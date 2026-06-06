//go:build darwin

package sia

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/fast"
)

// defaultKernelRunner returns the real MLX-backed runner on darwin.
func defaultKernelRunner() kernelRunner { return mlxKernelRunner{} }

// mlxKernelRunner JIT-compiles and runs a candidate kernel through MLX's Metal
// backend. The kernel's problem constants (dim, nrows, eps) are injected via a
// frozen header so the agent's source can reference them but cannot change them;
// the only things the agent controls are the body and the launch geometry.
type mlxKernelRunner struct{}

// frozenHeader is the read-only preamble prepended to every candidate kernel. It
// defines the problem constants as Metal `constant` values. The agent cannot
// override these — they are not part of the source it writes.
func frozenHeader(spec RMSNormSpec) string {
	return fmt.Sprintf("constant int dim = %d;\nconstant int nrows = %d;\nconstant float epsf = %g;\n",
		spec.Dim, spec.Rows, spec.Eps)
}

// run compiles source against the frozen inputs and times the evaluated work. The
// kernel is built once, then applied a warmup iteration (absorbing the one-time
// JIT compile) followed by iters timed iterations; PerIter is the timed total
// divided by iters. A malformed kernel yields kernelRun{CompileErr:...} and a nil
// Go error; only an inability to run at all (no Metal, cancelled context) is a Go
// error.
func (mlxKernelRunner) run(ctx context.Context, spec RMSNormSpec, source string, cfg launchConfig, iters int) (kernelRun, error) {
	if !mlx.MetalIsAvailable() {
		return kernelRun{}, fmt.Errorf("metal is not available on this host")
	}
	if err := ctx.Err(); err != nil {
		return kernelRun{}, err
	}

	x, w := spec.Inputs()
	xa := mlx.NewArray(x, spec.Rows, spec.Dim)
	defer xa.Free()
	wa := mlx.NewArray(w, spec.Dim)
	defer wa.Free()

	kernel, compileErr := buildKernel(spec, source)
	if compileErr != "" {
		return kernelRun{CompileErr: compileErr}, nil
	}
	defer kernel.Close()

	inputs := []*mlx.Array{xa, wa}
	outShapes := [][]int{{spec.Rows, spec.Dim}}
	outDtypes := []mlx.Dtype{mlx.Float32}

	// apply runs one Apply+Eval, returning the materialized output or a compile
	// error (the JIT failure surfaces at Eval, verified against live MLX).
	apply := func() (*mlx.Array, string) {
		var out *mlx.Array
		var cerr string
		func() {
			// fast.Kernel.Apply panics on some launch failures; treat as feedback.
			defer func() {
				if r := recover(); r != nil {
					out, cerr = nil, fmt.Sprintf("%v", r)
				}
			}()
			outs := kernel.Apply(inputs, cfg.Grid, cfg.ThreadGroup, outShapes, outDtypes)
			if len(outs) != 1 || outs[0] == nil {
				cerr = "kernel produced no output array"
				return
			}
			out = outs[0]
			if err := mlx.Eval(out); err != nil {
				out.Free()
				out, cerr = nil, err.Error()
			}
		}()
		return out, cerr
	}

	// Warmup iteration (only when timing multiple iters): absorbs JIT compile and
	// caches so the timed block measures steady-state kernel execution.
	if iters > 1 {
		out, cerr := apply()
		if cerr != "" {
			return kernelRun{CompileErr: cerr}, nil
		}
		out.Free()
		if err := ctx.Err(); err != nil {
			return kernelRun{}, err
		}
	}

	timed := iters
	if timed < 1 {
		timed = 1
	}

	var last *mlx.Array
	start := time.Now()
	for i := 0; i < timed; i++ {
		out, cerr := apply()
		if cerr != "" {
			if last != nil {
				last.Free()
			}
			return kernelRun{CompileErr: cerr}, nil
		}
		if last != nil {
			last.Free()
		}
		last = out
	}
	mlx.Synchronize()
	perIter := time.Since(start) / time.Duration(timed)
	defer last.Free()

	got, err := mlx.ToSlice[float32](last)
	if err != nil {
		return kernelRun{CompileErr: err.Error()}, nil
	}
	return kernelRun{Output: got, PerIter: perIter}, nil
}

// buildKernel constructs the candidate kernel with the frozen header. A
// constructor error becomes a compile-error string (not a Go error). The kernel
// is unevaluated; the actual MSL build error surfaces when its output is Eval'd.
func buildKernel(spec RMSNormSpec, source string) (kernel *fast.Kernel, compileErr string) {
	defer func() {
		if r := recover(); r != nil {
			kernel, compileErr = nil, fmt.Sprintf("%v", r)
		}
	}()
	k, err := fast.MetalKernelWithHeader(
		"sia_rmsnorm_candidate",
		[]string{"x", "w"},
		[]string{"out"},
		source,
		frozenHeader(spec),
		true,  // ensure_row_contiguous
		false, // atomic_outputs
	)
	if err != nil {
		return nil, err.Error()
	}
	return k, ""
}
