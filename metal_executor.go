package sia

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// launchConfig is the kernel's launch geometry: the Metal grid and threadgroup
// dimensions. It is the agent's second lever (after the source), supplied via the
// optional [KernelConfigName] sidecar; absent fields fall back to frozen defaults.
type launchConfig struct {
	Grid        [3]int `json:"grid"`         // total threads per dimension
	ThreadGroup [3]int `json:"thread_group"` // threads per threadgroup
}

// kernelRun is the result of JIT-compiling and running a candidate kernel: the
// flattened float32 output of the final iteration and the per-iteration GPU time
// (total evaluated time divided by the iteration count, with the first,
// JIT-compile-bearing iteration excluded as warmup). CompileErr is set (and
// Output nil) when the source failed to compile or the launch failed — the signal
// the agent gets back as feedback. A returned Go error means the runner itself
// could not run (e.g. Metal unavailable), distinct from a bad kernel.
type kernelRun struct {
	Output     []float32
	PerIter    time.Duration
	CompileErr string // non-empty => the kernel did not compile/run; not a runner failure
}

// kernelRunner JIT-compiles a Metal kernel source and runs it against the frozen
// inputs, returning the output and per-iteration timing. It is the single GPU
// seam used by both [MetalKernelExecutor] (does it run?) and [KernelBenchmarker]
// (how fast, and is it correct?). The real implementation lives behind a darwin
// build tag (metal_run_darwin.go); tests inject a fake.
type kernelRunner interface {
	// run compiles source, applies it to the spec's inputs with cfg, and times
	// the evaluated work over iters timed iterations after a warmup iteration
	// that absorbs the one-time JIT compile. iters <= 0 means a single timed run
	// (no warmup), which the executor uses for a pure does-it-run check. A
	// malformed kernel is reported via kernelRun.CompileErr (not a Go error); a Go
	// error means the runner could not execute at all.
	run(ctx context.Context, spec RMSNormSpec, source string, cfg launchConfig, iters int) (kernelRun, error)
}

// MetalKernelExecutor runs the generation's Metal kernel as the SIA target: it
// reads the gen's kernel source from the working directory, JIT-compiles it via
// MLX, and runs it once. A kernel that fails to compile or launch is reported as
// TargetResult{Success:false, ErrorMsg:<compiler error>} — never a Go error — so
// the feedback agent receives the compiler message and repairs the kernel next
// generation. It implements [TargetExecutor].
type MetalKernelExecutor struct {
	// Spec is the frozen RMS-norm problem the kernel solves.
	Spec RMSNormSpec
	// Timeout bounds a single JIT-compile-and-run; 0 uses DefaultKernelTimeout.
	// The agent will write kernels with pathological loops, so this is a hard cap.
	Timeout time.Duration
	// runner is the GPU seam; nil uses the platform default (real MLX on darwin,
	// an unavailable-stub elsewhere). Tests inject a fake.
	runner kernelRunner
}

// DefaultKernelTimeout bounds one compile-and-run. Generous enough for a cold
// JIT compile, tight enough that a runaway kernel does not stall the loop.
const DefaultKernelTimeout = 30 * time.Second

var _ TargetExecutor = (*MetalKernelExecutor)(nil)

// NewMetalKernelExecutor returns an executor for the given spec using the
// platform's default GPU runner.
func NewMetalKernelExecutor(spec RMSNormSpec) *MetalKernelExecutor {
	return &MetalKernelExecutor{Spec: spec}
}

// RunTarget reads the candidate kernel source from req.WorkingDir, compiles and
// runs it, and reports the outcome. Missing source or a compile/launch failure is
// Success=false with ErrorMsg (the feedback signal); a Go error is returned only
// when the runner itself cannot run.
func (e *MetalKernelExecutor) RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error) {
	source, cfg, err := e.loadKernel(req.WorkingDir)
	if err != nil {
		// No readable kernel source is a target-side failure the feedback agent
		// must fix (write kernel.metal), not a runner failure.
		return TargetResult{Success: false, ErrorMsg: err.Error()}, nil
	}

	timeout := e.Timeout
	if timeout <= 0 {
		timeout = DefaultKernelTimeout
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	res, runErr := e.run().run(runCtx, e.Spec, source, cfg, 1)
	if runErr != nil {
		// The runner could not execute (Metal unavailable, context cancelled).
		// This is an executor-side failure: report it but do not fail the loop.
		return TargetResult{Success: false, ErrorMsg: runErr.Error()}, fmt.Errorf("metal kernel runner: %w", runErr)
	}
	if res.CompileErr != "" {
		return TargetResult{Success: false, ErrorMsg: res.CompileErr}, nil
	}

	stdout := fmt.Sprintf("kernel compiled and ran in %s; %d outputs\n", res.PerIter, len(res.Output))
	if log := req.StdoutLog; log != "" {
		_ = os.WriteFile(log, []byte(stdout), 0o644)
	}
	return TargetResult{Success: true, Stdout: stdout}, nil
}

// loadKernel reads the kernel source (required) and launch config (optional) from
// workingDir, applying frozen defaults for any config field the agent omitted.
func (e *MetalKernelExecutor) loadKernel(workingDir string) (string, launchConfig, error) {
	return loadKernelSource(workingDir, e.Spec)
}

// loadKernelSource reads the candidate kernel source ([KernelSourceName],
// required) and optional launch config ([KernelConfigName]) from dir, applying
// spec's frozen launch defaults for omitted fields. It is shared by the executor
// (does-it-run) and the benchmarker (correctness + timing) so both read the
// agent's lever identically.
func loadKernelSource(dir string, spec RMSNormSpec) (string, launchConfig, error) {
	src, err := os.ReadFile(filepath.Join(dir, KernelSourceName))
	if err != nil {
		return "", launchConfig{}, fmt.Errorf("read kernel source %s: %w", KernelSourceName, err)
	}
	if len(src) == 0 {
		return "", launchConfig{}, fmt.Errorf("kernel source %s is empty", KernelSourceName)
	}
	cfg := spec.defaultLaunch()
	if cfgBytes, err := os.ReadFile(filepath.Join(dir, KernelConfigName)); err == nil {
		var parsed launchConfig
		if err := json.Unmarshal(cfgBytes, &parsed); err != nil {
			return "", launchConfig{}, fmt.Errorf("parse %s: %w", KernelConfigName, err)
		}
		cfg = spec.normalizeLaunch(parsed)
	}
	return string(src), cfg, nil
}

// run returns the configured runner or the platform default.
func (e *MetalKernelExecutor) run() kernelRunner {
	if e.runner != nil {
		return e.runner
	}
	return defaultKernelRunner()
}

// defaultLaunch is the frozen launch geometry for the seed kernel: one thread per
// row, 256-wide threadgroups. The agent may override via the sidecar.
func (s RMSNormSpec) defaultLaunch() launchConfig {
	return launchConfig{
		Grid:        [3]int{roundUp(s.Rows, 256), 1, 1},
		ThreadGroup: [3]int{256, 1, 1},
	}
}

// normalizeLaunch clamps an agent-supplied launch config to sane, non-zero
// values so a malformed sidecar cannot crash the launch (a zero threadgroup is a
// hard Metal error). Missing dimensions default to 1.
func (s RMSNormSpec) normalizeLaunch(cfg launchConfig) launchConfig {
	for i := range cfg.Grid {
		if cfg.Grid[i] <= 0 {
			cfg.Grid[i] = 1
		}
		if cfg.ThreadGroup[i] <= 0 {
			cfg.ThreadGroup[i] = 1
		}
	}
	return cfg
}

// roundUp rounds n up to the next multiple of m (m > 0).
func roundUp(n, m int) int {
	if m <= 0 {
		return n
	}
	return ((n + m - 1) / m) * m
}
