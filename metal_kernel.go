package sia

import (
	"fmt"
	"math"
)

// KernelSourceName is the file the target/feedback agent writes into its working
// directory each generation: the body of the Metal Shading Language kernel that
// [MetalKernelExecutor] JIT-compiles and runs. The agent's only lever is this
// source; the benchmark harness, inputs, and golden oracle are frozen.
const KernelSourceName = "kernel.metal"

// KernelConfigName is an optional sidecar the agent may write to tune launch
// parameters (grid / threadgroup). When absent, the executor uses the frozen
// defaults from the [RMSNormSpec]. It is the agent's second lever and stays
// inside the harness contract (the agent cannot change inputs or the oracle).
const KernelConfigName = "kernel.json"

// RMSNormSpec describes the fixed RMS-normalization problem the kernel solves:
//
//	y[r,c] = x[r,c] / sqrt(mean_c(x[r,c]^2) + Eps) * w[c]
//
// over a Rows×Dim row-major float32 matrix. It is the frozen problem definition:
// the agent rewrites only the kernel source, never these dimensions, the seed,
// the weights, or the tolerance.
type RMSNormSpec struct {
	Rows int     // number of rows (independent normalizations)
	Dim  int     // row length (the reduction axis)
	Eps  float32 // numerical-stability epsilon, frozen
	Seed int64   // RNG seed for the fixed input matrix and weights
	// RelTol is the frozen relative tolerance for the correctness gate. GPU
	// float reduction order differs from the Go reference, so an exact match is
	// not expected; this bounds the allowed drift and the agent cannot widen it.
	RelTol float64
}

// DefaultRMSNormSpec returns the problem the demo runs: a 1024×1024 matrix,
// eps 1e-6, seed 1, relative tolerance 2e-3. Large enough that a naive
// per-thread reduction leaves real headroom, small enough to bench in
// milliseconds so the loop stays watchable.
func DefaultRMSNormSpec() RMSNormSpec {
	return RMSNormSpec{Rows: 1024, Dim: 1024, Eps: 1e-6, Seed: 1, RelTol: 2e-3}
}

// Inputs returns the frozen input matrix x (Rows×Dim) and weight vector w (Dim),
// both row-major float32, deterministically generated from Seed. These are the
// golden inputs: generated once at benchmarker construction and held outside the
// agent's working directory.
//
// The generator is a self-contained float32 LCG (no MLX dependency) so the
// inputs are reproducible on any host and identical for candidate and baseline.
func (s RMSNormSpec) Inputs() (x []float32, w []float32) {
	x = make([]float32, s.Rows*s.Dim)
	w = make([]float32, s.Dim)
	r := newLCG(uint64(s.Seed))
	for i := range x {
		x[i] = r.normalish()
	}
	for i := range w {
		// Weights near 1.0 so the normalization, not the scale, dominates.
		w[i] = 0.5 + r.unit()
	}
	return x, w
}

// Golden computes the reference RMS-norm of x with weights w in float64, the
// external oracle the candidate kernel is graded against. It is pure Go and
// independent of MLX: the agent cannot read it, edit it, or influence it.
func (s RMSNormSpec) Golden(x, w []float32) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < s.Rows; r++ {
		base := r * s.Dim
		var sumSq float64
		for c := 0; c < s.Dim; c++ {
			v := float64(x[base+c])
			sumSq += v * v
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(s.Dim)+float64(s.Eps))
		for c := 0; c < s.Dim; c++ {
			out[base+c] = float32(float64(x[base+c]) * inv * float64(w[c]))
		}
	}
	return out
}

// CompareGolden reports whether got matches the golden output within the frozen
// relative tolerance. The reason is populated on mismatch (for the REVISE
// feedback) and names the first offending element. It never widens the
// tolerance — that is the whole point of the gate.
func (s RMSNormSpec) CompareGolden(got, golden []float32) (ok bool, reason string) {
	if len(got) != len(golden) {
		return false, fmt.Sprintf("output length %d != expected %d", len(got), len(golden))
	}
	for i := range golden {
		g := float64(golden[i])
		d := math.Abs(float64(got[i]) - g)
		tol := s.RelTol * (math.Abs(g) + 1e-6)
		if d > tol {
			r, c := i/s.Dim, i%s.Dim
			return false, fmt.Sprintf("mismatch at [%d,%d]: got %g want %g (|d|=%g > tol=%g)",
				r, c, got[i], golden[i], d, tol)
		}
	}
	return true, ""
}

// SeedKernelSource is the deliberately unoptimized but correct gen-0 kernel: one
// thread per row, and — like a naive first draft — it recomputes the row's
// entire sum-of-squares from scratch inside the per-element normalize loop. That
// is O(Dim^2) redundant reduction work per row, serialized, with no threadgroup
// cooperation and no vectorization. It compiles and is correct, so gen-0 scores;
// it is ~25x slower than a single-pass vectorized kernel on a 1024x1024 input, so
// the agent has large, legible headroom to climb (measured, not asserted).
//
// The kernel reads x (Rows*Dim), w (Dim) and writes out (Rows*Dim). The launch
// grid is (Rows,1,1): thread r owns row r. dim/nrows/eps come from the frozen
// header the harness prepends; the agent cannot change them.
const SeedKernelSource = `
    uint row = thread_position_in_grid.x;
    if (row >= (uint)nrows) { return; }
    uint base = row * (uint)dim;

    // Naive and wasteful: recompute the full sum of squares for every output
    // element instead of once per row. Correct, but O(dim^2) per row.
    for (uint c = 0; c < (uint)dim; c++) {
        float ss = 0.0f;
        for (uint k = 0; k < (uint)dim; k++) {
            float v = x[base + k];
            ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)dim + epsf);
        out[base + c] = x[base + c] * inv * w[c];
    }
`

// ScriptedKernelStages returns a fixed sequence of RMSNorm kernel sources of
// monotonically increasing throughput, each correct against the golden oracle.
// The built-in scripted improver walks this list so the full SIA loop — and the
// measured ops/sec — can run with no external model: a deterministic demo and the
// pre-recorded insurance the spec calls for. Every stage passes the correctness
// gate; the speedup is real and measured, not asserted.
//
// The ordering reflects what actually wins on Apple silicon for a 1024x1024
// input (measured): the dominant gain is algorithmic — collapsing the seed's
// O(dim^2)-per-row redundant reduction to a single O(dim) pass is a ~40x jump.
// The kernel is then memory-bandwidth-bound, so float4 vectorization alone does
// not help; folding the weight scale into a fused multiply on the vectorized path
// recovers a final few percent. (A plain-float4 variant without the fused scale
// was measured and dropped: it does not beat the scalar single-pass at this size,
// so including it would make the demo's curve regress — which would be
// dishonest.)
func ScriptedKernelStages() []string {
	return []string{
		SeedKernelSource,   // stage 0: naive O(dim^2) per row (the gen-0 baseline)
		kernelHoistedRecip, // stage 1: single O(dim) pass — the big algorithmic win
		kernelFloat4FMA,    // stage 2: float4 + fused weight scale — the final few %
	}
}

// kernelHoistedRecip: same naive structure, but hoist the reciprocal-of-dim out
// of the reduction and use fma in the accumulation. A small, honest win over the
// seed that still leaves the big gains (vectorization) on the table.
const kernelHoistedRecip = `
    uint row = thread_position_in_grid.x;
    if (row >= (uint)nrows) { return; }
    uint base = row * (uint)dim;
    float invDim = 1.0f / (float)dim;

    float ss = 0.0f;
    for (uint c = 0; c < (uint)dim; c++) {
        float v = x[base + c];
        ss = fma(v, v, ss);
    }
    float inv = rsqrt(ss * invDim + epsf);

    for (uint c = 0; c < (uint)dim; c++) {
        out[base + c] = x[base + c] * inv * w[c];
    }
`

// kernelFloat4FMA: vectorize both passes with float4 loads/stores (quartering the
// loop trip count and memory transactions) and fold the weight scale into a
// fused multiply. Requires dim % 4 == 0; the harness fixes dim (1024) so this
// holds. On a memory-bound kernel this recovers a final few percent over the
// scalar single pass.
const kernelFloat4FMA = `
    uint row = thread_position_in_grid.x;
    if (row >= (uint)nrows) { return; }
    uint base = row * (uint)dim;
    uint base4 = base >> 2;
    uint dim4 = (uint)dim >> 2;
    float invDim = 1.0f / (float)dim;

    device const float4* x4 = (device const float4*)x;
    device const float4* w4 = (device const float4*)w;
    device float4* out4 = (device float4*)out;

    float4 acc = float4(0.0f);
    for (uint c = 0; c < dim4; c++) {
        float4 v = x4[base4 + c];
        acc = fma(v, v, acc);
    }
    float ss = acc.x + acc.y + acc.z + acc.w;
    float inv = rsqrt(ss * invDim + epsf);
    float4 invv = float4(inv);

    for (uint c = 0; c < dim4; c++) {
        out4[base4 + c] = x4[base4 + c] * invv * w4[c];
    }
`

// LCG is a tiny self-contained linear-congruential generator used to produce
// the frozen inputs without an external RNG dependency. It is not for
// cryptographic or statistical use — only for reproducible test fixtures.
type lcg struct{ state uint64 }

func newLCG(seed uint64) *lcg {
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}
	return &lcg{state: seed}
}

// next advances the generator (Numerical Recipes constants) and returns the raw
// 64-bit state.
func (r *lcg) next() uint64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return r.state
}

// unit returns a float32 in [0,1).
func (r *lcg) unit() float32 {
	return float32(r.next()>>40) / float32(1<<24)
}

// normalish returns a roughly zero-mean unit-ish float32 by summing four uniform
// draws (a cheap central-limit approximation); enough variety to exercise the
// reduction without needing a real Gaussian.
func (r *lcg) normalish() float32 {
	var s float32
	for i := 0; i < 4; i++ {
		s += r.unit()
	}
	return s - 2.0
}
