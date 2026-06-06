package sia

import (
	"context"
	"fmt"
)

// KernelBenchmarker supplies the three runnables [ThroughputEvaluator] drives for
// the Metal-kernel target: it runs a generation's candidate kernel against an
// external Go golden oracle (the correctness gate), times the candidate, and
// times the frozen gen-0 baseline kernel interleaved with it. It implements
// [Benchmarker].
//
// The golden inputs and golden output are computed once in pure Go at
// construction (see [RMSNormSpec.Golden]) and held here, outside the agent's
// working directory: the agent cannot read them, widen the tolerance, or hardcode
// the answer. The baseline source is the frozen gen-0 kernel captured at
// construction, not whatever the agent later writes.
type KernelBenchmarker struct {
	// Spec is the frozen RMS-norm problem.
	Spec RMSNormSpec
	// Iters is the number of timed iterations per sample (after a warmup that
	// absorbs the JIT compile). <= 0 uses DefaultBenchIters.
	Iters int

	baselineSrc string       // frozen gen-0 kernel source
	golden      []float32    // Go reference output for the frozen inputs
	runner      kernelRunner // GPU seam; nil uses the platform default
}

// DefaultBenchIters is the per-sample timed-iteration count when Iters is unset.
// A handful of iterations keeps each sample stable without slowing the loop.
const DefaultBenchIters = 20

var _ Benchmarker = (*KernelBenchmarker)(nil)

// NewKernelBenchmarker returns a benchmarker for spec whose frozen gen-0 baseline
// is baselineSrc (typically [SeedKernelSource]). It precomputes the golden output
// from the spec's fixed-seed inputs so the correctness gate is independent of MLX.
func NewKernelBenchmarker(spec RMSNormSpec, baselineSrc string) *KernelBenchmarker {
	x, w := spec.Inputs()
	return &KernelBenchmarker{
		Spec:        spec,
		baselineSrc: baselineSrc,
		golden:      spec.Golden(x, w),
	}
}

// Unit reports the throughput unit written to results.json.
func (b *KernelBenchmarker) Unit() string { return "ops_per_sec" }

// Correct runs the generation's candidate kernel once and compares its output to
// the Go golden oracle within the frozen tolerance. A kernel that fails to
// compile or whose output drifts past tolerance is ok=false with a reason (the
// REVISE feedback); err is returned only when the check itself could not run
// (e.g. no readable source, Metal unavailable).
func (b *KernelBenchmarker) Correct(ctx context.Context, genDir string) (bool, string, error) {
	source, cfg, err := b.loadCandidate(genDir)
	if err != nil {
		return false, err.Error(), nil
	}
	res, runErr := b.run().run(ctx, b.Spec, source, cfg, 1)
	if runErr != nil {
		return false, "", fmt.Errorf("run candidate for correctness: %w", runErr)
	}
	if res.CompileErr != "" {
		return false, "kernel did not compile: " + res.CompileErr, nil
	}
	ok, reason := b.Spec.CompareGolden(res.Output, b.golden)
	return ok, reason, nil
}

// RunCandidate times the generation's candidate kernel and returns one throughput
// sample. The evaluator gates correctness before calling this, so a compile
// failure here is unexpected and reported as an error.
func (b *KernelBenchmarker) RunCandidate(ctx context.Context, genDir string) (Sample, error) {
	source, cfg, err := b.loadCandidate(genDir)
	if err != nil {
		return Sample{}, fmt.Errorf("load candidate: %w", err)
	}
	return b.sample(ctx, source, cfg)
}

// RunBaseline times the frozen gen-0 baseline kernel, interleaved with each
// candidate run. It ignores genDir's source on purpose — the baseline is fixed at
// construction so the reported delta is gen-N minus gen-0 under identical
// conditions.
func (b *KernelBenchmarker) RunBaseline(ctx context.Context, genDir string) (Sample, error) {
	return b.sample(ctx, b.baselineSrc, b.Spec.defaultLaunch())
}

// sample runs source for the configured iteration count and converts the
// per-iteration GPU time into a higher-is-better throughput: ops/sec over the
// problem's element count.
func (b *KernelBenchmarker) sample(ctx context.Context, source string, cfg launchConfig) (Sample, error) {
	iters := b.Iters
	if iters <= 0 {
		iters = DefaultBenchIters
	}
	res, err := b.run().run(ctx, b.Spec, source, cfg, iters)
	if err != nil {
		return Sample{}, err
	}
	if res.CompileErr != "" {
		return Sample{}, fmt.Errorf("kernel failed to run: %s", res.CompileErr)
	}
	if res.PerIter <= 0 {
		return Sample{}, fmt.Errorf("kernel reported non-positive per-iteration time")
	}
	// ops = elements normalized per launch; throughput = ops / seconds.
	ops := float64(b.Spec.Rows * b.Spec.Dim)
	return Sample{Throughput: ops / res.PerIter.Seconds()}, nil
}

// loadCandidate reads the candidate kernel source (required) and optional launch
// config from genDir, applying frozen launch defaults for omitted fields.
func (b *KernelBenchmarker) loadCandidate(genDir string) (string, launchConfig, error) {
	return loadKernelSource(genDir, b.Spec)
}

func (b *KernelBenchmarker) run() kernelRunner {
	if b.runner != nil {
		return b.runner
	}
	return defaultKernelRunner()
}
