package sia

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// goldenRunner is a [kernelRunner] that returns the spec's own golden output (so
// the candidate is "correct") at a configurable speed. It lets the benchmarker
// and ThroughputEvaluator be tested end-to-end without a GPU.
type goldenRunner struct {
	spec    RMSNormSpec
	perIter time.Duration
	wrong   bool // if set, perturb the output so the correctness gate fails
}

func (r goldenRunner) run(_ context.Context, spec RMSNormSpec, _ string, _ launchConfig, _ int) (kernelRun, error) {
	x, w := spec.Inputs()
	out := spec.Golden(x, w)
	if r.wrong {
		out[0] += 1000 // gross error the tolerance cannot absorb
	}
	return kernelRun{Output: out, PerIter: r.perIter}, nil
}

func newBench(t *testing.T, r kernelRunner) *KernelBenchmarker {
	t.Helper()
	spec := RMSNormSpec{Rows: 8, Dim: 16, Eps: 1e-6, Seed: 1, RelTol: 2e-3}
	b := NewKernelBenchmarker(spec, SeedKernelSource)
	b.runner = r
	b.Iters = 1
	return b
}

func TestKernelBenchmarkerUnit(t *testing.T) {
	if u := newBench(t, goldenRunner{}).Unit(); u != "ops_per_sec" {
		t.Errorf("Unit()=%q want ops_per_sec", u)
	}
}

func TestKernelBenchmarkerCorrect(t *testing.T) {
	b := newBench(t, goldenRunner{perIter: time.Millisecond})
	dir := t.TempDir()
	writeKernel(t, dir, "candidate source")

	ok, reason, err := b.Correct(context.Background(), dir)
	if err != nil {
		t.Fatalf("Correct: %v", err)
	}
	if !ok {
		t.Fatalf("a golden-matching kernel should pass; reason=%q", reason)
	}
}

func TestKernelBenchmarkerCorrectRejectsWrong(t *testing.T) {
	b := newBench(t, goldenRunner{perIter: time.Millisecond, wrong: true})
	dir := t.TempDir()
	writeKernel(t, dir, "fast but wrong")

	ok, reason, err := b.Correct(context.Background(), dir)
	if err != nil {
		t.Fatalf("Correct returned a Go error for a wrong candidate (should be ok=false): %v", err)
	}
	if ok {
		t.Fatal("a wrong kernel must NOT pass the correctness gate")
	}
	if reason == "" {
		t.Error("a failing gate should explain why (REVISE feedback)")
	}
}

func TestKernelBenchmarkerCompileFailureIsRevise(t *testing.T) {
	b := newBench(t, &fakeRunner{compileErr: "build error"})
	dir := t.TempDir()
	writeKernel(t, dir, "@@@")

	ok, reason, err := b.Correct(context.Background(), dir)
	if err != nil {
		t.Fatalf("a compile failure is REVISE feedback, not a Go error: %v", err)
	}
	if ok {
		t.Fatal("a kernel that did not compile cannot be correct")
	}
	if reason == "" {
		t.Error("compile failure should carry the compiler message")
	}
}

func TestKernelBenchmarkerThroughputHigherIsBetter(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "candidate")

	fast := newBench(t, goldenRunner{perIter: time.Millisecond})
	slow := newBench(t, goldenRunner{perIter: 10 * time.Millisecond})

	fs, err := fast.RunCandidate(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	ss, err := slow.RunCandidate(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if !(fs.Throughput > ss.Throughput) {
		t.Errorf("faster kernel must have higher throughput: fast=%g slow=%g", fs.Throughput, ss.Throughput)
	}
	// ops/sec for 8*16=128 ops in 1ms = 128_000.
	wantFast := float64(8*16) / time.Millisecond.Seconds()
	if d := fs.Throughput - wantFast; d < -1 || d > 1 {
		t.Errorf("throughput=%g want ~%g", fs.Throughput, wantFast)
	}
}

func TestKernelBenchmarkerBaselineIgnoresGenDir(t *testing.T) {
	// RunBaseline must time the frozen seed, not whatever is in genDir.
	b := newBench(t, goldenRunner{perIter: time.Millisecond})
	dir := t.TempDir() // intentionally has NO kernel.metal
	if _, err := b.RunBaseline(context.Background(), dir); err != nil {
		t.Fatalf("RunBaseline must not depend on genDir source: %v", err)
	}
}

// TestThroughputEvaluatorEndToEnd drives agent 3's evaluator with our
// benchmarker: a correct, faster candidate should PASS with a speedup; the
// results.json scalars the feedback agent reads must be populated.
func TestThroughputEvaluatorEndToEndPass(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "candidate that is correct and fast")

	// Candidate twice as fast as the (interleaved) baseline.
	spec := RMSNormSpec{Rows: 8, Dim: 16, Eps: 1e-6, Seed: 1, RelTol: 2e-3}
	b := NewKernelBenchmarker(spec, SeedKernelSource)
	b.Iters = 1
	b.runner = perSourceRunner{spec: spec, candidate: time.Millisecond, baseline: 2 * time.Millisecond}

	eval := &ThroughputEvaluator{Bench: b, Runs: 3}
	res, err := eval.Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Status != EvalSuccess {
		t.Fatalf("Status=%v reason=%q", res.Status, res.Reason)
	}

	var got struct {
		Verdict       string  `json:"verdict"`
		CorrectnessOK bool    `json:"correctness_ok"`
		Unit          string  `json:"unit"`
		TokensPerSec  float64 `json:"tokens_per_sec"`
		Speedup       float64 `json:"speedup"`
	}
	data, err := os.ReadFile(filepath.Join(dir, NameResultsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != VerdictPass {
		t.Errorf("verdict=%q want PASS", got.Verdict)
	}
	if !got.CorrectnessOK {
		t.Error("correctness_ok should be true on PASS")
	}
	if got.Unit != "ops_per_sec" {
		t.Errorf("unit=%q", got.Unit)
	}
	if got.Speedup <= 1.0 {
		t.Errorf("a 2x-faster candidate should report speedup>1, got %g", got.Speedup)
	}
}

func TestThroughputEvaluatorEndToEndRevise(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "fast but wrong")

	spec := RMSNormSpec{Rows: 8, Dim: 16, Eps: 1e-6, Seed: 1, RelTol: 2e-3}
	b := NewKernelBenchmarker(spec, SeedKernelSource)
	b.runner = goldenRunner{spec: spec, perIter: time.Microsecond, wrong: true}

	res, err := (&ThroughputEvaluator{Bench: b, Runs: 3}).Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	var got struct {
		Verdict       string `json:"verdict"`
		CorrectnessOK bool   `json:"correctness_ok"`
	}
	data, _ := os.ReadFile(filepath.Join(dir, NameResultsJSON))
	_ = json.Unmarshal(data, &got)
	if got.Verdict != VerdictRevise {
		t.Errorf("a wrong-but-fast kernel must be REVISE, got %q", got.Verdict)
	}
	if got.CorrectnessOK {
		t.Error("correctness_ok must be false for a wrong kernel")
	}
	_ = res
}

// perSourceRunner returns golden output but different timing for the candidate vs
// the frozen baseline source, so the interleaved-baseline speedup is exercised.
type perSourceRunner struct {
	spec                RMSNormSpec
	candidate, baseline time.Duration
}

func (r perSourceRunner) run(_ context.Context, spec RMSNormSpec, source string, _ launchConfig, _ int) (kernelRun, error) {
	x, w := spec.Inputs()
	out := spec.Golden(x, w)
	d := r.candidate
	if source == SeedKernelSource {
		d = r.baseline
	}
	return kernelRun{Output: out, PerIter: d}, nil
}
