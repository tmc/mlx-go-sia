package sia

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fakeBench is a deterministic Benchmarker for tests: no real model, no real
// timing — each run returns a fixed throughput, and correctness is dictated by
// the test. It also records how many times each runnable was invoked so tests
// can assert the baseline was interleaved and the median count is honored.
type fakeBench struct {
	ok           bool
	reason       string
	correctErr   error
	candidate    float64 // throughput RunCandidate returns
	baseline     float64 // throughput RunBaseline returns
	candErr      error
	baseErr      error
	unit         string
	candCalls    int
	baseCalls    int
	correctCalls int
}

func (f *fakeBench) Correct(_ context.Context, _ string) (bool, string, error) {
	f.correctCalls++
	return f.ok, f.reason, f.correctErr
}

func (f *fakeBench) RunCandidate(_ context.Context, _ string) (Sample, error) {
	f.candCalls++
	if f.candErr != nil {
		return Sample{}, f.candErr
	}
	return Sample{Throughput: f.candidate}, nil
}

func (f *fakeBench) RunBaseline(_ context.Context, _ string) (Sample, error) {
	f.baseCalls++
	if f.baseErr != nil {
		return Sample{}, f.baseErr
	}
	return Sample{Throughput: f.baseline}, nil
}

func (f *fakeBench) Unit() string {
	if f.unit == "" {
		return "tokens_per_sec"
	}
	return f.unit
}

// readResults loads and decodes genDir/results.json.
func readResults(t *testing.T, genDir string) throughputResults {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(genDir, NameResultsJSON))
	if err != nil {
		t.Fatalf("read results.json: %v", err)
	}
	var res throughputResults
	if err := json.Unmarshal(data, &res); err != nil {
		t.Fatalf("decode results.json: %v", err)
	}
	return res
}

func TestThroughputEvaluatorCorrectButSlow(t *testing.T) {
	// A correct candidate that is faster than baseline passes with a positive
	// delta; the advisory throughput number is present.
	dir := t.TempDir()
	bench := &fakeBench{ok: true, candidate: 120, baseline: 100}
	eval := &ThroughputEvaluator{Bench: bench, Runs: 3, Sleep: func(time.Duration) {}}

	got, err := eval.Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if got.Status != EvalSuccess {
		t.Fatalf("Status = %q, want %q", got.Status, EvalSuccess)
	}

	res := readResults(t, dir)
	if res.Verdict != VerdictPass {
		t.Errorf("verdict = %q, want %q", res.Verdict, VerdictPass)
	}
	if !res.CorrectnessOK {
		t.Errorf("correctness_ok = false, want true")
	}
	if res.TokensPerSec != 120 {
		t.Errorf("tokens_per_sec = %v, want 120", res.TokensPerSec)
	}
	if res.BaselineTokensPerSec != 100 {
		t.Errorf("baseline_tokens_per_sec = %v, want 100", res.BaselineTokensPerSec)
	}
	if res.DeltaTokensPerSec != 20 {
		t.Errorf("delta_tokens_per_sec = %v, want 20", res.DeltaTokensPerSec)
	}
	if res.Speedup != 1.2 {
		t.Errorf("speedup = %v, want 1.2", res.Speedup)
	}
}

func TestThroughputEvaluatorSlowerStillPasses(t *testing.T) {
	// Correct but SLOWER than baseline still passes correctness (verdict PASS);
	// the delta is simply negative. Correctness is the gate, not speed.
	dir := t.TempDir()
	bench := &fakeBench{ok: true, candidate: 80, baseline: 100}
	eval := &ThroughputEvaluator{Bench: bench, Runs: 1, Sleep: func(time.Duration) {}}

	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	res := readResults(t, dir)
	if res.Verdict != VerdictPass {
		t.Errorf("verdict = %q, want %q (correctness gates, speed does not)", res.Verdict, VerdictPass)
	}
	if res.DeltaTokensPerSec != -20 {
		t.Errorf("delta = %v, want -20", res.DeltaTokensPerSec)
	}
}

func TestThroughputEvaluatorFastButWrong(t *testing.T) {
	// A fast-but-wrong candidate must be REVISE — and must NOT run any timing.
	dir := t.TempDir()
	bench := &fakeBench{ok: false, reason: "token 3 mismatched golden", candidate: 9999, baseline: 100}
	eval := &ThroughputEvaluator{Bench: bench, Runs: 5, Sleep: func(time.Duration) {}}

	got, err := eval.Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if got.Status != EvalSuccess {
		t.Fatalf("Status = %q, want %q (a wrong candidate is a valid eval, not an error)", got.Status, EvalSuccess)
	}

	res := readResults(t, dir)
	if res.Verdict != VerdictRevise {
		t.Errorf("verdict = %q, want %q", res.Verdict, VerdictRevise)
	}
	if res.CorrectnessOK {
		t.Errorf("correctness_ok = true, want false")
	}
	if res.Reason != "token 3 mismatched golden" {
		t.Errorf("reason = %q, want the gate's reason", res.Reason)
	}
	if res.TokensPerSec != 0 {
		t.Errorf("tokens_per_sec = %v, want 0 (no number on REVISE)", res.TokensPerSec)
	}
	// Correctness gate runs BEFORE timing: no candidate/baseline runs happened.
	if bench.candCalls != 0 || bench.baseCalls != 0 {
		t.Errorf("timing ran on a wrong candidate: candCalls=%d baseCalls=%d, want 0/0",
			bench.candCalls, bench.baseCalls)
	}
}

func TestThroughputEvaluatorInterleavesBaseline(t *testing.T) {
	// Each timing cycle must run the baseline AND the candidate exactly once, so
	// the per-cycle delta cancels drift.
	dir := t.TempDir()
	const n = 4
	bench := &fakeBench{ok: true, candidate: 110, baseline: 100}
	eval := &ThroughputEvaluator{Bench: bench, Runs: n, Sleep: func(time.Duration) {}}

	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if bench.candCalls != n {
		t.Errorf("candidate runs = %d, want %d", bench.candCalls, n)
	}
	if bench.baseCalls != n {
		t.Errorf("baseline runs = %d, want %d (baseline must interleave every cycle)", bench.baseCalls, n)
	}
	res := readResults(t, dir)
	if res.Runs != n {
		t.Errorf("runs = %d, want %d", res.Runs, n)
	}
	if res.Baseline == nil || res.Candidate == nil {
		t.Fatalf("distributions missing: candidate=%v baseline=%v", res.Candidate, res.Baseline)
	}
	if got := len(res.Candidate.Samples); got != n {
		t.Errorf("candidate samples = %d, want %d", got, n)
	}
	if got := len(res.Baseline.Samples); got != n {
		t.Errorf("baseline samples = %d, want %d", got, n)
	}
}

func TestThroughputEvaluatorDefaultRuns(t *testing.T) {
	// Runs <= 0 falls back to DefaultThroughputRuns.
	dir := t.TempDir()
	bench := &fakeBench{ok: true, candidate: 110, baseline: 100}
	eval := &ThroughputEvaluator{Bench: bench, Sleep: func(time.Duration) {}}

	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if bench.candCalls != DefaultThroughputRuns {
		t.Errorf("candidate runs = %d, want default %d", bench.candCalls, DefaultThroughputRuns)
	}
}

func TestMedianOddEven(t *testing.T) {
	tests := []struct {
		name string
		in   []float64
		want float64
	}{
		{"odd", []float64{30, 10, 20}, 20},            // sorted 10,20,30
		{"even", []float64{40, 10, 30, 20}, 25},       // sorted 10,20,30,40 -> (20+30)/2
		{"single", []float64{42}, 42},                 //
		{"even-pair", []float64{100, 200}, 150},       //
		{"unsorted-odd", []float64{5, 1, 9, 3, 7}, 5}, // sorted 1,3,5,7,9
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := summarize(tt.in)
			if d.Median != tt.want {
				t.Errorf("median(%v) = %v, want %v", tt.in, d.Median, tt.want)
			}
		})
	}
}

func TestSummarizeMinMax(t *testing.T) {
	d := summarize([]float64{30, 10, 20, 50, 40})
	if d.Min != 10 {
		t.Errorf("min = %v, want 10", d.Min)
	}
	if d.Max != 50 {
		t.Errorf("max = %v, want 50", d.Max)
	}
	if d.Median != 30 {
		t.Errorf("median = %v, want 30", d.Median)
	}
	if len(d.Samples) != 5 {
		t.Errorf("samples len = %d, want 5", len(d.Samples))
	}
}

func TestThroughputEvaluatorMedianFromNoisySamples(t *testing.T) {
	// Verify the reported metric is the MEDIAN of varying samples, not the first
	// or the mean. A scripted Benchmarker returns a different value each run.
	dir := t.TempDir()
	candSeq := []float64{100, 130, 110, 500, 120} // median 120 (sorted 100,110,120,130,500)
	baseSeq := []float64{90, 100, 95, 105, 100}   // median 100 (sorted 90,95,100,100,105)
	bench := &seqBench{cand: candSeq, base: baseSeq}
	eval := &ThroughputEvaluator{Bench: bench, Runs: len(candSeq), Sleep: func(time.Duration) {}}

	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	res := readResults(t, dir)
	if res.TokensPerSec != 120 {
		t.Errorf("candidate median = %v, want 120 (resilient to the 500 outlier)", res.TokensPerSec)
	}
	if res.BaselineTokensPerSec != 100 {
		t.Errorf("baseline median = %v, want 100", res.BaselineTokensPerSec)
	}
	if res.DeltaTokensPerSec != 20 {
		t.Errorf("delta = %v, want 20", res.DeltaTokensPerSec)
	}
}

func TestThroughputEvaluatorCooldownBetweenCycles(t *testing.T) {
	// Cooldown is slept between cycles (N-1 times), never before the first.
	dir := t.TempDir()
	var slept int
	bench := &fakeBench{ok: true, candidate: 110, baseline: 100}
	eval := &ThroughputEvaluator{
		Bench:    bench,
		Runs:     4,
		Cooldown: 5 * time.Millisecond,
		Sleep:    func(d time.Duration) { slept++ },
	}
	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if slept != 3 {
		t.Errorf("cooldown sleeps = %d, want 3 (N-1)", slept)
	}
}

func TestThroughputEvaluatorNilBench(t *testing.T) {
	// A nil Benchmarker is a construction error the evaluator cannot start with —
	// the one case that IS a Go error.
	eval := &ThroughputEvaluator{}
	if _, err := eval.Evaluate(context.Background(), t.TempDir()); err == nil {
		t.Fatal("Evaluate with nil Benchmarker = nil error, want error")
	}
}

func TestThroughputEvaluatorCorrectCheckError(t *testing.T) {
	// If the correctness check itself cannot run, that is EvalError (not a Go
	// error that aborts the whole run), and results.json records the reason.
	dir := t.TempDir()
	bench := &fakeBench{correctErr: errors.New("oracle binary missing")}
	eval := &ThroughputEvaluator{Bench: bench, Sleep: func(time.Duration) {}}

	got, err := eval.Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate returned Go error: %v (should be EvalError status, not abort)", err)
	}
	if got.Status != EvalError {
		t.Fatalf("Status = %q, want %q", got.Status, EvalError)
	}
	res := readResults(t, dir)
	if res.CorrectnessOK {
		t.Errorf("correctness_ok = true, want false")
	}
}

func TestThroughputEvaluatorBenchRunError(t *testing.T) {
	// A timing run that errors mid-benchmark is EvalError, no Go error, no PASS.
	dir := t.TempDir()
	bench := &fakeBench{ok: true, baseErr: errors.New("warmup crashed")}
	eval := &ThroughputEvaluator{Bench: bench, Runs: 3, Sleep: func(time.Duration) {}}

	got, err := eval.Evaluate(context.Background(), dir)
	if err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if got.Status != EvalError {
		t.Errorf("Status = %q, want %q", got.Status, EvalError)
	}
}

func TestThroughputResultsTopLevelScalars(t *testing.T) {
	// The feedback agent extracts only top-level number/string/bool scalars from
	// results.json. Assert the demo metrics are present at the top level (not
	// buried in a nested object) so they flow to feedback.
	dir := t.TempDir()
	bench := &fakeBench{ok: true, candidate: 120, baseline: 100, unit: "ops_per_sec"}
	eval := &ThroughputEvaluator{Bench: bench, Runs: 1, Sleep: func(time.Duration) {}}
	if _, err := eval.Evaluate(context.Background(), dir); err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, NameResultsJSON))
	if err != nil {
		t.Fatalf("read results.json: %v", err)
	}
	var top map[string]any
	if err := json.Unmarshal(data, &top); err != nil {
		t.Fatalf("decode: %v", err)
	}
	for _, k := range []string{
		"verdict", "correctness_ok", "tokens_per_sec",
		"baseline_tokens_per_sec", "delta_tokens_per_sec", "speedup", "runs", "unit",
	} {
		if _, ok := top[k]; !ok {
			t.Errorf("results.json missing top-level scalar %q (feedback won't see it)", k)
		}
	}
	if top["unit"] != "ops_per_sec" {
		t.Errorf("unit = %v, want ops_per_sec (Benchmarker.Unit honored)", top["unit"])
	}
}

// seqBench returns a scripted sequence of throughputs, advancing each call.
type seqBench struct {
	cand, base []float64
	ci, bi     int
}

func (s *seqBench) Correct(context.Context, string) (bool, string, error) { return true, "", nil }
func (s *seqBench) RunCandidate(context.Context, string) (Sample, error) {
	v := s.cand[s.ci]
	s.ci++
	return Sample{Throughput: v}, nil
}
func (s *seqBench) RunBaseline(context.Context, string) (Sample, error) {
	v := s.base[s.bi]
	s.bi++
	return Sample{Throughput: v}, nil
}
func (s *seqBench) Unit() string { return "tokens_per_sec" }
