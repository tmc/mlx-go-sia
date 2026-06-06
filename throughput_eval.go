package sia

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Sample is one timed measurement of a runnable. Throughput is
// higher-is-better; supply tokens/sec, ops/sec, or 1e6/µs — any consistent
// unit where a larger number is a faster result.
type Sample struct {
	Throughput float64
}

// Benchmarker supplies the three runnables a [ThroughputEvaluator] drives, each
// keyed by the generation directory. It is the seam P3 (sampler) and P7 (Metal
// kernel) implement: the evaluator owns the honesty discipline (correctness
// before timing, median-of-N, interleaved baseline), the Benchmarker owns only
// "how to run the candidate, the frozen baseline, and the oracle".
//
// The golden oracle, fixed-seed inputs, and timing harness must live in a
// read-only directory captured OUTSIDE the agent's working directory, so the
// agent under optimization cannot widen tolerance, hardcode the golden output,
// or no-op the timing loop. The evaluator never sees those files; it only calls
// these methods.
type Benchmarker interface {
	// Correct runs the generation's candidate against the golden oracle on
	// fixed-seed inputs and reports whether the output matches (exact, or within
	// a frozen tolerance the agent cannot touch). ok=false yields a REVISE
	// verdict — a faster-but-wrong candidate is never a win. The returned error
	// is reserved for the check itself being unable to run; "the candidate is
	// wrong" must be reported as ok=false with a reason, not as an error.
	Correct(ctx context.Context, genDir string) (ok bool, reason string, err error)

	// RunCandidate executes the gen-N candidate once and returns its throughput
	// sample. One run only: the evaluator handles median-of-N and cooldown.
	RunCandidate(ctx context.Context, genDir string) (Sample, error)

	// RunBaseline executes the frozen gen-0 baseline once, interleaved with each
	// candidate run under identical conditions, so the reported metric is the
	// gen-N − gen-0 delta measured at that moment — cancelling thermal and cache
	// drift that would otherwise read as an agent improvement.
	RunBaseline(ctx context.Context, genDir string) (Sample, error)

	// Unit names the throughput unit for results.json, e.g. "tokens_per_sec" or
	// "ops_per_sec". It is advisory metadata for the chart and feedback agent.
	Unit() string
}

// ThroughputEvaluator gates correctness before timing, then reports the median
// throughput of a generation's candidate against an interleaved gen-0 baseline.
// It implements [Evaluator].
//
// The discipline is non-negotiable and lives here, outside the agent's reach:
//
//   - Correctness is checked BEFORE any timing. A candidate that fails the gate
//     is REVISE with no throughput number — speed never overrides correctness.
//   - Timing is median-of-N, not a single hot run; the full distribution
//     (min/median/max and raw samples) is written for the chart.
//   - Every cycle re-runs the frozen gen-0 baseline interleaved with the
//     candidate, so the demo metric is the gen-N − gen-0 delta, which cancels
//     thermal/cache drift.
//
// The zero value is not usable; construct with [NewThroughputEvaluator] or set
// Bench directly. Sleep is injectable for tests; nil uses time.Sleep. Timing
// itself is owned by the [Benchmarker]'s runnables, so the evaluator holds no
// clock of its own.
type ThroughputEvaluator struct {
	// Bench supplies the candidate, baseline, and oracle runnables. Required.
	Bench Benchmarker
	// Runs is N for the median-of-N timing. Values <= 0 default to
	// DefaultThroughputRuns.
	Runs int
	// Cooldown is slept between timing cycles to let the machine settle. Zero
	// means no cooldown.
	Cooldown time.Duration

	// Sleep pauses for the cooldown; nil uses time.Sleep. For tests.
	Sleep func(time.Duration)
}

// DefaultThroughputRuns is the median-of-N sample count when Runs is unset.
const DefaultThroughputRuns = 5

// Verdict strings written to results.json. PASS gates on correctness; the
// advisory throughput number is only present on PASS.
const (
	VerdictPass   = "PASS"
	VerdictRevise = "REVISE"
)

// NewThroughputEvaluator returns a [ThroughputEvaluator] driving b with the
// default run count and no cooldown.
func NewThroughputEvaluator(b Benchmarker) *ThroughputEvaluator {
	return &ThroughputEvaluator{Bench: b}
}

// throughputResults is the results.json schema. The top-level scalars
// (Verdict, CorrectnessOK, TokensPerSec, BaselineTokensPerSec, DeltaTokensPerSec,
// Speedup, Runs, Unit, Reason) are picked up verbatim by the feedback agent's
// metric extraction; the nested distributions are for the chart only.
type throughputResults struct {
	Verdict       string  `json:"verdict"`
	CorrectnessOK bool    `json:"correctness_ok"`
	Reason        string  `json:"reason,omitempty"`
	Unit          string  `json:"unit"`
	Runs          int     `json:"runs"`
	TokensPerSec  float64 `json:"tokens_per_sec,omitempty"`

	BaselineTokensPerSec float64 `json:"baseline_tokens_per_sec,omitempty"`
	DeltaTokensPerSec    float64 `json:"delta_tokens_per_sec,omitempty"`
	Speedup              float64 `json:"speedup,omitempty"`

	Candidate *distribution `json:"candidate,omitempty"`
	Baseline  *distribution `json:"baseline,omitempty"`
}

// distribution is the per-run spread for one runnable.
type distribution struct {
	Min     float64   `json:"min"`
	Median  float64   `json:"median"`
	Max     float64   `json:"max"`
	Samples []float64 `json:"samples"`
}

// Evaluate runs the correctness gate then the timed benchmark for genDir,
// writing results.json and returning an [EvalResult]. Consistent with the
// reference contract, an evaluation outcome — a wrong candidate, or a benchmark
// that could not run — is reported via Status, never as a Go error; a Go error
// is returned only when the evaluator cannot even start (a nil Benchmarker).
func (e *ThroughputEvaluator) Evaluate(ctx context.Context, genDir string) (EvalResult, error) {
	if e.Bench == nil {
		return EvalResult{}, fmt.Errorf("throughput evaluator: nil Benchmarker")
	}

	// 1. Correctness gate — before any timing.
	ok, reason, err := e.Bench.Correct(ctx, genDir)
	if err != nil {
		return e.fail(genDir, fmt.Sprintf("correctness check could not run: %v", err))
	}
	if !ok {
		if reason == "" {
			reason = "candidate output did not match the golden oracle"
		}
		return e.revise(genDir, reason)
	}

	// 2. Median-of-N timing with the gen-0 baseline interleaved each cycle.
	n := e.Runs
	if n <= 0 {
		n = DefaultThroughputRuns
	}
	cand := make([]float64, 0, n)
	base := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		if i > 0 {
			e.sleep(e.Cooldown)
		}
		// Baseline first, then candidate, back-to-back under identical
		// conditions so the per-cycle delta cancels drift.
		bs, err := e.Bench.RunBaseline(ctx, genDir)
		if err != nil {
			return e.fail(genDir, fmt.Sprintf("baseline run %d failed: %v", i, err))
		}
		cs, err := e.Bench.RunCandidate(ctx, genDir)
		if err != nil {
			return e.fail(genDir, fmt.Sprintf("candidate run %d failed: %v", i, err))
		}
		base = append(base, bs.Throughput)
		cand = append(cand, cs.Throughput)
	}

	// 3. PASS: report the median candidate and the interleaved-baseline delta.
	cd := summarize(cand)
	bd := summarize(base)
	res := throughputResults{
		Verdict:              VerdictPass,
		CorrectnessOK:        true,
		Unit:                 e.Bench.Unit(),
		Runs:                 n,
		TokensPerSec:         cd.Median,
		BaselineTokensPerSec: bd.Median,
		DeltaTokensPerSec:    cd.Median - bd.Median,
		Speedup:              speedup(cd.Median, bd.Median),
		Candidate:            &cd,
		Baseline:             &bd,
	}
	return e.write(genDir, res)
}

// revise writes a REVISE results.json and reports EvalSuccess: the evaluation
// itself ran fine, the candidate simply needs another pass.
func (e *ThroughputEvaluator) revise(genDir, reason string) (EvalResult, error) {
	return e.write(genDir, throughputResults{
		Verdict:       VerdictRevise,
		CorrectnessOK: false,
		Reason:        reason,
		Unit:          e.unit(),
	})
}

// fail reports EvalError when the benchmark machinery could not run, after
// recording the reason in results.json for the feedback agent.
func (e *ThroughputEvaluator) fail(genDir, reason string) (EvalResult, error) {
	res := throughputResults{
		Verdict:       VerdictRevise,
		CorrectnessOK: false,
		Reason:        reason,
		Unit:          e.unit(),
	}
	if _, werr := e.write(genDir, res); werr != nil {
		return EvalResult{Status: EvalError, Reason: reason}, nil
	}
	return EvalResult{
		Status:      EvalError,
		Reason:      reason,
		ResultsPath: filepath.Join(genDir, NameResultsJSON),
	}, nil
}

// write marshals res into genDir/results.json and returns an EvalSuccess result
// pointing at it. A write failure is reported as EvalError, not a Go error.
func (e *ThroughputEvaluator) write(genDir string, res throughputResults) (EvalResult, error) {
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return EvalResult{Status: EvalError, Reason: fmt.Sprintf("marshal results: %v", err)}, nil
	}
	path := filepath.Join(genDir, NameResultsJSON)
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return EvalResult{Status: EvalError, Reason: fmt.Sprintf("write results.json: %v", err)}, nil
	}
	return EvalResult{
		Status:      EvalSuccess,
		ResultsPath: path,
		Output:      string(data),
	}, nil
}

// unit returns the Benchmarker's unit, tolerating a nil Bench for the error
// paths.
func (e *ThroughputEvaluator) unit() string {
	if e.Bench == nil {
		return ""
	}
	return e.Bench.Unit()
}

func (e *ThroughputEvaluator) sleep(d time.Duration) {
	if d <= 0 {
		return
	}
	if e.Sleep != nil {
		e.Sleep(d)
		return
	}
	time.Sleep(d)
}

// summarize returns the min/median/max and raw samples of xs. xs must be
// non-empty.
func summarize(xs []float64) distribution {
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	return distribution{
		Min:     sorted[0],
		Median:  median(sorted),
		Max:     sorted[len(sorted)-1],
		Samples: append([]float64(nil), xs...),
	}
}

// median returns the median of a sorted, non-empty slice. For an even count it
// averages the two middle values.
func median(sorted []float64) float64 {
	n := len(sorted)
	mid := n / 2
	if n%2 == 1 {
		return sorted[mid]
	}
	return (sorted[mid-1] + sorted[mid]) / 2
}

// speedup is the candidate-over-baseline ratio, guarding a zero baseline.
func speedup(cand, base float64) float64 {
	if base == 0 {
		return 0
	}
	return cand / base
}
