package sia_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
)

// demoBench is a stand-in Benchmarker: it reports the candidate as correct and
// returns fixed throughputs, so the example is deterministic without a real
// model or kernel. P3 supplies a sampler-backed Benchmarker, P7 a Metal-kernel
// one; both keep the golden oracle in a read-only directory outside the agent's
// working dir.
type demoBench struct{}

func (demoBench) Correct(context.Context, string) (bool, string, error) { return true, "", nil }
func (demoBench) RunCandidate(context.Context, string) (sia.Sample, error) {
	return sia.Sample{Throughput: 1320}, nil
}
func (demoBench) RunBaseline(context.Context, string) (sia.Sample, error) {
	return sia.Sample{Throughput: 1100}, nil
}
func (demoBench) Unit() string { return "tokens_per_sec" }

// ExampleThroughputEvaluator gates correctness then reports the median
// throughput of a generation against an interleaved gen-0 baseline, writing
// results.json for the feedback agent and the chart.
func ExampleThroughputEvaluator() {
	genDir, _ := os.MkdirTemp("", "sia-gen")
	defer os.RemoveAll(genDir)

	eval := sia.NewThroughputEvaluator(demoBench{})
	eval.Runs = 5 // median-of-N

	res, err := eval.Evaluate(context.Background(), genDir)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	data, _ := os.ReadFile(filepath.Join(genDir, sia.NameResultsJSON))
	var r struct {
		Verdict      string  `json:"verdict"`
		TokensPerSec float64 `json:"tokens_per_sec"`
		Baseline     float64 `json:"baseline_tokens_per_sec"`
		Delta        float64 `json:"delta_tokens_per_sec"`
	}
	json.Unmarshal(data, &r)

	fmt.Println("status:", res.Status)
	fmt.Println("verdict:", r.Verdict)
	fmt.Printf("tokens/sec: %.0f (baseline %.0f, delta %.0f)\n", r.TokensPerSec, r.Baseline, r.Delta)
	// Output:
	// status: success
	// verdict: PASS
	// tokens/sec: 1320 (baseline 1100, delta 220)
}
