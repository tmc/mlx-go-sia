package main

import (
	"encoding/json"
	"fmt"
	"os"

	sia "github.com/tmc/sia-apple-silicon"
)

// genResults is the subset of each generation's results.json this command prints
// as the demo's climbing-throughput series. It mirrors the top-level scalars the
// ThroughputEvaluator writes.
type genResults struct {
	Verdict       string  `json:"verdict"`
	CorrectnessOK bool    `json:"correctness_ok"`
	Reason        string  `json:"reason"`
	Unit          string  `json:"unit"`
	TokensPerSec  float64 `json:"tokens_per_sec"`
	Baseline      float64 `json:"baseline_tokens_per_sec"`
	Delta         float64 `json:"delta_tokens_per_sec"`
	Speedup       float64 `json:"speedup"`
}

// reportThroughput prints the per-generation throughput series — the data behind
// the demo chart — by reading each generation's results.json. The reported
// number is the gen-N − gen-0 delta, which cancels thermal/cache drift; the
// chart should plot Delta (or Speedup), not raw tokens/sec.
func reportThroughput(layout sia.RunLayout, res sia.RunResult) {
	fmt.Println()
	fmt.Println("P3 throughput series (gen-N vs interleaved gen-0 baseline):")
	fmt.Printf("  %-5s %-7s %-14s %-14s %-12s %-8s\n", "gen", "verdict", "tokens/sec", "baseline", "delta", "speedup")
	for _, g := range res.Generations {
		gr, err := readGenResults(layout.ResultsJSON(g.Gen))
		if err != nil {
			fmt.Printf("  %-5d (no results.json: %v)\n", g.Gen, err)
			continue
		}
		if gr.Verdict == sia.VerdictPass {
			fmt.Printf("  %-5d %-7s %-14.2f %-14.2f %-+12.2f %-8.3f\n",
				g.Gen, gr.Verdict, gr.TokensPerSec, gr.Baseline, gr.Delta, gr.Speedup)
		} else {
			reason := gr.Reason
			if reason == "" {
				reason = "correctness gate not passed"
			}
			fmt.Printf("  %-5d %-7s %s\n", g.Gen, gr.Verdict, reason)
		}
	}
	fmt.Printf("\ncontext: %s\n", res.ContextPath)
	fmt.Println("chart data: each gen's results.json (delta_tokens_per_sec / speedup), oracle held read-only outside the agent's reach")
}

func readGenResults(path string) (genResults, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return genResults{}, err
	}
	var gr genResults
	if err := json.Unmarshal(data, &gr); err != nil {
		return genResults{}, err
	}
	return gr, nil
}
