//go:build darwin

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Gen is one generation of the weight-improvement loop as read from a
// results.json under runs-localtrain/run_N/gen_M. The fields mirror the
// WeightsResult written by mlx-go-sia's localtrain WeightsEvaluator.
//
// HasLoss is false when results.json omits test_loss (omitempty): the dashboard
// renders a gap rather than a zero so a missing value never reads as 0.0.
type Gen struct {
	Run        int
	Gen        int
	Verdict    string  // PASS, REVISE, SKIPPED, ...
	Trained    bool    // adapter was trained and present
	TestLoss   float64 // held-out cross-entropy; lower is better
	HasLoss    bool    // false => test_loss absent, render a gap
	Perplexity float64
	Metric     string
	Reason     string
}

// weightsResult matches the JSON schema written by the weight loop.
type weightsResult struct {
	Verdict    string  `json:"verdict"`
	Trained    bool    `json:"trained"`
	TestLoss   float64 `json:"test_loss"`
	Perplexity float64 `json:"perplexity"`
	Metric     string  `json:"metric"`
	Reason     string  `json:"reason"`
	HeldOut    string  `json:"held_out_dir"`
}

// Passed reports whether the generation's held-out gate accepted it.
func (g Gen) Passed() bool { return strings.EqualFold(g.Verdict, "PASS") }

// poll scans runsRoot for run_N/gen_M/results.json and returns the series sorted
// by (run, gen). It returns (series, true) when a live tree yields at least one
// generation, and (nil, false) when nothing readable is present so the caller can
// fall back to the captured canonical series.
func poll(runsRoot string) ([]Gen, bool) {
	runDirs, err := os.ReadDir(runsRoot)
	if err != nil {
		return nil, false
	}
	var gens []Gen
	for _, rd := range runDirs {
		if !rd.IsDir() {
			continue
		}
		runN, ok := suffixInt(rd.Name(), "run_")
		if !ok {
			continue
		}
		genDirs, err := os.ReadDir(filepath.Join(runsRoot, rd.Name()))
		if err != nil {
			continue
		}
		for _, gd := range genDirs {
			if !gd.IsDir() {
				continue
			}
			genM, ok := suffixInt(gd.Name(), "gen_")
			if !ok {
				continue
			}
			path := filepath.Join(runsRoot, rd.Name(), gd.Name(), "results.json")
			g, ok := readResults(path, runN, genM)
			if !ok {
				continue
			}
			gens = append(gens, g)
		}
	}
	if len(gens) == 0 {
		return nil, false
	}
	sort.Slice(gens, func(i, j int) bool {
		if gens[i].Run != gens[j].Run {
			return gens[i].Run < gens[j].Run
		}
		return gens[i].Gen < gens[j].Gen
	})
	return gens, true
}

// readResults parses one results.json. The "has test_loss" distinction comes from
// re-decoding into a raw map: omitempty means a regressing-but-untrained gen can
// legitimately lack the field, and we must render a gap, not zero.
func readResults(path string, runN, genM int) (Gen, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Gen{}, false
	}
	var wr weightsResult
	if err := json.Unmarshal(data, &wr); err != nil {
		return Gen{}, false
	}
	// Detect presence of test_loss independently of its (possibly zero) value.
	var raw map[string]json.RawMessage
	_ = json.Unmarshal(data, &raw)
	_, hasLoss := raw["test_loss"]
	return Gen{
		Run:        runN,
		Gen:        genM,
		Verdict:    wr.Verdict,
		Trained:    wr.Trained,
		TestLoss:   wr.TestLoss,
		HasLoss:    hasLoss && wr.TestLoss > 0,
		Perplexity: wr.Perplexity,
		Metric:     wr.Metric,
		Reason:     wr.Reason,
	}, true
}

// suffixInt parses the trailing integer of names like "run_3" or "gen_12".
func suffixInt(name, prefix string) (int, bool) {
	if !strings.HasPrefix(name, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(name, prefix))
	if err != nil {
		return 0, false
	}
	return n, true
}

// capturedSeries is the real, canonical P6 weight-loop run (run_1, 3 gens) from
// /tmp/p6-weightloop-FINAL.md. It is used only when no live run tree is readable,
// so the demo still renders the true numbers (never fabricated) when the run
// trees have been cleaned. gen1 generalizes best; gen2/gen3 overfit and the
// held-out gate REVISEs them.
func capturedSeries() []Gen {
	return []Gen{
		{Run: 1, Gen: 1, Verdict: "PASS", Trained: true, TestLoss: 2.4423, HasLoss: true, Perplexity: 11.50, Metric: "test_loss",
			Reason: "conservative first pass generalizes best (best-so-far)"},
		{Run: 1, Gen: 2, Verdict: "REVISE", Trained: true, TestLoss: 2.4744, HasLoss: true, Perplexity: 11.87, Metric: "test_loss",
			Reason: "held-out test_loss 2.4744 > best-so-far 2.4423 (gen 1): overfitting, rejected"},
		{Run: 1, Gen: 3, Verdict: "REVISE", Trained: true, TestLoss: 2.6210, HasLoss: true, Perplexity: 13.75, Metric: "test_loss",
			Reason: "held-out test_loss 2.6210 > best-so-far 2.4423 (gen 1): overfitting, rejected"},
	}
}

// bestSoFar returns the lowest test_loss (and its gen) among trained gens with a
// present test_loss, scanning only up to and including index i (causal). ok is
// false when no such gen exists yet.
func bestSoFar(gens []Gen, i int) (gen int, loss float64, ok bool) {
	for j := 0; j <= i && j < len(gens); j++ {
		g := gens[j]
		if !g.Trained || !g.HasLoss {
			continue
		}
		if !ok || g.TestLoss < loss {
			gen, loss, ok = g.Gen, g.TestLoss, true
		}
	}
	return
}

// fingerprint is a cheap change token for the series: the poller compares it
// between ticks and only triggers a SwiftUI rebuild when the data actually moved.
func fingerprint(gens []Gen) string {
	var b strings.Builder
	for _, g := range gens {
		fmt.Fprintf(&b, "%d/%d:%s:%v:%.4f:%v;", g.Run, g.Gen, g.Verdict, g.Trained, g.TestLoss, g.HasLoss)
	}
	return b.String()
}
