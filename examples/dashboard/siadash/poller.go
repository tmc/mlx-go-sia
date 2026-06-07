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
// The loop runs in one of two metric modes and the dashboard renders whichever
// the run emitted: held-out test_loss (lower is better) or held-out accuracy
// (higher is better). HasLoss / HasAccuracy is false when the corresponding
// field is omitted (omitempty), so a missing value renders as a gap, never 0.0.
type Gen struct {
	Run         int
	Gen         int
	Verdict     string  // PASS, REVISE, SKIPPED, ...
	Trained     bool    // adapter was trained and present
	TestLoss    float64 // held-out cross-entropy; lower is better
	HasLoss     bool    // false => test_loss absent, render a gap
	Accuracy    float64 // held-out accuracy in [0,1]; higher is better
	HasAccuracy bool    // false => accuracy absent, render a gap
	Correct     int     // held-out items scored correct (accuracy mode)
	Total       int     // held-out items scored (accuracy mode)
	Perplexity  float64
	Metric      string
	Reason      string
}

// HigherBetter reports whether this generation's plotted metric improves upward.
// Accuracy climbs; test_loss descends. A gen with accuracy present is an
// accuracy gen even if a (legacy) loss field is also set.
func (g Gen) HigherBetter() bool { return g.HasAccuracy }

// Value returns the plotted metric value and whether it is present. Accuracy
// takes precedence over loss when both happen to be set.
func (g Gen) Value() (float64, bool) {
	if g.HasAccuracy {
		return g.Accuracy, true
	}
	if g.HasLoss {
		return g.TestLoss, true
	}
	return 0, false
}

// weightsResult matches the JSON schema written by the weight loop. Both
// test_loss and accuracy are optional (omitempty): a run emits one or the other
// depending on its metric mode.
type weightsResult struct {
	Verdict    string  `json:"verdict"`
	Trained    bool    `json:"trained"`
	TestLoss   float64 `json:"test_loss"`
	Accuracy   float64 `json:"accuracy"`
	Correct    int     `json:"correct"`
	Total      int     `json:"total"`
	Perplexity float64 `json:"perplexity"`
	Metric     string  `json:"metric"`
	Reason     string  `json:"reason"`
	HeldOut    string  `json:"held_out_dir"`
}

// Passed reports whether the generation's held-out gate accepted it.
func (g Gen) Passed() bool { return strings.EqualFold(g.Verdict, "PASS") }

// metricLabel reports the human label and direction caption for a series. When
// any gen carries accuracy the series is an accuracy series (higher is better);
// otherwise it is a test_loss series (lower is better). An empty series defaults
// to test_loss so the captured fallback keeps its established wording.
func metricLabel(gens []Gen) (axis, caption, headline string, higherBetter bool) {
	for _, g := range gens {
		if g.HasAccuracy {
			return "held-out accuracy (higher = better)",
				"held-out accuracy · higher is better",
				"best accuracy", true
		}
	}
	return "held-out test_loss (lower = better)",
		"held-out test_loss · lower is better",
		"best test_loss", false
}

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
	sortGens(gens)
	return gens, true
}

func sortGens(gens []Gen) {
	sort.Slice(gens, func(i, j int) bool {
		if gens[i].Run != gens[j].Run {
			return gens[i].Run < gens[j].Run
		}
		return gens[i].Gen < gens[j].Gen
	})
}

// readResults parses one results.json. The "has X" distinction comes from
// re-decoding into a raw map: omitempty means a regressing-but-untrained gen can
// legitimately lack a metric field, and we must render a gap, not zero.
func readResults(path string, runN, genM int) (Gen, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Gen{}, false
	}
	return decodeResult(data, runN, genM)
}

// decodeResult turns one results.json blob into a Gen. It is shared by the live
// poller and the replay engine so both read exactly the same schema.
func decodeResult(data []byte, runN, genM int) (Gen, bool) {
	var wr weightsResult
	if err := json.Unmarshal(data, &wr); err != nil {
		return Gen{}, false
	}
	// Detect field presence independently of the (possibly zero) value.
	var raw map[string]json.RawMessage
	_ = json.Unmarshal(data, &raw)
	_, hasLoss := raw["test_loss"]
	_, hasAcc := raw["accuracy"]
	return Gen{
		Run:         runN,
		Gen:         genM,
		Verdict:     wr.Verdict,
		Trained:     wr.Trained,
		TestLoss:    wr.TestLoss,
		HasLoss:     hasLoss && wr.TestLoss > 0,
		Accuracy:    wr.Accuracy,
		HasAccuracy: hasAcc,
		Correct:     wr.Correct,
		Total:       wr.Total,
		Perplexity:  wr.Perplexity,
		Metric:      wr.Metric,
		Reason:      wr.Reason,
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

// capturedSeries is the real, canonical P6 weight-loop run (run_1, 3 gens). It is
// used only when no live run tree is readable, so the demo still renders the true
// numbers (never fabricated) when the run trees have been cleaned. gen1 generalizes
// best; gen2/gen3 overfit and the held-out gate REVISEs them.
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

// bestSoFar returns the best metric value (and its gen) among trained gens with a
// present value, scanning only up to and including index i (causal). Direction
// follows the series: accuracy maximizes, test_loss minimizes. ok is false when
// no such gen exists yet.
func bestSoFar(gens []Gen, i int) (gen int, val float64, ok bool) {
	_, _, _, higher := metricLabel(gens)
	for j := 0; j <= i && j < len(gens); j++ {
		g := gens[j]
		if !g.Trained {
			continue
		}
		v, has := g.Value()
		if !has {
			continue
		}
		if !ok || better(v, val, higher) {
			gen, val, ok = g.Gen, v, true
		}
	}
	return
}

// better reports whether candidate c improves on the current best b for the
// series direction (higher==true => larger is better).
func better(c, b float64, higher bool) bool {
	if higher {
		return c > b
	}
	return c < b
}

// worse reports whether candidate c is strictly worse than best b for the series
// direction. Equal is not worse (a tie holds the best, not a regression).
func worse(c, b float64, higher bool) bool {
	if higher {
		return c < b
	}
	return c > b
}

// fingerprint is a cheap change token for the series: the poller compares it
// between ticks and only triggers a SwiftUI rebuild when the data actually moved.
func fingerprint(gens []Gen) string {
	var b strings.Builder
	for _, g := range gens {
		v, _ := g.Value()
		fmt.Fprintf(&b, "%d/%d:%s:%v:%.4f:%v%v;", g.Run, g.Gen, g.Verdict, g.Trained, v, g.HasLoss, g.HasAccuracy)
	}
	return b.String()
}
