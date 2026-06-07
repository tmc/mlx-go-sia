package chartdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// gen builds a scored (trained, non-zero loss) generation with the given verdict.
func gen(n int, verdict string, loss, ppl float64) WeightsGen {
	return WeightsGen{Gen: n, Verdict: verdict, Trained: true, TestLoss: loss, Perplexity: ppl}
}

// untrained builds an untrained generation (no held-out number).
func untrained(n int, verdict, reason string) WeightsGen {
	return WeightsGen{Gen: n, Verdict: verdict, Trained: false, Reason: reason}
}

func series(gens ...WeightsGen) WeightsSeries { return WeightsSeries{Gens: gens} }

// TestWeightsQualityBars pins the quality glyph line for the real demo series and
// every edge case the design surfaced. Bars rise with the loss (taller = worse);
// a REVISE or untrained generation is a gap; a near-flat span renders mid.
func TestWeightsQualityBars(t *testing.T) {
	const P, R = verdictPass, verdictRevise
	tests := []struct {
		name string
		s    WeightsSeries
		want string
	}{
		{"real all-PASS rising", series(gen(1, P, 2.4183, 11.2), gen(2, P, 2.4920, 12.1), gen(3, P, 2.6322, 13.9)), "▁▃█"},
		{"real REVISE hero keeps bars (overfit, scored)", series(gen(1, P, 2.4183, 11.2), gen(2, R, 2.4920, 12.1), gen(3, R, 2.6322, 13.9)), "▁▃█"},
		{"no-adapter REVISE is a true gap (unscored)", series(gen(1, P, 2.4183, 11.2), untrained(2, R, "no adapter")), "▄·"},
		{"noise flat (no fake climb)", series(gen(1, P, 2.420, 11), gen(2, P, 2.421, 11), gen(3, P, 2.419, 11)), "▄▄▄"},
		{"span just below floor", series(gen(1, P, 2.4183, 11), gen(2, P, 2.4250, 11)), "▄▄"},
		{"span just above floor", series(gen(1, P, 2.40, 11), gen(2, P, 2.43, 11)), "▁█"},
		{"all REVISE but scored: bars shown (rejected, not invalid)", series(gen(1, R, 2.49, 12), gen(2, R, 2.63, 14)), "▁█"},
		{"all unscored: only true gaps", series(untrained(1, R, "no adapter"), untrained(2, R, "eval failed")), "··"},
		{"untrained gen1 excluded from scale", series(untrained(1, P, ""), gen(2, P, 2.49, 12), gen(3, P, 2.63, 14)), "·▁█"},
		{"single PASS, no slope", series(gen(1, P, 2.4183, 11)), "▄"},
		{"four-gen real improvement (gen4 best=short)", series(gen(1, P, 2.42, 11), gen(2, P, 2.49, 12), gen(3, P, 2.63, 14), gen(4, P, 2.30, 10)), "▄▅█▁"},
	}
	for _, tt := range tests {
		if got := tt.s.QualityBars(); got != tt.want {
			t.Errorf("%s: QualityBars() = %q, want %q", tt.name, got, tt.want)
		}
	}
}

// TestWeightsBestAnchor pins the best-PASS anchor and the signed Δ-vs-best, the
// view's headline. The best is the lowest-loss generation that cleared the gate;
// with no PASS generation there is no anchor.
func TestWeightsBestAnchor(t *testing.T) {
	const P, R = verdictPass, verdictRevise

	hero := series(gen(1, P, 2.4423, 11.5), gen(2, R, 2.4744, 11.9), gen(3, R, 2.6210, 13.7))
	if got := hero.bestPassIndex(); got != 0 {
		t.Errorf("REVISE hero: bestPassIndex() = %d, want 0 (gen1)", got)
	}

	// No PASS generation: no crown.
	none := series(gen(1, R, 2.49, 12), gen(2, R, 2.63, 14))
	if got := none.bestPassIndex(); got != -1 {
		t.Errorf("all-REVISE: bestPassIndex() = %d, want -1 (no crown)", got)
	}

	// best = argmin loss among PASS gens, not gen order.
	later := series(gen(1, P, 2.60, 13), gen(2, P, 2.40, 11), gen(3, P, 2.55, 12))
	if got := later.bestPassIndex(); got != 1 {
		t.Errorf("best-not-first: bestPassIndex() = %d, want 1", got)
	}
}

// TestWeightsTableCSVSignParity guards that the table and CSV report the same
// signed Δ-vs-best for every row: positive means worse, identically in both.
func TestWeightsTableCSVSignParity(t *testing.T) {
	const P, R = verdictPass, verdictRevise
	s := series(gen(1, P, 2.4423, 11.5), gen(2, R, 2.4744, 11.9), gen(3, R, 2.6210, 13.7))

	var table strings.Builder
	if err := s.Render(&table); err != nil {
		t.Fatalf("Render: %v", err)
	}
	var csv strings.Builder
	if err := s.WriteCSV(&csv); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}

	// Both must carry the worse-by gen2/gen3 deltas with a leading "+".
	for _, want := range []string{"+0.0321", "+0.1787"} {
		if !strings.Contains(table.String(), want) {
			t.Errorf("table missing signed delta %q", want)
		}
	}
	// CSV stores the bare signed value (no "+" needed for the column), positive.
	for _, want := range []string{"0.0321", "0.1787"} {
		if !strings.Contains(csv.String(), want) {
			t.Errorf("csv missing delta %q", want)
		}
	}
	// Neither may show a worse delta as negative.
	for _, bad := range []string{"-0.0321", "-0.1787"} {
		if strings.Contains(table.String(), bad) || strings.Contains(csv.String(), bad) {
			t.Errorf("a worse loss rendered as negative %q (sign desync)", bad)
		}
	}
}

// TestWeightsFlatness proves the throughput CV detector is NOT applied: the real
// series has a CV below the 0.06 throughput threshold yet still shows movement,
// while genuine near-equality (absolute span < flatLossSpan) flattens.
func TestWeightsFlatness(t *testing.T) {
	const P = verdictPass

	// Real series CV ≈ 0.035 (< 0.06): the CV detector would wrongly flatten it.
	real := series(gen(1, P, 2.4183, 11), gen(2, P, 2.4920, 12), gen(3, P, 2.6322, 14))
	if real.flat() {
		t.Error("real overfit series flagged flat (CV detector leaked into weights view)")
	}
	if bars := real.QualityBars(); bars == "▄▄▄" {
		t.Errorf("real series flattened to %q; want real movement", bars)
	}

	// Deterministic near-equality: absolute span 0.002 < flatLossSpan, flat.
	noise := series(gen(1, P, 2.420, 11), gen(2, P, 2.421, 11), gen(3, P, 2.419, 11))
	if !noise.flat() {
		t.Error("near-equal noise not flagged flat (would amplify into a fake climb)")
	}

	// When flat, the table's Δ column is neutralised so bars and numbers agree.
	var b strings.Builder
	if err := noise.Render(&b); err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(b.String(), "≈0") {
		t.Error("flat series Δ column not neutralised to ≈0 (flat bars vs nonzero delta desync)")
	}
}

// writeTree writes one results.json per gen under a fresh run dir and returns it.
func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for name, body := range files {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "results.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// TestIsWeightsTree pins schema detection: a positive witness routes the tree,
// detection scans past a bare-REVISE file, and an undecidable tree fails closed.
func TestIsWeightsTree(t *testing.T) {
	weights := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"PASS","trained":true,"test_loss":2.42,"metric":"test_loss"}`,
	})
	if w, ok := IsWeightsTree(weights); !ok || !w {
		t.Errorf("metric=test_loss tree: got (weights=%v, ok=%v), want (true, true)", w, ok)
	}

	// test_loss field with no metric key still witnesses weights.
	noMetric := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"PASS","trained":true,"test_loss":2.42}`,
	})
	if w, ok := IsWeightsTree(noMetric); !ok || !w {
		t.Errorf("test_loss-only tree: got (weights=%v, ok=%v), want (true, true)", w, ok)
	}

	throughput := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"PASS","tokens_per_sec":2500,"unit":"tokens_per_sec"}`,
	})
	if w, ok := IsWeightsTree(throughput); !ok || w {
		t.Errorf("throughput tree: got (weights=%v, ok=%v), want (false, true)", w, ok)
	}

	// A bare-REVISE file (neither witness) must not decide; detection scans on to
	// the decisive gen_2.
	bareThenWeights := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"REVISE","reason":"x"}`,
		"gen_2": `{"verdict":"PASS","trained":true,"test_loss":2.42,"metric":"test_loss"}`,
	})
	if w, ok := IsWeightsTree(bareThenWeights); !ok || !w {
		t.Errorf("bare-REVISE then weights: got (weights=%v, ok=%v), want (true, true)", w, ok)
	}

	// No decisive witness anywhere: fail closed (ok=false), do not guess.
	undecidable := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"REVISE","reason":"x"}`,
	})
	if _, ok := IsWeightsTree(undecidable); ok {
		t.Errorf("undecidable tree: got ok=true, want ok=false (fail closed)")
	}

	// A stray tokens_per_sec must not override a positive weights witness.
	strayField := writeTree(t, map[string]string{
		"gen_1": `{"verdict":"PASS","trained":true,"test_loss":2.42,"metric":"test_loss","tokens_per_sec":0}`,
	})
	if w, ok := IsWeightsTree(strayField); !ok || !w {
		t.Errorf("stray tokens_per_sec on weights file: got (weights=%v, ok=%v), want (true, true)", w, ok)
	}
}

// TestWeightsRevisedKeepsNumber confirms a REVISE-overfit generation keeps its
// raw test_loss visible (a plotted-but-rejected point), while an untrained REVISE
// is a true gap with no number.
func TestWeightsRevisedKeepsNumber(t *testing.T) {
	const P, R = verdictPass, verdictRevise
	s := series(
		gen(1, P, 2.4183, 11.2),
		gen(2, R, 2.4920, 12.1),       // overfit REVISE: number present
		untrained(3, R, "no adapter"), // no-adapter REVISE: true gap
	)
	var b strings.Builder
	if err := s.Render(&b); err != nil {
		t.Fatalf("Render: %v", err)
	}
	out := b.String()
	if !strings.Contains(out, "2.4920") {
		t.Error("overfit REVISE dropped its test_loss; the rising number must stay visible")
	}
	// An overfit REVISE (scored) carries a ✗ rejection marker beside its real bar.
	if !strings.Contains(out, "✗") {
		t.Error("overfit REVISE not marked ✗ (rejected-but-present bar)")
	}
	// The untrained gen has no loss to show.
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "3 ") && strings.Contains(line, "2.") {
			t.Errorf("untrained REVISE row shows a fabricated loss: %q", line)
		}
	}
}

// TestWeightsHeroShowsClimb is the money-shot guard: with the overfit gens
// rejected-but-scored, the hero sparkline must show the real rising climb
// (▁▃█), NOT collapse the rejected gens to gaps and hide it.
func TestWeightsHeroShowsClimb(t *testing.T) {
	const P, R = verdictPass, verdictRevise
	hero := series(gen(1, P, 2.4423, 11.5), gen(2, R, 2.4744, 11.9), gen(3, R, 2.6210, 13.7))
	if got := hero.QualityBars(); got != "▁▂█" {
		t.Errorf("hero climb hidden: QualityBars() = %q, want %q (the rise IS the story)", got, "▁▂█")
	}
}
