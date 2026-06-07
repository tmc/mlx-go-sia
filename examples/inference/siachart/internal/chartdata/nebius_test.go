package chartdata

import (
	"strings"
	"testing"
)

// nbDist builds a distribution with the given min/max plus two samples at the
// extremes, so the all-pairs unanimity check in classifySpeed has data to read.
// min and max bound the range the win/parity test compares; the two samples make
// the pairwise test agree with that range.
func nbDist(min, max float64) *Distribution {
	return &Distribution{Min: min, Max: max, Median: (min + max) / 2, Samples: []float64{min, max}}
}

// nbPass builds a PASS generation with a candidate and baseline distribution and
// a recorded speedup.
func nbPass(n int, speedup float64, cand, base *Distribution) Gen {
	return Gen{
		Gen: n, Verdict: verdictPass, CorrectnessOK: true,
		Speedup: speedup, Candidate: cand, BaselineDist: base,
	}
}

// nbRevise builds a REVISE generation that did not run (no distributions).
func nbRevise(n int, reason string) Gen {
	return Gen{Gen: n, Verdict: verdictRevise, CorrectnessOK: false, Reason: reason}
}

// TestClassifySpeed pins the win-vs-parity call, the view's core honesty: a win
// only when the candidate's slowest run clears the baseline's fastest. The real
// Nebius generations (whose distributions overlap) must classify as parity, and a
// recorded speedup above 1.0 must never alone make a win.
func TestClassifySpeed(t *testing.T) {
	tests := []struct {
		name string
		g    Gen
		want speedVerdict
	}{
		// The real deepseek gen_4: recorded 1.009x, but candidate min 2424.7 is far
		// below baseline max 3186.3 — overlapping, parity.
		{"deepseek gen4 parity", nbPass(4, 1.0094, nbDist(2424.687, 3235.344), nbDist(2822.956, 3186.258)), speedParity},
		// The real qwen PASS gens: every one overlaps, despite recorded speedups up
		// to 1.162x — the headline number must not override the ranges.
		{"qwen gen2 parity", nbPass(2, 1.0730, nbDist(2114.359, 2515.962), nbDist(2121.257, 2324.601)), speedParity},
		{"qwen gen3 parity", nbPass(3, 1.0372, nbDist(1941.55, 2367.186), nbDist(1915.707, 2347.895)), speedParity},
		{"qwen gen4 parity", nbPass(4, 1.1617, nbDist(2243.886, 2552.253), nbDist(2131.176, 2334.832)), speedParity},
		// A genuinely separated win: candidate min strictly above baseline max.
		{"separated win", nbPass(2, 1.45, nbDist(3300, 3500), nbDist(2200, 2300)), speedWin},
		// Boundary: candidate min exactly equals baseline max is NOT a win (touching
		// ranges are not separated; the rule is strictly greater).
		{"touching ranges is parity", nbPass(2, 1.30, nbDist(2300, 2600), nbDist(2000, 2300)), speedParity},
		// A REVISE that did not run has no distribution: no speed verdict.
		{"revise has no speed", nbRevise(1, "cannot slice h"), speedNone},
		// The real pi-agent gen_1: correct PASS, recorded 0.698x, ranges overlap
		// (cand.max 402 > base.min 296) so not separated either way — the recorded
		// sub-1.0 ratio makes it slower, not parity.
		{"pi-agent gen1 correct but slower", nbPass(1, 0.6979, nbDist(182.817, 402.48), nbDist(296.074, 776.588)), speedSlower},
		// A PASS recorded just under 1.0 but inside the noise band is parity.
		{"near-1.0 stays parity", nbPass(1, 0.99, nbDist(2400, 3200), nbDist(2800, 3180)), speedParity},
	}
	for _, tt := range tests {
		if got := classifySpeed(tt.g); got != tt.want {
			t.Errorf("%s: classifySpeed() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// TestSpeedLabelNeverFakesWin guards the most dangerous misread: a parity PASS
// must show its number but call itself parity, never "win"; a separated win shows
// "win" with the number.
func TestSpeedLabelNeverFakesWin(t *testing.T) {
	parity := nbPass(4, 1.0094, nbDist(2424.687, 3235.344), nbDist(2822.956, 3186.258))
	got := speedLabel(parity)
	if !strings.Contains(got, "parity") {
		t.Errorf("parity label = %q, want it to say parity", got)
	}
	// The directional recorded ratio must NOT appear: a lone "1.009x" reads as a
	// small win when skimmed. The label describes the overlap geometry instead.
	// ("no measurable win" is fine — it denies a win; "x win" would claim one.)
	if strings.Contains(got, "x win") || strings.Contains(got, "1.009") {
		t.Errorf("parity label = %q, must not show a directional ratio or claim a win", got)
	}
	if !strings.Contains(got, "overlap") {
		t.Errorf("parity label = %q, want it to describe the range overlap", got)
	}

	// A separated win (above the floor, all pairs faster) shows the speedup as a win.
	win := nbPass(2, 1.45, nbDist(3300, 3500), nbDist(2200, 2300))
	got = speedLabel(win)
	if !strings.Contains(got, "win") || !strings.Contains(got, "1.450") {
		t.Errorf("win label = %q, want it to say win with the speedup", got)
	}

	if got := speedLabel(nbRevise(1, "boom")); got != "—" {
		t.Errorf("no-timing label = %q, want %q", got, "—")
	}
}

// TestNoiseFloorRejectsHairSeparation pins the false-win guard: a separated pair
// whose margin is below the 3% floor is parity, not a win, so a chance razor-thin
// non-overlap at small sample counts never prints a celebratory win.
func TestNoiseFloorRejectsHairSeparation(t *testing.T) {
	// cand.min 3030 just clears base.max 3000 (1.01x), below the 1.03 floor.
	hair := nbPass(2, 1.05, nbDist(3030, 3200), nbDist(2800, 3000))
	if got := classifySpeed(hair); got != speedParity {
		t.Errorf("hair-separation classifySpeed() = %v, want speedParity (below noise floor)", got)
	}
}

// TestOutlierVetoConservative documents and locks the conservative cost: a
// high-median win with one throttled candidate run dipping into the baseline
// range is parity by design — the verdict is not promoted to win when the
// distributions are not separable, even though the median looks fast.
func TestOutlierVetoConservative(t *testing.T) {
	throttled := &Distribution{Min: 2900, Max: 3580, Median: 3570, Samples: []float64{3565, 3580, 3570, 3575, 2900}}
	base := &Distribution{Min: 3050, Max: 3150, Median: 3100, Samples: []float64{3060, 3100, 3150, 3090, 3050}}
	g := Gen{Gen: 1, Verdict: verdictPass, CorrectnessOK: true, Speedup: 1.15, Candidate: throttled, BaselineDist: base}
	if got := classifySpeed(g); got != speedParity {
		t.Errorf("outlier-vetoed classifySpeed() = %v, want speedParity (one throttled run breaks separation)", got)
	}
}

// TestTallyCountsHonestly pins the gate-record summary line: bugs REVISE'd,
// correct candidates PASS'd, and measurable wins — which is zero on a parity run.
func TestTallyCountsHonestly(t *testing.T) {
	deepseek := NebiusSeries{Gens: []Gen{
		nbRevise(1, "cannot slice h"),
		nbRevise(2, "cannot use rng"),
		nbRevise(3, "panic: index out of range"),
		nbPass(4, 1.0094, nbDist(2424.687, 3235.344), nbDist(2822.956, 3186.258)),
	}}
	if r, p, w := deepseek.tally(); r != 3 || p != 1 || w != 0 {
		t.Errorf("deepseek tally = (%d,%d,%d), want (3,1,0)", r, p, w)
	}

	qwen := NebiusSeries{Gens: []Gen{
		nbRevise(1, "cannot use &rng"),
		nbPass(2, 1.0730, nbDist(2114.359, 2515.962), nbDist(2121.257, 2324.601)),
		nbPass(3, 1.0372, nbDist(1941.55, 2367.186), nbDist(1915.707, 2347.895)),
		nbPass(4, 1.1617, nbDist(2243.886, 2552.253), nbDist(2131.176, 2334.832)),
	}}
	if r, p, w := qwen.tally(); r != 1 || p != 3 || w != 0 {
		t.Errorf("qwen tally = (%d,%d,%d), want (1,3,0)", r, p, w)
	}

	// A run with one genuinely separated win counts it.
	won := NebiusSeries{Gens: []Gen{nbPass(1, 1.45, nbDist(3300, 3500), nbDist(2200, 2300))}}
	if r, p, w := won.tally(); r != 0 || p != 1 || w != 1 {
		t.Errorf("won tally = (%d,%d,%d), want (0,1,1)", r, p, w)
	}
}

// TestBugSummary pins the bug extraction: a wrapped tool error's generic prefix
// is dropped in favour of the Go compiler diagnostic or the runtime panic — the
// real bug the gate caught.
func TestBugSummary(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   string
	}{
		{
			"compiler diagnostic past the location",
			"candidate did not run: run candidate (tokens): exit status 1: # command-line-arguments\n/tmp/x/gen_1/candidate.go:102:7: cannot slice h (variable of type *minHeap)",
			"cannot slice h (variable of type *minHeap)",
		},
		{
			"runtime panic on the prefix line",
			"candidate did not run: run candidate (tokens): exit status 1: panic: runtime error: index out of range [64] with length 64\n\ngoroutine 1 [running]:",
			"panic: runtime error: index out of range [64] with length 64",
		},
		{
			"no diagnostic falls back to first line",
			"candidate did not run: some opaque failure",
			"candidate did not run: some opaque failure",
		},
	}
	for _, tt := range tests {
		if got := bugSummary(tt.reason, 120); got != tt.want {
			t.Errorf("%s: bugSummary() = %q, want %q", tt.name, got, tt.want)
		}
	}
}

// TestNebiusRenderHonest renders the real deepseek series and asserts the honest
// invariants are present and the dishonest reads are absent: the gate summary
// reports zero measurable speedups, the bug is surfaced, the PASS is celebrated
// as correct, and the word "win" never appears (no fake speed win on a parity
// run).
func TestNebiusRenderHonest(t *testing.T) {
	s := NebiusSeries{Model: "deepseek", Gens: []Gen{
		nbRevise(1, "candidate did not run: exit status 1: # command-line-arguments\n/tmp/x/gen_1/candidate.go:102:7: cannot slice h (variable of type *minHeap)"),
		nbPass(4, 1.0094, nbDist(2424.687, 3235.344), nbDist(2822.956, 3186.258)),
	}}
	var b strings.Builder
	if err := s.Render(&b); err != nil {
		t.Fatalf("Render: %v", err)
	}
	out := b.String()

	wantPresent := []string{
		"verdict progression",                              // header is the gate story, not throughput
		"credited 0 measurable speedup",                    // gate gave out no win
		"cannot slice h (variable of type *minHeap)",       // real bug surfaced
		"correct (token-identical), no measurable speedup", // PASS celebrated, no win
		"parity — ranges overlap",                          // overlap geometry, not a directional ratio
	}
	for _, w := range wantPresent {
		if !strings.Contains(out, w) {
			t.Errorf("render missing %q\n--- got ---\n%s", w, out)
		}
	}
	// No fake win, no skimmable directional ratio, and never the "throughput
	// series" framing on this view.
	if strings.Contains(out, "x win") {
		t.Errorf("render must not claim a win on a parity run\n--- got ---\n%s", out)
	}
	if strings.Contains(out, "1.009") {
		t.Errorf("render must not surface the directional 1.009x ratio\n--- got ---\n%s", out)
	}
	if strings.Contains(out, "throughput series") {
		t.Errorf("correctness view must not use the throughput-series framing\n--- got ---\n%s", out)
	}
}

// TestNebiusRenderShowsRealWin pins the other direction: a genuinely separated
// generation is shown as a win with its speedup, so the view is not biased toward
// always saying parity.
func TestNebiusRenderShowsRealWin(t *testing.T) {
	s := NebiusSeries{Gens: []Gen{nbPass(1, 1.45, nbDist(3300, 3500), nbDist(2200, 2300))}}
	var b strings.Builder
	if err := s.Render(&b); err != nil {
		t.Fatalf("Render: %v", err)
	}
	out := b.String()
	if !strings.Contains(out, "1.450x win") {
		t.Errorf("render of a separated win missing the win label\n--- got ---\n%s", out)
	}
	if !strings.Contains(out, "credited 1 measurable speedup") {
		t.Errorf("render summary should credit the real win\n--- got ---\n%s", out)
	}
}

// TestNebiusRenderSlower pins the third speed flavor (the real pi-agent headline):
// a correct PASS recorded below 1.0 is shown as a regression that the gate credits
// for correctness but not speed — never "parity", never a win. This is the
// strongest anti-Goodhart point: the gate visibly declines to credit a regression.
func TestNebiusRenderSlower(t *testing.T) {
	s := NebiusSeries{Model: "pi-agent → DeepSeek", Gens: []Gen{
		nbPass(1, 0.6979, nbDist(182.817, 402.48), nbDist(296.074, 776.588)),
		nbRevise(2, "candidate did not run: exit status 1: # command-line-arguments\n/tmp/x/gen_2/candidate.go:205:5: declared and not used: maxIdx"),
		nbRevise(3, "candidate did not run: exit status 1: # command-line-arguments\n/tmp/x/gen_3/candidate.go:153:12: undefined: time"),
	}}
	var b strings.Builder
	if err := s.Render(&b); err != nil {
		t.Fatalf("Render: %v", err)
	}
	out := b.String()
	wantPresent := []string{
		"0.70x — slower, a regression (no credit)", // the slower speed cell
		"credited for correctness, not speed",      // the what-happened note
		"declared and not used: maxIdx",            // gen2 real bug
		"undefined: time",                          // gen3 real bug
		"credited 0 measurable speedup",            // gate gave out no win
	}
	for _, w := range wantPresent {
		if !strings.Contains(out, w) {
			t.Errorf("render missing %q\n--- got ---\n%s", w, out)
		}
	}
	// The gen_1 PASS row must not be labelled parity (the header explains the
	// parity rule, so check the row, not the whole output).
	if strings.Contains(out, "parity — ranges overlap") {
		t.Errorf("a recorded-slower PASS must not read as parity\n--- got ---\n%s", out)
	}
	if strings.Contains(out, "x win") {
		t.Errorf("a slower PASS must never claim a win\n--- got ---\n%s", out)
	}
}

// TestNebiusCSVParity asserts the CSV carries the data-derived speed verdict and
// is_real_win, not the recorded scalar alone — so a downstream consumer reads the
// same honest call as the table.
func TestNebiusCSVParity(t *testing.T) {
	s := NebiusSeries{Gens: []Gen{
		nbRevise(1, "cannot slice h"),
		nbPass(4, 1.0094, nbDist(2424.687, 3235.344), nbDist(2822.956, 3186.258)),
	}}
	var b strings.Builder
	if err := s.WriteCSV(&b); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	rows := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(rows) != 3 {
		t.Fatalf("CSV rows = %d, want 3 (header + 2)", len(rows))
	}
	if !strings.HasPrefix(rows[0], "gen,verdict,correctness_ok,speed_verdict,recorded_speedup,is_real_win") {
		t.Errorf("CSV header = %q", rows[0])
	}
	// gen4 PASS-at-parity: speed_verdict=parity, is_real_win=false, recorded number present.
	if !strings.Contains(rows[2], ",parity,") || !strings.Contains(rows[2], ",false,") {
		t.Errorf("gen4 row should be parity/false: %q", rows[2])
	}
	// REVISE gen1: speed_verdict=none, no recorded speedup.
	if !strings.Contains(rows[1], ",none,") {
		t.Errorf("gen1 row should be speed_verdict none: %q", rows[1])
	}
}
