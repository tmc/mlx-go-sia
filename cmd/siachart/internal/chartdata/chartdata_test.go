package chartdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeRun builds a synthetic run tree: runDir/gen_<n>/results.json for each
// supplied JSON body. It mirrors what the ThroughputEvaluator writes, so the
// reader is tested against the real on-disk schema without running a model.
func writeRun(t *testing.T, gens map[int]string) string {
	t.Helper()
	runDir := t.TempDir()
	for n, body := range gens {
		gd := filepath.Join(runDir, "gen_"+itoa(n))
		if err := os.MkdirAll(gd, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(gd, "results.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return runDir
}

func itoa(n int) string {
	// small local helper so the test does not depend on strconv import noise
	return strings.TrimSpace(sprintInt(n))
}
func sprintInt(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

const passGen = `{
  "verdict": "PASS", "correctness_ok": true, "unit": "tokens_per_sec", "runs": 5,
  "tokens_per_sec": 120, "baseline_tokens_per_sec": 100,
  "delta_tokens_per_sec": 20, "speedup": 1.2,
  "candidate": {"min": 110, "median": 120, "max": 135, "samples": [110,120,135,118,121]},
  "baseline": {"min": 95, "median": 100, "max": 106, "samples": [95,100,106,99,101]}
}`

const reviseGen = `{
  "verdict": "REVISE", "correctness_ok": false, "unit": "tokens_per_sec", "runs": 5,
  "reason": "token 3 mismatched golden"
}`

const fasterGen = `{
  "verdict": "PASS", "correctness_ok": true, "unit": "tokens_per_sec", "runs": 5,
  "tokens_per_sec": 150, "baseline_tokens_per_sec": 100,
  "delta_tokens_per_sec": 50, "speedup": 1.5,
  "candidate": {"min": 140, "median": 150, "max": 162, "samples": [140,150,162,148,151]},
  "baseline": {"min": 96, "median": 100, "max": 104, "samples": [96,100,104,99,101]}
}`

func TestReadSeriesOrdersAndParses(t *testing.T) {
	runDir := writeRun(t, map[int]string{0: passGen, 1: reviseGen, 2: fasterGen})
	s, err := ReadSeries(runDir)
	if err != nil {
		t.Fatalf("ReadSeries: %v", err)
	}
	if len(s.Gens) != 3 {
		t.Fatalf("gens = %d, want 3", len(s.Gens))
	}
	for i, want := range []int{0, 1, 2} {
		if s.Gens[i].Gen != want {
			t.Errorf("gen[%d].Gen = %d, want %d (must be ascending)", i, s.Gens[i].Gen, want)
		}
	}
	if !s.Gens[0].Passed() || s.Gens[1].Passed() || !s.Gens[2].Passed() {
		t.Errorf("Passed() mask = %v/%v/%v, want true/false/true",
			s.Gens[0].Passed(), s.Gens[1].Passed(), s.Gens[2].Passed())
	}
	if s.Gens[2].Candidate == nil || s.Gens[2].Candidate.Max != 162 {
		t.Errorf("gen2 candidate distribution not parsed: %+v", s.Gens[2].Candidate)
	}
	if s.Gens[1].Reason != "token 3 mismatched golden" {
		t.Errorf("revise reason = %q", s.Gens[1].Reason)
	}
}

func TestReadSeriesSkipsUnparseableAndNonGenDirs(t *testing.T) {
	runDir := writeRun(t, map[int]string{0: passGen, 1: fasterGen})
	// A non-gen dir and a gen dir with broken JSON should both be ignored.
	os.MkdirAll(filepath.Join(runDir, "_oracle"), 0o755)
	gd := filepath.Join(runDir, "gen_9")
	os.MkdirAll(gd, 0o755)
	os.WriteFile(filepath.Join(gd, "results.json"), []byte("{not json"), 0o644)

	s, err := ReadSeries(runDir)
	if err != nil {
		t.Fatalf("ReadSeries: %v", err)
	}
	if len(s.Gens) != 2 {
		t.Fatalf("gens = %d, want 2 (broken gen_9 + _oracle skipped)", len(s.Gens))
	}
}

func TestReadSeriesEmpty(t *testing.T) {
	if _, err := ReadSeries(t.TempDir()); err == nil {
		t.Fatal("ReadSeries on empty dir = nil error, want error")
	}
}

func TestUnitFromResults(t *testing.T) {
	// For P7 the unit is ops_per_sec even though the field is named tokens_per_sec.
	body := strings.Replace(passGen, `"unit": "tokens_per_sec"`, `"unit": "ops_per_sec"`, 1)
	s, err := ReadSeries(writeRun(t, map[int]string{0: body}))
	if err != nil {
		t.Fatal(err)
	}
	if s.Unit() != "ops_per_sec" {
		t.Errorf("Unit() = %q, want ops_per_sec (must read the column, not hardcode tokens)", s.Unit())
	}
}

func TestWriteCSVColumnsAndGap(t *testing.T) {
	runDir := writeRun(t, map[int]string{0: passGen, 1: reviseGen})
	s, _ := ReadSeries(runDir)
	var b strings.Builder
	if err := WriteCSV(s, &b); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("csv lines = %d, want 3 (header + 2 gens)", len(lines))
	}
	header := strings.Join(CSVColumns, ",")
	if lines[0] != header {
		t.Errorf("header = %q, want %q", lines[0], header)
	}
	// gen 0 (PASS) has all 15 columns populated.
	pass := strings.Split(lines[1], ",")
	if len(pass) != len(CSVColumns) {
		t.Fatalf("pass row cols = %d, want %d", len(pass), len(CSVColumns))
	}
	if pass[0] != "0" || pass[1] != "PASS" {
		t.Errorf("pass row gen/verdict = %q/%q", pass[0], pass[1])
	}
	if pass[10] != "120" { // cand_median
		t.Errorf("cand_median col = %q, want 120", pass[10])
	}
	// gen 1 (REVISE) has empty numeric columns (a plot gap), not zeros.
	rev := strings.Split(lines[2], ",")
	if rev[1] != "REVISE" {
		t.Errorf("revise verdict = %q", rev[1])
	}
	if rev[5] != "" { // tokens_per_sec must be empty, not "0"
		t.Errorf("revise tokens_per_sec col = %q, want empty (gap)", rev[5])
	}
	if rev[10] != "" { // cand_median empty
		t.Errorf("revise cand_median col = %q, want empty", rev[10])
	}
}

func TestCSVColumnCount(t *testing.T) {
	if len(CSVColumns) != 15 {
		t.Errorf("CSVColumns = %d, want 15 (locked with metalopt)", len(CSVColumns))
	}
	want := "gen,verdict,correctness_ok,unit,runs,tokens_per_sec,baseline_tokens_per_sec,delta_tokens_per_sec,speedup,cand_min,cand_median,cand_max,base_min,base_median,base_max"
	if got := strings.Join(CSVColumns, ","); got != want {
		t.Errorf("CSV header drifted from the locked format:\n got %q\nwant %q", got, want)
	}
}

func TestSparkline(t *testing.T) {
	// Ascending values -> low rune first, high rune last; a gap renders '·'.
	line := Sparkline([]float64{10, 0, 20, 30}, []bool{true, false, true, true})
	runes := []rune(line)
	if len(runes) != 4 {
		t.Fatalf("sparkline len = %d, want 4", len(runes))
	}
	if runes[1] != '·' {
		t.Errorf("gap rune = %q, want ·", string(runes[1]))
	}
	if runes[0] != sparkRunes[0] {
		t.Errorf("min point = %q, want lowest ramp rune %q", string(runes[0]), string(sparkRunes[0]))
	}
	if runes[3] != sparkRunes[len(sparkRunes)-1] {
		t.Errorf("max point = %q, want highest ramp rune", string(runes[3]))
	}
}

func TestSparklineAllRevise(t *testing.T) {
	line := Sparkline([]float64{0, 0}, []bool{false, false})
	if line != "··" {
		t.Errorf("all-revise sparkline = %q, want ··", line)
	}
}

func TestRenderTerminalReadsUnitAndShowsSpread(t *testing.T) {
	body := strings.Replace(passGen, `"unit": "tokens_per_sec"`, `"unit": "ops_per_sec"`, 1)
	s, _ := ReadSeries(writeRun(t, map[int]string{0: body, 1: reviseGen}))
	var b strings.Builder
	if err := RenderTerminal(s, MetricDelta, &b); err != nil {
		t.Fatalf("RenderTerminal: %v", err)
	}
	out := b.String()
	if !strings.Contains(out, "ops_per_sec") {
		t.Errorf("terminal output missing unit ops_per_sec (must not hardcode tokens/sec):\n%s", out)
	}
	if !strings.Contains(out, "gen-N − gen-0 delta") {
		t.Errorf("terminal output missing metric name:\n%s", out)
	}
	if !strings.Contains(out, "[110") { // candidate spread min shown
		t.Errorf("terminal output missing candidate spread:\n%s", out)
	}
	if !strings.Contains(out, "token 3 mismatched golden") {
		t.Errorf("terminal output missing REVISE reason:\n%s", out)
	}
}

func TestNumIsPlainDecimalNotScientific(t *testing.T) {
	// CSV numbers must be plain decimal so gnuplot/spreadsheets read them; a
	// billion-scale ops/sec value must NOT come out as 2.5e+09.
	got := num(2543821642.1440787)
	if strings.ContainsAny(got, "eE") {
		t.Errorf("num(2.54e9) = %q, want plain decimal (no scientific notation)", got)
	}
	if got != "2543821642.1440787" {
		t.Errorf("num lost precision: got %q", got)
	}
}

func TestSIUnit(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{2543821642, "2.54G"},
		{64391125, "64.4M"},
		{712643020, "713M"},
		{1045721181, "1.05G"},
		{518, "518.00"},
		{1500, "1.5K"},
		{2.5e12, "2.5T"},
	}
	for _, tt := range tests {
		if got := siUnit(tt.in); got != tt.want {
			t.Errorf("siUnit(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestCSVBooleanLowercase(t *testing.T) {
	// correctness_ok is lowercase true/false (RFC-CSV standard), not Python True.
	s, _ := ReadSeries(writeRun(t, map[int]string{0: passGen, 1: reviseGen}))
	var b strings.Builder
	WriteCSV(s, &b)
	if strings.Contains(b.String(), "True") || strings.Contains(b.String(), "False") {
		t.Errorf("CSV contains capitalized booleans, want lowercase true/false:\n%s", b.String())
	}
	if !strings.Contains(b.String(), ",true,") {
		t.Errorf("CSV missing lowercase true for PASS gen:\n%s", b.String())
	}
}

func TestSignedSINoDoubleSign(t *testing.T) {
	// A negative delta must render with a single minus, not "+-".
	if got := signedSI(-131.9); got != "-131.90" {
		t.Errorf("signedSI(-131.9) = %q, want -131.90 (no double sign)", got)
	}
	if got := signedSI(2.48e9); got != "+2.48G" {
		t.Errorf("signedSI(2.48e9) = %q, want +2.48G", got)
	}
	if got := signedSI(0); got != "+0.00" {
		t.Errorf("signedSI(0) = %q, want +0.00", got)
	}
}

func TestRenderTerminalNegativeDeltaSingleSign(t *testing.T) {
	// Regression: a slower-but-correct gen has a negative delta; the table must
	// not show "+-".
	slower := `{"verdict":"PASS","correctness_ok":true,"unit":"tokens_per_sec","runs":3,
	  "tokens_per_sec":80,"baseline_tokens_per_sec":100,"delta_tokens_per_sec":-20,"speedup":0.8,
	  "candidate":{"min":78,"median":80,"max":82,"samples":[78,80,82]},
	  "baseline":{"min":98,"median":100,"max":102,"samples":[98,100,102]}}`
	s, _ := ReadSeries(writeRun(t, map[int]string{0: slower}))
	var b strings.Builder
	RenderTerminal(s, MetricDelta, &b)
	if strings.Contains(b.String(), "+-") {
		t.Errorf("terminal shows double sign +- on negative delta:\n%s", b.String())
	}
	if !strings.Contains(b.String(), "-20.00") {
		t.Errorf("terminal missing single-signed -20.00:\n%s", b.String())
	}
}

func isFlat(line string) bool {
	runes := []rune(line)
	for i := 1; i < len(runes); i++ {
		if runes[i] != runes[0] {
			return false
		}
	}
	return true
}

// TestSparklineFlatness pins the flat-detector against the demo's money visual
// and its boundary cases. The flat-detector uses the coefficient of variation
// (stdev/|mean|), so a series is flat when its generations don't move relative
// to their magnitude — whether they sit at 1.0x (a 1B that can't improve) or at
// a steady 1.45x (a real optimizer written at gen-1). Both are honestly flat
// gen-to-gen; the table's speedup column carries the level. A genuine ramp or a
// step change must NOT be flattened (that would hide real movement).
func TestSparklineFlatness(t *testing.T) {
	all := func(n int) []bool {
		b := make([]bool, n)
		for i := range b {
			b[i] = true
		}
		return b
	}
	tests := []struct {
		name     string
		vals     []float64
		wantFlat bool
	}{
		// The two real demo series (CV 0.025 and 0.017 respectively).
		{"flat-pi run_1 (~1.0x noise)", []float64{0.9556, 1.0085, 1.0102, 0.9672}, true},
		{"scripted win run_3 (steady 1.45x)", []float64{1.4134, 1.4765, 1.4374, 1.4686}, true},
		// claude-climb, the demo's other money shot: a real win that keeps rising
		// must NOT be flattened (CV 0.079–0.092).
		{"claude-climb gen1 only", []float64{1.0, 1.1724}, false},
		{"claude-climb rising", []float64{1.0, 1.1724, 1.25}, false},
		// Degenerate and tiny series.
		{"identical", []float64{1.0, 1.0, 1.0}, true},
		// Genuine movement must survive (CV well above threshold).
		{"gentle real ramp", []float64{1.0, 1.1, 1.2, 1.3}, false},
		{"step change", []float64{1.0, 1.0, 1.5}, false},
		{"large climb", []float64{10, 20, 35, 60, 90}, false},
		// Signed delta series (Sparkline is metric-agnostic; MetricDelta centers
		// near zero). The scale = max(|mean|, span) denominator must not divide by
		// ~0 and must show a real two-sided swing rather than fabricate a flat line.
		{"delta real swing (mean≈0)", []float64{-50, 0, 80}, false},
		{"delta exactly-zero mean", []float64{-10, 10}, false},
		// All-identical zero delta: span==0 short-circuits, still flat, no 0/0.
		{"delta all zero", []float64{0, 0, 0}, true},
		// Raw tokens/sec: noise flat, real climb survives.
		{"tokens noise", []float64{2500, 2510, 2495, 2505}, true},
		{"tokens real climb", []float64{2500, 2800, 3200, 3600}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := Sparkline(tt.vals, all(len(tt.vals)))
			if got := isFlat(line); got != tt.wantFlat {
				t.Errorf("Sparkline(%v) = %q, flat=%v, want flat=%v", tt.vals, line, got, tt.wantFlat)
			}
		})
	}
}

func TestSparklineRealClimbScalesEndToEnd(t *testing.T) {
	// A genuine climb spans the full rune ramp low-to-high.
	vals := []float64{10, 20, 35, 60, 90}
	passed := []bool{true, true, true, true, true}
	runes := []rune(Sparkline(vals, passed))
	if runes[0] != sparkRunes[0] {
		t.Errorf("climb start = %q, want lowest ramp rune", string(runes[0]))
	}
	if runes[len(runes)-1] != sparkRunes[len(sparkRunes)-1] {
		t.Errorf("climb end = %q, want highest ramp rune", string(runes[len(runes)-1]))
	}
}

func TestFirstLineTruncatesMultilineReason(t *testing.T) {
	// A multi-line Metal compiler dump collapses to its headline so the demo
	// table layout survives.
	dump := "kernel did not compile: ArrayEval error\nutils.h:458: cannot initialize\nutils.h:475: undeclared identifier"
	got := firstLine(dump, 72)
	if strings.Contains(got, "\n") {
		t.Errorf("firstLine kept newlines: %q", got)
	}
	if got != "kernel did not compile: ArrayEval error" {
		t.Errorf("firstLine = %q, want the headline only", got)
	}
	// Over-long single line gets an ellipsis at the cap.
	long := strings.Repeat("x", 100)
	if r := []rune(firstLine(long, 72)); len(r) != 72 || r[71] != '…' {
		t.Errorf("firstLine long: len=%d last=%q, want 72 ending …", len(r), string(r[len(r)-1]))
	}
}

func TestRenderTerminalReviseReasonStaysOneLine(t *testing.T) {
	// The whole REVISE row must be a single line even when the reason is a
	// multi-line compiler error (real P7 gen_4 case).
	multi := `{"verdict":"REVISE","correctness_ok":false,"unit":"ops_per_sec","runs":1,
	  "reason":"kernel did not compile: ArrayEval\nutils.h:458: cannot initialize uint\nutils.h:475: undeclared simd_lane_id"}`
	s, _ := ReadSeries(writeRun(t, map[int]string{0: passGen, 1: multi}))
	var b strings.Builder
	RenderTerminal(s, MetricSpeedup, &b)
	// Count lines after the header/sparkline: the REVISE gen must add exactly one.
	lines := strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	for _, ln := range lines {
		if strings.Contains(ln, "utils.h:458") || strings.Contains(ln, "undeclared") {
			t.Errorf("multi-line compiler dump leaked into the table:\n%s", b.String())
		}
	}
	if !strings.Contains(b.String(), "kernel did not compile") {
		t.Errorf("REVISE headline missing:\n%s", b.String())
	}
}

func TestSeriesValuesMetricSelection(t *testing.T) {
	s, _ := ReadSeries(writeRun(t, map[int]string{0: passGen, 1: fasterGen}))
	delta, _ := s.values(MetricDelta)
	if delta[0] != 20 || delta[1] != 50 {
		t.Errorf("delta values = %v, want [20 50]", delta)
	}
	sp, _ := s.values(MetricSpeedup)
	if sp[0] != 1.2 || sp[1] != 1.5 {
		t.Errorf("speedup values = %v, want [1.2 1.5]", sp)
	}
	tok, _ := s.values(MetricTokensPerSec)
	if tok[0] != 120 || tok[1] != 150 {
		t.Errorf("tokens values = %v, want [120 150]", tok)
	}
}
