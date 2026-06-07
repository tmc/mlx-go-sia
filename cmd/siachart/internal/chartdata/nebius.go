// Correctness-progression view: the live-cloud Nebius beat. A frontier model
// drives the SIA loop and writes the same throughput results.json each
// generation, but the story this view tells is the GATE, not speed. Across the
// generations the exact-token correctness gate REVISEs the model's real compile
// and runtime bugs, then PASSes a provably-correct candidate — and it credits a
// speedup only when one is actually measurable.
//
// This view exists because the throughput series view would mislead here. A
// correct PASS at parity carries a recorded speedup near 1.0 (e.g. 1.009x); the
// throughput table prints that number and the sparkline draws a bar, which reads
// as a small win. But the candidate and baseline run-to-run distributions
// overlap — the +0.9% is measurement noise, not a speedup. So this view:
//
//   - leads with the verdict/correctness progression (✗ REVISE … ✓ PASS), with
//     each REVISE's bug reason inline — the gate catching real frontier-model
//     bugs is the visceral proof, not a speed bar;
//
//   - labels a PASS generation's speed from the DATA, not the recorded scalar: a
//     win only when the candidate's slowest run still beats the baseline's
//     fastest (candidate.Min > baseline.Max, the conservative non-overlap test);
//     otherwise "parity", so an overlapping ~1.0x is never dressed up as a win.
//
// The same render serves a genuine win: if a PASS generation's distributions do
// separate, its speedup is shown as a real win with the number. The win/parity
// call is made by the data, never by a flag, so the chart cannot disagree with
// the runs behind it.
package chartdata

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// NebiusSeries is a live-cloud run's generations in ascending order, read from
// the same results.json the throughput evaluator writes. It reuses the Gen type
// (verdict, correctness_ok, reason, the candidate/baseline distributions); only
// the rendering differs.
type NebiusSeries struct {
	RunDir string
	Model  string
	Gens   []Gen
}

// ReadNebiusSeries discovers gen_N directories under runDir and reads each
// results.json as a generation, in ascending order. It shares the throughput
// reader's tolerance: a missing or unparseable results.json is skipped (not yet
// evaluated, or mid-write); a runDir with no generations is an error.
func ReadNebiusSeries(runDir string) (NebiusSeries, error) {
	s, err := ReadSeries(runDir)
	if err != nil {
		return NebiusSeries{}, err
	}
	return NebiusSeries{RunDir: s.RunDir, Gens: s.Gens}, nil
}

// speedVerdict classifies a generation's measured speed against the baseline.
type speedVerdict int

const (
	// speedNone is a generation that produced no timing (a REVISE that did not
	// run, so it has no candidate distribution).
	speedNone speedVerdict = iota
	// speedParity is a PASS whose candidate and baseline distributions overlap and
	// whose recorded ratio sits near 1.0: any difference is within run-to-run
	// noise, neither a measurable win nor a measurable regression.
	speedParity
	// speedWin is a PASS whose candidate distribution clears the baseline's — the
	// slowest candidate run still beats the fastest baseline run — so the speedup
	// is real and reported.
	speedWin
	// speedSlower is a PASS whose recorded ratio is meaningfully below 1.0 while
	// the ranges do not separate upward: the correct candidate is slower than the
	// baseline. The gate credits its correctness but gives it no speed credit —
	// the honest, even stronger anti-Goodhart point that a correct regression is
	// not rewarded.
	speedSlower
)

// regressFloor is the recorded-ratio ceiling below which a PASS is called slower
// rather than parity: a candidate timed at 0.97x or below is meaningfully slower,
// outside the near-1.0 noise band.
const regressFloor = 0.97

// speedFloor is the margin a separated range must clear to be credited a win.
// candidate.min must beat baseline.max by at least 3%, so a razor-thin
// non-overlap that can occur by chance with a handful of samples is not promoted
// to a celebratory win. It is the false-win guard: separation alone is necessary
// but not sufficient.
const speedFloor = 1.03

// classifySpeed decides win versus parity from the distributions, never from the
// recorded speedup scalar. A PASS is a win only when the candidate's slowest run
// beats the baseline's fastest (no sample overlap), that separation clears the
// speedFloor margin, and every candidate sample beats every baseline sample. The
// three together refuse to credit a win the small-sample data cannot support: a
// median ratio above 1.0 is routinely noise, a bare non-overlap can be a fluke,
// and a single throttled run should not flip the verdict. Anything short of all
// three on a PASS is parity; a generation that did not time is speedNone. The
// rule is deliberately conservative and asymmetric — it defaults to parity under
// any doubt, so noise is never inflated into a win.
func classifySpeed(g Gen) speedVerdict {
	if g.Candidate == nil || g.BaselineDist == nil {
		return speedNone
	}
	separated := g.Candidate.Min > g.BaselineDist.Max
	clearsFloor := g.BaselineDist.Max > 0 && g.Candidate.Min/g.BaselineDist.Max >= speedFloor
	if separated && clearsFloor && allPairsFaster(g.Candidate.Samples, g.BaselineDist.Samples) {
		return speedWin
	}
	// A correct candidate timed meaningfully below 1.0 is slower, not parity —
	// but only when it is not separated upward (a win already returned above). The
	// recorded ratio is the in-loop median; on a noisy host the single-sample
	// ranges can overlap even when every contemporaneous pair is slower, so the
	// recorded ratio, not the range, decides the regression.
	if g.Speedup > 0 && g.Speedup <= regressFloor {
		return speedSlower
	}
	return speedParity
}

// allPairsFaster reports whether every candidate sample beats every baseline
// sample. It keeps the win rule sound when the two runnables have different
// sample counts (a single-run candidate against a five-run baseline), where the
// min/max separation test alone would lean on just two order statistics. With no
// samples on either side it is vacuously false: there is nothing to confirm a
// win, so the rule falls back to parity.
func allPairsFaster(cand, base []float64) bool {
	if len(cand) == 0 || len(base) == 0 {
		return false
	}
	for _, c := range cand {
		for _, b := range base {
			if c <= b {
				return false
			}
		}
	}
	return true
}

// speedLabel is the human-readable speed cell for a generation. A real win shows
// its speedup. A parity PASS is described by the overlap GEOMETRY — by how many
// tokens/sec the candidate and baseline ranges overlap — rather than by the
// recorded speedup scalar: a lone ratio just above 1.0 reads as a small win when
// skimmed, but the candidate's range dipping below the baseline's max is the
// honest "not separated, no measurable win". A generation that did not run has no
// speed cell.
func speedLabel(g Gen) string {
	switch classifySpeed(g) {
	case speedWin:
		return fmt.Sprintf("%.3fx win (cand min %.0f > base max %.0f)",
			g.Speedup, g.Candidate.Min, g.BaselineDist.Max)
	case speedSlower:
		return fmt.Sprintf("%.2fx — slower, a regression (no credit)", g.Speedup)
	case speedParity:
		overlap := g.BaselineDist.Max - g.Candidate.Min
		return fmt.Sprintf("parity — ranges overlap %.0f tok/s, no measurable win", overlap)
	default:
		return "—"
	}
}

// bugSummary pulls the meaningful diagnostic out of a candidate's failure
// reason. A failed candidate's reason is a wrapped tool error whose first line is
// a generic "candidate did not run …" prefix; the real bug is the Go compiler
// diagnostic ("cannot slice h …") or the runtime panic ("panic: index out of
// range …") below it. This surfaces that bug — the visceral evidence the gate
// acted on — rather than the prefix. It returns the panic line if present, else
// the text after the first "file.go:line:col:" location, else the first line.
func bugSummary(reason string, max int) string {
	for _, line := range strings.Split(reason, "\n") {
		line = strings.TrimSpace(line)
		if i := strings.Index(line, "panic:"); i >= 0 {
			return clip(line[i:], max)
		}
		if msg, ok := afterGoLocation(line); ok {
			return clip(msg, max)
		}
	}
	return firstLine(reason, max)
}

// afterGoLocation returns the message following a "…file.go:line:col:" prefix in
// line, if line carries one. A Go compiler diagnostic leads with the source
// location; the part after it is the human-readable cause.
func afterGoLocation(line string) (string, bool) {
	i := strings.Index(line, ".go:")
	if i < 0 {
		return "", false
	}
	rest := line[i+len(".go:"):]
	// Skip "line:col:" — two colon-terminated numeric fields — then return the
	// remaining message. If the shape doesn't match, this isn't a diagnostic.
	for fields := 0; fields < 2; fields++ {
		j := strings.IndexByte(rest, ':')
		if j < 0 {
			return "", false
		}
		if _, err := strconv.Atoi(rest[:j]); err != nil {
			return "", false
		}
		rest = rest[j+1:]
	}
	return strings.TrimSpace(rest), true
}

// clip truncates s to max runes with an ellipsis, mirroring firstLine's tail
// handling without re-splitting on newlines (the caller already has one line).
func clip(s string, max int) string {
	if r := []rune(s); len(r) > max {
		return string(r[:max-1]) + "…"
	}
	return s
}

// Render writes the correctness-progression view: a one-line gate summary, then
// a per-generation table of verdict, a correctness glyph, the speed verdict, and
// the bug reason inline on each REVISE. The speed column never draws a bar — a
// parity PASS is labelled, not plotted — so a ~1.0x can never read as a climb.
func (s NebiusSeries) Render(out io.Writer) error {
	w := func(format string, a ...any) error {
		_, err := fmt.Fprintf(out, format, a...)
		return err
	}
	model := s.Model
	if model == "" {
		model = "model"
	}
	if err := w("Nebius live-cloud verdict progression  ·  %s  ·  %s\n", s.RunDir, model); err != nil {
		return err
	}
	revised, passed, wins := s.tally()
	if err := w("the gate: REVISE'd %d real bug(s), PASS'd %d correct candidate(s), credited %d measurable speedup(s)\n",
		revised, passed, wins); err != nil {
		return err
	}
	if err := w("correctness ✓/✗ is the signal · a speedup is credited only when candidate/baseline ranges SEPARATE (else parity, never a fake win)\n\n"); err != nil {
		return err
	}

	if err := w("  %-4s %-7s %-4s %-52s %s\n", "gen", "verdict", "ok", "speed", "what happened"); err != nil {
		return err
	}
	for _, g := range s.Gens {
		ok := "✓"
		if !g.CorrectnessOK {
			ok = "✗"
		}
		what := ""
		switch {
		case g.Passed() && classifySpeed(g) == speedWin:
			what = "correct, measurably faster"
		case g.Passed() && classifySpeed(g) == speedSlower:
			what = "correct (token-identical) — credited for correctness, not speed"
		case g.Passed():
			what = "correct (token-identical), no measurable speedup"
		case g.Reason != "":
			what = bugSummary(g.Reason, 60)
		default:
			what = "correctness gate not passed"
		}
		if err := w("  %-4d %-7s %-4s %-52s %s\n", g.Gen, g.Verdict, ok, speedLabel(g), what); err != nil {
			return err
		}
	}
	return nil
}

// tally counts REVISE generations, PASS generations, and PASS generations with a
// measurable (range-separated) speedup. The three numbers are the gate's record:
// bugs caught, correct candidates credited, and real wins — typically zero on a
// parity run, which is the point.
func (s NebiusSeries) tally() (revised, passed, wins int) {
	for _, g := range s.Gens {
		switch {
		case g.Passed():
			passed++
			if classifySpeed(g) == speedWin {
				wins++
			}
		default:
			revised++
		}
	}
	return revised, passed, wins
}

// NebiusCSVColumns is the fixed column order for the correctness-progression CSV.
// speed_verdict is the data-derived win/parity/none classification; recorded_speedup
// is the raw scalar from results.json, kept so a consumer can see the number the
// loop reported alongside the honest verdict. is_real_win is true only when the
// distributions separate.
var NebiusCSVColumns = []string{
	"gen", "verdict", "correctness_ok", "speed_verdict", "recorded_speedup",
	"is_real_win", "cand_min", "cand_max", "base_min", "base_max", "reason",
}

// WriteCSV writes the series as CSV with NebiusCSVColumns. A generation that did
// not time leaves the distribution and speedup cells empty (not a misleading 0).
// speed_verdict carries the same honest call the table shows — win only when the
// distributions separate, slower when the recorded ratio is meaningfully below
// 1.0, else parity — and is_real_win is true only for a separated win.
func (s NebiusSeries) WriteCSV(out io.Writer) error {
	cw := csv.NewWriter(out)
	if err := cw.Write(NebiusCSVColumns); err != nil {
		return err
	}
	for _, g := range s.Gens {
		if err := cw.Write(s.csvRow(g)); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

func (s NebiusSeries) csvRow(g Gen) []string {
	sv := classifySpeed(g)
	verdict := map[speedVerdict]string{speedNone: "none", speedParity: "parity", speedWin: "win", speedSlower: "slower"}[sv]
	speedup, candMin, candMax, baseMin, baseMax := "", "", "", "", ""
	if sv != speedNone {
		speedup = num(g.Speedup)
		candMin, candMax = num(g.Candidate.Min), num(g.Candidate.Max)
		baseMin, baseMax = num(g.BaselineDist.Min), num(g.BaselineDist.Max)
	}
	return []string{
		strconv.Itoa(g.Gen), g.Verdict, strconv.FormatBool(g.CorrectnessOK),
		verdict, speedup, strconv.FormatBool(sv == speedWin),
		candMin, candMax, baseMin, baseMax, firstLine(g.Reason, 120),
	}
}
