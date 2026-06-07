// Weights view: the P6 held-out weight-loop. A weights generation reports a
// held-out test_loss (written by the WeightsEvaluator), where LOWER IS BETTER —
// the opposite polarity from the throughput Gen schema. The two views are kept
// fully separate: they share no numeric field and have opposite direction, so
// neither can silently corrupt the other.
//
// Two decisions distinguish this view from the throughput one:
//
//   - The throughput span-floored CV flat detector is NOT reused. That detector
//     is calibrated for repeated-sample timing jitter (thermal/cache noise); a
//     held-out test_loss is a single deterministic eval of a different adapter
//     each generation, so its coefficient of variation is tiny (values near 2.5,
//     small span) and the CV detector would wrongly flatten a real overfit climb.
//     Flatness here is gated by an absolute loss span (flatLossSpan) instead.
//
//   - The quality bars rise with the loss (native direction): a higher loss is a
//     taller bar, labelled "taller = worse". This keeps "the bar follows the
//     number" — the same reading the throughput view trains — so a rising bar, a
//     rising test_loss column, and a positive Δ-vs-best all point the same way.
//     The signed Δ-vs-best column is the headline; the bars are a secondary glyph
//     that can never desync from the printed numbers.
package chartdata

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// flatLossSpan is the absolute held-out test_loss span below which a weights
// series is treated as flat (no real gen-to-gen movement). 0.02 nats ≈ a ~2%
// perplexity change, below what is meaningful for this task. It is an absolute,
// metric-native floor — not a coefficient of variation — so it both refuses to
// amplify near-equal noise into a fake climb and lets a real overfit climb (the
// demo series spans ~0.21, 10× this floor) render as genuine movement.
const flatLossSpan = 0.02

// WeightsGen is one generation of the held-out weight-loop. The zero value
// (Trained false, TestLoss 0) is an untrained generation: it carries no held-out
// number, renders as a gap, and is excluded from the quality scale so a missing
// loss never becomes a tall "best" bar.
type WeightsGen struct {
	Gen        int     // generation number, from the gen_N directory
	Verdict    string  `json:"verdict"`
	Trained    bool    `json:"trained"`
	TestLoss   float64 `json:"test_loss"`
	Perplexity float64 `json:"perplexity"`
	Reason     string  `json:"reason"`
	HeldOutDir string  `json:"held_out_dir"`
}

// scored reports whether g contributes a real held-out loss to the quality
// scale. An untrained generation, or one whose held-out eval failed, has no
// loss and is not scored.
func (g WeightsGen) scored() bool { return g.Trained && g.TestLoss > 0 }

// passed reports whether g cleared the held-out gate.
func (g WeightsGen) passed() bool { return g.Verdict == verdictPass }

// WeightsSeries is a weights run's generations in ascending generation order.
type WeightsSeries struct {
	RunDir string
	Model  string
	Gens   []WeightsGen
}

// ReadWeightsSeries discovers gen_N directories under runDir, reads each
// results.json as a weights generation, and returns them in ascending order.
// Missing or unparseable results.json files are skipped (a generation may not
// have evaluated yet); a runDir with no gen directories is an error.
func ReadWeightsSeries(runDir string) (WeightsSeries, error) {
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return WeightsSeries{}, fmt.Errorf("read run dir: %w", err)
	}
	var gens []WeightsGen
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		n, ok := genNumber(e.Name())
		if !ok {
			continue
		}
		g, err := readWeightsGen(filepath.Join(runDir, e.Name(), "results.json"))
		if err != nil {
			continue // not evaluated yet, or mid-write
		}
		g.Gen = n
		gens = append(gens, g)
	}
	if len(gens) == 0 {
		return WeightsSeries{}, fmt.Errorf("no generations with results.json under %s", runDir)
	}
	sort.Slice(gens, func(i, j int) bool { return gens[i].Gen < gens[j].Gen })
	return WeightsSeries{RunDir: runDir, Gens: gens}, nil
}

func readWeightsGen(path string) (WeightsGen, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return WeightsGen{}, err
	}
	var g WeightsGen
	if err := json.Unmarshal(data, &g); err != nil {
		return WeightsGen{}, err
	}
	return g, nil
}

// IsWeightsTree reports whether the run tree at runDir is a weights tree, by
// sniffing the generations' results.json for a decisive schema witness. It
// routes on a positive signal — a "test_loss" metric or a test_loss field marks
// weights; a tokens_per_sec field marks throughput — never on the absence of a
// field owned by the other writer, since a bare-REVISE generation carries
// neither. It scans generations until one is decisive; ok is false when none is,
// so the caller can refuse to guess.
func IsWeightsTree(runDir string) (isWeights bool, ok bool) {
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return false, false
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, isGen := genNumber(e.Name()); !isGen {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(runDir, e.Name(), "results.json"))
		if err != nil {
			continue
		}
		var p struct {
			Metric       string   `json:"metric"`
			TokensPerSec *float64 `json:"tokens_per_sec"`
			TestLoss     *float64 `json:"test_loss"`
		}
		if json.Unmarshal(raw, &p) != nil {
			continue
		}
		switch {
		case p.Metric == "test_loss" || p.TestLoss != nil:
			return true, true // decisive: weights
		case p.TokensPerSec != nil:
			return false, true // decisive: throughput
		}
		// neither witness: bare REVISE or partial write; keep scanning
	}
	return false, false // no decisive witness anywhere
}

// bestPassIndex returns the index of the lowest-loss generation that cleared the
// gate, or -1 if no generation passed. The best PASS generation is the anchor
// for the Δ-vs-best column and wears the "best" tag; with no PASS generation
// there is no anchor and nothing is crowned.
func (s WeightsSeries) bestPassIndex() int {
	best := -1
	for i, g := range s.Gens {
		if !g.passed() || !g.scored() {
			continue
		}
		if best < 0 || g.TestLoss < s.Gens[best].TestLoss {
			best = i
		}
	}
	return best
}

// lossSpan returns the min and max held-out loss over the scored generations and
// whether any generation was scored. REVISE generations' losses are included:
// they set the scale against which a surviving bar's height is read.
func (s WeightsSeries) lossSpan() (min, max float64, any bool) {
	min, max = math.Inf(1), math.Inf(-1)
	for _, g := range s.Gens {
		if !g.scored() {
			continue
		}
		any = true
		if g.TestLoss < min {
			min = g.TestLoss
		}
		if g.TestLoss > max {
			max = g.TestLoss
		}
	}
	return min, max, any
}

// flat reports whether the scored losses span less than flatLossSpan — too
// little real movement to scale across the rune ramp without amplifying noise.
func (s WeightsSeries) flat() bool {
	min, max, any := s.lossSpan()
	return any && max-min < flatLossSpan
}

// QualityBars renders the per-generation quality glyph line. Bars rise with the
// loss (native: a higher loss is a taller bar = worse).
//
// A REVISE generation that still produced a held-out loss (an overfit rejected
// for being WORSE, not invalid) keeps its bar at its real height: that rising bar
// IS the overfit, and dropping it would hide the climb the gate caught. Only a
// generation with no held-out number at all — untrained, or whose eval failed —
// is a "·" gap. A REVISE generation's loss counts toward the scale either way. A
// near-flat series (span < flatLossSpan) renders every scored generation at the
// mid rune rather than amplifying noise into a fake slope.
func (s WeightsSeries) QualityBars() string {
	min, max, any := s.lossSpan()
	if !any {
		return repeatGap(len(s.Gens)) // all-untrained / all-eval-failed: no bars
	}
	span := max - min
	flat := span < flatLossSpan
	// A flat series renders at the low-middle rune: present (not an empty gap)
	// but visibly calm, so a deterministically-steady result never looks like a
	// climb. Index 3 of the 8-rune ramp ("▄") sits just below centre.
	const flatRune = 3

	var b []rune
	for _, g := range s.Gens {
		switch {
		case !g.scored():
			b = append(b, '·') // true gap: no held-out number to plot
		case flat:
			b = append(b, sparkRunes[flatRune])
		default:
			idx := int(math.Round((g.TestLoss - min) / span * float64(len(sparkRunes)-1)))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(sparkRunes) {
				idx = len(sparkRunes) - 1
			}
			b = append(b, sparkRunes[idx])
		}
	}
	return string(b)
}

func repeatGap(n int) string {
	r := make([]rune, n)
	for i := range r {
		r[i] = '·'
	}
	return string(r)
}

// WeightsCSVColumns is the fixed column order for the weights CSV. It is separate
// from the throughput CSVColumns (the two schemas share no numeric field), so a
// consumer never sees a 15-wide row with twelve empty cells. delta_loss_vs_best
// matches the table's sign convention: positive means worse than the best PASS
// generation.
var WeightsCSVColumns = []string{
	"gen", "verdict", "trained", "test_loss", "perplexity",
	"delta_loss_vs_best", "is_best", "reason", "held_out_dir",
}

// WriteCSV writes the weights series as CSV with WeightsCSVColumns header. An
// untrained generation leaves test_loss/perplexity/delta empty (not a misleading
// 0). delta_loss_vs_best is signed (+ worse); the best PASS generation is the
// only is_best=true row, and no row is best when nothing passed. The reason
// column is quoted as needed, so a comma in a REVISE reason never shifts columns.
func (s WeightsSeries) WriteCSV(out io.Writer) error {
	w := csv.NewWriter(out)
	if err := w.Write(WeightsCSVColumns); err != nil {
		return err
	}
	best := s.bestPassIndex()
	for i, g := range s.Gens {
		if err := w.Write(s.csvRow(g, i == best, best)); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func (s WeightsSeries) csvRow(g WeightsGen, isBest bool, best int) []string {
	testLoss, ppl, delta := "", "", ""
	if g.scored() {
		testLoss = num(g.TestLoss)
		ppl = num(g.Perplexity)
		if best >= 0 {
			// Fixed 4-decimal precision, matching the table's Δ column, so the
			// signed difference of two losses never exposes binary noise (e.g.
			// 0.03210000000000024) and the CSV and table agree value-for-value.
			delta = strconv.FormatFloat(g.TestLoss-s.Gens[best].TestLoss, 'f', 4, 64)
		}
	}
	return []string{
		strconv.Itoa(g.Gen), g.Verdict, strconv.FormatBool(g.Trained),
		testLoss, ppl, delta, strconv.FormatBool(isBest),
		firstLine(g.Reason, 120), g.HeldOutDir,
	}
}

// Render writes the weights series as a quality-bar line plus a per-generation
// table — the live-demo view. test_loss is shown lower-is-better; the bars rise
// with the loss (taller = worse); the signed Δ-vs-best column is the headline,
// with "+" meaning worse than the best generation that cleared the gate. A
// REVISE generation keeps its raw test_loss visible (the rising number is the
// evidence the gate acted on) and prints its reason inline.
func (s WeightsSeries) Render(out io.Writer) error {
	w := func(format string, a ...any) error {
		_, err := fmt.Fprintf(out, format, a...)
		return err
	}
	model := s.Model
	if model == "" {
		model = "model"
	}
	if err := w("P6 weight-loop  ·  %s  ·  %s\n", s.RunDir, model); err != nil {
		return err
	}
	if err := w("test_loss ↓ lower is better   ·   quality bars: taller = worse (rising loss)   ·   Δ vs best: + worse\n\n"); err != nil {
		return err
	}

	best := s.bestPassIndex()
	bestNote := ""
	if best >= 0 {
		bestNote = fmt.Sprintf("best = gen%d (held-out %.4f)", s.Gens[best].Gen, s.Gens[best].TestLoss)
	} else {
		bestNote = "no generation cleared the held-out gate"
	}
	if err := w("  loss  %s      %s\n\n", s.QualityBars(), bestNote); err != nil {
		return err
	}

	if err := w("  %-4s %-7s %-10s %-7s %-11s %s\n", "gen", "verdict", "test_loss", "ppl", "Δ vs best", "loss"); err != nil {
		return err
	}
	bars := []rune(s.QualityBars())
	for i, g := range s.Gens {
		loss, ppl, delta := "—", "—", "—"
		if g.scored() {
			loss = fmt.Sprintf("%.4f", g.TestLoss)
			ppl = fmt.Sprintf("%.2f", g.Perplexity)
			switch {
			case best < 0:
				delta = "—"
			case i == best:
				delta = "—  best"
			case s.flat():
				delta = "≈0"
			default:
				delta = signedLoss(g.TestLoss - s.Gens[best].TestLoss)
			}
		}
		bar := "·"
		if i < len(bars) {
			bar = string(bars[i])
		}
		// A rejected generation that still has a real bar (an overfit, rejected
		// for being worse — not invalid) is marked ✗ beside its bar, so the rising
		// bar reads as "present but gate-rejected". A true gap (no number) shows no
		// ✗ — there is nothing to reject, only nothing to plot.
		mark := " "
		if !g.passed() && g.scored() {
			mark = "✗"
		}
		trailer := ""
		if !g.passed() && g.Reason != "" {
			trailer = "   " + firstLine(g.Reason, 60)
		}
		if err := w("  %-4d %-7s %-10s %-7s %-11s %s %s%s\n",
			g.Gen, g.Verdict, loss, ppl, delta, bar, mark, trailer); err != nil {
			return err
		}
	}
	return nil
}

// signedLoss formats a Δ-vs-best loss with exactly one sign: "+0.0737" for a
// worse (higher) loss, "-0.0100" for a better one. Positive always means worse,
// matching the bars and the CSV's delta_loss_vs_best.
func signedLoss(d float64) string {
	if d < 0 {
		return strconv.FormatFloat(d, 'f', 4, 64)
	}
	return "+" + strconv.FormatFloat(d, 'f', 4, 64)
}
