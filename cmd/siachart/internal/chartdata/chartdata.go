// Package chartdata reads a SIA run tree's per-generation results.json files and
// renders the throughput series two ways: a terminal sparkline+table for the
// live demo, and a CSV for gnuplot or a spreadsheet. It is the demo-side reader
// for the ThroughputEvaluator's output, shared by P3 (cmd/inferopt) and P7
// (cmd/metalopt), which write the same results.json schema.
//
// The reported metric is the gen-N − gen-0 delta the evaluator already computed
// (cancelling thermal/cache drift); charts plot the delta or speedup, with the
// candidate/baseline distributions supplying error bars. A generation whose
// correctness gate did not pass carries no throughput number and renders as a
// gap.
package chartdata

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Verdict strings, mirroring the evaluator's results.json without importing the
// sia package (this tool only reads files).
const (
	verdictPass   = "PASS"
	verdictRevise = "REVISE"
)

// Distribution is the per-run spread of one runnable, mirroring the nested
// candidate/baseline objects the evaluator writes.
type Distribution struct {
	Min     float64   `json:"min"`
	Median  float64   `json:"median"`
	Max     float64   `json:"max"`
	Samples []float64 `json:"samples"`
}

// Gen is one generation's results.json, both the top-level scalars and the
// nested distributions. Fields absent from a REVISE generation stay zero.
type Gen struct {
	Gen           int           // generation number, from the gen_N directory
	Verdict       string        `json:"verdict"`
	CorrectnessOK bool          `json:"correctness_ok"`
	Reason        string        `json:"reason"`
	Unit          string        `json:"unit"`
	Runs          int           `json:"runs"`
	TokensPerSec  float64       `json:"tokens_per_sec"`
	Baseline      float64       `json:"baseline_tokens_per_sec"`
	Delta         float64       `json:"delta_tokens_per_sec"`
	Speedup       float64       `json:"speedup"`
	Candidate     *Distribution `json:"candidate"`
	BaselineDist  *Distribution `json:"baseline"`
}

// Passed reports whether the generation cleared the correctness gate and so
// carries a throughput number.
func (g Gen) Passed() bool { return g.Verdict == verdictPass }

// Series is a run's generations in ascending generation order.
type Series struct {
	RunDir string
	Gens   []Gen
}

// Unit returns the throughput unit of the first generation that reports one, or
// "tokens_per_sec" as a default.
func (s Series) Unit() string {
	for _, g := range s.Gens {
		if g.Unit != "" {
			return g.Unit
		}
	}
	return "tokens_per_sec"
}

// ReadSeries discovers gen_N directories under runDir, reads each results.json,
// and returns the generations in ascending order. Missing or unparseable
// results.json files are skipped (a generation may not have evaluated yet); a
// runDir with no gen directories is an error.
func ReadSeries(runDir string) (Series, error) {
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return Series{}, fmt.Errorf("read run dir: %w", err)
	}
	var gens []Gen
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		n, ok := genNumber(e.Name())
		if !ok {
			continue
		}
		g, err := readGen(filepath.Join(runDir, e.Name(), "results.json"))
		if err != nil {
			continue // not evaluated yet, or mid-write
		}
		g.Gen = n
		gens = append(gens, g)
	}
	if len(gens) == 0 {
		return Series{}, fmt.Errorf("no generations with results.json under %s", runDir)
	}
	sort.Slice(gens, func(i, j int) bool { return gens[i].Gen < gens[j].Gen })
	return Series{RunDir: runDir, Gens: gens}, nil
}

// genNumber parses a "gen_N" directory name, matching the run layout.
func genNumber(name string) (int, bool) {
	rest, ok := strings.CutPrefix(name, "gen_")
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(rest)
	if err != nil {
		return 0, false
	}
	return n, true
}

func readGen(path string) (Gen, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Gen{}, err
	}
	var g Gen
	if err := json.Unmarshal(data, &g); err != nil {
		return Gen{}, err
	}
	return g, nil
}

// CSVColumns is the fixed column order, coordinated with the metalopt series
// generator. Medians drive the line; min/max give error bars; verdict and
// correctness_ok let a consumer render REVISE generations as gaps.
var CSVColumns = []string{
	"gen", "verdict", "correctness_ok", "unit", "runs",
	"tokens_per_sec", "baseline_tokens_per_sec", "delta_tokens_per_sec", "speedup",
	"cand_min", "cand_median", "cand_max",
	"base_min", "base_median", "base_max",
}

// WriteCSV writes the series as CSV with CSVColumns header. A REVISE generation
// emits its gen/verdict/correctness and leaves the numeric columns empty so a
// plot shows a gap rather than a misleading zero.
func WriteCSV(s Series, out io.StringWriter) error {
	if _, err := out.WriteString(strings.Join(CSVColumns, ",") + "\n"); err != nil {
		return err
	}
	for _, g := range s.Gens {
		row := csvRow(g)
		if _, err := out.WriteString(strings.Join(row, ",") + "\n"); err != nil {
			return err
		}
	}
	return nil
}

func csvRow(g Gen) []string {
	cand := g.Candidate
	base := g.BaselineDist
	if !g.Passed() {
		// gen, verdict, correctness_ok, then unit/runs, then empty numerics.
		row := []string{strconv.Itoa(g.Gen), g.Verdict, strconv.FormatBool(g.CorrectnessOK), g.Unit, strconv.Itoa(g.Runs)}
		for i := len(row); i < len(CSVColumns); i++ {
			row = append(row, "")
		}
		return row
	}
	return []string{
		strconv.Itoa(g.Gen), g.Verdict, strconv.FormatBool(g.CorrectnessOK), g.Unit, strconv.Itoa(g.Runs),
		num(g.TokensPerSec), num(g.Baseline), num(g.Delta), num(g.Speedup),
		distField(cand, func(d Distribution) float64 { return d.Min }),
		distField(cand, func(d Distribution) float64 { return d.Median }),
		distField(cand, func(d Distribution) float64 { return d.Max }),
		distField(base, func(d Distribution) float64 { return d.Min }),
		distField(base, func(d Distribution) float64 { return d.Median }),
		distField(base, func(d Distribution) float64 { return d.Max }),
	}
}

func distField(d *Distribution, f func(Distribution) float64) string {
	if d == nil {
		return ""
	}
	return num(f(*d))
}

// num formats a float for CSV as a plain decimal (never scientific notation),
// so the output drops straight into gnuplot and spreadsheets. -1 precision
// keeps the shortest round-trippable form.
func num(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

// siUnit shortens a large number for the terminal table with a K/M/G/T suffix
// and three significant figures, e.g. 2543821642 -> "2.54G". Small magnitudes
// fall back to a two-decimal form. It is display-only; the CSV keeps full
// precision via num.
func siUnit(f float64) string {
	abs := math.Abs(f)
	switch {
	case abs >= 1e12:
		return fmt.Sprintf("%.3gT", f/1e12)
	case abs >= 1e9:
		return fmt.Sprintf("%.3gG", f/1e9)
	case abs >= 1e6:
		return fmt.Sprintf("%.3gM", f/1e6)
	case abs >= 1e3:
		return fmt.Sprintf("%.3gK", f/1e3)
	default:
		return fmt.Sprintf("%.2f", f)
	}
}

// firstLine returns the first line of s, trimmed and truncated to max runes
// with an ellipsis. A REVISE reason is often a multi-line compiler dump; the
// table shows only its headline so the layout stays intact (the full reason
// lives in the generation's results.json).
func firstLine(s string, max int) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > max {
		return string(r[:max-1]) + "…"
	}
	return s
}

// signedSI formats a delta with exactly one leading sign: "+2.48G" for a gain,
// "-131.90" for a regression — never the double sign "+-131.90". siUnit already
// carries the minus for negatives, so a "+" is prepended only when positive.
func signedSI(f float64) string {
	if f < 0 {
		return siUnit(f)
	}
	return "+" + siUnit(f)
}

// sparkRunes is a low-to-high ramp for the terminal sparkline.
var sparkRunes = []rune("▁▂▃▄▅▆▇█")

// Sparkline renders the passing generations' values as a single line of block
// runes scaled across [min,max]. REVISE generations render as a gap rune. With
// fewer than two points it returns the values verbatim.
func Sparkline(values []float64, passed []bool) string {
	min, max, sum, n := math.Inf(1), math.Inf(-1), 0.0, 0
	for i, v := range values {
		if !passed[i] {
			continue
		}
		n++
		sum += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if n == 0 {
		return strings.Repeat("·", len(values))
	}
	span := max - min
	mean := sum / float64(n)

	// A series whose run-to-run variation is tiny relative to its magnitude is
	// effectively flat — e.g. a 1B model bouncing on measurement noise around
	// 1.0x, or a real optimizer sitting rock-steady at 1.45x. In both cases the
	// generations don't move, so auto-scaling the wiggle to the full rune ramp
	// would fake a gen-to-gen climb that did not happen. We render a flat
	// mid-line instead — the honest "no real change between generations"; the
	// table's speedup column carries the actual level.
	//
	// We measure variation by the spread relative to scale, not max−min alone:
	// with only a handful of generations a single outlier inflates the span,
	// whereas the standard deviation is the stable "is this just noise?"
	// statistic. The scale is max(|mean|, span). When the magnitude is meaningful
	// — speedups near 1.0, raw token rates — |mean| dominates and the ratio is the
	// coefficient of variation, the natural scale-free noise measure. When the
	// mean is near zero — a signed gen-N−gen-0 delta series, which Sparkline also
	// plots — span dominates instead, so we neither divide by ~0 nor wrongly
	// flatten a real two-sided swing (whose stdev is a large fraction of its own
	// range). The real demo series land at 0.017 (steady 1.45x win) and 0.025
	// (1.0x pi noise) — both flat — while a genuine ramp like 1.0→1.3 is ~0.10 and
	// a step change higher still, so flatThreshold sits with margin on both sides.
	const flatThreshold = 0.06
	scale := math.Max(math.Abs(mean), span)
	flat := span == 0 || stdev(values, passed, mean)/scale < flatThreshold

	mid := len(sparkRunes) / 2
	var b strings.Builder
	for i, v := range values {
		if !passed[i] {
			b.WriteRune('·') // REVISE gap
			continue
		}
		idx := mid
		if !flat {
			idx = int((v - min) / span * float64(len(sparkRunes)-1))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(sparkRunes) {
				idx = len(sparkRunes) - 1
			}
		}
		b.WriteRune(sparkRunes[idx])
	}
	return b.String()
}

// stdev returns the population standard deviation of the passing values about
// the given mean. It is the spread half of the coefficient of variation used to
// decide whether a series is flat; callers pass the mean they already computed.
func stdev(values []float64, passed []bool, mean float64) float64 {
	sumSq, n := 0.0, 0
	for i, v := range values {
		if !passed[i] {
			continue
		}
		d := v - mean
		sumSq += d * d
		n++
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sumSq / float64(n))
}

// Metric selects which series a chart plots.
type Metric int

const (
	// MetricDelta plots gen-N − gen-0 delta (the demo's drift-cancelled number).
	MetricDelta Metric = iota
	// MetricSpeedup plots the candidate/baseline ratio.
	MetricSpeedup
	// MetricTokensPerSec plots raw candidate throughput.
	MetricTokensPerSec
)

// values returns the metric value per generation plus a passed mask.
func (s Series) values(m Metric) (vals []float64, passed []bool) {
	for _, g := range s.Gens {
		passed = append(passed, g.Passed())
		switch m {
		case MetricSpeedup:
			vals = append(vals, g.Speedup)
		case MetricTokensPerSec:
			vals = append(vals, g.TokensPerSec)
		default:
			vals = append(vals, g.Delta)
		}
	}
	return vals, passed
}

// RenderTerminal writes a sparkline plus a per-generation table to out — the
// live-demo view. It plots the chosen metric and shows the candidate spread
// (median ± half-range) so a reviewer sees the measurement noise behind each
// point.
func RenderTerminal(s Series, m Metric, out io.StringWriter) error {
	vals, passed := s.values(m)
	unit := s.Unit()
	w := func(str string) error { _, err := out.WriteString(str); return err }

	if err := w(fmt.Sprintf("SIA throughput series — %s (%s)\n", metricName(m), unit)); err != nil {
		return err
	}
	if err := w("  " + Sparkline(vals, passed) + "\n\n"); err != nil {
		return err
	}
	if err := w(fmt.Sprintf("  %-4s %-7s %-10s %-10s %-11s %-8s %s\n",
		"gen", "verdict", "cand med", "base med", "delta", "speedup", "cand spread")); err != nil {
		return err
	}
	for _, g := range s.Gens {
		if !g.Passed() {
			reason := g.Reason
			if reason == "" {
				reason = "correctness gate not passed"
			}
			if err := w(fmt.Sprintf("  %-4d %-7s %s\n", g.Gen, g.Verdict, firstLine(reason, 72))); err != nil {
				return err
			}
			continue
		}
		spread := "n/a"
		if g.Candidate != nil {
			spread = fmt.Sprintf("[%s … %s]", siUnit(g.Candidate.Min), siUnit(g.Candidate.Max))
		}
		if err := w(fmt.Sprintf("  %-4d %-7s %-10s %-10s %-11s %-8.3f %s\n",
			g.Gen, g.Verdict, siUnit(g.TokensPerSec), siUnit(g.Baseline), signedSI(g.Delta), g.Speedup, spread)); err != nil {
			return err
		}
	}
	return nil
}

func metricName(m Metric) string {
	switch m {
	case MetricSpeedup:
		return "speedup (cand/base)"
	case MetricTokensPerSec:
		return "candidate throughput"
	default:
		return "gen-N − gen-0 delta"
	}
}
