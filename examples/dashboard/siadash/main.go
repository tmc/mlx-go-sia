//go:build darwin

// Command siadash is a native macOS SwiftUI dashboard for the SIA
// weight-improvement loop alongside live pi-agent activity. It renders two
// panels side by side, both live-updating off a background ticker.
//
// LEFT — training (runs-localtrain). It tails the per-generation results.json
// under a runs-localtrain tree and renders:
//
//   - a held-out test_loss line chart (gen on X, test_loss on Y). Lower is
//     better, so a rising line means the model is getting WORSE; PASS and REVISE
//     generations are drawn with distinct color and symbol so the held-out gate
//     rejecting overfit reads at a glance.
//   - an agent-activity log panel streaming each generation's verdict, metric,
//     and reason ("held-out 2.47 > best 2.44: overfitting, rejected").
//
// RIGHT — pi-agent activity (github.com/tmc/cc). It scans the pi session JSONL
// under ~/.pi/agent/sessions (or -pi-dir) via the cc Pi collector and renders:
//
//   - an aggregate strip: total pi sessions, input/output tokens, tool calls.
//   - a live activity feed: the newest parsed pi entries, one row each with a
//     timestamp, role/kind, and a short summary (tool name or first ~60 chars).
//   - a tiny tool-call bar chart when any tools were used.
//
// The dashboard is a pure view: it shows the real numbers from results.json and
// the real parsed pi entries, and never fabricates. A generation whose
// results.json omits test_loss is drawn as a gap, not a zero. When no pi
// sessions are found the right panel says so explicitly rather than inventing
// rows. When no live training run tree is readable the left panel falls back to
// the captured canonical P6 series (see capturedSeries) so the demo still
// renders the true numbers.
//
// Usage:
//
//	go run ./examples/dashboard/siadash [-runs DIR] [-pi-dir DIR] [-interval 800ms]
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/tmc/swiftui"
	"github.com/tmc/swiftui/charts"
)

func init() { runtime.LockOSThread() }

const defaultRunsRoot = "/Users/tmc/go/src/github.com/tmc/mlx-go-sia/runs-localtrain"

// Palette, matching the charts example house style.
var (
	muted = swiftui.RGB(0.69, 0.73, 0.79)
	blue  = swiftui.RGB(0.23, 0.46, 0.94)
	green = swiftui.RGB(0.23, 0.71, 0.37)
	amber = swiftui.RGB(0.92, 0.62, 0.21)
	red   = swiftui.RGB(0.86, 0.29, 0.28)
)

// store holds the latest polled series behind a mutex. The ticker writes it; the
// SwiftUI builder (driven by DynamicView) reads a snapshot. version is the
// observable counter that triggers a rebuild whenever the data fingerprint moves.
type store struct {
	mu      sync.Mutex
	gens    []Gen
	live    bool // true => read from a live run tree, false => captured fallback
	version *swiftui.IntState
}

func (s *store) set(gens []Gen, live bool) {
	s.mu.Lock()
	s.gens = gens
	s.live = live
	s.mu.Unlock()
}

func (s *store) snapshot() ([]Gen, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Gen, len(s.gens))
	copy(out, s.gens)
	return out, s.live
}

// piStore holds the latest pi-agent snapshot behind a mutex, mirroring store:
// the ticker writes it; the SwiftUI builder reads a snapshot; version is the
// observable counter that drives a rebuild when the pi data fingerprint moves.
type piStore struct {
	mu      sync.Mutex
	snap    PiSnapshot
	version *swiftui.IntState
}

func (s *piStore) set(snap PiSnapshot) {
	s.mu.Lock()
	s.snap = snap
	s.mu.Unlock()
}

func (s *piStore) get() PiSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snap
}

func main() {
	runsRoot := flag.String("runs", defaultRunsRoot, "runs-localtrain root to tail (run_N/gen_M/results.json)")
	piDir := flag.String("pi-dir", "", "pi sessions dir to scan (default: ~/.pi/agent/sessions via cc; PI_CODING_AGENT_DIR honored)")
	interval := flag.Duration("interval", 800*time.Millisecond, "poll interval")
	flag.Parse()

	st := &store{version: swiftui.NewIntState(0)}
	pst := &piStore{version: swiftui.NewIntState(0)}

	// Initial training load: prefer the live tree, fall back to the captured series.
	if gens, ok := poll(*runsRoot); ok {
		st.set(gens, true)
	} else {
		st.set(capturedSeries(), false)
	}

	// Initial pi load.
	ctx := context.Background()
	pst.set(pollPi(ctx, *piDir))

	// Background training poller: re-scan the tree and bump version only when the
	// series actually changes, so the chart animates as a run appears/advances.
	go func() {
		last := func() string { g, _ := st.snapshot(); return fingerprint(g) }()
		tick := time.NewTicker(*interval)
		defer tick.Stop()
		for range tick.C {
			gens, ok := poll(*runsRoot)
			live := true
			if !ok {
				gens, live = capturedSeries(), false
			}
			fp := fingerprint(gens)
			if fp == last {
				continue
			}
			last = fp
			st.set(gens, live)
			st.version.Set(st.version.Get() + 1)
		}
	}()

	// Background pi poller: re-scan the pi sessions dir and bump version only when
	// the pi data fingerprint moves, so the right panel updates live as sessions
	// appear/disappear or new entries land — without a restart.
	go func() {
		last := piFingerprint(pst.get())
		tick := time.NewTicker(*interval)
		defer tick.Stop()
		for range tick.C {
			snap := pollPi(ctx, *piDir)
			fp := piFingerprint(snap)
			if fp == last {
				continue
			}
			last = fp
			pst.set(snap)
			pst.version.Set(pst.version.Get() + 1)
		}
	}()

	left := swiftui.VStackSpaced(14,
		header(st),
		swiftui.DynamicView(st.version, func(_ int) swiftui.View {
			gens, live := st.snapshot()
			return body(gens, live, *runsRoot)
		}).MaxFrame(-1, -1),
	).MaxFrame(-1, -1)

	right := swiftui.VStackSpaced(14,
		piHeader(pst),
		swiftui.DynamicView(pst.version, func(_ int) swiftui.View {
			return piBody(pst.get())
		}).MaxFrame(-1, -1),
	).MaxFrame(-1, -1)

	root := swiftui.HStackSpaced(16, left, right).
		Padding(16).
		BackgroundStyle("windowBackground")

	if err := swiftui.Run(swiftui.App{Windows: []swiftui.WindowConfig{{
		Title:  "SIA — weight loop (left) + live pi-agent activity (right)",
		Width:  1480,
		Height: 760,
		Root:   root,
	}}}); err != nil {
		log.Fatal(err)
	}
}

func header(st *store) swiftui.View {
	return swiftui.DynamicView(st.version, func(_ int) swiftui.View {
		gens, live := st.snapshot()
		src := "captured P6 series (no live run tree)"
		dot := amber
		if live {
			src = "live: tailing run tree"
			dot = green
		}
		best := "-"
		if g, l, ok := overallBest(gens); ok {
			best = fmt.Sprintf("%.4f (gen %d)", l, g)
		}
		return swiftui.VStackSpaced(6,
			swiftui.HStack(
				swiftui.Label("SIA Weight-Improvement Loop", "scalemass.fill").
					Font(swiftui.FontTitle2).
					FontWeight(swiftui.WeightBold),
				swiftui.Spacer(),
				swiftui.Image("circle.fill").ForegroundStyle(rgba(dot, 1)),
				swiftui.Text(src).Font(swiftui.FontCaption).ForegroundStyleNamed("secondary"),
			),
			swiftui.HStack(
				swiftui.Text("Held-out cross-entropy per generation. Lower is better — a rising line means the model is overfitting and getting worse on data it never saw.").
					Font(swiftui.FontCallout).
					ForegroundStyleNamed("secondary").
					LineLimit(0),
				swiftui.Spacer(),
				swiftui.Text("best test_loss: "+best).
					Font(swiftui.FontCallout).
					FontWeight(swiftui.WeightSemibold).
					MonospacedDigit(),
			),
		).Padding(14).
			BackgroundRoundedRect(rgba(swiftui.RGB(0.10, 0.13, 0.18), 0.92), 18).
			Border(rgba(muted, 0.18), 1)
	})
}

func body(gens []Gen, live bool, runsRoot string) swiftui.View {
	return swiftui.VStackSpaced(14,
		swiftui.GroupBox("Held-out test_loss (lower = better)",
			chartCard(gens),
		).MaxFrame(-1, 0),
		swiftui.GroupBox("Agent activity",
			activityPanel(gens, live, runsRoot),
		).MaxFrame(-1, -1),
	)
}

// chartCard builds the live line chart. PASS gens are green circles, REVISE gens
// are red diamonds; a connecting line runs through gens that have a test_loss.
// Gens without a test_loss are simply omitted (a gap), never plotted as zero.
func chartCard(gens []Gen) swiftui.View {
	if len(plottable(gens)) == 0 {
		return swiftui.VStackSpaced(8,
			swiftui.Spacer(),
			swiftui.Image("hourglass").ForegroundStyleNamed("secondary").ImageScale(swiftui.ImageScaleLarge),
			swiftui.Text("Waiting for a trained generation with a held-out test_loss...").
				Font(swiftui.FontCallout).ForegroundStyleNamed("secondary"),
			swiftui.Spacer(),
		).Frame(0, 300).Padding(12)
	}

	pts := plottable(gens)
	loMin, loMax := lossRange(pts)
	maxGen := 0
	for _, g := range pts {
		if g.Gen > maxGen {
			maxGen = g.Gen
		}
	}

	marks := make([]charts.Mark, 0, len(pts)*2+1)
	// Connecting line across all gens that have a loss (drawn first, under points).
	for _, g := range pts {
		marks = append(marks,
			charts.LineMark(
				charts.XInt("Generation", g.Gen),
				charts.YFloat("test_loss", g.TestLoss),
			).
				ForegroundStyle(muted).
				Interpolation(charts.InterpolationMonotone).
				LineStyle(charts.Stroke(2.4)),
		)
	}
	// Verdict-distinct points on top.
	for _, g := range pts {
		sym := charts.SymbolDiamond
		series := "REVISE"
		if g.Passed() {
			sym, series = charts.SymbolCircle, "PASS"
		}
		pt := charts.PointMark(
			charts.XInt("Generation", g.Gen),
			charts.YFloat("test_loss", g.TestLoss),
		).
			ForegroundStyleBy("verdict", series).
			Symbol(sym).
			SymbolSize(150).
			ZIndex(5)
		pt = pt.TextAnnotation(fmt.Sprintf("gen %d: %.4f", g.Gen, g.TestLoss), charts.AnnotationTop)
		marks = append(marks, pt)
	}
	// Best-so-far reference line: the bar the gate holds new gens to.
	if _, l, ok := overallBest(pts); ok {
		marks = append(marks,
			charts.RuleMark(charts.YFloat("best", l)).
				ForegroundStyle(green).
				LineStyle(charts.Stroke(1.4, 5, 4)).
				TextAnnotation(fmt.Sprintf("best-so-far %.4f", l), charts.AnnotationBottom),
		)
	}

	pad := (loMax - loMin) * 0.25
	if pad < 0.02 {
		pad = 0.02
	}
	chart := charts.Chart(marks...).
		ChartXScaleDomain(charts.IntegerDomain(0, maxGen+1)).
		ChartYScaleDomain(charts.NumberDomain(loMin-pad, loMax+pad)).
		ChartForegroundStyleScale(
			charts.StyleScale("PASS", green),
			charts.StyleScale("REVISE", red),
		).
		ChartLegend(charts.LegendVisible(charts.LegendPositionTop, charts.LegendAlignmentLeading, 8)).
		ChartXAxisLabel("generation").
		ChartYAxisLabel("held-out test_loss (lower = better)")

	// Re-render once on appear so the native chart lays out at full size.
	version := swiftui.NewIntState(0)
	return swiftui.DynamicView(version, func(_ int) swiftui.View {
		return chart.Frame(900, 300)
	}).OnAppear(func() { version.Set(1) }).Padding(8)
}

// activityPanel streams a row per generation (newest considerations last), each
// showing the verdict badge, the metric value, and the gate's reason.
func activityPanel(gens []Gen, live bool, runsRoot string) swiftui.View {
	rows := make([]swiftui.Viewable, 0, len(gens)+1)
	if !live {
		rows = append(rows, noteRow("No live run tree at "+runsRoot+" — showing captured P6 series."))
	}
	for i, g := range gens {
		rows = append(rows, activityRow(g, gens, i))
	}
	if len(gens) == 0 {
		rows = append(rows, noteRow("No generations yet."))
	}
	return swiftui.ScrollView(
		swiftui.VStackSpaced(8, rows...).Padding(10),
	).MaxFrame(-1, -1)
}

func activityRow(g Gen, gens []Gen, i int) swiftui.View {
	badgeColor := red
	icon := "xmark.octagon.fill"
	if g.Passed() {
		badgeColor, icon = green, "checkmark.seal.fill"
	} else if !g.Trained {
		badgeColor, icon = amber, "exclamationmark.triangle.fill"
	}

	metric := "test_loss n/a (gap)"
	if g.HasLoss {
		metric = fmt.Sprintf("test_loss %.4f", g.TestLoss)
		if g.Perplexity > 0 {
			metric += fmt.Sprintf("  ppl %.2f", g.Perplexity)
		}
	}

	reason := g.Reason
	if reason == "" {
		if best, l, ok := bestSoFar(gens, i); ok && g.HasLoss {
			if g.TestLoss <= l {
				reason = fmt.Sprintf("new best (<= prior best %.4f from gen %d)", l, best)
			} else {
				reason = fmt.Sprintf("held-out %.4f > best-so-far %.4f (gen %d): overfitting, rejected", g.TestLoss, l, best)
			}
		} else {
			reason = "no prior baseline to compare against"
		}
	}

	return swiftui.HStackSpaced(10,
		swiftui.Image(icon).ForegroundStyle(rgba(badgeColor, 1)).Frame(20, 0),
		swiftui.VStackSpaced(3,
			swiftui.HStackSpaced(8,
				swiftui.Text(fmt.Sprintf("run %d · gen %d", g.Run, g.Gen)).
					Font(swiftui.FontHeadline).FontWeight(swiftui.WeightSemibold),
				badge(g.Verdict, badgeColor),
				swiftui.Text(metric).Font(swiftui.FontCallout).MonospacedDigit().ForegroundStyleNamed("secondary"),
				swiftui.Spacer(),
			),
			swiftui.HStack(
				swiftui.Text(reason).Font(swiftui.FontCaption).ForegroundStyleNamed("secondary").LineLimit(0),
				swiftui.Spacer(),
			),
		),
	).Padding(10).
		BackgroundRoundedRect(rgba(swiftui.RGB(0.16, 0.18, 0.22), 0.55), 10).
		Border(rgba(badgeColor, 0.30), 1)
}

func badge(text string, color swiftui.Color) swiftui.View {
	return swiftui.Text(text).
		Font(swiftui.FontCaption2).
		FontWeight(swiftui.WeightBold).
		ForegroundStyle(rgba(color, 1)).
		AsView().
		Padding(5).
		BackgroundRoundedRect(rgba(color, 0.16), 7)
}

func noteRow(text string) swiftui.View {
	return swiftui.HStackSpaced(8,
		swiftui.Image("info.circle.fill").ForegroundStyle(rgba(amber, 1)),
		swiftui.Text(text).Font(swiftui.FontCaption).ForegroundStyleNamed("secondary").LineLimit(0),
		swiftui.Spacer(),
	).Padding(8).BackgroundRoundedRect(rgba(amber, 0.10), 8)
}

// plottable returns gens that have a present, positive test_loss (the only ones
// that belong on the chart). Gens without a loss are gaps, not zeros.
func plottable(gens []Gen) []Gen {
	out := make([]Gen, 0, len(gens))
	for _, g := range gens {
		if g.HasLoss && g.TestLoss > 0 {
			out = append(out, g)
		}
	}
	return out
}

func lossRange(gens []Gen) (min, max float64) {
	min, max = gens[0].TestLoss, gens[0].TestLoss
	for _, g := range gens {
		if g.TestLoss < min {
			min = g.TestLoss
		}
		if g.TestLoss > max {
			max = g.TestLoss
		}
	}
	return
}

// overallBest returns the lowest test_loss across the whole series.
func overallBest(gens []Gen) (gen int, loss float64, ok bool) {
	for _, g := range gens {
		if !g.HasLoss || g.TestLoss <= 0 {
			continue
		}
		if !ok || g.TestLoss < loss {
			gen, loss, ok = g.Gen, g.TestLoss, true
		}
	}
	return
}

func rgba(c swiftui.Color, a float64) swiftui.Color {
	return swiftui.RGBA(c.R, c.G, c.B, a)
}

// -------------------- pi-agent (right) panel --------------------

// piHeader renders the title and a live/empty status dot for the pi panel.
func piHeader(pst *piStore) swiftui.View {
	return swiftui.DynamicView(pst.version, func(_ int) swiftui.View {
		snap := pst.get()
		src := "no pi runs found"
		dot := amber
		switch {
		case snap.Error != "":
			src, dot = "scan error", red
		case snap.Agg.Found && snap.Agg.Sessions > 0:
			src, dot = fmt.Sprintf("live: %d pi session(s)", snap.Agg.Sessions), green
		case snap.Agg.Found:
			src, dot = "pi dir present, no sessions", amber
		}
		return swiftui.VStackSpaced(6,
			swiftui.HStack(
				swiftui.Label("Live pi-agent activity", "terminal.fill").
					Font(swiftui.FontTitle2).
					FontWeight(swiftui.WeightBold),
				swiftui.Spacer(),
				swiftui.Image("circle.fill").ForegroundStyle(rgba(dot, 1)),
				swiftui.Text(src).Font(swiftui.FontCaption).ForegroundStyleNamed("secondary"),
			),
			swiftui.HStack(
				swiftui.Text("Real parsed pi session JSONL via github.com/tmc/cc. Every number folds actual entries; absent fields show n/a, never a zero that reads as data.").
					Font(swiftui.FontCallout).
					ForegroundStyleNamed("secondary").
					LineLimit(0),
				swiftui.Spacer(),
			),
		).Padding(14).
			BackgroundRoundedRect(rgba(swiftui.RGB(0.10, 0.13, 0.18), 0.92), 18).
			Border(rgba(muted, 0.18), 1)
	})
}

func piBody(snap PiSnapshot) swiftui.View {
	return swiftui.VStackSpaced(14,
		swiftui.GroupBox("Aggregate (all pi sessions)",
			piAggregateStrip(snap.Agg),
		).MaxFrame(-1, 0),
		toolBarCard(snap.Agg),
		swiftui.GroupBox("Live activity feed (newest last)",
			piFeedPanel(snap),
		).MaxFrame(-1, -1),
	)
}

// piAggregateStrip renders the real rollup: sessions, input tokens, output
// tokens (flagged approximate, since pi JSONL carries only the streaming-start
// snapshot), tool calls, and user turns. Zero is shown only when zero is the
// true folded value (e.g. 0 tool calls); a missing pi dir shows n/a instead.
func piAggregateStrip(a PiAgg) swiftui.View {
	if !a.Found {
		return swiftui.HStackSpaced(12,
			statCard("sessions", "n/a", muted),
			statCard("input tok", "n/a", muted),
			statCard("output tok", "n/a", muted),
			statCard("tool calls", "n/a", muted),
		).Padding(8)
	}

	outVal := commas(a.OutputTokens)
	outLabel := "output tok"
	if !a.OutputTokensReliable && a.OutputTokens > 0 {
		outVal = "~" + outVal
		outLabel = "output tok (approx)"
	}
	toolColor := muted
	if a.ToolCalls > 0 {
		toolColor = blue
	}
	return swiftui.VStackSpaced(8,
		swiftui.HStackSpaced(12,
			statCard("sessions", commas(a.Sessions), blue),
			statCard("user turns", commas(a.Turns), green),
			statCard("tool calls", commas(a.ToolCalls), toolColor),
		),
		swiftui.HStackSpaced(12,
			statCard("input tok", commas(a.InputTokens), amber),
			statCard(outLabel, outVal, amber),
			statCard("cache tok", commas(a.CacheTokens), muted),
		),
	).Padding(8)
}

func statCard(label, value string, color swiftui.Color) swiftui.View {
	return swiftui.VStackSpaced(2,
		swiftui.Text(value).
			Font(swiftui.FontTitle3).
			FontWeight(swiftui.WeightBold).
			MonospacedDigit().
			ForegroundStyle(rgba(color, 1)),
		swiftui.Text(label).
			Font(swiftui.FontCaption2).
			ForegroundStyleNamed("secondary"),
	).Padding(10).
		MaxFrame(-1, 0).
		BackgroundRoundedRect(rgba(color, 0.12), 10).
		Border(rgba(color, 0.30), 1)
}

// toolBarCard draws a tiny per-tool count bar chart when any tools were used.
// pi chat sessions with zero tool calls render an honest "no tool calls"
// note rather than an empty or fabricated chart.
func toolBarCard(a PiAgg) swiftui.View {
	if !a.Found || a.ToolCalls == 0 || len(a.ToolBreak) == 0 {
		note := "No tool calls recorded in these pi sessions yet."
		if !a.Found {
			note = "No pi sessions dir — tool breakdown n/a."
		}
		return swiftui.GroupBox("Tool calls by name",
			swiftui.HStackSpaced(8,
				swiftui.Image("wrench.and.screwdriver").ForegroundStyleNamed("secondary"),
				swiftui.Text(note).Font(swiftui.FontCaption).ForegroundStyleNamed("secondary").LineLimit(0),
				swiftui.Spacer(),
			).Padding(10),
		).MaxFrame(-1, 0)
	}

	type tc struct {
		name string
		n    int
	}
	tools := make([]tc, 0, len(a.ToolBreak))
	for name, n := range a.ToolBreak {
		tools = append(tools, tc{name, n})
	}
	sort.Slice(tools, func(i, j int) bool {
		if tools[i].n != tools[j].n {
			return tools[i].n > tools[j].n
		}
		return tools[i].name < tools[j].name
	})

	marks := make([]charts.Mark, 0, len(tools))
	for _, t := range tools {
		marks = append(marks,
			charts.BarMark(
				charts.YString("tool", t.name),
				charts.XInt("calls", t.n),
			).
				ForegroundStyle(blue).
				TextAnnotation(fmt.Sprintf("%d", t.n), charts.AnnotationTrailing),
		)
	}
	chart := charts.Chart(marks...).
		ChartXAxisLabel("calls").
		ChartYAxisLabel("tool")

	version := swiftui.NewIntState(0)
	return swiftui.GroupBox("Tool calls by name",
		swiftui.DynamicView(version, func(_ int) swiftui.View {
			return chart.Frame(0, float64(40+len(tools)*26))
		}).OnAppear(func() { version.Set(1) }).Padding(8),
	).MaxFrame(-1, 0)
}

// piFeedPanel renders the live activity rows, newest last. An empty feed shows
// an explicit empty state; rows never invent data.
func piFeedPanel(snap PiSnapshot) swiftui.View {
	rows := make([]swiftui.Viewable, 0, len(snap.Feed)+1)
	if snap.Error != "" {
		rows = append(rows, noteRow("pi scan error: "+snap.Error))
	}
	if !snap.Agg.Found {
		rows = append(rows, noteRow("No pi sessions directory found (looked for ~/.pi/agent/sessions). Nothing to show — not fabricating rows."))
	} else if len(snap.Feed) == 0 {
		rows = append(rows, noteRow("pi dir present but no parsed activity yet."))
	}
	for _, e := range snap.Feed {
		rows = append(rows, piFeedRow(e))
	}
	return swiftui.ScrollView(
		swiftui.VStackSpaced(8, rows...).Padding(10),
	).MaxFrame(-1, -1)
}

func piFeedRow(e PiEntry) swiftui.View {
	icon, color := feedIcon(e)

	when := "  --:--:--"
	if e.HasTime {
		when = e.When.Local().Format("15:04:05")
	}

	tag := e.Role
	if tag == "" {
		tag = e.Kind
	}

	return swiftui.HStackSpaced(10,
		swiftui.Image(icon).ForegroundStyle(rgba(color, 1)).Frame(20, 0),
		swiftui.VStackSpaced(3,
			swiftui.HStackSpaced(8,
				swiftui.Text(when).
					Font(swiftui.FontCaption).MonospacedDigit().ForegroundStyleNamed("secondary"),
				badge(tag, color),
				swiftui.Text(e.Session).Font(swiftui.FontCaption2).ForegroundStyleNamed("secondary").LineLimit(1),
				swiftui.Spacer(),
			),
			swiftui.HStack(
				swiftui.Text(e.Summary).Font(swiftui.FontCallout).ForegroundStyleNamed("primary").LineLimit(0),
				swiftui.Spacer(),
			),
		),
	).Padding(10).
		BackgroundRoundedRect(rgba(swiftui.RGB(0.16, 0.18, 0.22), 0.55), 10).
		Border(rgba(color, 0.30), 1)
}

// feedIcon picks an SF Symbol and color for a feed row by kind/role.
func feedIcon(e PiEntry) (string, swiftui.Color) {
	switch e.Kind {
	case "tool_use":
		return "hammer.fill", blue
	case "tool_result":
		return "arrow.turn.down.right", green
	}
	switch e.Role {
	case "user":
		return "person.fill", amber
	case "assistant":
		return "sparkles", green
	}
	return "circle.dotted", muted
}

// commas formats an int with thousands separators for the stat cards.
func commas(n int) string {
	s := fmt.Sprintf("%d", n)
	neg := false
	if len(s) > 0 && s[0] == '-' {
		neg, s = true, s[1:]
	}
	var b []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			b = append(b, ',')
		}
		b = append(b, c)
	}
	if neg {
		return "-" + string(b)
	}
	return string(b)
}
