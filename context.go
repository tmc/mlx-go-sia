package sia

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// GenerationRecord captures what happened in one generation, the input the
// [ContextManager] needs to render that generation's context.md entry. It
// mirrors the reference's add_generation gen_data dict.
type GenerationRecord struct {
	GenNum          int
	Success         bool
	Timestamp       string  // formatted "2006-01-02 15:04:05"
	Duration        float64 // seconds
	AgentPath       string  // path to this generation's target_agent.py / train.py
	GenDir          string  // path to the generation directory
	ImprovementPath string  // empty if no improvement.md was written
	ExecutionType   string  // "Single" or "Multi-trajectory"
}

// agentStats holds the file statistics the context manager reports for a
// generation's agent (mirrors _get_agent_stats).
type agentStats struct {
	size  int64
	lines int
}

// genState is the per-generation state the context manager retains for delta
// and summary calculations (mirrors the dicts appended to self.generations).
type genState struct {
	genNum  int
	stats   agentStats
	metrics map[string]any
	success bool
}

// ContextSummarizer produces the optional "Evolution Summary (LLM Analysis)"
// block the reference generates with a model call (_generate_llm_summary). It is
// the seam for the one genuinely SDK-bound part of the context manager; the
// orchestrator's agent engine is not wired in here by default, so the zero
// (nil) summarizer omits the block, matching the reference's behavior when the
// summary call returns None.
type ContextSummarizer interface {
	// Summarize returns a 2-4 sentence evolution summary for genNum (>= 2),
	// given the rendered prompt context. An empty string omits the block.
	Summarize(genNum int, promptContext string) string
}

// ContextManager accumulates the run's evolution history into context.md, the
// file the feedback agent reads before each rewrite. It is a faithful port of
// the reference's context_manager.py: per-generation agent stats, extracted
// metrics, improvement.md insights, metric deltas, an optional LLM evolution
// summary, and a closing summary with best-generation and code-growth stats.
// The zero value is not usable; construct with [NewContextManager].
type ContextManager struct {
	layout     RunLayout
	runConfig  map[string]string
	cfg        Config
	summarizer ContextSummarizer
	header     string     // the initialize() header, retained for rewrites
	entries    []string   // rendered per-generation entries
	state      []genState // retained per-generation state for deltas/summary
}

// NewContextManager creates a context manager rooted at the run's layout.
// runConfig holds run-level metadata (task_dir, meta_model, task_model,
// agent_impl, max_gen) surfaced in the context.md header. It uses
// [DefaultConfig] tunables and no LLM summarizer; see [ContextManager.WithConfig]
// and [ContextManager.WithSummarizer].
func NewContextManager(layout RunLayout, runConfig map[string]string) *ContextManager {
	return &ContextManager{layout: layout, runConfig: runConfig, cfg: DefaultConfig()}
}

// WithConfig sets the truncation/limit tunables and returns c.
func (c *ContextManager) WithConfig(cfg Config) *ContextManager {
	c.cfg = cfg
	return c
}

// WithSummarizer installs the optional evolution-summary engine and returns c.
func (c *ContextManager) WithSummarizer(s ContextSummarizer) *ContextManager {
	c.summarizer = s
	return c
}

// Initialize writes context.md with the run header.
func (c *ContextManager) Initialize() error {
	c.header = c.renderHeader()
	return c.write()
}

// AddGeneration records a generation and appends its entry to context.md.
func (c *ContextManager) AddGeneration(rec GenerationRecord) error {
	stats := c.agentStats(rec.AgentPath)

	var sizePct float64
	var linesDelta int
	if len(c.state) > 0 {
		prev := c.state[len(c.state)-1].stats
		if prev.size != 0 {
			sizePct = float64(stats.size-prev.size) / float64(prev.size) * 100
		}
		linesDelta = stats.lines - prev.lines
	}

	metrics := c.extractMetrics(rec.GenDir)

	var insights []string
	if rec.ImprovementPath != "" && isFile(rec.ImprovementPath) {
		insights = c.extractInsights(rec.ImprovementPath)
	}

	summary := c.generateSummary(rec, metrics)

	entry := c.formatEntry(rec, stats, sizePct, linesDelta, metrics, insights, summary)
	c.entries = append(c.entries, entry)
	c.state = append(c.state, genState{
		genNum:  rec.GenNum,
		stats:   stats,
		metrics: metrics,
		success: rec.Success,
	})
	return c.write()
}

// Finalize appends the closing summary statistics to context.md.
func (c *ContextManager) Finalize() error {
	if len(c.state) == 0 {
		return nil
	}
	c.entries = append(c.entries, c.renderSummary())
	return c.write()
}

// Generations returns the recorded generations in order.
func (c *ContextManager) Generations() []GenerationRecord {
	out := make([]GenerationRecord, 0, len(c.state))
	for _, s := range c.state {
		out = append(out, GenerationRecord{GenNum: s.genNum, Success: s.success})
	}
	return out
}

func (c *ContextManager) write() error {
	if err := os.MkdirAll(c.layout.RunDir, 0o755); err != nil {
		return err
	}
	var b strings.Builder
	b.WriteString(c.header)
	for _, e := range c.entries {
		b.WriteString(e)
	}
	return os.WriteFile(c.layout.ContextMD(), []byte(b.String()), 0o644)
}

// renderHeader mirrors initialize(): the run header block.
func (c *ContextManager) renderHeader() string {
	get := func(k string) string {
		if v, ok := c.runConfig[k]; ok && v != "" {
			return v
		}
		return "N/A"
	}
	return fmt.Sprintf(`# Run Context: %s

**Task**: %s
**Meta Model**: %s
**Task Model**: %s
**Agent impl**: %s
**Started**: %s
**Max Generations**: %s

---

`,
		baseName(c.layout.RunDir),
		get("task_dir"), get("meta_model"), get("task_model"), get("agent_impl"),
		get("started"), get("max_gen"))
}

// formatEntry mirrors _format_generation_entry. Each entry is the markdown block
// followed by the reference's "\n---\n\n" separator that add_generation appends.
func (c *ContextManager) formatEntry(rec GenerationRecord, stats agentStats, sizePct float64, linesDelta int, metrics map[string]any, insights []string, llmSummary string) string {
	status := "✗ FAILED"
	if rec.Success {
		status = "✓ SUCCESS"
	}

	var b strings.Builder
	fmt.Fprintf(&b, `## Generation %d

**Status**: %s
**Timestamp**: %s
**Duration**: %.1fs

### Target Agent Changes
`, rec.GenNum, status, orNA(rec.Timestamp), rec.Duration)

	if rec.GenNum == 1 {
		fmt.Fprintf(&b, "- Initial agent created by meta-agent\n- File size: %s bytes\n- Lines of code: %d\n",
			commaInt(stats.size), stats.lines)
	} else {
		sizeStr := fmt.Sprintf("%.1f%%", sizePct)
		if sizePct > 0 {
			sizeStr = fmt.Sprintf("+%.1f%%", sizePct)
		}
		linesStr := strconv.Itoa(linesDelta)
		if linesDelta > 0 {
			linesStr = "+" + linesStr
		}
		fmt.Fprintf(&b, "- Modified by feedback agent\n- File size: %s bytes (%s)\n- Lines: %d (%s lines)\n",
			commaInt(stats.size), sizeStr, stats.lines, linesStr)
		if len(insights) > 0 {
			b.WriteString("- Key changes from improvement.md:\n")
			for _, ins := range firstN(insights, 3) {
				if len(ins) > c.cfg.InsightPreviewLimit {
					ins = ins[:c.cfg.InsightPreviewLimit] + "..."
				}
				fmt.Fprintf(&b, "  * %s\n", ins)
			}
		}
	}

	if llmSummary != "" {
		fmt.Fprintf(&b, "\n### Evolution Summary (LLM Analysis)\n%s\n", llmSummary)
	}

	fmt.Fprintf(&b, `
### Execution Summary
- Execution status: %s
- Output format: %s

### Performance Metrics
`, status, orUnknown(rec.ExecutionType))

	if len(metrics) > 0 {
		for _, k := range sortedAnyKeys(metrics) {
			b.WriteString(formatMetricLine(k, metrics[k]))
		}
	} else {
		b.WriteString("- No structured metrics found\n")
	}

	if rec.GenNum > 1 && len(c.state) > 0 {
		prevMetrics := c.state[len(c.state)-1].metrics
		var changes []string
		for _, k := range sortedAnyKeys(metrics) {
			pv, ok := prevMetrics[k]
			if !ok {
				continue
			}
			cur, curOK := numericValue(metrics[k])
			prev, prevOK := numericValue(pv)
			if curOK && prevOK {
				changes = append(changes, fmt.Sprintf("- %s: %+.2f", k, cur-prev))
			}
		}
		if len(changes) > 0 {
			b.WriteString("\n### Changes vs Previous Generation\n")
			b.WriteString(strings.Join(changes, "\n") + "\n")
		}
	}

	return b.String() + "\n---\n\n"
}

// generateSummary mirrors _generate_llm_summary: skip for gen 1, otherwise build
// the prompt context and ask the (optional) summarizer. With no summarizer the
// block is omitted, matching the reference's None path.
func (c *ContextManager) generateSummary(rec GenerationRecord, metrics map[string]any) string {
	if rec.GenNum == 1 || c.summarizer == nil {
		return ""
	}
	curCode, ok := readFileLimited(rec.AgentPath)
	if !ok {
		return ""
	}
	prevAgent := joinPath(c.layout.GenDir(rec.GenNum-1), agentFileName(rec.AgentPath))
	prevCode, ok := readFileLimited(prevAgent)
	if !ok {
		prevCode = "Not available"
	}
	improvement := ""
	if rec.ImprovementPath != "" && isFile(rec.ImprovementPath) {
		if s, ok := readFileLimited(rec.ImprovementPath); ok {
			improvement = s
		}
	}
	var prevMetrics map[string]any
	if len(c.state) > 0 {
		prevMetrics = c.state[len(c.state)-1].metrics
	}
	comparison := c.formatMetricsComparison(prevMetrics, metrics)

	improvementBody := improvement
	if improvementBody == "" {
		improvementBody = "No improvement.md found"
	}
	prompt := fmt.Sprintf(`You are analyzing the evolution of an AI agent across generations.

**TASK**: Provide a concise summary (2-4 sentences) of what changed between Generation %d and Generation %d, focusing on:
1. Key code/structural changes made
2. The reasoning behind these changes (from improvement.md)
3. Impact on performance metrics (if any)

**IMPROVEMENT PLAN** (improvement.md from gen_%d):
%s
%s
%s

**PREVIOUS AGENT CODE** (gen_%d/target_agent.py):
%spython
%s%s
%s

**CURRENT AGENT CODE** (gen_%d/target_agent.py):
%spython
%s%s
%s

**METRICS COMPARISON**:
%s

**YOUR SUMMARY** (2-4 sentences, be specific about what changed and why):
`,
		rec.GenNum-1, rec.GenNum,
		rec.GenNum, fence, improvementBody, fence,
		rec.GenNum-1, fence, clipPreview(prevCode, c.cfg.AgentCodePreviewLimit), ellipsisIf(prevCode, c.cfg.AgentCodePreviewLimit), fence,
		rec.GenNum, fence, clipPreview(curCode, c.cfg.AgentCodePreviewLimit), ellipsisIf(curCode, c.cfg.AgentCodePreviewLimit), fence,
		comparison)

	return strings.TrimSpace(c.summarizer.Summarize(rec.GenNum, prompt))
}

// formatMetricsComparison mirrors _format_metrics_comparison.
func (c *ContextManager) formatMetricsComparison(prev, cur map[string]any) string {
	if len(prev) == 0 && len(cur) == 0 {
		return "No metrics available for comparison"
	}
	keys := map[string]struct{}{}
	for k := range prev {
		keys[k] = struct{}{}
	}
	for k := range cur {
		keys[k] = struct{}{}
	}
	all := make([]string, 0, len(keys))
	for k := range keys {
		all = append(all, k)
	}
	sort.Strings(all)

	var lines []string
	for _, k := range all {
		pv := metricStr(prev, k)
		cv := metricStr(cur, k)
		delta := ""
		if pn, ok1 := numericValue(prev[k]); ok1 {
			if cn, ok2 := numericValue(cur[k]); ok2 {
				if _, hasP := prev[k]; hasP {
					if _, hasC := cur[k]; hasC {
						delta = fmt.Sprintf(" (%+.2f)", cn-pn)
					}
				}
			}
		}
		lines = append(lines, fmt.Sprintf("- %s: %s → %s%s", k, pv, cv, delta))
	}
	if len(lines) == 0 {
		return "No metrics to compare"
	}
	return strings.Join(lines, "\n")
}

// renderSummary mirrors finalize(): closing statistics.
func (c *ContextManager) renderSummary() string {
	first := c.state[0]
	last := c.state[len(c.state)-1]

	successful := 0
	for _, g := range c.state {
		if g.success {
			successful++
		}
	}

	bestGen := -1
	bestMetric := math.Inf(-1)
	for _, g := range c.state {
		if acc, ok := numericValue(g.metrics["accuracy"]); ok {
			if acc > bestMetric {
				bestMetric = acc
				bestGen = g.genNum
			}
		}
	}

	evolution := "N/A"
	if fa, ok1 := numericValue(first.metrics["accuracy"]); ok1 {
		if la, ok2 := numericValue(last.metrics["accuracy"]); ok2 {
			evolution = fmt.Sprintf("%.2f%% → %.2f%% (%+.2f%%)", fa, la, la-fa)
		}
	}

	bestStr := "N/A"
	if bestGen >= 0 {
		bestStr = strconv.Itoa(bestGen)
	}
	// best_metric stays -inf when no generation reported accuracy; Python's
	// f"{-inf:.2f}" renders "-inf", which a Go %.2f would render "-Inf".
	bestAccStr := pyFixed2(bestMetric)

	return fmt.Sprintf(`## Summary Statistics

**Total Generations**: %d
**Successful Executions**: %d
**Best Performance**: Generation %s (%s%% accuracy)

**Evolution**:
- %s

**Code Growth**:
- Initial: %d lines (%s bytes)
- Final: %d lines (%s bytes)
- Growth: %d lines (%s bytes)
`,
		len(c.state), successful, bestStr, bestAccStr,
		evolution,
		first.stats.lines, commaInt(first.stats.size),
		last.stats.lines, commaInt(last.stats.size),
		last.stats.lines-first.stats.lines, commaSignedInt(last.stats.size-first.stats.size))
}

// agentStats mirrors _get_agent_stats: line count + byte size of the agent file.
func (c *ContextManager) agentStats(path string) agentStats {
	data, err := os.ReadFile(path)
	if err != nil {
		return agentStats{}
	}
	return agentStats{size: int64(len(data)), lines: countLines(data)}
}

// extractMetrics mirrors _extract_metrics: results.json scalars first, then
// detailed_results.json, then stdout regex — each only when nothing earlier hit.
func (c *ContextManager) extractMetrics(genDir string) map[string]any {
	metrics := map[string]any{}

	collectScalars := func(data map[string]any) {
		for k, v := range data {
			if s, ok := scalarMetric(v); ok {
				metrics[k] = s
			}
		}
	}

	if data, ok := loadJSONObject(joinPath(genDir, NameResultsJSON)); ok {
		collectScalars(data)
	}

	if len(metrics) == 0 {
		if data, ok := loadJSONObject(joinPath(genDir, NameDetailedResults)); ok {
			collectScalars(data)
		}
	}

	if len(metrics) == 0 {
		for _, name := range []string{NameStdoutLog, NameTrainStdoutLog} {
			p := joinPath(genDir, name)
			if isFile(p) {
				maps.Copy(metrics, c.parseStdoutMetrics(p))
				break
			}
		}
	}

	return metrics
}

var stdoutMetricPatterns = []struct {
	name     string
	patterns []*regexp.Regexp
}{
	{"accuracy", []*regexp.Regexp{
		regexp.MustCompile(`(?i)accuracy[:\s=]+(\d+\.?\d*)\s*%?`),
		regexp.MustCompile(`(?i)final\s+accuracy[:\s=]+(\d+\.?\d*)\s*%?`),
		regexp.MustCompile(`(?i)test\s+accuracy[:\s=]+(\d+\.?\d*)\s*%?`),
	}},
	{"validation", []*regexp.Regexp{
		regexp.MustCompile(`(?i)validation[:\s=]+(\d+\.?\d*)`),
		regexp.MustCompile(`(?i)val[:\s=]+(\d+\.?\d*)`),
	}},
	{"correct", []*regexp.Regexp{
		regexp.MustCompile(`(?i)(\d+)\s*/\s*\d+\s+correct`),
		regexp.MustCompile(`(?i)correct[:\s=]+(\d+)`),
	}},
	{"total", []*regexp.Regexp{
		regexp.MustCompile(`(?i)\d+\s*/\s*(\d+)\s+(?:questions|samples|total)`),
	}},
}

// parseStdoutMetrics mirrors _parse_stdout_metrics: regex patterns over the log.
func (c *ContextManager) parseStdoutMetrics(path string) map[string]any {
	metrics := map[string]any{}
	data, err := os.ReadFile(path)
	if err != nil {
		return metrics
	}
	content := string(data)
	for _, mp := range stdoutMetricPatterns {
		for _, re := range mp.patterns {
			if m := re.FindStringSubmatch(content); m != nil {
				if f, err := strconv.ParseFloat(m[1], 64); err == nil {
					metrics[mp.name] = f
					break
				}
			}
		}
	}
	return metrics
}

var (
	bulletRe   = regexp.MustCompile(`(?m)^[-*]\s+(.+)$`)
	numberedRe = regexp.MustCompile(`(?m)^\d+\.\s+(.+)$`)
)

// extractInsights mirrors _extract_insights: bullets + numbered items from
// improvement.md, filtered to len>20 and not ending in ":", first 5.
func (c *ContextManager) extractInsights(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	content := string(data)
	var all []string
	for _, m := range bulletRe.FindAllStringSubmatch(content, -1) {
		all = append(all, m[1])
	}
	for _, m := range numberedRe.FindAllStringSubmatch(content, -1) {
		all = append(all, m[1])
	}
	var meaningful []string
	for _, ins := range all {
		t := strings.TrimSpace(ins)
		if len(t) > 20 && !strings.HasSuffix(t, ":") {
			meaningful = append(meaningful, t)
		}
	}
	return firstN(meaningful, 5)
}

// --- small helpers --------------------------------------------------------

func orNA(s string) string {
	if s == "" {
		return "N/A"
	}
	return s
}

func orUnknown(s string) string {
	if s == "" {
		return "Unknown"
	}
	return s
}

// countLines mirrors Python f.readlines(): the number of lines, where a final
// line without a trailing newline still counts.
func countLines(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := 0
	for _, b := range data {
		if b == '\n' {
			n++
		}
	}
	if data[len(data)-1] != '\n' {
		n++
	}
	return n
}

func firstN[T any](s []T, n int) []T {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// commaInt formats n with thousands separators, like Python's "{:,}".
func commaInt(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := strconv.FormatInt(n, 10)
	var out strings.Builder
	if neg {
		out.WriteByte('-')
	}
	pre := len(s) % 3
	if pre == 0 {
		pre = 3
	}
	out.WriteString(s[:pre])
	for i := pre; i < len(s); i += 3 {
		out.WriteByte(',')
		out.WriteString(s[i : i+3])
	}
	return out.String()
}

// commaSignedInt formats n like Python's "{:+,}" (always-signed, grouped).
func commaSignedInt(n int64) string {
	if n >= 0 {
		return "+" + commaInt(n)
	}
	return commaInt(n)
}

// scalarMetric converts a decoded JSON value into the Go representation the
// context manager stores for a metric — int64 (integer-form number), float64
// (float-form number), string, or bool — mirroring Python's int/float/str/bool
// from json.load. It reports ok=false for containers (dict/list), which the
// reference skips. Numbers arrive as json.Number because results.json is decoded
// with UseNumber, so 40.0 stays a float and 12 stays an int.
func scalarMetric(v any) (any, bool) {
	switch x := v.(type) {
	case json.Number:
		return jsonNumberValue(x), true
	case string, bool:
		return x, true
	case float64: // values not from a UseNumber decoder
		return x, true
	default:
		return nil, false
	}
}

// numericValue mirrors the reference's float(str(v).rstrip("%")) coercion used
// for deltas and accuracy comparisons. bool is not numeric (Python's isinstance
// checks exclude it from the delta paths that matter here).
func numericValue(v any) (float64, bool) {
	switch x := v.(type) {
	case int64:
		return float64(x), true
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	case bool:
		return 0, false
	case string:
		f, err := strconv.ParseFloat(strings.TrimSuffix(x, "%"), 64)
		return f, err == nil
	default:
		return 0, false
	}
}

// metricStr renders a metric value for the comparison block, or "N/A" if absent,
// matching prev_metrics.get(key, "N/A").
func metricStr(m map[string]any, k string) string {
	v, ok := m[k]
	if !ok {
		return "N/A"
	}
	return scalarString(v)
}

// scalarString renders a stored scalar the way Python str() would: ints plain,
// floats via Python's repr-ish formatting, strings verbatim, bools True/False.
func scalarString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		if x {
			return "True"
		}
		return "False"
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return pyFloatStr(x)
	default:
		return fmt.Sprint(v)
	}
}

// pyFixed2 formats a float like Python's f"{x:.2f}", which renders non-finite
// values lowercase ("inf", "-inf", "nan") where Go's %.2f would render "Inf".
func pyFixed2(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	default:
		return fmt.Sprintf("%.2f", f)
	}
}

// pyFloatStr formats a float the way Python str(float) does for the values that
// reach context.md: an integral float keeps a trailing ".0" (e.g. 40.0 -> "40.0").
func pyFloatStr(f float64) string {
	if f == math.Trunc(f) && !math.IsInf(f, 0) && !math.IsNaN(f) {
		return strconv.FormatFloat(f, 'f', 1, 64)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// formatMetricLine mirrors the reference's Performance Metrics rendering:
// a Python float prints with %.2f (so 40.0 -> "40.00"), everything else via str().
func formatMetricLine(k string, v any) string {
	if f, ok := v.(float64); ok {
		return fmt.Sprintf("- %s: %.2f\n", k, f)
	}
	return fmt.Sprintf("- %s: %s\n", k, scalarString(v))
}

func sortedAnyKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// loadJSONObject loads path as a JSON object, decoding numbers as json.Number so
// integer- and float-form numbers can be distinguished the way Python's json
// does (40.0 -> float, 12 -> int). Returns ok=false on read/parse error or a
// non-object top level.
func loadJSONObject(path string) (map[string]any, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, false
	}
	obj, ok := v.(map[string]any)
	return obj, ok
}

// jsonNumberValue converts a json.Number to a Go int64 (integer form) or float64
// (float form), matching Python's int/float distinction from json.load.
func jsonNumberValue(n json.Number) any {
	s := string(n)
	if strings.ContainsAny(s, ".eE") {
		f, _ := n.Float64()
		return f
	}
	if i, err := n.Int64(); err == nil {
		return i
	}
	f, _ := n.Float64()
	return f
}

func readFileLimited(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(data), true
}

func clipPreview(s string, limit int) string {
	if len(s) > limit {
		return s[:limit]
	}
	return s
}

func ellipsisIf(s string, limit int) string {
	if len(s) > limit {
		return "..."
	}
	return ""
}

func agentFileName(path string) string {
	return baseName(path)
}

func baseName(p string) string {
	p = strings.TrimRight(p, "/")
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}
