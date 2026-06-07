package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	sia "github.com/tmc/mlx-go-sia"
)

// evidenceKeys is the fixed display order for the seven coverage-map evidence
// booleans, weightiest first. It matches the rubric weighting so the capture
// reads top-to-bottom in order of how much each key matters.
var evidenceKeys = []string{
	"validator_command",
	"artifact_manifest_hash",
	"model_backed_or_opt_in_command",
	"falsifier_rows",
	"control_rows",
	"fixture_row",
	"heavy_skip_narrowed_or_cleared",
}

// genCapture is the per-generation slice of results.json the capture emitter
// reads. It is intentionally the honest-recompute output — the evidence_state
// booleans the evaluator derived itself and the per-key recompute trace — not
// anything the simulated agent self-reported.
type genCapture struct {
	Verdict       string            `json:"verdict"`
	AdvisoryScore float64           `json:"advisory_score"`
	FastCheckOK   bool              `json:"fast_check_ok"`
	EvidenceState map[string]bool   `json:"evidence_state"`
	Blockers      []string          `json:"blockers"`
	Detail        map[string]string `json:"evidence_detail"`
	RowID         string            `json:"row_id"`
	Status        string            `json:"status"`

	// claimed is the agent's planted self-report, read from agentSelfReportName
	// beside results.json in the SAME gen dir. nil for honest prototypes (none
	// planted). It is shown only to contrast with EvidenceState; never trusted.
	claimed *selfReport
}

// captureProto holds one prototype's per-generation captures for rendering.
type captureProto struct {
	id     string
	status string
	gens   []genCapture
}

// readGenCapture loads one generation's results.json. A missing file yields a
// zero-value capture with verdict "MISSING" so the table still renders.
func readGenCapture(path string) genCapture {
	data, err := os.ReadFile(path)
	if err != nil {
		return genCapture{Verdict: "MISSING"}
	}
	var g genCapture
	if err := json.Unmarshal(data, &g); err != nil {
		return genCapture{Verdict: "INVALID"}
	}
	return g
}

// selfReport is the agent-authored claim planted in a gamed gen dir. Only these
// fields are read back for the captured claimed-vs-recomputed diff; the rest of
// the planted JSON is the agent's noise.
type selfReport struct {
	Verdict       string          `json:"verdict"`
	AdvisoryScore float64         `json:"advisory_score"`
	EvidenceState map[string]bool `json:"evidence_state"`
}

// readSelfReport loads an agent-authored self-report from path, or nil if absent
// or malformed — an honest prototype plants none. This is the only place the
// agent's claim enters the capture; it is shown, never trusted.
func readSelfReport(path string) *selfReport {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var sr selfReport
	if err := json.Unmarshal(data, &sr); err != nil {
		return nil
	}
	return &sr
}

// writeCapture renders the captured run as a self-contained demo artifact: the
// climbing-score series and, for each prototype, the evidence_state booleans
// flipping false->true generation over generation with the evaluator's
// recompute reason. Every boolean shown was recomputed by the evaluator, never
// read from the agent's self-report.
//
// path "-" (or "") prints the Markdown to stdout. Otherwise both a Markdown
// report and a machine-readable JSON sibling are written: the path's extension
// selects the primary file, and the other format is written alongside it (so
// "capture.md" also yields "capture.json", and vice versa). The JSON mirrors
// the Markdown so a chart or deck builder can consume the same numbers.
func writeCapture(path string, protos []captureProto, maxGen int) error {
	if path == "-" || path == "" {
		var b strings.Builder
		writeCaptureMarkdown(&b, protos, maxGen)
		fmt.Print(b.String())
		return nil
	}

	mdPath, jsonPath := captureSiblings(path)

	var b strings.Builder
	writeCaptureMarkdown(&b, protos, maxGen)
	if err := os.WriteFile(mdPath, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write capture %s: %w", mdPath, err)
	}

	data, err := json.MarshalIndent(captureJSON(protos, maxGen), "", "  ")
	if err != nil {
		return fmt.Errorf("marshal capture json: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(jsonPath, data, 0o644); err != nil {
		return fmt.Errorf("write capture %s: %w", jsonPath, err)
	}

	if anyGamed(protos) {
		var gb strings.Builder
		writeGamedCallout(&gb, protos)
		gamedPath := strings.TrimSuffix(mdPath, filepath.Ext(mdPath)) + ".gamed.md"
		if err := os.WriteFile(gamedPath, []byte(gb.String()), 0o644); err != nil {
			return fmt.Errorf("write capture %s: %w", gamedPath, err)
		}
		fmt.Printf("gamed-attempt capture written to %s\n", gamedPath)
	}

	fmt.Printf("\ncapture written to %s and %s\n", mdPath, jsonPath)
	return nil
}

// captureSiblings returns the markdown and json output paths for a -capture
// path, deriving the missing-format sibling from the given path's stem.
func captureSiblings(path string) (md, jsonPath string) {
	ext := strings.ToLower(filepath.Ext(path))
	stem := strings.TrimSuffix(path, filepath.Ext(path))
	switch ext {
	case ".json":
		return stem + ".md", path
	case ".md":
		return path, stem + ".json"
	default:
		return path + ".md", path + ".json"
	}
}

func writeCaptureMarkdown(b *strings.Builder, protos []captureProto, maxGen int) {
	fmt.Fprintf(b, "# P1 paperbench — paper-reproduction scored by honest recompute\n\n")
	fmt.Fprintf(b, "**Claim:** the `PASS`/`REVISE` verdict is gated categorically on seven evidence booleans the **real** ")
	fmt.Fprintf(b, "`PaperEvaluator` *independently recomputes* from each generation's artifacts — running the frozen validator, ")
	fmt.Fprintf(b, "re-hashing artifacts, parsing fixtures — never reading the agent's self-reported state. The advisory score is ")
	fmt.Fprintf(b, "a weighted **proxy** that climbs as booleans clear; it is *not* the pass criterion (a gamed run below climbs the ")
	fmt.Fprintf(b, "score yet stays REVISE).\n\n")
	fmt.Fprintf(b, "**What is real vs simulated:** the evaluator is production code; only the agent and target are simulated. ")
	fmt.Fprintf(b, "The SIA meta/feedback learning loop is a **no-op stub** in this demo — there is no agent under optimization ")
	fmt.Fprintf(b, "pressure. The agent writes a progressively more complete artifact ladder (scripted, one new artifact per gen) ")
	fmt.Fprintf(b, "and the evaluator independently verifies each addition actually satisfies the check, holding the boolean false ")
	fmt.Fprintf(b, "if it does not. What is demonstrated is the evaluator's honest recompute, not an agent learning to fix blockers.\n\n")
	fmt.Fprintf(b, "The gamed prototype's `%s` is agent-authored fixture data the demo plants on purpose — ", agentSelfReportName)
	fmt.Fprintf(b, "not the output of a real optimization loop — read back only to contrast with the evaluator's recompute.\n\n")
	fmt.Fprintf(b, "**Adversarial control:** one prototype (`%s`) tries to cheat the gate four ways every generation — ", gamedProtoID)
	fmt.Fprintf(b, "including a self-report claiming all seven keys true; the evaluator caught every one (section below).\n\n")

	// Headline climbing-score series across all prototypes.
	fmt.Fprintf(b, "## Climbing series — verdict is categorical and tier-gated; advisory score is a proxy\n\n")
	fmt.Fprintf(b, "Same recompute machinery across contrasting status tiers (not a self-improvement-generalizes claim — no learning loop runs).\n\n")
	fmt.Fprintf(b, "| prototype | tier | %s |\n", genHeader(maxGen))
	fmt.Fprintf(b, "|---|---|%s\n", strings.Repeat("---|", maxGen))
	for _, p := range protos {
		var cells []string
		for _, g := range p.gens {
			cells = append(cells, fmt.Sprintf("%s %.2f", g.Verdict, g.AdvisoryScore))
		}
		fmt.Fprintf(b, "| `%s` | %s | %s |\n", p.id, p.status, strings.Join(cells, " | "))
	}
	fmt.Fprintf(b, "\n")

	// Per-prototype evidence_state flip table + recompute reasons.
	for _, p := range protos {
		writeProtoSection(b, p, maxGen)
	}

	writeGamedCallout(b, protos)

	fmt.Fprintf(b, "## Why recompute is structural, not a prose claim\n\n")
	fmt.Fprintf(b, "A reason string proves nothing on its own — what makes each boolean a *recompute* is that the evaluator owns ")
	fmt.Fprintf(b, "the code path a judge can re-run:\n\n")
	fmt.Fprintf(b, "- **The validator is frozen and checksummed.** `validator_command` runs the rubric-owned validator (verified ")
	fmt.Fprintf(b, "against a pinned sha256 manifest at the start of every `Evaluate`), never a script from the agent's gen dir. ")
	fmt.Fprintf(b, "It must accept a frozen positive (exit 0) **and** reject a frozen negative and a tampered positive (exit 1..125). ")
	fmt.Fprintf(b, "An agent that only self-reports — writes the booleans plus an `exit 0` stub — cannot touch this gate.\n")
	fmt.Fprintf(b, "- **Hashing is evaluator-side.** `artifact_manifest_hash` re-hashes the named gen-dir artifact and compares to the ")
	fmt.Fprintf(b, "manifest's claimed sha; a fabricated hash, a missing file, or a byte-size lie all fail.\n")
	fmt.Fprintf(b, "- **Falsifiers must be validator-rejected.** `falsifier_rows` requires a row the frozen validator rejects as a ")
	fmt.Fprintf(b, "claim near-miss; a relabeled passing row is validator-*accepted* and does not count.\n")
	fmt.Fprintf(b, "- **Re-run to verify.** Every number here is read from each `gen_N/results.json`; point the evaluator at the same ")
	fmt.Fprintf(b, "gen dir and the booleans recompute identically. The negative-control behavior is pinned by ")
	fmt.Fprintf(b, "`TestValidatorCommand_RejectsExitZeroStub`, `TestPaperEvaluator_IgnoresSelfReportedEvidence`, and ")
	fmt.Fprintf(b, "`TestManifestHash_GamingVectors` in `paper_eval_test.go`.\n")
}

// writeGamedCallout renders the adversarial-control exhibit: the gamed
// prototype's planted all-true self-report next to the evaluator's honest
// recompute of the SAME gen dir, a per-trick "fake -> why it failed" list, and a
// one-line stage frame. Every mark, count, verdict, and reason is read from a
// real run (the planted self-report and results.json), never asserted. It is a
// no-op when the run contained no gamed prototype.
func writeGamedCallout(b *strings.Builder, protos []captureProto) {
	for _, p := range protos {
		if !isGamedProto(p) || len(p.gens) == 0 {
			continue
		}
		g := p.gens[0] // gamed prototype plants the same fakes every gen
		fmt.Fprintf(b, "## We tried to cheat our own gate — no trick survived the recompute\n\n")
		fmt.Fprintf(b, "*We tried to cheat our own gate four ways — including a self-report claiming all seven keys PASS; ")
		fmt.Fprintf(b, "the evaluator recomputed from ground truth and not one trick survived.*\n\n")
		fmt.Fprintf(b, "`%s` ships four named fabrications every generation. The evaluator reads none of them; ", p.id)
		fmt.Fprintf(b, "it recomputes each boolean from ground truth in the gen dir. The agent even planted ")
		fmt.Fprintf(b, "`%s` claiming **all seven keys true** with a fabricated `verdict: PASS`. ", agentSelfReportName)
		fmt.Fprintf(b, "Below, the claim sits beside the honest recompute, both read from the same gen dir:\n\n")

		fmt.Fprintf(b, "| evidence key | agent claimed | evaluator recomputed | recompute reason |\n")
		fmt.Fprintf(b, "|---|:--:|:--:|---|\n")
		for _, key := range evidenceKeys {
			fmt.Fprintf(b, "| `%s` | %s | %s | %s |\n",
				key, tick(claimedKey(g.claimed, key)), tick(g.EvidenceState[key]), g.Detail[key])
		}
		fmt.Fprintf(b, "| **verdict** | **%s** (claimed) | **%s** (recomputed) | %s; held by open blockers %s |\n",
			claimedVerdict(g.claimed), g.Verdict, g.Detail["fast_check"], blockerList(g.Blockers))
		fmt.Fprintf(b, "| **advisory_score** | **%.2f** (claimed) | **%.2f** (recomputed) | weighted proxy, not the gate |\n\n",
			claimedScore(g.claimed), g.AdvisoryScore)

		claimed := countTrue(claimedState(g.claimed))
		recomp := countTrue(g.EvidenceState)
		fmt.Fprintf(b, "Agent claimed **%d/%d** keys true; the evaluator recomputed **%d/%d**. ",
			claimed, len(evidenceKeys), recomp, len(evidenceKeys))
		fmt.Fprintf(b, "The self-report changed nothing: verdict **%s**, with open blockers %s. ",
			g.Verdict, blockerList(g.Blockers))
		fmt.Fprintf(b, "`paper_eval.go`: *\"A self-reported evidence_state in the gen dir is never read.\"*\n\n")
		fmt.Fprintf(b, "The four recomputed-true keys are not leaked fakes: the agent shipped one genuine control row ")
		fmt.Fprintf(b, "(earning `fixture_row` + `control_rows`), and `validator_command` + `model_backed_or_opt_in_command` ")
		fmt.Fprintf(b, "pass their own frozen checks regardless of the agent. That real credit is exactly why the gamed score ")
		fmt.Fprintf(b, "reaches %.2f — and exactly why the score is a proxy, not the gate.\n\n", g.AdvisoryScore)

		fmt.Fprintf(b, "Four named tricks, each defeated by an evaluator-owned recompute:\n\n")
		for _, t := range gamedTricks { // mapping mirrors writeGamedArtifacts in main.go
			reason := g.Detail[t.key]
			switch t.key {
			case "validator_command":
				// The stub is never run; the recompute is the frozen validator's.
				reason = "stub never run — the evaluator runs the frozen validator instead (" + reason + ")"
			case "":
				// The self-report trick targets no single key — the verdict itself.
				reason = fmt.Sprintf("never read; recomputed %d/%d true, so the verdict stays %s", recomp, len(evidenceKeys), g.Verdict)
			}
			fmt.Fprintf(b, "%d. **%s** — %s — %s\n", t.n, t.fake, displayKey(t.key), reason)
		}
		fmt.Fprintf(b, "\nScore gamed up to %.2f, verdict held at %s: the cleanest proof the advisory number is **not** the pass criterion.\n\n",
			g.AdvisoryScore, g.Verdict)
		return
	}
}

// gamedTricks maps each fabrication writeGamedArtifacts plants to the evidence
// key it targets, so the capture can pair each fake with the recompute reason
// (read from results.json) that defeated it. Only the trick->key mapping and the
// human-readable fake descriptions are static; the reason text is always read
// from g.Detail. This list MUST mirror writeGamedArtifacts in main.go.
var gamedTricks = []struct {
	n    int
	fake string
	key  string
}{
	{1, "exit-0 validator stub (`validator.sh`)", "validator_command"},
	{2, "fabricated sha256 manifest over a non-existent `ghost.jsonl`", "artifact_manifest_hash"},
	{3, "a second validator-accepted row offered as a falsifier", "falsifier_rows"},
	{4, "self-reported `evidence_state` claiming all seven keys true", ""},
}

// tick renders a captured boolean as the capture's ✓ / · glyphs.
func tick(v bool) string {
	if v {
		return "✓"
	}
	return "·"
}

// countTrue counts true values in an evidence map (claimed or recomputed).
func countTrue(m map[string]bool) int {
	n := 0
	for _, v := range m {
		if v {
			n++
		}
	}
	return n
}

// displayKey labels the evidence key a trick targets; the self-report trick
// targets no single key but the verdict itself.
func displayKey(key string) string {
	if key == "" {
		return "targets the verdict, not one key"
	}
	return "targets `" + key + "`"
}

// blockerList renders the captured open-blocker keys, or "none" if empty.
func blockerList(blockers []string) string {
	if len(blockers) == 0 {
		return "none"
	}
	var qs []string
	for _, b := range blockers {
		qs = append(qs, "`"+b+"`")
	}
	return strings.Join(qs, ", ")
}

// claimedState returns the agent's claimed evidence map, or nil for an honest
// prototype (which plants no self-report). Indexing a nil map is false, so
// callers need no extra guard.
func claimedState(c *selfReport) map[string]bool {
	if c == nil {
		return nil
	}
	return c.EvidenceState
}

func claimedKey(c *selfReport, key string) bool { return claimedState(c)[key] }

func claimedVerdict(c *selfReport) string {
	if c == nil {
		return "—"
	}
	return c.Verdict
}

func claimedScore(c *selfReport) float64 {
	if c == nil {
		return 0
	}
	return c.AdvisoryScore
}

// gamedProtoID is the id of the adversarial-control prototype the gamed-attempt
// callout describes.
const gamedProtoID = "gamed-attempt"

// agentSelfReportName is the agent-authored, deliberately-untrusted self-report
// the gamed prototype plants in its gen dir. It is NOT NameResultsJSON
// ("results.json"): the evaluator writes results.json itself and would clobber a
// plant by that name. The evaluator never reads this file; the capture reads it
// only to show the claim the honest recompute overrules.
const agentSelfReportName = "agent-self-report.json"

// isGamedProto reports whether p is the adversarial-control prototype.
func isGamedProto(p captureProto) bool {
	return p.id == gamedProtoID
}

// anyGamed reports whether protos contains the adversarial-control prototype.
func anyGamed(protos []captureProto) bool {
	for _, p := range protos {
		if isGamedProto(p) {
			return true
		}
	}
	return false
}

// writeProtoSection renders one prototype's per-generation evidence table.
func writeProtoSection(b *strings.Builder, p captureProto, maxGen int) {
	fmt.Fprintf(b, "## %s (tier: %s)\n\n", p.id, p.status)
	fmt.Fprintf(b, "Evidence booleans, recomputed each generation (✓ = evaluator verified true, · = false):\n\n")
	fmt.Fprintf(b, "| evidence key | %s |\n", genHeader(maxGen))
	fmt.Fprintf(b, "|---|%s\n", strings.Repeat("---|", maxGen))

	for _, key := range evidenceKeys {
		var cells []string
		prev := false
		for i, g := range p.gens {
			cur := g.EvidenceState[key]
			mark := "·"
			if cur {
				mark = "✓"
			}
			// Flag the generation where this key first flips to true.
			if cur && (i == 0 && cur || (i > 0 && !prev)) {
				mark = "✓ (flip)"
			}
			cells = append(cells, mark)
			prev = cur
		}
		fmt.Fprintf(b, "| `%s` | %s |\n", key, strings.Join(cells, " | "))
	}

	// Verdict / score / blockers footer row.
	var vcells, scells, bcells []string
	for _, g := range p.gens {
		vcells = append(vcells, g.Verdict)
		scells = append(scells, fmt.Sprintf("%.2f", g.AdvisoryScore))
		if len(g.Blockers) == 0 {
			bcells = append(bcells, "—")
		} else {
			bcells = append(bcells, fmt.Sprintf("%d", len(g.Blockers)))
		}
	}
	fmt.Fprintf(b, "| **verdict** | %s |\n", strings.Join(vcells, " | "))
	fmt.Fprintf(b, "| **advisory_score** | %s |\n", strings.Join(scells, " | "))
	fmt.Fprintf(b, "| **open blockers** | %s |\n\n", strings.Join(bcells, " | "))

	// Recompute reasons for the final (PASS) generation: proof the booleans
	// were derived, not asserted.
	if last := p.gens[len(p.gens)-1]; len(last.Detail) > 0 {
		fmt.Fprintf(b, "Final-generation recompute trace (proof of independent verification):\n\n")
		keys := make([]string, 0, len(last.Detail))
		for k := range last.Detail {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(b, "- `%s`: %s\n", k, last.Detail[k])
		}
		fmt.Fprintf(b, "\n")
	}
}

// genHeader returns "gen 1 | gen 2 | ..." sized to maxGen.
func genHeader(maxGen int) string {
	var h []string
	for g := 1; g <= maxGen; g++ {
		h = append(h, fmt.Sprintf("gen %d", g))
	}
	return strings.Join(h, " | ")
}

// reportCapture prints the terminal-style per-generation series for one
// prototype, matching the house style of cmd/inferopt/report.go: a compact
// table read straight from each generation's results.json.
func reportCapture(layout sia.RunLayout, p captureProto, maxGen int) {
	fmt.Printf("  %-4s %-7s %-9s %-7s %s\n", "gen", "verdict", "advisory", "blkrs", "newly-cleared evidence (honest recompute)")
	var prev map[string]bool
	for i, g := range p.gens {
		cleared := newlyCleared(prev, g.EvidenceState)
		fmt.Printf("  %-4d %-7s %-9.2f %-7d %s\n",
			i+1, g.Verdict, g.AdvisoryScore, len(g.Blockers), strings.Join(cleared, ", "))
		prev = g.EvidenceState
	}
}

// newlyCleared returns the evidence keys that flipped false->true between prev
// and cur, in weight order.
func newlyCleared(prev, cur map[string]bool) []string {
	var out []string
	for _, k := range evidenceKeys {
		if cur[k] && (prev == nil || !prev[k]) {
			out = append(out, k)
		}
	}
	return out
}

// captureReport is the machine-readable JSON capture: a per-prototype climbing
// series with each generation's verdict, advisory score, the honestly
// recomputed evidence booleans, the keys newly cleared that generation, and the
// recompute trace. It deliberately uses the same vocabulary as results.json and
// siachart (verdict REVISE/PASS) so a deck or chart builder can consume it.
type captureReport struct {
	Thesis      string             `json:"thesis"`
	EvidenceKey []string           `json:"evidence_keys"`
	Prototypes  []capturePrototype `json:"prototypes"`
}

type capturePrototype struct {
	ID     string           `json:"id"`
	Status string           `json:"status"`
	Gens   []captureGenJSON `json:"generations"`
}

type captureGenJSON struct {
	Gen           int               `json:"gen"`
	Verdict       string            `json:"verdict"`
	AdvisoryScore float64           `json:"advisory_score"`
	FastCheckOK   bool              `json:"fast_check_ok"`
	EvidenceState map[string]bool   `json:"evidence_state"`
	NewlyCleared  []string          `json:"newly_cleared"`
	OpenBlockers  []string          `json:"open_blockers"`
	Detail        map[string]string `json:"evidence_detail"`

	// Agent self-report, shown for contrast and never trusted. Present only for
	// the gamed prototype; omitted for honest prototypes (which plant none).
	ClaimedState   map[string]bool `json:"claimed_state,omitempty"`
	ClaimedVerdict string          `json:"claimed_verdict,omitempty"`
	ClaimedScore   float64         `json:"claimed_score,omitempty"`
}

// captureJSON builds the machine-readable capture from the per-prototype runs.
func captureJSON(protos []captureProto, maxGen int) captureReport {
	rep := captureReport{
		Thesis: "Paper-reproduction quality scored as a game-resistant SIA loop: each evidence boolean is " +
			"independently recomputed by the evaluator (frozen validator, artifact hashing, fixture parsing), " +
			"never trusted from the agent's self-report; the verdict climbs REVISE->PASS as blockers honestly clear.",
		EvidenceKey: evidenceKeys,
	}
	for _, p := range protos {
		cp := capturePrototype{ID: p.id, Status: p.status}
		var prev map[string]bool
		for i, g := range p.gens {
			blockers := g.Blockers
			if blockers == nil {
				blockers = []string{}
			}
			gj := captureGenJSON{
				Gen:           i + 1,
				Verdict:       g.Verdict,
				AdvisoryScore: g.AdvisoryScore,
				FastCheckOK:   g.FastCheckOK,
				EvidenceState: g.EvidenceState,
				NewlyCleared:  newlyCleared(prev, g.EvidenceState),
				OpenBlockers:  blockers,
				Detail:        g.Detail,
			}
			// Only the gamed prototype plants a self-report; honest gens stay clean.
			if g.claimed != nil {
				gj.ClaimedState = g.claimed.EvidenceState
				gj.ClaimedVerdict = g.claimed.Verdict
				gj.ClaimedScore = g.claimed.AdvisoryScore
			}
			cp.Gens = append(cp.Gens, gj)
			prev = g.EvidenceState
		}
		rep.Prototypes = append(rep.Prototypes, cp)
	}
	return rep
}
