// Command paperbench runs the SIA self-improvement loop with a [sia.PaperEvaluator]:
// each generation, a simulated agent extends its research-prototype implementation
// and the evaluator honest-recomputes the paper-roadmap coverage-map rubric from
// the generation's artifacts, so the verdict climbs from REVISE to PASS as the
// agent closes the named blockers.
//
// The demo is fully self-contained and offline: it scaffolds a minimal task tree,
// a frozen evaluator-owned rubric, and a meta/feedback/target engine that needs no
// network or model. It runs over N>=3 contrasting prototypes (mirroring the SIA
// paper's "generalizes across three domains" claim) and prints a per-generation
// verdict and advisory score — the climbing-score curve that is the demo's payoff.
//
// Usage:
//
//	paperbench                 # run the built-in 3-prototype demo
//	paperbench -max-gen 4       # more generations per prototype
//	paperbench -runs-root /tmp/pb   # where run_* dirs are written
//
// The evaluator is the real one from package sia; only the agent and target are
// simulated, so what climbs is a genuine honest recompute, not a scripted number.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	sia "github.com/tmc/mlx-go-sia"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("paperbench: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	fs := flag.NewFlagSet("paperbench", flag.ContinueOnError)
	var (
		maxGen   = fs.Int("max-gen", 4, "self-improvement generations per prototype")
		runsRoot = fs.String("runs-root", "", "directory runs are written under (default: a temp dir)")
		capture  = fs.String("capture", "", "write a Markdown demo capture to this path (\"-\" for stdout) showing the per-generation evidence_state booleans flipping")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}
	if *maxGen < 1 {
		return fmt.Errorf("-max-gen must be >= 1")
	}

	root := *runsRoot
	if root == "" {
		tmp, err := os.MkdirTemp("", "paperbench-")
		if err != nil {
			return err
		}
		root = tmp
		fmt.Printf("writing runs under %s\n", root)
	}

	// One frozen rubric is shared by all prototypes in this demo; in production
	// each coverage-map row would carry its own validator + schema.
	rubricDir := filepath.Join(root, "_rubric")
	if err := scaffoldRubric(rubricDir); err != nil {
		return fmt.Errorf("scaffold rubric: %w", err)
	}
	rubric, err := sia.LoadRubric(rubricDir)
	if err != nil {
		return fmt.Errorf("load rubric: %w", err)
	}

	// Three contrasting prototypes, each with real headroom in its evidence
	// state. The demo simulates the agent closing the blockers generation over
	// generation; the evaluator recomputes the truth. The fourth prototype is an
	// adversarial control: it submits the four named fakes (an exit-0 validator
	// stub, a fabricated manifest hash, a relabeled control row passed off as a
	// falsifier, and a self-reported evidence_state claiming all seven keys true)
	// every generation, so the captured run shows the real evaluator holding the
	// targeted booleans false and the verdict at REVISE even as the advisory
	// score is gamed upward.
	prototypes := []prototype{
		{id: "dflash-ddtree", status: "lightweight"},
		{id: "eagle3-vocab-translation", status: "lightweight"},
		{id: "gnosis-trace-compression", status: "covered"},
		{id: gamedProtoID, status: "covered", gamed: true},
	}

	ctx := context.Background()
	var captures []captureProto
	for i, p := range prototypes {
		fmt.Printf("\n=== prototype %d/%d: %s (tier %s) ===\n", i+1, len(prototypes), p.id, p.status)
		cap, err := runPrototype(ctx, root, i+1, p, rubric, *maxGen)
		if err != nil {
			return fmt.Errorf("prototype %s: %w", p.id, err)
		}
		captures = append(captures, cap)
	}
	fmt.Printf("\ndone: scored %d prototype(s) over %d generation(s) each.\n", len(prototypes), *maxGen)

	if *capture != "" {
		if err := writeCapture(*capture, captures, *maxGen); err != nil {
			return err
		}
	}
	return nil
}

// prototype names one demo target and its coverage-map status tier. A gamed
// prototype submits fabricated evidence every generation instead of honestly
// closing blockers, so the evaluator's rejection of the fakes is captured.
type prototype struct {
	id     string
	status string
	gamed  bool
}

// runPrototype runs the SIA loop for one prototype: the simulated target writes
// progressively more complete artifacts each generation, and the real
// PaperEvaluator scores them. It prints the per-generation verdict + score.
func runPrototype(ctx context.Context, runsRoot string, runID int, p prototype, rubric sia.Rubric, maxGen int) (captureProto, error) {
	cap := captureProto{id: p.id, status: p.status}
	taskRoot := filepath.Join(runsRoot, "tasks", p.id)
	task, err := scaffoldTask(taskRoot, p.id)
	if err != nil {
		return cap, err
	}
	resolved, err := sia.DefaultAgentReference.Resolve(task)
	if err != nil {
		return cap, err
	}
	taskFiles, err := sia.LoadTaskFiles(task, resolved)
	if err != nil {
		return cap, err
	}
	layout, err := sia.SetupRunDirectory(filepath.Join(runsRoot, "runs", p.id), runID)
	if err != nil {
		return cap, err
	}

	// The meta/feedback engine is a no-op: the orchestrator only needs it to
	// "produce" a target agent; our target executor does the real work below.
	engine := sia.FuncRunner{ImplName: "demo", Fn: func(context.Context, sia.AgentRequest) error { return nil }}

	// The simulated agent improves across generations: gen 1 ships an empty
	// fixture (REVISE), and each later gen adds the artifacts that flip another
	// blocker, until it reaches a PASS. WorkingDir is the gen dir the evaluator
	// scores, so writing here is exactly what an honest agent would do. A gamed
	// prototype instead submits fabricated evidence every generation.
	target := sia.FuncTargetExecutor(func(_ context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
		gen := genOf(layout, req.WorkingDir, maxGen)
		if p.gamed {
			writeGamedArtifacts(req.WorkingDir)
		} else {
			writeGenArtifacts(req.WorkingDir, gen)
		}
		return sia.TargetResult{Success: true, Stdout: fmt.Sprintf("generation %d artifacts written", gen)}, nil
	})

	eval := &sia.PaperEvaluator{
		Row: sia.CoverageRow{
			ID:        p.id,
			Status:    p.status,
			FastCheck: "true", // offline demo: stand in for the row's real fast_check
			Examples:  []string{"fixtures/" + p.id + ".jsonl"},
		},
		RepoRoot: taskRoot,
		Rubric:   rubric,
	}

	orch := &sia.Orchestrator{
		Meta:   engine,
		Target: target,
		Eval:   eval,
		Logf:   func(string, ...any) {}, // quiet; we print our own summary
	}

	if _, err := orch.Run(ctx, sia.RunOptions{
		Layout:      layout,
		Task:        task,
		TaskFiles:   taskFiles,
		MetaProfile: demoMetaProfile(),
		Target:      demoTargetProfile(),
		Resolved:    resolved,
		MaxGen:      maxGen,
		Focus:       sia.FocusHarness,
	}); err != nil {
		return cap, err
	}

	// Collect each generation's honestly-recomputed results.json and, for a gamed
	// prototype, the agent's planted self-report from the SAME gen dir, so the
	// capture's claimed-vs-recomputed diff is two reads from one directory.
	for gen := 1; gen <= maxGen; gen++ {
		gc := readGenCapture(layout.ResultsJSON(gen))
		gc.claimed = readSelfReport(filepath.Join(layout.GenDir(gen), agentSelfReportName))
		cap.gens = append(cap.gens, gc)
	}
	reportCapture(layout, cap, maxGen)
	return cap, nil
}

// genOf infers the 1-based generation number from the working dir path.
func genOf(layout sia.RunLayout, workingDir string, maxGen int) int {
	for gen := 1; gen <= maxGen; gen++ {
		if filepath.Clean(layout.GenDir(gen)) == filepath.Clean(workingDir) {
			return gen
		}
	}
	return 1
}

// writeGenArtifacts simulates the agent's generation-n implementation: each
// generation adds the artifacts that honestly satisfy one more evidence key.
//
//	gen 1: empty fixture           -> fixture_row false, REVISE
//	gen 2: valid control fixture   -> fixture_row + control_rows true
//	gen 3: + a claim falsifier row -> falsifier_rows true
//	gen 4: + evidence manifest     -> artifact_manifest_hash true (PASS for covered tier)
func writeGenArtifacts(dir string, gen int) {
	fixtures := filepath.Join(dir, "fixtures")
	_ = os.MkdirAll(fixtures, 0o755)
	// Resolve the prototype id from the fixtures we are asked to write: the
	// evaluator looks for fixtures/<id>.jsonl, so name it from the run path.
	id := protoIDFromGenDir(dir)
	fixturePath := filepath.Join(fixtures, id+".jsonl")

	var lines []string
	if gen >= 2 {
		lines = append(lines, demoRow("control-1", "gpu", true))
	}
	if gen >= 3 {
		lines = append(lines, demoRow("control-1", "gpu", false)) // claim near-miss
	}
	if gen >= 1 {
		writeLines(fixturePath, lines) // gen 1 writes an empty fixture
	}

	if gen >= 4 {
		// A hash-verified artifact + manifest flips artifact_manifest_hash.
		artPath := filepath.Join(dir, "artifact.jsonl")
		writeLines(artPath, []string{demoRow("art", "gpu", true)})
		writeManifest(dir, "artifact.jsonl")
		// Cover the frozen scope with the same gpu artifact (heavy_skip).
		writeFile(filepath.Join(dir, "scope-manifest.json"),
			`{"covered":["gpu_overlap"],"links":{"gpu_overlap":"a-gpu"}}`)
	}
}

// writeGamedArtifacts simulates an agent gaming the rubric: it submits the
// four fabrications the spec's threat model names, all at once, every
// generation. None of them survives the evaluator's honest recompute:
//
//   - an exit-0 validator stub: ignored — the evaluator runs the FROZEN,
//     checksummed rubric validator, never a script from the gen dir, so
//     validator_command is decided by the real validator regardless.
//   - a fabricated manifest hash pointing at no real artifact: re-hashing finds
//     no match, so artifact_manifest_hash stays false.
//   - a validator-accepted row relabeled as a "falsifier": the frozen validator
//     accepts it, so it is not a claim near-miss and falsifier_rows stays false.
//   - a self-reported evidence_state claiming all seven keys true: the evaluator
//     never reads an agent-authored evidence_state, so the claim is overruled
//     key by key by recompute.
//
// The agent does ship a real, validator-accepted control row, so fixture_row
// and control_rows do go true and the advisory score is gamed upward — but the
// covered-tier verdict stays REVISE because the high-weight booleans the fakes
// targeted are held false by recompute. Score up, verdict not: the cleanest
// proof that the advisory number is not the pass criterion.
func writeGamedArtifacts(dir string) {
	id := protoIDFromGenDir(dir)
	fixtures := filepath.Join(dir, "fixtures")
	_ = os.MkdirAll(fixtures, 0o755)

	// A genuine, validator-accepted control row (earns fixture_row + control_rows)
	// plus a SECOND row that is also validator-accepted but relabeled as a
	// falsifier — there is no real validator-rejected near-miss, so falsifier_rows
	// will not flip.
	writeLines(filepath.Join(fixtures, id+".jsonl"), []string{
		demoRow("control-1", "gpu", true),
		demoRow("control-2", "gpu", true), // relabeled "falsifier" — validator still accepts it
	})

	// An exit-0 validator stub the agent hopes the evaluator will run. It will
	// not: validator_command uses the frozen rubric validator.
	writeFile(filepath.Join(dir, "validator.sh"), "#!/bin/sh\nexit 0\n")

	// A manifest claiming a fabricated sha256 for an artifact that does not exist
	// in the gen dir; re-hashing finds no match.
	fakeHash := "sha256:" + strings.Repeat("ab", 32)
	writeFile(filepath.Join(dir, "evidence-manifest.json"), fmt.Sprintf(
		`{"schema_version":"mlx_go_evidence_manifest.v0","manifest_id":"gamed","run_id":"gamed",`+
			`"artifacts":[{"artifact_id":"ghost","artifact_kind":"fixture","path":"ghost.jsonl",`+
			`"sha256":%q,"byte_size":123,"source_gaps":[]}],"source_gaps":[]}`, fakeHash))

	// The fourth named trick: a self-reported evidence_state claiming every key
	// true, with a fabricated PASS/1.00. The evaluator never reads it
	// (paper_eval.go: "A self-reported evidence_state in the gen dir is never
	// read"); we plant it under an agent-owned name — never results.json, which
	// the evaluator writes itself — so the capture can show the claim beside the
	// honest recompute that overrules it.
	selfReport := map[string]any{
		"_comment":       "AGENT-AUTHORED CLAIM — not read by the evaluator; planted to prove the recompute ignores it.",
		"verdict":        "PASS",
		"advisory_score": 1.0,
		"evidence_state": map[string]bool{
			"validator_command":              true,
			"artifact_manifest_hash":         true,
			"model_backed_or_opt_in_command": true,
			"falsifier_rows":                 true,
			"control_rows":                   true,
			"fixture_row":                    true,
			"heavy_skip_narrowed_or_cleared": true,
		},
	}
	if data, err := json.MarshalIndent(selfReport, "", "  "); err == nil {
		writeFileBytes(filepath.Join(dir, agentSelfReportName), append(data, '\n'))
	}
}

// protoIDFromGenDir extracts the prototype id from a .../runs/<id>/run_N/gen_M path.
func protoIDFromGenDir(genDir string) string {
	// genDir = <runsRoot>/runs/<id>/run_N/gen_M
	d := filepath.Dir(filepath.Dir(genDir)) // .../runs/<id>
	return filepath.Base(d)
}

const demoSchemaTag = "paperbench-demo-row/v1"

func demoRow(runID, device string, accepted bool) string {
	digest := sha256.Sum256([]byte(runID + device))
	return fmt.Sprintf(`{"schema":%q,"run_id":%q,"device":%q,"accepted":%t,"digest":%q}`,
		demoSchemaTag, runID, device, accepted, "sha256:"+hex.EncodeToString(digest[:]))
}

func writeLines(path string, lines []string) {
	var data []byte
	for _, l := range lines {
		data = append(data, l...)
		data = append(data, '\n')
	}
	writeFileBytes(path, data)
}

func writeManifest(genDir, rel string) {
	data, err := os.ReadFile(filepath.Join(genDir, rel))
	if err != nil {
		return
	}
	sum := sha256.Sum256(data)
	m := map[string]any{
		"schema_version": "mlx_go_evidence_manifest.v0",
		"manifest_id":    "demo-m",
		"run_id":         "demo-r",
		"artifacts": []map[string]any{{
			"artifact_id":   "a-gpu",
			"artifact_kind": "fixture",
			"path":          rel,
			"sha256":        "sha256:" + hex.EncodeToString(sum[:]),
			"byte_size":     len(data),
			"source_gaps":   []string{},
		}},
		"source_gaps": []string{},
	}
	b, _ := json.MarshalIndent(m, "", "  ")
	writeFileBytes(filepath.Join(genDir, "evidence-manifest.json"), b)
}

// --- demo scaffolding (offline, no network) ---

// demoMetaProfile / demoTargetProfile are minimal profiles for the offline loop;
// the FuncRunner engine never contacts a provider.
func demoMetaProfile() sia.MetaAgentProfile {
	return sia.MetaAgentProfile{ProfileID: "demo", Name: "demo", AgentImpl: "demo", Model: "demo"}
}

func demoTargetProfile() sia.TargetAgentProfile {
	return sia.TargetAgentProfile{ProfileID: "demo", Name: "demo", Model: "demo", AgentReference: sia.DefaultAgentReference}
}

// scaffoldTask writes the minimal task tree the orchestrator's prompt rendering
// requires: task.md, sample descriptions, a seed reference agent, and a shared
// sample execution.
func scaffoldTask(taskRoot, id string) (sia.TaskLayout, error) {
	taskDir := filepath.Join(taskRoot, "task")
	shared := filepath.Join(taskRoot, "_shared")
	files := map[string]string{
		filepath.Join(taskDir, "data", "public", "task.md"):                "# " + id + "\nImplement the prototype to satisfy its coverage-map rubric.\n",
		filepath.Join(taskDir, "reference", "SAMPLE_TASK_DESCRIPTIONS.md"): "Close the evidence blockers reported in results.json.\n",
		filepath.Join(taskDir, "reference", "reference_target_agent.py"):   "# seed prototype implementation\n",
		filepath.Join(shared, "sample_agent_execution.json"):               `[{"role":"user","content":"implement"}]`,
	}
	for p, content := range files {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return sia.TaskLayout{}, err
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			return sia.TaskLayout{}, err
		}
	}
	return sia.TaskLayout{TaskDir: taskDir, SharedDir: shared}, nil
}

func writeFile(path, content string) { writeFileBytes(path, []byte(content)) }
func writeFileBytes(path string, data []byte) {
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	_ = os.WriteFile(path, data, 0o644)
}
