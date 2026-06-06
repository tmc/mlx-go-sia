package sia

import (
	"os"
	"path/filepath"
	"testing"
)

// TestContextManagerGoldenFullRun verifies that the context.md the
// ContextManager produces is byte-identical to the reference's
// context_manager.py output (testdata/context/full_run.md, generated from the
// Python reference with the SDK-bound LLM summary neutralized). It exercises the
// gen-1 entry, gen-2 deltas + improvement.md insights, gen-3 failure +
// stdout-regex metrics, metric float/int rendering, and the closing summary with
// best-generation and code-growth statistics.
func TestContextManagerGoldenFullRun(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "run_1")

	// resultsJSON is written verbatim (not via json.Marshal) so the integer- vs
	// float-form numbers match the Python golden's results.json exactly: Go's
	// json.Marshal would render 40.0 as "40", erasing the float distinction the
	// reference (and this port) key off when rendering metrics.
	writeGen := func(n int, files map[string]string, resultsJSON string) string {
		gd := filepath.Join(runDir, "gen_"+itoa(n))
		if err := os.MkdirAll(gd, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, body := range files {
			writeTestFile(t, filepath.Join(gd, name), body)
		}
		if resultsJSON != "" {
			writeTestFile(t, filepath.Join(gd, NameResultsJSON), resultsJSON)
		}
		return gd
	}

	// Gen 1: 3-line agent, results.json with a float, a string, and an int
	// (plus a nested dict + a list, both of which must be skipped).
	writeGen(1,
		map[string]string{NameTargetAgent: "# gen1 agent\nimport os\nprint('hi')\n"},
		`{"accuracy": 40.0, "model": "qwen", "n_correct": 12, "nested": {"a": 1}, "items": [1, 2]}`)

	// Gen 2: 5-line agent, improvement.md mixing bullets/numbered (some filtered
	// by the len>20 / not-ending-in-colon rule), higher accuracy.
	writeGen(2,
		map[string]string{
			NameTargetAgent: "# gen2 improved agent\nimport os\nimport sys\nprint('hello world')\nx = 1\n",
			NameImprovementMD: "# Improvement Plan\n\n" +
				"- This is a sufficiently long meaningful bullet about retries and backoff\n" +
				"- short\n" +
				"- Header ending with colon:\n" +
				"1. Added a numbered improvement that is also long enough to count here\n" +
				"* Another long enough starred bullet describing tool selection logic changes\n\n" +
				"Some prose that should be ignored.\n",
		},
		`{"accuracy": 55.5, "model": "qwen", "n_correct": 18}`)

	// Gen 3: failure, no results.json — metrics come from the stdout regex.
	writeGen(3,
		map[string]string{
			NameTargetAgent: "# gen3 agent regressed\nprint('x')\n",
			NameStdoutLog:   "running...\nFinal accuracy: 48.5%\n3 / 10 correct\n",
		},
		"")

	layout := RunLayout{RunDir: runDir}
	cm := NewContextManager(layout, map[string]string{
		"task_dir":   "/tasks/mytask",
		"meta_model": "haiku",
		"task_model": "claude-haiku-4-5-20251001",
		"agent_impl": "claude",
		"max_gen":    "3",
		"started":    "2026-06-06 12:00:00",
	})
	if err := cm.Initialize(); err != nil {
		t.Fatal(err)
	}

	add := func(n int, success bool) {
		gd := filepath.Join(runDir, "gen_"+itoa(n))
		rec := GenerationRecord{
			GenNum:        n,
			Success:       success,
			Timestamp:     "2026-06-06 12:00:00",
			Duration:      1.5,
			AgentPath:     filepath.Join(gd, NameTargetAgent),
			GenDir:        gd,
			ExecutionType: "Single",
		}
		imp := filepath.Join(gd, NameImprovementMD)
		if isFile(imp) {
			rec.ImprovementPath = imp
		}
		if err := cm.AddGeneration(rec); err != nil {
			t.Fatal(err)
		}
	}
	add(1, true)
	add(2, true)
	add(3, false)
	if err := cm.Finalize(); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(layout.ContextMD())
	if err != nil {
		t.Fatal(err)
	}
	wantBytes, err := os.ReadFile(filepath.Join("testdata", "context", "full_run.md"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	want := string(wantBytes)
	if string(got) != want {
		t.Errorf("context.md does not match reference golden\n%s", firstDiff(string(got), want))
	}
}

// TestContextManagerGoldenNoAccuracy verifies the closing summary's no-metric
// edge case is byte-identical to the reference: with no accuracy anywhere the
// best generation is "N/A" and best_metric stays -inf, which Python renders as
// "-inf" (not Go's "-Inf"). The code-growth signs (+0) must match too.
func TestContextManagerGoldenNoAccuracy(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "run_1")
	gd := filepath.Join(runDir, "gen_1")
	if err := os.MkdirAll(gd, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(gd, NameTargetAgent), "# a\nprint(1)\n")

	layout := RunLayout{RunDir: runDir}
	cm := NewContextManager(layout, map[string]string{
		"task_dir":   "/t",
		"meta_model": "haiku",
		"task_model": "m",
		"agent_impl": "claude",
		"max_gen":    "1",
		"started":    "2026-06-06 12:00:00",
	})
	if err := cm.Initialize(); err != nil {
		t.Fatal(err)
	}
	if err := cm.AddGeneration(GenerationRecord{
		GenNum: 1, Success: true, Timestamp: "2026-06-06 12:00:00", Duration: 1.5,
		AgentPath: filepath.Join(gd, NameTargetAgent), GenDir: gd, ExecutionType: "Single",
	}); err != nil {
		t.Fatal(err)
	}
	if err := cm.Finalize(); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(layout.ContextMD())
	if err != nil {
		t.Fatal(err)
	}
	wantBytes, err := os.ReadFile(filepath.Join("testdata", "context", "no_accuracy.md"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if string(got) != string(wantBytes) {
		t.Errorf("context.md (no-accuracy) does not match reference golden\n%s", firstDiff(string(got), string(wantBytes)))
	}
}
