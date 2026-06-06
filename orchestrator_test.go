package sia

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// buildTestTask writes a minimal task tree under root and returns its layout.
func buildTestTask(t *testing.T, root string) TaskLayout {
	t.Helper()
	taskDir := filepath.Join(root, "mytask")
	writeTestFile(t, filepath.Join(taskDir, NameTaskMD), "# Task\nSolve it.\n")
	writeTestFile(t, filepath.Join(taskDir, NameSampleTaskDescriptions), "desc a\ndesc b")
	writeTestFile(t, filepath.Join(taskDir, NameReferenceAgent), "# seed agent\n")
	shared := filepath.Join(root, NameSharedDir)
	writeTestFile(t, filepath.Join(shared, NameSharedSampleExecution), `[{"role":"user","content":"hi"}]`)
	return TaskLayout{TaskDir: taskDir, SharedDir: shared}
}

// fixedClock returns a deterministic time for reproducible context.md output.
func fixedClock() func() time.Time {
	tm := time.Date(2026, 6, 6, 12, 0, 0, 0, time.UTC)
	return func() time.Time { return tm }
}

func TestOrchestratorFullLoop(t *testing.T) {
	root := t.TempDir()
	task := buildTestTask(t, root)
	ref := DefaultAgentReference
	resolved, err := ref.Resolve(task)
	if err != nil {
		t.Fatal(err)
	}
	taskFiles, err := LoadTaskFiles(task, resolved)
	if err != nil {
		t.Fatal(err)
	}

	runsRoot := filepath.Join(root, "runs")
	layout, err := SetupRunDirectory(runsRoot, 1)
	if err != nil {
		t.Fatal(err)
	}

	// The fake engine writes a target_agent.py on the meta call, then an
	// improvement.md + target_agent.py on each feedback call.
	metaStep := FakeStep{Files: map[string]string{NameTargetAgent: "# gen1 agent\n"}}
	fb := FakeStep{Files: map[string]string{
		NameTargetAgent:   "# improved agent\n",
		NameImprovementMD: "## what changed\n",
	}}
	engine := NewFakeRunner("fake", metaStep, fb, fb) // meta + 2 feedback (3 gens => 2 feedback)

	// The fake target writes a single-trajectory log and succeeds.
	var targetCalls int
	target := FuncTargetExecutor(func(_ context.Context, req TargetRequest) (TargetResult, error) {
		targetCalls++
		// Verify the orchestrator passes the fixed CLI contract.
		if !filepath.IsAbs(req.DatasetDir) {
			t.Errorf("dataset_dir not absolute: %q", req.DatasetDir)
		}
		writeTestFile(t, filepath.Join(req.WorkingDir, NameAgentExecution), `[{"role":"assistant","content":"done"}]`)
		writeTestFile(t, req.StdoutLog, "line1\nline2\n")
		return TargetResult{Success: true, Stdout: "line1\nline2\n"}, nil
	})

	orch := &Orchestrator{Meta: engine, Target: target, Now: fixedClock()}
	res, err := orch.Run(context.Background(), RunOptions{
		Layout:      layout,
		Task:        task,
		TaskFiles:   taskFiles,
		MetaProfile: mustMeta(t, "default-meta"),
		Target:      mustTarget(t, "default-target"),
		Resolved:    resolved,
		MaxGen:      3,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 3 generations recorded.
	if len(res.Generations) != 3 {
		t.Fatalf("generations = %d, want 3", len(res.Generations))
	}
	// Engine called once for meta + twice for feedback.
	if engine.Calls() != 3 {
		t.Errorf("engine calls = %d, want 3 (meta + 2 feedback)", engine.Calls())
	}
	// Target executed once per generation.
	if targetCalls != 3 {
		t.Errorf("target calls = %d, want 3", targetCalls)
	}

	// Each generation directory has the expected artifacts.
	for gen := 1; gen <= 3; gen++ {
		mustExist(t, layout.TargetAgent(gen))
		mustExist(t, layout.AgentExecutionJSON(gen))
	}
	// The meta prompt is saved in gen 1; feedback prompts in gens 2 and 3.
	mustExist(t, layout.MetaPrompt(1))
	mustExist(t, layout.FeedbackPrompt(2))
	mustExist(t, layout.FeedbackPrompt(3))
	mustExist(t, layout.ImprovementMD(2))

	// context.md exists and records all three generations + a final summary.
	ctxBytes, err := os.ReadFile(res.ContextPath)
	if err != nil {
		t.Fatal(err)
	}
	ctx := string(ctxBytes)
	// The format matches the reference context_manager.py: a run header, one
	// "## Generation N" block per generation with status/stats/metrics, and a
	// closing "## Summary Statistics" block.
	for _, want := range []string{
		"# Run Context:",
		"## Generation 1",
		"## Generation 2",
		"## Generation 3",
		"### Target Agent Changes",
		"Initial agent created by meta-agent",
		"### Performance Metrics",
		"## Summary Statistics",
		"**Total Generations**: 3",
		"**Successful Executions**: 3",
		"**Code Growth**:",
	} {
		if !strings.Contains(ctx, want) {
			t.Errorf("context.md missing %q", want)
		}
	}

	// The feedback prompt for gen 2 should embed gen 1's agent body and a
	// requirements note is absent (default reference ships no requirements.txt).
	fbPrompt, _ := os.ReadFile(layout.FeedbackPrompt(2))
	if !strings.Contains(string(fbPrompt), "# gen1 agent") {
		t.Error("feedback prompt should embed the previous generation's agent body")
	}
}

func TestOrchestratorRejectsExistingRun(t *testing.T) {
	root := t.TempDir()
	runsRoot := filepath.Join(root, "runs")
	if _, err := SetupRunDirectory(runsRoot, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := SetupRunDirectory(runsRoot, 1); err == nil {
		t.Error("second SetupRunDirectory with same run_id should error")
	}
}

func TestOrchestratorTargetFailureStillFeedsBack(t *testing.T) {
	root := t.TempDir()
	task := buildTestTask(t, root)
	resolved, _ := DefaultAgentReference.Resolve(task)
	taskFiles, _ := LoadTaskFiles(task, resolved)
	layout, _ := SetupRunDirectory(filepath.Join(root, "runs"), 1)

	engine := NewFakeRunner("fake",
		FakeStep{Files: map[string]string{NameTargetAgent: "# gen1\n"}},
		FakeStep{Files: map[string]string{NameTargetAgent: "# gen2\n", NameImprovementMD: "fix\n"}},
	)
	// Target always "fails" (non-zero exit semantics).
	target := FuncTargetExecutor(func(_ context.Context, req TargetRequest) (TargetResult, error) {
		writeTestFile(t, req.StdoutLog, "boom\n")
		return TargetResult{Success: false, ErrorMsg: "Target agent failed with exit code 1", Stdout: "boom\n"}, nil
	})

	orch := &Orchestrator{Meta: engine, Target: target, Now: fixedClock()}
	res, err := orch.Run(context.Background(), RunOptions{
		Layout: layout, Task: task, TaskFiles: taskFiles,
		MetaProfile: mustMeta(t, "default-meta"), Target: mustTarget(t, "default-target"),
		Resolved: resolved, MaxGen: 2,
	})
	if err != nil {
		t.Fatalf("Run should not error on target failure: %v", err)
	}
	// The feedback agent still ran (engine called meta + 1 feedback).
	if engine.Calls() != 2 {
		t.Errorf("engine calls = %d, want 2 despite target failure", engine.Calls())
	}
	if res.Generations[0].Target.Success {
		t.Error("gen 1 should be recorded as failed")
	}
	// The feedback prompt reflects the FAILED status.
	fb, _ := os.ReadFile(layout.FeedbackPrompt(2))
	if !strings.Contains(string(fb), "FAILED:") {
		t.Error("feedback prompt should reflect FAILED execution status")
	}
}

func TestOrchestratorWeightsEarlyStop(t *testing.T) {
	root := t.TempDir()
	task := buildTestTask(t, root)
	resolved, _ := DefaultAgentReference.Resolve(task)
	taskFiles, _ := LoadTaskFiles(task, resolved)
	layout, _ := SetupRunDirectory(filepath.Join(root, "runs"), 1)

	// Feedback writes train.py + a COMPLETED marker in the next gen dir to
	// signal early stopping.
	engine := NewFakeRunner("fake",
		FakeStep{Files: map[string]string{NameTrainScript: "# train gen1\n"}},
		FakeStep{Files: map[string]string{
			NameTrainScript:     "# train gen2\n",
			NameImprovementMD:   "rl fix\n",
			NameCompletedMarker: "done\n",
		}},
	)
	target := FuncTargetExecutor(func(_ context.Context, req TargetRequest) (TargetResult, error) {
		writeTestFile(t, filepath.Join(req.WorkingDir, NameAgentExecution), `[]`)
		writeTestFile(t, req.StdoutLog, "ok\n")
		return TargetResult{Success: true, Stdout: "ok\n"}, nil
	})

	orch := &Orchestrator{Meta: engine, Target: target, Now: fixedClock()}
	res, err := orch.Run(context.Background(), RunOptions{
		Layout: layout, Task: task, TaskFiles: taskFiles,
		MetaProfile: mustMeta(t, "default-meta"), Target: mustTarget(t, "default-target"),
		Resolved: resolved, MaxGen: 5, Focus: FocusWeights,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.StoppedEarly {
		t.Error("expected early stop on COMPLETED marker")
	}
	// Only gen 1 ran before the marker triggered the stop.
	if len(res.Generations) != 1 {
		t.Errorf("generations = %d, want 1 (early stop)", len(res.Generations))
	}
}

func mustMeta(t *testing.T, name string) MetaAgentProfile {
	t.Helper()
	p, err := LoadMetaProfile(name, LoadProvider, nil)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustTarget(t *testing.T, name string) TargetAgentProfile {
	t.Helper()
	p, err := LoadTargetProfile(name, LoadProvider)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected file to exist: %s (%v)", path, err)
	}
}
