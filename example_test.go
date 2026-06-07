package sia_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
)

// Example runs a two-generation self-improvement loop with a scripted engine and
// a scripted target executor — the shape of a real run, without a live model.
func Example() {
	root, _ := os.MkdirTemp("", "sia-example")
	defer os.RemoveAll(root)

	// A minimal task: a task.md, sample descriptions, a seed reference agent, and
	// a shared sample execution trajectory.
	taskDir := filepath.Join(root, "mytask")
	write := func(rel, content string) {
		p := filepath.Join(taskDir, rel)
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte(content), 0o644)
	}
	write(sia.NameTaskMD, "# Task\nAnswer the question.\n")
	write(sia.NameSampleTaskDescriptions, "example question")
	write(sia.NameReferenceAgent, "# seed agent\n")
	shared := filepath.Join(root, sia.NameSharedDir)
	os.MkdirAll(shared, 0o755)
	os.WriteFile(filepath.Join(shared, sia.NameSharedSampleExecution), []byte(`[{"role":"user","content":"hi"}]`), 0o644)

	task := sia.NewTaskLayout(taskDir, shared)
	resolved, _ := sia.DefaultAgentReference.Resolve(task)
	taskFiles, _ := sia.LoadTaskFiles(task, resolved)

	// The engine (meta + feedback) writes target_agent.py each invocation.
	engine := sia.NewFakeRunner("fake",
		sia.FakeStep{Files: map[string]string{sia.NameTargetAgent: "# gen1\n"}},
		sia.FakeStep{Files: map[string]string{
			sia.NameTargetAgent:   "# gen2 (improved)\n",
			sia.NameImprovementMD: "## improvements\n",
		}},
	)
	// The target executor records a trajectory and succeeds.
	target := sia.FuncTargetExecutor(func(_ context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
		os.WriteFile(filepath.Join(req.WorkingDir, sia.NameAgentExecution), []byte(`[{"role":"assistant","content":"done"}]`), 0o644)
		os.WriteFile(req.StdoutLog, []byte("ok\n"), 0o644)
		return sia.TargetResult{Success: true, Stdout: "ok\n"}, nil
	})

	layout, _ := sia.SetupRunDirectory(filepath.Join(root, "runs"), 1)
	orch := sia.NewOrchestrator(engine, target)

	res, err := orch.Run(context.Background(), sia.RunOptions{
		Layout:      layout,
		Task:        task,
		TaskFiles:   taskFiles,
		MetaProfile: mustLoadMeta(),
		Target:      mustLoadTarget(),
		Resolved:    resolved,
		MaxGen:      2,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("generations: %d\n", len(res.Generations))
	fmt.Printf("gen1 success: %v\n", res.Generations[0].Target.Success)
	fmt.Printf("feedback ran after gen1: %v\n", res.Generations[0].FeedbackRan)
	// Output:
	// generations: 2
	// gen1 success: true
	// feedback ran after gen1: true
}

func mustLoadMeta() sia.MetaAgentProfile {
	p, _ := sia.LoadMetaProfile("default-meta", sia.LoadProvider, nil)
	return p
}

func mustLoadTarget() sia.TargetAgentProfile {
	p, _ := sia.LoadTargetProfile("default-target", sia.LoadProvider)
	return p
}
