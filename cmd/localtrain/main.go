// Command localtrain runs the P6 totally-local weight-training demo: a SIA
// FocusWeights loop where the agent rewrites a LoRA training spec (train.py),
// the MLXTrainExecutor trains a small model on-device via mlx-lm-train (no
// Python runtime, no cloud), and a WeightsEvaluator scores the trained adapter
// on a HELD-OUT test set kept outside the agent's reach. The number that goes up
// (well, down) is held-out loss after on-device training.
//
// This is the T1-RECORDED tier of the demo: it is runnable, but a live training
// run heats the GPU and a multi-minute training loop has no place adjacent to
// the P3 inference benchmark (thermal drift fakes benchmark wins). Record it;
// do not run it live next to cmd/inferopt.
//
// By default it runs with -dry-run (scaffolds data + spec, uses a no-op engine,
// and skips the GPU training/eval) so `go run ./cmd/localtrain` is a green
// self-test. Drop -dry-run with -agent-cmd/-engine and a base model to do a real
// local LoRA generation.
//
// Usage:
//
//	localtrain                                    # dry-run self-test (no GPU)
//	localtrain -dry-run=false -engine pi -max-gen 1 \
//	    -model mlx-community/Qwen3-0.6B-4bit       # one real local LoRA generation
package main

import (
	"context"
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
	log.SetPrefix("localtrain: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	fs := flag.NewFlagSet("localtrain", flag.ContinueOnError)
	var (
		runsRoot  = fs.String("runs-root", "./runs", "directory runs are written under")
		runID     = fs.Int("run-id", 1, "unique run identifier")
		maxGen    = fs.Int("max-gen", 2, "generations (P6: small, 1-2 — training is slow/thermal)")
		maxTurns  = fs.Int("max-turns", 20, "engine turn budget per invocation")
		model     = fs.String("model", "mlx-community/Qwen3-0.6B-4bit", "base model to LoRA-fine-tune")
		engine    = fs.String("engine", "", "offline engine: \"pi\" for pi-mlx; empty uses -agent-cmd or a no-op")
		agentCmd  = fs.String("agent-cmd", "", "external agent CLI for the meta/feedback engine (e.g. claude)")
		agentArgs = fs.String("agent-args", "", "comma-separated args for -agent-cmd; %MODEL%/%MAXTURNS%/%WORKDIR% substituted")
		trainBin  = fs.String("train-bin", "mlx-lm-train", "mlx-lm-train executable")
		dryRun    = fs.Bool("dry-run", true, "scaffold + wire but skip GPU training/eval (default; the safe self-test)")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	layout, err := sia.SetupRunDirectory(*runsRoot, *runID)
	if err != nil {
		return err
	}

	// Data lives in two separated trees:
	//   _data       — train/valid, given to the agent's training executor.
	//   _heldout    — read-only test.jsonl, the evaluator's only; never the agent's.
	dataDir := filepath.Join(*runsRoot, "_data", fmt.Sprintf("run_%d", *runID))
	heldOutDir := filepath.Join(*runsRoot, "_heldout", fmt.Sprintf("run_%d", *runID))
	if err := scaffoldData(dataDir, heldOutDir); err != nil {
		return fmt.Errorf("scaffold data: %w", err)
	}

	taskLayout, resolved, taskFiles, err := scaffoldTask(filepath.Join(*runsRoot, "_taskroot", fmt.Sprintf("run_%d", *runID)))
	if err != nil {
		return fmt.Errorf("scaffold task: %w", err)
	}

	meta, engineName, err := buildEngine(*engine, *agentCmd, *agentArgs)
	if err != nil {
		return err
	}

	// Target: the agent's train.py is read as a declarative spec and run via
	// mlx-lm-train (LoRA on a 4-bit base) — no Python runtime. DataDir is the
	// train/valid tree only; the held-out test set never reaches the executor.
	mlxTarget := sia.NewMLXTrainExecutor(*model, dataDir)
	mlxTarget.TrainBin = *trainBin
	// Seed the gen's train.py from the reference spec when the engine writes none
	// (the no-op self-test), so the demo exercises the real spec-parse path.
	target := &seedingTarget{inner: mlxTarget, seed: seedTrainSpec}

	eval := &WeightsEvaluator{
		TrainBin:   *trainBin,
		BaseModel:  *model,
		HeldOutDir: heldOutDir,
		DryRun:     *dryRun,
	}

	log.Printf("engine=%s model=%s max-gen=%d dry-run=%v", engineName, *model, *maxGen, *dryRun)
	if *dryRun {
		log.Print("DRY-RUN: scaffolding + wiring only; no GPU training/eval. Drop -dry-run for a real local LoRA generation.")
	} else {
		log.Print("LIVE local training: T1-RECORDED tier — do NOT run adjacent to the P3 benchmark (thermal drift).")
	}

	orch := &sia.Orchestrator{
		Meta:   meta,
		Target: target,
		Eval:   eval,
		Logf:   func(format string, args ...any) { log.Printf(format, args...) },
	}

	res, err := orch.Run(context.Background(), sia.RunOptions{
		Layout:    layout,
		Task:      taskLayout,
		TaskFiles: taskFiles,
		MetaProfile: sia.MetaAgentProfile{
			ProfileID: "localtrain-meta", Name: "localtrain meta", AgentImpl: meta.Name(), Model: *model,
		},
		Target: sia.TargetAgentProfile{
			ProfileID: "localtrain-target", Name: "localtrain LoRA", Model: *model,
			AgentReference: sia.DefaultAgentReference,
		},
		Resolved: resolved,
		MaxGen:   *maxGen,
		MaxTurns: *maxTurns,
		Focus:    sia.FocusWeights,
		// SandboxLocal means "train on this host" — the totally-local path. The
		// MLXTrainExecutor reads train.py as a spec and runs mlx-lm-train itself,
		// so the sandbox enum is operationally a no-op (it only selects an advisory
		// prompt block); SandboxLocal is the honest label for the demo. See
		// 06-local.md and the sia package's sandbox_local.go.
		TrainingSandbox: sia.SandboxLocal,
		RunConfig: map[string]string{
			"demo":     "P6 totally-local LoRA fine-tune",
			"engine":   engineName,
			"base":     *model,
			"held_out": heldOutDir,
			"tier":     "T1-recorded",
		},
	})
	if err != nil {
		return err
	}

	reportWeights(layout, res)
	return nil
}

// buildEngine selects the meta/feedback engine: offline pi-mlx, an external CLI,
// or a no-op so the dry-run self-test produces a real (untrained) gen-0.
func buildEngine(engine, agentCmd, agentArgs string) (sia.AgentRunner, string, error) {
	switch {
	case engine == "pi":
		return sia.NewPiRunner(""), "pi-mlx", nil
	case engine != "":
		return nil, "", fmt.Errorf("unknown -engine %q (want \"pi\" or empty)", engine)
	case agentCmd != "":
		return &sia.ExecRunner{ImplName: "claude", Command: agentCmd, Args: splitArgs(agentArgs)}, agentCmd, nil
	default:
		log.Print("no -engine/-agent-cmd: no-op engine (the seed train.py spec is used as-is)")
		return sia.FuncRunner{ImplName: "noop", Fn: func(context.Context, sia.AgentRequest) error { return nil }}, "noop", nil
	}
}

func splitArgs(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// reportWeights prints the held-out metric series from each generation's
// results.json — the demo's "loss goes down across generations" story.
func reportWeights(layout sia.RunLayout, res sia.RunResult) {
	fmt.Println()
	fmt.Println("P6 held-out metric series (lower test_loss is better):")
	fmt.Printf("  %-5s %-8s %-8s %-12s %-12s\n", "gen", "verdict", "trained", "test_loss", "perplexity")
	for _, g := range res.Generations {
		wr, err := readWeightsResults(layout.ResultsJSON(g.Gen))
		if err != nil {
			fmt.Printf("  %-5d (no results.json: %v)\n", g.Gen, err)
			continue
		}
		fmt.Printf("  %-5d %-8s %-8v %-12.4f %-12.3f %s\n",
			g.Gen, wr.Verdict, wr.Trained, wr.TestLoss, wr.Perplexity, wr.Reason)
	}
	if res.StoppedEarly {
		fmt.Println("(feedback agent signaled completion; stopped early)")
	}
	fmt.Printf("\ncontext: %s\n", res.ContextPath)
	fmt.Println("held-out test.jsonl kept read-only outside the agent's reach (no leakage)")
}
