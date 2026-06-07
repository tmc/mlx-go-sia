// Command sia-train runs the SIA self-improvement loop in totally-local weights
// mode: each generation the engine writes a train.py, and an [sia.MLXTrainExecutor]
// reads it as a declarative hyperparameter spec and runs the Go mlx-lm-train
// binary on this host — no Python MLX/tinker runtime, no cloud sandbox.
//
// Usage:
//
//	sia-train -task lawbench -tasks-root ./tasks -max-gen 3 -run-id 1 \
//	    -base-model mlx-community/Qwen3-0.6B-4bit -train-data ./data/train \
//	    -agent-cmd claude -agent-args '-p,--model,%MODEL%'
//
// The executor never executes the agent's code: it extracts a whitelisted set of
// hyperparameters (learning_rate, lora_rank, num_layers, iters, batch_size,
// fine_tune_type, data_mix) and translates them to mlx-lm-train flags. The
// adapter weights are written into each generation's working directory.
//
// Honesty discipline: -train-data must point at a directory holding train (and
// optionally valid) JSONL only. The held-out test set is the evaluator's, kept
// in a read-only directory outside the agent's reach; this command never hands
// it to the trainer, so a metric gain is generalization, not memorization.
//
// The held-out weights evaluator is a separate component (see docs/specs/06-local.md);
// until one is wired here, evaluation falls back to the task's evaluate.py (if any)
// or is skipped. The seam is the orchestrator's Eval field, set below.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	sia "github.com/tmc/sia-apple-silicon"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sia-train: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	cfg := sia.ConfigFromEnv()
	fs := flag.NewFlagSet("sia-train", flag.ContinueOnError)

	var (
		task      = fs.String("task", "", "bundled task name ("+strings.Join(sia.BundledTasks, ", ")+"); mutually exclusive with -task-dir")
		taskDir   = fs.String("task-dir", "", "path to an external task directory")
		tasksRoot = fs.String("tasks-root", "./tasks", "directory holding bundled task subdirectories and _shared/")
		maxGen    = fs.Int("max-gen", cfg.DefaultMaxGenerations, "number of self-improvement generations")
		runID     = fs.Int("run-id", cfg.DefaultRunID, "unique run identifier")
		runsRoot  = fs.String("runs-root", sia.NameRunsRoot, "directory runs are written under")
		maxTurns  = fs.Int("max-turns", cfg.DefaultMaxTurns, "engine turn budget per invocation")

		metaProf   = fs.String("meta-agent-profile", cfg.DefaultMetaAgentProfile, "meta/feedback agent profile (name or path to .json)")
		targetProf = fs.String("target-agent-profile", cfg.DefaultTargetAgentProfile, "target agent profile (name or path to .json)")

		baseModel = fs.String("base-model", "mlx-community/Qwen3-0.6B-4bit", "base model directory or HuggingFace ID to fine-tune")
		trainData = fs.String("train-data", "", "directory with train/valid JSONL for the trainer (NOT the held-out test set); required to train")
		trainBin  = fs.String("train-bin", "mlx-lm-train", "mlx-lm-train executable (resolved on PATH if unqualified)")

		evalCmd   = fs.String("eval-interpreter", "python3", "interpreter for the task's evaluate.py fallback (empty runs it directly)")
		agentCmd  = fs.String("agent-cmd", "", "external agent CLI for the meta/feedback engine (required to run a live engine)")
		agentArgs = fs.String("agent-args", "", "comma-separated args for -agent-cmd; tokens %MODEL%/%MAXTURNS%/%WORKDIR% are substituted")
		dryRun    = fs.Bool("dry-run", false, "set up the run and write the meta prompt, but use a no-op engine (the executor still runs if a train.py exists)")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	if *trainData == "" && !*dryRun {
		return fmt.Errorf("-train-data is required (a directory of train/valid JSONL; the held-out test set must live elsewhere)")
	}

	taskLayout, err := sia.ResolveTaskDir(*tasksRoot, *task, *taskDir)
	if err != nil {
		return err
	}

	metaProfile, err := sia.LoadMetaProfile(*metaProf, sia.LoadProvider, nil)
	if err != nil {
		return err
	}
	targetProfile, err := sia.LoadTargetProfile(*targetProf, sia.LoadProvider)
	if err != nil {
		return err
	}
	resolved, err := targetProfile.AgentReference.Resolve(taskLayout)
	if err != nil {
		return err
	}
	taskFiles, err := sia.LoadTaskFiles(taskLayout, resolved)
	if err != nil {
		return err
	}
	layout, err := sia.SetupRunDirectory(*runsRoot, *runID)
	if err != nil {
		return err
	}

	// Engine: a live external agent CLI, or a no-op in dry-run.
	var meta sia.AgentRunner
	if *dryRun || *agentCmd == "" {
		if !*dryRun {
			log.Print("no -agent-cmd given; running in dry-run mode (no-op engine)")
		}
		meta = sia.FuncRunner{ImplName: metaProfile.AgentImpl, Fn: func(context.Context, sia.AgentRequest) error { return nil }}
	} else {
		meta = &sia.ExecRunner{
			ImplName: metaProfile.AgentImpl,
			Command:  *agentCmd,
			Args:     splitArgs(*agentArgs),
			Env:      providerEnv(metaProfile.Provider),
		}
	}

	// Target: the local weights bridge. It runs mlx-lm-train on this host,
	// treating each gen's train.py as a declarative spec.
	target := &sia.MLXTrainExecutor{
		TrainBin:  *trainBin,
		BaseModel: *baseModel,
		DataDir:   *trainData,
	}

	// Evaluator seam: the held-out weights evaluator plugs in here. Until then,
	// fall back to the task's evaluate.py, or skip.
	var eval sia.Evaluator = sia.NopEvaluator{}
	if script := taskLayout.EvaluateScript(); script != "" && !*dryRun {
		eval = &sia.ExecEvaluator{Script: script, Interpreter: *evalCmd}
	}

	orch := &sia.Orchestrator{
		Meta:   meta,
		Target: target,
		Eval:   eval,
		Logf:   func(format string, args ...any) { log.Printf(format, args...) },
	}

	res, err := orch.Run(context.Background(), sia.RunOptions{
		Layout:          layout,
		Task:            taskLayout,
		TaskFiles:       taskFiles,
		MetaProfile:     metaProfile,
		Target:          targetProfile,
		Resolved:        resolved,
		MaxGen:          *maxGen,
		MaxTurns:        *maxTurns,
		Focus:           sia.FocusWeights,
		TrainingSandbox: sia.SandboxLocal,
		MaxLogSize:      cfg.MaxExecutionLogSize,
		RunConfig: map[string]string{
			"task_dir":   taskLayout.TaskDir,
			"meta_model": metaProfile.Model,
			"task_model": targetProfile.Model,
			"base_model": *baseModel,
			"focus":      string(sia.FocusWeights),
			"sandbox":    string(sia.SandboxLocal),
			"max_gen":    fmt.Sprintf("%d", *maxGen),
		},
	})
	if err != nil {
		return err
	}

	fmt.Printf("completed %d generation(s); context: %s\n", len(res.Generations), res.ContextPath)
	if res.StoppedEarly {
		fmt.Println("(stopped early on COMPLETED marker)")
	}
	return nil
}

// providerEnv mirrors the engine provider's base URL and API key onto the
// conventional OpenAI-compatible env vars so a generic OpenAI CLI reaches the
// provider. It returns nil for native providers.
func providerEnv(p sia.Provider) []string {
	if p.ClientKind != sia.ClientOpenAI || p.BaseURL == "" {
		return nil
	}
	env := []string{"OPENAI_BASE_URL=" + p.BaseURL}
	if p.APIKeyEnv != "" {
		if key := os.Getenv(p.APIKeyEnv); key != "" {
			env = append(env, "OPENAI_API_KEY="+key)
		}
	}
	return env
}

// splitArgs splits a comma-separated argument list, trimming spaces and dropping
// empties.
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
