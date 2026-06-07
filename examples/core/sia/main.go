// Command sia runs the SIA self-improvement loop: a meta agent seeds a target
// agent for a task, then each generation runs the target, evaluates it, and a
// feedback agent rewrites it for the next generation.
//
// Usage:
//
//	sia -task gpqa -tasks-root ./tasks -max-gen 5 -run-id 1 \
//	    -agent-cmd claude -agent-args '-p,--model,%MODEL%' \
//	    -interpreter python3
//
// The meta/feedback engine is an external agent CLI (see -agent-cmd): sia writes
// each prompt to the engine's stdin and runs it in the generation's working
// directory, where it is expected to write target_agent.py (and improvement.md
// for the feedback step). The target agent is run by -interpreter with the fixed
// --dataset_dir/--working_dir contract. No Python venv is created; dependency
// setup is the operator's responsibility.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	sia "github.com/tmc/mlx-go-sia"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sia: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	cfg := sia.ConfigFromEnv()
	fs := flag.NewFlagSet("sia", flag.ContinueOnError)

	var (
		task         = fs.String("task", "", "bundled task name ("+strings.Join(sia.BundledTasks, ", ")+"); mutually exclusive with -task-dir")
		taskDir      = fs.String("task-dir", "", "path to an external task directory")
		tasksRoot    = fs.String("tasks-root", "./tasks", "directory holding bundled task subdirectories and _shared/")
		maxGen       = fs.Int("max-gen", cfg.DefaultMaxGenerations, "number of self-improvement generations")
		runID        = fs.Int("run-id", cfg.DefaultRunID, "unique run identifier")
		runsRoot     = fs.String("runs-root", sia.NameRunsRoot, "directory runs are written under")
		metaProf     = fs.String("meta-agent-profile", cfg.DefaultMetaAgentProfile, "meta/feedback agent profile (name or path to .json)")
		targetProf   = fs.String("target-agent-profile", cfg.DefaultTargetAgentProfile, "target agent profile (name or path to .json)")
		focus        = fs.String("focus", "harness", "improvement focus: harness or weights")
		trainSandbox = fs.String("training-sandbox", "modal", "weights focus: modal or sandboxfusion")
		maxTurns     = fs.Int("max-turns", cfg.DefaultMaxTurns, "engine turn budget per invocation")

		agentCmd  = fs.String("agent-cmd", "", "external agent CLI for the meta/feedback engine (required to run a live engine)")
		agentArgs = fs.String("agent-args", "", "comma-separated args for -agent-cmd; tokens %MODEL%/%MAXTURNS%/%WORKDIR% are substituted")

		interpreter = fs.String("interpreter", "python3", "interpreter that runs the target agent (empty runs it directly)")
		evalCmd     = fs.String("eval-interpreter", "python3", "interpreter that runs the task's evaluate.py (empty runs it directly)")
		dryRun      = fs.Bool("dry-run", false, "set up the run and write the meta prompt, but use a no-op engine and target")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	focusMode := sia.Focus(*focus)
	if focusMode != sia.FocusHarness && focusMode != sia.FocusWeights {
		return fmt.Errorf("invalid -focus %q: want harness or weights", *focus)
	}

	// Resolve the task.
	taskLayout, err := sia.ResolveTaskDir(*tasksRoot, *task, *taskDir)
	if err != nil {
		return err
	}

	// Resolve profiles + providers against the built-in registries / disk.
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

	// Build the engine + target executor.
	var meta sia.AgentRunner
	var target sia.TargetExecutor
	if *dryRun || *agentCmd == "" {
		if !*dryRun {
			log.Print("no -agent-cmd given; running in dry-run mode (no-op engine, no-op target)")
		}
		meta = sia.FuncRunner{ImplName: metaProfile.AgentImpl, Fn: func(context.Context, sia.AgentRequest) error { return nil }}
		target = sia.FuncTargetExecutor(func(_ context.Context, _ sia.TargetRequest) (sia.TargetResult, error) {
			return sia.TargetResult{Success: false, ErrorMsg: "dry-run: no target executor"}, nil
		})
	} else {
		meta = &sia.ExecRunner{
			ImplName: metaProfile.AgentImpl,
			Command:  *agentCmd,
			Args:     splitArgs(*agentArgs),
			// Export the engine provider's base URL and API key under the
			// conventional OpenAI-compatible env vars so a generic OpenAI CLI
			// reaches the provider (e.g. Nebius Token Factory) without extra
			// flags. The provider's own api_key_env is expected to be set in the
			// environment already; this mirrors it onto OPENAI_API_KEY.
			Env: providerEnv(metaProfile.Provider),
		}
		target = &sia.ExecTargetExecutor{Interpreter: *interpreter, InterpreterArgs: []string{"-u"}, Progress: os.Stdout}
	}

	var eval sia.Evaluator = sia.NopEvaluator{}
	if script := taskLayout.EvaluateScript(); script != "" && !*dryRun {
		eval = &sia.ExecEvaluator{
			Script:      script,
			Interpreter: *evalCmd,
			Timeout:     time.Duration(cfg.EvalTimeout) * time.Second,
		}
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
		Focus:           focusMode,
		TrainingSandbox: sia.TrainingSandbox(*trainSandbox),
		MaxLogSize:      cfg.MaxExecutionLogSize,
		RunConfig: map[string]string{
			"task_dir":   taskLayout.TaskDir,
			"meta_model": metaProfile.Model,
			"task_model": targetProfile.Model,
			"agent_impl": metaProfile.AgentImpl,
			"max_gen":    fmt.Sprintf("%d", *maxGen),
			"focus":      string(focusMode),
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

// providerEnv returns extra environment for the engine subprocess so a generic
// OpenAI-compatible CLI reaches the provider. For an OpenAI-compatible provider
// with a base URL it sets OPENAI_BASE_URL, and mirrors the resolved API key
// (read from the provider's api_key_env) onto OPENAI_API_KEY. It returns nil for
// native providers (e.g. Anthropic), leaving the inherited environment unchanged.
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
