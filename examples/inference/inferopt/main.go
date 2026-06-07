// Command inferopt runs the P3 "money-shot" loop: it points the SIA
// self-improvement loop at a decode-step sampler and lets the agent optimize it
// for throughput, with a tokens/sec evaluator that gates on exact-token
// correctness. Each generation the agent rewrites candidate.go (a top-k/top-p/
// temperature sampler); the evaluator runs a frozen golden oracle the agent
// cannot touch, then — only if the tokens match — times decode throughput
// against an interleaved gen-0 baseline. The per-generation results.json carries
// the climbing-throughput series for the demo chart.
//
// It is self-contained: with no flags it scaffolds a synthetic sampler-optimization
// task, captures the golden oracle outside the agent's reach, and runs with a
// no-op engine (the seed candidate as gen-0) so `go run ./cmd/inferopt` produces
// a real, gradable run with no model download. Point -agent-cmd at the claude
// CLI, or -engine pi at the offline pi-mlx wrapper, to let an agent actually
// optimize the sampler.
//
// Usage:
//
//	inferopt                                   # self-test: seed-only, no engine
//	inferopt -agent-cmd claude -agent-args '-p,--model,%MODEL%' -max-gen 6
//	inferopt -engine pi -max-gen 6             # fully offline via pi-mlx
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
	"github.com/tmc/mlx-go-sia/examples/inference/inferopt/internal/oracle"
	"github.com/tmc/mlx-go-sia/examples/inference/inferopt/internal/seed"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("inferopt: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	fs := flag.NewFlagSet("inferopt", flag.ContinueOnError)
	var (
		runsRoot  = fs.String("runs-root", "./runs", "directory runs are written under")
		runID     = fs.Int("run-id", 1, "unique run identifier")
		maxGen    = fs.Int("max-gen", 6, "number of self-improvement generations (P3: 5-8)")
		maxTurns  = fs.Int("max-turns", 30, "engine turn budget per invocation")
		engine    = fs.String("engine", "", "engine: \"pi\" (pi-mlx), \"scripted\" (deterministic verified-optimizer fallback); empty uses -agent-cmd or a no-op")
		agentCmd  = fs.String("agent-cmd", "", "external agent CLI for the meta/feedback engine (e.g. claude)")
		agentArgs = fs.String("agent-args", "", "comma-separated args for -agent-cmd; %MODEL%/%MAXTURNS%/%WORKDIR% are substituted")
		piScript  = fs.String("pi-script", "", "pi-mlx wrapper path for -engine pi (empty auto-detects scripts/pi-mlx, else PATH)")
		model     = fs.String("model", "", "model the engine drives (empty uses the engine default)")
		runs      = fs.Int("bench-runs", 5, "median-of-N timing runs per generation")
		steps     = fs.Int("steps", 256, "decode steps in the fixtures (sequence length to sample)")
		vocab     = fs.Int("vocab", 4096, "vocabulary size of each logits row")
		temp      = fs.Float64("temperature", 0.8, "sampler temperature")
		topK      = fs.Int("top-k", 64, "sampler top-k (0 disables)")
		topP      = fs.Float64("top-p", 0.95, "sampler top-p (1.0 disables)")
		seedVal   = fs.Uint64("seed", 0x51A0_0B73, "fixed RNG seed for fixtures + sampling")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	layout, err := sia.SetupRunDirectory(*runsRoot, *runID)
	if err != nil {
		return err
	}

	// Capture the protected oracle OUTSIDE the agent's working directories: a
	// sibling _oracle dir under the run root. The agent's WorkingDir is each
	// gen_N/ under the run dir; _oracle is never copied or handed to the agent.
	oracleDir := filepath.Join(*runsRoot, "_oracle", fmt.Sprintf("run_%d", *runID))
	cfg := oracle.Config{Temperature: *temp, TopK: *topK, TopP: *topP}
	harness, err := oracle.Capture(oracleDir, cfg, *seedVal, *steps, *vocab)
	if err != nil {
		return fmt.Errorf("capture oracle: %w", err)
	}

	// Freeze the gen-0 baseline candidate next to the oracle (also outside the
	// agent's reach) so RunBaseline always times the original, unmodified sampler.
	baselinePath := filepath.Join(oracleDir, "baseline_"+candidateFile)
	if err := os.WriteFile(baselinePath, []byte(seed.Candidate), 0o644); err != nil {
		return fmt.Errorf("freeze baseline: %w", err)
	}

	bench := newSamplerBench(harness, baselinePath)

	// Build the self-contained task + reference seed. The reference seed is the
	// naive candidate; the meta agent improves it into gen-1's candidate.go.
	taskLayout, resolved, taskFiles, err := scaffoldTask(*runsRoot, *runID, cfg, *steps, *vocab, *seedVal)
	if err != nil {
		return fmt.Errorf("scaffold task: %w", err)
	}

	meta, engineName, err := buildEngine(*engine, *agentCmd, *agentArgs, *piScript)
	if err != nil {
		return err
	}
	log.Printf("engine=%s max-gen=%d steps=%d vocab=%d temp=%.2f top-k=%d top-p=%.2f seed=0x%x",
		engineName, *maxGen, *steps, *vocab, *temp, *topK, *topP, *seedVal)

	orch := &sia.Orchestrator{
		Meta:   meta,
		Target: &samplerTarget{bench: bench},
		Eval:   &sia.ThroughputEvaluator{Bench: bench, Runs: *runs},
		Logf:   func(format string, args ...any) { log.Printf(format, args...) },
	}

	metaModel := *model
	if metaModel == "" {
		metaModel = sia.DefaultPiModel
	}
	res, err := orch.Run(context.Background(), sia.RunOptions{
		Layout:    layout,
		Task:      taskLayout,
		TaskFiles: taskFiles,
		MetaProfile: sia.MetaAgentProfile{
			ProfileID: "inferopt-meta", Name: "inferopt meta", AgentImpl: meta.Name(), Model: metaModel,
		},
		Target: sia.TargetAgentProfile{
			ProfileID: "inferopt-target", Name: "inferopt sampler", Model: metaModel,
			AgentReference: resolved.reference,
		},
		Resolved: resolved.resolved,
		MaxGen:   *maxGen,
		MaxTurns: *maxTurns,
		Focus:    sia.FocusHarness,
		RunConfig: map[string]string{
			"demo":   "P3 mlx-inference-opt (decode sampler)",
			"engine": engineName,
			"oracle": oracleDir,
		},
	})
	if err != nil {
		return err
	}

	reportThroughput(layout, res)
	return nil
}

// buildEngine selects the meta/feedback engine: the offline pi-mlx wrapper, an
// external agent CLI, or — when neither is given — a no-op runner so the loop
// still produces a seed-only baseline run for a self-test.
func buildEngine(engine, agentCmd, agentArgs, piScript string) (sia.AgentRunner, string, error) {
	switch {
	case engine == "scripted":
		return newScriptedImprover(), "scripted", nil
	case engine == "pi":
		runner := sia.NewPiRunner("")
		// The pi-mlx wrapper runs in each generation's WorkingDir, so the script
		// path must be absolute. Honor an explicit -pi-script, else use the
		// repo's scripts/pi-mlx when present (it is not installed on PATH), else
		// fall back to the bare PATH name.
		if script, err := resolvePiScript(piScript); err != nil {
			return nil, "", err
		} else if script != "" {
			runner.Script = script
		}
		return runner, "pi-mlx", nil
	case engine != "":
		return nil, "", fmt.Errorf("unknown -engine %q (want \"pi\" or empty)", engine)
	case agentCmd != "":
		return &sia.ExecRunner{ImplName: "claude", Command: agentCmd, Args: splitArgs(agentArgs)}, agentCmd, nil
	default:
		log.Print("no -engine/-agent-cmd: running seed-only (no-op engine); gen-0 baseline still graded")
		return sia.FuncRunner{ImplName: "noop", Fn: func(context.Context, sia.AgentRequest) error { return nil }}, "noop", nil
	}
}

// resolvePiScript returns the absolute pi-mlx wrapper path to use. An explicit
// path is made absolute and must exist. Empty auto-detects the repository's
// scripts/pi-mlx (relative to this command's source tree); if that is missing it
// returns "" so the runner falls back to the bare DefaultPiScript on PATH.
func resolvePiScript(explicit string) (string, error) {
	if explicit != "" {
		abs, err := filepath.Abs(explicit)
		if err != nil {
			return "", fmt.Errorf("resolve -pi-script: %w", err)
		}
		if _, err := os.Stat(abs); err != nil {
			return "", fmt.Errorf("-pi-script %q: %w", explicit, err)
		}
		return abs, nil
	}
	// cmd/inferopt → repo root is two levels up; scripts/pi-mlx sits there.
	for _, rel := range []string{"scripts/pi-mlx", "../../scripts/pi-mlx"} {
		if abs, err := filepath.Abs(rel); err == nil {
			if _, statErr := os.Stat(abs); statErr == nil {
				log.Printf("pi-mlx: using repo wrapper %s (not on PATH)", abs)
				return abs, nil
			}
		}
	}
	return "", nil
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
