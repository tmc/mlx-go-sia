// Command metalopt runs a SIA self-improvement loop over an MLX Metal kernel.
//
// Each generation the engine rewrites a Metal Shading Language kernel source
// (kernel.metal); MLX JIT-compiles it; an external Go oracle gates correctness;
// and the score is throughput (ops/sec) measured against an interleaved,
// frozen gen-0 baseline. The number that goes up is real GPU-kernel speed on
// Apple silicon — the SIA paper's flagship domain (GPU-kernel optimization),
// extended to Metal, which the paper does not cover.
//
// Usage:
//
//	# Self-contained scripted demo (no LLM): a built-in improver rewrites the
//	# kernel through a tuned-but-honest optimization sequence; correctness is
//	# still gated against the frozen Go oracle, so a wrong step is REVISE.
//	metalopt -run-id 1
//
//	# Live engine: claude rewrites kernel.metal from the prompt each generation.
//	metalopt -run-id 2 -agent-cmd claude -agent-args '-p,--model,%MODEL%'
//
// The harness (inputs, golden oracle, baseline, timing) is frozen and lives
// outside the engine's working directory; the engine's only lever is the kernel
// source and its launch geometry (kernel.json).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	sia "github.com/tmc/mlx-go-sia"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("metalopt: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(argv []string) error {
	fs := flag.NewFlagSet("metalopt", flag.ContinueOnError)
	var (
		runsRoot  = fs.String("runs-root", sia.NameRunsRoot, "directory runs are written under")
		runID     = fs.Int("run-id", 1, "unique run identifier")
		maxGen    = fs.Int("max-gen", 6, "number of self-improvement generations")
		rows      = fs.Int("rows", 0, "RMSNorm rows (0 = spec default)")
		dim       = fs.Int("dim", 0, "RMSNorm row length (0 = spec default)")
		runsN     = fs.Int("bench-runs", 5, "median-of-N timed cycles per generation")
		iters     = fs.Int("bench-iters", 0, "timed iterations per sample (0 = default)")
		cooldown  = fs.Duration("cooldown", 0, "sleep between timed cycles (thermal settle)")
		timeout   = fs.Duration("kernel-timeout", sia.DefaultKernelTimeout, "per-generation compile-and-run cap")
		agentCmd  = fs.String("agent-cmd", "", "external engine CLI (e.g. claude); empty uses the built-in scripted improver")
		agentArgs = fs.String("agent-args", "", "comma-separated args for -agent-cmd; %MODEL%/%MAXTURNS%/%WORKDIR% are substituted")
		model     = fs.String("model", "claude-opus-4-8", "model the engine drives")
		maxTurns  = fs.Int("max-turns", 0, "engine turn budget per invocation (0 = engine default)")
	)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	spec := sia.DefaultRMSNormSpec()
	if *rows > 0 {
		spec.Rows = *rows
	}
	if *dim > 0 {
		spec.Dim = *dim
	}

	// Lay out the run, then scaffold a self-contained task whose reference is the
	// frozen seed kernel. CopyInto drops kernel.metal into each generation dir, so
	// gen-1 starts correct and the engine edits the source in place.
	layout, err := sia.SetupRunDirectory(*runsRoot, *runID)
	if err != nil {
		return err
	}
	task, taskFiles, resolved, err := scaffoldKernelTask(layout.RunDir, spec)
	if err != nil {
		return fmt.Errorf("scaffold kernel task: %w", err)
	}

	// Engine: a live agent CLI if given, else the built-in scripted improver that
	// walks a tuned-but-honest optimization sequence (correctness still gated).
	var meta sia.AgentRunner
	if *agentCmd != "" {
		meta = &sia.ExecRunner{ImplName: "exec", Command: *agentCmd, Args: splitArgs(*agentArgs)}
	} else {
		log.Print("no -agent-cmd given; using the built-in scripted kernel improver")
		meta = scriptedImprover{}
	}

	executor := &sia.MetalKernelExecutor{Spec: spec, Timeout: *timeout}
	bench := sia.NewKernelBenchmarker(spec, sia.SeedKernelSource)
	bench.Iters = *iters

	orch := &sia.Orchestrator{
		Meta:   meta,
		Target: executor,
		Eval:   &sia.ThroughputEvaluator{Bench: bench, Runs: *runsN, Cooldown: *cooldown},
		Logf:   func(format string, args ...any) { log.Printf(format, args...) },
	}

	res, err := orch.Run(context.Background(), sia.RunOptions{
		Layout:      layout,
		Task:        task,
		TaskFiles:   taskFiles,
		MetaProfile: sia.MetaAgentProfile{ProfileID: "metalopt-engine", AgentImpl: meta.Name(), Model: *model},
		Target:      sia.TargetAgentProfile{ProfileID: "metal-kernel", Model: *model, AgentReference: resolved.reference},
		Resolved:    resolved.resolved,
		MaxGen:      *maxGen,
		MaxTurns:    *maxTurns,
		Focus:       sia.FocusHarness,
		RunConfig: map[string]string{
			"problem":     fmt.Sprintf("rmsnorm %dx%d", spec.Rows, spec.Dim),
			"unit":        bench.Unit(),
			"max_gen":     fmt.Sprintf("%d", *maxGen),
			"engine":      meta.Name(),
			"bench_runs":  fmt.Sprintf("%d", *runsN),
			"kernel_seed": sia.KernelSourceName,
		},
	})
	if err != nil {
		return err
	}

	fmt.Printf("completed %d generation(s) optimizing the %dx%d RMSNorm Metal kernel; context: %s\n",
		len(res.Generations), spec.Rows, spec.Dim, res.ContextPath)
	return nil
}

// resolvedTask bundles the resolved reference with the AgentReference that
// produced it, so RunOptions can carry both.
type resolvedTask struct {
	reference sia.AgentReference
	resolved  sia.ResolvedAgentReference
}

// scaffoldKernelTask writes a complete, minimal SIA task under runDir/task whose
// reference directory holds the frozen seed kernel. It returns the resolved task
// layout, loaded task files, and the resolved reference. The task is a directory
// reference (RefDir) with kernel.metal as the entrypoint, so every generation dir
// is seeded with the working seed kernel that the engine then rewrites.
func scaffoldKernelTask(runDir string, spec sia.RMSNormSpec) (sia.TaskLayout, sia.TaskFiles, resolvedTask, error) {
	taskDir := filepath.Join(runDir, "task")
	refDir := filepath.Join(taskDir, sia.NameReferenceDir)
	publicDir := filepath.Join(taskDir, sia.NameDataPublic)
	sharedDir := filepath.Join(runDir, sia.NameSharedDir)
	for _, d := range []string{refDir, publicDir, sharedDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return sia.TaskLayout{}, sia.TaskFiles{}, resolvedTask{}, err
		}
	}

	files := map[string]string{
		filepath.Join(taskDir, sia.NameTaskMD):                  kernelTaskMD(spec),
		filepath.Join(refDir, sia.KernelSourceName):             sia.SeedKernelSource,
		filepath.Join(refDir, "SAMPLE_TASK_DESCRIPTIONS.md"):    kernelTaskMD(spec),
		filepath.Join(sharedDir, sia.NameSharedSampleExecution): "{}\n",
	}
	for path, content := range files {
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			return sia.TaskLayout{}, sia.TaskFiles{}, resolvedTask{}, err
		}
	}

	task := sia.NewTaskLayout(taskDir, sharedDir)
	ref := sia.AgentReference{Kind: sia.RefDir, Source: refDir, Entrypoint: sia.KernelSourceName}
	resolved, err := ref.Resolve(task)
	if err != nil {
		return sia.TaskLayout{}, sia.TaskFiles{}, resolvedTask{}, err
	}
	taskFiles, err := sia.LoadTaskFiles(task, resolved)
	if err != nil {
		return sia.TaskLayout{}, sia.TaskFiles{}, resolvedTask{}, err
	}
	return task, taskFiles, resolvedTask{reference: ref, resolved: resolved}, nil
}

// kernelTaskMD is the task brief shown to the engine: optimize the RMSNorm Metal
// kernel for throughput while staying correct against the frozen oracle.
func kernelTaskMD(spec sia.RMSNormSpec) string {
	return fmt.Sprintf(`# Optimize a Metal RMSNorm kernel

Rewrite the Metal Shading Language kernel in %q to compute RMS normalization

    y[r,c] = x[r,c] / sqrt(mean_c(x[r,c]^2) + eps) * w[c]

over a %d x %d row-major float32 matrix, as fast as possible (ops/sec).

Inputs (bound by the harness): x (%d*%d), w (%d). Output: out (%d*%d).
Constants are injected by a frozen header — your source may use them directly:
constant int dim; constant int nrows; constant float epsf.

Rules:
- Edit only %q (and optionally %q to set grid / thread_group).
- Correctness is gated against an external reference you cannot see or change;
  a faster-but-wrong kernel is REVISE, not a win.
- The seed kernel is correct but deliberately naive (each thread recomputes the
  whole row's sum of squares). Real wins: threadgroup-parallel reduction,
  float4 vectorization, a single fused pass.
`, sia.KernelSourceName, spec.Rows, spec.Dim,
		spec.Rows, spec.Dim, spec.Dim, spec.Rows, spec.Dim,
		sia.KernelSourceName, sia.KernelConfigName)
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

// scriptedImprover is the built-in offline engine: a deterministic [AgentRunner]
// that rewrites kernel.metal toward a faster, still-correct kernel each time it
// runs. It lets the full loop run — and the number go up — with no external
// model, and serves as the demo's pre-recorded insurance. It walks a fixed list
// of honest optimizations; once exhausted it leaves the best kernel in place.
type scriptedImprover struct{}

func (scriptedImprover) Name() string { return "scripted" }

func (scriptedImprover) Run(_ context.Context, req sia.AgentRequest) error {
	// Advance by generation number, not by the dir's current contents: the
	// orchestrator re-seeds each generation dir with the gen-0 baseline before the
	// engine runs, so the working copy is always the seed. The dir is named
	// "gen_<N>"; gen N gets stage N (gen-0's seed is the frozen baseline), and once
	// the sequence is exhausted the best kernel is held.
	stages := sia.ScriptedKernelStages()
	stage := genStage(req.WorkingDir, len(stages))

	dst := filepath.Join(req.WorkingDir, sia.KernelSourceName)
	if err := os.MkdirAll(req.WorkingDir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(dst, []byte(stages[stage]), 0o644); err != nil {
		return fmt.Errorf("scripted improver: write %s: %w", sia.KernelSourceName, err)
	}
	// A real engine also writes improvement.md; mirror that for parity.
	_ = os.WriteFile(filepath.Join(req.WorkingDir, "improvement.md"),
		[]byte(fmt.Sprintf("scripted optimization stage %d\n", stage)), 0o644)
	return nil
}

// genStage maps a "gen_<N>" working directory to the scripted stage index it
// should hold, clamped to the available stages. A directory that does not encode
// a generation number falls back to the first improvement stage.
func genStage(workingDir string, numStages int) int {
	n := 1
	if base := filepath.Base(workingDir); strings.HasPrefix(base, "gen_") {
		if v, err := strconv.Atoi(strings.TrimPrefix(base, "gen_")); err == nil {
			n = v
		}
	}
	if n >= numStages {
		return numStages - 1
	}
	if n < 1 {
		return 1
	}
	return n
}
