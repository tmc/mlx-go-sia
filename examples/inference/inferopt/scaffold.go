package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
	"github.com/tmc/mlx-go-sia/examples/inference/inferopt/internal/oracle"
	"github.com/tmc/mlx-go-sia/examples/inference/inferopt/internal/seed"
)

// resolvedRef bundles the agent reference and its resolution so main can pass
// both into RunOptions without re-resolving.
type resolvedRef struct {
	reference sia.AgentReference
	resolved  sia.ResolvedAgentReference
}

// scaffoldTask writes a self-contained sampler-optimization task under
// runsRoot/_task/run_N and returns the layout, resolved reference (seed code +
// deps), and loaded task files. The task steers the meta/feedback agent to
// rewrite candidate.go — a decode sampler — for throughput while preserving the
// exact token sequence the hidden golden oracle checks.
func scaffoldTask(runsRoot string, runID int, cfg oracle.Config, steps, vocab int, seedVal uint64) (sia.TaskLayout, resolvedRef, sia.TaskFiles, error) {
	root := filepath.Join(runsRoot, "_task", fmt.Sprintf("run_%d", runID))
	taskDir := root
	sharedDir := filepath.Join(filepath.Dir(taskDir), sia.NameSharedDir)

	dataPublic := filepath.Join(taskDir, sia.NameDataPublic)
	refDir := filepath.Join(taskDir, sia.NameReferenceDir)
	for _, d := range []string{dataPublic, refDir, sharedDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("mkdir %s: %w", d, err)
		}
	}

	// task.md — the contract the agent optimizes against.
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameTaskMD), []byte(taskMarkdown(cfg, steps, vocab, seedVal)), 0o644); err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("write task.md: %w", err)
	}
	// SAMPLE_TASK_DESCRIPTIONS.md and the seed reference agent (the naive sampler).
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameSampleTaskDescriptions), []byte(sampleDescriptions), 0o644); err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("write sample descriptions: %w", err)
	}
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameReferenceAgent), []byte(seed.Candidate), 0o644); err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("write reference agent: %w", err)
	}
	// A minimal valid sample execution trajectory (the loader requires valid JSON).
	if err := os.WriteFile(filepath.Join(sharedDir, sia.NameSharedSampleExecution), sampleExecution, 0o644); err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("write sample execution: %w", err)
	}

	taskLayout := sia.NewTaskLayout(taskDir, sharedDir)
	ref := sia.DefaultAgentReference // RefDefault: embeds reference_target_agent.py inline
	resolved, err := ref.Resolve(taskLayout)
	if err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("resolve reference: %w", err)
	}
	taskFiles, err := sia.LoadTaskFiles(taskLayout, resolved)
	if err != nil {
		return sia.TaskLayout{}, resolvedRef{}, sia.TaskFiles{}, fmt.Errorf("load task files: %w", err)
	}
	return taskLayout, resolvedRef{reference: ref, resolved: resolved}, taskFiles, nil
}

// taskMarkdown renders the task description shown to the agent. It deliberately
// states the algorithm + RNG contract precisely (the agent must preserve tokens)
// while inviting any throughput-preserving rewrite.
func taskMarkdown(cfg oracle.Config, steps, vocab int, seedVal uint64) string {
	return fmt.Sprintf(`# Task: optimize a decode-step sampler for throughput

You are improving a single Go file, %s, a decode-step sampler that turns model
logits into a token. Your ONLY goal is to make it sample tokens FASTER while
emitting the EXACT SAME token sequence as the original. A faster-but-different
sampler is rejected (verdict REVISE); only an identical-output, faster sampler
is a win (verdict PASS, with a tokens/sec number that should climb each
generation).

## What you may and may not change
- You MAY rewrite the body of Sample, add helper functions/types, precompute,
  avoid allocations, use partial selection instead of full sorts, etc.
- You MUST keep candidate.go a single, standalone, stdlib-only %spackage main%s
  program (it is run with %sgo run candidate.go%s) that:
  - reads the fixtures JSON on stdin,
  - with -mode=tokens, prints one token id per line (one per decode step),
  - with -mode=bench, prints a single line %stokens_per_sec <float>%s.
- You MUST NOT change the sampling algorithm or RNG (doing so changes the tokens
  and fails the hidden correctness oracle). You cannot see the golden tokens.

## The fixed algorithm (reproduce exactly)
For each decode step's logits row:
1. If temperature < 0.01: emit argmax(logits) (lowest index on ties); do not
   advance the RNG.
2. Otherwise: scaled = logits / temperature; keep the top_k highest-logit indices
   (top_k <= 0 keeps all; ties broken by lower index); numerically-stable softmax
   over the kept set; top_p (nucleus) truncation over indices in descending
   probability (lower index on ties) — the smallest prefix whose cumulative
   probability first reaches top_p, keeping at least one; renormalize; draw one
   token with the splitMix64 stream seeded from Seed and advanced once per step.

The RNG is splitMix64: state += 0x9e3779b97f4a7c15; z := state;
z = (z ^ (z>>30)) * 0xbf58476d1ce4e5b9; z = (z ^ (z>>27)) * 0x94d049bb133111eb;
return z ^ (z>>31); and float64() = (next() >> 11) / 2^53.

## This run's configuration
temperature=%.3f  top_k=%d  top_p=%.3f  steps=%d  vocab=%d  seed=0x%x

Write your improved candidate.go into the working directory. Keep it correct.
`, candidateFile, "`", "`", "`", "`", "`", "`", cfg.Temperature, cfg.TopK, cfg.TopP, steps, vocab, seedVal)
}

const sampleDescriptions = `# Sample task descriptions

- Optimize the decode sampler so median tokens/sec rises across generations
  while the emitted token sequence stays bit-identical to the golden.
- Example win: replace the full O(V log V) sort with a partial top-k selection,
  reuse buffers across steps, and skip softmax normalization where it does not
  affect the argmax/draw — all while preserving the exact tokens.
`

// sampleExecution is a minimal but valid agent-execution trajectory; the loader
// only requires it to be valid JSON shaped like the reference's samples.
var sampleExecution = json.RawMessage(`{
  "task": "optimize-decode-sampler",
  "messages": [
    {"role": "system", "content": "Improve candidate.go for throughput; keep tokens identical."},
    {"role": "assistant", "content": "Rewrote Sample to use partial top-k selection and buffer reuse."}
  ],
  "result": {"verdict": "PASS", "tokens_per_sec": 0}
}`)
