package main

import (
	"context"
	"fmt"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
	"github.com/tmc/mlx-go-sia/cmd/inferopt/internal/oracle"
)

// candidateFile is the single Go source the agent edits each generation. It is a
// standalone package main program (see the seed) read from the generation's
// working directory; the protected oracle compiles, runs, and grades it.
const candidateFile = "candidate.go"

// samplerBench is the P3 [sia.Benchmarker]: it bridges the agent's per-generation
// candidate.go to the protected [oracle.Harness]. Correctness is an exact-token
// match against a golden sequence the agent never sees; throughput is the
// candidate's reported decode tokens/sec. The frozen gen-0 baseline is the seed
// candidate, captured once at construction outside the agent's reach.
type samplerBench struct {
	oracle   *oracle.Harness // golden + fixtures, in a read-only dir outside WorkingDir
	baseline string          // path to the frozen gen-0 candidate (also outside WorkingDir)
}

// newSamplerBench builds the protected harness and freezes the baseline. h.Dir
// and baselinePath must live outside any agent working directory; the caller
// (main) places them under the run's _oracle dir, which the agent is never given.
func newSamplerBench(h *oracle.Harness, baselinePath string) *samplerBench {
	return &samplerBench{oracle: h, baseline: baselinePath}
}

// candidatePath returns the gen's candidate source path.
func (b *samplerBench) candidatePath(genDir string) string {
	return filepath.Join(genDir, candidateFile)
}

// Correct runs the gen's candidate against the golden oracle and reports an exact
// match. A wrong candidate is ok=false (REVISE); a Go error means the check could
// not run (the candidate failed to compile), which the evaluator surfaces as a
// failed cycle rather than a silent pass.
func (b *samplerBench) Correct(ctx context.Context, genDir string) (bool, string, error) {
	ok, reason, err := b.oracle.Correct(ctx, b.candidatePath(genDir))
	if err != nil {
		// A compile/run failure of the candidate is the agent's fault, not the
		// harness's: report it as REVISE feedback, not a run-aborting Go error.
		return false, fmt.Sprintf("candidate did not run: %v", err), nil
	}
	return ok, reason, nil
}

// RunCandidate times the gen's candidate once and returns its decode throughput.
func (b *samplerBench) RunCandidate(ctx context.Context, genDir string) (sia.Sample, error) {
	tps, err := b.oracle.Throughput(ctx, b.candidatePath(genDir))
	if err != nil {
		return sia.Sample{}, fmt.Errorf("time candidate: %w", err)
	}
	return sia.Sample{Throughput: tps}, nil
}

// RunBaseline times the frozen gen-0 baseline once, interleaved with each
// candidate run so the reported delta cancels thermal and cache drift.
func (b *samplerBench) RunBaseline(ctx context.Context, _ string) (sia.Sample, error) {
	tps, err := b.oracle.Throughput(ctx, b.baseline)
	if err != nil {
		return sia.Sample{}, fmt.Errorf("time baseline: %w", err)
	}
	return sia.Sample{Throughput: tps}, nil
}

// Unit names the throughput unit written to results.json.
func (b *samplerBench) Unit() string { return "tokens_per_sec" }
