package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
	"github.com/tmc/mlx-go-sia/cmd/inferopt/internal/seed"
)

// samplerTarget is the P3 [sia.TargetExecutor]. In this demo the "target" the
// agent produces is a Go sampler (candidate.go), not a Python program, so this
// executor does not run the agent's code itself — the protected oracle owns
// running and timing. Its job is to make the generation runnable: ensure a
// candidate.go is present (seeding the very first generation from the embedded
// baseline so the loop produces a real gen-0 even with a no-op engine) and
// confirm it compiles, reporting a compile failure as feedback rather than a
// crash.
type samplerTarget struct {
	bench *samplerBench
}

// RunTarget ensures the generation's candidate.go exists and compiles. A missing
// candidate is seeded from the frozen baseline (gen-0 parity); a candidate that
// does not compile is reported as Success=false with the compiler error so the
// feedback agent can repair it — never a Go error, which would abort the run.
func (t *samplerTarget) RunTarget(ctx context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
	candidate := filepath.Join(req.WorkingDir, candidateFile)
	if _, err := os.Stat(candidate); err != nil {
		// The engine wrote no candidate (e.g. dry-run/fake): seed the baseline so
		// the generation is still a real, gradable gen-0.
		if writeErr := os.WriteFile(candidate, []byte(seed.Candidate), 0o644); writeErr != nil {
			return sia.TargetResult{}, fmt.Errorf("seed candidate: %w", writeErr)
		}
	}

	// A quick correctness probe doubles as a compile check: if the candidate
	// cannot even run, surface that to the feedback agent as a failed target.
	ok, reason, err := t.bench.oracle.Correct(ctx, candidate)
	if err != nil {
		return sia.TargetResult{Success: false, ErrorMsg: fmt.Sprintf("candidate did not run: %v", err)}, nil
	}
	if !ok {
		// Still a successful target *run* — the candidate executed; correctness is
		// the evaluator's verdict. Report the mismatch as advisory stdout.
		return sia.TargetResult{Success: true, Stdout: "candidate ran; correctness: " + reason}, nil
	}
	return sia.TargetResult{Success: true, Stdout: "candidate ran; correctness: exact match"}, nil
}
