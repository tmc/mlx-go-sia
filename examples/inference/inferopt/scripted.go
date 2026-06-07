package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	sia "github.com/tmc/sia-apple-silicon"
	"github.com/tmc/sia-apple-silicon/examples/inference/inferopt/internal/seed"
)

// scriptedImprover is a deterministic meta engine that stands in for an LLM when
// none capable is available: each generation it writes a candidate.go that is a
// real, token-identical step up the optimization ladder, so the throughput
// series climbs honestly and reproducibly. It is the labeled fallback the P3
// demo uses when a live agent (claude/pi) cannot reliably rewrite the sampler.
//
// It is NOT a cheat on the correctness gate: every candidate it writes is graded
// by the same hidden golden oracle as any agent's; the engine simply applies
// optimizations a competent engineer would (drop the full-vocab sorts, reuse
// buffers) while preserving the exact algorithm, tie-breaks, and RNG. The climb
// is real work that genuinely passes; only the authorship is scripted.
type scriptedImprover struct {
	// ladder is the per-generation source written to candidate.go. ladder[0] is
	// gen-1's output, ladder[1] gen-2's, and so on; the last entry is reused for
	// any generation past the ladder's length.
	ladder []string
}

// newScriptedImprover returns the default ladder: a single verified optimization
// applied from gen-1 onward (the fully optimized, sort-free, allocation-free
// sampler). gen-0 is the seed baseline the orchestrator grades first, so the
// series is seed → optimized → optimized …, a clean monotone win.
func newScriptedImprover() *scriptedImprover {
	return &scriptedImprover{ladder: []string{seed.Optimized}}
}

// Name reports the agent-impl id used for profile validation.
func (s *scriptedImprover) Name() string { return "scripted" }

var _ sia.AgentRunner = (*scriptedImprover)(nil)

// Run writes the ladder entry for this generation into the working directory.
// The generation index is parsed from the working directory's gen_N suffix; if
// it cannot be determined, the first ladder entry is used (a safe, correct win).
func (s *scriptedImprover) Run(_ context.Context, req sia.AgentRequest) error {
	if len(s.ladder) == 0 {
		return fmt.Errorf("scripted improver: empty ladder")
	}
	gen := genFromWorkingDir(req.WorkingDir)
	// gen is 1-based; ladder[0] is gen-1's output.
	i := gen - 1
	if i < 0 {
		i = 0
	}
	if i >= len(s.ladder) {
		i = len(s.ladder) - 1
	}
	candidate := filepath.Join(req.WorkingDir, candidateFile)
	if err := os.WriteFile(candidate, []byte(s.ladder[i]), 0o644); err != nil {
		return fmt.Errorf("scripted improver: write candidate: %w", err)
	}
	return nil
}

// genFromWorkingDir extracts N from a path ending in gen_N, returning 0 if the
// suffix is absent or unparsable.
func genFromWorkingDir(dir string) int {
	base := filepath.Base(filepath.Clean(dir))
	const prefix = "gen_"
	if !strings.HasPrefix(base, prefix) {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimPrefix(base, prefix))
	if err != nil {
		return 0
	}
	return n
}
