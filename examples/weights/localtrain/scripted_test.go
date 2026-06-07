package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	sia "github.com/tmc/mlx-go-sia"
)

// TestSpecLadderWritesPerGenSpec verifies the ladder writes a distinct, valid
// train.py for each generation, parsed from the gen_N working directory, and
// that the tuning story is well formed: iters increase monotonically (more
// training capacity each rung) and every rung carries its rationale comment so
// the gen-over-gen diff reads like a meta-agent's reasoning. This locks the
// shape of the demo's spec series.
func TestSpecLadderWritesPerGenSpec(t *testing.T) {
	s := newScriptedSpecLadder()
	root := t.TempDir()

	var prevIters int
	for gen := 1; gen <= len(s.rungs); gen++ {
		genDir := filepath.Join(root, "gen_"+strconv.Itoa(gen))
		if err := os.MkdirAll(genDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := s.Run(context.Background(), sia.AgentRequest{WorkingDir: genDir}); err != nil {
			t.Fatalf("gen %d Run: %v", gen, err)
		}
		got, err := os.ReadFile(filepath.Join(genDir, sia.NameTrainScript))
		if err != nil {
			t.Fatalf("gen %d: train.py not written: %v", gen, err)
		}
		spec := string(got)

		// The rationale comment must be present (the narrative the diff tells).
		if want := s.rungs[gen-1].rationale; !strings.Contains(spec, want) {
			t.Errorf("gen %d: spec missing rationale %q", gen, want)
		}
		// Every spec must keep fine_tune_type = lora (the 4-bit base needs it).
		if !strings.Contains(spec, `fine_tune_type = "lora"`) {
			t.Errorf("gen %d: spec must keep fine_tune_type = lora", gen)
		}
		iters := s.rungs[gen-1].iters
		if iters <= prevIters {
			t.Errorf("gen %d: iters=%d not greater than previous %d (ladder should add training each rung)", gen, iters, prevIters)
		}
		prevIters = iters
		if !strings.Contains(spec, "iters = "+strconv.Itoa(iters)) {
			t.Errorf("gen %d: spec does not carry iters = %d", gen, iters)
		}
	}

	// The first rung is the conservative underfit baseline; the last pushes
	// hardest. Sanity-check the bookends so a future edit cannot silently flip
	// the underfit -> push direction the demo narrates.
	first, last := s.rungs[0], s.rungs[len(s.rungs)-1]
	if last.iters <= first.iters || last.loraRank < first.loraRank {
		t.Errorf("ladder should escalate: first{iters=%d rank=%d} last{iters=%d rank=%d}",
			first.iters, first.loraRank, last.iters, last.loraRank)
	}
}

// TestSpecLadderUnknownGenUsesBaseline checks that a working directory without a
// gen_N suffix falls back to the conservative first rung rather than failing.
func TestSpecLadderUnknownGenUsesBaseline(t *testing.T) {
	s := newScriptedSpecLadder()
	dir := t.TempDir() // no gen_N suffix
	if err := s.Run(context.Background(), sia.AgentRequest{WorkingDir: dir}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, sia.NameTrainScript))
	if err != nil {
		t.Fatalf("train.py not written: %v", err)
	}
	if !strings.Contains(string(got), s.rungs[0].rationale) {
		t.Error("unknown gen should fall back to the first (baseline) rung")
	}
}

// TestGenFromWorkingDir checks the rung-index parse.
func TestGenFromWorkingDir(t *testing.T) {
	tests := []struct {
		dir  string
		want int
	}{
		{"/runs/run_1/gen_1", 1},
		{"/runs/run_1/gen_3", 3},
		{"gen_12/", 12},
		{"/no/gen/here", 0},
		{"/runs/gen_x", 0},
	}
	for _, tt := range tests {
		if got := genFromWorkingDir(tt.dir); got != tt.want {
			t.Errorf("genFromWorkingDir(%q) = %d, want %d", tt.dir, got, tt.want)
		}
	}
}
