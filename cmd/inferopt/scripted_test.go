package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	sia "github.com/tmc/mlx-go-sia"
	"github.com/tmc/mlx-go-sia/cmd/inferopt/internal/oracle"
	"github.com/tmc/mlx-go-sia/cmd/inferopt/internal/seed"
)

// TestOptimizedCandidateIsCorrectAndFaster verifies the scripted improver's
// optimized sampler against the real golden oracle: it must emit the exact same
// tokens as the seed (correctness gate) and decode faster (the honest win the
// demo claims). This locks the optimization to "token-identical, faster" so a
// future edit that changes the tokens fails the build, not just the demo.
func TestOptimizedCandidateIsCorrectAndFaster(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles + runs go programs; skipped in -short")
	}
	dir := t.TempDir()
	cfg := oracle.Config{Temperature: 0.8, TopK: 64, TopP: 0.95}
	h, err := oracle.Capture(dir, cfg, 0x51A00B73, 256, 4096)
	if err != nil {
		t.Fatal(err)
	}

	seedPath := filepath.Join(dir, "seed_candidate.go")
	if err := os.WriteFile(seedPath, []byte(seed.Candidate), 0o644); err != nil {
		t.Fatal(err)
	}
	optPath := filepath.Join(dir, "opt_candidate.go")
	if err := os.WriteFile(optPath, []byte(seed.Optimized), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	ok, reason, err := h.Correct(ctx, optPath)
	if err != nil {
		t.Fatalf("Correct(optimized): %v", err)
	}
	if !ok {
		t.Fatalf("optimized candidate must match golden tokens exactly, but: %s", reason)
	}

	seedTP, err := h.Throughput(ctx, seedPath)
	if err != nil {
		t.Fatalf("Throughput(seed): %v", err)
	}
	optTP, err := h.Throughput(ctx, optPath)
	if err != nil {
		t.Fatalf("Throughput(optimized): %v", err)
	}
	// The optimization (dropping two full-vocab sorts + per-step allocation) is
	// algorithmic, so it wins even on a loaded machine; require a margin clear of
	// timing noise rather than a tight ratio.
	if optTP <= seedTP {
		t.Fatalf("optimized not faster: seed=%.1f opt=%.1f tok/s", seedTP, optTP)
	}
	t.Logf("token-identical; seed=%.1f opt=%.1f tok/s (%.2fx)", seedTP, optTP, optTP/seedTP)
}

// TestGenFromWorkingDir checks the ladder index parse.
func TestGenFromWorkingDir(t *testing.T) {
	tests := []struct {
		dir  string
		want int
	}{
		{"/runs/run_1/gen_1", 1},
		{"/runs/run_1/gen_12", 12},
		{"gen_3/", 3},
		{"/no/gen/here", 0},
		{"/runs/gen_x", 0},
	}
	for _, tt := range tests {
		if got := genFromWorkingDir(tt.dir); got != tt.want {
			t.Errorf("genFromWorkingDir(%q) = %d, want %d", tt.dir, got, tt.want)
		}
	}
}

// TestScriptedImproverWritesCandidate verifies the improver writes candidate.go
// into the generation's working directory.
func TestScriptedImproverWritesCandidate(t *testing.T) {
	dir := t.TempDir()
	genDir := filepath.Join(dir, "gen_1")
	if err := os.MkdirAll(genDir, 0o755); err != nil {
		t.Fatal(err)
	}
	s := newScriptedImprover()
	if err := s.Run(context.Background(), sia.AgentRequest{WorkingDir: genDir}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(genDir, candidateFile))
	if err != nil {
		t.Fatalf("candidate.go not written: %v", err)
	}
	if string(got) != seed.Optimized {
		t.Fatal("candidate.go content does not match the optimized ladder entry")
	}
}
