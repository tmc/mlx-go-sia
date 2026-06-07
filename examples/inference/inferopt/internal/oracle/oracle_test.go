package oracle

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// testConfig is a representative stochastic sampler config (top-k + top-p +
// temperature all active) so the oracle exercises the full path.
var testConfig = Config{Temperature: 0.8, TopK: 16, TopP: 0.9}

// captureTest builds a small protected harness for tests.
func captureTest(t *testing.T) *Harness {
	t.Helper()
	h, err := Capture(t.TempDir(), testConfig, 0xC0FFEE, 24, 128)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	return h
}

// writeCandidate writes src to a candidate.go under a fresh dir and returns its path.
func writeCandidate(t *testing.T, src string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "candidate.go")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatalf("write candidate: %v", err)
	}
	return path
}

func TestCorrect(t *testing.T) {
	h := captureTest(t)
	ctx := context.Background()

	tests := []struct {
		name    string
		src     string
		wantOK  bool
		wantErr bool
	}{
		{name: "reference seed matches golden", src: seedSource, wantOK: true},
		{name: "faster variant matches golden", src: fasterSource, wantOK: true},
		{name: "wrong variant fails oracle", src: wrongSource, wantOK: false},
		{name: "uncompilable variant errors", src: brokenSource, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ok, reason, err := h.Correct(ctx, writeCandidate(t, tt.src))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("want a run error, got ok=%v reason=%q", ok, reason)
				}
				return
			}
			if err != nil {
				t.Fatalf("Correct: %v", err)
			}
			if ok != tt.wantOK {
				t.Fatalf("ok = %v (reason %q), want %v", ok, reason, tt.wantOK)
			}
		})
	}
}

func TestThroughput(t *testing.T) {
	h := captureTest(t)
	ctx := context.Background()
	tps, err := h.Throughput(ctx, writeCandidate(t, seedSource))
	if err != nil {
		t.Fatalf("Throughput: %v", err)
	}
	if tps <= 0 {
		t.Fatalf("throughput = %v, want > 0", tps)
	}
}

// seedSource is the reference seed itself; it must match the golden by construction.
const seedSource = candidateContract + naiveSample

// fasterSource keeps the exact algorithm but avoids the full-vocab sort on the
// top-k step by selecting via a partial scan — same tokens, less work.
const fasterSource = candidateContract + fasterSample

// wrongSource perturbs the algorithm (skips temperature scaling) so the drawn
// tokens diverge from the golden — the oracle must reject it.
const wrongSource = candidateContract + wrongSample

// brokenSource does not compile.
const brokenSource = "package main\nfunc main() { this is not go }\n"
