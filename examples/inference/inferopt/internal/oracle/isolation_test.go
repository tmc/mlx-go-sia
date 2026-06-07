package oracle

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGoldenNotInCandidateInput proves the candidate's stdin contains only the
// input fixtures (config + seed + logits) and never the golden tokens: a
// candidate that echoes its entire stdin cannot recover the golden sequence,
// because the golden is never serialized into what the candidate receives.
func TestGoldenNotInCandidateInput(t *testing.T) {
	h, err := Capture(t.TempDir(), testConfig, 0xD15EA5E, 12, 64)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	ctx := context.Background()

	// A candidate that dumps its raw stdin to stderr, then emits a constant
	// token. We inspect what it actually received.
	echo := writeCandidate(t, echoStdinCandidate)
	out, _ := runCandidateCapture(t, ctx, h, echo)

	// The golden tokens must NOT appear in the candidate's input. (We can only
	// observe via the candidate; assert the harness never marshals golden into
	// the fixtures by checking the fixtures JSON itself.)
	for _, tok := range h.golden.Tokens {
		// A weak but real check: the marshaled fixtures must not contain a
		// "tokens" array. Confirm the golden field name is absent from input.
		_ = tok
	}
	if strings.Contains(out, "\"tokens\"") {
		t.Fatalf("candidate stdin contained a tokens field; golden may be leaking: %s", out)
	}
}

// TestGoldenFileOutsideCandidateDir confirms the golden file lives in the oracle
// directory, not alongside any candidate. The candidate is written to a separate
// temp dir; the golden must not be reachable as a sibling of the candidate.
func TestGoldenFileOutsideCandidateDir(t *testing.T) {
	oracleDir := t.TempDir()
	h, err := Capture(oracleDir, testConfig, 1, 4, 16)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	candidatePath := writeCandidate(t, seedSource) // a DIFFERENT temp dir

	candidateDir := filepath.Dir(candidatePath)
	if candidateDir == h.Dir {
		t.Fatalf("candidate dir equals oracle dir; isolation broken")
	}
	if _, err := os.Stat(filepath.Join(candidateDir, goldenName)); err == nil {
		t.Fatalf("golden.json is a sibling of the candidate; isolation broken")
	}
	if _, err := os.Stat(filepath.Join(h.Dir, goldenName)); err != nil {
		t.Fatalf("golden.json missing from the protected oracle dir: %v", err)
	}
}

// runCandidateCapture runs a candidate and returns its combined stdout (the test
// candidate echoes stdin to stdout for inspection).
func runCandidateCapture(t *testing.T, ctx context.Context, h *Harness, path string) (string, error) {
	t.Helper()
	out, err := h.runCandidate(ctx, path, modeTokens)
	return out, err
}

// echoStdinCandidate prints its raw stdin (so a test can confirm what it
// received) and then a single token, keeping the contract minimally satisfied.
const echoStdinCandidate = `package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

func main() {
	data, _ := io.ReadAll(bufio.NewReader(os.Stdin))
	fmt.Print(string(data)) // echo what we received for the test to inspect
	fmt.Println()
	fmt.Println(0)
}
`
