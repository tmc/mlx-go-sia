// Package oracle is the protected correctness-and-timing harness for the P3
// sampler-optimization demo. It owns the golden token sequences, the fixed-seed
// input fixtures, and the frozen baseline sampler — everything the agent under
// optimization must not be able to touch.
//
// The honesty discipline (P3 spec): the agent edits only candidate.go (a decode
// sampler that turns logits into a token). The golden outputs and the reference
// implementation that produced them live here, captured into a read-only
// directory OUTSIDE the agent's working directory. The agent receives only the
// input logits; it never sees the golden tokens or this code, so it cannot
// widen a tolerance, hardcode the answer, or no-op the timing loop. It can only
// make the same computation faster.
package oracle

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Fixtures is the input half of the oracle: a batch of decode steps, each a
// vocabulary-sized logits row, plus the sampling configuration and seed every
// run shares. It is handed to the candidate verbatim; the golden token sequence
// it produces is kept separately (see [Golden]) and never shipped to the agent.
type Fixtures struct {
	Config Config      `json:"config"`
	Seed   uint64      `json:"seed"`
	Steps  [][]float32 `json:"steps"` // each row is one decode step's logits over the vocab
}

// Config is the decode-sampler configuration shared by the reference and every
// candidate. It mirrors the knobs of mlx-go-lm's sampler (temperature, top-k,
// top-p) so the demo optimizes a real sampling path, not a toy.
type Config struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"` // 0 disables
	TopP        float64 `json:"top_p"` // 1.0 disables
}

// Golden is the output half of the oracle: the exact token sequence the frozen
// reference sampler emits for [Fixtures]. Correctness is an exact match against
// this sequence — a rigid categorical pass/fail, no fuzzy tolerance.
type Golden struct {
	Tokens []int `json:"tokens"`
}

// fixturesName and goldenName are the files written into the protected dir.
const (
	fixturesName = "fixtures.json" // inputs only — safe to expose to the candidate
	goldenName   = "golden.json"   // expected tokens — NEVER exposed to the candidate
)

// candidateMode and the mode flag values are the contract between the harness
// and a candidate program: it reads fixtures.json on stdin and prints either the
// token sequence or its decode throughput.
const (
	modeTokens = "tokens" // print one token id per line for every step
	modeBench  = "bench"  // print a single "tokens_per_sec <float>" line
)

// Harness is the protected oracle rooted at a read-only directory the agent
// cannot reach. Construct it with [Capture]; afterward Dir holds only inputs the
// candidate is allowed to see and the golden file it is not.
type Harness struct {
	Dir      string   // read-only protected directory (outside any agent WorkingDir)
	fixtures Fixtures // in-memory copy of the inputs
	golden   Golden   // in-memory copy of the expected tokens
}

// Capture builds the protected harness under dir: it generates the fixed-seed
// fixtures, runs the frozen reference sampler to produce the golden tokens, and
// writes both files. dir is created if missing. The caller is expected to keep
// dir outside any agent's WorkingDir and treat it as read-only thereafter.
func Capture(dir string, cfg Config, seed uint64, steps, vocab int) (*Harness, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create oracle dir: %w", err)
	}
	fx := generateFixtures(cfg, seed, steps, vocab)
	golden := Golden{Tokens: referenceTokens(fx)}

	if err := writeJSON(filepath.Join(dir, fixturesName), fx); err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(dir, goldenName), golden); err != nil {
		return nil, err
	}
	return &Harness{Dir: dir, fixtures: fx, golden: golden}, nil
}

// FixturesPath is the input file a candidate is allowed to read. It contains
// logits + config + seed, never the golden tokens.
func (h *Harness) FixturesPath() string { return filepath.Join(h.Dir, fixturesName) }

// Correct runs candidatePath against the golden tokens and reports whether its
// emitted sequence matches exactly. ok=false (with a human reason) means the
// candidate is wrong — a REVISE, never a Go error. A Go error is returned only
// when the check itself could not run (the candidate failed to compile or the
// harness could not invoke it).
func (h *Harness) Correct(ctx context.Context, candidatePath string) (ok bool, reason string, err error) {
	out, runErr := h.runCandidate(ctx, candidatePath, modeTokens)
	if runErr != nil {
		return false, "", runErr
	}
	got, parseErr := parseTokens(out)
	if parseErr != nil {
		return false, fmt.Sprintf("candidate produced unparseable token output: %v", parseErr), nil
	}
	if len(got) != len(h.golden.Tokens) {
		return false, fmt.Sprintf("token count mismatch: got %d, want %d", len(got), len(h.golden.Tokens)), nil
	}
	for i, tok := range got {
		if tok != h.golden.Tokens[i] {
			return false, fmt.Sprintf("token %d mismatch: got %d, want %d", i, tok, h.golden.Tokens[i]), nil
		}
	}
	return true, "", nil
}

// Throughput runs candidatePath in benchmark mode and returns the decode
// throughput it reports, in tokens/sec. It is one timed run; the evaluator
// handles median-of-N and cooldown.
func (h *Harness) Throughput(ctx context.Context, candidatePath string) (float64, error) {
	out, err := h.runCandidate(ctx, candidatePath, modeBench)
	if err != nil {
		return 0, err
	}
	return parseThroughput(out)
}

// runCandidate compiles and runs the candidate program with the given mode,
// feeding it the fixtures on stdin and returning its stdout. The candidate is a
// single-file stdlib-only Go program (package main); running it via `go run`
// keeps the agent's editable surface to exactly one file with no module setup.
func (h *Harness) runCandidate(ctx context.Context, candidatePath, mode string) (string, error) {
	if _, statErr := os.Stat(candidatePath); statErr != nil {
		return "", fmt.Errorf("candidate not found: %s", candidatePath)
	}
	fxBytes, err := json.Marshal(h.fixtures)
	if err != nil {
		return "", fmt.Errorf("marshal fixtures: %w", err)
	}

	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "go", "run", candidatePath, "-mode="+mode)
	cmd.Stdin = bytes.NewReader(fxBytes)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if runErr := cmd.Run(); runErr != nil {
		return "", fmt.Errorf("run candidate (%s): %w: %s", mode, runErr, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

// parseTokens reads one integer token id per non-blank line.
func parseTokens(s string) ([]int, error) {
	var tokens []int
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil {
			return nil, fmt.Errorf("line %q: %w", line, err)
		}
		tokens = append(tokens, n)
	}
	return tokens, sc.Err()
}

// parseThroughput reads the "tokens_per_sec <float>" line the candidate prints
// in benchmark mode, tolerating extra log lines around it.
func parseThroughput(s string) (float64, error) {
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) == 2 && fields[0] == "tokens_per_sec" {
			v, err := strconv.ParseFloat(fields[1], 64)
			if err != nil {
				return 0, fmt.Errorf("parse throughput %q: %w", fields[1], err)
			}
			return v, nil
		}
	}
	if err := sc.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("no tokens_per_sec line in candidate output")
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal %s: %w", filepath.Base(path), err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", filepath.Base(path), err)
	}
	return nil
}
