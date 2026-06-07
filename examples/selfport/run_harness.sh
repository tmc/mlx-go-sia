#!/usr/bin/env bash
# run_harness.sh — SIA self-port harness: porting agent -> go test grader
#
# Anti-leak design:
#   AGENT SEES:  port_prompt.txt (Python reference + behavior contract + test signatures)
#   AGENT DOES NOT SEE: jsonutil_test.go expected values, original Go source
#
# Model is swappable via PI_MLX_MODEL env var (default: Llama-3.2-1B-Instruct-4bit).
# For P6 improved adapters: PI_MLX_MODEL=<adapter-path> ./run_harness.sh
#
# Usage:
#   ./run_harness.sh             # use base weights
#   PI_MLX_MODEL=<path> ./run_harness.sh   # use improved weights/adapter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$SCRIPT_DIR/jsonutil_port"
PROMPT_FILE="$SCRIPT_DIR/port_prompt.txt"
CANDIDATE_FILE="$PORT_DIR/jsonutil.go"
PI_MLX="$SCRIPT_DIR/../../../../../../../go/src/github.com/tmc/mlx-go-examples/mlx-go-sia/scripts/pi-mlx"

# Resolve pi-mlx absolute path
MAIN_REPO="/Users/tmc/go/src/github.com/tmc/mlx-go-examples/mlx-go-sia"
PI_MLX="$MAIN_REPO/scripts/pi-mlx"

MODEL="${PI_MLX_MODEL:-mlx-community/Llama-3.2-1B-Instruct-4bit}"
MAXTOK="${PI_MLX_MAXTOK:-512}"

echo "=== SIA Self-Port Harness ==="
echo "Target: jsonutil.go (missingKeys + joinSorted)"
echo "Model: $MODEL"
echo "Max tokens: $MAXTOK"
echo ""

# --- Step 1: Verify oracle (tests fail without implementation) ---
echo "--- Step 1: Oracle check (expect build failure) ---"
if cd "$PORT_DIR" && go test ./... 2>&1 | grep -q "undefined: missingKeys"; then
    echo "PASS: Oracle correctly rejects missing implementation"
else
    echo "WARN: Oracle state unexpected — check jsonutil.go exists?"
fi
echo ""

# --- Step 2: Run porting agent via mlx-pi ---
echo "--- Step 2: Porting agent (mlx-pi / $MODEL) ---"
echo "Prompt length: $(wc -c < "$PROMPT_FILE") bytes"

# Remove any previous candidate
rm -f "$CANDIDATE_FILE"

# Run the porting agent: feed prompt via stdin, capture stdout as candidate code
RAW_OUTPUT=""
if PI_MLX_MODEL="$MODEL" PI_MLX_MAXTOK="$MAXTOK" "$PI_MLX" < "$PROMPT_FILE" > /tmp/sia_port_raw.txt 2>/tmp/sia_port_err.txt; then
    RAW_OUTPUT=$(cat /tmp/sia_port_raw.txt)
    echo "Agent generated $(echo "$RAW_OUTPUT" | wc -l) lines of output"
else
    echo "ERROR: mlx-pi failed"
    cat /tmp/sia_port_err.txt
    exit 1
fi

# --- Step 3: Extract Go code from output ---
echo ""
echo "--- Step 3: Extract Go code from agent output ---"

# Try to extract code between ```go fences first, then fall back to raw
if echo "$RAW_OUTPUT" | grep -q '```'; then
    CANDIDATE=$(echo "$RAW_OUTPUT" | sed -n '/^```\(go\)\?$/,/^```$/p' | grep -v '^```' | head -100)
else
    # Raw mode: look for package declaration
    CANDIDATE=$(echo "$RAW_OUTPUT" | sed -n '/^package /,$ p' | head -100)
fi

if [ -z "$CANDIDATE" ]; then
    echo "ERROR: Could not extract Go code from agent output"
    echo "Raw output:"
    echo "$RAW_OUTPUT" | head -30
    exit 1
fi

echo "Extracted candidate Go file ($(echo "$CANDIDATE" | wc -l) lines):"
echo "---"
echo "$CANDIDATE"
echo "---"

# Write candidate to the port directory
echo "$CANDIDATE" > "$CANDIDATE_FILE"
echo ""

# --- Step 4: Format check ---
echo "--- Step 4: gofmt check ---"
if gofmt -l "$CANDIDATE_FILE" | grep -q .; then
    echo "INFO: Candidate needs formatting — applying gofmt"
    gofmt -w "$CANDIDATE_FILE"
else
    echo "PASS: Candidate is gofmt clean"
fi

# --- Step 5: Grade via go test ---
echo ""
echo "--- Step 5: Grade (go test ./...) ---"
cd "$PORT_DIR"
TEST_OUTPUT=$(go test ./... -v 2>&1)
TEST_EXIT=$?

echo "$TEST_OUTPUT"
echo ""

# --- Step 6: Report ---
echo "=== RESULTS ==="
echo "Model: $MODEL"
PASS_COUNT=$(echo "$TEST_OUTPUT" | grep -c "^--- PASS:" || true)
FAIL_COUNT=$(echo "$TEST_OUTPUT" | grep -c "^--- FAIL:" || true)
echo "Tests passed: $PASS_COUNT"
echo "Tests failed: $FAIL_COUNT"

if [ $TEST_EXIT -eq 0 ]; then
    echo "GRADE: PASS — ported code makes go test pass"
    echo "'Eating its own tail': the 1B model successfully ported its own substrate."
else
    echo "GRADE: FAIL — ported code does not pass go test"
    echo "Capability finding: see failed tests above for where the 1B broke."
fi

echo ""
echo "Candidate file saved at: $CANDIDATE_FILE"
echo "To swap in P6 improved weights: PI_MLX_MODEL=<adapter-path> ./run_harness.sh"
