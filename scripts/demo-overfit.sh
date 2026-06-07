#!/usr/bin/env bash
# demo-overfit.sh — the anti-Goodhart demo: held-out test_loss RISES as later
# generations overfit a tiny train set, and the held-out gate REVISEs them. This
# is the thesis on screen — a frozen held-out verifier refusing to reward a model
# that got worse on data it never saw.
#
# Usage:
#   scripts/demo-overfit.sh [speed]
#
# Replays the committed fixture testdata/weights-loss-replay.json over time at the
# run's own recorded cadence (speed-scaled). No /tmp tree, no GPU. The header reads
# "REPLAY · RECORDED" (blue): real recorded test_loss, shown over time. gen 1 PASSes
# (best-so-far); gens 2-3 regress and the gate marks them REVISE. macOS only.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEED="${1:-1}"
FIXTURE="$REPO/examples/dashboard/siadash/testdata/weights-loss-replay.json"

echo "siadash replay <- $FIXTURE (speed=${SPEED}x)"
echo "  anti-Goodhart: held-out test_loss rises, the gate REVISEs the overfit gens"
exec go -C "$REPO" run ./examples/dashboard/siadash -replay "$FIXTURE" -speed "$SPEED"
