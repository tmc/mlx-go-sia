#!/usr/bin/env bash
# demo-accuracy.sh — the hero demo: held-out ACCURACY climbing 0.50 -> 0.69 -> 1.00
# over three generations, replayed over time so the curve animates as if live.
#
# Usage:
#   scripts/demo-accuracy.sh [speed]
#
# speed (default 1) compresses the recorded cadence; e.g. 2 plays twice as fast.
# Replays the committed fixture testdata/weights-accuracy-replay.json, so it needs
# no /tmp run tree and no GPU. The window header reads "REPLAY · RECORDED" (blue)
# — the numbers are the real recorded held-out accuracy, shown over time, never a
# live measurement and never fabricated. macOS only (siadash is //go:build darwin).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEED="${1:-1}"
FIXTURE="$REPO/examples/dashboard/siadash/testdata/weights-accuracy-replay.json"

echo "siadash replay <- $FIXTURE (speed=${SPEED}x)"
echo "  hero: held-out accuracy 0.50 -> 0.69 -> 1.00, the number that goes UP"
exec go -C "$REPO" run ./examples/dashboard/siadash -replay "$FIXTURE" -speed "$SPEED"
