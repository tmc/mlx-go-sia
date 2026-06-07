#!/usr/bin/env bash
# demo-all.sh — run the SIA demo sweep. Prints the headless throughput chart, then
# opens the two animated siadash replays (accuracy hero, then the anti-Goodhart
# overfit-caught run). Each replay loops until you close its window.
#
# Usage:
#   scripts/demo-all.sh [speed]
#
# speed (default 2) compresses the siadash replay cadence for a snappier demo.
# The siadash windows are launched in the background so the script can sequence
# them; close a window to move on. macOS only for the siadash halves.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEED="${1:-2}"

echo "==> [1/3] P7 throughput (headless siachart)"
bash "$REPO/scripts/demo-throughput.sh" || true

echo
echo "==> [2/3] accuracy hero (siadash replay, ${SPEED}x) — close the window to continue"
bash "$REPO/scripts/demo-accuracy.sh" "$SPEED"

echo
echo "==> [3/3] anti-Goodhart overfit-caught (siadash replay, ${SPEED}x)"
bash "$REPO/scripts/demo-overfit.sh" "$SPEED"
