#!/usr/bin/env bash
# show-run-siadash.sh — open a previous SIA run in the siadash macOS dashboard.
#
# Usage:
#   scripts/show-run-siadash.sh [runs-root]
#
# runs-root defaults to the climbing-accuracy demo run (0.50 -> 0.69 -> 1.00).
# The dashboard tails <runs-root>/run_N/gen_M/results.json and renders the
# per-generation metric chart live; pointed at a finished run it shows the
# completed series. macOS only (siadash is //go:build darwin).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-/tmp/sia-acc-runs2}"

if [ ! -d "$RUNS" ]; then
	echo "no run tree at: $RUNS" >&2
	echo "pick one of:" >&2
	ls -d /tmp/sia-*runs* 2>/dev/null | sed 's/^/  /' >&2
	exit 1
fi

echo "siadash <- $RUNS"
exec go -C "$REPO" run ./examples/dashboard/siadash -runs "$RUNS" -interval 800ms
