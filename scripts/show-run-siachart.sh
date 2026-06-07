#!/usr/bin/env bash
# show-run-siachart.sh — render a previous SIA run in the terminal with siachart.
#
# Usage:
#   scripts/show-run-siachart.sh [run-dir] [metric]
#
# run-dir defaults to the P7 metalopt throughput run (the 40.8x kernel speedup);
# metric is one of delta | speedup | tokens | correctness (default speedup).
# No GUI — prints a sparkline + table from <run-dir>/gen_N/results.json, so it
# works headless. siachart renders the throughput/correctness schema; for the
# weights/accuracy run use show-run-siadash.sh instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-/tmp/metalopt-runs/run_901}"
METRIC="${2:-speedup}"

if [ ! -d "$RUN" ]; then
	echo "no run directory at: $RUN" >&2
	echo "expected a run_N dir containing gen_M/results.json; available:" >&2
	find /tmp -maxdepth 2 -type d -name 'run_*' 2>/dev/null | sed 's/^/  /' >&2
	exit 1
fi

echo "siachart <- $RUN (metric=$METRIC)"
exec go -C "$REPO" run ./examples/inference/siachart -run "$RUN" -metric "$METRIC"
