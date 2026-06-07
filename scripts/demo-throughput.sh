#!/usr/bin/env bash
# demo-throughput.sh — the P7 Metal-kernel demo: a SIA loop optimizing a Metal
# RMSNorm kernel, scored by a golden-output oracle, with 8-13x throughput speedup
# over the baseline across generations. Rendered headless in the terminal with
# siachart (the throughput/correctness schema), so it works without a GUI.
#
# Usage:
#   scripts/demo-throughput.sh [run-dir] [metric]
#
# run-dir defaults to the recorded P7 run; metric is delta|speedup|tokens|
# correctness (default speedup). For the weights/accuracy runs use the siadash
# demos (demo-accuracy.sh / demo-overfit.sh) instead — siachart renders the
# throughput schema, siadash renders the weights schema.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-/tmp/metalopt-runs/run_901}"
METRIC="${2:-speedup}"

if [ ! -d "$RUN" ]; then
	echo "no run directory at: $RUN" >&2
	echo "expected a run_N dir with gen_M/results.json; available:" >&2
	find /tmp -maxdepth 2 -type d -name 'run_*' 2>/dev/null | sed 's/^/  /' >&2
	exit 1
fi

echo "siachart <- $RUN (metric=$METRIC)"
echo "  P7: Metal RMSNorm kernel, golden-oracle gated, ~8-13x throughput speedup"
exec go -C "$REPO" run ./examples/inference/siachart -run "$RUN" -metric "$METRIC"
