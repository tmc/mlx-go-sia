#!/usr/bin/env bash
# demo-live.sh — the LIVE proof: run REAL on-device LoRA training and watch the
# dashboard climb in real time. This is the counterpart to the replay demos
# (demo-accuracy.sh et al.): nothing recorded, nothing replayed — localtrain
# fine-tunes the base model with mlx-lm-train and scores each generation on a
# held-out test set, while siadash tails the run tree and shows each real
# results.json as it lands (green "LIVE · tailing run tree", never REPLAY).
#
# Usage:
#   scripts/demo-live.sh [gens] [model]
#
# gens  (default 3) generations to train; each is ~30-60s of GPU + eval.
# model (default mlx-community/Qwen3-0.6B-4bit) base model to LoRA-fine-tune.
#
# Held-out ACCURACY is the metric (the number that goes UP); the held-out gate
# REVISEs any generation that fails to improve. Requires mlx-lm-train on PATH (or
# ~/go/bin) and an Apple-silicon GPU. macOS only (siadash is //go:build darwin).
#
# Training runs in the background; siadash runs in the foreground. Closing the
# siadash window (or Ctrl-C) stops the demo and the training process.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENS="${1:-3}"
MODEL="${2:-mlx-community/Qwen3-0.6B-4bit}"
RUNS="${RUNS:-/tmp/sia-live-demo}"
TRAIN_BIN="${TRAIN_BIN:-mlx-lm-train}"

# Resolve mlx-lm-train: PATH first, then the conventional ~/go/bin install.
if ! command -v "$TRAIN_BIN" >/dev/null 2>&1; then
	if [ -x "$HOME/go/bin/mlx-lm-train" ]; then
		TRAIN_BIN="$HOME/go/bin/mlx-lm-train"
	else
		echo "demo-live: mlx-lm-train not found on PATH or in ~/go/bin." >&2
		echo "  install it, or set TRAIN_BIN=/path/to/mlx-lm-train." >&2
		exit 1
	fi
fi

rm -rf "$RUNS"
mkdir -p "$RUNS"
TRAIN_LOG="$RUNS/training.log"

echo "demo-live: REAL on-device training — $GENS gens, model=$MODEL"
echo "  metric: held-out accuracy (higher is better); gate REVISEs non-improving gens"
echo "  runs:   $RUNS"
echo "  log:    $TRAIN_LOG"
echo

# Launch the real training run in the background. LOCALTRAIN_METRIC=accuracy
# selects the accuracy ladder/metric (it has no flag); -dry-run=false is what
# makes it actually train on the GPU rather than just scaffolding.
LOCALTRAIN_METRIC=accuracy go -C "$REPO" run ./examples/weights/localtrain \
	-dry-run=false -engine scripted -max-gen "$GENS" -run-id 1 \
	-runs-root "$RUNS" -model "$MODEL" -train-bin "$TRAIN_BIN" \
	>"$TRAIN_LOG" 2>&1 &
train_pid=$!

# Stop training when the dashboard exits (window closed or Ctrl-C).
cleanup() { kill "$train_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "demo-live: training PID $train_pid; opening siadash (tail $TRAIN_LOG for training detail)"
echo
exec go -C "$REPO" run ./examples/dashboard/siadash -runs "$RUNS" -interval 800ms
