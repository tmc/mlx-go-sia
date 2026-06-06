# localtrain — P6 totally-local weight-training demo

Runs the entire SIA loop on-device: the agent rewrites a LoRA training spec
(`train.py`), `MLXTrainExecutor` trains a small model via `mlx-lm-train` (no
Python runtime, no cloud), and `WeightsEvaluator` scores the trained adapter on a
held-out test set kept outside the agent's reach. The number that goes down is
held-out loss after on-device training.

This is the **T1-RECORDED** tier: a live training run heats the GPU, and the
thermal drift fakes inference-benchmark wins — so do not run it live next to
`cmd/inferopt` (P3). Record it.

## Run

    go run ./cmd/localtrain                                 # dry-run self-test (no GPU)
    go run ./cmd/localtrain -dry-run=false -engine pi -max-gen 1 \
        -model mlx-community/Qwen3-0.6B-4bit                # one real local LoRA generation

Default is `-dry-run` (scaffold + wire, skip the GPU) so the no-flag form is a
green self-test. Drop `-dry-run` with `-engine`/`-agent-cmd` for a real generation.

## Honesty

Training data (`runs/_data/`) holds train/valid only. The held-out `test.jsonl`
lives in `runs/_heldout/`, the evaluator's alone — it never reaches `mlx-lm-train`
via the agent, so a metric gain is generalization, not memorization. The
evaluator scores a resumed adapter with `mlx-lm-train -test` (eval-only, no
`-train`).
