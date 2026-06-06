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

    # One real local LoRA generation (RECORDED, T1) — verified command:
    go run ./cmd/localtrain -dry-run=false -max-gen 1 \
        -runs-root ./runs-localtrain -run-id 1 \
        -model mlx-community/Qwen3-0.6B-4bit

Default is `-dry-run` (scaffold + wire, skip the GPU) so the no-flag form is a
green self-test. Drop `-dry-run` for a real generation. The no-op engine trains
the clean seed spec as-is; add `-engine pi` or `-agent-cmd claude` to let an
agent tune the hyperparameters first (the offline `pi` 1B may rewrite the
declarative spec into invalid Python — the no-op engine is the reliable record).

A recorded real run on Qwen3-0.6B-4bit trains a 2.884M-parameter LoRA adapter
(112 adapters over 16 layers, 100 iters) in well under a minute and writes
`adapters/adapters.safetensors`; the evaluator then scores it on the held-out
test set (a real `test_loss`/`perplexity`). The 14-row demo dataset is for the
mechanism, not a benchmark — the number is honest, just not competitive.

Use a separate `-runs-root` (e.g. `./runs-localtrain`) so the LoRA adapter
binary and data trees do not collide with a P3 `cmd/inferopt` run under `./runs`.

## Honesty

Training data (`runs/_data/`) holds train/valid only. The held-out `test.jsonl`
lives in `runs/_heldout/`, the evaluator's alone — it never reaches `mlx-lm-train`
via the agent, so a metric gain is generalization, not memorization. The
evaluator scores a resumed adapter with `mlx-lm-train -test` (eval-only, no
`-train`).
