# localtrain — P6 totally-local weight-training demo

Runs the entire SIA loop on-device: each generation the agent revises a LoRA
training spec (`train.py`), `MLXTrainExecutor` trains a small model via
`mlx-lm-train` (no Python runtime, no cloud), and `WeightsEvaluator` scores the
trained adapter on a held-out test set kept outside the agent's reach. The number
that moves is held-out loss after on-device training — and the held-out gate
REVISES any generation that regresses.

This is the **T1-RECORDED** tier: a live training run heats the GPU, and the
thermal drift fakes inference-benchmark wins — so do not run it live next to
`inferopt` (P3). Record it.

## Run

    go run ./examples/weights/localtrain                   # dry-run self-test (no GPU)

    # The multi-gen weight-improvement loop (RECORDED, T1) — verified command:
    go run ./examples/weights/localtrain -dry-run=false -engine scripted -max-gen 3 \
        -runs-root ./runs-localtrain -run-id 1 \
        -model mlx-community/Qwen3-0.6B-4bit

Default is `-dry-run` (scaffold + wire, skip the GPU) so the no-flag form is a
green self-test. Drop `-dry-run` for real generations.

Engines: `-engine scripted` walks a deterministic spec-tuning ladder (the
labeled-honest fallback, like `inferopt`'s scripted engine — see below);
`-engine pi` / `-agent-cmd claude` let an LLM tune the hyperparameters (the
offline `pi` 1B may rewrite the declarative spec into invalid Python — `scripted`
is the reliable record); the no-op engine (no `-engine`) trains the clean seed
spec as-is.

## The scripted ladder (the on-thesis demo)

`-engine scripted` revises `train.py` each generation along an
underfit → diagnose → push story a meta-agent would walk:

| gen | spec (the agent's revision)                | iters | LR   | rank |
|-----|--------------------------------------------|-------|------|------|
| 1   | conservative first pass — expected underfit| 100   | 5e-6 | 8    |
| 2   | diagnosed underfitting; raise iters/LR/rank| 200   | 2e-5 | 16   |
| 3   | push further to chase the last gains       | 400   | 3e-5 | 16   |

A recorded run on Qwen3-0.6B-4bit (10 train / 4 valid / 4 held-out rows) trains
real, distinct LoRA adapters per generation (rank 8 ≈ 11.5 MB, rank 16 ≈ 23 MB)
in well under a minute each. On this tiny train set the conservative gen 1
generalizes BEST on held-out (`test_loss ≈ 2.44`); gens 2–3 OVERFIT — held-out
loss RISES (≈2.47, ≈2.62) — so the gate flags them REVISE: *"held-out test_loss
2.4744 > best-so-far 2.4423 (gen 1): overfitting, rejected."* The held-out gate
refusing to bless a model that got worse on data it never saw is the point. (Run
to run, the absolute losses vary slightly — LoRA training is nondeterministic —
but the overfit-caught shape is stable.) The dataset is for the mechanism, not a
benchmark; the number is honest, just not competitive.

Use a separate `-runs-root` (e.g. `./runs-localtrain`) so the LoRA adapter
binaries and data trees do not collide with a P3 `inferopt` run under
`./runs`. Note `./runs` is gitignored but `./runs-localtrain` is not — clean it
(and never stage the `.safetensors`).

## Honesty

Training data (`_data/`) holds train/valid only. The held-out `test.jsonl` lives
in `_heldout/`, the evaluator's alone — it never reaches `mlx-lm-train` via the
agent, so a metric move is generalization, not memorization.

**Adapter-aware eval.** The evaluator must score the TRAINED adapter, not the
bare base model. `mlx-lm-train` only attaches and resumes a LoRA adapter inside
its training path, so a plain `-test` (no `-train`) evaluates the base weights and
returns the same loss for every generation regardless of training. The evaluator
therefore runs `-test -train -iters 0 -resume-adapter-file …`: it enters the
training path (attach + resume the adapter) but takes ZERO optimizer steps, a pure
adapter-aware evaluation that does not perturb the resumed weights. Verified: the
bare base model scores `4.1875` while three different real adapters score
`2.44 / 2.47 / 2.62`.

**Causal gate.** A generation is REVISE'd when its held-out `test_loss` exceeds
the best of all *strictly prior* generations (gen N reads only gen 1…N-1's
`results.json`), so a later result never retroactively changes an earlier
verdict — an honest online loop. The raw `test_loss` is always kept in
`results.json`, never hidden behind the verdict.
