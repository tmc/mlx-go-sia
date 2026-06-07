# P6 — Totally-local self-improvement (inference + training on-device)

**One-liner (the goal):** Run the *entire* SIA loop — agent engine, target
execution, **and real model-weight training** — fully on-device on Apple silicon
via the Go + MLX stack, so we can show SIA improving a model **with no cloud at
all**. This is the headline aspiration; Nebius (`05-nebius.md`) is the cloud
contrast.

**Status: goal / design doc.** This is the doc the user asked to iterate with
NotebookLM. It is deliberately ambitious; the "Tiered fallback" section defines
what we actually demo if the full weight-update loop doesn't land in time.

## Why this is the strongest story

SIA's thesis is "an agent that rewrites both its harness **and** its weights."
Most teams will only demo harness rewriting (it needs no GPU). Showing **weight
updates running locally** — the model literally getting better on-device — is the
fullest expression of the paper, and "no network, no API, no cloud" is a claim
nobody can wave away. It also exercises the part of `mlx-go-sia` that the harness
demos don't: `FocusWeights`.

## What exists (verified) — the local building blocks

**Inference (offline, done):** `mlx-go-lm` — 81 architectures, decode ≥ Python
mlx-lm, prefill 1.4–2.8× faster. `mlx-lm-generate` / `mlx-lm-chat` already on
PATH. This is the offline *inference* half — already real today.

**Local training:** `mlx-lm-train` is **on PATH** and the stack documents
training (`mlx-go-lm/docs/QUICKSTART_TRAINING.md`; README §"Train or fine-tune").
Sibling harnesses add methods: `mlx-go-rl` (GRPO RL), `mlx-go-sdpo`
(self-distilled policy optimization), `mlx-go-distill` (on-policy distillation),
`mlx-go-rlsd` (self-distilled RLVR). All MLX-native, Apple-silicon-first.

**SIA weights mode (verified in source):** `mlx-go-sia` already models weight
self-improvement:

- `Focus = FocusWeights` ("weights") "rewrites an RL training script (train.py)
  that tunes model weights" (`layout.go:52-53`).
- Per-gen artifact is `train.py` (`NameTrainScript`, `layout.go:17`); weights mode
  logs to `train_stdout.log` (`layout.go:116-117`).
- `RunOptions.TrainingSandbox` is "weights focus only" (`orchestrator.go:53`);
  weights mode supports **feedback-agent early-stop** (`orchestrator.go:110-111`).
- Note (`doc.go:26`): the reference's Python training ecosystem (venv/pip/tinker)
  is *not* copied — the Go port abstracts the engine behind the runner/executor
  seams. So "local training" means **wiring SIA's `train.py` execution to the Go
  MLX training stack**, not running the reference's tinker pipeline.

## The totally-local loop (target architecture)

```
            ┌─────────────────────────── all on-device, no network ───────────────────────────┐
 meta/feedback engine            target = train.py                 evaluator
 (AgentRunner)                   (TargetExecutor)                  (Evaluator)
 ─ option A: claude (local-ish,  ─ runs the gen's train.py under   ─ scores the *trained* model:
   needs net; pre-record)          the MLX training stack            run held-out eval via
 ─ option B: pi-mlx wrapper        (mlx-lm-train / mlx-go-rl GRPO)   mlx-lm-generate, compute the
   on mlx-lm-generate (offline)  ─ produces updated weights +        task metric, write results.json
                                    train_stdout.log                ─ FocusWeights early-stop signal
            └──────────────────── Focus = FocusWeights, TrainingSandbox set ───────────────────┘
```

Each generation: the feedback agent rewrites `train.py` (e.g. tweaks the RL
reward, LR, LoRA rank, data mix); the executor runs it to produce updated
weights; the evaluator scores the *trained* model on a held-out task; the metric
feeds back. The number that "goes up" is **model quality after on-device
training**, not just kernel speed.

## BLOCKER (verified): the weights prompt targets a Python runtime we don't have

The NLM review flagged this; verification refines it:

- `prompts_weights.go:45` — weights mode "substitut[es] the task fields into the
  **embedded template**" and the per-gen artifact is `train.py` (`layout.go:17`).
- `doc.go:26` — the Go port deliberately does **not** copy "the reference's Python
  ecosystem (venv, pip, the Claude Agent SDK, **tinker**)."

So the agent is steered to write a `train.py`, but **no Python MLX/tinker runtime
exists on the demo machine** — and we want to drive the *Go* `mlx-lm-train`. We
must close this seam. Two options (decide before building):

1. **Custom weights `TargetExecutor`** (preferred): the executor treats `train.py`
   as a *spec*, parses the chosen hyperparameters (LR, LoRA rank, num-layers,
   data mix), and translates them into a shell call to
   `mlx-lm-train --model %MODEL% --data %WORKDIR% --fine-tune-type lora ...`. The
   agent's "code" becomes a structured intent the Go stack executes.
2. **Override the weights prompt/profile** so the agent emits an `mlx-lm-train`
   invocation (or a tiny config the executor runs) directly, instead of a
   Python/tinker `train.py`.

(NLM stated the prompt "embeds the Tinker-Cookbook verbatim"; verified that's the
*reference's* behavior described in the mlx-go-sia README, while the Go port's
`doc.go` says tinker is **not** ported. The risk — agent writes for a missing
runtime — is real regardless; the fix above stands.)

## CLOSED DESIGN: the `MLXTrainExecutor` bridge (the blocker fix, detailed)

This is the crux that was "a sentence" — now designed against the **verified**
contracts on both sides.

**SIA side (verified `target.go`):** `TargetExecutor` is an *interface*
(`RunTarget(ctx, TargetRequest) (TargetResult, error)`). `TargetRequest` carries
`AgentPath` (= `train.py` in weights mode, `orchestrator.go:175`), `DatasetDir`
(ro), `WorkingDir` (rw, the gen dir), `StdoutLog`. The stock `ExecTargetExecutor`
runs `AgentPath` under an interpreter with the fixed contract
`--dataset_dir <ro> --working_dir <rw>` (`target.go:69`). **Because it's an
interface, we don't have to make `train.py` executable Python — we supply our own
executor.**

**`mlx-lm-train` side (verified flags):** `-model`, `-data <dir with
train/valid/test.jsonl>`, `-fine-tune-type lora`, `-lora-rank`, `-learning-rate`,
`-iters`, `-batch-size`, `-num-layers`, `-adapter-path <out>`.

**The bridge — `MLXTrainExecutor` (implements `sia.TargetExecutor`):**

```go
// MLXTrainExecutor treats the agent's train.py as a DECLARATIVE training spec
// and runs it via the Go mlx-lm-train binary — no Python runtime needed.
type MLXTrainExecutor struct {
    TrainBin  string // "mlx-lm-train" (on PATH)
    BaseModel string // e.g. "mlx-community/Qwen3-0.6B-4bit"
    DataDir   string // train/valid(/test) jsonl; test stays OUT (evaluator owns it)
    Defaults  TrainSpec // safe defaults; the agent's spec overrides a whitelist
}

func (e *MLXTrainExecutor) RunTarget(ctx context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
    spec, err := parseTrainSpec(req.AgentPath) // read train.py as a spec (below)
    // on parse failure: TargetResult{Success:false, ErrorMsg:...} (NOT a Go error)
    args := e.buildArgs(spec, req)             // -> mlx-lm-train flags
    // run mlx-lm-train, stream combined output to req.StdoutLog,
    // write adapter weights into req.WorkingDir (-adapter-path),
    // return TargetResult{Success: exit==0, ...}
}
```

**How `train.py` becomes a spec (two options, pick #1):**

1. **Hyperparameter-block contract (preferred):** the weights *prompt* (overridden,
   below) instructs the agent to emit `train.py` containing a single fenced config
   the executor parses — a small whitelisted set: `learning_rate`, `lora_rank`,
   `num_layers`, `iters`, `batch_size`, `fine_tune_type`, `data_mix`. Anything
   outside the whitelist is ignored. This is robust, ungameable on the
   correctness axis, and keeps the agent's lever exactly the training knobs.
2. **Parse-Python-AST fallback:** extract assignments from a real `train.py`. More
   fragile; only if #1 proves too constraining.

**Prompt override (the other half of the blocker fix) — VERIFIED, needs a code
seam, not config.** The weights templates are `//go:embed`'d *verbatim*
(`prompts_weights.go:23-32`: `weights_meta_prompt.txt`, `weights_feedback_prompt.txt`,
`rl_guide.txt`) and built by unexported funcs (`buildWeightsMetaPromptSandbox`,
`buildWeightsFeedbackPrompt`). They are **not swappable via a profile** — changing
the instruction to "emit an mlx-lm-train hyperparameter block, not a tinker
pipeline" requires either:
- (a) editing the embedded `weights_meta_prompt.txt` / `rl_guide.txt` in our copy
  of the module (cleanest for the demo), or
- (b) adding a small exported override seam to `prompts_weights.go`.

**Sandbox constraint — VERIFIED, real wrinkle.** `TrainingSandbox` has only two
values (`prompts_weights.go:13-15`): `SandboxModal` ("modal", cloud, the empty
default) and `SandboxFusion` ("sandboxfusion", a local SandboxFusion *service*).
`sandboxInstruction(s)` appends a block telling the agent how to run its code in
that sandbox. **Neither is "run training directly on this host."** Two ways to
handle it:
- **Bypass (preferred, no new sandbox):** our `MLXTrainExecutor` treats `train.py`
  as a *spec* and runs `mlx-lm-train` itself, so the sandbox instruction in the
  prompt is irrelevant — we just pick the sandbox value whose prompt block is
  least misleading and ignore it operationally. Combined with the prompt override
  above, the agent is told to emit hyperparameters, not sandbox-run code.
- **Add `SandboxLocal`:** a small code addition (a new `TrainingSandbox` value +
  a `weights_sandbox_local.txt` instruction that says "your train spec runs via
  the local Go MLX trainer"). Cleaner story, slightly more code.

For the hackathon, **bypass + edit the embedded prompt** is the minimal path;
`SandboxLocal` is the tidy version if time allows.

## Remaining build pieces

2. **Weights `Evaluator` (`WeightsEvaluator`):** loads the trained adapter from
   `WorkingDir` onto `BaseModel`, runs a held-out eval via `mlx-lm-generate`,
   computes the task metric, writes `results.json`. Honesty discipline: the
   **held-out `test.jsonl` lives outside the agent's reach** (see honesty section)
   — the agent's `DataDir` has train/valid only.
3. **Glue:** `o := sia.NewOrchestrator(meta, &MLXTrainExecutor{...})`;
   `o.Eval = &WeightsEvaluator{...}`; `o.Run(ctx, RunOptions{ Focus: FocusWeights,
   TrainingSandbox: ..., MaxGen, ... })` (injection seam per `00-master.md`).
   **Confirm `TrainingSandbox`'s required value/constraints from source before
   building** — it's "weights focus only" and may gate execution.
4. **Task + model:** the locked recipe below (Qwen3-0.6B-4bit + LoRA + narrow task).

## Locked feasibility recipe (verified against the training quickstart)

The crux: an on-device weight update must move a held-out metric *visibly within
one generation*, fast enough to watch. Decided, grounded in
`mlx-go-lm/docs/QUICKSTART_TRAINING.md`:

- **Model:** `mlx-community/Qwen3-0.6B-4bit` (quickstart's first example; Qwen3
  0.6B/1.8B/4B + 4/8-bit are the tested set).
- **Method:** **LoRA** (`--fine-tune-type lora`, `--lora-rank 8`, tune
  `--num-layers`). **Not GRPO/RLVR** for the demo (richer story, far higher risk).
  ⚠️ **QLoRA is explicitly unsupported** (quickstart) — LoRA on a 4-bit *base*, not
  quantized-LoRA.
- **Footprint:** Peak mem ~**4.27 GB stable** across 100 iters (quickstart) — fits
  the laptop comfortably; a ~50–100 step update is seconds-to-low-minutes.
- **Task:** a narrow classification/text task (e.g. a small LawBench slice) where
  tweaking LR / `--num-layers` makes the loss drop or a val metric tick up within
  ~50 steps — small enough to be legible, real enough to be honest.

## Thermal sequencing (HIGH, from review): don't let P6 ruin P3

A live local training run heats the GPU; the roadmap warns thermal drift fakes
benchmark wins, and P3's money-shot needs a clean thermal baseline. **Do not run a
live P6 training loop adjacent to the live P3 inference benchmark.** Plus, a
3–4 min demo has no room for a live training loop.

## Tiered fallback (what we actually demo) — DECISION

Per the review, commit to **T0 live + T1 recorded**:

- **Tier 0 (LIVE, certain): local inference.** SIA loop with `pi-mlx` engine +
  harness focus, fully offline (P2 + P3). The live floor.
- **Tier 1 (RECORDED, target): local LoRA fine-tune, single gen.** One
  `mlx-lm-train` LoRA run inside a SIA `FocusWeights` generation; the recording
  shows the held-out val metric move. Proves on-device *training* end-to-end
  through SIA — **recorded** to protect stage time *and* P3's thermal baseline.
- **Tier 2 (stretch, recorded only if it lands): multi-gen local weight
  self-improvement.** ≥3 gens, feedback agent edits the training config, held-out
  metric climbs. The full thesis, locally.
- **Tier 3: explicitly out of scope for the hackathon** (local GRPO/RLVR via
  `mlx-go-rl`/`mlx-go-rlsd`) — noted as future work only.

The demo claims exactly the tier we land — no overclaiming.

## Honesty: held-out data outside the agent's reach (HIGH, from review)

Same discipline as P3's evaluator-sandbox isolation, applied to weights — the
paper's coupled-Goodhart warning bites harder here because the *model itself* can
memorize the eval split:

- The `WeightsEvaluator` keeps `test.jsonl` in a **read-only dir entirely outside
  the agent's `WorkingDir`**, checksummed.
- At eval time it mounts the agent's trained weights from `WorkingDir` but streams
  eval prompts from the pristine external set. The agent can read/train on its
  *training* data only and **never sees the held-out rows** — so a metric gain is
  generalization, not leakage.

## Local vs cloud (the paired demo with P5)

| | P6 local (Apple/MLX) | P5 Nebius cloud |
|---|---|---|
| Inference | `mlx-go-lm` on-device | hosted open models |
| Training | `mlx-lm-train`/`mlx-go-rl`, laptop | H200 credits, larger models |
| Story | "no network at all" | sponsor compute, scale |
| SIA focus | FocusWeights local | FocusWeights / harness cloud |

Showing both = the complete compute story: it runs offline on a laptop *and*
scales to sponsor GPUs with the same loop and config.

## Open questions for the NLM design review (iterate here)

- **Feasibility crux:** what (model size, method, task, step budget) makes an
  on-device weight update move a held-out metric *visibly within one generation*
  on this Mac, fast enough to watch? LoRA on a ≤3B model? Which task?
- Is wiring SIA `train.py` → `mlx-lm-train` (LoRA) the right first target, or is
  `mlx-go-rl` GRPO close enough to demo and a stronger story?
- For a 6pm demo: is Tier 1 (single local LoRA gen) realistic to land solidly, or
  should we plan the demo around Tier 0 + a recorded Tier 1?
- Honesty: how do we keep the weights `Evaluator`'s held-out data out of the
  agent's reach so a "trained" win isn't memorization/leakage?
- Does `TrainingSandbox` (weights-only) impose constraints we must design around?
  (Resolve from `mlx-go-sia` source before building.)
- Thermal/time budget: a real training run heats the laptop — does that perturb a
  *subsequent* inference benchmark (P3)? Sequencing on stage.
