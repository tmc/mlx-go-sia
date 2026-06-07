# P3 — `mlx-inference-opt` (Applied track) — the money-shot

**One-liner:** Point the SIA loop at an `mlx-go-lm` inference path and let it
optimize for throughput, with a tokens/sec `Evaluator` — the visceral
"number goes up across generations" demo, echoing SIA's own GPU-kernel result on
Apple silicon.

## Why

This is the single most legible 60 seconds of the demo: one chart, tokens/sec
climbing gen-over-gen on a real local model. It mirrors the SIA paper's GPU-kernel
domain (the paper reports 12.4% faster kernels; marketing says 91.9% runtime
reduction) but on **Apple silicon via MLX**, which the paper does not cover —
a genuine new domain for the framework.

## The target the agent optimizes (LOCKED: decode-step sampler)

**Decision (from NLM review + stack README):** lock the target to a **decode-step
sampler kernel** (top-k / top-p / temperature). Rationale:

- **Exact correctness oracle:** with a fixed seed, the emitted token sequence must
  match the golden exactly — a rigid categorical pass/fail, no fuzzy tolerance.
- **Avoids an already-optimized ceiling:** `mlx-go-lm` prefill is *already*
  1.4–2.8× faster than Python mlx-lm (stack README), and the roadmap warns prefill
  timing is volatile — low headroom, high noise. The sampler path is a better
  speedup-headroom : oracle-cleanliness ratio for a ≤8-gen claude-driven loop.

Fused ops / prefill micro-paths are explicitly *not* the target (floating-point
drift makes their correctness ambiguous).

**Hard requirement:** the agent may only change the implementation, never the
correctness contract. The evaluator runs the correctness gate *before* timing,
from a harness the agent cannot touch (see Sandbox isolation below).

## Scoring model

Categorical gate + advisory number, same discipline as P1:

- **Verdict:** `PASS` iff the correctness gate passes (output matches the
  golden within tolerance). A faster-but-wrong implementation is `REVISE`,
  full stop — speed never overrides correctness.
- **Advisory score:** median tokens/sec (or μs/op) over N runs, **only**
  reported when correctness passes. This is the demo curve.

## Evaluator sandbox isolation (BLOCKER fix from NLM review)

If the correctness oracle, golden outputs, and timing scripts live inside the
agent's `WorkingDir`, the agent can cheat — widen the float tolerance, hardcode
the golden output, or no-op the timing loop. So:

- The `ThroughputEvaluator` keeps its **golden oracle, test inputs, and timing
  harness in a read-only directory entirely outside the agent's `WorkingDir`**,
  checksummed from a pristine copy each cycle.
- At evaluation time the evaluator **injects/links the agent's inference routine
  into the protected harness** (the agent supplies only the routine under test).
- The agent never has read or write access to the test files. The correctness
  contract is owned by the evaluator, not discoverable or editable by the agent.

This is the same honesty discipline as P1's honest-recompute: the verifier must
be outside the optimizer's reach (cf. the SIA paper's coupled-Goodhart warning).

## The `Evaluator` (the contribution)

```go
// ThroughputEvaluator runs a correctness gate then a benchmark, writing the
// median throughput to results.json. Implements sia.Evaluator.
type ThroughputEvaluator struct {
    CorrectnessCmd string // exits 0 iff the gen's impl is correct (golden compare)
    BenchCmd       string // prints a parseable tokens/sec or ns/op
    Runs           int    // N (median, not single hot run)
    Cooldown       time.Duration
}

func (e *ThroughputEvaluator) Evaluate(ctx context.Context, genDir string) (sia.EvalResult, error) {
    // 1. run CorrectnessCmd; if non-zero -> results.json{verdict:REVISE, reason}, EvalSuccess
    // 2. else run BenchCmd N times with Cooldown between, parse, take median
    // 3. results.json{verdict:PASS, tokens_per_sec, runs:[...], correctness_ok:true}
}
```

## Benchmark methodology (the trap the roadmap warns about)

`paper-roadmap` has an explicit `apple-silicon-benchmark-methodology` row:
"Thermal state, memory pressure, run order, and MoE cache hysteresis can create
false Apple Silicon benchmark wins." Median-of-N + cooldown alone is **not enough**
to defend a stage demo. The most likely false win: **environmental drift** — gen-N
runs after gen-(N-1) and benefits from a warm cache or suffers gradual thermal
throttle, so a baseline shift reads as an agent improvement.

Mandatory controls (the evaluator owns all of these):

- **Mandatory interleaved gen-0 baseline:** every evaluation cycle re-runs the
  gen-0 (baseline) code *immediately alongside* the gen-N code, under identical
  conditions. **The demo chart plots the gen-N − gen-0 delta measured at that
  moment, not raw tokens/sec** — this mathematically cancels thermal/cache drift.
  (This was "optional null-matrix"; it is now required.)
- **Median of N**, not a single run; report the distribution (min/median/max).
- **Cooldown** between runs; fixed run order; identical warmup.
- Same model, prompt, seed, thread/stream config across gens.

A demo where the number went up only because the laptop warmed differently is a
loss; the methodology *is* part of the contribution.

## Wiring into the loop

- `NewOrchestrator(metaRunner, targetExecutor)`; `metaRunner` = `claude` (or
  `pi`). The target the agent edits is the inference routine; `TargetExecutor`
  builds+runs it.
- Inject `ThroughputEvaluator` (same open API question as P1: where the
  evaluator attaches).
- `RunOptions{ MaxGen: 5–8, Focus: FocusHarness, ... }` — harness focus
  (rewrite the code), not weights.
- Per-gen `results.json` → the throughput chart.

## Deliverables

1. The chosen inference routine isolated with a golden correctness oracle and a
   stable benchmark command.
2. `ThroughputEvaluator` implementing `sia.Evaluator` with the methodology
   controls above.
3. The loop run producing the climbing-throughput series + the chart script
   (shared with P1's plotter).

## Open questions for the NLM design review

- Which routine has the best **correctness-oracle + speedup-headroom** ratio?
  Sampler vs a fused op vs prefill micro-path?
- Is the optimization space rich enough that a `claude`-driven agent finds real
  wins in ≤8 gens, or will it plateau at gen-1?
- Methodology: is median-of-N + cooldown enough, or do we need the full
  null-matrix / cooldown-order matrix to defend the number on stage?
- Risk that the agent "optimizes" by weakening the correctness tolerance — is the
  correctness gate truly independent of the code under test?
- Does FocusWeights (real weight updates via Nebius/MLX training) add a stronger
  story than FocusHarness here, or is it scope we cut for the demo?
