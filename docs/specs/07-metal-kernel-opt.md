# P7 — `metal-kernel-opt` (Applied) — SIA optimizing Metal kernels

**One-liner:** A SIA-style loop that iterates an **MLX Metal kernel** — the agent
rewrites the kernel *source*, MLX JIT-compiles it, correctness is gated against
the reference op, and the score is µs/op (or tokens/sec). The fullest echo of
SIA's flagship domain (GPU-kernel optimization) on **Apple silicon / Metal**,
which the SIA paper does not cover.

**Status: candidate money-shot — likely SUPERSEDES P3.** P3 optimized a Go
*sampler*; P7 optimizes an actual *Metal kernel* (what the GPU runs). Same demo
shape (number-goes-up across generations), much stronger story and more on-thesis.
Recommend: make P7 the Applied flagship, keep P3 as the lower-risk fallback.

## Why this is the right idea

- **SIA's own headline domain.** The paper's marquee result is GPU-kernel
  optimization (12.4% faster kernels / "91.9% runtime reduction" marketing).
  Doing the *same thing* on Metal/Apple silicon is the most direct, legible
  "we reproduced and extended SIA" claim.
- **It's a real lever, not a toy.** The agent edits Metal Shading Language source
  that MLX JIT-compiles and runs — genuine kernel engineering, auto-measured.
- **The roadmap already frames it.** `autonomous-metal-fusion`
  ("verifier-driven agents can discover and reject MLX Go Metal fusion
  candidates") and `mlxvet-static-fusion-analysis` are existing roadmap rows —
  P7 is the *active* version of that research direction.

## Verified substrate (this is buildable)

- **Custom Metal kernel API in Go (verified):** MLX exposes
  `mlx_fast_metal_kernel_apply` — supply Metal source + a `FastMetalKernelConfig`,
  MLX **JIT-compiles and runs it**. Wrapped zero-alloc in the Go bindings as
  `FastMetalKernelApplyVec` / `FastMetalKernel` / `FastMetalKernelConfig`
  (`mlx-go/internal/mlxc/fastfunc.go:114-120`, surfaced via the `fast` package).
- **A real kernel to optimize (verified):** `mlx-go-lm` ships a production **fused
  MoE Metal kernel** — `mlxlm/llm/models/metal_kernel.go`,
  `qwen3_next_flashmoe_metal_darwin.go`, design doc
  `mlx-go-lm/docs/fused-moe-metal-design.md`. It has an obvious correctness
  reference: the **unfused** MoE path.
- **A feedback signal source (verified):** `mlxvet` statically flags fusion
  opportunities (`fast.SDPA`, `fast.RMSNorm`, unfused norms) — the feedback agent
  can consume its output as hints.

## The loop

```
 meta/feedback engine        target = the Metal kernel              evaluator
 (AgentRunner)               (TargetExecutor)                       (ThroughputEvaluator, P3-style)
 ─ claude (pre-rec) or       ─ takes the gen's kernel SOURCE,       ─ 1. CORRECTNESS GATE: run kernel vs
   pi-mlx (offline)            JIT-compiles via MLX                    reference op on fixed inputs/seed;
                             ─ if compile fails -> Success=false,      exact/within-tol match or REVISE
                               error fed back (agent fixes it)      ─ 2. BENCH: median µs/op over N, with
                                                                       interleaved gen-0 baseline (anti-thermal)
                                                                     ─ results.json -> feedback
```

Each generation the agent rewrites the Metal kernel (tiling, vectorization,
threadgroup memory, fusing ops, dtype tricks); the executor JIT-compiles it; the
evaluator gates correctness then measures µs/op; the metric feeds back. **The
number that goes up is real GPU-kernel speed.**

## What we build

1. **A self-contained kernel target.** Start *simpler than fused-MoE* for demo
   reliability: a single well-known op as a Metal kernel with a trivial
   correctness oracle (e.g. a fused RMSNorm, a softmax, or an element-wise fused
   activation) where a fixed-seed input has an exact reference output. The
   fused-MoE kernel is the *stretch* target (richer story, harder oracle).
2. **`MetalKernelExecutor` (implements `sia.TargetExecutor`):** reads the gen's
   kernel source from the working dir, builds a `FastMetalKernel` via the `fast`
   API, runs it. **Compile/JIT failure → `TargetResult{Success:false, ErrorMsg}`**
   (a Go error only if the executor itself can't run) so the agent gets the
   compiler error as feedback and fixes the kernel next gen.
3. **`ThroughputEvaluator` (reuse P3's, verbatim discipline):** correctness gate
   BEFORE timing; median-of-N; **mandatory interleaved gen-0 baseline** (plot the
   gen-N − gen-0 delta to cancel thermal drift); golden oracle in a **read-only
   dir outside the agent's working dir** so the agent can't widen tolerance.
4. **Glue:** `o := NewOrchestrator(meta, &MetalKernelExecutor{...})`;
   `o.Eval = &ThroughputEvaluator{...}`; `RunOptions{ Focus: FocusHarness, MaxGen:
   5–8 }` (harness focus = rewrite the kernel source).

## Honesty / anti-Goodhart (same threat model as P1/P3/P6)

- Correctness gate is **independent and external** — golden inputs/outputs the
  agent can't read or edit; a faster-but-wrong kernel is `REVISE`, never a win.
- Report median + distribution; interleaved baseline cancels thermal/cache drift.
- The agent's only lever is the kernel source; the eval harness is frozen.

## Risks & mitigations

- **JIT compile churn:** the agent will write kernels that don't compile. That's
  *fine* — it's the feedback signal — but bound it: per-gen compile timeout,
  cap iterations, and seed gen-0 with a known-compiling baseline kernel.
- **Correctness of Metal numerics:** GPU float order differs; use a sensible
  tolerance for the reference compare (documented, frozen, agent can't touch it).
- **Headroom:** a naive gen-0 kernel must have real room to improve. Seed gen-0
  with a *deliberately unoptimized* but correct kernel so gains are visible.
- **Demo time/thermal:** same as P3 — the live benchmark needs a clean thermal
  baseline; the interleaved gen-0 control protects it. Pre-record as insurance.

## Relationship to the other projects

- **vs P3:** P7 is the stronger version of the same money-shot; if P7 lands, it
  *is* the Applied demo and P3 becomes the safety net.
- **Shares** the `ThroughputEvaluator` + chart with P3, and the `pi-mlx` offline
  engine with P2 — low marginal cost given P3 is already specced.
- **Same honesty discipline** as P1/P6 (external frozen verifier).

## Open questions for the NLM design review

- Which **starter kernel** has the best (trivial correctness oracle) × (real
  speedup headroom) for a ≤8-gen claude loop — fused RMSNorm? softmax? a small
  element-wise fusion? (Fused-MoE is the stretch.)
- Can a `claude`-driven agent write *compiling, improving* Metal source in a few
  gens, or does it need the kernel scaffolded so it only edits the inner loop?
- Is JIT-compile latency per gen fast enough to keep the loop watchable?
- Does P7 fully replace P3 in the demo, or do we show both (P3 as the safe
  opener, P7 as the wow)?
