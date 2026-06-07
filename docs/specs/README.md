# SIA Hackathon — Design Specs

Process artifacts that guide the build and the NotebookLM design review. Not
shippable module content.

- [`00-master.md`](00-master.md) — thesis, ecosystem assets, verified
  `mlx-go-sia` API seams, the four projects, demo plan, risks.
- [`01-paperbench.md`](01-paperbench.md) — **flagship (Research)**: score an
  agent implementing a paper-roadmap prototype via the `coverage-map` rubric,
  wired as a SIA `Evaluator`.
- [`02-pi.md`](02-pi.md) — **(Framework)**: minimal Go agent `AgentRunner`,
  `claude`-CLI then local `mlx-go-lm` backend, for offline runs.
- [`03-mlx-inference-opt.md`](03-mlx-inference-opt.md) — **money-shot (Applied)**:
  SIA loop optimizing an `mlx-go-lm` inference path; tokens/sec `Evaluator`.
- [`04-self-mod.md`](04-self-mod.md) — **kicker (Framework, stretch)**: loop
  points at `mlx-go-sia` itself; test-gate `Evaluator`.
- [`05-nebius.md`](05-nebius.md) — **compute (cloud)**: Nebius Token Factory
  provider (built-in, verified) + the `%BASEURL%`/`%APIKEY%` token fix; sponsor
  open-model engine. Working now.
- [`06-local.md`](06-local.md) — **compute (local), the goal**: totally-local
  self-improvement — on-device inference *and* weight training via
  `mlx-lm-train` + SIA `FocusWeights`. Tiered fallback. Iterated with NotebookLM.
- [`07-metal-kernel-opt.md`](07-metal-kernel-opt.md) — **Applied, candidate
  money-shot**: SIA loop iterates an **MLX Metal kernel** source (JIT-compiled via
  `FastMetalKernel`), correctness-gated, µs/op scored. SIA's flagship GPU-kernel
  domain on Apple silicon. Likely supersedes P3 (which becomes the fallback).

## Demo priority

P1 + P3 are must-haves. P2 is the offline-credibility stretch. P4 is a 30s
kicker, built only if green and cheap.

## Notebook

Design iteration runs in the NotebookLM notebook `SIA hackathon — design
2026-06-06`. The notebook is a *grounded reviewer*: every finding is verified
against this filesystem before any spec is changed.
