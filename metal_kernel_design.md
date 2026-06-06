# P7 metal-kernel-opt — design (Agent 4)

A SIA loop that iterates an **MLX Metal kernel source**: the agent rewrites the
kernel each generation, MLX JIT-compiles it, correctness is gated against an
external Go reference, and the score is throughput (ops/sec). The number that
goes up is real GPU-kernel speed on Apple silicon.

## Files this adds (all new; touches no shared file)

- `metal_kernel.go`     — starter kernel source, fixed-seed inputs, Go golden oracle, RMSNorm spec
- `metal_executor.go`   — `MetalKernelExecutor` (`sia.TargetExecutor`): JIT-compile + run the gen's source
- `metal_benchmarker.go`— `KernelBenchmarker` (`sia.Benchmarker`): Correct / RunCandidate / RunBaseline / Unit
- `cmd/metalopt/main.go` — self-contained orchestrator wiring
- `*_test.go`           — roundtrip, compile-failure→Success:false, oracle, benchmarker

## Chosen starter kernel: fused RMSNorm

`y[r,c] = x[r,c] / sqrt(mean_c(x[r,c]^2) + eps) * w[c]`, over `[rows, dim]` f32.

Why RMSNorm over softmax / element-wise:
- **Exact, external oracle.** The golden output is computed in pure Go `float64`
  (`metal_kernel.go`), independent of MLX — the agent cannot influence it. Frozen
  relative tolerance covers GPU float-order differences.
- **Real, legible headroom.** The seeded gen-0 kernel is deliberately naive: each
  thread recomputes the whole row's sum-of-squares (O(dim^2) per row, one thread
  per row, no vectorization). The optimization path is obvious and large —
  threadgroup-parallel reduction, `float4` vectorization, single-pass — so the
  ops/sec genuinely climbs across generations.
- **On-thesis.** RMSNorm is a real LLM kernel; mlx-go ships a fused production
  `fast.RMSNorm` (the op we are racing) and `mlxvet` flags unfused norms as a
  fusion opportunity. Fused-MoE (mlx-go-lm) is the documented stretch target.

The agent's only lever is the kernel **source** (FocusHarness). Grid/threadgroup
choice is also the agent's to tune, declared in a small JSON sidecar next to the
source (see the executor contract) so the harness stays frozen.

## Verified substrate (probed against live MLX on this host, 2026-06-06)

The brief said the kernel API is "the `fast` package"; the real importable path
is **`github.com/tmc/mlx-go/mlx/fast`** (`mlx-go/fast` does not exist).

- `fast.MetalKernel(name, inNames, outNames, source) (*Kernel, error)` — ctor.
- `(*Kernel).Apply(inputs []*mlx.Array, grid, threadGroup [3]int, outShapes [][]int, outDtypes []mlx.Dtype) []*mlx.Array`.
- `mlx.NewArray[T](data, shape...)`, `mlx.ToSlice[float32]`, `mlx.Eval(...)`, `mlx.Synchronize()`, `mlx.MetalIsAvailable()`.

**Compile-failure semantics (probed, not assumed):** a malformed MSL string does
**not** error at `MetalKernel()` and does **not** panic at `Apply()`. MLX compiles
the kernel lazily; the build error surfaces when the outputs are evaluated:
`mlx.Eval(outputs...)` returns `ArrayEval: [metal::Device] Unable to build metal
library from source`. So the executor:
1. builds via `fast.MetalKernel` (catch ctor error),
2. `Apply` inside a `recover()` (defense-in-depth — launch errors *can* panic),
3. **`mlx.Eval` the outputs and check its error** — the real compile-failure gate.
Any failure → `TargetResult{Success:false, ErrorMsg:<compiler error>}`, no Go
error, so the feedback agent sees the compiler message and fixes it next gen.

## Honesty / anti-Goodhart

- Golden inputs + outputs are computed in Go from a frozen seed at benchmarker
  construction, held outside the agent's WorkingDir; the agent cannot read or
  widen them. A faster-but-wrong kernel returns `ok=false` → `REVISE`.
- Throughput uses `mlx.Synchronize()` to force GPU completion before stopping the
  clock (an unsynced timer would measure dispatch, not compute).
- Agent 3's `ThroughputEvaluator` owns median-of-N + interleaved gen-0 baseline
  (cancels thermal/cache drift). We supply only the three runnables + `Unit()`.

## Wiring

```go
bench := sia.NewKernelBenchmarker(...)            // captures golden oracle (read-only)
o := sia.NewOrchestrator(meta, sia.NewMetalKernelExecutor(...))
o.Eval = sia.NewThroughputEvaluator(bench)        // agent 3's evaluator
o.Run(ctx, sia.RunOptions{Focus: sia.FocusHarness, MaxGen: 6, ...})
```

Unit: `ops_per_sec`. Sample.Throughput = `1e6 / median_µs` (higher = better).
