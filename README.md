# mlx-go-sia

Go port of [SIA](https://github.com/hexo-ai/sia) (Self-Improving AI with Harness
& Weight Updates, Hebbar et al. 2026, [arXiv:2605.27276](https://arxiv.org/abs/2605.27276)).

SIA runs a three-agent loop over successive generations:

- a **Meta-Agent** seeds an initial target agent from a task specification,
- a **Target Agent** runs the task and records an execution trajectory,
- a **Feedback-Agent** reads the trajectory and evaluation results and rewrites
  the target agent for the next generation.

## What It Covers

- the orchestration loop (`Orchestrator`) over `runs/run_{id}/gen_{n}/`
- the filesystem `Layout` (path/filename constants matching the reference)
- the meta and feedback prompt builders — byte-identical to the reference's
  golden-locked text (verified in `prompts_test.go` against fixtures generated
  from the Python reference)
- the execution-trajectory format and the single- vs multi-trajectory detection
  rule (`LoadExecution`)
- the JSON `Provider` / `MetaAgentProfile` / `TargetAgentProfile` configuration
  model, with the built-in default providers and profiles the reference ships
- `harness` and `weights` improvement focus (the weights-mode RL prompt embeds
  the reference's Tinker-Cookbook integration guide verbatim)

## What Is Abstracted

The reference's Python ecosystem is not copied. The engine that runs the
meta/feedback agent is the `AgentRunner` interface — the same seam SIA exposes as
"agent impls":

- `ExecRunner` shells out to an external agent CLI (e.g. `claude`), passing the
  prompt on stdin and running it in the generation's working directory.
- `FakeRunner` is a scripted engine for deterministic tests and examples.

The target agent is run by a `TargetExecutor` (`ExecTargetExecutor` runs it under
a configurable interpreter with the fixed `--dataset_dir` / `--working_dir`
contract), and scoring is a pluggable `Evaluator` (`ExecEvaluator` runs the
task's `evaluate.py`). No Python venv or pip is created — dependency setup is the
operator's responsibility.

## Run

```bash
# Dry run: set up the run + write the meta prompt, with a no-op engine/target.
go run ./examples/core/sia -task-dir ./tasks/mytask -max-gen 3 -run-id 1 -dry-run

# Live run: drive an external agent CLI as the meta/feedback engine, and run the
# target agent under python3 with the fixed CLI contract.
go run ./examples/core/sia -task-dir ./tasks/mytask -max-gen 5 -run-id 1 \
    -agent-cmd claude -agent-args '-p,--model,%MODEL%' \
    -interpreter python3
```

`%MODEL%`, `%MAXTURNS%`, `%WORKDIR%`, `%BASEURL%`, `%APIKEY_ENV%`, and `%APIKEY%`
in `-agent-args` are substituted per invocation. The last three resolve from the
engine profile's provider, so an OpenAI-compatible engine CLI can be pointed at a
provider without hardcoding the endpoint:

```bash
# Drive the meta/feedback engine against Nebius Token Factory (OpenAI-compatible).
export NEBIUS_API_KEY=...
go run ./examples/core/sia -task-dir ./tasks/mytask -max-gen 1 -run-id 1 \
    -meta-agent-profile kimi-nebius-meta \
    -target-agent-profile qwen-nebius-target \
    -agent-cmd oai-agent \
    -agent-args '--base-url,%BASEURL%,--model,%MODEL%' \
    -interpreter python3
```

For an OpenAI-compatible engine provider with a base URL, the CLI also exports
`OPENAI_BASE_URL` and `OPENAI_API_KEY` (mirrored from the provider's `api_key_env`)
into the engine subprocess, so a CLI that reads those env vars needs no extra
flags. Built-in providers include `anthropic`, `openai`, `gemini`, `nebius`
(Nebius Token Factory), and `together`; built-in Nebius profiles are
`kimi-nebius-meta`, `qwen-nebius-target`, `gptoss-nebius-target`, and
`kimi-nebius-target`. (The default `claude` engine impl requires an `anthropic`
provider; use a non-claude meta profile to drive a non-Anthropic engine.)

Artifacts land under `runs/run_{id}/gen_{n}/`:

- `target_agent.py` — the agent for that generation (`train.py` in weights mode)
- `agent_execution.json` or `agent_execution/execution_q*.json` — the trajectory
- `improvement.md` — the feedback rationale (gen 2+)
- `meta_agent_prompt.txt` / `feedback_agent_prompt.txt` — the saved prompts
- `results.json` — evaluation output (when an evaluator runs)

`runs/run_{id}/context.md` accumulates the cross-generation summary the
feedback agent reads before each rewrite.

## Task Layout

```
mytask/
├── data/
│   ├── public/
│   │   ├── task.md          # task description (SIA reads this)
│   │   ├── evaluate.py      # optional scorer (writes results.json)
│   │   └── ...              # inputs the agent may read
│   └── private/             # held-out eval data; never exposed to the agent
└── reference/
    ├── reference_target_agent.py   # the improvable seed
    └── SAMPLE_TASK_DESCRIPTIONS.md
```

A sibling `_shared/sample_agent_execution.json` supplies the trajectory format
shown to the meta agent.

## Library

```go
orch := sia.NewOrchestrator(engine, target) // engine: AgentRunner; target: TargetExecutor
res, err := orch.Run(ctx, sia.RunOptions{
    Layout: layout, Task: task, TaskFiles: taskFiles,
    MetaProfile: metaProfile, Target: targetProfile, Resolved: resolved,
    MaxGen: 5, Focus: sia.FocusHarness,
})
```

See the runnable package `Example` in `example_test.go`.

## Examples

The commands live under themed `examples/` directories. Each is a self-contained
`package main` runnable from the repo root with `go run ./examples/<group>/<name>`;
each has a `Command` doc comment with full usage.

| Command | Run | What it does |
|---|---|---|
| sia | `go run ./examples/core/sia` | The SIA loop CLI: meta-seed, run, evaluate, feedback-rewrite per generation. |
| metalopt | `go run ./examples/metal/metalopt` | SIA loop that rewrites an MLX Metal kernel; correctness-gated GPU throughput. |
| localtrain | `go run ./examples/weights/localtrain` | P6 totally-local weight-training demo; held-out gate REVISEs overfit. |
| sia-train | `go run ./examples/weights/sia-train` | Weights mode via a declarative `train.py` spec → `mlx-lm-train`. |
| leakguard-demo | `go run ./examples/weights/leakguard-demo` | Proves the held-out set is structurally unreachable by the training agent. |
| paperbench | `go run ./examples/paper/paperbench` | SIA loop with a `PaperEvaluator` that honest-recomputes a paper-roadmap rubric. |
| inferopt | `go run ./examples/inference/inferopt` | P3 sampler-optimization loop; tokens/sec gated on exact-token correctness. |
| siachart | `go run ./examples/inference/siachart` | Renders a run's throughput series (terminal sparkline+table or CSV). |
| sia-traindata | `go run ./examples/cloud/sia-traindata` | Renders recorded trajectories into token-level `train_data.jsonl`. |
| sia-tinker-train | `go run ./examples/cloud/sia-tinker-train` | Trains a LoRA on that data via a local localtinker coordinator. |
| siadash | `go run ./examples/dashboard/siadash` | Native macOS SwiftUI dashboard (`//go:build darwin`) for the weights loop. |

`examples/selfport/` is a script-driven self-port task (an agent ports a Python
reference to Go, graded by `go test`), not a Go command.

Design material lives under `docs/`: `docs/sia-rlsd-seam.go` is a `//go:build
ignore` design sketch of an RLSD training seam, and `docs/specs/` holds the
SIA design specification suite (`00-master.md` plus per-track specs).

## Fidelity

This is a faithful port of the reference's load-bearing contracts (the
orchestrator↔target CLI contract, the layout, the trajectory format + detection
rule, and the prompt text), not a byte-for-byte runtime clone. The agent engine,
target interpreter, and evaluator are abstracted behind interfaces so the loop
runs without a Python runtime. The prompt builders are checked byte-for-byte
against fixtures generated from the reference.

## Weights mode: local training data (post-hoc)

The reference's weights mode never tokenizes trajectories itself: the
orchestrator drops the raw trajectory JSON into the feedback prompt as text, and
in weights mode it emits a prompt instructing the *generated* `train.py` to
build a renderer with `tinker_cookbook` and train against the hosted Tinker
service. The `traindata` and `localtrain` packages provide a Go-native,
hosted-Tinker-free analog of that delegated pipeline. They are **additive**:
nothing in the orchestration core changes, and `package sia`'s import graph stays
dependency-free — the renderer/tinker edge lives only in these subpackages and
their commands.

- `traindata` renders a recorded `sia.Execution` into token-level
  `(token_ids, loss_mask)` samples via the target model's
  [renderer](https://github.com/tmc/mlx-go-experiments/tree/main/renderer), and
  writes them as JSONL — the line-delimited form a `train.py` reads.
- `localtrain` maps those samples onto the
  [localtinker](https://github.com/tmc/localtinker) `tinker.Datum` contract and
  runs `CreateLoRA → ForwardBackward → OptimStep → Save` against a local
  coordinator on MLX.

```sh
# 1. Render a finished run's trajectories into training data.
sia-traindata -run-dir ./runs/run_1 -all \
    -model Qwen/Qwen3-8B -tokenizer /models/Qwen3-8B -mask rl

# 2. Train a LoRA locally against a running localtinker coordinator.
#    (in another shell: localtinker serve -addr 127.0.0.1:8080 -home /tmp/lt-home)
sia-tinker-train -data ./runs/run_1/gen_3/train_data.jsonl \
    -base-model Qwen/Qwen3-8B -model-path /models/Qwen3-8B -rank 8
```

This is one of two coexisting weights-mode backends: `sia-tinker-train` drives
the renderer→tinker path above, while `sia-train` drives a separate
declarative-spec path that translates the generated `train.py` into
`mlx-lm-train` flags. The `train_data.jsonl` from `sia-traindata` feeds either.
