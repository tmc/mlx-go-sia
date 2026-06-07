# SIA Hackathon — Master Spec

**Event:** AGI House "Self-Improving AI (SIA) Agents", 2026-06-06, SF.
**Demo:** 6:00 PM, 3–4 minutes. **Awards:** 7:30 PM. **Prize:** Meta Ray-Bans per track.
**Sponsor compute:** Nebius (H200 + hosted Deepseek/Qwen/GLM/Gemma).
**Constraints:** demo must be **offline/local-capable**; solo/small team; `claude` CLI as
the agent engine of record (local `mlx-go-lm` model as a stretch); multiple
projects built by parallel agents, dispatched and integrated by the lead.

> This is a *process artifact*, not shippable module content. It lives under
> `docs/specs/` to guide the build and the NotebookLM design review. It is not
> what a `git clone` of any published module should treat as code.

## 1. The thesis (one sentence)

> **Turn the SIA self-improvement loop into a benchmark: how fast and how well
> can an agent implement a research paper — measured automatically, scored
> locally, on a real Go + MLX ML stack.**

Why it wins across tracks from one codebase:

- **Research:** a novel, automatic eval for self-improvement — paper-implementation
  velocity/quality scored by a machine-checkable rubric (not human judging).
- **Applied:** point the same loop at a concrete `mlx-go-lm` inference path and
  show throughput climbing — a real domain task with an objective number.
- **Framework:** the loop, evaluator seam, and a minimal local agent are
  contributions back to the SIA framework shape (here, its Go port).

## 2. What already exists (reuse, do not rebuild)

All paths under `/Users/tmc/go/src/github.com/tmc`.

| Asset | Role in this hackathon |
|---|---|
| `mlx-go` | Go bindings for Apple MLX (the array runtime). |
| `mlx-go-lm` | 81 LM architectures, offline inference; decode ≥ Python mlx-lm, prefill 1.4–2.8× faster. **The target model + the inference path we optimize.** |
| `mlx-go-examples/mlx-go-sia` | **Build-green Go port of SIA.** Full `Orchestrator` loop over `runs/run_{id}/gen_{n}/`, golden-locked meta/feedback prompts, pluggable `AgentRunner`/`TargetExecutor`/`Evaluator`. |
| `mlx-go-examples/paper-roadmap` | `coverage-map.jsonl`: ~14 paper-prototype ideas, each with a `claim`, a runnable `fast_check`, an `examples` list, a `gap`, and an `evidence_state` of booleans. **The eval dataset + machine-checkable rubric.** |
| `claude` CLI 2.1.157 | The meta/feedback agent engine (`AgentRunner` via `ExecRunner`). |
| Nebius credits | Real open-model / GPU path for a weight-update or heavier demo. |

**No Python MLX is installed on the demo machine.** That is deliberate leverage:
the offline-local demo path is Go + MLX, which most teams cannot match.

## 3. The mlx-go-sia API seams we build against (verified)

```go
type AgentRunner interface {
    Name() string                                   // agent-impl id, e.g. "claude"
    Run(ctx context.Context, req AgentRequest) error
}

type TargetExecutor interface {
    RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error)
}

type Evaluator interface {
    // Evaluate scores a generation dir; failure is reported via Status, not error.
    Evaluate(ctx context.Context, genDir string) (EvalResult, error)
}

func NewOrchestrator(meta AgentRunner, target TargetExecutor) *Orchestrator
func (o *Orchestrator) Run(ctx context.Context, opts RunOptions) (RunResult, error)
```

Supporting shapes (fields named exactly):

- `AgentRequest{ Model, Prompt, WorkingDir string; MaxTurns int; Provider Provider }`
- `TargetRequest{ AgentPath, DatasetDir, WorkingDir, StdoutLog string }`
- `TargetResult{ Success bool; Stdout, Stderr, ErrorMsg string }`
- `EvalResult{ Status EvalStatus; Reason, ResultsPath, Output string }`
- `EvalStatus ∈ {skipped, success, warning, error}`; `success` requires a
  `results.json`.
- `RunOptions{ Layout, Task, TaskFiles, MetaProfile, Target, Resolved,
  MaxGen, MaxTurns, Focus, TrainingSandbox, MaxLogSize, RunConfig }`
- `RunResult{ Focus; Generations []GenResult; StoppedEarly bool; ContextPath }`

**Evaluator injection seam (RESOLVED from source, orchestrator.go):**
`Orchestrator.Eval` is an exported struct field. `NewOrchestrator(meta, target)`
leaves it nil; `Run` defaults nil → `NopEvaluator{}` (orchestrator.go:67). Inject
a custom evaluator by **setting the field after construction**:

```go
o := sia.NewOrchestrator(metaRunner, targetExec)
o.Eval = &PaperEvaluator{...}   // or ThroughputEvaluator / TestGateEvaluator
o.Run(ctx, opts)
```

`o.Eval.Evaluate(ctx, genDir)` is called once per generation (orchestrator.go:173)
and a returned **Go error aborts the whole run** — so our evaluators must report
task failure via `EvalResult.Status` (`EvalError`/`EvalWarning`), never a Go
error. (`EvalError` is for the evaluator being unable to run; a `REVISE`/failed
task is still `EvalSuccess` with the verdict inside `results.json`.) The CLI uses
the same seam: `var eval sia.Evaluator = sia.NopEvaluator{}` (cmd/sia/main.go:122).
This is the injection point all of P1/P3/P4 use.

Module path: `github.com/tmc/mlx-go-sia`, a member of `mlx-go-examples/go.work`.

## 4. The four projects

| # | Name | Track lead | One-liner |
|---|---|---|---|
| P1 | `paperbench` | Research | Score an agent implementing a paper-roadmap prototype via the `coverage-map` rubric (honest-recompute, Goodhart-resistant), wired as a SIA `Evaluator`. **Flagship. Must-have.** |
| P3 | `mlx-inference-opt` | Applied | SIA loop optimizing an `mlx-go-lm` decode sampler with a tokens/sec `Evaluator` (interleaved gen-0 baseline). **"Number goes up" money-shot. Must-have.** |
| P2 | offline engine | Framework | **Trimmed:** point the existing `ExecRunner` at a thin `mlx-lm-generate` wrapper — offline claim, no new Go agent loop. **Stretch.** |
| P4 | `self-mod` | Framework | **Cut to recorded kicker:** loop edits a `cp -r` copy of `mlx-go-sia` with a *planted* bug, ends green. **Recorded 30s only, build last.** |

Priority order for a solo build: **P1 → P3 → (P2 wrapper) → (P4 recording).**

See `01-paperbench.md` … `04-self-mod.md`.

## 5. Demo plan (3–4 minutes)

1. **(0:00) Frame (20s):** "Self-improvement needs a *measurement*. We built one
   for the hardest agent task we know — implementing research papers — and ran
   SIA against it, fully local on Go + MLX."
2. **(0:20) P3 money-shot (60s):** chart of tokens/sec climbing across
   generations on a real `mlx-go-lm` path. The objective number.
3. **(1:20) P1 flagship (90s):** paperbench rubric score climbing as the agent
   fills in `evidence_state` booleans / passes `fast_check` for a roadmap
   prototype. The novel eval, the self-improvement story.
4. **(2:50) P4 kicker (30s):** "It even improves its own harness" — the loop
   editing `mlx-go-sia` and re-running green tests.
5. **(3:20) Close (20s):** offline, local, Go + MLX; P2 is why it runs with no
   cloud. Tracks: Research (P1), Applied (P3), Framework (P2/P4).

**Offline safety:** every demo segment must have a pre-recorded fallback (asciinema
or screen capture) in case of wifi/API/thermal trouble. Live is the goal; recorded
is the insurance.

## 6. Risks & mitigations

- **The score must move convincingly.** The whole demo hinges on a visible delta.
  Mitigation: pick tasks where gen-0 is genuinely weak and the rubric has
  headroom (unflipped `evidence_state` booleans); validate the delta before
  building polish. (Owner: P1, P3.)
- **`claude` needs network at run time.** Mitigation: pre-record; pursue P2
  local-model engine as the offline-live path.
- **Evaluator injection seam unknown.** Mitigation: resolve §3 open question
  first; it blocks P1.
- **Thermal/benchmark noise on Apple silicon** (the roadmap itself warns of this).
  Mitigation: P3 evaluator must control run order / cooldown and report median of
  N, not a single hot run.
- **Scope for a small team.** Mitigation: P1 + P3 are the must-haves; P2 stretch;
  P4 only if green and cheap.

## 7. Success criteria

- A recorded (ideally live) demo where **at least one objective number improves
  across SIA generations**, end-to-end through the real `mlx-go-sia` loop.
- A clean submission repo (`sia-hackathon`) importing `mlx-go-sia` / `mlx-go-lm`,
  with the custom `Evaluator`(s) and the `pi` agent runner as the visible
  contributions.
- A crisp one-paragraph claim per track grounded in the demo.
