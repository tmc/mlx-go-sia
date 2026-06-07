# P4 — `self-mod` (Framework track) — the recursive kicker

**One-liner:** Point the SIA loop at `mlx-go-sia` *itself* — the Feedback-Agent
rewrites the harness's own code and re-runs its test suite — so the demo shows
"it improves its own harness," SIA's literal premise made visible.

**Status after design review: CUT from the primary plan.** P1 and P3 are the
must-haves; for a solo builder, hunting a *real* bug a weak agent can reliably fix
in ≤5 gens is an unpredictable time sink and a dangerous distraction. P4 ships
**only as a pre-recorded 30s kicker**, and **only with a planted synthetic bug**
(below) so the outcome is deterministic — never as a live, find-a-real-bug demo.

Decisions locked from the review:
- **Isolation: in-gen copy (`cp -r`), not a git worktree** — a worktree adds git
  state + cleanup failure modes for no demo benefit.
- **Target: a planted synthetic bug** (e.g. comment out a line so a known test
  fails), so the agent's fix is trivial and the recording is reliable. A "real
  bug" target is explicitly out of scope for the hackathon.
- Editing a *copy* is honest — a compiler/test gate doesn't care if it's the live
  tree — and it is the necessary safety measure so a bad gen can't break the loop
  that's running the demo.

## Why

SIA's headline is "an agent that rewrites both its own harness and weights."
Most teams will demo SIA improving *some other* artifact. Showing SIA improving
**the SIA harness** is the most on-theme possible image — recursion the judges
can see.

## The task (planted synthetic bug, deterministic)

A bounded, **planted** defect in a `cp -r` copy of `mlx-go-sia` with an objective
gate:

- **Target:** comment out / break one line so a known existing test fails at
  gen-0 (a synthetic, predictable defect — *not* a hunt for a real bug).
- **Correctness gate:** `go build ./...`, `go vet ./...`, `go test ./...` on the
  copied tree must all pass; the previously-failing target test must flip
  false→true.

The agent only succeeds if the package builds and all tests (including the target
test) are green. Deterministic by construction, so the 30s recording is reliable.

## The `Evaluator`

Reuse the throughput/rubric pattern, but the gate is the Go test suite:

```go
// TestGateEvaluator builds and tests a Go module copy in genDir, writing
// pass/fail + counts to results.json. Implements sia.Evaluator.
type TestGateEvaluator struct {
    ModuleDir   string   // the mlx-go-sia copy the gen edits
    TargetTest  string   // the test that must flip false->true (the objective)
    Gates       []string // ["go build ./...", "go vet ./...", "go test ./..."]
}
```

`PASS` iff all gates green **and** `TargetTest` passes. `results.json` carries
the failing→passing transition and the test counts — the feedback signal and the
demo's proof.

## Safety: isolate the self-modification

The loop edits **a copy** of `mlx-go-sia` inside the gen dir (or a git worktree),
never the live source tree. This prevents a bad gen from breaking the harness
that is running the loop — a real foot-gun for a recursive demo. Worktree
isolation is the clean option.

## Wiring

- `NewOrchestrator(metaRunner, targetExecutor)` where the target the agent edits
  is the `mlx-go-sia` copy; `TargetExecutor` runs `go build`/`go test`.
- Inject `TestGateEvaluator`.
- `RunOptions{ MaxGen: 3–5, Focus: FocusHarness, ... }`.

## Deliverables (only if time after P1+P3)

1. A chosen, bounded self-improvement target in `mlx-go-sia` + a failing target
   test.
2. `TestGateEvaluator` implementing `sia.Evaluator`.
3. The isolated (worktree/copy) loop run, ending with green tests on the
   modified harness — recorded for the 30s kicker.

## Open questions for the NLM design review

- What is a target that is **real** (not contrived) yet **small enough** to land
  in ≤5 gens with a `claude`-driven agent?
- Worktree vs in-gen copy for isolation — which is simpler and safer under demo
  time pressure?
- Is the recursion *legible in 30 seconds*, or does it need too much setup to land
  as a kicker? If the latter, cut it.
- Does editing a copy (not the live tree) undercut the "it improved *itself*"
  claim, or is "it improved a copy of itself, verified green" honest enough?
