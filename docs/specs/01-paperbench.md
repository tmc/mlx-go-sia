# P1 — `paperbench` (Research track, flagship)

**One-liner:** Score an agent implementing a paper-roadmap prototype against the
`coverage-map.jsonl` rubric, wired as a SIA `Evaluator`, so the SIA loop drives
the agent to improve its implementation generation over generation.

## Why this is the flagship

SIA's whole premise is a *measured* feedback loop. The hardest, most legible
agent task we have local infra for is **implementing a research prototype**, and
the `paper-roadmap` already ships a machine-checkable rubric for exactly that.
So paperbench is: a **novel automatic eval** (Research) that **plugs into SIA's
own loop** (self-improvement) and runs **fully local** (offline demo).

## The dataset & rubric (already exists)

`mlx-go-examples/paper-roadmap/coverage-map.jsonl` — one row per prototype idea.
Each row (verified shape):

```jsonc
{
  "id": "ane-zero-cost-drafting",
  "source": "papers/prototypes/.../README.md",
  "claim": "<the falsifiable claim the prototype must support>",
  "status": "lightweight | covered | external | source_gap",
  "coverage": "<what artifacts exist>",
  "examples": ["<paths to code/tests/schemas/traces>"],
  "fast_check": "<a runnable command that must pass>",
  "heavy_skip": "<what is intentionally out of scope>",
  "evidence_state": {                 // <-- the machine-checkable rubric
    "fixture_row": true,
    "validator_command": true,
    "artifact_manifest_hash": false,
    "model_backed_or_opt_in_command": false,
    "control_rows": true,
    "falsifier_rows": false,
    "heavy_skip_narrowed_or_cleared": false
  },
  "gap": "<prose: exactly what is still missing>"
}
```

The `evidence_state` booleans are the score. A gen-0 agent inherits a row with
several `false`s; an improving agent flips them to `true` by writing the missing
fixtures, validators, falsifier rows, manifest hashes, etc. — and the
`fast_check` must still pass.

All 10 rows that carry an `evidence_state` use exactly these 7 keys (verified
against `coverage-map.jsonl`): `fixture_row`, `validator_command`,
`artifact_manifest_hash`, `model_backed_or_opt_in_command`, `control_rows`,
`falsifier_rows`, `heavy_skip_narrowed_or_cleared`.

## Threat model: Coupled co-evolutionary Goodhart (named by the SIA paper)

The SIA paper has a section titled **"Coupled co-evolutionary Goodhart"** warning
that when harness search and weight updates "both optimise against the same fixed
verifier V," the joint fixed point "can appear strong on the training verifier
while being fragile under any perturbation" (verified: `main.tex`). Our evaluator
*is* that fixed verifier, so paperbench is squarely exposed to this failure mode.

The naive rubric is trivially gameable — and an agent under optimization
pressure *will* game it:

| Boolean | Naive check | How an agent fakes it |
|---|---|---|
| `artifact_manifest_hash` | manifest field matches `^sha256:[0-9a-f]{64}$` (real regex in `schemas/evidence-manifest.schema.json`) | emit a random 64-hex string; no real artifact |
| `fixture_row`/`control_rows`/`falsifier_rows` | JSONL rows validate against the row schema (enums, numeric bounds) | synthesize schema-valid rows with fabricated numbers |
| `validator_command`/`model_backed_or_opt_in_command` | a named script exits 0 | write a script that just `exit 0`s — no inference |
| `heavy_skip_narrowed_or_cleared` | `heavy_skip` text shrank | delete the text, do no work |

**Honest-recompute requirements (the core of the contribution):** the evaluator
must *independently establish ground truth*, never trust the agent's assertions:

- `artifact_manifest_hash` → the evaluator itself hashes the referenced artifact
  in the gen dir and compares; a manifest hash with no matching artifact = false.
- `validator_command`/`model_backed_or_opt_in_command` → run them in a **frozen,
  evaluator-owned context** the agent cannot edit (the validator script and its
  inputs are checksummed from a pristine copy, not read from the agent's tree),
  with a positive *and* a negative input — a script that ignores its input and
  exits 0 fails the negative case.
- `falsifier_rows` → require ≥1 row that the validator *rejects* (a true negative
  control); a fixture with only passing rows = false.
- `heavy_skip_narrowed_or_cleared` → the narrowed scope must be backed by a new
  passing artifact, not just deleted text.

This honest recompute is what makes the climbing score *mean* something — and is
exactly what the NLM design review (and the paper) flagged as the make-or-break.

## Scoring model (categorical gate + weighted advisory number)

Per the repo's judging conventions (`skills/CONVENTIONS.md`): **gate
categorically, keep the number advisory.**

- **Verdict (categorical):** `PASS` iff `fast_check` exits 0 **and** no
  `blocker`-level evidence gap remains. Blockers = the subset of
  `evidence_state` keys the row marks as required for its `status` tier.
- **Advisory score (weighted, not flat):** a flat "fraction of booleans" weighs a
  spoofable JSON-parse check the same as real execution. Instead weight the hard,
  honest-recompute booleans heavily and gate them on a prerequisite chain:
  - cheap/structural (`fixture_row`, `heavy_skip_narrowed_or_cleared`): low weight.
  - execution/falsification (`validator_command`,
    `model_backed_or_opt_in_command`, `falsifier_rows`, `artifact_manifest_hash`):
    high weight, and **worth zero unless `validator_command` is true** (no credit
    for hashes/falsifiers without a working validator).
  Surfaced for the demo's "number goes up" line and trend-watching only; it does
  **not** decide the verdict.

This avoids the "pickier reviewer scores lower" failure mode, resists Goodhart,
and still gives the demo a clean PASS/REVISE signal and a meaningful curve.

## The `Evaluator` (the contribution)

```go
// PaperEvaluator scores one generation against a coverage-map row.
// Implements mlx-go-sia's Evaluator.
type PaperEvaluator struct {
    Row        CoverageRow // the target prototype's rubric row
    RepoRoot   string      // where fast_check / paths resolve
    RequireKeys []string   // evidence_state keys that gate the verdict (blockers)
}

func (e *PaperEvaluator) Evaluate(ctx context.Context, genDir string) (sia.EvalResult, error) {
    // 1. run e.Row.FastCheck in RepoRoot (bounded timeout)
    // 2. recompute evidence_state from the gen's artifacts (don't trust the row's cached bools)
    // 3. write results.json into genDir: {verdict, advisory_score, evidence_state, fast_check_ok}
    // 4. EvalSuccess iff results.json written; the loop reads it for feedback
}
```

Returns `EvalResult{Status: EvalSuccess, ResultsPath: <genDir>/results.json, ...}`
on a completed scoring run (even a `REVISE` verdict is `EvalSuccess` —
"failed the task" is data in `results.json`, not a Go error, matching the
reference's evaluator contract: `EvalError` is reserved for the evaluator
itself being unable to run).

**`results.json` schema (the feedback signal):**

```json
{
  "verdict": "PASS | REVISE",
  "advisory_score": 0.57,
  "fast_check_ok": true,
  "evidence_state": { "falsifier_rows": false, "...": true },
  "blockers": ["falsifier_rows", "artifact_manifest_hash"],
  "notes": "human-readable gap summary fed back to the Feedback-Agent"
}
```

The Feedback-Agent reads `results.json` (the loop already wires evaluation
output into the feedback prompt) and rewrites the target agent to close the
named blockers next generation.

## Recomputing `evidence_state` (the honest part)

We must not trust the row's cached booleans — that would let the agent "win" by
editing the rubble. The evaluator **recomputes** each boolean from artifacts in
the gen dir, e.g.:

- `fixture_row` → a JSONL trace fixture exists and validates against its schema.
- `validator_command` → the row's validator command runs and exits 0.
- `falsifier_rows` → the trace contains ≥1 row that would *fail* the claim
  (a negative control), not just supporting rows.
- `artifact_manifest_hash` → a manifest with a content hash is present and the
  hash matches the artifact.

Each recompute is a small, deterministic Go check. This is the core of the
contribution and the thing the NotebookLM review should pressure-test hardest:
**is each boolean an honest, gameable-resistant check?**

## Task selection for the demo (N≥3, verified candidates)

A Research-track "self-improvement generalizes" claim needs **N≥3** prototypes —
the SIA paper itself anchors generalization on *three contrasting domains*
(Chinese legal classification, GPU-kernel optimization, single-cell RNA
denoising). One prototype is an anecdote, not a result.

Pick rows where gen-0 has real headroom (unflipped booleans) and a fast
`fast_check`. **Verified against `coverage-map.jsonl`:**

- ✅ **`dflash-ddtree`** — fast Go test `fast_check`; missing
  `{artifact_manifest_hash, model_backed_or_opt_in_command, falsifier_rows,
  heavy_skip_narrowed_or_cleared}` (4-boolean headroom). Good target.
- ✅ **`eagle3-vocab-translation`** — targeted Go test `fast_check`; same 4-boolean
  gap. Good target.
- ⚠️ **`lazy-graph-masking`** — the NLM review proposed this, but its
  `evidence_state` is **all-true (zero headroom)** when checked against the file.
  **Do not use as a demo target** — there is nothing for the agent to improve.
  (Caught by filesystem verification; the notebook was wrong on this one.)

Third target: pick another row with a non-empty missing set and a fast
`fast_check` (candidates with schemas + sample traces: `byte-exact-replay`,
`floating-point-drift-circuit`, `ane-zero-cost-drafting`). Verify its
`evidence_state` has headroom before committing — do not trust the cached bools.

## Wiring into the loop

- `NewOrchestrator(metaRunner, targetExecutor)` with `metaRunner` = `ExecRunner`
  driving `claude` (P2 swaps in a local engine).
- Inject `PaperEvaluator` (resolve the **open API question** in the master spec
  §3: field vs `RunOptions`; default is `NopEvaluator`).
- `RunOptions{ MaxGen: 4–6, Focus: FocusHarness, Task: <the prototype>, ... }`.
- Artifacts land under `runs/run_{id}/gen_{n}/` with `results.json` per gen.

## Deliverables

1. `CoverageRow` loader for `coverage-map.jsonl` (+ schema validation).
2. `PaperEvaluator` implementing `sia.Evaluator`, with per-boolean recompute
   checks and `results.json` emission.
3. A small CLI / example that runs the loop on one prototype and prints the
   per-gen verdict + advisory score (the demo data source).
4. A plot script turning per-gen `results.json` into the climbing-score chart.

## Open questions for the NLM design review

- Is each `evidence_state` recompute **gameable**? Where could an agent flip a
  boolean true without honestly satisfying the claim?
- Is the **blocker set** (gating keys) defensible per `status` tier, or arbitrary?
- Is "advisory score = fraction of booleans" the right curve, or should it weight
  the harder booleans (falsifiers, manifest hash) more?
- Does recomputing from the gen dir vs the repo root create a path-resolution
  hazard (the agent edits the repo, not the gen dir)?
- Is one prototype enough for a Research claim, or do we need ≥3 for the
  "generalizes" story?
