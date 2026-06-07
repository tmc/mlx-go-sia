# P5 — Nebius Token Factory integration (compute: cloud open-models)

**One-liner:** Drive SIA's meta/feedback (and target) agents against
Nebius-hosted open models (Deepseek / Qwen / GLM / Gemma / Kimi / GPT-OSS) over
Token Factory's OpenAI-compatible API — the hackathon's sponsor-compute path and
the cloud half of the "full compute story" (the local half is `06-local.md`).

**Status: enablement, working now, COMMITTED.** All facts below are **verified
against `mlx-go-sia` source** (cross-checked by this session and the agent in
57D8). The module — including the Nebius token/env work — landed as the initial
import **`20909c4`** ("mlx-go-sia: initial Go port of SIA self-improvement loop")
on branch `gnosify-compiled-trainstep-cli` (46 files, no go.sum/binaries,
git-note with provenance, 30 tests `-race` green; verified by this session via
`git log`). The module was previously untracked, so this is one initial commit,
not an isolated Nebius commit; the unrelated context.md `max_gen` change rode
along intentionally (splitting from a zero-baseline import would fabricate a
never-tested pre-state). Branch is local (not pushed).

## What already exists (zero config change)

`Provider` struct (`provider.go`) JSON fields:
`provider_id`, `name`, `client_kind` (`anthropic`|`openai`|`google`), `base_url`,
`api_key_env`.

`nebius` is a **built-in provider** (`provider.go:59`, verified):

```json
{
  "provider_id": "nebius",
  "name": "Nebius Token Factory",
  "client_kind": "openai",
  "base_url": "https://api.tokenfactory.us-central1.nebius.com/v1/",
  "api_key_env": "NEBIUS_API_KEY"
}
```

> Authoritative base URL is the **tokenfactory** host above — *not*
> `api.studio.nebius.com`. (`together` and `openai` are also built in.)

**Built-in Nebius profiles** (`profile.go:139–146`, verified verbatim):

| Profile | Role | agent_impl | model |
|---|---|---|---|
| `kimi-nebius-meta` | meta | `openhands` | `moonshotai/Kimi-K2.6` |
| `qwen-nebius-target` | target | (default) | `Qwen/Qwen3-Next-80B-A3B-Thinking-fast` |
| `gptoss-nebius-target` | target | (default) | `openai/gpt-oss-120b-fast` |
| `kimi-nebius-target` | target | (default) | `moonshotai/Kimi-K2.6` |

Model ids are HF-style `org/model`, passed verbatim in the profile `model` field.
To add **Deepseek / GLM / Gemma**, write a target profile JSON with
`provider_id:"nebius"` and the model id (`deepseek-ai/DeepSeek-V3`,
`zai-org/GLM-4.6`, a Gemma id, …) and pass it via `-target-agent-profile
./that.json` — **no code change**.

## The one real gap — CLOSED (first-class, verified on disk)

Originally `ExecRunner` only substituted `%MODEL%`/`%MAXTURNS%`/`%WORKDIR%` and
didn't inject the provider's base URL/key. **Now fixed** (landed on branch
`gnosify-compiled-trainstep-cli`, verified by this session: `go test -race`
green):

- `runner_helpers.go` `newTokenReplacer` adds **`%BASEURL%`**, **`%APIKEY_ENV%`**,
  and **`%APIKEY%`** (the last reads the secret from `Provider.APIKeyEnv` via
  `os.Getenv` at substitution time). `%MODEL%`/`%MAXTURNS%`/`%WORKDIR%` unchanged.
- `cmd/sia/main.go` `providerEnv()` (line 177) sets `ExecRunner.Env`: for an
  OpenAI-compatible provider with a `base_url`, it exports **`OPENAI_BASE_URL`**
  and mirrors the resolved key onto **`OPENAI_API_KEY`**; returns nil for native
  providers (so the `claude`/Anthropic path is unchanged).
- No provider/profile JSON changed — still byte-compatible with the reference.
- Tests: `runner_test.go` — `TestTokenReplacer` (all 6 tokens incl. nebius
  base_url+key), `TestTokenReplacerEmptyAPIKey`, `TestExecRunnerSubstitutesArgs`
  (end-to-end via `/bin/sh`, captures argv + stdin).

**Exact working `-agent-args` forms (use these):**

```text
# base URL on a flag (env also exported, so this is belt-and-suspenders):
-agent-args '--base-url,%BASEURL%,--model,%MODEL%'

# if the CLI wants the key on a flag too:
-agent-args '--base-url,%BASEURL%,--api-key,%APIKEY%,--model,%MODEL%'

# pure env (CLI reads OPENAI_BASE_URL + OPENAI_API_KEY, now auto-exported):
-agent-args '--model,%MODEL%'
```

## Working now (no code change) — two run modes

**A. Agent CLI reads OpenAI env (most OpenAI-compatible CLIs):**

```bash
export NEBIUS_API_KEY=...
export OPENAI_BASE_URL=https://api.tokenfactory.us-central1.nebius.com/v1/
export OPENAI_API_KEY=$NEBIUS_API_KEY
```

**B. Agent CLI takes base URL on a flag:** hardcode it in `-agent-args`
(`--base-url,https://api.tokenfactory.us-central1.nebius.com/v1/,--model,%MODEL%`).

### Ready-to-run 1-gen loop against Nebius

```bash
cd .../mlx-go-examples/mlx-go-sia
export NEBIUS_API_KEY=...
export OPENAI_BASE_URL=https://api.tokenfactory.us-central1.nebius.com/v1/
export OPENAI_API_KEY=$NEBIUS_API_KEY

go run ./cmd/sia \
  -task-dir ./tasks/mytask -run-id 1 -max-gen 1 \
  -meta-agent-profile kimi-nebius-meta \
  -target-agent-profile qwen-nebius-target \
  -agent-cmd oai-agent -agent-args '--model,%MODEL%' \
  -interpreter python3
```

Dry-run first (writes the full `runs/run_1/gen_1/` tree + prompts, **no model
calls, no credits burned**):

```bash
go run ./cmd/sia -task-dir ./tasks/mytask -run-id 1 -max-gen 1 \
  -meta-agent-profile kimi-nebius-meta -target-agent-profile qwen-nebius-target -dry-run
```

## Verified caveat: claude-impl rejects non-anthropic providers

`-meta-agent-profile` defaults to `default-meta` (agent_impl `claude` + provider
`anthropic`). `ParseMetaAgentProfile` **rejects** `claude` + a non-anthropic
provider (`profile.go`). For an all-Nebius run, use a non-claude meta profile
(`kimi-nebius-meta`, agent_impl `openhands`). Target profiles have no such
restriction.

## How P5 threads into the other projects

- **P1 paperbench / P3 inference-opt:** Nebius is an alternative *engine* — run the
  meta/feedback agent on a strong open model (Kimi/Qwen) instead of `claude`. Lets
  us A/B "which engine self-improves better" — a nice Research-track data point.
- **P6 totally-local:** P5 is the *contrast*. The demo story is "here's SIA on
  sponsor cloud compute (Nebius) **and** the same loop fully on-device (P6)."
- **FocusWeights (cloud):** Nebius H200 credits make real weight-update runs
  feasible at sizes the laptop can't — the cloud counterpart to P6's local
  training.

## Deliverables

1. (57D8) `%BASEURL%`/`%APIKEY%` token + env wiring + test; report commit + final
   `-agent-args` form → fold into this doc.
2. A verified end-to-end 1-gen Nebius run (dry-run first, then one real gen on a
   cheap model) captured for the demo.
3. Optional Deepseek/GLM/Gemma target profile JSONs (no code change) for engine
   A/B.

## Open questions for the NLM design review

- Is an OpenAI-compatible *agentic* CLI (reads prompt on stdin, writes
  `target_agent.py`, honors `--model`) available, or do we need a thin one? This
  is the practical prerequisite for any non-`claude` meta engine.
- For the demo, is Nebius the *primary* engine or the *contrast* to local (P6)?
- Which Nebius model is the best price/capability for driving the SIA loop in the
  hackathon budget?
