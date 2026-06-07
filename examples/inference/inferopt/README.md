# inferopt — P3 sampler-optimization loop

Points the SIA self-improvement loop at a decode-step sampler and lets the agent
optimize it for throughput, gated on exact-token correctness. The visceral
"tokens/sec goes up across generations" demo.

Each generation the agent rewrites `candidate.go` — a top-k / top-p / temperature
sampler. The `ThroughputEvaluator` runs a frozen golden oracle the agent cannot
touch, and only if the emitted tokens match exactly does it time decode
throughput against an interleaved gen-0 baseline. The per-generation
`results.json` carries the climbing-throughput series (plot `delta_tokens_per_sec`
or `speedup`, which cancels thermal/cache drift).

## Run

    go run ./cmd/inferopt                                   # self-test: seed-only, no engine, no model
    go run ./cmd/inferopt -engine pi -max-gen 6             # fully offline via pi-mlx
    go run ./cmd/inferopt -agent-cmd claude -agent-args '-p,--model,%MODEL%' -max-gen 6

## Honesty

The golden tokens, fixtures, and frozen baseline live in `runs/_oracle/`, outside
every generation's working directory. The candidate receives only the input
logits on stdin (config + seed + logits) and never the golden tokens or this
path; correctness is compared after it exits. A faster-but-different sampler is
`REVISE`, never a win. See `internal/oracle/` and its tests.
