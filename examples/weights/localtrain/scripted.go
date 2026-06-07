package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	sia "github.com/tmc/sia-apple-silicon"
)

// scriptedSpecLadder is a deterministic meta/feedback engine that stands in for
// an LLM when none capable of safely editing a hyperparameter spec is available.
// Each generation it writes a revised train.py — a real step on a tuning ladder a
// competent meta-agent would walk after reading the held-out signal — so the
// weight-improvement loop runs end-to-end and reproducibly. It is the labeled
// fallback the P6 demo uses, the same pattern as cmd/inferopt's scripted engine.
//
// It is NOT a cheat on the held-out gate: every spec it writes is trained for
// real on-device by mlx-lm-train and scored by the WeightsEvaluator on a test set
// the engine never sees. The ladder encodes a meta-agent's reasoning — start
// conservative (underfit), diagnose the high held-out loss as underfitting and
// raise capacity/LR/iters, then push further — but whether each rung actually
// lowers the held-out loss is decided by the real evaluator. On a tiny dataset
// the final push may OVERFIT and the held-out loss may rise; that regression is a
// true outcome the gate catches, not something the ladder suppresses.
type scriptedSpecLadder struct {
	// rungs are the per-generation train.py specs. rungs[0] is gen-1's spec,
	// rungs[1] gen-2's, and so on; the last rung is reused past the ladder length.
	rungs []ladderRung
}

// ladderRung is one generation's spec plus the human-readable rationale that
// names why a meta-agent would choose it. The rationale is written into the
// train.py as a comment, so the gen-over-gen spec diff reads like agent
// reasoning rather than a knob being cranked.
type ladderRung struct {
	rationale    string
	learningRate string // emitted verbatim (e.g. "5e-6") to keep the spec legible
	loraRank     int
	numLayers    int
	iters        int
	batchSize    int
}

// defaultLadder is the three-rung underfit -> diagnose -> push tuning story.
//
//	gen 1: deliberately conservative — small LR, low rank, the fewest iters. This
//	       UNDERFITS the task, so the held-out loss starts high. It is the honest
//	       baseline a cautious first pass produces.
//	gen 2: the meta-agent reads the high held-out loss as underfitting and responds
//	       the way an engineer would: more iterations, a larger (but still safe) LR,
//	       and more LoRA capacity (rank 8 -> 16). This should lower the held-out loss.
//	gen 3: push the same direction harder (more iters, slightly higher LR) to wring
//	       out the last gains. On a tiny 10-sample train set this may instead OVERFIT
//	       — the train loss keeps dropping while the held-out loss rises. That is a
//	       real, desirable outcome for the demo: the held-out gate catches the
//	       regression and the evaluator flags it, the weights analogue of P7's gen_4
//	       bad-kernel catch.
//
// Every rung's iters is a multiple of mlx-lm-train's default -save-every (100):
// the trainer only writes adapters.safetensors at a save checkpoint, so an iters
// value below 100 trains real weights but saves no adapter for the evaluator to
// score. Keeping iters on the 100 grid guarantees each generation produces a
// scorable adapter (100 is still the conservative baseline against 200 and 400).
var defaultLadder = []ladderRung{
	{
		rationale:    "gen1 baseline: conservative first pass (fewest iters, small LR, low rank) — expected to underfit",
		learningRate: "5e-6", loraRank: 8, numLayers: 16, iters: 100, batchSize: 2,
	},
	{
		rationale:    "gen2: held-out loss was high -> diagnosed underfitting; raise iters, LR, and LoRA rank",
		learningRate: "2e-5", loraRank: 16, numLayers: 16, iters: 200, batchSize: 2,
	},
	{
		rationale:    "gen3: push further (more iters, higher LR) to chase the last gains — risks overfitting a tiny train set",
		learningRate: "3e-5", loraRank: 16, numLayers: 16, iters: 400, batchSize: 2,
	},
}

// newScriptedSpecLadder returns the default underfit->diagnose->push ladder.
func newScriptedSpecLadder() *scriptedSpecLadder {
	return &scriptedSpecLadder{rungs: defaultLadder}
}

// Name reports the agent-impl id used for profile validation.
func (s *scriptedSpecLadder) Name() string { return "scripted" }

var _ sia.AgentRunner = (*scriptedSpecLadder)(nil)

// Run writes this generation's train.py spec into the working directory. The
// generation index is parsed from the working directory's gen_N suffix; if it
// cannot be determined, the first rung (the conservative baseline) is used.
func (s *scriptedSpecLadder) Run(_ context.Context, req sia.AgentRequest) error {
	if len(s.rungs) == 0 {
		return fmt.Errorf("scripted spec ladder: empty ladder")
	}
	gen := genFromWorkingDir(req.WorkingDir)
	i := gen - 1 // gen is 1-based; rungs[0] is gen-1's spec
	if i < 0 {
		i = 0
	}
	if i >= len(s.rungs) {
		i = len(s.rungs) - 1
	}
	specPath := filepath.Join(req.WorkingDir, sia.NameTrainScript)
	if err := os.WriteFile(specPath, []byte(s.rungs[i].spec()), 0o644); err != nil {
		return fmt.Errorf("scripted spec ladder: write train.py: %w", err)
	}
	return nil
}

// spec renders the rung as a declarative train.py the MLXTrainExecutor parses.
// The rationale is a leading comment so the gen-over-gen diff narrates the
// agent's reasoning; the whitelisted keys (learning_rate, lora_rank, num_layers,
// iters, batch_size, fine_tune_type) are the only lines the executor reads.
func (r ladderRung) spec() string {
	var b strings.Builder
	b.WriteString("# train.py — declarative LoRA training spec (parsed, not executed).\n")
	b.WriteString("# " + r.rationale + "\n\n")
	fmt.Fprintf(&b, "learning_rate = %s\n", r.learningRate)
	fmt.Fprintf(&b, "lora_rank = %d\n", r.loraRank)
	fmt.Fprintf(&b, "num_layers = %d\n", r.numLayers)
	fmt.Fprintf(&b, "iters = %d\n", r.iters)
	fmt.Fprintf(&b, "batch_size = %d\n", r.batchSize)
	b.WriteString("fine_tune_type = \"lora\"\n")
	return b.String()
}

// genFromWorkingDir extracts N from a path ending in gen_N, returning 0 if the
// suffix is absent or unparsable.
func genFromWorkingDir(dir string) int {
	base := filepath.Base(filepath.Clean(dir))
	const prefix = "gen_"
	if !strings.HasPrefix(base, prefix) {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimPrefix(base, prefix))
	if err != nil {
		return 0
	}
	return n
}
