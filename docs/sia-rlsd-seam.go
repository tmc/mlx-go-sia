// Package docs — SIA-RLSD seam sketch
//
// This file shows the minimal wiring to replace SIA's SFT (mlx-lm-train /
// MLXTrainExecutor) weight loop with an RLSD objective
// (mlx-go-rlsd.NewTeacherTrainer / RLSDTrainer.Step).
//
// The "seam" is identical at the training.LossFunc level:
//
//   SFT path:  mlx-lm-train → cross-entropy loss (opaque, inside Python)
//   RLSD path: rlsd.MakeLossFunc(model, adapters, cfg) → training.LossFunc
//              then rlsd.NewTeacherTrainer(...).Step(ctx, prompts, env, ...)
//
// Both return an adapter weight file; WeightsEvaluator scores it against the
// held-out test set the same way — that's the "same seam."
//
// Objective precision: RLSD is NOT cross-entropy SFT. It is a GRPO-family
// policy-gradient loss where per-token advantages are teacher-interpolated:
//
//   advantage_t = A_group * ((1-λ)*1 + λ*clip(exp(sign(A)*Δ), 1-ε, 1+ε))
//   Δ = teacher_logp(y_t) − student_logp(y_t)   (stop-gradient)
//   λ decays from 0.5 → 0 over the first cfg.LambdaDecaySteps steps
//
// This file does NOT compile — it imports from modules not wired into this
// worktree's go.mod. It is a prose+code seam document only.

//go:build ignore

package docs

import (
	"context"
	"fmt"
	"log"

	sia "github.com/tmc/sia-apple-silicon"
	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	rlsd "github.com/tmc/mlx-go/examples/mlx-go-rlsd"
	"github.com/tmc/mlx-go/mlx"
)

// RLSDTrainExecutor wraps rlsd.RLSDTrainer as a sia.TargetExecutor.
// It replaces MLXTrainExecutor for a FocusWeights run whose update objective
// is RLSD self-distillation / RLVR, not SFT cross-entropy.
//
// Wire it the same way as MLXTrainExecutor:
//
//	o := sia.NewOrchestrator(meta, NewRLSDTrainExecutor(cfg))
//	o.Run(ctx, sia.RunOptions{Focus: sia.FocusWeights, ...})
type RLSDTrainExecutor struct {
	// ModelPath is the HuggingFace model ID or local path, e.g.
	// "mlx-community/Llama-3.2-1B-Instruct-4bit".
	ModelPath string

	// TrainData is the GSM8K-format JSONL path for RLSD rollout collection.
	// Never pass the held-out test set here; it belongs to WeightsEvaluator.
	TrainData string

	// Steps is how many RLSD optimizer steps to run per generation.
	// Keep this small (3–10) for demo runs — GPU is shared.
	Steps int

	// AdapterOut is the path where the trained adapter will be written.
	AdapterOut string

	// Cfg is the RLSD objective config. Zero value → rlsd.DefaultConfig().
	Cfg rlsd.RLSDConfig

	// LR is the learning rate for AdamW.
	LR float32
}

// RLSDSpec is the subset of RLSD hyperparameters the meta-agent can tune.
// The meta-agent writes these as top-level assignments in a "spec.py" file;
// the executor reads them and translates them into an RLSDConfig + run.
type RLSDSpec struct {
	// Algorithm selects the RLSD variant: "rlsd" | "opsd" | "grpo-opsd".
	Algorithm string // default: "rlsd"
	// LR is the AdamW learning rate.
	LR float32 // default: 1e-5
	// LoRARank is the LoRA adapter rank.
	LoRARank int // default: 8
	// GroupSize is the number of completions per prompt for advantage estimation.
	GroupSize int // default: 8
	// Steps is the number of RLSD optimizer steps.
	Steps int // default: 10
	// TeacherSyncPeriod is how often the teacher snapshot is refreshed.
	TeacherSyncPeriod int // default: 10
}

// RunTarget implements sia.TargetExecutor.
//
// It:
//  1. Loads the base model.
//  2. Attaches LoRA adapters.
//  3. Creates an rlsd.RLSDTrainer with NewTeacherTrainer (RLSD objective).
//  4. Collects rollouts from the GSM8K verifier reward.
//  5. Calls trainer.Step() in a loop (RLSD update, NOT SFT).
//  6. Saves the adapter to AdapterOut so WeightsEvaluator can score it.
func (e *RLSDTrainExecutor) RunTarget(ctx context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
	// --- load model ---
	model, tok, err := mlxlm.Load(ctx, e.ModelPath)
	if err != nil {
		return sia.TargetResult{Success: false, ErrorMsg: fmt.Sprintf("load model: %v", err)}, nil
	}

	// --- read spec written by the meta-agent ---
	spec, err := parseRLSDSpec(req.AgentPath)
	if err != nil {
		return sia.TargetResult{Success: false, ErrorMsg: fmt.Sprintf("parse spec: %v", err)}, nil
	}

	algorithm := firstNonEmptyStr(spec.Algorithm, "rlsd")
	lr := firstPositiveFloat32(spec.LR, e.LR, 1e-5)
	loraRank := firstPositiveInt(spec.LoRARank, 8)
	groupSize := firstPositiveInt(spec.GroupSize, 8)
	steps := firstPositiveInt(spec.Steps, e.Steps, 10)

	// --- wire RLSD config ---
	cfg := e.Cfg
	if cfg == (rlsd.RLSDConfig{}) {
		cfg = rlsd.DefaultConfig()
	}
	cfg.GroupSize = groupSize
	if spec.TeacherSyncPeriod > 0 {
		cfg.TeacherSyncPeriod = spec.TeacherSyncPeriod
	}

	// --- create LoRA adapters and trainer ---
	adapters, err := createLoRAAdapters(model, loraRank)
	if err != nil {
		return sia.TargetResult{Success: false, ErrorMsg: err.Error()}, nil
	}
	defer adapters.Free()

	// This is the KEY LINE: MakeLossFunc returns a training.LossFunc —
	// the same interface SFT uses — but the objective is RLSD self-distillation:
	//   loss = −E[advantage_t(λ) * log p_student(y_t | x)]
	//
	// NewTeacherTrainer wraps MakeLossFunc in a stateful trainer that also manages:
	//   * teacher-snapshot sync (periodic copy of current LoRA params)
	//   * rollout collection (via rlsd.CollectPadded)
	//   * EnrichRollouts (attaches teacher log-probs to each rollout)
	var trainer *rlsd.RLSDTrainer
	switch algorithm {
	case "opsd":
		trainer = rlsd.NewOPSDTrainer(model, adapters, cfg, lr)
	case "grpo-opsd":
		trainer = rlsd.NewGRPOOPSDTrainer(model, adapters, cfg, lr)
	default:
		// "rlsd" — the RLSD self-distillation / RLVR objective
		trainer = rlsd.NewTeacherTrainer(model, adapters, cfg, lr)
	}
	defer trainer.Free()

	// --- load training prompts and build GSM8K verifier ---
	examples, err := loadGSM8KExamples(e.TrainData)
	if err != nil {
		return sia.TargetResult{}, fmt.Errorf("load train data: %w", err)
	}
	env, privilegedContextFn, err := newGSM8KVerifier(tok, examples)
	if err != nil {
		return sia.TargetResult{}, fmt.Errorf("gsm8k verifier: %w", err)
	}

	collectCfg := rlsd.DefaultCollectConfig()
	collectCfg.GroupSize = groupSize
	collectCfg.MaxSeqLen = cfg.MaxSeqLen
	collectCfg.MaxNewTokens = cfg.MaxSeqLen / 4
	collectCfg.Temperature = 1.0
	collectCfg.EOSTokens = tok.EOSTokenIDs()

	// --- RLSD training loop (NOT SFT) ---
	log.Printf("RLSD loop: %d steps, algorithm=%s, lr=%.2e, lora-rank=%d", steps, algorithm, lr, loraRank)
	for step := 0; step < steps; step++ {
		// Pick a batch of prompts cyclically.
		batch := cyclicBatch(examples, step, 2 /*promptBatch*/)

		// trainer.Step runs:
		//   1. CollectPadded → rollouts (model generates completions, env scores)
		//   2. EnrichRollouts → attach teacher log-probs to each rollout
		//   3. For each rollout: Forward (RLSD loss) + Apply (AdamW update)
		// The loss is teacher-interpolated RLVR — not SFT cross-entropy.
		result, err := trainer.Step(ctx, batch, env, &rlTokenizer{tok: tok}, collectCfg, privilegedContextFn)
		if err != nil {
			return sia.TargetResult{Success: false, ErrorMsg: fmt.Sprintf("step %d: %v", step, err)}, nil
		}
		log.Printf("  step %d: reward=%.3f loss=%.6f lambda=%.3f rollouts=%d",
			step, result.MeanReward, result.MeanLoss, result.Lambda, result.NumRollouts)
		mlx.FlushFreeQueue()
	}

	// --- save adapter (same as SFT path) ---
	if err := trainer.Save(e.AdapterOut); err != nil {
		return sia.TargetResult{Success: false, ErrorMsg: fmt.Sprintf("save adapter: %v", err)}, nil
	}

	return sia.TargetResult{
		Success: true,
		Stdout:  fmt.Sprintf("RLSD trained %d steps, adapter saved to %s", steps, e.AdapterOut),
	}, nil
}

// --- Seam comparison: SFT vs RLSD ---
//
// SFT (current SIA loop):
//   Orchestrator → MLXTrainExecutor.RunTarget()
//     → parseTrainSpec(train.py)       // reads LR, rank, iters, batch_size
//     → exec mlx-lm-train -train ...   // SFT cross-entropy, opaque Python
//     → TargetResult{Stdout: log}
//   WeightsEvaluator reads adapter from WorkingDir/adapters/
//
// RLSD (this file):
//   Orchestrator → RLSDTrainExecutor.RunTarget()
//     → parseRLSDSpec(spec.py)         // reads LR, rank, group_size, steps, algorithm
//     → rlsd.NewTeacherTrainer(...)     // RLSD objective, pure Go/MLX
//     → trainer.Step() × steps         // teacher-interpolated RLVR update
//     → trainer.Save(AdapterOut)
//   WeightsEvaluator reads adapter from AdapterOut — IDENTICAL downstream
//
// The meta-agent's "tuning knobs" change from {LR, rank, iters, batch_size}
// to {LR, rank, group_size, steps, algorithm, teacher_sync_period}.
// The eval + feedback loop is unchanged.
//
// What changes in the demo story:
//   "We train with RLSD — reward-guided self-distillation — not cross-entropy.
//    The model collects completions on GSM8K, scores them with a verifiable
//    math reward, and the teacher snapshot provides a KL anchor. Loss descends
//    because the adapter learns to increase probability of high-reward tokens."

// Stubs so the file is parseable (not compiled due to //go:build ignore):

type rlTokenizer struct{ tok interface{ Encode(string) ([]int32, error) } }

func (r *rlTokenizer) Encode(s string) ([]int32, error) { return r.tok.Encode(s) }

func parseRLSDSpec(_ string) (RLSDSpec, error)                                              { return RLSDSpec{}, nil }
func createLoRAAdapters(_ models.LanguageModel, _ int) (interface{ Free() }, error)         { return nil, nil }
func loadGSM8KExamples(_ string) ([]interface{}, error)                                     { return nil, nil }
func newGSM8KVerifier(_ interface{}, _ []interface{}) (rl.Environment, func(string, string) ([]int32, error), error) {
	return nil, nil, nil
}
func cyclicBatch(_ []interface{}, _, _ int) []string { return nil }
func firstNonEmptyStr(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
func firstPositiveFloat32(vals ...float32) float32 {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}
func firstPositiveInt(vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}
