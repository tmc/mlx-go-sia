package localtrain

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/localtinker/tinker"
)

// TrainOptions configures a training run.
type TrainOptions struct {
	// BaseModel is the Tinker model id to fine-tune (e.g. "Qwen/Qwen3-8B").
	BaseModel string
	// Rank is the LoRA rank.
	Rank int
	// Loss is the training loss; nil uses [tinker.CrossEntropy], the loss the
	// coordinator advertises and the one [DatumFromSample] shapes data for.
	Loss tinker.Loss
	// Optim is the optimizer; the zero value uses [tinker.DefaultAdamW].
	Optim tinker.AdamW
	// Epochs is the number of forward-backward/optimizer passes over the batch
	// (default 1).
	Epochs int
	// CheckpointName names the saved checkpoint (default "sia-lora").
	CheckpointName string
	// TrainMLP and TrainAttn select which LoRA targets to train; both default
	// to true when neither is set.
	TrainMLP  bool
	TrainAttn bool
	// Logf logs per-epoch progress; nil discards.
	Logf func(format string, args ...any)
}

// Train fine-tunes a LoRA on batch using the localtinker coordinator behind
// client. It runs CreateLoRA, then for each epoch a ForwardBackward followed by
// an OptimStep, and finally Save, returning the saved checkpoint.
//
// Train surfaces tinker.ErrUnsupported unchanged when the coordinator cannot
// execute MLX forward/backward, so callers can detect a validate-only run.
func Train(ctx context.Context, client *tinker.Client, batch []tinker.Datum, opts TrainOptions) (tinker.Checkpoint, error) {
	if client == nil {
		return tinker.Checkpoint{}, errors.New("localtrain: client is nil")
	}
	if len(batch) == 0 {
		return tinker.Checkpoint{}, errors.New("localtrain: batch is empty")
	}
	loss := opts.Loss
	if loss == nil {
		loss = tinker.CrossEntropy{}
	}
	optim := opts.Optim
	if optim == (tinker.AdamW{}) {
		optim = tinker.DefaultAdamW()
	}
	epochs := opts.Epochs
	if epochs <= 0 {
		epochs = 1
	}
	name := opts.CheckpointName
	if name == "" {
		name = "sia-lora"
	}
	trainMLP, trainAttn := opts.TrainMLP, opts.TrainAttn
	if !trainMLP && !trainAttn {
		trainMLP, trainAttn = true, true
	}
	logf := opts.Logf
	if logf == nil {
		logf = func(string, ...any) {}
	}

	trainer, err := client.CreateLoRA(ctx, tinker.CreateLoRARequest{
		BaseModel: opts.BaseModel,
		Rank:      opts.Rank,
		TrainMLP:  trainMLP,
		TrainAttn: trainAttn,
	})
	if err != nil {
		return tinker.Checkpoint{}, fmt.Errorf("localtrain: create lora: %w", err)
	}
	defer trainer.Close()

	for epoch := 1; epoch <= epochs; epoch++ {
		fb, err := trainer.ForwardBackward(ctx, batch, loss)
		if err != nil {
			return tinker.Checkpoint{}, fmt.Errorf("localtrain: epoch %d forward-backward: %w", epoch, err)
		}
		step, err := trainer.OptimStep(ctx, optim)
		if err != nil {
			return tinker.Checkpoint{}, fmt.Errorf("localtrain: epoch %d optim step: %w", epoch, err)
		}
		logf("epoch %d/%d: loss=%v optim=%v", epoch, epochs, fb.Loss, step.Metrics)
	}

	ckpt, err := trainer.Save(ctx, name)
	if err != nil {
		return tinker.Checkpoint{}, fmt.Errorf("localtrain: save %q: %w", name, err)
	}
	return ckpt, nil
}
