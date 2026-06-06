package localtrain

import (
	"github.com/tmc/localtinker/tinker"
	"github.com/tmc/mlx-go-sia/traindata"
)

// DatumFromSample maps a rendered SIA training sample onto the Tinker
// cross-entropy contract. The rendered token stream is the model input; the
// target at each position is the next token (a causal-LM shift), and the loss
// mask becomes per-token weights (1 for a trainable position, 0 otherwise).
//
// Input, TargetTokens and Weights are parallel arrays of equal length, as the
// Tinker batch validator and the hosted SDK require (see
// testdata/minimal_tinker_training.py). The final position has no next token,
// so its weight is forced to 0 regardless of the mask.
func DatumFromSample(s traindata.TrainingSample) tinker.Datum {
	n := len(s.TokenIDs)
	input := make([]int, n)
	targets := make([]int, n)
	weights := make([]float32, n)
	for i, id := range s.TokenIDs {
		input[i] = int(id)
	}
	for i := 0; i < n; i++ {
		if i+1 < n {
			targets[i] = int(s.TokenIDs[i+1])
			if i < len(s.LossMask) && s.LossMask[i+1] {
				weights[i] = 1
			}
		} else {
			targets[i] = int(s.TokenIDs[i]) // no next token; weight 0 masks it out
		}
	}
	return tinker.Datum{
		Input: tinker.FromTokens(input),
		LossInput: tinker.LossInput{
			TargetTokens: targets,
			Weights:      weights,
		},
	}
}

// BatchFromSamples maps each sample to a [tinker.Datum]. Empty samples (no
// tokens) are skipped so the batch passes the Tinker validator.
func BatchFromSamples(samples []traindata.TrainingSample) []tinker.Datum {
	batch := make([]tinker.Datum, 0, len(samples))
	for _, s := range samples {
		if len(s.TokenIDs) == 0 {
			continue
		}
		batch = append(batch, DatumFromSample(s))
	}
	return batch
}
