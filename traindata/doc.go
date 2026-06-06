// Package traindata renders recorded SIA agent trajectories into token-level
// training samples.
//
// It is the Go-native analog of the tinker_cookbook.renderers the SIA reference
// delegates to the generated train.py. SIA's own orchestration core never
// tokenizes: it loads a trajectory and drops the raw JSON into the feedback
// prompt as text (see package sia). In weights mode the reference only emits a
// prompt instructing the generated train.py to build a renderer with
// tinker_cookbook. This package provides the same capability natively and
// post-hoc, converting a recorded [github.com/tmc/mlx-go-sia.Execution] into
// [TrainingSample] values via a [github.com/tmc/mlx-go-experiments/renderer.Renderer].
//
// The package is intentionally separate from package sia so the faithful
// orchestration core stays free of renderer imports; nothing here runs during
// a run. Convert with [MessagesFromTrajectory] and [SamplesFromExecution], then
// serialize with [WriteJSONL] (the JSONL the reference's train.py reads) or hand
// the samples to package localtrain to train a local model.
package traindata

// NameTrainData is the conventional filename for the JSONL emitted by
// [WriteJSONL] into a generation directory. It is an artifact of this additive
// layer, not a SIA reference layout name.
const NameTrainData = "train_data.jsonl"
