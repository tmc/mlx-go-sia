// Package localtrain trains a local model on SIA-derived training samples.
//
// It drives a localtinker coordinator (github.com/tmc/localtinker, served by
// cmd/localtinker) through its Go tinker API — the Go-native, hosted-Tinker-free
// version of what the SIA reference delegates to the generated train.py. The
// reference's weights mode emits a prompt telling train.py to build a renderer
// and run against the hosted Tinker service; this package consumes the
// equivalent rendered samples (see package traindata) and runs the same
// CreateLoRA → ForwardBackward → OptimStep → Save loop locally on MLX.
//
// [DatumFromSample] maps a [github.com/tmc/sia-apple-silicon/traindata.TrainingSample]
// onto the Tinker training contract: the rendered token stream is the model
// input, its next-token shift is the cross-entropy target, and the loss mask
// becomes per-token weights. [Train] runs the loop against a [tinker.Client].
//
// The in-process tinker handle validates batches and shapes data; actual MLX
// forward/backward executes through a running coordinator. When execution is
// not available the underlying call returns tinker.ErrUnsupported, which [Train]
// surfaces unchanged.
package localtrain
