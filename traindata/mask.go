package traindata

import "github.com/tmc/mlx-go-experiments/renderer"

// RLMask is the default loss policy for reinforcement-learning samples: a nil
// role filter, which makes [renderer.BuildTrainingSample] mask to the
// renderer's SampledMask — every token the model would produce at inference is
// trainable. This matches GRPO's assistant-tokens-trainable contract.
var RLMask func(renderer.Message) bool = nil

// SFTMask is a supervised-fine-tuning loss policy: train on assistant tokens.
// Combine with [AssistantPlusToolSFTRoles] to also supervise tool-reply bodies.
func SFTMask(m renderer.Message) bool {
	return m.Role == "assistant"
}

// AssistantPlusToolSFTRoles opts tool-role message bodies into supervision when
// passed as the contentSFTRoles argument of [SamplesFromExecution] (via
// [EmitOptions.ContentSFTRoles]). Their non-body scaffolding stays masked.
var AssistantPlusToolSFTRoles = map[string]bool{"tool": true}
