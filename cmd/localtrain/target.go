package main

import (
	"context"
	"os"

	sia "github.com/tmc/mlx-go-sia"
)

// seedingTarget wraps a [sia.TargetExecutor] and seeds the generation's train.py
// from the reference spec when the engine produced none (e.g. the no-op engine
// in the dry-run self-test). This keeps the demo runnable end-to-end without an
// agent while exercising the real spec-parse path of the wrapped executor; with
// a live engine the agent's own train.py is already present and is left untouched.
type seedingTarget struct {
	inner sia.TargetExecutor
	seed  string // train.py spec written when the gen has none
}

// RunTarget writes the seed spec into req.AgentPath if it is missing, then
// delegates to the wrapped executor.
func (t *seedingTarget) RunTarget(ctx context.Context, req sia.TargetRequest) (sia.TargetResult, error) {
	if _, err := os.Stat(req.AgentPath); err != nil {
		if writeErr := os.WriteFile(req.AgentPath, []byte(t.seed), 0o644); writeErr != nil {
			return sia.TargetResult{Success: false, ErrorMsg: "seed train.py: " + writeErr.Error()}, nil
		}
	}
	return t.inner.RunTarget(ctx, req)
}
