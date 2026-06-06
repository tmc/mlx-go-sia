package traindata

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/mlx-go-experiments/renderer"
	sia "github.com/tmc/mlx-go-sia"
)

// TrainingSample is one rendered trajectory ready for token-level training.
// TokenIDs is the full rendered token stream; LossMask is true for the tokens
// the loss is computed on (per [EmitOptions]). TrajectoryIndex is the position
// in the source execution, so a multi-trajectory sample aligns with its
// execution_qN.json file.
type TrainingSample struct {
	TrajectoryIndex int     `json:"trajectory_index"`
	TokenIDs        []int32 `json:"token_ids"`
	LossMask        []bool  `json:"loss_mask"`
}

// EmitOptions configures how an execution is rendered into samples.
type EmitOptions struct {
	// Tools is the tool schema in effect for the trajectories; usually nil for
	// SIA trajectories, which record tool calls inline.
	Tools []renderer.ToolSpec
	// RoleToMask selects which message roles contribute loss tokens. Nil uses
	// the renderer's sampled mask (see [RLMask]); [SFTMask] restricts to
	// assistant tokens.
	RoleToMask func(renderer.Message) bool
	// ContentSFTRoles opts additional roles into body-only supervision (see
	// [AssistantPlusToolSFTRoles]).
	ContentSFTRoles map[string]bool
}

// SamplesFromExecution renders every well-formed trajectory in exec to a
// [TrainingSample] using r. A single-file execution yields at most one sample
// at index 0; a multi-trajectory execution yields one sample per well-formed
// trajectory, preserving the original index. Error-object trajectories (a
// failed sample) are skipped, mirroring [sia.Execution.Summary] treating
// non-list trajectories as failed. A trajectory that is a list but malformed in
// a way the renderer rejects is also skipped, with the error collected and
// returned (joined) so the caller can report it without losing the good
// samples.
func SamplesFromExecution(r renderer.Renderer, exec sia.Execution, opts EmitOptions) ([]TrainingSample, error) {
	raws := exec.Trajectories
	if !exec.MultiTrajectory {
		raws = []json.RawMessage{exec.Single}
	}
	var samples []TrainingSample
	var errs []error
	for i, raw := range raws {
		msgs, err := MessagesFromTrajectory(raw)
		if errors.Is(err, ErrNotTrajectory) {
			continue // failed/error-object trajectory: not training data
		}
		if err != nil {
			errs = append(errs, fmt.Errorf("trajectory %d: %w", i, err))
			continue
		}
		if len(msgs) == 0 {
			continue
		}
		ids, mask, err := renderer.BuildTrainingSample(r, msgs, opts.RoleToMask, opts.Tools, opts.ContentSFTRoles)
		if err != nil {
			errs = append(errs, fmt.Errorf("trajectory %d: render: %w", i, err))
			continue
		}
		samples = append(samples, TrainingSample{TrajectoryIndex: i, TokenIDs: ids, LossMask: mask})
	}
	return samples, errors.Join(errs...)
}

// WriteJSONL writes samples as one compact JSON object per line and returns the
// number written. This is the line-delimited form the reference's generated
// train.py reads (json.loads per line).
func WriteJSONL(w io.Writer, samples []TrainingSample) (int, error) {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i, s := range samples {
		if err := enc.Encode(s); err != nil {
			return i, fmt.Errorf("traindata: encode sample %d: %w", i, err)
		}
	}
	if err := bw.Flush(); err != nil {
		return len(samples), fmt.Errorf("traindata: flush: %w", err)
	}
	return len(samples), nil
}
