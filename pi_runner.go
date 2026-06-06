package sia

import (
	"context"
	"fmt"
	"os"
)

// DefaultPiModel is the local model the pi-mlx engine drives when none is given.
// It is a 4-bit Llama that is small and fast enough to keep the loop watchable
// live; the cached Qwen2.5 family does not load in the current mlx-lm-generate
// build. Override per request via [AgentRequest.Model] or globally via the
// PI_MLX_MODEL environment variable read by the pi-mlx wrapper.
const DefaultPiModel = "mlx-community/Llama-3.2-1B-Instruct-4bit"

// DefaultPiScript is the wrapper command [PiRunner] invokes. The bare name is
// resolved on PATH; to run the repository's scripts/pi-mlx without installing
// it, set [PiRunner.Script] to its ABSOLUTE path. A relative path would be
// resolved against the request's working directory (the run/generation dir),
// not the repository, and so would not be found.
const DefaultPiScript = "pi-mlx"

// PiRunner is the offline SIA meta/feedback engine: an [AgentRunner] that drives
// a local model through the pi-mlx wrapper around mlx-lm-generate, so a run can
// improve itself with no network and no in-process model dependency. It is the
// offline counterpart to pointing an [ExecRunner] at the network `claude` CLI —
// both are just a command behind the same seam.
//
// The wrapper reads the prompt on stdin and writes the completion to stdout;
// PiRunner runs it in the request's working directory and selects the model via
// the PI_MLX_MODEL environment variable (the request's model takes precedence
// over the runner's, which defaults to [DefaultPiModel]).
//
// The zero value is usable: it drives [DefaultPiModel] through the [DefaultPiScript]
// command on PATH. [NewPiRunner] is the convenience constructor for a custom model.
type PiRunner struct {
	// Model is the default model id passed to the wrapper when a request does
	// not name one. Empty means [DefaultPiModel].
	Model string
	// Script is the wrapper command (default [DefaultPiScript], resolved on
	// PATH). To run the repository copy without installing it, set this to the
	// ABSOLUTE path of scripts/pi-mlx: the command runs in the request's working
	// directory, so a relative path resolves against that dir, not the repo.
	Script string
	// Stdout receives the generated text; nil means os.Stdout. The orchestrator
	// captures the working directory afterward, so a run typically lets the
	// agent's output stream here while it writes files in WorkingDir.
	Stdout *os.File
	// Stderr receives the wrapper's diagnostics; nil means os.Stderr.
	Stderr *os.File
}

var _ AgentRunner = (*PiRunner)(nil)

// NewPiRunner returns a [PiRunner] that drives the given local model offline. An
// empty model uses [DefaultPiModel]. This is the constructor the feedback and
// self-improvement engines call to obtain a fully local AgentRunner.
//
// The wrapper command defaults to [DefaultPiScript] resolved on PATH; to run the
// repository's copy without installing it, set the returned runner's Script field
// to the ABSOLUTE path of scripts/pi-mlx (a relative path resolves against the
// request's working directory, not the repository).
func NewPiRunner(model string) *PiRunner {
	return &PiRunner{Model: model}
}

// Name reports the agent-impl id used for profile validation.
func (r *PiRunner) Name() string { return "pi-mlx" }

// Run drives the pi-mlx wrapper for one invocation: it writes req.Prompt to the
// wrapper's stdin, runs it in req.WorkingDir, and exports the selected model as
// PI_MLX_MODEL. It returns an error only when the wrapper itself fails to run;
// whether the agent produced the expected files is the orchestrator's concern.
func (r *PiRunner) Run(ctx context.Context, req AgentRequest) error {
	model := req.Model
	if model == "" {
		model = r.Model
	}
	if model == "" {
		model = DefaultPiModel
	}
	script := r.Script
	if script == "" {
		script = DefaultPiScript
	}

	exec := &ExecRunner{
		ImplName: r.Name(),
		Command:  script,
		Stdout:   r.Stdout,
		Stderr:   r.Stderr,
		Env:      []string{"PI_MLX_MODEL=" + model},
	}
	// Carry the resolved model on the request too, so any %MODEL% token a caller
	// adds to the wrapper's args substitutes the same model the env selects.
	req.Model = model
	if err := exec.Run(ctx, req); err != nil {
		return fmt.Errorf("pi-mlx runner: %w", err)
	}
	return nil
}
