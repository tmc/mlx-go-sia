package sia

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// EvalStatus classifies an evaluation outcome, mirroring the reference's status
// strings.
type EvalStatus string

const (
	EvalSkipped EvalStatus = "skipped" // no evaluator / no evaluate.py
	EvalSuccess EvalStatus = "success" // ran and produced results.json
	EvalWarning EvalStatus = "warning" // ran but no results.json
	EvalError   EvalStatus = "error"   // failed or timed out
)

// EvalResult summarizes one generation's evaluation.
type EvalResult struct {
	Status      EvalStatus
	Reason      string // populated for skipped/warning/error
	ResultsPath string // path to results.json when Status == EvalSuccess
	Output      string // combined evaluator stdout/stderr
}

// Evaluator scores a generation's output. The reference runs a task-provided
// evaluate.py that compares the agent's output against held-out ground truth and
// writes results.json into the generation directory. Abstracting it lets the Go
// port plug in a different scorer or skip evaluation entirely.
type Evaluator interface {
	// Evaluate scores the generation directory. It returns an [EvalResult]; a
	// failed evaluation is reported via Status (not a Go error). A Go error is
	// returned only when the evaluator cannot run at all.
	Evaluate(ctx context.Context, genDir string) (EvalResult, error)
}

// NopEvaluator skips evaluation, reporting [EvalSkipped]. It is the default when
// no evaluator is configured.
type NopEvaluator struct{}

// Evaluate reports that evaluation was skipped.
func (NopEvaluator) Evaluate(_ context.Context, _ string) (EvalResult, error) {
	return EvalResult{Status: EvalSkipped, Reason: "no evaluator configured"}, nil
}

// ExecEvaluator runs the task's evaluate.py as a subprocess, matching the
// reference contract: `<interpreter> <script> --gen-dir <genDir>`, after which
// genDir/results.json is expected. The combined output is written to
// genDir/evaluation.log.
type ExecEvaluator struct {
	// Script is the path to evaluate.py. If empty, [ExecEvaluator.Evaluate]
	// reports [EvalSkipped] (the task ships no evaluator).
	Script string
	// Interpreter runs the script (e.g. "python3"); empty runs the script directly.
	Interpreter string
	// Env is extra environment appended to os.Environ.
	Env []string
	// Timeout bounds a single evaluate.py run. A hung or runaway evaluator is
	// killed once it elapses (the reference passes EVAL_TIMEOUT to subprocess.run
	// for the same reason). Zero means no bound beyond the caller's context.
	Timeout time.Duration
}

// Evaluate runs evaluate.py against genDir.
func (e *ExecEvaluator) Evaluate(ctx context.Context, genDir string) (EvalResult, error) {
	if e.Script == "" {
		return EvalResult{Status: EvalSkipped, Reason: "evaluate.py not found"}, nil
	}

	name, args := e.Interpreter, []string{}
	if name == "" {
		name = e.Script
	} else {
		args = append(args, e.Script)
	}
	args = append(args, "--gen-dir", genDir)

	if e.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, e.Timeout)
		defer cancel()
	}

	var buf bytes.Buffer
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = append(os.Environ(), e.Env...)
	runErr := cmd.Run()
	output := buf.String()

	logPath := filepath.Join(genDir, NameEvalLog)
	_ = os.WriteFile(logPath, []byte(output), 0o644)

	if runErr != nil {
		return EvalResult{
			Status: EvalError,
			Reason: fmt.Sprintf("evaluate.py failed: %v", runErr),
			Output: output,
		}, nil
	}

	resultsPath := filepath.Join(genDir, NameResultsJSON)
	if isFile(resultsPath) {
		return EvalResult{Status: EvalSuccess, ResultsPath: resultsPath, Output: output}, nil
	}
	return EvalResult{
		Status: EvalWarning,
		Reason: "results.json not created by evaluate.py",
		Output: output,
	}, nil
}

// FuncEvaluator adapts a function to an [Evaluator] for tests.
type FuncEvaluator func(ctx context.Context, genDir string) (EvalResult, error)

// Evaluate calls the wrapped function.
func (f FuncEvaluator) Evaluate(ctx context.Context, genDir string) (EvalResult, error) {
	return f(ctx, genDir)
}
