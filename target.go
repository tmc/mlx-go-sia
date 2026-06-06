package sia

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
)

// TargetResult is the outcome of running a target agent for one generation.
type TargetResult struct {
	Success  bool
	Stdout   string
	Stderr   string
	ErrorMsg string
}

// TargetExecutor runs the generated target agent (or train.py) for one
// generation. The orchestrator-to-target contract is fixed by the reference: the
// agent is invoked with --dataset_dir (read-only data) and --working_dir
// (read-write, the generation directory) and must write its trajectory there.
//
// Abstracting this lets the Go port run a Python target agent, a different
// interpreter, or a test fake — without changing the orchestrator.
type TargetExecutor interface {
	// RunTarget runs the agent at agentPath with the fixed CLI contract,
	// streaming combined output to stdoutLog. It returns a [TargetResult]; a
	// non-zero exit is reported as Success=false (not a Go error), matching the
	// reference's "continue with feedback despite target failure" behavior. A Go
	// error is returned only when the executor itself cannot run.
	RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error)
}

// TargetRequest is one target-agent invocation.
type TargetRequest struct {
	AgentPath  string // path to target_agent.py / train.py
	DatasetDir string // absolute --dataset_dir (read-only)
	WorkingDir string // absolute --working_dir (read-write; the gen dir)
	StdoutLog  string // path the combined stdout/stderr is written to
}

// ExecTargetExecutor runs the target agent as a subprocess of an interpreter
// (e.g. python). It is the production target executor and replaces the
// reference's venv-python invocation; the interpreter is configurable so the Go
// port does not assume a managed venv.
type ExecTargetExecutor struct {
	// Interpreter is the executable that runs the agent (e.g. "python3"). If
	// empty, the agent file is executed directly (it must be executable).
	Interpreter string
	// InterpreterArgs are prepended before the agent path (e.g. ["-u"] for
	// unbuffered output, matching the reference).
	InterpreterArgs []string
	// Env is extra environment appended to os.Environ (e.g. SANDBOX_URL).
	Env []string
}

// RunTarget runs the target agent with the fixed --dataset_dir/--working_dir
// contract and streams its output to the log file.
func (e *ExecTargetExecutor) RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error) {
	if !isFile(req.AgentPath) {
		return TargetResult{Success: false, ErrorMsg: fmt.Sprintf("Target agent file not found: %s", req.AgentPath)},
			fmt.Errorf("target agent file not found: %s", req.AgentPath)
	}

	name, args := e.commandFor(req.AgentPath)
	args = append(args, "--dataset_dir", req.DatasetDir, "--working_dir", req.WorkingDir)

	logFile, err := os.Create(req.StdoutLog)
	if err != nil {
		return TargetResult{}, fmt.Errorf("create stdout log: %w", err)
	}
	defer logFile.Close()

	var buf bytes.Buffer
	out := io.MultiWriter(logFile, &buf)

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = req.WorkingDir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.Env = append(os.Environ(), e.Env...)

	runErr := cmd.Run()
	stdout := buf.String()
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			msg := fmt.Sprintf("Target agent failed with exit code %d", exitErr.ExitCode())
			return TargetResult{Success: false, Stdout: stdout, ErrorMsg: msg}, nil
		}
		// The executor could not run the process at all.
		return TargetResult{Success: false, Stdout: stdout, ErrorMsg: runErr.Error()}, fmt.Errorf("run target agent: %w", runErr)
	}
	return TargetResult{Success: true, Stdout: stdout}, nil
}

// commandFor returns the executable and leading args for running agentPath.
func (e *ExecTargetExecutor) commandFor(agentPath string) (string, []string) {
	if e.Interpreter == "" {
		return agentPath, nil
	}
	args := append([]string(nil), e.InterpreterArgs...)
	args = append(args, agentPath)
	return e.Interpreter, args
}

// FuncTargetExecutor adapts a function to a [TargetExecutor], for tests that
// simulate a target agent by writing trajectory files directly.
type FuncTargetExecutor func(ctx context.Context, req TargetRequest) (TargetResult, error)

// RunTarget calls the wrapped function.
func (f FuncTargetExecutor) RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error) {
	return f(ctx, req)
}
