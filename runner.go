package sia

import (
	"context"
	"fmt"
	"os"
	"os/exec"
)

// AgentRunner runs a meta or feedback agent: it is given a prompt and a working
// directory and is expected to produce files in that directory (a
// target_agent.py for the meta agent; improvement.md + target_agent.py, or
// train.py, for the feedback agent). This is the seam the reference exposes as
// "agent impls"; the orchestrator drives it without knowing whether the engine
// is a local CLI, a hosted API, or a test fake.
//
// A runner must honor ctx cancellation. It returns an error only when the engine
// itself fails to run; whether the agent produced the right files is the
// orchestrator's concern (it inspects the working directory afterward).
type AgentRunner interface {
	// Name is the agent-impl id (e.g. "claude") used for profile validation.
	Name() string
	// Run drives the agent for one invocation.
	Run(ctx context.Context, req AgentRequest) error
}

// AgentRequest is one invocation of an [AgentRunner].
type AgentRequest struct {
	Model      string   // model the engine should drive
	Prompt     string   // the full prompt text
	WorkingDir string   // cwd for the agent; it writes its output files here
	MaxTurns   int      // turn budget (0 means the runner's default)
	Provider   Provider // endpoint/credentials for the engine
}

// ExecRunner runs an external agent CLI as a subprocess, passing the prompt on
// stdin and running it in the request's working directory. It is the production
// runner: it shells out to a command (e.g. the `claude` CLI) rather than
// embedding a model SDK, so the Go port has no model-client dependency.
//
// The command is invoked as: Command Args... with the prompt written to stdin.
// %MODEL%, %MAXTURNS%, and %WORKDIR% tokens in Args are replaced per request.
// Stdout/stderr stream to Stdout/Stderr (defaulting to the process's).
type ExecRunner struct {
	ImplName string   // value reported by Name() (defaults to "exec")
	Command  string   // executable, e.g. "claude"
	Args     []string // arguments; %MODEL%/%MAXTURNS%/%WORKDIR% are substituted
	Stdout   *os.File // defaults to os.Stdout
	Stderr   *os.File // defaults to os.Stderr
	Env      []string // extra environment (appended to os.Environ); empty inherits
}

// Name reports the agent-impl id.
func (r *ExecRunner) Name() string {
	if r.ImplName == "" {
		return "exec"
	}
	return r.ImplName
}

// Run executes the configured command with the prompt on stdin.
func (r *ExecRunner) Run(ctx context.Context, req AgentRequest) error {
	if r.Command == "" {
		return fmt.Errorf("exec runner: no command configured")
	}
	args := make([]string, len(r.Args))
	repl := newTokenReplacer(req)
	for i, a := range r.Args {
		args[i] = repl.Replace(a)
	}

	cmd := exec.CommandContext(ctx, r.Command, args...)
	cmd.Dir = req.WorkingDir
	cmd.Stdin = stringReader(req.Prompt)
	cmd.Stdout = orStdout(r.Stdout)
	cmd.Stderr = orStderr(r.Stderr)
	cmd.Env = append(os.Environ(), r.Env...)

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("exec runner: %s: %w", r.Command, err)
	}
	return nil
}

// FuncRunner adapts a function to an [AgentRunner]. It is the simplest way to
// supply a custom or scripted engine — see [FakeRunner] for the test-oriented
// variant that writes canned files.
type FuncRunner struct {
	ImplName string
	Fn       func(ctx context.Context, req AgentRequest) error
}

// Name reports the agent-impl id (defaults to "func").
func (r FuncRunner) Name() string {
	if r.ImplName == "" {
		return "func"
	}
	return r.ImplName
}

// Run calls the wrapped function.
func (r FuncRunner) Run(ctx context.Context, req AgentRequest) error {
	if r.Fn == nil {
		return fmt.Errorf("func runner: no function configured")
	}
	return r.Fn(ctx, req)
}
