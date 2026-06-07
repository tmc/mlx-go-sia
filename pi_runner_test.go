package sia

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakePiScript writes a hermetic stand-in for scripts/pi-mlx that records its
// stdin (the prompt), working directory, and the PI_MLX_MODEL it was invoked
// with, so a test can assert the contract without driving the real model. It
// returns the script path and the three record-file paths.
func fakePiScript(t *testing.T) (script, promptFile, cwdFile, modelFile string) {
	t.Helper()
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	dir := t.TempDir()
	promptFile = filepath.Join(dir, "prompt.txt")
	cwdFile = filepath.Join(dir, "cwd.txt")
	modelFile = filepath.Join(dir, "model.txt")
	script = filepath.Join(dir, "pi-mlx")
	writeTestFile(t, script, "#!/bin/sh\n"+
		"cat > '"+promptFile+"'\n"+
		"pwd > '"+cwdFile+"'\n"+
		"printf '%s' \"$PI_MLX_MODEL\" > '"+modelFile+"'\n")
	if err := os.Chmod(script, 0o755); err != nil {
		t.Fatal(err)
	}
	return script, promptFile, cwdFile, modelFile
}

// TestPiRunnerName checks the agent-impl id used for profile validation.
func TestPiRunnerName(t *testing.T) {
	if got := NewPiRunner("").Name(); got != "pi-mlx" {
		t.Errorf("Name() = %q, want %q", got, "pi-mlx")
	}
}

// TestPiRunnerRun runs PiRunner against the fake wrapper and confirms the prompt
// reaches stdin, the request's working directory is honored, and the effective
// model is exported as PI_MLX_MODEL with request > runner > default precedence.
func TestPiRunnerRun(t *testing.T) {
	script, promptFile, cwdFile, modelFile := fakePiScript(t)
	workDir := t.TempDir()
	// macOS routes TempDir through /var -> /private/var; resolve symlinks so the
	// `pwd` the script reports compares equal to the directory we requested.
	wantCwd, err := filepath.EvalSymlinks(workDir)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name        string
		runnerModel string // model passed to NewPiRunner
		reqModel    string // model on the request (overrides runner)
		wantModel   string
	}{
		{"default", "", "", DefaultPiModel},
		{"runner override", "runner/model", "", "runner/model"},
		{"request wins", "runner/model", "req/model", "req/model"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := NewPiRunner(c.runnerModel)
			r.Script = script
			err := r.Run(context.Background(), AgentRequest{
				Model:      c.reqModel,
				Prompt:     "improve the target agent",
				WorkingDir: workDir,
			})
			if err != nil {
				t.Fatalf("Run: %v", err)
			}

			if got := readTestFile(t, promptFile); strings.TrimSpace(got) != "improve the target agent" {
				t.Errorf("prompt on stdin = %q, want %q", got, "improve the target agent")
			}
			if got := strings.TrimSpace(readTestFile(t, cwdFile)); got != wantCwd {
				t.Errorf("working dir = %q, want %q", got, wantCwd)
			}
			if got := readTestFile(t, modelFile); got != c.wantModel {
				t.Errorf("PI_MLX_MODEL = %q, want %q", got, c.wantModel)
			}
		})
	}
}

// TestPiRunnerStdout confirms the runner routes the wrapper's stdout to the
// configured Stdout file rather than the process stdout, the path the
// orchestrator uses to capture generated text.
func TestPiRunnerStdout(t *testing.T) {
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "pi-mlx")
	writeTestFile(t, script, "#!/bin/sh\ncat >/dev/null\nprintf 'generated text'\n")
	if err := os.Chmod(script, 0o755); err != nil {
		t.Fatal(err)
	}
	out, err := os.CreateTemp(dir, "stdout-*")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	r := NewPiRunner("")
	r.Script = script
	r.Stdout = out
	if err := r.Run(context.Background(), AgentRequest{WorkingDir: dir}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := readTestFile(t, out.Name()); got != "generated text" {
		t.Errorf("Stdout = %q, want %q", got, "generated text")
	}
}

// TestPiRunnerContextCancellation confirms the runner honors ctx cancellation
// (the AgentRunner contract): a canceled context aborts the wrapper and Run
// returns an error.
func TestPiRunnerContextCancellation(t *testing.T) {
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "pi-mlx")
	writeTestFile(t, script, "#!/bin/sh\ncat >/dev/null\nsleep 30\n")
	if err := os.Chmod(script, 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already canceled: the wrapper must not be allowed to run to completion
	r := NewPiRunner("")
	r.Script = script
	if err := r.Run(ctx, AgentRequest{WorkingDir: dir}); err == nil {
		t.Fatal("expected error from canceled context, got nil")
	}
}

// TestPiRunnerMissingScript confirms a runner pointed at a nonexistent wrapper
// reports an error (rather than silently succeeding), since a failed engine
// must surface to the orchestrator.
func TestPiRunnerMissingScript(t *testing.T) {
	r := NewPiRunner("")
	r.Script = filepath.Join(t.TempDir(), "does-not-exist")
	err := r.Run(context.Background(), AgentRequest{WorkingDir: t.TempDir()})
	if err == nil {
		t.Fatal("expected error for missing wrapper, got nil")
	}
	if !strings.Contains(err.Error(), "pi-mlx runner") {
		t.Errorf("error %q should mention the pi-mlx runner", err)
	}
}

// TestPiRunnerScriptPathResolution documents the integration rule the example
// commands rely on: because the wrapper runs in the request's working directory (a
// run/generation dir), a relative Script resolves against that dir — not the
// repository — and is not found, while an absolute Script works from any
// working directory.
func TestPiRunnerScriptPathResolution(t *testing.T) {
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	// A real wrapper living in scriptDir, addressed two ways.
	scriptDir := t.TempDir()
	abs := filepath.Join(scriptDir, "pi-mlx")
	writeTestFile(t, abs, "#!/bin/sh\ncat >/dev/null\n")
	if err := os.Chmod(abs, 0o755); err != nil {
		t.Fatal(err)
	}
	// WorkingDir is a *different* directory, standing in for runs/run_1/gen_1.
	workDir := t.TempDir()

	t.Run("relative path is not found from a foreign working dir", func(t *testing.T) {
		r := NewPiRunner("")
		r.Script = "pi-mlx" // bare/relative: resolved against workDir, where it does not exist
		if err := r.Run(context.Background(), AgentRequest{WorkingDir: workDir}); err == nil {
			t.Fatal("expected error for relative script not in working dir, got nil")
		}
	})
	t.Run("absolute path runs from any working dir", func(t *testing.T) {
		r := NewPiRunner("")
		r.Script = abs
		if err := r.Run(context.Background(), AgentRequest{WorkingDir: workDir}); err != nil {
			t.Fatalf("absolute script should run: %v", err)
		}
	})
}

func readTestFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}
