package sia

import (
	"os"
	"path/filepath"
	"testing"
)

// writeTestFile writes content to path, creating parent directories.
func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadExecutionSingle(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), `[{"role":"user","content":"hi"}]`)

	e := LoadExecution(dir, 0)
	if e.MultiTrajectory {
		t.Fatal("expected single-trajectory")
	}
	count, ok, fail := e.Summary()
	if count != 1 || ok != 1 || fail != 0 {
		t.Errorf("summary = (%d,%d,%d), want (1,1,0)", count, ok, fail)
	}
}

func TestLoadExecutionMulti(t *testing.T) {
	dir := t.TempDir()
	execDir := filepath.Join(dir, NameAgentExecutionDir)
	writeTestFile(t, filepath.Join(execDir, "execution_q0.json"), `[{"role":"user"}]`)
	writeTestFile(t, filepath.Join(execDir, "execution_q1.json"), `[{"role":"assistant"}]`)
	writeTestFile(t, filepath.Join(execDir, "execution_q2.json"), `{"error":"boom"}`) // a failed trajectory

	e := LoadExecution(dir, 0)
	if !e.MultiTrajectory {
		t.Fatal("expected multi-trajectory")
	}
	count, ok, fail := e.Summary()
	if count != 3 || ok != 2 || fail != 1 {
		t.Errorf("summary = (%d,%d,%d), want (3,2,1)", count, ok, fail)
	}
}

func TestLoadExecutionMultiTakesPrecedence(t *testing.T) {
	// When both the folder and the single file exist, the folder wins (matches
	// the reference's detection order).
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), `[]`)
	writeTestFile(t, filepath.Join(dir, NameAgentExecutionDir, "execution_q0.json"), `[]`)
	e := LoadExecution(dir, 0)
	if !e.MultiTrajectory {
		t.Error("folder should take precedence over single file")
	}
}

func TestLoadExecutionMissing(t *testing.T) {
	e := LoadExecution(t.TempDir(), 0)
	if e.MultiTrajectory {
		t.Fatal("missing log should be reported as single with an error object")
	}
	if !hasError(e.Single) {
		t.Errorf("missing log should carry an error object, got %s", e.Single)
	}
}

func TestLoadExecutionOversized(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), `[{"big":"payload-exceeding-cap"}]`)
	e := LoadExecution(dir, 4) // tiny cap forces "File too large"
	if !hasError(e.Single) {
		t.Errorf("oversized log should carry an error object, got %s", e.Single)
	}
}

func TestLoadExecutionMalformed(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), `[{"role": "user"`) // truncated
	e := LoadExecution(dir, 0)
	if !hasError(e.Single) {
		t.Errorf("malformed log should carry an error object, got %s", e.Single)
	}
}
