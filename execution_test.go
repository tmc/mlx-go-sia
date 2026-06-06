package sia

import (
	"encoding/json"
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

// fields decodes a json.RawMessage into a map for shape assertions.
func fields(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("error object is not an object: %s (%v)", raw, err)
	}
	return m
}

// TestLoadExecutionSingleMalformedShape pins the exact single-file parse-error
// contract from the reference (orchestrator.py load_agent_execution, locked by
// tests/test_load_execution_formats.py:test_malformed_single_returns_partial_preview):
// {"error":"Parse error","raw_preview":<raw>,"parse_error":<msg>,"file_size":N}
// with NO "file" key.
func TestLoadExecutionSingleMalformedShape(t *testing.T) {
	dir := t.TempDir()
	raw := "{not valid json"
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), raw)
	e := LoadExecution(dir, 0)
	if e.MultiTrajectory {
		t.Fatal("expected single-trajectory")
	}
	m := fields(t, e.Single)
	if m["error"] != "Parse error" {
		t.Errorf(`error = %v, want "Parse error"`, m["error"])
	}
	if m["raw_preview"] != raw {
		t.Errorf("raw_preview = %v, want %q", m["raw_preview"], raw)
	}
	if got := m["file_size"]; got != float64(len(raw)) {
		t.Errorf("file_size = %v, want %d", got, len(raw))
	}
	if _, ok := m["parse_error"]; !ok {
		t.Error(`missing "parse_error" key (reference includes the decoder message)`)
	}
	if _, ok := m["file"]; ok {
		t.Error(`single-file parse error must NOT carry a "file" key`)
	}
}

// TestLoadExecutionSingleOversizedShape pins the single-file oversized shape:
// {"error":"File too large","size":N} with NO "file" key.
func TestLoadExecutionSingleOversizedShape(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecution), `[{"big":"payload-exceeding-cap"}]`)
	e := LoadExecution(dir, 4)
	m := fields(t, e.Single)
	if m["error"] != "File too large" {
		t.Errorf(`error = %v, want "File too large"`, m["error"])
	}
	if _, ok := m["size"]; !ok {
		t.Error(`missing "size" key`)
	}
	if _, ok := m["file"]; ok {
		t.Error(`single-file oversized error must NOT carry a "file" key`)
	}
}

// TestLoadExecutionMultiMalformedShape pins the per-file multi-trajectory
// parse-error contract: {"error":<decoder message>,"file":<basename>} with NO
// raw_preview/file_size (matches orchestrator.py's multi branch).
func TestLoadExecutionMultiMalformedShape(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, NameAgentExecutionDir, "execution_q0.json"), `[{"role":"user"}]`)
	writeTestFile(t, filepath.Join(dir, NameAgentExecutionDir, "execution_q1.json"), `{bad json`)
	e := LoadExecution(dir, 0)
	if !e.MultiTrajectory || len(e.Trajectories) != 2 {
		t.Fatalf("expected 2 multi trajectories, got multi=%v n=%d", e.MultiTrajectory, len(e.Trajectories))
	}
	m := fields(t, e.Trajectories[1])
	if m["file"] != "execution_q1.json" {
		t.Errorf(`file = %v, want "execution_q1.json"`, m["file"])
	}
	if m["error"] == "Parse error" || m["error"] == nil || m["error"] == "" {
		t.Errorf("multi parse error should be the decoder message, got %v", m["error"])
	}
	if _, ok := m["raw_preview"]; ok {
		t.Error(`multi per-file parse error must NOT carry "raw_preview"`)
	}
}
