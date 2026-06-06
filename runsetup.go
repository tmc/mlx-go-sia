package sia

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadTaskFiles loads the task reference files the prompts are built from. The
// seed shown to the meta agent comes from resolved (its InlineSeed for a
// default/file reference, empty for a directory reference the agent reads from
// disk).
func LoadTaskFiles(task TaskLayout, resolved ResolvedAgentReference) (TaskFiles, error) {
	sampleDesc, err := os.ReadFile(task.SampleDescriptions())
	if err != nil {
		return TaskFiles{}, fmt.Errorf("read sample descriptions: %w", err)
	}
	taskMD, err := os.ReadFile(task.TaskMD())
	if err != nil {
		return TaskFiles{}, fmt.Errorf("read task.md: %w", err)
	}
	sampleExecBytes, err := os.ReadFile(task.SampleExecution())
	if err != nil {
		return TaskFiles{}, fmt.Errorf("read sample execution: %w", err)
	}
	if !json.Valid(sampleExecBytes) {
		return TaskFiles{}, fmt.Errorf("sample execution %s is not valid JSON", task.SampleExecution())
	}

	return TaskFiles{
		SampleTaskDescriptions: string(sampleDesc),
		ReferenceTargetAgentPy: resolved.InlineSeed,
		SampleAgentExecution:   json.RawMessage(sampleExecBytes),
		TaskMD:                 string(taskMD),
	}, nil
}

// SetupRunDirectory creates the run directory and generation-1 working
// directory, refusing to overwrite an existing run. It returns the run layout.
//
// Unlike the reference, no Python venv is created: the Go port's target executor
// and evaluator are configured externally (interpreter, env), so dependency
// management is the operator's concern, not the orchestrator's.
func SetupRunDirectory(runsRoot string, runID int) (RunLayout, error) {
	layout := RunLayoutForID(runsRoot, runID)
	if _, err := os.Stat(layout.RunDir); err == nil {
		return RunLayout{}, fmt.Errorf("run directory already exists: %s (use a different run_id or remove it)", layout.RunDir)
	}
	if err := os.MkdirAll(layout.GenDir(1), 0o755); err != nil {
		return RunLayout{}, fmt.Errorf("create run directories: %w", err)
	}
	return layout, nil
}
