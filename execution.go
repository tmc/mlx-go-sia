package sia

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Execution is the result of loading a generation's recorded agent execution.
// Exactly one of Single / Multi is populated, indicated by MultiTrajectory.
type Execution struct {
	// MultiTrajectory reports whether the generation used the per-sample
	// agent_execution/ folder (true) or the single agent_execution.json (false).
	MultiTrajectory bool

	// Single holds the parsed single-file trajectory (MultiTrajectory == false).
	// On a parse or read problem it carries a diagnostic object with an "error"
	// field, matching the reference's degraded-but-usable behavior.
	Single json.RawMessage

	// Trajectories holds the parsed per-sample files in filename order
	// (MultiTrajectory == true). An unreadable file becomes an error object.
	Trajectories []json.RawMessage
}

// LoadExecution loads a generation's execution log with automatic format
// detection, mirroring the reference's load_agent_execution:
//
//   - if genDir/agent_execution/ exists, it is multi-trajectory: every
//     execution_q*.json is loaded in sorted filename order;
//   - else if genDir/agent_execution.json exists, it is single-trajectory;
//   - else the result carries a single error object.
//
// maxLogSize caps each file; an oversized file is replaced by an error object
// rather than read. LoadExecution never returns a nil error for a missing or
// malformed log — the degraded data is carried in the [Execution] so the
// feedback agent can still reason about the failure.
func LoadExecution(genDir string, maxLogSize int64) Execution {
	execDir := filepath.Join(genDir, NameAgentExecutionDir)
	execFile := filepath.Join(genDir, NameAgentExecution)

	if isDir(execDir) {
		return loadMultiTrajectory(execDir, maxLogSize)
	}
	if isFile(execFile) {
		return Execution{Single: loadOneTrajectory(execFile, maxLogSize)}
	}
	return Execution{Single: errorObject(map[string]any{"error": "Execution log not found"})}
}

func loadMultiTrajectory(execDir string, maxLogSize int64) Execution {
	matches, _ := filepath.Glob(filepath.Join(execDir, NameExecutionGlob))
	sort.Strings(matches)
	if len(matches) == 0 {
		return Execution{
			MultiTrajectory: true,
			Trajectories:    []json.RawMessage{errorObject(map[string]any{"error": "Empty execution folder", "type": "multi-trajectory"})},
		}
	}
	trajectories := make([]json.RawMessage, 0, len(matches))
	for _, f := range matches {
		trajectories = append(trajectories, loadOneTrajectory(f, maxLogSize))
	}
	return Execution{MultiTrajectory: true, Trajectories: trajectories}
}

// loadOneTrajectory reads and validates one trajectory file, returning the raw
// JSON on success or an error object describing the failure.
func loadOneTrajectory(path string, maxLogSize int64) json.RawMessage {
	base := filepath.Base(path)
	info, err := os.Stat(path)
	if err != nil {
		return errorObject(map[string]any{"error": err.Error(), "file": base})
	}
	if maxLogSize > 0 && info.Size() > maxLogSize {
		return errorObject(map[string]any{"error": "File too large", "file": base, "size": info.Size()})
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return errorObject(map[string]any{"error": err.Error(), "file": base})
	}
	if !json.Valid(data) {
		preview := data
		if len(preview) > 1000 {
			preview = preview[:1000]
		}
		return errorObject(map[string]any{
			"error":       "Parse error",
			"file":        base,
			"raw_preview": string(preview),
			"file_size":   len(data),
		})
	}
	return json.RawMessage(data)
}

// errorObject marshals a diagnostic map into a json.RawMessage. The map is
// fully serializable, so the error is dropped.
func errorObject(m map[string]any) json.RawMessage {
	b, err := json.Marshal(m)
	if err != nil {
		return json.RawMessage(fmt.Sprintf(`{"error":%q}`, err.Error()))
	}
	return b
}

// hasError reports whether raw is a JSON object carrying a truthy "error" field,
// matching the reference's `isinstance(t, dict) and t.get("error")` check: an
// empty string, null, false, or 0 is falsy and so is not treated as an error.
func hasError(raw json.RawMessage) bool {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return false
	}
	return jsonTruthy(obj["error"])
}

// jsonTruthy reports whether raw is present and Python-truthy: a non-empty
// string, a non-zero number, true, a non-empty array/object. A missing key
// (nil raw), null, "", 0, false, [], or {} is falsy.
func jsonTruthy(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return false
	}
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case float64:
		return x != 0
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	default:
		return true
	}
}

// isList reports whether raw is a JSON array (a well-formed trajectory; the
// reference records a trajectory as a list of messages).
func isList(raw json.RawMessage) bool {
	var arr []json.RawMessage
	return json.Unmarshal(raw, &arr) == nil
}

// Summary returns the trajectory count and the successful/failed split for a
// multi-trajectory execution, mirroring the reference: a trajectory counts as
// successful when it is a JSON list and failed when it is an error object. For a
// single-file execution it reports a count of one and whether the single
// trajectory parsed cleanly.
func (e Execution) Summary() (count, successful, failed int) {
	if !e.MultiTrajectory {
		if hasError(e.Single) {
			return 1, 0, 1
		}
		return 1, 1, 0
	}
	count = len(e.Trajectories)
	for _, t := range e.Trajectories {
		switch {
		case isList(t):
			successful++
		case hasError(t):
			failed++
		}
	}
	return count, successful, failed
}
