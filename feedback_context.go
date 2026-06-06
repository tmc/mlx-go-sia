package sia

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// fence is a markdown code fence; spelled as a const so format strings below can
// stay plain (Go raw-string literals cannot contain a backtick).
const fence = "```"

// feedbackContext builds the (execution_status, execution_section) blocks the
// feedback prompt is assembled from, porting the reference's
// _build_feedback_context. execution_section renders the trajectory (single or
// multi); execution_status reports SUCCESS/FAILED with evaluation results and a
// tail of the target's stdout.
func (o *Orchestrator) feedbackContext(opts RunOptions, gen int, tr TargetResult, ev EvalResult, exec Execution) (status, section string) {
	previewLimit := DefaultConfig().TrajectoryPreviewLimit

	section = renderExecutionSection(opts.Layout, gen, exec, previewLimit)
	evalSection := renderEvalSection(opts.Layout, gen, ev, opts.MaxLogSize)
	status = renderExecutionStatus(tr, evalSection, opts.Layout.StdoutLog(gen, opts.Focus))
	return status, section
}

// renderExecutionSection renders the trajectory block, matching the reference's
// single- vs multi-trajectory formatting.
func renderExecutionSection(layout RunLayout, gen int, exec Execution, previewLimit int) string {
	if exec.MultiTrajectory {
		count, successful, failed := exec.Summary()
		var samples strings.Builder
		for i, traj := range exec.Trajectories {
			if i >= 3 {
				break
			}
			tj := truncate(jsonIndent(traj), previewLimit)
			fmt.Fprintf(&samples, "\n### Trajectory %d\n%sjson\n%s\n%s\n", i, fence, tj, fence)
		}
		execDir := layout.AgentExecutionDir(gen)
		return strings.Join([]string{
			"",
			"**MULTI-TRAJECTORY EXECUTION**:",
			"",
			fmt.Sprintf("The agent executed %d separate trajectories (e.g., different questions/samples).", count),
			"",
			"**Summary**:",
			fmt.Sprintf("- Total trajectories: %d", count),
			fmt.Sprintf("- Successful: %d", successful),
			fmt.Sprintf("- Failed: %d", failed),
			fmt.Sprintf("- Execution folder: %s", execDir),
			"",
			"**Sample Trajectories** (first 3 shown, you can read others from the folder):",
			samples.String(),
			"",
			"**To analyze all trajectories**:",
			fmt.Sprintf("- Read files from: %s", execDir),
			fmt.Sprintf("- Files named: execution_q0.json, execution_q1.json, ..., execution_q%d.json", count-1),
			"",
			"**Analysis guidance**:",
			"- Look for common failure patterns across trajectories",
			"- Check if trajectories are properly isolated",
			"- Ensure consistent behavior across all samples",
			"",
		}, "\n")
	}

	tj := truncate(jsonIndent(exec.Single), previewLimit)
	return strings.Join([]string{
		"",
		"Here is the target agent execution trajectory:",
		fence + "json",
		tj,
		fence,
		"",
		`NOTE: If you see an "error" field in the above JSON, it means the execution log was malformed or missing. Focus on making the agent more robust.`,
		"",
	}, "\n")
}

// renderEvalSection renders the evaluation-results block from results.json,
// matching the reference's behavior when the file is present/absent/oversized.
func renderEvalSection(layout RunLayout, gen int, _ EvalResult, maxLogSize int64) string {
	resultsPath := layout.ResultsJSON(gen)
	info, err := os.Stat(resultsPath)
	if err != nil {
		return "\n**EVALUATION RESULTS**: No results.json found (evaluation may not have run or may have failed)\n"
	}
	if maxLogSize > 0 && info.Size() > maxLogSize {
		return fmt.Sprintf("\n**EVALUATION RESULTS**: results.json too large (%d bytes)\n", info.Size())
	}
	data, err := os.ReadFile(resultsPath)
	if err != nil {
		return fmt.Sprintf("\n**EVALUATION RESULTS**: Error loading results.json: %v\n", err)
	}
	if !json.Valid(data) {
		return "\n**EVALUATION RESULTS**: Error loading results.json: invalid JSON\n"
	}
	return fmt.Sprintf("\n\n**EVALUATION RESULTS**:\n%sjson\n%s\n%s\n", fence, jsonIndent(data), fence)
}

// renderExecutionStatus renders the SUCCESS/FAILED status block with a tail of
// the target agent's stdout.
func renderExecutionStatus(tr TargetResult, evalSection, stdoutLog string) string {
	last10 := lastLines(tr.Stdout, 10)
	if tr.Success {
		return strings.Join([]string{
			"SUCCESS: Target agent completed execution successfully.",
			evalSection,
			"",
			"**Last 10 lines of output**:",
			fence,
			last10,
			fence,
			"",
			fmt.Sprintf("Full logs available at: %s", stdoutLog),
			"",
		}, "\n")
	}
	return strings.Join([]string{
		fmt.Sprintf("FAILED: %s", tr.ErrorMsg),
		evalSection,
		"",
		"**Last 10 lines of output**:",
		fence,
		last10,
		fence,
		"",
		fmt.Sprintf("Full logs available at: %s", stdoutLog),
		"",
		"STDERR:",
		tr.Stderr,
		"",
	}, "\n")
}

// truncate clips s to limit characters with a trailing marker, matching the
// reference's preview truncation.
func truncate(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	return s[:limit] + "\n  ... (truncated)"
}

// lastLines returns the last n lines of s (or all of s if it has <= n lines),
// matching the reference's "\n".join(lines[-10:]) when there are more than n.
func lastLines(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) <= n {
		return s
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}
