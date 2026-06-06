package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	sia "github.com/tmc/mlx-go-sia"
)

// WeightsEvaluator scores a weights-mode generation on a HELD-OUT test set the
// agent can never see. It loads the adapter the generation trained (written into
// the agent's working directory by mlx-lm-train) and evaluates it against
// test.jsonl kept in a read-only directory OUTSIDE the agent's reach, then
// writes the held-out loss to results.json.
//
// Honesty discipline (P6 spec): the agent's training executor is given a data
// directory with train/valid only; this evaluator owns the held-out test split.
// Because mlx-lm-train can evaluate a resumed adapter on a test set without
// training (-test without -train), the evaluator computes a real generalization
// metric the agent cannot have memorized — the model never saw the eval rows.
//
// It implements [sia.Evaluator]. A missing adapter or a failed eval is reported
// via results.json + EvalResult.Status, never as a Go error (which would abort
// the run); a Go error is returned only when the evaluator cannot start at all.
type WeightsEvaluator struct {
	// TrainBin is the mlx-lm-train executable used in eval-only mode. Empty
	// defaults to "mlx-lm-train" (resolved on PATH).
	TrainBin string
	// BaseModel is the base model the adapter attaches to. Required.
	BaseModel string
	// HeldOutDir is the read-only directory holding test.jsonl (and nothing the
	// agent's training was given). Required. It must live outside any agent
	// WorkingDir.
	HeldOutDir string
	// AdapterSubdir is the path, relative to the generation's WorkingDir, where
	// the executor wrote the trained adapter. Defaults to "adapters" (matching
	// MLXTrainExecutor).
	AdapterSubdir string
	// TestBatches caps the test batches (-1 = all). Defaults to -1.
	TestBatches int
	// Env is extra environment appended to os.Environ for the eval process.
	Env []string
	// DryRun skips invoking mlx-lm-train and writes a placeholder result, so the
	// loop is demonstrable without a GPU/model download. Off for a real run.
	DryRun bool
}

// weightsResults is the results.json schema for a weights generation. The
// top-level scalars flow to the feedback agent; lower held-out loss is better,
// so improvement is a decreasing test_loss across generations.
type weightsResults struct {
	Verdict    string  `json:"verdict"`
	Trained    bool    `json:"trained"`
	TestLoss   float64 `json:"test_loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
	Metric     string  `json:"metric"`
	Reason     string  `json:"reason,omitempty"`
	HeldOut    string  `json:"held_out_dir"`
}

// Evaluate loads the generation's trained adapter and scores it on the held-out
// test set, writing results.json into genDir.
func (e *WeightsEvaluator) Evaluate(ctx context.Context, genDir string) (sia.EvalResult, error) {
	if e.BaseModel == "" {
		return sia.EvalResult{}, fmt.Errorf("weights evaluator: BaseModel is required")
	}
	if e.HeldOutDir == "" {
		return sia.EvalResult{}, fmt.Errorf("weights evaluator: HeldOutDir is required")
	}

	sub := e.AdapterSubdir
	if sub == "" {
		sub = "adapters"
	}
	adapterDir := filepath.Join(genDir, sub)

	if e.DryRun {
		return e.write(genDir, weightsResults{
			Verdict: "SKIPPED", Trained: false, Metric: "test_loss",
			Reason:  "dry-run: held-out eval not executed (no training/GPU)",
			HeldOut: e.HeldOutDir,
		})
	}

	if !isDir(adapterDir) {
		// No adapter produced (training did not run or failed): report as feedback.
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: false, Metric: "test_loss",
			Reason:  fmt.Sprintf("no trained adapter at %s; training likely failed", adapterDir),
			HeldOut: e.HeldOutDir,
		})
	}

	loss, out, err := e.evalHeldOut(ctx, adapterDir)
	logPath := filepath.Join(genDir, sia.NameEvalLog)
	_ = os.WriteFile(logPath, []byte(out), 0o644)
	if err != nil {
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: true, Metric: "test_loss",
			Reason:  fmt.Sprintf("held-out eval failed: %v", err),
			HeldOut: e.HeldOutDir,
		})
	}

	res := weightsResults{
		Verdict:    "PASS",
		Trained:    true,
		TestLoss:   loss,
		Perplexity: perplexity(loss),
		Metric:     "test_loss",
		HeldOut:    e.HeldOutDir,
	}
	return e.write(genDir, res)
}

// evalHeldOut runs mlx-lm-train in eval-only mode (-test, no -train) with the
// generation's adapter resumed, pointing -data at the held-out directory. It
// returns the parsed test loss and the combined output.
func (e *WeightsEvaluator) evalHeldOut(ctx context.Context, adapterDir string) (float64, string, error) {
	bin := e.TrainBin
	if bin == "" {
		bin = "mlx-lm-train"
	}
	batches := e.TestBatches
	if batches == 0 {
		batches = -1
	}
	args := []string{
		"-test",
		"-model", e.BaseModel,
		"-data", e.HeldOutDir,
		"-adapter-path", adapterDir,
		"-resume-adapter-file", filepath.Join(adapterDir, "adapters.safetensors"),
		"-test-batches", strconv.Itoa(batches),
	}

	var buf bytes.Buffer
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = append(os.Environ(), e.Env...)
	runErr := cmd.Run()
	out := buf.String()
	if runErr != nil {
		return 0, out, fmt.Errorf("%v", runErr)
	}
	loss, perr := parseTestLoss(out)
	if perr != nil {
		return 0, out, perr
	}
	return loss, out, nil
}

// testLossRe matches a reported test loss line, tolerant of mlx-lm-train's
// phrasing ("Test loss 1.234", "test_loss: 1.234").
var testLossRe = regexp.MustCompile(`(?i)test[ _]loss[:= ]+([0-9]+\.?[0-9]*)`)

// parseTestLoss extracts the held-out test loss from the eval output.
func parseTestLoss(out string) (float64, error) {
	if m := testLossRe.FindStringSubmatch(out); m != nil {
		return strconv.ParseFloat(m[1], 64)
	}
	// Fallback: scan for a "Test ... <float>" line.
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(strings.ToLower(line), "test") && strings.Contains(strings.ToLower(line), "loss") {
			for _, f := range strings.Fields(line) {
				if v, err := strconv.ParseFloat(f, 64); err == nil {
					return v, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("no test loss in eval output")
}

// perplexity is exp(loss); a friendlier number for the demo chart.
func perplexity(loss float64) float64 { return math.Exp(loss) }

// write marshals res into genDir/results.json and returns the EvalResult.
func (e *WeightsEvaluator) write(genDir string, res weightsResults) (sia.EvalResult, error) {
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return sia.EvalResult{Status: sia.EvalError, Reason: fmt.Sprintf("marshal results: %v", err)}, nil
	}
	path := filepath.Join(genDir, sia.NameResultsJSON)
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return sia.EvalResult{Status: sia.EvalError, Reason: fmt.Sprintf("write results.json: %v", err)}, nil
	}
	status := sia.EvalSuccess
	if res.Verdict == "REVISE" || res.Verdict == "SKIPPED" {
		// A REVISE/SKIPPED still produced results.json; treat it as a warning so
		// the feedback agent sees the reason without the run being marked failed.
		status = sia.EvalWarning
	}
	return sia.EvalResult{Status: status, ResultsPath: path, Output: string(data)}, nil
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// readWeightsResults loads a generation's results.json for the demo report.
func readWeightsResults(path string) (weightsResults, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return weightsResults{}, err
	}
	var wr weightsResults
	if err := json.Unmarshal(data, &wr); err != nil {
		return weightsResults{}, err
	}
	return wr, nil
}
