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
	// Metric selects the held-out score. "" or "test_loss" (the default) parses a
	// generalization loss from an adapter-aware mlx-lm-train eval — lower is
	// better. "accuracy" instead generates a label per held-out row with the
	// trained adapter and reports correct/total — higher is better. The accuracy
	// path mirrors upstream SIA's LawBench evaluator (pred_label == label).
	Metric string
	// TestBatches caps the test batches (-1 = all). Defaults to -1.
	TestBatches int
	// Env is extra environment appended to os.Environ for the eval process.
	Env []string
	// DryRun skips invoking mlx-lm-train and writes a placeholder result, so the
	// loop is demonstrable without a GPU/model download. Off for a real run.
	DryRun bool
}

// weightsResults is the results.json schema for a weights generation. The
// top-level scalars flow to the feedback agent. In the default test_loss metric
// a lower held-out loss is better, so improvement is a decreasing test_loss
// across generations. In the accuracy metric (Metric: "accuracy") the trained
// model labels each held-out row and the score is correct/total in [0,1] — a
// number that goes UP, mirroring upstream SIA's LawBench evaluator.
type weightsResults struct {
	Verdict    string  `json:"verdict"`
	Trained    bool    `json:"trained"`
	TestLoss   float64 `json:"test_loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
	Accuracy   float64 `json:"accuracy,omitempty"`
	Correct    int     `json:"correct,omitempty"`
	Total      int     `json:"total,omitempty"`
	Metric     string  `json:"metric"`
	Reason     string  `json:"reason,omitempty"`
	HeldOut    string  `json:"held_out_dir"`
}

// The held-out metrics. metricTestLoss (the default) is a decreasing
// generalization loss; metricAccuracy is an increasing correct/total share.
const (
	metricTestLoss = "test_loss"
	metricAccuracy = "accuracy"
)

// metric returns the selected held-out metric, defaulting to test_loss.
func (e *WeightsEvaluator) metric() string {
	if e.Metric == metricAccuracy {
		return metricAccuracy
	}
	return metricTestLoss
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
	metric := e.metric()

	if e.DryRun {
		return e.write(genDir, weightsResults{
			Verdict: "SKIPPED", Trained: false, Metric: metric,
			Reason:  "dry-run: held-out eval not executed (no training/GPU)",
			HeldOut: e.HeldOutDir,
		})
	}

	if !isDir(adapterDir) {
		// No adapter produced (training did not run or failed): report as feedback.
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: false, Metric: metric,
			Reason:  fmt.Sprintf("no trained adapter at %s; training likely failed", adapterDir),
			HeldOut: e.HeldOutDir,
		})
	}

	if metric == metricAccuracy {
		return e.evaluateAccuracy(ctx, genDir, adapterDir)
	}

	loss, out, err := e.evalHeldOut(ctx, adapterDir)
	logPath := filepath.Join(genDir, sia.NameEvalLog)
	_ = os.WriteFile(logPath, []byte(out), 0o644)
	if err != nil {
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: true, Metric: metric,
			Reason:  fmt.Sprintf("held-out eval failed: %v", err),
			HeldOut: e.HeldOutDir,
		})
	}

	res := weightsResults{
		Verdict:    "PASS",
		Trained:    true,
		TestLoss:   loss,
		Perplexity: perplexity(loss),
		Metric:     metric,
		HeldOut:    e.HeldOutDir,
	}
	// The held-out gate is the hero: a generation that trains and evaluates fine
	// but whose held-out loss is WORSE than the best of all strictly-prior
	// generations has regressed (overfitting) and must not be silently blessed.
	// Flag it REVISE with the comparison spelled out, while keeping the raw
	// test_loss above so the number is never hidden. The lookup is causal —
	// generation N reads only generations 1..N-1 — so a later result can never
	// retroactively change an earlier verdict (an honest online loop). A new
	// best (or the first generation, which has no prior) stays PASS.
	if bestGen, bestLoss, ok := e.bestPriorLoss(genDir); ok && loss > bestLoss {
		res.Verdict = "REVISE"
		res.Reason = fmt.Sprintf("held-out test_loss %.4f > best-so-far %.4f (gen %d): overfitting, rejected", loss, bestLoss, bestGen)
	}
	return e.write(genDir, res)
}

// bestPriorLoss scans the results.json of every generation strictly before the
// one at genDir and returns the lowest held-out test_loss found and the
// generation that produced it. It reads only earlier generations (causal: gen N
// sees gen 1..N-1), so a verdict never depends on a future result. ok is false
// when genDir is not a gen_N directory or no prior generation has a usable loss.
func (e *WeightsEvaluator) bestPriorLoss(genDir string) (bestGen int, bestLoss float64, ok bool) {
	gen := genFromWorkingDir(genDir)
	if gen <= 1 {
		return 0, 0, false // gen 1 (or unparsable) has no prior to compare against
	}
	runDir := filepath.Dir(genDir)
	for prev := 1; prev < gen; prev++ {
		wr, err := readWeightsResults(filepath.Join(runDir, fmt.Sprintf("gen_%d", prev), sia.NameResultsJSON))
		if err != nil || !wr.Trained || wr.TestLoss <= 0 {
			continue // skip generations that did not produce a usable held-out loss
		}
		if !ok || wr.TestLoss < bestLoss {
			bestGen, bestLoss, ok = prev, wr.TestLoss, true
		}
	}
	return bestGen, bestLoss, ok
}

// evaluateAccuracy scores the trained adapter on the held-out set the upstream
// LawBench way: for each held-out row it builds the prompt with the label
// WITHHELD, generates a few tokens with the trained adapter attached, and counts
// a row correct when the generated label equals the true label. accuracy =
// correct/total in [0,1] — higher is better. The model never saw these rows, so
// it is a real generalization measurement, not memorization.
func (e *WeightsEvaluator) evaluateAccuracy(ctx context.Context, genDir, adapterDir string) (sia.EvalResult, error) {
	samples, err := readHeldOutSamples(filepath.Join(e.HeldOutDir, "test.jsonl"))
	if err != nil {
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: true, Metric: metricAccuracy,
			Reason:  fmt.Sprintf("held-out accuracy eval failed: %v", err),
			HeldOut: e.HeldOutDir,
		})
	}

	// Read the trained adapter's architecture so the generation attaches the SAME
	// shape. mlx-lm-train defaults to 16 layers / rank 8 when these are unset, and
	// a shape mismatch loads the wrong (or no) adapter — silently scoring a model
	// that is not the one trained. Honesty requires the eval match the rung.
	cfg, err := readAdapterConfig(filepath.Join(adapterDir, "adapter_config.json"))
	if err != nil {
		return e.write(genDir, weightsResults{
			Verdict: "REVISE", Trained: true, Metric: metricAccuracy,
			Reason:  fmt.Sprintf("held-out accuracy eval failed: read adapter config: %v", err),
			HeldOut: e.HeldOutDir,
		})
	}

	var log strings.Builder
	correct := 0
	for i, s := range samples {
		pred, out, perr := e.predictLabel(ctx, adapterDir, cfg, s.prompt)
		fmt.Fprintf(&log, "row %d: want=%q pred=%q ok=%v\n%s\n", i, s.label, pred, pred == s.label, out)
		if perr != nil {
			return e.write(genDir, weightsResults{
				Verdict: "REVISE", Trained: true, Metric: metricAccuracy,
				Reason:  fmt.Sprintf("held-out accuracy eval failed on row %d: %v", i, perr),
				HeldOut: e.HeldOutDir,
			})
		}
		if pred == s.label {
			correct++
		}
	}
	_ = os.WriteFile(filepath.Join(genDir, sia.NameEvalLog), []byte(log.String()), 0o644)

	total := len(samples)
	acc := 0.0
	if total > 0 {
		acc = float64(correct) / float64(total)
	}
	res := weightsResults{
		Verdict:  "PASS",
		Trained:  true,
		Accuracy: acc,
		Correct:  correct,
		Total:    total,
		Metric:   metricAccuracy,
		HeldOut:  e.HeldOutDir,
	}
	// Higher-is-better gate (mirror of the loss gate, inverted): a generation
	// whose held-out accuracy is LOWER than the best of all strictly-prior
	// generations has regressed and must not be silently blessed. The lookup is
	// causal — generation N reads only generations 1..N-1 — so a later result can
	// never retroactively change an earlier verdict. A new best (or gen 1, which
	// has no prior) stays PASS.
	if bestGen, bestAcc, ok := e.bestPriorAccuracy(genDir); ok && acc < bestAcc {
		res.Verdict = "REVISE"
		res.Reason = fmt.Sprintf("held-out accuracy %.4f < best-so-far %.4f (gen %d): regressed, rejected", acc, bestAcc, bestGen)
	}
	return e.write(genDir, res)
}

// bestPriorAccuracy scans the results.json of every generation strictly before
// the one at genDir and returns the highest held-out accuracy found and the
// generation that produced it. It reads only earlier generations (causal: gen N
// sees gen 1..N-1), so a verdict never depends on a future result. ok is false
// when genDir is not a gen_N directory or no prior generation has a usable score.
func (e *WeightsEvaluator) bestPriorAccuracy(genDir string) (bestGen int, bestAcc float64, ok bool) {
	gen := genFromWorkingDir(genDir)
	if gen <= 1 {
		return 0, 0, false // gen 1 (or unparsable) has no prior to compare against
	}
	runDir := filepath.Dir(genDir)
	for prev := 1; prev < gen; prev++ {
		wr, err := readWeightsResults(filepath.Join(runDir, fmt.Sprintf("gen_%d", prev), sia.NameResultsJSON))
		if err != nil || !wr.Trained || wr.Total <= 0 {
			continue // skip generations that did not produce a usable accuracy
		}
		if !ok || wr.Accuracy > bestAcc {
			bestGen, bestAcc, ok = prev, wr.Accuracy, true
		}
	}
	return bestGen, bestAcc, ok
}

// predictLabel generates a label for one held-out prompt with the trained
// adapter attached, returning the parsed label ("positive"/"negative", or "" if
// neither was generated) and the raw output.
//
// mlx-lm-generate cannot attach a LoRA adapter (it has no -adapter-path /
// -resume-adapter-file flag), and mlx-lm-fuse mis-handles the 4-bit quantized
// base (it drops the untouched and scale/bias tensors, yielding an unloadable
// model). The only tool that attaches and resumes the adapter for generation is
// mlx-lm-train's sample-generation hook. We run it with -learning-rate 0 and
// -iters 1: the single step is a no-op (zero LR moves no weights), so the sample
// reflects the PURE resumed adapter, while -gen-every 1 fires the generation.
// -save-every is set huge so the no-op step never rewrites the saved adapter.
// cfg carries the adapter's layer count and rank so the attached shape matches
// the trained adapter (a mismatch silently loads the wrong weights).
func (e *WeightsEvaluator) predictLabel(ctx context.Context, adapterDir string, cfg adapterConfig, prompt string) (string, string, error) {
	bin := e.TrainBin
	if bin == "" {
		bin = "mlx-lm-train"
	}
	args := []string{
		"-train", "-iters", "1",
		"-learning-rate", "0",
		"-batch-size", "2",
		"-save-every", "1000000",
		"-model", e.BaseModel,
		"-data", e.HeldOutDir,
		"-adapter-path", adapterDir,
		"-resume-adapter-file", filepath.Join(adapterDir, "adapters.safetensors"),
		"-num-layers", strconv.Itoa(cfg.NumLayers),
		"-lora-rank", strconv.Itoa(cfg.LoRAParameters.Rank),
		"-gen-every", "1",
		"-gen-prompt", prompt,
		"-gen-tokens", "4",
	}

	var buf bytes.Buffer
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = append(os.Environ(), e.Env...)
	if runErr := cmd.Run(); runErr != nil {
		return "", buf.String(), fmt.Errorf("%v", runErr)
	}
	out := buf.String()
	return parsePredictedLabel(out, prompt), out, nil
}

// genSampleRe captures the quoted sample mlx-lm-train emits at -gen-every: an
// `Iter N: "<prompt + generation>"` block whose body spans multiple lines.
var genSampleRe = regexp.MustCompile(`(?s)Iter\s+\d+:\s*"(.*?)"`)

// parsePredictedLabel extracts the model's predicted label from a sample
// generation. mlx-lm-train echoes the prompt plus the continuation inside the
// quoted Iter block; the prompt withholds the label, so the first "positive" or
// "negative" appearing AFTER the prompt is the model's prediction. It returns
// "" when neither label was generated.
func parsePredictedLabel(out, prompt string) string {
	body := out
	if m := genSampleRe.FindStringSubmatch(out); m != nil {
		body = m[1]
	}
	// Strip the echoed prompt so a stray label word in the sentence cannot be
	// mistaken for the prediction; the prompt itself withholds the label.
	if i := strings.LastIndex(body, "Sentiment:"); i >= 0 {
		body = body[i+len("Sentiment:"):]
	} else if i := strings.Index(body, prompt); i >= 0 {
		body = body[i+len(prompt):]
	}
	body = strings.ToLower(body)
	pos := strings.Index(body, "positive")
	neg := strings.Index(body, "negative")
	switch {
	case pos < 0 && neg < 0:
		return ""
	case neg < 0:
		return "positive"
	case pos < 0:
		return "negative"
	case pos < neg:
		return "positive"
	default:
		return "negative"
	}
}

// adapterConfig mirrors the adapter_config.json mlx-lm-train writes beside the
// adapter weights: the layer count and LoRA rank the generation trained with.
// The eval must reattach the SAME shape or the resume loads the wrong weights.
type adapterConfig struct {
	NumLayers      int `json:"num_layers"`
	LoRAParameters struct {
		Rank int `json:"rank"`
	} `json:"lora_parameters"`
}

// readAdapterConfig loads the trained adapter's architecture. It defaults a
// missing layer count or rank to mlx-lm-train's own defaults (16 layers, rank
// 8) so an older adapter without the field still attaches.
func readAdapterConfig(path string) (adapterConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return adapterConfig{}, err
	}
	var cfg adapterConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return adapterConfig{}, fmt.Errorf("parse adapter config: %w", err)
	}
	if cfg.NumLayers == 0 {
		cfg.NumLayers = 16
	}
	if cfg.LoRAParameters.Rank == 0 {
		cfg.LoRAParameters.Rank = 8
	}
	return cfg, nil
}

// heldOutSample is one held-out row: the prompt with its label withheld plus the
// true label to score against.
type heldOutSample struct {
	prompt string
	label  string
}

// labelLineRe splits a scaffolded "text" row into its sentence prompt and label.
// Rows are "Classify the sentiment.\nSentence: <text>\nSentiment: <label>"; the
// prompt is everything through "Sentiment:" (label withheld), the label the tail.
var labelLineRe = regexp.MustCompile(`(?s)^(.*Sentiment:)\s*(\w+)\s*$`)

// readHeldOutSamples loads the held-out test.jsonl and returns each row as a
// prompt (label withheld) plus its true label, ready to score for accuracy.
func readHeldOutSamples(path string) ([]heldOutSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var samples []heldOutSample
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("parse held-out row: %w", err)
		}
		m := labelLineRe.FindStringSubmatch(row.Text)
		if m == nil {
			return nil, fmt.Errorf("held-out row missing 'Sentiment: <label>': %q", row.Text)
		}
		samples = append(samples, heldOutSample{
			prompt: strings.TrimRight(m[1], " "),
			label:  strings.ToLower(m[2]),
		})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("no held-out rows in %s", path)
	}
	return samples, nil
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
	// The held-out eval MUST score the trained adapter, not the bare base model.
	// mlx-lm-train only attaches and resumes the LoRA adapter inside its training
	// path; a plain `-test` (no `-train`) evaluates bundle.Model with no adapter,
	// so it returns the identical base-model loss for every generation regardless
	// of what was trained. Running `-train -iters 0` enters that path — attach +
	// resume the adapter — but takes ZERO optimizer steps, so it is a pure
	// adapter-aware evaluation that does not perturb the resumed weights. Verified:
	// base model scores 4.1875 while two different real adapters score 2.69 / 2.50.
	args := []string{
		"-test",
		"-train", "-iters", "0",
		"-batch-size", "2",
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
