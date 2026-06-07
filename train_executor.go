package sia

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// MLXTrainExecutor runs a weights-mode generation by treating the agent's
// train.py as a DECLARATIVE training spec and translating it into a call to the
// Go mlx-lm-train binary. No Python runtime is needed: the agent's "code" is
// read as a whitelisted set of hyperparameters, never executed.
//
// It implements [TargetExecutor]; wire it as the orchestrator's Target for a
// FocusWeights run that trains real model weights on-device:
//
//	o := sia.NewOrchestrator(meta, &sia.MLXTrainExecutor{
//		BaseModel: "mlx-community/Qwen3-0.6B-4bit",
//		DataDir:   "/data/train", // train/valid jsonl only — never the held-out test
//	})
//	o.Run(ctx, sia.RunOptions{Focus: sia.FocusWeights, TrainingSandbox: sia.SandboxLocal, ...})
//
// A train.py the executor cannot parse, or a non-zero mlx-lm-train exit, is
// reported as TargetResult{Success:false, ErrorMsg:...} so the feedback agent
// can repair the next generation; a Go error is returned only when the executor
// itself cannot run (the binary is missing, the log cannot be created).
//
// Honesty discipline: DataDir must contain train (and optionally valid) data
// only. The held-out test set is the evaluator's, kept in a read-only directory
// outside the agent's reach; this executor never passes it to mlx-lm-train, so
// the agent can train on but never see the eval rows.
type MLXTrainExecutor struct {
	// TrainBin is the mlx-lm-train executable. Empty defaults to "mlx-lm-train"
	// (resolved on PATH).
	TrainBin string
	// BaseModel is the model directory or HuggingFace ID to fine-tune, e.g.
	// "mlx-community/Qwen3-0.6B-4bit". Required.
	BaseModel string
	// DataDir holds {train,valid}.jsonl for mlx-lm-train -data. It must NOT
	// contain the held-out test.jsonl (see the honesty note above). Required.
	DataDir string
	// Defaults are the hyperparameters used for any key the agent's spec does
	// not override. The zero value is filled from buildArgs' built-in safe
	// defaults, so a zero Defaults is usable.
	Defaults TrainSpec
	// Env is extra environment appended to os.Environ for the training process.
	Env []string
	// Progress, when non-nil, also receives mlx-lm-train's combined output as it
	// is produced, so a long-running fine-tune can be watched live. The default
	// (nil) keeps the executor quiet — output still goes to StdoutLog and the
	// returned TargetResult — so tests and library callers are unaffected. CLIs
	// set this to os.Stdout.
	Progress io.Writer
}

// NewMLXTrainExecutor returns an executor that fine-tunes baseModel on the
// train/valid data in dataDir. dataDir must NOT contain the held-out test set;
// that belongs to the evaluator (see the type doc's honesty note). The trainer
// binary defaults to mlx-lm-train on PATH; set [MLXTrainExecutor.TrainBin],
// [MLXTrainExecutor.Defaults], or [MLXTrainExecutor.Env] on the result to
// customize.
func NewMLXTrainExecutor(baseModel, dataDir string) *MLXTrainExecutor {
	return &MLXTrainExecutor{BaseModel: baseModel, DataDir: dataDir}
}

// TrainSpec is the whitelisted set of training hyperparameters parsed from a
// weights-mode train.py. Only these keys influence the run; anything else in
// the file is ignored. A nil/zero field means "unset" and falls back to the
// executor's Defaults and then the built-in defaults in buildArgs.
type TrainSpec struct {
	LearningRate *float64 // learning_rate -> -learning-rate (must be > 0)
	LoRARank     *int     // lora_rank     -> -lora-rank      (must be > 0)
	NumLayers    *int     // num_layers    -> -num-layers     (> 0, or -1 for all)
	Iters        *int     // iters         -> -iters          (must be > 0)
	BatchSize    *int     // batch_size    -> -batch-size     (must be > 0)
	FineTuneType string   // fine_tune_type-> -fine-tune-type (lora|dora|full)
	DataMix      string   // data_mix      -> selects a DataDir subdirectory
}

// RunTarget reads req.AgentPath (the gen's train.py) as a declarative spec and
// runs mlx-lm-train with the translated flags, streaming combined output to
// req.StdoutLog and writing adapter weights into req.WorkingDir.
func (e *MLXTrainExecutor) RunTarget(ctx context.Context, req TargetRequest) (TargetResult, error) {
	if e.BaseModel == "" {
		return TargetResult{}, errors.New("base model is required")
	}
	if e.DataDir == "" {
		return TargetResult{}, errors.New("data dir is required")
	}

	specBytes, err := os.ReadFile(req.AgentPath)
	if err != nil {
		// No train.py written (meta/feedback agent failed): report as feedback,
		// not a Go error, so the loop can repair the next generation.
		return TargetResult{Success: false, ErrorMsg: fmt.Sprintf("train spec not found: %s", req.AgentPath)}, nil
	}

	spec, err := parseTrainSpec(specBytes)
	if err != nil {
		return TargetResult{Success: false, ErrorMsg: fmt.Sprintf("parse train spec: %v", err)}, nil
	}

	args, err := e.buildArgs(spec)
	if err != nil {
		return TargetResult{Success: false, ErrorMsg: err.Error()}, nil
	}

	logFile, err := os.Create(req.StdoutLog)
	if err != nil {
		return TargetResult{Success: false, ErrorMsg: err.Error()}, fmt.Errorf("create stdout log: %w", err)
	}
	defer logFile.Close()

	var buf bytes.Buffer
	out := teeProgress(logFile, &buf, e.Progress)

	bin := e.TrainBin
	if bin == "" {
		bin = "mlx-lm-train"
	}
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Dir = req.WorkingDir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.Env = append(os.Environ(), e.Env...)

	runErr := cmd.Run()
	stdout := buf.String()
	if runErr != nil {
		// A cancelled or timed-out context is request cancellation, not an
		// executor failure: report it as feedback (Success=false), not a Go
		// error. (On unix the kill surfaces as an *exec.ExitError, so this also
		// normalizes the message across platforms.)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return TargetResult{Success: false, Stdout: stdout, ErrorMsg: fmt.Sprintf("training cancelled: %v", ctxErr)}, nil
		}
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			// Training ran but exited non-zero: feedback, not a Go error.
			msg := fmt.Sprintf("mlx-lm-train exited with code %d", exitErr.ExitCode())
			return TargetResult{Success: false, Stdout: stdout, ErrorMsg: msg}, nil
		}
		// The executor could not launch the trainer at all (e.g. binary missing).
		return TargetResult{Success: false, Stdout: stdout, ErrorMsg: runErr.Error()},
			fmt.Errorf("run mlx-lm-train: %w", runErr)
	}
	return TargetResult{Success: true, Stdout: stdout}, nil
}

// buildArgs translates a parsed spec plus the executor's defaults into the
// mlx-lm-train flag list. It always trains LoRA-family fine-tuning on a 4-bit
// base (QLoRA is unsupported), writes the adapter into the gen's WorkingDir, and
// points -data at DataDir (optionally a data_mix subdirectory) — never at the
// held-out test set.
func (e *MLXTrainExecutor) buildArgs(spec TrainSpec) ([]string, error) {
	// dataDir resolves a whitelisted data_mix to a subdirectory, defending
	// against path escapes (a mix like "../test" must not reach test data).
	dataDir := e.DataDir
	if mix := firstNonEmpty(spec.DataMix, e.Defaults.DataMix); mix != "" {
		if !isSafeMix(mix) {
			return nil, fmt.Errorf("invalid data_mix %q", mix)
		}
		dataDir = filepath.Join(e.DataDir, mix)
	}
	// Honesty defense-in-depth: refuse to train if the held-out test set is in
	// the data the trainer would read. The agent must never learn the eval rows;
	// test.jsonl belongs in the evaluator's read-only dir outside WorkingDir.
	if isFile(filepath.Join(dataDir, "test.jsonl")) {
		return nil, fmt.Errorf("data dir %q contains test.jsonl: held-out test data must stay outside the trainer", dataDir)
	}

	fineTune := firstNonEmpty(spec.FineTuneType, e.Defaults.FineTuneType, "lora")
	switch fineTune {
	case "lora", "dora", "full":
	default:
		return nil, fmt.Errorf("invalid fine_tune_type %q (want lora, dora, or full)", fineTune)
	}

	loraRank := intOr(spec.LoRARank, e.Defaults.LoRARank, 8)
	numLayers := intOr(spec.NumLayers, e.Defaults.NumLayers, 16)
	iters := intOr(spec.Iters, e.Defaults.Iters, 100)
	batchSize := intOr(spec.BatchSize, e.Defaults.BatchSize, 4)
	learningRate := floatOr(spec.LearningRate, e.Defaults.LearningRate, 1e-5)

	// Reject nonsensical hyperparameters up front so the agent gets clear
	// feedback instead of a cryptic trainer crash. num_layers == -1 means "all
	// layers" to mlx-lm-train, so it is allowed; the rest must be positive.
	for _, c := range []struct {
		name string
		val  int
	}{{"lora_rank", loraRank}, {"iters", iters}, {"batch_size", batchSize}} {
		if c.val <= 0 {
			return nil, fmt.Errorf("%s must be positive, got %d", c.name, c.val)
		}
	}
	if numLayers <= 0 && numLayers != -1 {
		return nil, fmt.Errorf("num_layers must be positive or -1 (all), got %d", numLayers)
	}
	if learningRate <= 0 {
		return nil, fmt.Errorf("learning_rate must be positive, got %s", formatFloat(learningRate))
	}

	adapterPath := "adapters" // relative to cmd.Dir (the gen WorkingDir)

	args := []string{
		"-train",
		"-model", e.BaseModel,
		"-data", dataDir,
		"-fine-tune-type", fineTune,
		"-adapter-path", adapterPath,
		"-lora-rank", itoa(loraRank),
		"-num-layers", itoa(numLayers),
		"-iters", itoa(iters),
		"-batch-size", itoa(batchSize),
		"-learning-rate", formatFloat(learningRate),
	}
	return args, nil
}

// trainSpecKeys is the whitelist of hyperparameter keys parsed from train.py.
// Keys outside this set are ignored, and the file is never executed.
var trainSpecKeys = map[string]bool{
	"learning_rate":  true,
	"lora_rank":      true,
	"num_layers":     true,
	"iters":          true,
	"batch_size":     true,
	"fine_tune_type": true,
	"data_mix":       true,
}

// parseTrainSpec reads a train.py as a declarative spec, extracting only the
// whitelisted top-level assignments (key = value or key: value). It does NOT
// execute the file: every line is matched as a simple assignment and anything
// that is not a whitelisted key is skipped, so arbitrary Python is inert.
//
// A returned error means a whitelisted key had a value that could not be parsed
// as its expected type; an empty spec (no recognized keys) is not an error.
func parseTrainSpec(data []byte) (TrainSpec, error) {
	var spec TrainSpec
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		key, val, ok := splitAssignment(sc.Text())
		if !ok || !trainSpecKeys[key] {
			continue
		}
		if err := assignSpecField(&spec, key, val); err != nil {
			return TrainSpec{}, err
		}
	}
	if err := sc.Err(); err != nil {
		return TrainSpec{}, err
	}
	return spec, nil
}

// splitAssignment parses a single "key = value" or "key: value" line, ignoring
// surrounding whitespace, a trailing comment, an optional trailing comma, and
// quotes around the value. It reports ok=false for blank lines, comments, and
// any line that is not a bare top-level assignment (e.g. indented code, calls).
func splitAssignment(line string) (key, val string, ok bool) {
	// Reject indented lines (nested code) and comments outright.
	if line == "" || line[0] == ' ' || line[0] == '\t' || strings.HasPrefix(strings.TrimSpace(line), "#") {
		return "", "", false
	}
	sep := strings.IndexAny(line, "=:")
	if sep < 0 {
		return "", "", false
	}
	key = strings.TrimSpace(line[:sep])
	rest := line[sep+1:]
	// Strip a trailing comment, then trailing comma and whitespace.
	if h := strings.IndexByte(rest, '#'); h >= 0 {
		rest = rest[:h]
	}
	rest = strings.TrimSpace(rest)
	rest = strings.TrimSuffix(rest, ",")
	rest = strings.TrimSpace(rest)
	val = unquote(rest)
	if key == "" || !isIdent(key) {
		return "", "", false
	}
	return key, val, true
}

// assignSpecField sets the spec field named by a whitelisted key, parsing val to
// the field's type. A type-mismatched value is an error (it signals a malformed
// spec the agent should fix), not a silently ignored line.
func assignSpecField(spec *TrainSpec, key, val string) error {
	switch key {
	case "learning_rate":
		f, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return fmt.Errorf("learning_rate %q: %w", val, err)
		}
		spec.LearningRate = &f
	case "lora_rank":
		n, err := parseIntField("lora_rank", val)
		if err != nil {
			return err
		}
		spec.LoRARank = &n
	case "num_layers":
		n, err := parseIntField("num_layers", val)
		if err != nil {
			return err
		}
		spec.NumLayers = &n
	case "iters":
		n, err := parseIntField("iters", val)
		if err != nil {
			return err
		}
		spec.Iters = &n
	case "batch_size":
		n, err := parseIntField("batch_size", val)
		if err != nil {
			return err
		}
		spec.BatchSize = &n
	case "fine_tune_type":
		spec.FineTuneType = val
	case "data_mix":
		spec.DataMix = val
	}
	return nil
}

func parseIntField(name, val string) (int, error) {
	n, err := strconv.Atoi(val)
	if err != nil {
		return 0, fmt.Errorf("%s %q: %w", name, val, err)
	}
	return n, nil
}

// unquote strips a single matched pair of surrounding single or double quotes.
func unquote(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// isIdent reports whether s is a plausible identifier (letters, digits,
// underscore; not starting with a digit), so a stray "a = b == c" or URL does
// not look like an assignment key.
func isIdent(s string) bool {
	for i, r := range s {
		switch {
		case r == '_':
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return s != ""
}

// isSafeMix reports whether a data_mix value is a single safe path segment (no
// separators, no parent-dir escape), so it can only select a subdirectory of
// DataDir and never reach the held-out test set.
func isSafeMix(mix string) bool {
	if mix == "" || mix == "." || mix == ".." {
		return false
	}
	if strings.ContainsAny(mix, "/\\") || strings.Contains(mix, "..") {
		return false
	}
	return true
}

// firstNonEmpty returns the first non-empty string, or "" if all are empty.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// intOr resolves an int hyperparameter by precedence: the agent's spec value if
// set, else the executor default if set, else the built-in safe value.
func intOr(spec, def *int, builtin int) int {
	if spec != nil {
		return *spec
	}
	if def != nil {
		return *def
	}
	return builtin
}

// floatOr is [intOr] for float64 hyperparameters.
func floatOr(spec, def *float64, builtin float64) float64 {
	if spec != nil {
		return *spec
	}
	if def != nil {
		return *def
	}
	return builtin
}

// formatFloat renders a learning rate compactly without losing precision,
// preferring the shortest representation (e.g. 1e-05, 0.0003).
func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'g', -1, 64)
}
