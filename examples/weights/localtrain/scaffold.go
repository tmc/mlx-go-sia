package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/sia-apple-silicon"
)

// dataset is a tiny, deterministic instruction→response set for the narrow demo
// task: classify a short sentence's sentiment. It is intentionally small and
// legible so a LoRA update can move the held-out loss within one generation.
//
// The rows are split into train, valid, and a HELD-OUT test set. The agent's
// training executor is given only the train/valid directory; the held-out test
// directory is the evaluator's and never reaches mlx-lm-train via the agent.
var dataset = struct{ train, valid, test []sample }{
	train: []sample{
		{"The food was delicious and the staff were kind.", "positive"},
		{"I loved every minute of the show.", "positive"},
		{"This is the best purchase I have made all year.", "positive"},
		{"The room was clean, bright, and welcoming.", "positive"},
		{"Absolutely wonderful experience from start to finish.", "positive"},
		{"The product broke after a single use.", "negative"},
		{"I waited an hour and no one helped me.", "negative"},
		{"The worst customer service I have ever dealt with.", "negative"},
		{"It was a complete waste of my money.", "negative"},
		{"The app crashes every time I open it.", "negative"},
	},
	// Validation needs at least batch_size examples (the seed spec uses 2), so
	// keep at least four here to leave headroom for a slightly larger batch.
	valid: []sample{
		{"A delightful little cafe with great coffee.", "positive"},
		{"The flight was delayed and the seats were cramped.", "negative"},
		{"The team delivered exactly what they promised.", "positive"},
		{"Nothing about this lived up to the hype.", "negative"},
	},
	// Held-out: never seen during training; the evaluator scores generalization.
	test: []sample{
		{"What a fantastic and memorable evening.", "positive"},
		{"The package arrived damaged and late.", "negative"},
		{"They went out of their way to help me.", "positive"},
		{"Overpriced and underwhelming in every way.", "negative"},
	},
}

// sample is one labeled example.
type sample struct {
	text  string
	label string
}

// scaffoldData writes train/valid into dataDir (the agent's training data) and
// the held-out test.jsonl into heldOutDir (the evaluator's, kept outside the
// agent's reach). Both are created if missing. mlx-lm-train reads {train,valid,
// test}.jsonl with a "text" field per line.
func scaffoldData(dataDir, heldOutDir string) error {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return fmt.Errorf("mkdir data: %w", err)
	}
	if err := os.MkdirAll(heldOutDir, 0o755); err != nil {
		return fmt.Errorf("mkdir held-out: %w", err)
	}
	if err := writeJSONL(filepath.Join(dataDir, "train.jsonl"), dataset.train); err != nil {
		return err
	}
	if err := writeJSONL(filepath.Join(dataDir, "valid.jsonl"), dataset.valid); err != nil {
		return err
	}
	// The held-out test set lives ONLY in heldOutDir. mlx-lm-train -test reads
	// {train,valid,test}.jsonl, so the held-out dir gets its own train/valid
	// stubs (a copy of train) plus the real held-out test.jsonl.
	if err := writeJSONL(filepath.Join(heldOutDir, "train.jsonl"), dataset.train); err != nil {
		return err
	}
	if err := writeJSONL(filepath.Join(heldOutDir, "valid.jsonl"), dataset.valid); err != nil {
		return err
	}
	return writeJSONL(filepath.Join(heldOutDir, "test.jsonl"), dataset.test)
}

// writeJSONL writes samples as mlx-lm-train "text"-format JSONL.
func writeJSONL(path string, samples []sample) error {
	var buf []byte
	for _, s := range samples {
		line, err := json.Marshal(map[string]string{
			"text": fmt.Sprintf("Classify the sentiment.\nSentence: %s\nSentiment: %s", s.text, s.label),
		})
		if err != nil {
			return fmt.Errorf("marshal sample: %w", err)
		}
		buf = append(buf, line...)
		buf = append(buf, '\n')
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", filepath.Base(path), err)
	}
	return nil
}

// scaffoldTask writes a self-contained weights-focus task and returns the layout,
// resolved reference, and loaded task files. The reference seed is the train.py
// spec the agent starts from; the weights meta prompt (embedded in the sia
// package) steers the feedback agent to edit the training hyperparameters.
func scaffoldTask(root string) (sia.TaskLayout, sia.ResolvedAgentReference, sia.TaskFiles, error) {
	taskDir := filepath.Join(root, "_task")
	sharedDir := filepath.Join(taskDir, "..", sia.NameSharedDir)
	dataPublic := filepath.Join(taskDir, sia.NameDataPublic)
	refDir := filepath.Join(taskDir, sia.NameReferenceDir)
	for _, d := range []string{dataPublic, refDir, sharedDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, fmt.Errorf("mkdir %s: %w", d, err)
		}
	}
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameTaskMD), []byte(weightsTaskMD), 0o644); err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameSampleTaskDescriptions), []byte(weightsSampleDesc), 0o644); err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}
	// The reference seed is the train.py spec (the agent improves its knobs).
	if err := os.WriteFile(filepath.Join(taskDir, sia.NameReferenceAgent), []byte(seedTrainSpec), 0o644); err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}
	if err := os.WriteFile(filepath.Join(sharedDir, sia.NameSharedSampleExecution), weightsSampleExecution, 0o644); err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}

	taskLayout := sia.NewTaskLayout(taskDir, sharedDir)
	resolved, err := sia.DefaultAgentReference.Resolve(taskLayout)
	if err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}
	taskFiles, err := sia.LoadTaskFiles(taskLayout, resolved)
	if err != nil {
		return sia.TaskLayout{}, sia.ResolvedAgentReference{}, sia.TaskFiles{}, err
	}
	return taskLayout, resolved, taskFiles, nil
}

// seedTrainSpec is the initial train.py: a declarative hyperparameter block the
// MLXTrainExecutor parses (it is never executed as Python). The feedback agent
// tweaks these knobs (LR, LoRA rank, layers, iters) to lower the held-out loss.
const seedTrainSpec = `# train.py — declarative LoRA training spec (parsed, not executed).
# The runner translates these whitelisted keys into an mlx-lm-train invocation.
# Tune them to lower the held-out test loss; keep fine_tune_type = lora (the
# 4-bit base does not support QLoRA).

learning_rate = 1e-5
lora_rank = 8
num_layers = 16
iters = 100
batch_size = 2
fine_tune_type = "lora"
`

const weightsTaskMD = `# Task: improve a model on a sentiment task via local LoRA fine-tuning

You are improving train.py, a DECLARATIVE training spec (not executable Python).
Each generation you may tune the whitelisted hyperparameters to lower the model's
held-out test loss after a local LoRA fine-tune:

  learning_rate, lora_rank, num_layers, iters, batch_size, fine_tune_type

Rules:
- Keep fine_tune_type = lora (the 4-bit base model does not support QLoRA).
- The training data has train/valid only; the test set is held out and scored by
  the evaluator — you cannot see it, so improvements must generalize.
- Lower test_loss across generations is the goal.

Write the updated train.py (the same hyperparameter-block format) into the
working directory.
`

const weightsSampleDesc = `# Sample task descriptions

- Lower held-out test loss by tuning LoRA rank, learning rate, number of tuned
  layers, and iteration count for a local on-device fine-tune of a small model.
`

var weightsSampleExecution = json.RawMessage(`{
  "task": "local-lora-finetune",
  "messages": [
    {"role": "system", "content": "Tune the LoRA hyperparameters to lower held-out loss."},
    {"role": "assistant", "content": "Raised lora_rank to 16 and iters to 150."}
  ],
  "result": {"verdict": "PASS", "test_loss": 0}
}`)
