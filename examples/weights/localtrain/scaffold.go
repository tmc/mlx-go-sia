package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
)

// dataset is a tiny, deterministic instruction→response set for the narrow demo
// task: classify a short sentence's sentiment. It is intentionally small and
// legible so a LoRA update can move the held-out loss within one generation.
//
// The rows are split into train, valid, and a HELD-OUT test set. The agent's
// training executor is given only the train/valid directory; the held-out test
// directory is the evaluator's and never reaches mlx-lm-train via the agent.
// The train set is large and varied enough that added LoRA capacity across
// generations generalizes rather than memorizing — so the held-out loss can
// genuinely descend gen-over-gen, not just overfit. train/valid/test are
// disjoint. All three are balanced positive/negative.
var dataset = struct{ train, valid, test []sample }{
	train: []sample{
		// positive
		{"The food was delicious and the staff were kind.", "positive"},
		{"I loved every minute of the show.", "positive"},
		{"This is the best purchase I have made all year.", "positive"},
		{"The room was clean, bright, and welcoming.", "positive"},
		{"Absolutely wonderful experience from start to finish.", "positive"},
		{"The service was fast and everyone was friendly.", "positive"},
		{"This book kept me hooked until the last page.", "positive"},
		{"The hotel exceeded every one of my expectations.", "positive"},
		{"A brilliant performance by the entire cast.", "positive"},
		{"The new update made the app so much smoother.", "positive"},
		{"Fresh ingredients and a cozy atmosphere.", "positive"},
		{"They refunded me instantly without any hassle.", "positive"},
		{"The scenery on the hike was breathtaking.", "positive"},
		{"My order arrived a day early and well packaged.", "positive"},
		{"The instructor explained everything clearly and patiently.", "positive"},
		{"Such a comfortable bed; I slept wonderfully.", "positive"},
		{"The coffee here is rich and perfectly brewed.", "positive"},
		{"Customer support solved my issue in two minutes.", "positive"},
		{"The movie was funny, heartfelt, and beautifully shot.", "positive"},
		{"Great value for the price and excellent quality.", "positive"},
		{"The garden was peaceful and full of color.", "positive"},
		{"I would happily recommend this to all my friends.", "positive"},
		{"The keyboard feels premium and types like a dream.", "positive"},
		{"Their pastries are the best I have ever tasted.", "positive"},
		{"The concert was electric from the first note.", "positive"},
		{"A thoughtful gift that she absolutely adored.", "positive"},
		{"The trail was well marked and easy to follow.", "positive"},
		{"This jacket is warm, light, and looks fantastic.", "positive"},
		{"The tutorial was concise and genuinely helpful.", "positive"},
		{"Everyone went above and beyond to make us feel welcome.", "positive"},
		{"The battery lasts all day on a single charge.", "positive"},
		{"A charming town with the friendliest locals.", "positive"},
		{"The dish was seasoned perfectly and beautifully plated.", "positive"},
		{"The headphones have crisp, immersive sound.", "positive"},
		{"This was money well spent; I use it every day.", "positive"},
		// negative
		{"The product broke after a single use.", "negative"},
		{"I waited an hour and no one helped me.", "negative"},
		{"The worst customer service I have ever dealt with.", "negative"},
		{"It was a complete waste of my money.", "negative"},
		{"The app crashes every time I open it.", "negative"},
		{"The room smelled musty and the sheets were stained.", "negative"},
		{"They never responded to a single one of my emails.", "negative"},
		{"The food was cold, bland, and overpriced.", "negative"},
		{"The plot was predictable and the acting was wooden.", "negative"},
		{"My package arrived crushed and missing parts.", "negative"},
		{"The update slowed everything down to a crawl.", "negative"},
		{"Rude staff and an impossibly long wait.", "negative"},
		{"The fabric tore the first time I wore it.", "negative"},
		{"Nothing worked as advertised; deeply disappointing.", "negative"},
		{"The hotel was nothing like the photos online.", "negative"},
		{"I regretted this purchase almost immediately.", "negative"},
		{"The instructions were confusing and incomplete.", "negative"},
		{"The battery died within an hour of unboxing it.", "negative"},
		{"Overcrowded, noisy, and not worth the ticket price.", "negative"},
		{"The seat was cramped and the flight was delayed twice.", "negative"},
		{"Their support line kept me on hold and then hung up.", "negative"},
		{"The coffee tasted burnt and watered down.", "negative"},
		{"A frustrating experience I would not repeat.", "negative"},
		{"The screen cracked even inside its protective case.", "negative"},
		{"The meal made me feel sick the rest of the night.", "negative"},
		{"Cheaply made and falling apart already.", "negative"},
		{"The tour was rushed and poorly organized.", "negative"},
		{"They charged me twice and refused to fix it.", "negative"},
		{"The sound was tinny and the bass was nonexistent.", "negative"},
		{"Dull, forgettable, and far too long.", "negative"},
		{"The website crashed during checkout three times.", "negative"},
		{"The bed was lumpy and the room was freezing.", "negative"},
		{"This was the most disappointing meal of the trip.", "negative"},
		{"The keys stick and the trackpad barely responds.", "negative"},
		{"I want my money back; it never worked at all.", "negative"},
	},
	// Validation: disjoint from train, balanced, large enough for a real batch.
	valid: []sample{
		{"A delightful little cafe with great coffee.", "positive"},
		{"The team delivered exactly what they promised.", "positive"},
		{"The staff remembered my name and my usual order.", "positive"},
		{"A smooth, relaxing, and thoroughly enjoyable trip.", "positive"},
		{"The repair was quick and the price was fair.", "positive"},
		{"The view from the balcony was simply stunning.", "positive"},
		{"The flight was delayed and the seats were cramped.", "negative"},
		{"Nothing about this lived up to the hype.", "negative"},
		{"The delivery driver left it out in the rain.", "negative"},
		{"The interface is clunky and constantly freezes.", "negative"},
		{"They cancelled my reservation without telling me.", "negative"},
		{"The portions were tiny for what they charged.", "negative"},
	},
	// Held-out: never seen during training; the evaluator scores generalization.
	test: []sample{
		{"What a fantastic and memorable evening.", "positive"},
		{"They went out of their way to help me.", "positive"},
		{"The presentation was clear, sharp, and engaging.", "positive"},
		{"Cozy, clean, and exactly what we needed.", "positive"},
		{"The upgrade was seamless and well worth it.", "positive"},
		{"A warm welcome and impeccable service throughout.", "positive"},
		{"The soundtrack alone was worth the price of admission.", "positive"},
		{"Sturdy, reliable, and beautifully designed.", "positive"},
		{"The package arrived damaged and late.", "negative"},
		{"Overpriced and underwhelming in every way.", "negative"},
		{"The line never moved and the staff ignored us.", "negative"},
		{"It stopped charging after the second day.", "negative"},
		{"The room was dirty and the shower was broken.", "negative"},
		{"A tedious, overlong, and joyless film.", "negative"},
		{"They lost my order and blamed me for it.", "negative"},
		{"The material pilled and faded after one wash.", "negative"},
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
