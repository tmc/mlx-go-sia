package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	sia "github.com/tmc/mlx-go-sia"
)

func TestParseTestLoss(t *testing.T) {
	tests := []struct {
		name string
		out  string
		want float64
		ok   bool
	}{
		{"spaced", "Iter 100: done\nTest loss 1.2345\n", 1.2345, true},
		{"colon", "test_loss: 0.873\n", 0.873, true},
		{"equals", "Test loss = 2.0\n", 2.0, true},
		{"fallback line", "Final Test Loss value 3.14 ppl 23\n", 3.14, true},
		{"absent", "training complete, no eval\n", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTestLoss(tt.out)
			if tt.ok != (err == nil) {
				t.Fatalf("err = %v, want ok=%v", err, tt.ok)
			}
			if tt.ok && got != tt.want {
				t.Fatalf("loss = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEvaluateDryRun(t *testing.T) {
	e := &WeightsEvaluator{BaseModel: "m", HeldOutDir: t.TempDir(), DryRun: true}
	genDir := t.TempDir()
	res, err := e.Evaluate(context.Background(), genDir)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Status != sia.EvalWarning {
		t.Fatalf("status = %v, want EvalWarning (dry-run SKIPPED)", res.Status)
	}
	if _, statErr := os.Stat(filepath.Join(genDir, sia.NameResultsJSON)); statErr != nil {
		t.Fatalf("results.json not written: %v", statErr)
	}
}

func TestEvaluateNoAdapter(t *testing.T) {
	// A non-dry run with no trained adapter must report REVISE feedback, not a
	// Go error (which would abort the loop).
	e := &WeightsEvaluator{BaseModel: "m", HeldOutDir: t.TempDir()}
	res, err := e.Evaluate(context.Background(), t.TempDir())
	if err != nil {
		t.Fatalf("Evaluate returned a Go error (should be feedback): %v", err)
	}
	if res.Status != sia.EvalWarning {
		t.Fatalf("status = %v, want EvalWarning", res.Status)
	}
}

// writeGenResult lays down a gen_N/results.json with the given held-out loss, as
// the evaluator would after that generation. trained=false stubs an untrained gen.
func writeGenResult(t *testing.T, runDir string, gen int, loss float64, trained bool) {
	t.Helper()
	genDir := filepath.Join(runDir, "gen_"+strconv.Itoa(gen))
	if err := os.MkdirAll(genDir, 0o755); err != nil {
		t.Fatal(err)
	}
	wr := weightsResults{Verdict: "PASS", Trained: trained, TestLoss: loss, Metric: "test_loss"}
	data, _ := json.MarshalIndent(wr, "", "  ")
	if err := os.WriteFile(filepath.Join(genDir, sia.NameResultsJSON), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestBestPriorLossIsCausal verifies the held-out gate's best-so-far lookup is
// causal (reads only strictly-earlier generations) and ignores untrained gens.
func TestBestPriorLossIsCausal(t *testing.T) {
	e := &WeightsEvaluator{BaseModel: "m", HeldOutDir: "x"}
	runDir := t.TempDir()
	// Series: gen1=2.42 (best), gen2=2.49 (worse), gen3=2.63 (worse), gen4 untrained.
	writeGenResult(t, runDir, 1, 2.4183, true)
	writeGenResult(t, runDir, 2, 2.4920, true)
	writeGenResult(t, runDir, 3, 2.6322, true)
	writeGenResult(t, runDir, 4, 0, false)

	// gen1 has no prior -> not ok.
	if _, _, ok := e.bestPriorLoss(filepath.Join(runDir, "gen_1")); ok {
		t.Error("gen1 should have no prior best")
	}
	// gen2 sees only gen1.
	if g, l, ok := e.bestPriorLoss(filepath.Join(runDir, "gen_2")); !ok || g != 1 || l != 2.4183 {
		t.Errorf("gen2 best-prior = gen%d loss%.4f ok=%v, want gen1 2.4183", g, l, ok)
	}
	// gen3 sees gen1+gen2; gen1 is still the best (causal, not affected by gen3 itself).
	if g, l, ok := e.bestPriorLoss(filepath.Join(runDir, "gen_3")); !ok || g != 1 || l != 2.4183 {
		t.Errorf("gen3 best-prior = gen%d loss%.4f ok=%v, want gen1 2.4183", g, l, ok)
	}
	// gen5 (after an untrained gen4) still finds gen1 as best and skips the untrained one.
	if g, l, ok := e.bestPriorLoss(filepath.Join(runDir, "gen_5")); !ok || g != 1 || l != 2.4183 {
		t.Errorf("gen5 best-prior = gen%d loss%.4f ok=%v, want gen1 2.4183 (untrained gen4 skipped)", g, l, ok)
	}
}

func TestParsePredictedLabel(t *testing.T) {
	prompt := "Classify the sentiment.\nSentence: What a fantastic evening.\nSentiment:"
	tests := []struct {
		name string
		out  string
		want string
	}{
		{
			"trained positive",
			"Iter 1: \"Classify the sentiment.\nSentence: What a fantastic evening.\nSentiment: positive<|im_end|>!!\"\n",
			"positive",
		},
		{
			"base negative",
			"Iter 1: \"Classify the sentiment.\nSentence: What a fantastic evening.\nSentiment: Negative.\nSentence:\"\n",
			"negative",
		},
		{
			"no label generated",
			"Iter 1: \"Classify the sentiment.\nSentence: What a fantastic evening.\nSentiment: 12345\"\n",
			"",
		},
		{
			// A label word appearing only inside the echoed prompt must NOT count as a
			// prediction: the prompt withholds the label, so the strip past "Sentiment:"
			// keeps the sentence text from leaking in.
			"label word in sentence is not the prediction",
			"Iter 1: \"Classify the sentiment.\nSentence: a positive review.\nSentiment: negative<|im_end|>\"\n",
			"negative",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parsePredictedLabel(tt.out, prompt); got != tt.want {
				t.Fatalf("parsePredictedLabel = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestReadHeldOutSamples(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.jsonl")
	body := `{"text":"Classify the sentiment.\nSentence: What a fantastic evening.\nSentiment: positive"}
{"text":"Classify the sentiment.\nSentence: The package arrived damaged.\nSentiment: negative"}
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := readHeldOutSamples(path)
	if err != nil {
		t.Fatalf("readHeldOutSamples: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d samples, want 2", len(got))
	}
	if got[0].label != "positive" || got[1].label != "negative" {
		t.Errorf("labels = %q,%q, want positive,negative", got[0].label, got[1].label)
	}
	// The prompt must withhold the label (end at "Sentiment:") so generation predicts it.
	if !strings.HasSuffix(got[0].prompt, "Sentiment:") {
		t.Errorf("prompt does not withhold label: %q", got[0].prompt)
	}
	if strings.Contains(got[0].prompt, "positive") {
		t.Errorf("prompt leaks the label: %q", got[0].prompt)
	}
}

// writeGenAccuracy lays down a gen_N/results.json with the given held-out
// accuracy, as the evaluator would in accuracy mode.
func writeGenAccuracy(t *testing.T, runDir string, gen int, acc float64, correct, total int, trained bool) {
	t.Helper()
	genDir := filepath.Join(runDir, "gen_"+strconv.Itoa(gen))
	if err := os.MkdirAll(genDir, 0o755); err != nil {
		t.Fatal(err)
	}
	wr := weightsResults{Verdict: "PASS", Trained: trained, Accuracy: acc, Correct: correct, Total: total, Metric: "accuracy"}
	data, _ := json.MarshalIndent(wr, "", "  ")
	if err := os.WriteFile(filepath.Join(genDir, sia.NameResultsJSON), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestBestPriorAccuracyIsCausal verifies the higher-is-better gate's best-so-far
// lookup is causal (reads only strictly-earlier generations), ignores untrained
// gens, and tracks the MAXIMUM accuracy.
func TestBestPriorAccuracyIsCausal(t *testing.T) {
	e := &WeightsEvaluator{BaseModel: "m", HeldOutDir: "x"}
	runDir := t.TempDir()
	// Series: gen1=0.62, gen2=0.81 (best), gen3=0.75 (worse), gen4 untrained.
	writeGenAccuracy(t, runDir, 1, 0.62, 10, 16, true)
	writeGenAccuracy(t, runDir, 2, 0.81, 13, 16, true)
	writeGenAccuracy(t, runDir, 3, 0.75, 12, 16, true)
	writeGenAccuracy(t, runDir, 4, 0, 0, 0, false)

	// gen1 has no prior -> not ok.
	if _, _, ok := e.bestPriorAccuracy(filepath.Join(runDir, "gen_1")); ok {
		t.Error("gen1 should have no prior best")
	}
	// gen2 sees only gen1.
	if g, a, ok := e.bestPriorAccuracy(filepath.Join(runDir, "gen_2")); !ok || g != 1 || a != 0.62 {
		t.Errorf("gen2 best-prior = gen%d acc%.4f ok=%v, want gen1 0.62", g, a, ok)
	}
	// gen3 sees gen1+gen2; gen2 is the best (max accuracy).
	if g, a, ok := e.bestPriorAccuracy(filepath.Join(runDir, "gen_3")); !ok || g != 2 || a != 0.81 {
		t.Errorf("gen3 best-prior = gen%d acc%.4f ok=%v, want gen2 0.81", g, a, ok)
	}
	// gen5 (after untrained gen4) still finds gen2 as best and skips the untrained one.
	if g, a, ok := e.bestPriorAccuracy(filepath.Join(runDir, "gen_5")); !ok || g != 2 || a != 0.81 {
		t.Errorf("gen5 best-prior = gen%d acc%.4f ok=%v, want gen2 0.81 (untrained gen4 skipped)", g, a, ok)
	}
}

func TestEvaluateRequiresConfig(t *testing.T) {
	if _, err := (&WeightsEvaluator{HeldOutDir: "x"}).Evaluate(context.Background(), t.TempDir()); err == nil {
		t.Fatal("want error for missing BaseModel")
	}
	if _, err := (&WeightsEvaluator{BaseModel: "m"}).Evaluate(context.Background(), t.TempDir()); err == nil {
		t.Fatal("want error for missing HeldOutDir")
	}
}
