package main

import (
	"context"
	"os"
	"path/filepath"
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

func TestEvaluateRequiresConfig(t *testing.T) {
	if _, err := (&WeightsEvaluator{HeldOutDir: "x"}).Evaluate(context.Background(), t.TempDir()); err == nil {
		t.Fatal("want error for missing BaseModel")
	}
	if _, err := (&WeightsEvaluator{BaseModel: "m"}).Evaluate(context.Background(), t.TempDir()); err == nil {
		t.Fatal("want error for missing HeldOutDir")
	}
}
