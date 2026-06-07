// Command leakguard-demo proves the anti-leakage honesty guard in
// sia.MLXTrainExecutor on camera: the held-out test set is structurally
// unreachable by the training agent.
//
// The P6 demo climax is a real LoRA weight-training loop whose gate catches
// overfitting via a held-out test_loss. The obvious skeptic's attack is "the
// agent just trained on the eval set, so test_loss is meaningless." This
// command answers that attack directly. It drives the REAL executor
// (sia.MLXTrainExecutor.RunTarget, the same entry point the orchestrator uses)
// through three acts:
//
//	ACT 1  data dir holds train/valid only        -> executor TRAINS; the
//	                                                  mlx-lm-train command line
//	                                                  it would run is printed,
//	                                                  proving the held-out path
//	                                                  is never an argument.
//	ACT 2  test.jsonl is dropped INTO the data dir -> executor REFUSES (fatal,
//	                                                  reported as feedback).
//	ACT 3  the spec asks for data_mix=../_heldout  -> path escape REFUSED before
//	                                                  any trainer runs.
//
// By default the trainer is a hermetic fake (a shell script that echoes its
// argv) so the demo runs in well under a second with no model download; the
// guard itself is exercised through the production code path either way. Point
// -train-bin at the real mlx-lm-train to run an actual fine-tune for ACT 1.
//
// Usage:
//
//	go run ./examples/weights/leakguard-demo                 # hermetic, fast, demoable
//	go run ./examples/weights/leakguard-demo -train-bin mlx-lm-train -base mlx-community/Qwen3-0.6B-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	sia "github.com/tmc/mlx-go-sia"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "leakguard-demo:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		trainBin  = flag.String("train-bin", "", "trainer binary; empty uses a hermetic fake that echoes its argv")
		baseModel = flag.String("base", "mlx-community/Qwen3-0.6B-4bit", "base model for -model")
		root      = flag.String("root", "", "scratch root; empty uses a temp dir")
	)
	flag.Parse()

	work, cleanup, err := scratchRoot(*root)
	if err != nil {
		return err
	}
	defer cleanup()

	bin := *trainBin
	hermetic := bin == ""
	if hermetic {
		bin, err = writeFakeTrainer(work)
		if err != nil {
			return err
		}
	}

	// The data dir the agent's trainer reads: train + valid ONLY. The held-out
	// test set lives in a SEPARATE _heldout/ dir that the executor is never told
	// about — it belongs to the evaluator, not the trainer.
	dataDir := filepath.Join(work, "data")
	heldoutDir := filepath.Join(work, "_heldout")
	if err := mkdirs(dataDir, heldoutDir); err != nil {
		return err
	}
	if err := writeFile(filepath.Join(dataDir, "train.jsonl"), sampleRows); err != nil {
		return err
	}
	if err := writeFile(filepath.Join(dataDir, "valid.jsonl"), sampleRows); err != nil {
		return err
	}
	// The eval rows. Note the path: OUTSIDE dataDir.
	if err := writeFile(filepath.Join(heldoutDir, "test.jsonl"), sampleRows); err != nil {
		return err
	}

	exec := &sia.MLXTrainExecutor{TrainBin: bin, BaseModel: *baseModel, DataDir: dataDir}

	banner("SIA P6 anti-leakage guard — live proof")
	fmt.Printf("  base model : %s\n", *baseModel)
	fmt.Printf("  trainer    : %s%s\n", bin, hermeticNote(hermetic))
	fmt.Printf("  -data dir  : %s   (train.jsonl, valid.jsonl)\n", dataDir)
	fmt.Printf("  held-out   : %s   (test.jsonl — NOT under -data)\n\n", heldoutDir)

	ctx := context.Background()

	// ── ACT 1 ───────────────────────────────────────────────────────────────
	// Clean data dir: the executor trains. We print the exact argv it ran so the
	// audience can SEE the held-out path is nowhere in it.
	act(1, "data dir holds train/valid only  →  executor TRAINS")
	spec := "lora_rank = 8\niters = 5\nlearning_rate = 1e-5\n"
	res := runGen(ctx, exec, work, "gen_1", spec)
	report(res)
	if hermetic {
		fmt.Printf("    mlx-lm-train argv (echoed by fake trainer):\n")
		printArgv(res.Stdout, dataDir, heldoutDir)
	}
	if !res.Success {
		return fmt.Errorf("ACT 1 expected success, got refusal: %s", res.ErrorMsg)
	}
	fmt.Println()

	// ── ACT 2 ───────────────────────────────────────────────────────────────
	// Now plant test.jsonl INSIDE the data dir — simulate an agent (or a careless
	// data-prep step) leaking the eval rows into the trainer's reach. The guard
	// must fatally refuse.
	act(2, "test.jsonl appears IN the -data dir  →  executor REFUSES")
	if err := writeFile(filepath.Join(dataDir, "test.jsonl"), sampleRows); err != nil {
		return err
	}
	fmt.Printf("    planted: %s\n", filepath.Join(dataDir, "test.jsonl"))
	res = runGen(ctx, exec, work, "gen_2", spec)
	report(res)
	if res.Success {
		return fmt.Errorf("ACT 2 LEAK: executor trained with test.jsonl present")
	}
	// Clean it back up so ACT 3 tests the orthogonal escape path.
	if err := os.Remove(filepath.Join(dataDir, "test.jsonl")); err != nil {
		return err
	}
	fmt.Println()

	// ── ACT 3 ───────────────────────────────────────────────────────────────
	// The agent tries to STEER -data at the held-out dir via a malicious data_mix
	// path escape. The mix whitelist refuses any value with a separator or "..".
	act(3, `spec sets data_mix="../_heldout"  →  path escape REFUSED`)
	escapeSpec := spec + `data_mix = "../_heldout"` + "\n"
	res = runGen(ctx, exec, work, "gen_3", escapeSpec)
	report(res)
	if res.Success {
		return fmt.Errorf("ACT 3 LEAK: data_mix path escape was not blocked")
	}
	fmt.Println()

	banner("VERDICT: held-out test set is structurally unreachable by the agent")
	fmt.Println("  • the trainer's -data is only ever train/valid (ACT 1 argv proves it)")
	fmt.Println("  • test.jsonl inside -data is a fatal refusal, not a silent train (ACT 2)")
	fmt.Println("  • data_mix can only pick a SUBDIR of -data, never escape it (ACT 3)")
	fmt.Println("  ⇒ the held-out test_loss the gate reads cannot have been trained on.")
	return nil
}

// runGen writes spec as train.py into a fresh gen dir and drives the real
// executor exactly as the orchestrator does.
func runGen(ctx context.Context, e *sia.MLXTrainExecutor, root, gen, spec string) sia.TargetResult {
	genDir := filepath.Join(root, gen)
	if err := os.MkdirAll(genDir, 0o755); err != nil {
		return sia.TargetResult{ErrorMsg: err.Error()}
	}
	specPath := filepath.Join(genDir, "train.py")
	if err := writeFile(specPath, spec); err != nil {
		return sia.TargetResult{ErrorMsg: err.Error()}
	}
	res, err := e.RunTarget(ctx, sia.TargetRequest{
		AgentPath:  specPath,
		WorkingDir: genDir,
		StdoutLog:  filepath.Join(genDir, "train_stdout.log"),
	})
	if err != nil {
		// An executor-level Go error (e.g. trainer binary missing) — still show it.
		res.ErrorMsg = firstNonEmpty(res.ErrorMsg, err.Error())
	}
	return res
}

func report(res sia.TargetResult) {
	if res.Success {
		fmt.Printf("    → RESULT: TRAINED  (Success=true)\n")
		return
	}
	fmt.Printf("    → RESULT: REFUSED  (Success=false)\n")
	fmt.Printf("    → reason : %s\n", res.ErrorMsg)
}

// printArgv pulls the echoed argv line out of the fake trainer's stdout and
// flags whether the held-out dir appears anywhere in it.
func printArgv(stdout, dataDir, heldoutDir string) {
	for _, line := range strings.Split(strings.TrimRight(stdout, "\n"), "\n") {
		if strings.HasPrefix(line, "ARGV ") {
			fmt.Printf("      %s\n", strings.TrimPrefix(line, "ARGV "))
		}
	}
	leaked := strings.Contains(stdout, heldoutDir)
	mark := "✓ held-out path absent from argv"
	if leaked {
		mark = "✗ LEAK: held-out path present in argv"
	}
	fmt.Printf("      %s\n", mark)
}

// writeFakeTrainer drops a hermetic shell script that echoes its argv (so we can
// show exactly what the executor would have invoked) and exits 0.
func writeFakeTrainer(dir string) (string, error) {
	path := filepath.Join(dir, "fake-mlx-lm-train")
	script := "#!/bin/sh\nprintf 'ARGV %s\\n' \"$*\"\nexit 0\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		return "", err
	}
	return path, nil
}

func scratchRoot(root string) (string, func(), error) {
	if root != "" {
		if err := os.MkdirAll(root, 0o755); err != nil {
			return "", nil, err
		}
		return root, func() {}, nil
	}
	dir, err := os.MkdirTemp("", "leakguard-demo-")
	if err != nil {
		return "", nil, err
	}
	return dir, func() { os.RemoveAll(dir) }, nil
}

func mkdirs(dirs ...string) error {
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}
	return nil
}

func writeFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func hermeticNote(hermetic bool) string {
	if hermetic {
		return "  (hermetic fake; pass -train-bin for a real fine-tune)"
	}
	return ""
}

func banner(s string) {
	line := strings.Repeat("═", 74)
	fmt.Printf("%s\n%s\n%s\n", line, s, line)
}

func act(n int, title string) {
	fmt.Printf("ACT %d  %s\n", n, title)
}

// sampleRows is a couple of trivially-valid chat-format rows; content is
// irrelevant to the guard, which keys on the test.jsonl path, not row contents.
const sampleRows = `{"messages":[{"role":"user","content":"2+2"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"3+3"},{"role":"assistant","content":"6"}]}
`
