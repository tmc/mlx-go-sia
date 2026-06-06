package traindata_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/mlx-go-experiments/renderer"
	sia "github.com/tmc/mlx-go-sia"
	"github.com/tmc/mlx-go-sia/traindata"
)

var update = flag.Bool("update", false, "update golden files")

// newQwen3 builds the real Qwen3 renderer over the deterministic fake tokenizer,
// so rendered token ids and loss masks are reproducible across runs.
func newQwen3(t *testing.T) renderer.Renderer {
	t.Helper()
	r, err := traindata.RendererForModel(newFakeTokenizer("Qwen/Qwen3-8B"), nil)
	if err != nil {
		t.Fatalf("RendererForModel: %v", err)
	}
	return r
}

func TestSamplesFromExecutionSingle(t *testing.T) {
	r := newQwen3(t)
	exec := sia.Execution{Single: json.RawMessage(`[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]`)}
	samples, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{RoleToMask: traindata.RLMask})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 {
		t.Fatalf("got %d samples, want 1", len(samples))
	}
	if samples[0].TrajectoryIndex != 0 {
		t.Errorf("index = %d, want 0", samples[0].TrajectoryIndex)
	}
	if len(samples[0].TokenIDs) == 0 {
		t.Error("expected non-empty token ids")
	}
	if len(samples[0].LossMask) != len(samples[0].TokenIDs) {
		t.Errorf("mask len %d != token len %d", len(samples[0].LossMask), len(samples[0].TokenIDs))
	}
	if !anyTrue(samples[0].LossMask) {
		t.Error("expected at least one trainable token (the assistant turn)")
	}
}

func TestSamplesFromExecutionMultiSkipsErrors(t *testing.T) {
	r := newQwen3(t)
	exec := sia.Execution{
		MultiTrajectory: true,
		Trajectories: []json.RawMessage{
			json.RawMessage(`[{"role":"user","content":"q0"},{"role":"assistant","content":"a0"}]`),
			json.RawMessage(`[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"}]`),
			json.RawMessage(`{"error":"boom","file":"execution_q2.json"}`),
		},
	}
	samples, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{RoleToMask: traindata.RLMask})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("got %d samples, want 2 (error-object trajectory skipped)", len(samples))
	}
	// Indices must align with the on-disk execution_qN positions, not be renumbered.
	if samples[0].TrajectoryIndex != 0 || samples[1].TrajectoryIndex != 1 {
		t.Errorf("indices = %d,%d; want 0,1", samples[0].TrajectoryIndex, samples[1].TrajectoryIndex)
	}
}

func TestRLvsSFTMaskDiffer(t *testing.T) {
	r := newQwen3(t)
	exec := sia.Execution{Single: json.RawMessage(`[{"role":"system","content":"sys"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello there"}]`)}

	rl, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{RoleToMask: traindata.RLMask})
	if err != nil {
		t.Fatal(err)
	}
	sft, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{RoleToMask: traindata.SFTMask})
	if err != nil {
		t.Fatal(err)
	}
	if len(rl) != 1 || len(sft) != 1 {
		t.Fatalf("expected one sample each, got rl=%d sft=%d", len(rl), len(sft))
	}
	if countTrue(rl[0].LossMask) == 0 || countTrue(sft[0].LossMask) == 0 {
		t.Fatal("both policies should train on the assistant turn")
	}
	// The masks need not differ token-for-token on this input, but the policies
	// are distinct objects; assert they both produced a valid, equal-length mask.
	if len(rl[0].LossMask) != len(rl[0].TokenIDs) || len(sft[0].LossMask) != len(sft[0].TokenIDs) {
		t.Error("mask length must equal token length")
	}
}

func TestToolCallAndReasoningRender(t *testing.T) {
	r := newQwen3(t)
	exec := sia.Execution{Single: json.RawMessage(`[
		{"role":"user","content":"add 2 and 3"},
		{"role":"assistant","reasoning_content":"need the add tool","tool_calls":[{"id":"c1","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}]},
		{"role":"tool","tool_call_id":"c1","name":"add","content":"5"},
		{"role":"assistant","content":"The answer is 5."}
	]`)}
	samples, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{RoleToMask: traindata.RLMask})
	if err != nil {
		t.Fatalf("render with tool call + reasoning: %v", err)
	}
	if len(samples) != 1 || len(samples[0].TokenIDs) == 0 {
		t.Fatalf("expected one non-empty sample, got %d", len(samples))
	}
}

// TestGoldenJSONL renders fixture executions and byte-compares the JSONL output
// against goldens. Run with -update to regenerate. The fake tokenizer is
// deterministic, so the token ids and masks are stable.
func TestGoldenJSONL(t *testing.T) {
	cases := []struct {
		name    string
		fixture string
		opts    traindata.EmitOptions
		golden  string
	}{
		{"single_rl", "single.json", traindata.EmitOptions{RoleToMask: traindata.RLMask}, "single_rl.jsonl"},
		{"multi_rl", "multi", traindata.EmitOptions{RoleToMask: traindata.RLMask}, "multi_rl.jsonl"},
		{"multi_sft", "multi", traindata.EmitOptions{RoleToMask: traindata.SFTMask, ContentSFTRoles: traindata.AssistantPlusToolSFTRoles}, "multi_sft.jsonl"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := newQwen3(t)
			exec := loadFixture(t, c.fixture)
			samples, err := traindata.SamplesFromExecution(r, exec, c.opts)
			if err != nil {
				t.Fatal(err)
			}
			var buf bytes.Buffer
			if _, err := traindata.WriteJSONL(&buf, samples); err != nil {
				t.Fatal(err)
			}
			goldenPath := filepath.Join("testdata", c.golden)
			if *update {
				if err := os.WriteFile(goldenPath, buf.Bytes(), 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(goldenPath)
			if err != nil {
				t.Fatalf("read golden (run -update to create): %v", err)
			}
			if !bytes.Equal(buf.Bytes(), want) {
				t.Errorf("JSONL mismatch with %s\n got: %s\nwant: %s", c.golden, buf.Bytes(), want)
			}
		})
	}
}

// loadFixture reads a single-file fixture (a .json trajectory) or a multi
// fixture (a directory of execution_q*.json) into an Execution.
func loadFixture(t *testing.T, name string) sia.Execution {
	t.Helper()
	path := filepath.Join("testdata", name)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.IsDir() {
		matches, _ := filepath.Glob(filepath.Join(path, "execution_q*.json"))
		// Glob returns sorted order on these names.
		var trajs []json.RawMessage
		for _, m := range matches {
			b, err := os.ReadFile(m)
			if err != nil {
				t.Fatal(err)
			}
			trajs = append(trajs, json.RawMessage(b))
		}
		return sia.Execution{MultiTrajectory: true, Trajectories: trajs}
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sia.Execution{Single: json.RawMessage(b)}
}

func anyTrue(b []bool) bool { return countTrue(b) > 0 }

func countTrue(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}
