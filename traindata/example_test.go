package traindata_test

import (
	"encoding/json"
	"fmt"

	sia "github.com/tmc/sia-apple-silicon"
	"github.com/tmc/sia-apple-silicon/traindata"
)

// Example renders a recorded single-trajectory execution into one training
// sample and reports its token count and how many tokens the loss trains on.
func Example() {
	// In production, load a real tokenizer with mlxlm.LoadTokenizer and adapt it
	// with renderer.Adapt; here a deterministic fake stands in.
	r, err := traindata.RendererForModel(newFakeTokenizer("Qwen/Qwen3-8B"), nil)
	if err != nil {
		panic(err)
	}

	exec := sia.Execution{Single: json.RawMessage(
		`[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]`)}

	samples, err := traindata.SamplesFromExecution(r, exec, traindata.EmitOptions{
		RoleToMask: traindata.RLMask,
	})
	if err != nil {
		panic(err)
	}

	trainable := 0
	for _, m := range samples[0].LossMask {
		if m {
			trainable++
		}
	}
	fmt.Printf("samples=%d tokens=%d trainable=%d\n",
		len(samples), len(samples[0].TokenIDs), trainable)
	// Output: samples=1 tokens=23 trainable=13
}
