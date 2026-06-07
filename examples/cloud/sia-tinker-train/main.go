// Command sia-tinker-train trains a LoRA on SIA-derived training data using a
// local localtinker coordinator.
//
// It is the Go-native, hosted-Tinker-free version of what the SIA reference
// delegates to the generated train.py: read a train_data.jsonl produced by
// sia-traindata, map each sample onto the Tinker cross-entropy contract, and
// run CreateLoRA → ForwardBackward → OptimStep → Save against a localtinker
// coordinator on MLX. It does NOT collide with sia-train, which drives the
// separate mlx-lm-train declarative-spec backend.
//
// Usage:
//
//	# in one shell: localtinker serve -addr 127.0.0.1:8080 -home /tmp/lt-home
//	sia-tinker-train -data ./runs/run_1/gen_3/train_data.jsonl \
//	    -base-model Qwen/Qwen3-8B -model-path /models/Qwen3-8B -rank 8 -epochs 1
//
// The in-process tinker handle validates and shapes the batch; actual MLX
// forward/backward executes through the coordinator. When execution is not
// available the run reports tinker.ErrUnsupported.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/tmc/localtinker/tinker"
	"github.com/tmc/sia-apple-silicon/localtrain"
	"github.com/tmc/sia-apple-silicon/traindata"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sia-tinker-train: ")

	data := flag.String("data", "", "train_data.jsonl produced by sia-traindata (required)")
	baseModel := flag.String("base-model", "", "Tinker base model id, e.g. Qwen/Qwen3-8B (required)")
	modelPath := flag.String("model-path", "", "local model asset directory the coordinator loads (required)")
	root := flag.String("root", "", "client root dir for run state (default a temp dir)")
	rank := flag.Int("rank", 8, "LoRA rank")
	epochs := flag.Int("epochs", 1, "number of forward-backward/optim passes over the batch")
	out := flag.String("out", "sia-lora", "checkpoint name")
	maxContext := flag.Int("max-context", 0, "model max context (0 = unset)")
	flag.Parse()

	if *data == "" || *baseModel == "" || *modelPath == "" {
		flag.Usage()
		log.Fatal("-data, -base-model and -model-path are required")
	}

	samples, err := readSamples(*data)
	if err != nil {
		log.Fatalf("read %s: %v", *data, err)
	}
	batch := localtrain.BatchFromSamples(samples)
	if len(batch) == 0 {
		log.Fatalf("no usable samples in %s", *data)
	}
	log.Printf("loaded %d sample(s), %d datum(s)", len(samples), len(batch))

	rootDir := *root
	if rootDir == "" {
		rootDir, err = os.MkdirTemp("", "sia-tinker-train-")
		if err != nil {
			log.Fatalf("temp root: %v", err)
		}
	}
	client, err := tinker.New(tinker.Config{
		RootDir: rootDir,
		Models: staticRegistry{spec: tinker.ModelSpec{
			Name:       *baseModel,
			Path:       *modelPath,
			MaxContext: *maxContext,
		}},
	})
	if err != nil {
		log.Fatalf("new client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()
	ckpt, err := localtrain.Train(ctx, client, batch, localtrain.TrainOptions{
		BaseModel:      *baseModel,
		Rank:           *rank,
		Epochs:         *epochs,
		CheckpointName: *out,
		Logf:           log.Printf,
	})
	if err != nil {
		log.Fatalf("train: %v", err)
	}
	fmt.Printf("saved checkpoint: %+v\n", ckpt)
}

// readSamples reads JSONL training samples written by traindata.WriteJSONL.
func readSamples(path string) ([]traindata.TrainingSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var samples []traindata.TrainingSample
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<26) // allow long token-id lines
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var s traindata.TrainingSample
		if err := json.Unmarshal(line, &s); err != nil {
			return nil, fmt.Errorf("decode sample: %w", err)
		}
		samples = append(samples, s)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return samples, nil
}

// staticRegistry resolves one base model to a fixed local asset.
type staticRegistry struct{ spec tinker.ModelSpec }

func (r staticRegistry) Resolve(_ context.Context, name string) (tinker.ModelSpec, error) {
	if name != r.spec.Name {
		return tinker.ModelSpec{}, fmt.Errorf("unknown model %q (have %q)", name, r.spec.Name)
	}
	return r.spec, nil
}

func (r staticRegistry) List(context.Context) ([]tinker.ModelInfo, error) {
	return []tinker.ModelInfo{{Name: r.spec.Name, MaxContext: r.spec.MaxContext}}, nil
}
