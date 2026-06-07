// Command sia-traindata renders recorded SIA agent trajectories into
// token-level training data (train_data.jsonl).
//
// It is the post-hoc, Go-native analog of the tinker_cookbook.renderers the SIA
// reference delegates to the generated train.py: it walks a finished run,
// loads each generation's recorded execution with the faithful loader, renders
// every well-formed trajectory through the target model's renderer, and writes
// (token_ids, loss_mask) samples as JSONL. The orchestrator itself never does
// this; sia-traindata is an opt-in tool.
//
// Usage:
//
//	sia-traindata -run-dir ./runs/run_1 -model Qwen/Qwen3-8B -tokenizer /path/to/model
//	sia-traindata -run-dir ./runs/run_1 -all -mask sft -tokenizer /path/to/model
//
// The JSONL it writes feeds either weights-mode backend: the mlx-lm-train
// declarative path or sia-tinker-train (localtinker).
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/tmc/mlx-go-experiments/renderer"
	"github.com/tmc/mlx-go-lm/mlxlm"
	sia "github.com/tmc/sia-apple-silicon"
	"github.com/tmc/sia-apple-silicon/traindata"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sia-traindata: ")

	runDir := flag.String("run-dir", "", "run directory to render (e.g. ./runs/run_1) (required)")
	gen := flag.Int("gen", 0, "render only this generation number; 0 renders the latest")
	all := flag.Bool("all", false, "render every generation")
	model := flag.String("model", "", "target model id for renderer family selection (e.g. Qwen/Qwen3-8B) (required)")
	tokDir := flag.String("tokenizer", "", "tokenizer/model directory for mlxlm.LoadTokenizer (required)")
	mask := flag.String("mask", "rl", "loss policy: rl (sampled mask) or sft (assistant tokens)")
	out := flag.String("o", "", "output path; '-' for stdout; empty writes <genDir>/train_data.jsonl per gen")
	maxLogSize := flag.Int64("max-log-size", 0, "per-trajectory-file size cap in bytes (0 = no cap)")
	flag.Parse()

	if *runDir == "" || *model == "" || *tokDir == "" {
		flag.Usage()
		log.Fatal("-run-dir, -model and -tokenizer are required")
	}

	opts, err := maskOptions(*mask)
	if err != nil {
		log.Fatal(err)
	}

	tok, err := loadRendererTokenizer(*tokDir, *model)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}
	r, err := traindata.RendererForModel(tok, nil)
	if err != nil {
		log.Fatalf("build renderer for %q: %v", *model, err)
	}

	gens, err := selectGenerations(*runDir, *gen, *all)
	if err != nil {
		log.Fatal(err)
	}
	if len(gens) == 0 {
		log.Fatalf("no generation directories under %s", *runDir)
	}

	layout := sia.NewRunLayout(*runDir)
	total := 0
	for _, n := range gens {
		genDir := layout.GenDir(n)
		exec := sia.LoadExecution(genDir, *maxLogSize)
		samples, rerr := traindata.SamplesFromExecution(r, exec, opts)
		if rerr != nil {
			log.Printf("gen %d: %v", n, rerr) // skipped trajectories are reported, not fatal
		}
		w, closeFn, dest, err := openOutput(*out, genDir)
		if err != nil {
			log.Fatalf("gen %d: open output: %v", n, err)
		}
		written, err := traindata.WriteJSONL(w, samples)
		closeFn()
		if err != nil {
			log.Fatalf("gen %d: write: %v", n, err)
		}
		log.Printf("gen %d: wrote %d sample(s) -> %s", n, written, dest)
		total += written
		if *out == "-" {
			// stdout aggregates across gens; do not reopen per gen header.
		}
	}
	log.Printf("done: %d sample(s) across %d generation(s)", total, len(gens))
}

func maskOptions(mask string) (traindata.EmitOptions, error) {
	switch mask {
	case "rl":
		return traindata.EmitOptions{RoleToMask: traindata.RLMask}, nil
	case "sft":
		return traindata.EmitOptions{
			RoleToMask:      traindata.SFTMask,
			ContentSFTRoles: traindata.AssistantPlusToolSFTRoles,
		}, nil
	default:
		return traindata.EmitOptions{}, fmt.Errorf("unknown -mask %q (want rl or sft)", mask)
	}
}

// selectGenerations resolves which gen_N directories to render. With all, every
// gen is returned in order; with gen>0, just that one; otherwise the latest.
func selectGenerations(runDir string, gen int, all bool) ([]int, error) {
	if gen > 0 {
		return []int{gen}, nil
	}
	matches, err := filepath.Glob(filepath.Join(runDir, "gen_*"))
	if err != nil {
		return nil, fmt.Errorf("glob generations: %w", err)
	}
	var nums []int
	for _, m := range matches {
		info, err := os.Stat(m)
		if err != nil || !info.IsDir() {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(filepath.Base(m), "gen_%d", &n); err == nil {
			nums = append(nums, n)
		}
	}
	sort.Ints(nums)
	if all || len(nums) == 0 {
		return nums, nil
	}
	return nums[len(nums)-1:], nil
}

// openOutput returns the writer for a generation's samples. dest is a label for
// logging. The caller must call closeFn.
func openOutput(out, genDir string) (w *os.File, closeFn func(), dest string, err error) {
	if out == "-" {
		return os.Stdout, func() {}, "stdout", nil
	}
	path := out
	if path == "" {
		path = filepath.Join(genDir, traindata.NameTrainData)
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, nil, "", err
	}
	return f, func() { f.Close() }, path, nil
}

// loadRendererTokenizer loads an mlx-go-lm BPE tokenizer from dir and adapts it
// to a renderer.Tokenizer carrying modelName for family selection. It wires the
// public mlxlm offset helper so the renderer can attribute mixed body/scaffold
// segments; a tokenizer without offsets degrades to plain encoding.
func loadRendererTokenizer(dir, modelName string) (renderer.Tokenizer, error) {
	tok, err := mlxlm.LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	mlxTok := mlxTokenizer{Tokenizer: tok}

	// Probe offset support once; if absent, pass nil so the renderer falls back
	// to plain encoding rather than failing on every call.
	if _, perr := mlxlm.EncodeWithOffsets(tok, ""); perr == mlxlm.ErrOffsetsUnsupported {
		return renderer.Adapt(mlxTok, modelName, nil), nil
	}
	offsets := func(text string) ([]renderer.MLXOffsetToken, error) {
		ots, err := mlxlm.EncodeWithOffsets(tok, text)
		if err != nil {
			return nil, err
		}
		out := make([]renderer.MLXOffsetToken, len(ots))
		for i, o := range ots {
			out[i] = renderer.MLXOffsetToken{ID: o.ID, Start: o.Start, End: o.End}
		}
		return out, nil
	}
	return renderer.Adapt(mlxTok, modelName, offsets), nil
}

// mlxTokenizer adapts an mlx-go-lm Tokenizer to renderer.MLXTokenizer. The
// Tokenizer interface lacks TokenByName; mlxlm's pathTokenizer forwards it, so
// the embedded value already answers TokenByName.
type mlxTokenizer struct {
	mlxlm.Tokenizer
}

func (m mlxTokenizer) TokenByName(name string) int32 {
	if tn, ok := m.Tokenizer.(interface{ TokenByName(string) int32 }); ok {
		return tn.TokenByName(name)
	}
	return -1
}
