package traindata_test

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/tmc/mlx-go-experiments/renderer"
)

// fakeTokenizer is a deterministic, reversible tokenizer for tests. It registers
// the Qwen special tokens and reports offsets — the capabilities the Qwen3
// renderer needs — so rendered token ids and loss masks are reproducible and
// byte-assertable without any native tokenizer. It is not a real BPE. Modeled
// on the renderer package's example tokenizer.
type fakeTokenizer struct {
	model     string
	specByID  map[int32]string
	specByStr map[string]int32
	vocab     map[string]int32
	rev       map[int32]string
	next      int32
	eos       []int32
}

func newFakeTokenizer(model string) *fakeTokenizer {
	t := &fakeTokenizer{
		model:     model,
		specByID:  map[int32]string{},
		specByStr: map[string]int32{},
		vocab:     map[string]int32{},
		rev:       map[int32]string{},
		next:      100000,
	}
	var id int32 = 1
	for _, s := range []string{
		"<|im_start|>", "<|im_end|>", "<|endoftext|>",
		"<think>", "</think>", "<tool_call>", "</tool_call>",
		"<tool_response>", "</tool_response>",
	} {
		t.specByStr[s] = id
		t.specByID[id] = s
		id++
	}
	t.eos = []int32{t.specByStr["<|im_end|>"], t.specByStr["<|endoftext|>"]}
	return t
}

func fakeChunks(text string) []string {
	var out []string
	r := []rune(text)
	for i := 0; i < len(r); {
		if unicode.IsLetter(r[i]) || unicode.IsDigit(r[i]) {
			j := i + 1
			for j < len(r) && (unicode.IsLetter(r[j]) || unicode.IsDigit(r[j])) {
				j++
			}
			out = append(out, string(r[i:j]))
			i = j
		} else {
			out = append(out, string(r[i]))
			i++
		}
	}
	return out
}

func (t *fakeTokenizer) idFor(c string) int32 {
	if id, ok := t.vocab[c]; ok {
		return id
	}
	id := t.next
	t.next++
	t.vocab[c] = id
	t.rev[id] = c
	return id
}

func (t *fakeTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}
	cs := fakeChunks(text)
	out := make([]int32, len(cs))
	for i, c := range cs {
		out[i] = t.idFor(c)
	}
	return out, nil
}

func (t *fakeTokenizer) EncodeWithOffsets(text string) ([]renderer.OffsetToken, error) {
	if text == "" {
		return nil, nil
	}
	cs := fakeChunks(text)
	out := make([]renderer.OffsetToken, len(cs))
	pos := 0
	for i, c := range cs {
		out[i] = renderer.OffsetToken{ID: t.idFor(c), Start: pos, End: pos + len(c)}
		pos += len(c)
	}
	return out, nil
}

func (t *fakeTokenizer) DecodeWithOptions(tokens []int32, skipSpecial bool) (string, error) {
	var b strings.Builder
	for _, id := range tokens {
		if name, ok := t.specByID[id]; ok {
			if !skipSpecial {
				b.WriteString(name)
			}
			continue
		}
		c, ok := t.rev[id]
		if !ok {
			return "", fmt.Errorf("unknown id %d", id)
		}
		b.WriteString(c)
	}
	return b.String(), nil
}

func (t *fakeTokenizer) TokenByName(name string) int32 {
	if id, ok := t.specByStr[name]; ok {
		return id
	}
	return -1
}

func (t *fakeTokenizer) EOSTokenIDs() []int32 { return t.eos }
func (t *fakeTokenizer) ModelName() string    { return t.model }
