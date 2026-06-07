package oracle

// This file holds candidate program sources used by the oracle tests. Each is a
// standalone package main combining candidateContract (stdin I/O + RNG + the two
// run modes) with one Sample body. They mirror the shipped seed's contract so a
// correct body reproduces the golden token sequence exactly.

// candidateContract is everything except Sample: it is byte-for-byte compatible
// with the shipped seed so a correct Sample body yields identical tokens.
const candidateContract = `package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"time"
)

type Config struct {
	Temperature float64 ` + "`json:\"temperature\"`" + `
	TopK        int     ` + "`json:\"top_k\"`" + `
	TopP        float64 ` + "`json:\"top_p\"`" + `
}

type Fixtures struct {
	Config Config      ` + "`json:\"config\"`" + `
	Seed   uint64      ` + "`json:\"seed\"`" + `
	Steps  [][]float32 ` + "`json:\"steps\"`" + `
}

func main() {
	mode := flag.String("mode", "tokens", "tokens | bench")
	flag.Parse()
	var fx Fixtures
	if err := json.NewDecoder(bufio.NewReader(os.Stdin)).Decode(&fx); err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
	switch *mode {
	case "tokens":
		w := bufio.NewWriter(os.Stdout)
		defer w.Flush()
		rng := newSplitMix64(fx.Seed)
		for _, logits := range fx.Steps {
			fmt.Fprintln(w, Sample(logits, fx.Config, &rng))
		}
	case "bench":
		const budget = 80 * time.Millisecond
		var tokens int
		start := time.Now()
		for time.Since(start) < budget {
			rng := newSplitMix64(fx.Seed)
			for _, logits := range fx.Steps {
				_ = Sample(logits, fx.Config, &rng)
				tokens++
			}
		}
		fmt.Printf("tokens_per_sec %.3f\n", float64(tokens)/time.Since(start).Seconds())
	}
}

type splitMix64 struct{ state uint64 }

func newSplitMix64(seed uint64) splitMix64 { return splitMix64{state: seed} }

func (s *splitMix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (s *splitMix64) float64() float64 {
	return float64(s.next()>>11) / float64(uint64(1)<<53)
}

var _ = sort.Ints
var _ = math.Exp
`

// naiveSample is the seed body (full sort), identical to the shipped seed.
const naiveSample = `
func Sample(logits []float32, cfg Config, rng *splitMix64) int {
	if cfg.Temperature < 0.01 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = float64(v) / cfg.Temperature
	}
	idx := make([]int, len(scaled))
	for i := range idx {
		idx[i] = i
	}
	if cfg.TopK > 0 && cfg.TopK < len(idx) {
		sort.Slice(idx, func(a, b int) bool {
			ia, ib := idx[a], idx[b]
			if scaled[ia] != scaled[ib] {
				return scaled[ia] > scaled[ib]
			}
			return ia < ib
		})
		idx = idx[:cfg.TopK]
	}
	maxLogit := math.Inf(-1)
	for _, i := range idx {
		if scaled[i] > maxLogit {
			maxLogit = scaled[i]
		}
	}
	probs := make([]float64, len(scaled))
	var sum float64
	for _, i := range idx {
		e := math.Exp(scaled[i] - maxLogit)
		probs[i] = e
		sum += e
	}
	for _, i := range idx {
		probs[i] /= sum
	}
	order := make([]int, len(idx))
	copy(order, idx)
	sort.Slice(order, func(a, b int) bool {
		pa, pb := probs[order[a]], probs[order[b]]
		if pa != pb {
			return pa > pb
		}
		return order[a] < order[b]
	})
	if cfg.TopP > 0 && cfg.TopP < 1.0 {
		var cum float64
		cut := len(order)
		for i, t := range order {
			cum += probs[t]
			if cum >= cfg.TopP {
				cut = i + 1
				break
			}
		}
		order = order[:cut]
	}
	var total float64
	for _, t := range order {
		total += probs[t]
	}
	r := rng.float64() * total
	var acc float64
	for _, t := range order {
		acc += probs[t]
		if r < acc {
			return t
		}
	}
	return order[len(order)-1]
}
`

// fasterSample keeps the same algorithm but sorts only the kept indices once
// (the full-vocab sort + the prob re-sort collapse into a single sort on the
// top-k slice). It emits the identical token sequence.
const fasterSample = `
func Sample(logits []float32, cfg Config, rng *splitMix64) int {
	if cfg.Temperature < 0.01 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	n := len(logits)
	scaled := make([]float64, n)
	for i, v := range logits {
		scaled[i] = float64(v) / cfg.Temperature
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		if scaled[ia] != scaled[ib] {
			return scaled[ia] > scaled[ib]
		}
		return ia < ib
	})
	keep := n
	if cfg.TopK > 0 && cfg.TopK < keep {
		keep = cfg.TopK
	}
	order := idx[:keep]
	maxLogit := scaled[order[0]]
	probs := make([]float64, n)
	var sum float64
	for _, i := range order {
		e := math.Exp(scaled[i] - maxLogit)
		probs[i] = e
		sum += e
	}
	for _, i := range order {
		probs[i] /= sum
	}
	if cfg.TopP > 0 && cfg.TopP < 1.0 {
		var cum float64
		cut := len(order)
		for i, t := range order {
			cum += probs[t]
			if cum >= cfg.TopP {
				cut = i + 1
				break
			}
		}
		order = order[:cut]
	}
	var total float64
	for _, t := range order {
		total += probs[t]
	}
	r := rng.float64() * total
	var acc float64
	for _, t := range order {
		acc += probs[t]
		if r < acc {
			return t
		}
	}
	return order[len(order)-1]
}
`

// wrongSample drops the temperature scaling, so its probabilities — and the
// tokens it draws — differ from the golden. The oracle must reject it.
const wrongSample = `
func Sample(logits []float32, cfg Config, rng *splitMix64) int {
	if cfg.Temperature < 0.01 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = float64(v) // BUG: no temperature scaling
	}
	idx := make([]int, len(scaled))
	for i := range idx {
		idx[i] = i
	}
	if cfg.TopK > 0 && cfg.TopK < len(idx) {
		sort.Slice(idx, func(a, b int) bool {
			ia, ib := idx[a], idx[b]
			if scaled[ia] != scaled[ib] {
				return scaled[ia] > scaled[ib]
			}
			return ia < ib
		})
		idx = idx[:cfg.TopK]
	}
	maxLogit := math.Inf(-1)
	for _, i := range idx {
		if scaled[i] > maxLogit {
			maxLogit = scaled[i]
		}
	}
	probs := make([]float64, len(scaled))
	var sum float64
	for _, i := range idx {
		e := math.Exp(scaled[i] - maxLogit)
		probs[i] = e
		sum += e
	}
	for _, i := range idx {
		probs[i] /= sum
	}
	order := make([]int, len(idx))
	copy(order, idx)
	sort.Slice(order, func(a, b int) bool {
		pa, pb := probs[order[a]], probs[order[b]]
		if pa != pb {
			return pa > pb
		}
		return order[a] < order[b]
	})
	var total float64
	for _, t := range order {
		total += probs[t]
	}
	r := rng.float64() * total
	var acc float64
	for _, t := range order {
		acc += probs[t]
		if r < acc {
			return t
		}
	}
	return order[len(order)-1]
}
`
