package oracle

import (
	"math"
	"sort"
)

// This file is the FROZEN reference sampler. It defines the exact decode-sampling
// algorithm and the exact RNG the candidate must reproduce token-for-token. It
// lives in the protected oracle package and is never copied into an agent's
// working directory — the agent reimplements the same contract in candidate.go
// and is scored on producing the identical token sequence, only faster.
//
// The algorithm, in order, matches mlx-go-lm's sampler structure:
//
//  1. Greedy when temperature < 0.01: emit argmax(logits) (ties broken by lowest
//     index). The RNG is not consumed.
//  2. Otherwise: divide logits by temperature; keep the top-k highest logits
//     (k <= 0 keeps all); convert to probabilities with a numerically-stable
//     softmax; apply top-p (nucleus) truncation over the descending-probability
//     order; renormalize; draw one token with the shared RNG.
//
// The RNG is a splitmix64 seeded once from Fixtures.Seed and advanced once per
// decode step, so the sequence is fully reproducible and independent of timing.

// referenceTokens runs the frozen reference sampler over every step of fx and
// returns the emitted token sequence — the golden output.
func referenceTokens(fx Fixtures) []int {
	rng := newSplitMix64(fx.Seed)
	tokens := make([]int, len(fx.Steps))
	for i, logits := range fx.Steps {
		tokens[i] = referenceSample(logits, fx.Config, &rng)
	}
	return tokens
}

// referenceSample turns one logits row into one token id using the frozen
// algorithm. It is intentionally written for clarity, not speed — the whole
// point of the demo is that a candidate can make an equivalent routine faster.
func referenceSample(logits []float32, cfg Config, rng *splitMix64) int {
	// 1. Greedy fast path: no RNG consumption, lowest-index tie-break.
	if cfg.Temperature < 0.01 {
		return argmax(logits)
	}

	// 2a. Temperature scaling.
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = float64(v) / cfg.Temperature
	}

	// Index set we still consider; starts as all vocab indices.
	idx := make([]int, len(scaled))
	for i := range idx {
		idx[i] = i
	}

	// 2b. Top-k: keep the k highest-logit indices. Sort by logit desc, then by
	// index asc for a deterministic order among equal logits.
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

	// 2c. Softmax over the kept indices (numerically stable).
	probs := softmax(scaled, idx)

	// 2d. Top-p (nucleus): over indices sorted by probability desc (index asc on
	// ties), keep the smallest prefix whose cumulative probability first reaches
	// top_p; always keep at least one. Then renormalize.
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

	// 2e. Renormalize the surviving probabilities and draw one token.
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
	return order[len(order)-1] // floating-point guard: return the last kept token
}

// argmax returns the index of the largest logit, breaking ties by lowest index.
func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// softmax returns a full-length probability slice that is non-zero only at the
// indices in keep, computed with the standard max-shift for stability.
func softmax(scaled []float64, keep []int) []float64 {
	maxLogit := math.Inf(-1)
	for _, i := range keep {
		if scaled[i] > maxLogit {
			maxLogit = scaled[i]
		}
	}
	probs := make([]float64, len(scaled))
	var sum float64
	for _, i := range keep {
		e := math.Exp(scaled[i] - maxLogit)
		probs[i] = e
		sum += e
	}
	for _, i := range keep {
		probs[i] /= sum
	}
	return probs
}

// generateFixtures builds a deterministic batch of decode-step logits rows from
// seed, so the demo is fully reproducible with no model download. A separate
// splitmix64 stream (seed^fixturesSalt) fills the logits, keeping the sampling
// RNG stream (seeded from seed) independent of fixture generation.
func generateFixtures(cfg Config, seed uint64, steps, vocab int) Fixtures {
	const fixturesSalt = 0x9e3779b97f4a7c15
	gen := newSplitMix64(seed ^ fixturesSalt)
	rows := make([][]float32, steps)
	for s := range rows {
		row := make([]float32, vocab)
		for v := range row {
			// Logits in roughly [-6, 6); a few peaks emerge from the draws so
			// top-k/top-p actually bite.
			row[v] = float32(gen.float64()*12 - 6)
		}
		rows[s] = row
	}
	return Fixtures{Config: cfg, Seed: seed, Steps: rows}
}

// splitMix64 is a tiny, fully-specified PRNG. The candidate must reproduce it
// exactly (same constants, same advance) to match the golden token sequence; it
// is documented in the candidate contract so the agent reimplements it verbatim.
type splitMix64 struct{ state uint64 }

func newSplitMix64(seed uint64) splitMix64 { return splitMix64{state: seed} }

// next advances the generator and returns the next 64-bit value.
func (s *splitMix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// float64 returns a value in [0, 1) using the top 53 bits, the standard
// construction so the candidate can reproduce it bit-for-bit.
func (s *splitMix64) float64() float64 {
	return float64(s.next()>>11) / float64(uint64(1)<<53)
}
