// Package seed holds the initial decode-step sampler the meta agent starts from.
// It is the correct-but-naive candidate the agent then optimizes for throughput;
// the same source also produces gen-0's golden tokens, so gen-0 always passes
// the correctness gate and serves as the interleaved baseline.
//
// The seed is stored as candidate.go.txt (not a .go file) so it is not compiled
// into this package — it is a standalone package main program run on its own.
package seed

import _ "embed"

// Candidate is the seed sampler source, written into gen-1's working dir for the
// meta agent and used to produce the frozen golden baseline.
//
//go:embed candidate.go.txt
var Candidate string
