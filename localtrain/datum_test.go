package localtrain_test

import (
	"testing"

	"github.com/tmc/sia-apple-silicon/localtrain"
	"github.com/tmc/sia-apple-silicon/traindata"
)

func TestDatumFromSample(t *testing.T) {
	s := traindata.TrainingSample{
		TrajectoryIndex: 0,
		TokenIDs:        []int32{10, 11, 12, 13},
		LossMask:        []bool{false, false, true, true},
	}
	d := localtrain.DatumFromSample(s)

	gotInput := d.Input.Tokens()
	wantInput := []int{10, 11, 12, 13}
	if !eqInt(gotInput, wantInput) {
		t.Errorf("input = %v, want %v", gotInput, wantInput)
	}

	// Targets are the next-token shift; the last position repeats (masked out).
	wantTargets := []int{11, 12, 13, 13}
	if !eqInt(d.LossInput.TargetTokens, wantTargets) {
		t.Errorf("targets = %v, want %v", d.LossInput.TargetTokens, wantTargets)
	}

	// Weight at position i reflects LossMask[i+1] (the token being predicted);
	// the final position is always 0 (no next token).
	wantWeights := []float32{0, 1, 1, 0}
	if !eqF32(d.LossInput.Weights, wantWeights) {
		t.Errorf("weights = %v, want %v", d.LossInput.Weights, wantWeights)
	}

	if len(d.Input.Tokens()) != len(d.LossInput.TargetTokens) {
		t.Error("input and target lengths must match (Tinker batch contract)")
	}
}

func TestBatchFromSamplesSkipsEmpty(t *testing.T) {
	samples := []traindata.TrainingSample{
		{TokenIDs: []int32{1, 2}, LossMask: []bool{true, true}},
		{TokenIDs: nil}, // empty: skipped so the batch passes validation
		{TokenIDs: []int32{3, 4, 5}, LossMask: []bool{false, true, true}},
	}
	batch := localtrain.BatchFromSamples(samples)
	if len(batch) != 2 {
		t.Fatalf("got %d datums, want 2 (empty sample skipped)", len(batch))
	}
	if batch[0].Input.Len() != 2 || batch[1].Input.Len() != 3 {
		t.Errorf("datum lengths = %d,%d; want 2,3", batch[0].Input.Len(), batch[1].Input.Len())
	}
}

func eqInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func eqF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
