package localtrain_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/localtinker/tinker"
	"github.com/tmc/sia-apple-silicon/localtrain"
	"github.com/tmc/sia-apple-silicon/traindata"
)

// testRegistry resolves a single fake local model.
type testRegistry struct{}

func (testRegistry) Resolve(context.Context, string) (tinker.ModelSpec, error) {
	return tinker.ModelSpec{Name: "test-model", Path: "/models/test", MaxContext: 2048}, nil
}

func (testRegistry) List(context.Context) ([]tinker.ModelInfo, error) {
	return []tinker.ModelInfo{{Name: "test-model", MaxContext: 2048}}, nil
}

func newTestClient(t *testing.T) *tinker.Client {
	t.Helper()
	c, err := tinker.New(tinker.Config{RootDir: t.TempDir(), Models: testRegistry{}})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestTrainValidatesArgs(t *testing.T) {
	ctx := context.Background()
	batch := localtrain.BatchFromSamples([]traindata.TrainingSample{
		{TokenIDs: []int32{1, 2, 3}, LossMask: []bool{false, true, true}},
	})

	if _, err := localtrain.Train(ctx, nil, batch, localtrain.TrainOptions{}); err == nil {
		t.Error("nil client should error")
	}
	if _, err := localtrain.Train(ctx, newTestClient(t), nil, localtrain.TrainOptions{}); err == nil {
		t.Error("empty batch should error")
	}
}

// TestTrainDrivesLoop confirms Train shapes a valid batch and reaches the
// coordinator's execution call. The in-process tinker handle does not execute
// MLX yet, so the forward-backward step returns tinker.ErrUnsupported; Train
// must surface it rather than fail earlier on a malformed batch.
func TestTrainDrivesLoop(t *testing.T) {
	ctx := context.Background()
	client := newTestClient(t)
	batch := localtrain.BatchFromSamples([]traindata.TrainingSample{
		{TokenIDs: []int32{10, 11, 12, 13}, LossMask: []bool{false, false, true, true}},
	})

	_, err := localtrain.Train(ctx, client, batch, localtrain.TrainOptions{
		BaseModel: "test-model",
		Rank:      8,
		Epochs:    1,
	})
	if !errors.Is(err, tinker.ErrUnsupported) {
		t.Fatalf("err = %v, want errors.Is tinker.ErrUnsupported (batch reached execution)", err)
	}
}
