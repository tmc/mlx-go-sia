package sia

import (
	"math"
	"strings"
	"testing"
)

func TestRMSNormGolden(t *testing.T) {
	// A 2x4 case computed by hand-independent reference.
	spec := RMSNormSpec{Rows: 2, Dim: 4, Eps: 0, RelTol: 1e-6}
	x := []float32{1, 2, 3, 4, 2, 2, 2, 2}
	w := []float32{1, 1, 1, 1}
	got := spec.Golden(x, w)

	// Row 0: mean(1,4,9,16)=7.5, inv=1/sqrt(7.5).
	inv0 := 1.0 / math.Sqrt(7.5)
	want := []float64{1 * inv0, 2 * inv0, 3 * inv0, 4 * inv0, 1, 1, 1, 1} // row1: all 2, mean=4, inv=0.5 → 2*0.5=1
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-5 {
			t.Errorf("Golden[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestCompareGoldenWithinTolerance(t *testing.T) {
	spec := DefaultRMSNormSpec()
	golden := []float32{1.0, 2.0, 3.0}

	// Within tolerance.
	if ok, reason := spec.CompareGolden([]float32{1.0001, 2.0, 3.0}, golden); !ok {
		t.Errorf("tiny drift should pass, got reason %q", reason)
	}
	// Past tolerance.
	if ok, _ := spec.CompareGolden([]float32{1.5, 2.0, 3.0}, golden); ok {
		t.Error("a 50% error should fail the gate")
	}
	// Length mismatch.
	if ok, _ := spec.CompareGolden([]float32{1.0}, golden); ok {
		t.Error("length mismatch should fail")
	}
}

func TestInputsDeterministic(t *testing.T) {
	spec := RMSNormSpec{Rows: 4, Dim: 8, Seed: 7}
	x1, w1 := spec.Inputs()
	x2, w2 := spec.Inputs()
	if len(x1) != spec.Rows*spec.Dim || len(w1) != spec.Dim {
		t.Fatalf("input shapes wrong: x=%d w=%d", len(x1), len(w1))
	}
	for i := range x1 {
		if x1[i] != x2[i] {
			t.Fatalf("inputs not deterministic at x[%d]: %g vs %g", i, x1[i], x2[i])
		}
	}
	for i := range w1 {
		if w1[i] != w2[i] {
			t.Fatalf("weights not deterministic at w[%d]", i)
		}
	}

	// A different seed should produce different inputs.
	other := RMSNormSpec{Rows: 4, Dim: 8, Seed: 8}
	xo, _ := other.Inputs()
	if xo[0] == x1[0] && xo[1] == x1[1] {
		t.Error("different seed produced identical leading inputs")
	}
}

func TestScriptedKernelStages(t *testing.T) {
	stages := ScriptedKernelStages()
	if len(stages) < 2 {
		t.Fatalf("want at least 2 stages, got %d", len(stages))
	}
	if strings.TrimSpace(stages[0]) != strings.TrimSpace(SeedKernelSource) {
		t.Error("stage 0 must be the seed kernel")
	}
	seen := map[string]bool{}
	for i, s := range stages {
		key := strings.TrimSpace(s)
		if key == "" {
			t.Errorf("stage %d is empty", i)
		}
		if seen[key] {
			t.Errorf("stage %d duplicates an earlier stage (the improver would stall)", i)
		}
		seen[key] = true
	}
}
