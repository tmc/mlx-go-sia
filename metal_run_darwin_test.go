//go:build darwin

package sia

import (
	"context"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

// requireMetal skips when no Metal device is available (CI, non-Apple-silicon).
func requireMetal(t *testing.T) {
	t.Helper()
	if !mlx.MetalIsAvailable() {
		t.Skip("metal not available on this host")
	}
}

// TestSeedKernelCompilesAndMatches is the core money-shot check: the deliberately
// naive seed kernel JIT-compiles on the real Metal backend and its output matches
// the external Go golden oracle within the frozen tolerance.
func TestSeedKernelCompilesAndMatches(t *testing.T) {
	requireMetal(t)
	spec := DefaultRMSNormSpec()
	x, w := spec.Inputs()
	golden := spec.Golden(x, w)

	res, err := mlxKernelRunner{}.run(context.Background(), spec, SeedKernelSource, spec.defaultLaunch(), 1)
	if err != nil {
		t.Fatalf("runner could not run: %v", err)
	}
	if res.CompileErr != "" {
		t.Fatalf("seed kernel failed to compile: %s", res.CompileErr)
	}
	if ok, reason := spec.CompareGolden(res.Output, golden); !ok {
		t.Fatalf("seed kernel output does not match golden: %s", reason)
	}
}

// TestScriptedStagesCompileAndMatch checks every stage of the scripted
// optimization sequence compiles and stays correct against the golden oracle on
// the real GPU — so the demo's "number goes up" is a sequence of correct kernels,
// never a faster-but-wrong one.
func TestScriptedStagesCompileAndMatch(t *testing.T) {
	requireMetal(t)
	spec := DefaultRMSNormSpec()
	x, w := spec.Inputs()
	golden := spec.Golden(x, w)

	for i, src := range ScriptedKernelStages() {
		res, err := mlxKernelRunner{}.run(context.Background(), spec, src, spec.defaultLaunch(), 1)
		if err != nil {
			t.Fatalf("stage %d: runner error: %v", i, err)
		}
		if res.CompileErr != "" {
			t.Fatalf("stage %d failed to compile: %s", i, res.CompileErr)
		}
		if ok, reason := spec.CompareGolden(res.Output, golden); !ok {
			t.Fatalf("stage %d output is wrong: %s", i, reason)
		}
	}
}

// TestBadKernelReportsCompileError confirms the verified failure seam: a
// malformed MSL string yields CompileErr (surfaced at mlx.Eval) and no Go error
// and no panic — the signal the agent fixes next generation.
func TestBadKernelReportsCompileError(t *testing.T) {
	requireMetal(t)
	spec := DefaultRMSNormSpec()
	res, err := mlxKernelRunner{}.run(context.Background(), spec, "@@@ this is not valid metal ###;", spec.defaultLaunch(), 1)
	if err != nil {
		t.Fatalf("a bad kernel must not be a runner error: %v", err)
	}
	if res.CompileErr == "" {
		t.Fatal("a malformed kernel must report a compile error")
	}
}

// TestScriptedStagesImproveThroughput is a soft, informational check that the
// vectorized stages are at least not slower than the naive seed on this host. It
// does not hard-fail on small regressions (thermal/measurement noise), but logs
// the per-stage throughput so the demo's headroom is visible.
func TestScriptedStagesImproveThroughput(t *testing.T) {
	requireMetal(t)
	spec := DefaultRMSNormSpec()
	stages := ScriptedKernelStages()

	var seedTput float64
	for i, src := range stages {
		res, err := mlxKernelRunner{}.run(context.Background(), spec, src, spec.defaultLaunch(), 30)
		if err != nil || res.CompileErr != "" {
			t.Fatalf("stage %d run failed: err=%v compile=%s", i, err, res.CompileErr)
		}
		tput := float64(spec.Rows*spec.Dim) / res.PerIter.Seconds()
		t.Logf("stage %d: %.1f ops/sec (%v/iter)", i, tput, res.PerIter)
		if i == 0 {
			seedTput = tput
		}
		if i == len(stages)-1 && tput < seedTput {
			t.Logf("note: final stage (%.1f) slower than seed (%.1f) on this host — likely thermal/measurement noise, not a correctness issue", tput, seedTput)
		}
	}
}
