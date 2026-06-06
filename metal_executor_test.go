package sia

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fakeRunner is a [kernelRunner] for the cross-platform tests: it returns a
// scripted result instead of touching the GPU, so the executor and benchmarker
// logic is exercised on any host.
type fakeRunner struct {
	out        []float32
	perIter    time.Duration
	compileErr string
	runErr     error
	gotSource  string
	gotCfg     launchConfig
	gotIters   int
	calls      int
}

func (f *fakeRunner) run(_ context.Context, _ RMSNormSpec, source string, cfg launchConfig, iters int) (kernelRun, error) {
	f.calls++
	f.gotSource, f.gotCfg, f.gotIters = source, cfg, iters
	if f.runErr != nil {
		return kernelRun{}, f.runErr
	}
	return kernelRun{Output: f.out, PerIter: f.perIter, CompileErr: f.compileErr}, nil
}

func writeKernel(t *testing.T, dir, source string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, KernelSourceName), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestMetalKernelExecutorRoundtrip(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "out[0] = x[0];")
	if err := os.WriteFile(filepath.Join(dir, KernelConfigName),
		[]byte(`{"grid":[512,1,1],"thread_group":[128,1,1]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	fr := &fakeRunner{out: []float32{1}, perIter: time.Millisecond}
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: fr}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir})
	if err != nil {
		t.Fatalf("RunTarget: %v", err)
	}
	if !res.Success {
		t.Fatalf("Success=false, ErrorMsg=%q", res.ErrorMsg)
	}
	if fr.gotSource != "out[0] = x[0];" {
		t.Errorf("runner saw source %q", fr.gotSource)
	}
	if fr.gotCfg.Grid != [3]int{512, 1, 1} || fr.gotCfg.ThreadGroup != [3]int{128, 1, 1} {
		t.Errorf("runner saw cfg %+v, want grid 512/tg 128", fr.gotCfg)
	}
	if fr.gotIters != 1 {
		t.Errorf("executor should run a single iteration, got %d", fr.gotIters)
	}
}

func TestMetalKernelExecutorCompileFailure(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "@@@ not valid metal ###")

	fr := &fakeRunner{compileErr: "Unable to build metal library from source"}
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: fr}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir})
	if err != nil {
		t.Fatalf("compile failure must NOT be a Go error: %v", err)
	}
	if res.Success {
		t.Fatal("Success=true for a kernel that did not compile")
	}
	if res.ErrorMsg == "" {
		t.Fatal("ErrorMsg empty; the agent gets no compiler feedback")
	}
	if res.ErrorMsg != fr.compileErr {
		t.Errorf("ErrorMsg=%q, want the compiler error %q", res.ErrorMsg, fr.compileErr)
	}
}

func TestMetalKernelExecutorMissingSource(t *testing.T) {
	dir := t.TempDir() // no kernel.metal written
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: &fakeRunner{}}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir})
	if err != nil {
		t.Fatalf("missing source is target-side feedback, not a Go error: %v", err)
	}
	if res.Success || res.ErrorMsg == "" {
		t.Fatalf("missing source must be Success=false with ErrorMsg, got %+v", res)
	}
}

func TestMetalKernelExecutorEmptySource(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "")
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: &fakeRunner{}}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir})
	if err != nil {
		t.Fatalf("empty source: unexpected Go error: %v", err)
	}
	if res.Success {
		t.Fatal("empty source must not be Success=true")
	}
}

func TestMetalKernelExecutorRunnerError(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "out[0] = x[0];")
	fr := &fakeRunner{runErr: errors.New("metal is not available on this host")}
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: fr}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir})
	// A runner that cannot run at all is an executor-side failure: Go error AND
	// Success=false (the orchestrator treats the Go error as non-fatal).
	if err == nil {
		t.Fatal("a runner that cannot run should return a Go error")
	}
	if res.Success {
		t.Fatal("Success must be false when the runner cannot run")
	}
}

func TestMetalKernelExecutorWritesStdoutLog(t *testing.T) {
	dir := t.TempDir()
	writeKernel(t, dir, "out[0] = x[0];")
	logPath := filepath.Join(dir, "stdout.log")
	fr := &fakeRunner{out: []float32{1, 2}, perIter: 2 * time.Millisecond}
	e := &MetalKernelExecutor{Spec: DefaultRMSNormSpec(), runner: fr}

	res, err := e.RunTarget(context.Background(), TargetRequest{WorkingDir: dir, StdoutLog: logPath})
	if err != nil || !res.Success {
		t.Fatalf("RunTarget: err=%v success=%v", err, res.Success)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Errorf("stdout log not written: %v", err)
	}
}

func TestNormalizeLaunchClampsZeros(t *testing.T) {
	spec := DefaultRMSNormSpec()
	cfg := spec.normalizeLaunch(launchConfig{Grid: [3]int{0, 0, 0}, ThreadGroup: [3]int{0, 0, 0}})
	for i := range cfg.Grid {
		if cfg.Grid[i] < 1 || cfg.ThreadGroup[i] < 1 {
			t.Fatalf("zero dimension not clamped: %+v", cfg)
		}
	}
}

func TestDefaultLaunchCoversAllRows(t *testing.T) {
	spec := RMSNormSpec{Rows: 1000, Dim: 16}
	cfg := spec.defaultLaunch()
	if cfg.Grid[0] < spec.Rows {
		t.Errorf("grid %d does not cover %d rows", cfg.Grid[0], spec.Rows)
	}
	if cfg.Grid[0]%cfg.ThreadGroup[0] != 0 {
		t.Errorf("grid %d not a multiple of threadgroup %d", cfg.Grid[0], cfg.ThreadGroup[0])
	}
}
