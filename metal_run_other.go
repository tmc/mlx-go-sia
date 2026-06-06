//go:build !darwin

package sia

import (
	"context"
	"fmt"
)

// defaultKernelRunner returns a runner that reports Metal is unavailable on
// non-darwin platforms. The executor and benchmarker logic still compile and are
// unit-testable everywhere via an injected fake runner; only the real GPU path is
// darwin-only (MLX custom Metal kernels require Apple silicon).
func defaultKernelRunner() kernelRunner { return unavailableRunner{} }

type unavailableRunner struct{}

func (unavailableRunner) run(_ context.Context, _ RMSNormSpec, _ string, _ launchConfig, _ int) (kernelRun, error) {
	return kernelRun{}, fmt.Errorf("metal custom kernels are only supported on darwin/Apple silicon")
}
