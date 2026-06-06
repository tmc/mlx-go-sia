package sia

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FakeRunner is a scripted [AgentRunner] for tests and examples. Each call to
// Run pops the next [FakeStep] from Steps and applies it: it writes the named
// files into the request's working directory and records the request. It never
// drives a model, so orchestrator behavior can be exercised deterministically.
//
// The zero value runs zero steps (and errors on the first call). Construct with
// [NewFakeRunner].
type FakeRunner struct {
	ImplName string

	mu       sync.Mutex
	steps    []FakeStep
	next     int
	requests []AgentRequest
}

// FakeStep is one scripted agent invocation: a set of files to write into the
// working directory, an optional inspection hook, and an optional error to
// return (simulating an engine failure).
type FakeStep struct {
	// Files maps a path (relative to the working dir, or absolute) to its
	// contents. Parent directories are created as needed.
	Files map[string]string
	// Inspect, if set, is called with the request before files are written.
	Inspect func(req AgentRequest)
	// Err, if set, is returned after Inspect/Files run (the files are still
	// written, mirroring an engine that produced partial output then failed).
	Err error
}

// NewFakeRunner returns a [FakeRunner] that applies steps in order.
func NewFakeRunner(implName string, steps ...FakeStep) *FakeRunner {
	return &FakeRunner{ImplName: implName, steps: steps}
}

// Name reports the agent-impl id (defaults to "fake").
func (r *FakeRunner) Name() string {
	if r.ImplName == "" {
		return "fake"
	}
	return r.ImplName
}

// Run applies the next scripted step.
func (r *FakeRunner) Run(_ context.Context, req AgentRequest) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.requests = append(r.requests, req)
	if r.next >= len(r.steps) {
		return fmt.Errorf("fake runner: no scripted step for call %d", r.next+1)
	}
	step := r.steps[r.next]
	r.next++

	if step.Inspect != nil {
		step.Inspect(req)
	}
	for name, content := range step.Files {
		path := name
		if !filepath.IsAbs(path) {
			path = filepath.Join(req.WorkingDir, name)
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			return err
		}
	}
	return step.Err
}

// Requests returns the requests Run has received so far, in order.
func (r *FakeRunner) Requests() []AgentRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]AgentRequest(nil), r.requests...)
}

// Calls reports how many times Run has been invoked.
func (r *FakeRunner) Calls() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.requests)
}
