package sia

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Orchestrator runs the SIA self-improvement loop: it seeds an initial target
// agent with the meta agent, then for each generation runs the target agent,
// evaluates it, and (except on the last generation) runs the feedback agent to
// produce the next generation's agent. Construct with [NewOrchestrator].
type Orchestrator struct {
	// Meta runs the meta and feedback agents (the engine). Required.
	Meta AgentRunner
	// Target runs the generated target agent each generation. Required.
	Target TargetExecutor
	// Eval scores each generation; nil uses [NopEvaluator].
	Eval Evaluator
	// Now returns the current time for timestamps; nil uses time.Now.
	Now func() time.Time
	// Logf logs progress; nil discards.
	Logf func(format string, args ...any)
	// Config supplies context-manager tunables (preview/insight limits); the
	// zero value uses DefaultConfig.
	Config Config
	// Summarizer optionally produces the per-generation evolution summary the
	// reference generates with an LLM call; nil omits the block.
	Summarizer ContextSummarizer
}

// NewOrchestrator returns an orchestrator with the given engine and target
// executor and defaults for the optional fields.
func NewOrchestrator(meta AgentRunner, target TargetExecutor) *Orchestrator {
	return &Orchestrator{Meta: meta, Target: target}
}

// RunOptions configures a full [Orchestrator.Run].
type RunOptions struct {
	Layout          RunLayout              // run directory layout (required)
	Task            TaskLayout             // resolved task (required)
	TaskFiles       TaskFiles              // loaded task reference files (required)
	MetaProfile     MetaAgentProfile       // engine model + provider (required)
	Target          TargetAgentProfile     // target model + provider + reference (required)
	Resolved        ResolvedAgentReference // resolved seed + deps (required)
	MaxGen          int                    // number of generations (>= 1)
	MaxTurns        int                    // engine turn budget (0 = profile default later)
	Focus           Focus                  // FocusHarness (default) or FocusWeights
	TrainingSandbox TrainingSandbox        // weights focus only
	MaxLogSize      int64                  // per-trajectory-file cap (0 = no cap)
	RunConfig       map[string]string      // metadata surfaced in context.md
}

// Run executes the full loop. It returns a [RunResult] summarizing each
// generation. A returned error indicates the loop could not proceed (e.g. the
// meta agent engine failed to run); a target agent that exits non-zero does not
// fail the run — the feedback agent still gets a chance to repair it, matching
// the reference.
func (o *Orchestrator) Run(ctx context.Context, opts RunOptions) (RunResult, error) {
	if opts.MaxGen < 1 {
		return RunResult{}, fmt.Errorf("max_gen must be >= 1, got %d", opts.MaxGen)
	}
	if o.Meta == nil {
		return RunResult{}, fmt.Errorf("orchestrator: Meta runner is required")
	}
	if o.Target == nil {
		return RunResult{}, fmt.Errorf("orchestrator: Target executor is required")
	}
	if o.Eval == nil {
		o.Eval = NopEvaluator{}
	}
	cfg := o.Config
	if cfg.InsightPreviewLimit == 0 {
		cfg = DefaultConfig()
	}

	// Stamp the run start with the orchestrator clock so context.md is
	// reproducible under a fixed clock; do not clobber a caller-set value.
	runConfig := opts.RunConfig
	if _, ok := runConfig["started"]; !ok {
		merged := make(map[string]string, len(runConfig)+1)
		maps.Copy(merged, runConfig)
		merged["started"] = o.now().Format("2006-01-02 15:04:05")
		runConfig = merged
	}

	cm := NewContextManager(opts.Layout, runConfig).WithConfig(cfg).WithSummarizer(o.Summarizer)
	if err := cm.Initialize(); err != nil {
		return RunResult{}, fmt.Errorf("initialize context: %w", err)
	}

	// Section: seed the initial target agent with the meta agent.
	if err := o.runMeta(ctx, opts); err != nil {
		return RunResult{}, err
	}

	result := RunResult{Focus: opts.Focus}
	for gen := 1; gen <= opts.MaxGen; gen++ {
		o.logf("==== generation %d of %d ====", gen, opts.MaxGen)
		genResult, err := o.runGeneration(ctx, opts, cm, gen)
		if err != nil {
			return result, err
		}
		result.Generations = append(result.Generations, genResult)

		// Weights mode early-stop: the feedback agent may signal completion.
		if opts.Focus == FocusWeights && gen < opts.MaxGen {
			if isFile(opts.Layout.CompletedMarker(gen + 1)) {
				o.logf("feedback agent signaled completion; stopping early")
				result.StoppedEarly = true
				break
			}
		}
	}

	if err := cm.Finalize(); err != nil {
		return result, fmt.Errorf("finalize context: %w", err)
	}
	result.ContextPath = opts.Layout.ContextMD()
	return result, nil
}

// runMeta builds the meta prompt, copies the reference into gen 1's working dir,
// saves the prompt for transparency, and runs the meta agent to create the
// initial target agent.
func (o *Orchestrator) runMeta(ctx context.Context, opts RunOptions) error {
	workDir := opts.Layout.GenDir(1)
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return fmt.Errorf("create gen 1 dir: %w", err)
	}
	if err := opts.Resolved.CopyInto(workDir); err != nil {
		return fmt.Errorf("copy reference into gen 1: %w", err)
	}

	referenceDir := ""
	if opts.Resolved.RefDir != "" {
		referenceDir = workDir
	}
	provider := opts.Target.Provider
	prompt := BuildMetaPrompt(MetaPromptOptions{
		TaskFiles:       opts.TaskFiles,
		TaskModel:       opts.Target.Model,
		WorkingDir:      workDir,
		Provider:        &provider,
		ReferenceDir:    referenceDir,
		Focus:           opts.Focus,
		TrainingSandbox: opts.TrainingSandbox,
	})
	if err := os.WriteFile(filepath.Join(workDir, NameMetaPrompt), []byte(prompt), 0o644); err != nil {
		return fmt.Errorf("save meta prompt: %w", err)
	}
	o.logf("running meta agent (model=%s)", opts.MetaProfile.Model)
	if err := o.Meta.Run(ctx, AgentRequest{
		Model:      opts.MetaProfile.Model,
		Prompt:     prompt,
		WorkingDir: workDir,
		MaxTurns:   opts.MaxTurns,
		Provider:   opts.MetaProfile.Provider,
	}); err != nil {
		return fmt.Errorf("meta agent: %w", err)
	}
	return nil
}

// runGeneration runs one generation: target agent, evaluation, context record,
// and (unless it's the last generation) the feedback agent.
func (o *Orchestrator) runGeneration(ctx context.Context, opts RunOptions, cm *ContextManager, gen int) (GenResult, error) {
	genDir := opts.Layout.GenDir(gen)
	agentPath := opts.Layout.TargetAgent(gen)
	if opts.Focus == FocusWeights {
		agentPath = opts.Layout.TrainScript(gen)
	}
	stdoutLog := opts.Layout.StdoutLog(gen, opts.Focus)

	start := o.now()
	tr, err := o.Target.RunTarget(ctx, TargetRequest{
		AgentPath:  agentPath,
		DatasetDir: opts.Task.AbsDatasetDir(),
		WorkingDir: genDir,
		StdoutLog:  stdoutLog,
	})
	duration := o.now().Sub(start).Seconds()
	// A target executor error (cannot run at all) is non-fatal here: the
	// reference continues to the feedback agent so it can repair the agent.
	if err != nil {
		o.logf("target agent could not run: %v", err)
	}

	o.logf("evaluating generation %d", gen)
	evalResult, evalErr := o.Eval.Evaluate(ctx, genDir)
	if evalErr != nil {
		return GenResult{}, fmt.Errorf("evaluate gen %d: %w", gen, evalErr)
	}

	improvementPath := opts.Layout.ImprovementMD(gen)
	execType := "Single"
	if isDir(opts.Layout.AgentExecutionDir(gen)) {
		execType = "Multi-trajectory"
	}
	rec := GenerationRecord{
		GenNum:        gen,
		Success:       tr.Success,
		Timestamp:     o.now().Format("2006-01-02 15:04:05"),
		Duration:      duration,
		AgentPath:     agentPath,
		GenDir:        genDir,
		ExecutionType: execType,
	}
	if isFile(improvementPath) {
		rec.ImprovementPath = improvementPath
	}
	if err := cm.AddGeneration(rec); err != nil {
		return GenResult{}, fmt.Errorf("record gen %d: %w", gen, err)
	}

	gres := GenResult{
		Gen:       gen,
		Target:    tr,
		Eval:      evalResult,
		DurationS: duration,
		MultiTraj: execType == "Multi-trajectory",
	}

	// Run the feedback agent for all but the final generation.
	if gen < opts.MaxGen {
		if err := o.runFeedback(ctx, opts, gen, tr, evalResult); err != nil {
			return gres, err
		}
		gres.FeedbackRan = true
	}
	return gres, nil
}

// runFeedback builds the feedback prompt from the generation's execution +
// evaluation, copies the reference forward, saves the prompt, and runs the
// feedback agent to write the next generation's agent.
func (o *Orchestrator) runFeedback(ctx context.Context, opts RunOptions, gen int, tr TargetResult, ev EvalResult) error {
	genDir := opts.Layout.GenDir(gen)
	nextGenDir := opts.Layout.GenDir(gen + 1)
	if err := os.MkdirAll(nextGenDir, 0o755); err != nil {
		return fmt.Errorf("create gen %d dir: %w", gen+1, err)
	}
	// Carry the reference forward so the improved agent can import helpers and
	// declared deps install.
	if err := opts.Resolved.CopyInto(nextGenDir); err != nil {
		return fmt.Errorf("copy reference into gen %d: %w", gen+1, err)
	}

	agentPath := opts.Layout.TargetAgent(gen)
	if opts.Focus == FocusWeights {
		agentPath = opts.Layout.TrainScript(gen)
	}
	agentPy, err := os.ReadFile(agentPath)
	if err != nil {
		// The target agent file is missing (meta/feedback agent failed to write
		// it); feed an empty body so the feedback agent can still repair.
		agentPy = nil
	}

	exec := LoadExecution(genDir, opts.MaxLogSize)
	status, section := o.feedbackContext(opts, gen, tr, ev, exec)

	requirementsDir := ""
	if opts.Resolved.Requirements != "" {
		requirementsDir = nextGenDir
	}
	provider := opts.Target.Provider
	prompt := BuildFeedbackPrompt(FeedbackPromptOptions{
		CurrentGen:       gen,
		MaxGen:           opts.MaxGen,
		TaskFiles:        opts.TaskFiles,
		AgentPy:          string(agentPy),
		Task:             opts.TaskFiles.TaskMD,
		ExecutionStatus:  status,
		ExecutionSection: section,
		RunDir:           opts.Layout.RunDir,
		NextGenDir:       nextGenDir,
		PreviousGens:     previousGens(gen),
		TaskModel:        opts.Target.Model,
		Provider:         &provider,
		RequirementsDir:  requirementsDir,
		Focus:            opts.Focus,
	})
	if err := os.WriteFile(filepath.Join(nextGenDir, NameFeedbackPrompt), []byte(prompt), 0o644); err != nil {
		return fmt.Errorf("save feedback prompt: %w", err)
	}

	o.logf("running feedback agent for generation %d", gen)
	if err := o.Meta.Run(ctx, AgentRequest{
		Model:      opts.MetaProfile.Model,
		Prompt:     prompt,
		WorkingDir: nextGenDir,
		MaxTurns:   opts.MaxTurns,
		Provider:   opts.MetaProfile.Provider,
	}); err != nil {
		return fmt.Errorf("feedback agent: %w", err)
	}
	return nil
}

func (o *Orchestrator) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}

func (o *Orchestrator) logf(format string, args ...any) {
	if o.Logf != nil {
		o.Logf(format, args...)
	}
}

// previousGens renders the previous-generation list for the feedback prompt:
// "None" before gen 2, else "1", "1, 2", etc.
func previousGens(currentGen int) string {
	if currentGen <= 1 {
		return "None"
	}
	parts := make([]string, 0, currentGen-1)
	for i := 1; i < currentGen; i++ {
		parts = append(parts, itoa(i))
	}
	return strings.Join(parts, ", ")
}

// RunResult summarizes a full orchestrator run.
type RunResult struct {
	Focus        Focus
	Generations  []GenResult
	StoppedEarly bool
	ContextPath  string
}

// GenResult summarizes one generation.
type GenResult struct {
	Gen         int
	Target      TargetResult
	Eval        EvalResult
	DurationS   float64
	MultiTraj   bool
	FeedbackRan bool
}
