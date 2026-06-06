package sia

import (
	"path/filepath"
)

// BundledTasks are the task names the reference ships inside its wheel. The Go
// port keeps the names for parity but resolves tasks from a task directory
// rather than embedded package data; see [ResolveTaskDir].
var BundledTasks = []string{"gpqa", "lawbench", "longcot-chess", "spaceship-titanic"}

// Filename and relative-path literals used by a run or a task. These mirror the
// reference's layout.Names so artifacts are laid out identically.
const (
	// Run / generation artifacts.
	NameTargetAgent       = "target_agent.py"
	NameTrainScript       = "train.py"
	NameAgentExecution    = "agent_execution.json"
	NameAgentExecutionDir = "agent_execution"
	NameExecutionGlob     = "execution_q*.json"
	NameStdoutLog         = "target_agent_stdout.log"
	NameTrainStdoutLog    = "train_stdout.log"
	NameEvalLog           = "evaluation.log"
	NameResultsJSON       = "results.json"
	NameDetailedResults   = "detailed_results.json"
	NameContextMD         = "context.md"
	NameImprovementMD     = "improvement.md"
	NameMetaPrompt        = "meta_agent_prompt.txt"
	NameFeedbackPrompt    = "feedback_agent_prompt.txt"
	NameRequirementsTxt   = "requirements.txt"
	NameRunsRoot          = "./runs"

	// Task inputs.
	NameDataPublic             = "data/public"
	NameTaskMD                 = "data/public/task.md"
	NameEvaluatePy             = "evaluate.py"
	NameSharedSampleExecution  = "sample_agent_execution.json"
	NameReferenceDir           = "reference"
	NameReferenceAgentFile     = "reference_target_agent.py"
	NameSampleTaskDescriptions = NameReferenceDir + "/SAMPLE_TASK_DESCRIPTIONS.md"
	NameReferenceAgent         = NameReferenceDir + "/" + NameReferenceAgentFile
	NameSharedDir              = "_shared"
	NameCompletedMarker        = "COMPLETED"
)

// Focus selects what the self-improvement loop rewrites each generation.
type Focus string

const (
	// FocusHarness rewrites the target agent's scaffold (its code and prompts).
	FocusHarness Focus = "harness"
	// FocusWeights rewrites an RL training script (train.py) that tunes model weights.
	FocusWeights Focus = "weights"
)

// RunLayout builds the paths under a single run directory (e.g. ./runs/run_1).
//
// The zero value is not usable; construct with [NewRunLayout] or
// [RunLayoutForID].
type RunLayout struct {
	RunDir string
}

// NewRunLayout returns the layout rooted at runDir.
func NewRunLayout(runDir string) RunLayout {
	return RunLayout{RunDir: runDir}
}

// RunLayoutForID returns the layout for runs/run_{id} under runsRoot. Pass
// [NameRunsRoot] for the default ./runs root.
func RunLayoutForID(runsRoot string, runID int) RunLayout {
	return RunLayout{RunDir: filepath.Join(runsRoot, runDirName(runID))}
}

func runDirName(runID int) string {
	return "run_" + itoa(runID)
}

func genDirName(n int) string {
	return "gen_" + itoa(n)
}

// GenDir returns the absolute path to generation n's directory.
func (l RunLayout) GenDir(n int) string {
	abs, err := filepath.Abs(filepath.Join(l.RunDir, genDirName(n)))
	if err != nil {
		return filepath.Join(l.RunDir, genDirName(n))
	}
	return abs
}

// GenDirRel returns the relative path to generation n's directory.
func (l RunLayout) GenDirRel(n int) string {
	return filepath.Join(l.RunDir, genDirName(n))
}

// ContextMD returns the path to the run's cross-generation context summary.
func (l RunLayout) ContextMD() string {
	return filepath.Join(l.RunDir, NameContextMD)
}

// TargetAgent returns the path to generation n's target_agent.py.
func (l RunLayout) TargetAgent(n int) string {
	return filepath.Join(l.GenDir(n), NameTargetAgent)
}

// TrainScript returns the path to generation n's train.py (weights focus).
func (l RunLayout) TrainScript(n int) string {
	return filepath.Join(l.GenDir(n), NameTrainScript)
}

// StdoutLog returns the path to generation n's target-agent stdout log. The log
// name depends on focus: weights mode logs to train_stdout.log.
func (l RunLayout) StdoutLog(n int, focus Focus) string {
	name := NameStdoutLog
	if focus == FocusWeights {
		name = NameTrainStdoutLog
	}
	return filepath.Join(l.GenDir(n), name)
}

// ImprovementMD returns the path to generation n's improvement.md.
func (l RunLayout) ImprovementMD(n int) string {
	return filepath.Join(l.GenDir(n), NameImprovementMD)
}

// AgentExecutionDir returns the path to generation n's multi-trajectory folder.
func (l RunLayout) AgentExecutionDir(n int) string {
	return filepath.Join(l.GenDir(n), NameAgentExecutionDir)
}

// AgentExecutionJSON returns the path to generation n's single-trajectory file.
func (l RunLayout) AgentExecutionJSON(n int) string {
	return filepath.Join(l.GenDir(n), NameAgentExecution)
}

// MetaPrompt returns the path to generation n's saved meta-agent prompt.
func (l RunLayout) MetaPrompt(n int) string {
	return filepath.Join(l.GenDir(n), NameMetaPrompt)
}

// FeedbackPrompt returns the path to generation n's saved feedback-agent prompt.
func (l RunLayout) FeedbackPrompt(n int) string {
	return filepath.Join(l.GenDir(n), NameFeedbackPrompt)
}

// ResultsJSON returns the path to generation n's evaluation results.
func (l RunLayout) ResultsJSON(n int) string {
	return filepath.Join(l.GenDir(n), NameResultsJSON)
}

// CompletedMarker returns the path to generation n's COMPLETED marker, written
// by the feedback agent in weights mode to signal early stopping.
func (l RunLayout) CompletedMarker(n int) string {
	return filepath.Join(l.GenDir(n), NameCompletedMarker)
}

// itoa formats a non-negative int without importing strconv at every call site.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
