package sia

import (
	_ "embed"
	"strings"
)

// TrainingSandbox selects where train.py executes code in weights mode.
type TrainingSandbox string

const (
	// SandboxModal runs code in Modal cloud functions (the reference default).
	SandboxModal TrainingSandbox = "modal"
	// SandboxFusion runs code against a local SandboxFusion service.
	SandboxFusion TrainingSandbox = "sandboxfusion"
)

// Prompt templates ported verbatim from the reference's prompts.py. They are
// product-critical text (locked by the reference's golden-master tests), so they
// are embedded rather than reconstructed and only have well-known fields
// substituted.
var (
	//go:embed internal/prompttext/rl_guide.txt
	rlGuide string
	//go:embed internal/prompttext/weights_meta_prompt.txt
	weightsMetaTemplate string
	//go:embed internal/prompttext/weights_feedback_prompt.txt
	weightsFeedbackTemplate string
	//go:embed internal/prompttext/weights_sandbox_modal.txt
	weightsSandboxModal string
	//go:embed internal/prompttext/weights_sandbox_sandboxfusion.txt
	weightsSandboxFusion string
)

// sandboxInstruction returns the trailing sandbox-configuration block for the
// weights meta prompt, matching the reference's branch on training_sandbox.
func sandboxInstruction(s TrainingSandbox) string {
	if s == SandboxFusion {
		return weightsSandboxFusion
	}
	return weightsSandboxModal
}

// buildWeightsMetaPromptSandbox builds the meta-agent prompt for RL-based weight
// tuning (train.py), substituting the task fields into the embedded template.
// The reference threads training_sandbox through to select the trailing
// sandbox-configuration block.
func buildWeightsMetaPromptSandbox(tf TaskFiles, taskModel, workingDir string, sandbox TrainingSandbox) string {
	r := strings.NewReplacer(
		"{{RL_GUIDE}}", rlGuide,
		"{{TASK_MD}}", tf.TaskMD,
		"{{SAMPLE_DESCRIPTIONS}}", tf.SampleTaskDescriptions,
		"{{REFERENCE_AGENT}}", tf.ReferenceTargetAgentPy,
		"{{SAMPLE_EXECUTION}}", jsonIndent(tf.SampleAgentExecution),
		"{{WORKING_DIR}}", workingDir,
		"{{TASK_MODEL}}", taskModel,
		"{{SANDBOX_INSTRUCTION}}", sandboxInstruction(sandbox),
	)
	return r.Replace(weightsMetaTemplate)
}

// buildWeightsFeedbackPrompt builds the feedback-agent prompt for the weights
// focus, substituting fields into the embedded template.
func buildWeightsFeedbackPrompt(opts FeedbackPromptOptions, contextMD string) string {
	r := strings.NewReplacer(
		"{{CURRENT_GEN}}", itoa(opts.CurrentGen),
		"{{PREVIOUS_GENS}}", opts.PreviousGens,
		"{{CONTEXT_MD}}", contextMD,
		"{{AGENT_PY}}", opts.AgentPy,
		"{{TASK}}", opts.Task,
		"{{EXECUTION_STATUS}}", opts.ExecutionStatus,
		"{{EXECUTION_SECTION}}", opts.ExecutionSection,
		"{{NEXT_GEN_DIR}}", opts.NextGenDir,
	)
	return r.Replace(weightsFeedbackTemplate)
}
