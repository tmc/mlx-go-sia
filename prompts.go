package sia

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// TaskFiles bundles the task reference files the prompts are built from.
type TaskFiles struct {
	SampleTaskDescriptions string
	ReferenceTargetAgentPy string // the inline seed (empty for a directory reference)
	SampleAgentExecution   json.RawMessage
	TaskMD                 string
}

// referenceSection returns the reference paragraph of the meta prompt. With an
// empty referenceDir the seed code is embedded verbatim (default/single-file);
// otherwise the agent is pointed at the on-disk multi-file reference.
func referenceSection(tf TaskFiles, referenceDir string) string {
	if referenceDir == "" {
		return "Here is a sample target_agent.py showing the complete implementation pattern " +
			"(READ THE ENTIRE FILE):\n" + tf.ReferenceTargetAgentPy
	}
	return fmt.Sprintf(
		"Your reference agent implementation has been placed in your working directory (%s). "+
			"It may span multiple files. READ IT YOURSELF with your file tools (Read/Glob/Grep) — study the "+
			"entrypoint and any helper modules — then write your target_agent.py in the same directory.\n"+
			"If your target_agent.py needs third-party packages, list them (one per line) in a requirements.txt "+
			"in your working directory; they are installed before the target agent runs.",
		referenceDir)
}

// MetaPromptOptions configures [BuildMetaPrompt].
type MetaPromptOptions struct {
	TaskFiles       TaskFiles
	TaskModel       string
	WorkingDir      string
	Provider        *Provider       // nil for the default/native case
	ReferenceDir    string          // non-empty only for a multi-file directory reference
	Focus           Focus           // FocusHarness (default) or FocusWeights
	TrainingSandbox TrainingSandbox // weights focus only; empty defaults to SandboxModal
}

// BuildMetaPrompt builds the meta-agent prompt for creating the initial target
// agent. For Anthropic/Google providers (and a nil provider) the text matches
// the reference verbatim; for an OpenAI-compatible provider a client-setup block
// is prepended. The weights focus delegates to [buildWeightsMetaPrompt].
func BuildMetaPrompt(opts MetaPromptOptions) string {
	if opts.Focus == FocusWeights {
		sandbox := opts.TrainingSandbox
		if sandbox == "" {
			sandbox = SandboxModal
		}
		return buildWeightsMetaPromptSandbox(opts.TaskFiles, opts.TaskModel, opts.WorkingDir, sandbox)
	}

	sample := jsonIndent(opts.TaskFiles.SampleAgentExecution)
	base := fmt.Sprintf(`You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here is the FULL TASK SPECIFICATION that your target_agent.py will need to solve:
%s

Here are a couple of sample task descriptions which the target agent has to solve:
%s

%s

Here is a sample agent execution trajectory:
%s

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is %s. Create the target_agent.py in the current working directory itself.

2. The target_agent.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The target_agent.py must INCLUDE these paths in the prompt it sends to %s. %s MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let %s search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The target agent can ONLY read from the dataset directory provided via --dataset_dir, and can ONLY write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.

5. EXECUTION LOGGING - CRITICAL:

   The target_agent.py must log its execution trajectory properly. The format depends on the task type:

   **FOR TASKS WITH MULTIPLE INDEPENDENT SAMPLES** (e.g., GPQA with 198 questions, multiple test cases):
   - Create a folder: agent_execution/ in the working directory
   - Save each sample separately: execution_q0.json, execution_q1.json, execution_q2.json, etc.
   - Each file contains the complete trajectory for that ONE sample only
   - Files must be named sequentially: execution_q0.json, execution_q1.json, ...

   **FOR TASKS WITH SINGLE EXECUTION** (e.g., building one ML model, analyzing one dataset):
   - Save to a single file: agent_execution.json in the working directory
   - File contains the complete execution trajectory

   **HOW TO DETERMINE WHICH FORMAT**:
   - Read the task description carefully
   - If it mentions "independent items", "dataset with multiple records to process separately"
     → Use multi-trajectory (folder with multiple files)
   - If it's about "build a model", "analyze the dataset", "create one solution", "optimize one system"
     → Use single-trajectory (one JSON file)

   **FORMAT REQUIREMENTS** (both formats):
   - Use the same format as the sample agent execution trajectory provided above
   - Include all messages, tool calls, and their results
   - Ensure valid JSON (properly close all arrays/objects)
   - Make sure to properly close the JSON file(s) to avoid corruption

6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The target_agent.py should use only the "%s" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the target_agent.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to %s in the prompt.

Example invocation (paths will vary at runtime):
    python target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
`,
		opts.TaskFiles.TaskMD,
		opts.TaskFiles.SampleTaskDescriptions,
		referenceSection(opts.TaskFiles, opts.ReferenceDir),
		sample,
		opts.WorkingDir,
		opts.TaskModel, opts.TaskModel,
		opts.TaskModel,
		opts.TaskModel,
		opts.TaskModel,
	)

	if opts.Provider == nil || opts.Provider.ClientKind != ClientOpenAI {
		return base
	}
	return BuildTargetClientSetup(*opts.Provider, opts.TaskModel) + base
}

// BuildTargetClientSetup returns the prompt block telling the meta-agent how to
// reach an OpenAI-compatible target model. It is prepended to the meta/feedback
// prompt for OpenAI-compatible providers.
func BuildTargetClientSetup(provider Provider, taskModel string) string {
	return fmt.Sprintf(`=== TARGET MODEL CLIENT SETUP (OpenAI-compatible provider: %s) ===

The target model "%s" is served by an OpenAI-compatible API. The reference
target_agent.py shown below may use a different SDK (e.g. the Gemini SDK) — you MUST
refactor your target_agent.py to use the `+"`openai`"+` SDK configured for this provider
(do NOT use the anthropic or google SDK):

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="%s",
        api_key=os.environ["%s"],
    )

Call client.chat.completions.create(model="%s", ...) using OpenAI-style
messages (and OpenAI function calling / response_format where the reference uses
structured output). Do NOT compute a dollar cost: per-provider pricing is unknown, so
set any cost field to 0 (token counts from the API response are still fine to record).

`,
		provider.Name, taskModel, provider.BaseURL, provider.APIKeyEnv, taskModel)
}

// FeedbackPromptOptions configures [BuildFeedbackPrompt].
type FeedbackPromptOptions struct {
	CurrentGen       int
	MaxGen           int
	TaskFiles        TaskFiles
	AgentPy          string // current generation's agent source (target_agent.py or train.py)
	Task             string // the task.md text
	ExecutionStatus  string
	ExecutionSection string
	RunDir           string
	NextGenDir       string
	PreviousGens     string // e.g. "None" or "1, 2"
	TaskModel        string
	Provider         *Provider
	RequirementsDir  string // non-empty when the reference declares dependencies
	Focus            Focus
}

// BuildFeedbackPrompt builds the feedback-agent prompt for improving the target
// agent (harness focus) or train.py (weights focus). The text matches the
// reference verbatim for Anthropic/Google/native providers; an OpenAI-compatible
// provider prepends the client-setup block.
func BuildFeedbackPrompt(opts FeedbackPromptOptions) string {
	contextMD := joinPath(opts.RunDir, NameContextMD)

	if opts.Focus == FocusWeights {
		return buildWeightsFeedbackPrompt(opts, contextMD)
	}

	base := fmt.Sprintf(`You are an expert AI Engineer analyzing agent scaffolds for iterative improvement.

**GENERATION CONTEXT**:
- Current generation: %d
- Previous generations: %s
- Evolution history: %s

**BEFORE ANALYZING - READ THE FULL HISTORY**:
1. Read %s to understand:
   - What improvements were tried in each previous generation
   - Performance trends across generations
   - What worked and what didn't work
2. Review previous improvement.md files from earlier generations if helpful
3. Don't repeat failed approaches from earlier generations
4. Build upon successful patterns that improved performance

---

**SAMPLE TASK DESCRIPTIONS**:
`+"```"+`
%s
`+"```"+`

**CURRENT TARGET AGENT** (Generation %d):
`+"```python"+`
%s
`+"```"+`

**TASK WORKED ON**:
`+"```"+`
%s
`+"```"+`

**EXECUTION STATUS**:
`+"```"+`
%s
`+"```"+`

**EXECUTION LOGS**:
%s

---

**YOUR TASK**:

You must create exactly TWO files in %s/:
1. improvement.md - Analysis and improvement plan
2. target_agent.py - The improved agent implementation

Follow these steps:

**STEP 1: Analyze the execution**:
   - For multi-trajectory: Look for patterns across all trajectories
   - For single-trajectory: Analyze the full execution flow
   - Identify what worked well and what failed
   - Check for consistency and robustness

**STEP 2: Review evolution history**:
   - Read context.md to see the full evolution
   - Understand what was tried in previous generations
   - Build upon successful patterns
   - Avoid repeating failed approaches

**STEP 3: Write improvement.md**:
   - MUST save to: %s/improvement.md
   - Document your analysis and planned improvements
   - Focus on structural improvements to the agent scaffold
   - Make the agent more robust and generalizable
   - Don't optimize for this specific task
   - Reference insights from previous generations if applicable

**STEP 4: Create improved target_agent.py**:
   - MUST save to: %s/target_agent.py
   - Implement the improvements documented in improvement.md
   - Apply all the planned improvements from step 3
   - Do not create or modify any other files besides these two

**RULES**:
- Focus on agent structure, not task-specific optimizations
- Make the agent work well across diverse task types (see sample task descriptions)
- If execution failed, fix the root cause
- If multi-trajectory: ensure each trajectory is properly isolated and logged
- Consider error handling, logging mechanisms, and robustness
- Build upon successful patterns from previous generations (check context.md)
- If execution log shows errors or is incomplete, suggest improvements to ensure proper logging

NOTE: The agent execution log may be incomplete or contain errors if the target agent crashed. If you see an "error" field, focus on making the agent more robust to prevent such failures.
`,
		opts.CurrentGen,
		opts.PreviousGens,
		contextMD,
		contextMD,
		opts.TaskFiles.SampleTaskDescriptions,
		opts.CurrentGen,
		opts.AgentPy,
		opts.Task,
		opts.ExecutionStatus,
		opts.ExecutionSection,
		opts.NextGenDir,
		opts.NextGenDir,
		opts.NextGenDir,
	)

	if opts.RequirementsDir != "" {
		base += fmt.Sprintf("\nNOTE ON DEPENDENCIES: You may also create or edit a requirements.txt in %s "+
			"(one package per line) to declare third-party packages your target_agent.py needs; they are "+
			"installed before the target agent runs. This is the one exception to the \"only two files\" rule above.\n",
			opts.RequirementsDir)
	}
	if opts.Provider == nil || opts.Provider.ClientKind != ClientOpenAI {
		return base
	}
	return BuildTargetClientSetup(*opts.Provider, opts.TaskModel) + base
}

// jsonIndent re-indents raw with two-space indentation, matching Python's
// json.dumps(indent=2). It indents the original bytes (via [json.Indent]) rather
// than round-tripping through a map, so object key order is preserved — the
// reference relies on insertion order in the sample-execution block. On a
// non-JSON value it returns the raw bytes unchanged.
func jsonIndent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "null"
	}
	// Compact first so any incoming whitespace is normalized, then indent.
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		return string(raw)
	}
	var out bytes.Buffer
	if err := json.Indent(&out, compact.Bytes(), "", "  "); err != nil {
		return string(raw)
	}
	return out.String()
}

// joinPath joins with a forward slash like the reference's os.path.join on the
// run/gen paths used in prompts, keeping prompt text stable across platforms.
func joinPath(dir, name string) string {
	if dir == "" {
		return name
	}
	return strings.TrimRight(dir, "/") + "/" + name
}
