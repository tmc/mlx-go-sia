// Package sia ports the SIA self-improving-AI orchestrator to Go.
//
// SIA (Hebbar et al., 2026, arXiv:2605.27276) runs a three-agent loop over
// successive generations:
//
//   - a Meta-Agent seeds an initial target agent from a task specification,
//   - a Target Agent runs the task and records an execution trajectory,
//   - a Feedback-Agent reads the trajectory and evaluation results and rewrites
//     the target agent for the next generation.
//
// The orchestrator coordinates the loop, lays out per-run and per-generation
// artifacts under runs/run_{id}/gen_{n}/, and accumulates a cross-generation
// context summary the Feedback-Agent reads before each rewrite.
//
// This port is faithful to the reference's load-bearing contracts:
//
//   - the orchestrator-to-target CLI contract (--dataset_dir, --working_dir),
//   - the filesystem [Layout] (path and filename constants),
//   - the execution-trajectory format and the single- vs multi-trajectory
//     detection rule (see [LoadExecution]),
//   - the meta and feedback [BuildMetaPrompt]/[BuildFeedbackPrompt] text,
//   - the JSON [Profile]/[Provider] configuration model.
//
// The meta/feedback agent engine is abstracted behind the [AgentRunner]
// interface — the same seam the reference exposes as "agent impls". The
// reference's Python ecosystem (venv, pip, the Claude Agent SDK, tinker) is not
// copied: [ExecRunner] shells out to an external agent CLI, and tests use a
// scripted [FakeRunner]. The port does not depend on a Python runtime.
package sia
