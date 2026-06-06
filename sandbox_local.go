package sia

// SandboxLocal is a [TrainingSandbox] value meaning "run training directly on
// this host" — the totally-local path. The stock sandbox values ([SandboxModal]
// and [SandboxFusion]) both delegate code execution to an external service
// (Modal cloud, or a SandboxFusion daemon); neither runs training on the demo
// machine.
//
// SandboxLocal is the value to pass alongside an [MLXTrainExecutor], which
// treats the gen's train.py as a declarative spec and runs mlx-lm-train on the
// host itself. Because the executor never executes the agent's code in a
// sandbox, the sandbox enum is operationally a no-op for this path: it does not
// change how training runs, only which advisory block the weights meta prompt
// appends.
//
// On the prompt side, [sandboxInstruction] falls through to the Modal block for
// any value it does not recognize, including SandboxLocal. That block is
// inert for the local path (the executor ignores sandbox-run instructions
// entirely), so it is harmless; the local-training story is carried by the
// MLXTrainExecutor and its prompt, not by the sandbox enum. The accompanying
// command (cmd/sia-train) sets up the prompt so the agent is steered to emit a
// hyperparameter block rather than sandbox-run code.
const SandboxLocal TrainingSandbox = "local"

// IsLocalSandbox reports whether s selects on-host training (SandboxLocal). The
// orchestrator does not branch on this — [MLXTrainExecutor] always runs on the
// host — but callers that build a UI or log line can use it to label the run.
func IsLocalSandbox(s TrainingSandbox) bool {
	return s == SandboxLocal
}
