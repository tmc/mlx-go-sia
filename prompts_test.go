package sia

import (
	"os"
	"path/filepath"
	"testing"
)

// goldenTaskFiles mirrors the TaskFiles the reference fixtures were generated
// from (see testdata/prompts; produced by the SIA reference's prompt builders).
func goldenTaskFiles() TaskFiles {
	return TaskFiles{
		SampleTaskDescriptions: "SAMPLE DESC LINE 1\nSAMPLE DESC LINE 2",
		ReferenceTargetAgentPy: "# reference agent\nprint('hi')\n",
		SampleAgentExecution:   []byte(`[{"role":"user","content":"hi"}]`),
		TaskMD:                 "# Task\nDo the thing.\n",
	}
}

func readGolden(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "prompts", name))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	return string(b)
}

// firstDiff reports the byte offset and surrounding context of the first
// difference between got and want, for readable failures.
func firstDiff(got, want string) string {
	n := min(len(got), len(want))
	for i := range n {
		if got[i] != want[i] {
			lo := max(0, i-40)
			return "first diff at byte " + itoa(i) + "\n  got:  ..." + clip(got, lo, i+40) +
				"\n  want: ..." + clip(want, lo, i+40)
		}
	}
	if len(got) != len(want) {
		return "lengths differ: got " + itoa(len(got)) + " want " + itoa(len(want)) +
			"; common prefix matches"
	}
	return ""
}

func clip(s string, lo, hi int) string {
	if hi > len(s) {
		hi = len(s)
	}
	if lo < 0 {
		lo = 0
	}
	return s[lo:hi]
}

func TestBuildMetaPromptHarnessNative(t *testing.T) {
	got := BuildMetaPrompt(MetaPromptOptions{
		TaskFiles:  goldenTaskFiles(),
		TaskModel:  "claude-haiku-4-5-20251001",
		WorkingDir: "/work/gen_1",
		Focus:      FocusHarness,
	})
	want := readGolden(t, "meta_harness_native.txt")
	if got != want {
		t.Errorf("meta prompt (harness/native) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildMetaPromptHarnessOpenAI(t *testing.T) {
	prov := Provider{
		ProviderID: "nebius", Name: "Nebius Token Factory", ClientKind: ClientOpenAI,
		BaseURL: "https://api.tokenfactory.us-central1.nebius.com/v1/", APIKeyEnv: "NEBIUS_API_KEY",
	}
	got := BuildMetaPrompt(MetaPromptOptions{
		TaskFiles:  goldenTaskFiles(),
		TaskModel:  "Qwen/Qwen3",
		WorkingDir: "/work/gen_1",
		Provider:   &prov,
		Focus:      FocusHarness,
	})
	want := readGolden(t, "meta_harness_openai.txt")
	if got != want {
		t.Errorf("meta prompt (harness/openai) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildMetaPromptHarnessDirReference(t *testing.T) {
	got := BuildMetaPrompt(MetaPromptOptions{
		TaskFiles:    goldenTaskFiles(),
		TaskModel:    "claude-haiku-4-5-20251001",
		WorkingDir:   "/work/gen_1",
		ReferenceDir: "/work/gen_1",
		Focus:        FocusHarness,
	})
	want := readGolden(t, "meta_harness_dir.txt")
	if got != want {
		t.Errorf("meta prompt (harness/dir) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildFeedbackPromptHarnessNative(t *testing.T) {
	got := BuildFeedbackPrompt(FeedbackPromptOptions{
		CurrentGen:       2,
		MaxGen:           5,
		TaskFiles:        goldenTaskFiles(),
		AgentPy:          "# agent gen2\n",
		Task:             "# Task\nDo the thing.\n",
		ExecutionStatus:  "SUCCESS: ...",
		ExecutionSection: "\nHere is the trajectory\n",
		RunDir:           "/work",
		NextGenDir:       "/work/gen_3",
		PreviousGens:     "1",
		TaskModel:        "claude-haiku-4-5-20251001",
		RequirementsDir:  "/work/gen_3",
		Focus:            FocusHarness,
	})
	want := readGolden(t, "feedback_harness_native.txt")
	if got != want {
		t.Errorf("feedback prompt (harness/native) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildMetaPromptWeightsModal(t *testing.T) {
	got := BuildMetaPrompt(MetaPromptOptions{
		TaskFiles:       goldenTaskFiles(),
		TaskModel:       "Qwen/Qwen3-4B-Instruct-2507",
		WorkingDir:      "/work/gen_1",
		Focus:           FocusWeights,
		TrainingSandbox: SandboxModal,
	})
	want := readGolden(t, "meta_weights_modal.txt")
	if got != want {
		t.Errorf("meta prompt (weights/modal) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildFeedbackPromptWeights(t *testing.T) {
	got := BuildFeedbackPrompt(FeedbackPromptOptions{
		CurrentGen:       2,
		MaxGen:           5,
		TaskFiles:        goldenTaskFiles(),
		AgentPy:          "# train gen2\n",
		Task:             "# Task\nDo the thing.\n",
		ExecutionStatus:  "SUCCESS: ...",
		ExecutionSection: "\ntrajectory\n",
		RunDir:           "/work",
		NextGenDir:       "/work/gen_3",
		PreviousGens:     "1",
		TaskModel:        "Qwen/Qwen3",
		Focus:            FocusWeights,
	})
	want := readGolden(t, "feedback_weights.txt")
	if got != want {
		t.Errorf("feedback prompt (weights) does not match reference golden\n%s", firstDiff(got, want))
	}
}

func TestBuildTargetClientSetup(t *testing.T) {
	prov := Provider{
		ProviderID: "nebius", Name: "Nebius Token Factory", ClientKind: ClientOpenAI,
		BaseURL: "https://api.tokenfactory.us-central1.nebius.com/v1/", APIKeyEnv: "NEBIUS_API_KEY",
	}
	got := BuildTargetClientSetup(prov, "Qwen/Qwen3")
	want := readGolden(t, "client_setup.txt")
	if got != want {
		t.Errorf("client setup block does not match reference golden\n%s", firstDiff(got, want))
	}
}
