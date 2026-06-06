package sia

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestTokenReplacer verifies that the ExecRunner argument tokens substitute from
// the request and provider, including the provider endpoint/key tokens used to
// point an OpenAI-compatible engine CLI at a provider such as Nebius.
func TestTokenReplacer(t *testing.T) {
	t.Setenv("NEBIUS_API_KEY", "secret-key-123")
	req := AgentRequest{
		Model:      "Qwen/Qwen3-Next-80B-A3B-Thinking-fast",
		MaxTurns:   7,
		WorkingDir: "/runs/run_1/gen_1",
		Provider: Provider{
			ProviderID: "nebius",
			ClientKind: ClientOpenAI,
			BaseURL:    "https://api.tokenfactory.us-central1.nebius.com/v1/",
			APIKeyEnv:  "NEBIUS_API_KEY",
		},
	}
	repl := newTokenReplacer(req)

	cases := []struct {
		in, want string
	}{
		{"%MODEL%", "Qwen/Qwen3-Next-80B-A3B-Thinking-fast"},
		{"%MAXTURNS%", "7"},
		{"%WORKDIR%", "/runs/run_1/gen_1"},
		{"%BASEURL%", "https://api.tokenfactory.us-central1.nebius.com/v1/"},
		{"%APIKEY_ENV%", "NEBIUS_API_KEY"},
		{"%APIKEY%", "secret-key-123"},
		{"--base-url=%BASEURL%", "--base-url=https://api.tokenfactory.us-central1.nebius.com/v1/"},
		{"--model=%MODEL%", "--model=Qwen/Qwen3-Next-80B-A3B-Thinking-fast"},
	}
	for _, c := range cases {
		if got := repl.Replace(c.in); got != c.want {
			t.Errorf("replace(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTokenReplacerEmptyAPIKey verifies %APIKEY% is empty when the named env var
// is unset, rather than substituting a stale value.
func TestTokenReplacerEmptyAPIKey(t *testing.T) {
	t.Setenv("SIA_TEST_UNSET_KEY", "")
	os.Unsetenv("SIA_TEST_UNSET_KEY")
	req := AgentRequest{Provider: Provider{APIKeyEnv: "SIA_TEST_UNSET_KEY"}}
	if got := newTokenReplacer(req).Replace("[%APIKEY%]"); got != "[]" {
		t.Errorf("expected empty APIKEY substitution, got %q", got)
	}
}

// TestExecRunnerSubstitutesArgs runs ExecRunner against a small shell script that
// records the args it was invoked with, confirming end-to-end token
// substitution and that the prompt is delivered on stdin.
func TestExecRunnerSubstitutesArgs(t *testing.T) {
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	t.Setenv("NEBIUS_API_KEY", "k-456")
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	promptFile := filepath.Join(dir, "prompt.txt")
	// Script: write "$@" and stdin to files for inspection.
	script := filepath.Join(dir, "engine.sh")
	writeTestFile(t, script, "#!/bin/sh\nprintf '%s\\n' \"$@\" > '"+argsFile+"'\ncat > '"+promptFile+"'\n")
	if err := os.Chmod(script, 0o755); err != nil {
		t.Fatal(err)
	}

	runner := &ExecRunner{
		Command: "/bin/sh",
		Args:    []string{script, "--base-url", "%BASEURL%", "--model", "%MODEL%", "--key", "%APIKEY%"},
	}
	err := runner.Run(context.Background(), AgentRequest{
		Model:      "Qwen/Qwen3",
		Prompt:     "hello prompt",
		WorkingDir: dir,
		Provider: Provider{
			ClientKind: ClientOpenAI,
			BaseURL:    "https://api.tokenfactory.us-central1.nebius.com/v1/",
			APIKeyEnv:  "NEBIUS_API_KEY",
		},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	gotArgs, _ := os.ReadFile(argsFile)
	for _, want := range []string{
		"--base-url",
		"https://api.tokenfactory.us-central1.nebius.com/v1/",
		"--model",
		"Qwen/Qwen3",
		"--key",
		"k-456",
	} {
		if !strings.Contains(string(gotArgs), want) {
			t.Errorf("engine args missing %q\ngot:\n%s", want, gotArgs)
		}
	}
	gotPrompt, _ := os.ReadFile(promptFile)
	if strings.TrimSpace(string(gotPrompt)) != "hello prompt" {
		t.Errorf("prompt on stdin = %q, want %q", gotPrompt, "hello prompt")
	}
}
