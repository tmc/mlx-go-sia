package sia

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestRunLayoutPaths(t *testing.T) {
	l := RunLayoutForID("./runs", 1)
	if got, want := l.RunDir, filepath.Join("runs", "run_1"); got != want {
		t.Errorf("RunDir = %q, want %q", got, want)
	}
	// GenDir is absolute and ends with the relative gen path.
	gen := l.GenDir(2)
	if !filepath.IsAbs(gen) {
		t.Errorf("GenDir not absolute: %q", gen)
	}
	if !strings.HasSuffix(gen, filepath.Join("runs", "run_1", "gen_2")) {
		t.Errorf("GenDir = %q, want suffix runs/run_1/gen_2", gen)
	}

	tests := []struct {
		name string
		got  string
		want string
	}{
		{"TargetAgent", filepath.Base(l.TargetAgent(1)), NameTargetAgent},
		{"TrainScript", filepath.Base(l.TrainScript(1)), NameTrainScript},
		{"ImprovementMD", filepath.Base(l.ImprovementMD(1)), NameImprovementMD},
		{"MetaPrompt", filepath.Base(l.MetaPrompt(1)), NameMetaPrompt},
		{"FeedbackPrompt", filepath.Base(l.FeedbackPrompt(1)), NameFeedbackPrompt},
		{"ResultsJSON", filepath.Base(l.ResultsJSON(1)), NameResultsJSON},
		{"AgentExecutionDir", filepath.Base(l.AgentExecutionDir(1)), NameAgentExecutionDir},
		{"ContextMD", filepath.Base(l.ContextMD()), NameContextMD},
		{"CompletedMarker", filepath.Base(l.CompletedMarker(1)), NameCompletedMarker},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s base = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

func TestStdoutLogFocus(t *testing.T) {
	l := RunLayoutForID("./runs", 1)
	if got := filepath.Base(l.StdoutLog(1, FocusHarness)); got != NameStdoutLog {
		t.Errorf("harness stdout log = %q, want %q", got, NameStdoutLog)
	}
	if got := filepath.Base(l.StdoutLog(1, FocusWeights)); got != NameTrainStdoutLog {
		t.Errorf("weights stdout log = %q, want %q", got, NameTrainStdoutLog)
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Setenv("SIA_MAX_GENERATIONS", "7")
	t.Setenv("SIA_TASK_MODEL", "my-model")
	t.Setenv("SIA_MAX_TURNS", "not-a-number") // ignored, keeps default
	cfg := ConfigFromEnv()
	if cfg.DefaultMaxGenerations != 7 {
		t.Errorf("DefaultMaxGenerations = %d, want 7", cfg.DefaultMaxGenerations)
	}
	if cfg.DefaultTaskModel != "my-model" {
		t.Errorf("DefaultTaskModel = %q, want my-model", cfg.DefaultTaskModel)
	}
	if cfg.DefaultMaxTurns != DefaultConfig().DefaultMaxTurns {
		t.Errorf("DefaultMaxTurns = %d, want default %d (unparseable env ignored)", cfg.DefaultMaxTurns, DefaultConfig().DefaultMaxTurns)
	}
}

func TestParseProvider(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr string
	}{
		{"valid", `{"provider_id":"x","name":"X","client_kind":"openai","api_key_env":"X_KEY"}`, ""},
		{"missing keys", `{"provider_id":"x"}`, "missing required keys"},
		{"bad client_kind", `{"provider_id":"x","name":"X","client_kind":"bogus","api_key_env":"K"}`, "invalid client_kind"},
		{"bad json", `{`, "invalid provider json"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseProvider([]byte(tt.json), "test")
			switch {
			case tt.wantErr == "" && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
				t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestParseMetaProfileClaudeRequiresAnthropic(t *testing.T) {
	// claude impl + non-anthropic provider must be rejected.
	j := `{"profile_id":"p","name":"P","agent_impl":"claude","model":"m","provider_id":"nebius"}`
	_, err := ParseMetaAgentProfile([]byte(j), "test", DefaultProviderLoader, nil)
	if err == nil || !strings.Contains(err.Error(), "requires an anthropic provider") {
		t.Fatalf("error = %v, want claude/anthropic mismatch", err)
	}

	// claude impl + anthropic provider is accepted.
	ok := `{"profile_id":"p","name":"P","agent_impl":"claude","model":"m","provider_id":"anthropic"}`
	p, err := ParseMetaAgentProfile([]byte(ok), "test", DefaultProviderLoader, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Provider.ClientKind != ClientAnthropic {
		t.Errorf("provider client_kind = %q, want anthropic", p.Provider.ClientKind)
	}
}

func TestLoadDefaultProfiles(t *testing.T) {
	mp, err := LoadMetaProfile("default-meta", LoadProvider, nil)
	if err != nil {
		t.Fatalf("LoadMetaProfile: %v", err)
	}
	if mp.AgentImpl != "claude" || mp.Provider.ProviderID != "anthropic" {
		t.Errorf("default-meta = %+v, want claude/anthropic", mp)
	}
	tp, err := LoadTargetProfile("default-target", LoadProvider)
	if err != nil {
		t.Fatalf("LoadTargetProfile: %v", err)
	}
	if tp.AgentReference.Kind != RefDefault {
		t.Errorf("default-target reference kind = %q, want default", tp.AgentReference.Kind)
	}
	if _, err := LoadMetaProfile("nope", LoadProvider, nil); err == nil {
		t.Error("LoadMetaProfile(nope) should error on unknown name")
	}
}

func TestParseAgentReference(t *testing.T) {
	if r, _ := ParseAgentReference("default", ""); r.Kind != RefDefault {
		t.Errorf("default kind = %q", r.Kind)
	}
	if r, _ := ParseAgentReference(nil, ""); r.Kind != RefDefault {
		t.Errorf("nil kind = %q", r.Kind)
	}
	// A file source.
	dir := t.TempDir()
	file := filepath.Join(dir, "agent.py")
	writeTestFile(t, file, "print('x')")
	r, err := ParseAgentReference(map[string]any{"source": file}, "")
	if err != nil {
		t.Fatalf("file ref: %v", err)
	}
	if r.Kind != RefFile || r.Source != file {
		t.Errorf("file ref = %+v", r)
	}
	// A directory source.
	rd, err := ParseAgentReference(map[string]any{"source": dir, "entrypoint": "agent.py"}, "")
	if err != nil {
		t.Fatalf("dir ref: %v", err)
	}
	if rd.Kind != RefDir || rd.Entrypoint != "agent.py" {
		t.Errorf("dir ref = %+v", rd)
	}
	// Bad spec.
	if _, err := ParseAgentReference(map[string]any{"nope": 1}, ""); err == nil {
		t.Error("missing source should error")
	}
}
