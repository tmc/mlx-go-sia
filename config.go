package sia

import (
	"os"
	"strconv"
)

// Config holds the SIA orchestration defaults. The zero value is not usable;
// construct with [DefaultConfig] or [ConfigFromEnv].
type Config struct {
	// Agent profile defaults (resolved on the CLI; see profiles).
	DefaultMetaAgentProfile   string
	DefaultTargetAgentProfile string

	// Model fallbacks for context metadata / env overrides.
	DefaultMetaModel string
	DefaultTaskModel string

	// Generation defaults.
	DefaultMaxGenerations int
	DefaultRunID          int

	// Agent execution.
	DefaultMaxTurns  int
	DefaultAgentImpl string

	// Truncation limits (bytes / characters in preview blocks).
	AgentCodePreviewLimit  int
	TrajectoryPreviewLimit int
	InsightPreviewLimit    int

	// Context-manager LLM summary (turns budget for the optional summarizer).
	ContextSummaryMaxTurns int

	// Timeouts (seconds).
	EvalTimeout int

	// Sandbox.
	SandboxMode string // "none" or "docker"

	// File size limits (bytes).
	MaxExecutionLogSize int64
}

// DefaultConfig returns the reference's built-in defaults.
func DefaultConfig() Config {
	return Config{
		DefaultMetaAgentProfile:   "default-meta",
		DefaultTargetAgentProfile: "default-target",
		DefaultMetaModel:          "haiku",
		DefaultTaskModel:          "claude-haiku-4-5-20251001",
		DefaultMaxGenerations:     3,
		DefaultRunID:              1,
		DefaultMaxTurns:           20,
		DefaultAgentImpl:          "claude",
		AgentCodePreviewLimit:     3000,
		TrajectoryPreviewLimit:    1000,
		InsightPreviewLimit:       200,
		ContextSummaryMaxTurns:    5,
		EvalTimeout:               600,
		SandboxMode:               "none",
		MaxExecutionLogSize:       50_000_000,
	}
}

// ConfigFromEnv returns [DefaultConfig] with SIA_* environment overrides
// applied. Unparseable numeric values are ignored (the default is kept).
func ConfigFromEnv() Config {
	cfg := DefaultConfig()
	if v, ok := os.LookupEnv("SIA_META_AGENT_PROFILE"); ok {
		cfg.DefaultMetaAgentProfile = v
	}
	if v, ok := os.LookupEnv("SIA_TARGET_AGENT_PROFILE"); ok {
		cfg.DefaultTargetAgentProfile = v
	}
	if v, ok := os.LookupEnv("SIA_META_MODEL"); ok {
		cfg.DefaultMetaModel = v
	}
	if v, ok := os.LookupEnv("SIA_TASK_MODEL"); ok {
		cfg.DefaultTaskModel = v
	}
	if v, ok := os.LookupEnv("SIA_AGENT_IMPL"); ok {
		cfg.DefaultAgentImpl = v
	}
	if v, ok := os.LookupEnv("SIA_SANDBOX_MODE"); ok {
		cfg.SandboxMode = v
	}
	if v, ok := os.LookupEnv("SIA_MAX_GENERATIONS"); ok {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DefaultMaxGenerations = n
		}
	}
	if v, ok := os.LookupEnv("SIA_MAX_TURNS"); ok {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DefaultMaxTurns = n
		}
	}
	return cfg
}
