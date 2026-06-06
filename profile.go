package sia

import (
	"encoding/json"
	"fmt"
)

// MetaAgentProfile is the full configuration for the meta/feedback agent role:
// the engine that runs inside SIA via an [AgentRunner], plus its model and
// provider.
type MetaAgentProfile struct {
	ProfileID string   // stable id (also the --meta-agent-profile value)
	Name      string   // human-readable display name
	AgentImpl string   // a registered agent impl (e.g. "claude")
	Model     string   // model the engine drives
	Provider  Provider // endpoint/credentials
}

// TargetAgentProfile is the full configuration for the target agent role. The
// target is generated code SIA never runs as an engine; it is seeded from an
// [AgentReference] and iteratively improved.
type TargetAgentProfile struct {
	ProfileID      string         // stable id (also the --target-agent-profile value)
	Name           string         // human-readable display name
	Model          string         // model the generated target_agent.py calls
	Provider       Provider       // endpoint/credentials for that model
	AgentReference AgentReference // where the seed code + deps come from
}

// metaProfileJSON / targetProfileJSON mirror the on-disk profile schema.
type metaProfileJSON struct {
	ProfileID  string `json:"profile_id"`
	Name       string `json:"name"`
	AgentImpl  string `json:"agent_impl"`
	Model      string `json:"model"`
	ProviderID string `json:"provider_id"`
}

type targetProfileJSON struct {
	ProfileID      string          `json:"profile_id"`
	Name           string          `json:"name"`
	Model          string          `json:"model"`
	ProviderID     string          `json:"provider_id"`
	AgentReference json.RawMessage `json:"agent_reference"`
}

// ProviderLoader resolves a provider id (or path) to a [Provider]. Pass
// [DefaultProviderLoader] to use the built-in registry, or a custom loader to
// read provider files from disk.
type ProviderLoader func(idOrPath string) (Provider, error)

// DefaultProviderLoader resolves provider ids against the built-in
// [DefaultProviders] registry.
func DefaultProviderLoader(id string) (Provider, error) {
	p, ok := DefaultProviders[id]
	if !ok {
		return Provider{}, fmt.Errorf("unknown provider: %q", id)
	}
	return p, nil
}

// AgentImplValidator reports whether an agent impl name is registered. Pass nil
// to skip validation.
type AgentImplValidator func(name string) bool

// ParseMetaAgentProfile parses and validates a meta-agent profile from JSON.
// source is used only in error messages; providers is the provider loader;
// validImpl validates the agent_impl (nil skips that check).
func ParseMetaAgentProfile(data []byte, source string, providers ProviderLoader, validImpl AgentImplValidator) (MetaAgentProfile, error) {
	if missing := missingKeys(data, "profile_id", "name", "agent_impl", "model", "provider_id"); len(missing) > 0 {
		return MetaAgentProfile{}, fmt.Errorf("profile at %s is missing required keys: %s", source, joinSorted(missing))
	}
	var j metaProfileJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return MetaAgentProfile{}, fmt.Errorf("invalid profile json at %s: %w", source, err)
	}
	provider, err := providers(j.ProviderID)
	if err != nil {
		return MetaAgentProfile{}, fmt.Errorf("profile at %s: %w", source, err)
	}
	p := MetaAgentProfile{
		ProfileID: j.ProfileID,
		Name:      j.Name,
		AgentImpl: j.AgentImpl,
		Model:     j.Model,
		Provider:  provider,
	}
	if validImpl != nil && !validImpl(p.AgentImpl) {
		return MetaAgentProfile{}, fmt.Errorf("profile at %s has invalid agent_impl %q", source, p.AgentImpl)
	}
	// The Claude agent impl only talks to Anthropic; pairing it with another
	// provider would silently authenticate against the wrong endpoint.
	if p.AgentImpl == "claude" && p.Provider.ClientKind != ClientAnthropic {
		return MetaAgentProfile{}, fmt.Errorf(
			"profile at %s pairs agent_impl 'claude' with provider %q (client_kind=%s); "+
				"the claude agent impl requires an anthropic provider",
			source, p.Provider.Name, p.Provider.ClientKind)
	}
	return p, nil
}

// ParseTargetAgentProfile parses and validates a target-agent profile from JSON.
// baseDir resolves a relative agent_reference source (empty uses the cwd).
func ParseTargetAgentProfile(data []byte, source, baseDir string, providers ProviderLoader) (TargetAgentProfile, error) {
	if missing := missingKeys(data, "profile_id", "name", "model", "provider_id"); len(missing) > 0 {
		return TargetAgentProfile{}, fmt.Errorf("profile at %s is missing required keys: %s", source, joinSorted(missing))
	}
	var j targetProfileJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return TargetAgentProfile{}, fmt.Errorf("invalid profile json at %s: %w", source, err)
	}
	provider, err := providers(j.ProviderID)
	if err != nil {
		return TargetAgentProfile{}, fmt.Errorf("profile at %s: %w", source, err)
	}
	rawRef, err := rawAgentReference(j.AgentReference)
	if err != nil {
		return TargetAgentProfile{}, fmt.Errorf("profile at %s: invalid agent_reference: %w", source, err)
	}
	ref, err := ParseAgentReference(rawRef, baseDir)
	if err != nil {
		return TargetAgentProfile{}, fmt.Errorf("profile at %s: %w", source, err)
	}
	return TargetAgentProfile{
		ProfileID:      j.ProfileID,
		Name:           j.Name,
		Model:          j.Model,
		Provider:       provider,
		AgentReference: ref,
	}, nil
}

// DefaultMetaProfiles / DefaultTargetProfiles are the bundled profiles the
// reference ships under defaults/profiles/, kept as raw JSON so a run works
// without on-disk profile files. Resolve them with [LoadMetaProfile] /
// [LoadTargetProfile].
var DefaultMetaProfiles = map[string]string{
	"default-meta":     `{"profile_id":"default-meta","name":"Default meta agent (Claude Haiku)","agent_impl":"claude","model":"haiku","provider_id":"anthropic"}`,
	"kimi-nebius-meta": `{"profile_id":"kimi-nebius-meta","name":"Kimi K2.6 on Nebius","agent_impl":"openhands","model":"moonshotai/Kimi-K2.6","provider_id":"nebius"}`,
}

var DefaultTargetProfiles = map[string]string{
	"default-target":       `{"profile_id":"default-target","name":"Default target agent (Claude Haiku)","model":"claude-haiku-4-5-20251001","provider_id":"anthropic","agent_reference":"default"}`,
	"gptoss-nebius-target": `{"profile_id":"gptoss-nebius-target","name":"GPT OSS 120B on Nebius","model":"openai/gpt-oss-120b-fast","provider_id":"nebius","agent_reference":"default"}`,
	"kimi-nebius-target":   `{"profile_id":"kimi-nebius-target","name":"Kimi K2.6 on Nebius","model":"moonshotai/Kimi-K2.6","provider_id":"nebius","agent_reference":"default"}`,
	"qwen-nebius-target":   `{"profile_id":"qwen-nebius-target","name":"Qwen 80B on Nebius","model":"Qwen/Qwen3-Next-80B-A3B-Thinking-fast","provider_id":"nebius","agent_reference":"default"}`,
}
