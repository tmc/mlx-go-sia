package sia

import (
	"encoding/json"
	"fmt"
)

// ClientKind names the SDK family a generated or meta agent uses to reach a
// model provider.
type ClientKind string

const (
	ClientAnthropic ClientKind = "anthropic"
	ClientOpenAI    ClientKind = "openai"
	ClientGoogle    ClientKind = "google"
)

func (k ClientKind) valid() bool {
	switch k {
	case ClientAnthropic, ClientOpenAI, ClientGoogle:
		return true
	}
	return false
}

// Provider describes how to reach a model provider's API. The zero value is not
// usable; load one with [LoadProvider] or [ParseProvider].
type Provider struct {
	ProviderID string     `json:"provider_id"` // stable id, referenced by a profile's provider_id
	Name       string     `json:"name"`        // human-readable display name
	ClientKind ClientKind `json:"client_kind"` // anthropic | openai | google
	BaseURL    string     `json:"base_url"`    // empty for native endpoints; set for OpenAI-compatible
	APIKeyEnv  string     `json:"api_key_env"` // env var holding the API key
}

// ParseProvider parses and validates a provider definition from JSON. source is
// used only in error messages.
func ParseProvider(data []byte, source string) (Provider, error) {
	var p Provider
	if err := json.Unmarshal(data, &p); err != nil {
		return Provider{}, fmt.Errorf("invalid provider json at %s: %w", source, err)
	}
	if missing := missingKeys(data, "provider_id", "name", "client_kind", "api_key_env"); len(missing) > 0 {
		return Provider{}, fmt.Errorf("provider at %s is missing required keys: %s", source, joinSorted(missing))
	}
	if !p.ClientKind.valid() {
		return Provider{}, fmt.Errorf("provider at %s has invalid client_kind %q: expected one of: anthropic, openai, google", source, p.ClientKind)
	}
	return p, nil
}

// DefaultProviders are the providers the reference ships under
// defaults/providers/. The Go port keeps them as a built-in registry so a run
// works without on-disk provider files.
var DefaultProviders = map[string]Provider{
	"anthropic": {ProviderID: "anthropic", Name: "Anthropic", ClientKind: ClientAnthropic, APIKeyEnv: "ANTHROPIC_API_KEY"},
	"gemini":    {ProviderID: "gemini", Name: "Google Gemini", ClientKind: ClientGoogle, APIKeyEnv: "GEMINI_API_KEY"},
	"openai":    {ProviderID: "openai", Name: "OpenAI", ClientKind: ClientOpenAI, BaseURL: "https://api.openai.com/v1", APIKeyEnv: "OPENAI_API_KEY"},
	"nebius":    {ProviderID: "nebius", Name: "Nebius Token Factory", ClientKind: ClientOpenAI, BaseURL: "https://api.tokenfactory.us-central1.nebius.com/v1/", APIKeyEnv: "NEBIUS_API_KEY"},
	"together":  {ProviderID: "together", Name: "Together AI", ClientKind: ClientOpenAI, BaseURL: "https://api.together.ai/v1", APIKeyEnv: "TOGETHER_API_KEY"},
	"tinker":    {ProviderID: "tinker", Name: "Tinker API", ClientKind: ClientOpenAI, BaseURL: "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", APIKeyEnv: "TINKER_API_KEY"},
}
