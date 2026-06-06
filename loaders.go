package sia

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LoadMetaProfile resolves a meta-agent profile by built-in name or by path to a
// .json file. A value ending in ".json" (or containing a path separator) is read
// from disk; otherwise it is looked up in [DefaultMetaProfiles]. providers
// resolves the referenced provider; validImpl validates the agent_impl (nil
// skips).
func LoadMetaProfile(nameOrPath string, providers ProviderLoader, validImpl AgentImplValidator) (MetaAgentProfile, error) {
	data, source, baseDir, err := readConfig(nameOrPath, DefaultMetaProfiles, "meta profile")
	if err != nil {
		return MetaAgentProfile{}, err
	}
	_ = baseDir
	return ParseMetaAgentProfile(data, source, providers, validImpl)
}

// LoadTargetProfile resolves a target-agent profile by built-in name or path.
// The base directory for a relative agent_reference source is the profile file's
// directory (or the cwd for a built-in).
func LoadTargetProfile(nameOrPath string, providers ProviderLoader) (TargetAgentProfile, error) {
	data, source, baseDir, err := readConfig(nameOrPath, DefaultTargetProfiles, "target profile")
	if err != nil {
		return TargetAgentProfile{}, err
	}
	return ParseTargetAgentProfile(data, source, baseDir, providers)
}

// LoadProvider resolves a provider by built-in name or by path to a .json file.
func LoadProvider(nameOrPath string) (Provider, error) {
	if isConfigPath(nameOrPath) {
		data, err := os.ReadFile(nameOrPath)
		if err != nil {
			return Provider{}, fmt.Errorf("read provider %s: %w", nameOrPath, err)
		}
		return ParseProvider(data, nameOrPath)
	}
	p, err := DefaultProviderLoader(nameOrPath)
	if err != nil {
		return Provider{}, err
	}
	return p, nil
}

// readConfig returns the JSON bytes for nameOrPath: from disk when it looks like
// a path, else from the built-in registry. source is a human label and baseDir
// is the directory for resolving relative references (empty for built-ins).
func readConfig(nameOrPath string, builtins map[string]string, kind string) (data []byte, source, baseDir string, err error) {
	if isConfigPath(nameOrPath) {
		b, rerr := os.ReadFile(nameOrPath)
		if rerr != nil {
			return nil, "", "", fmt.Errorf("read %s %s: %w", kind, nameOrPath, rerr)
		}
		return b, nameOrPath, filepath.Dir(nameOrPath), nil
	}
	raw, ok := builtins[nameOrPath]
	if !ok {
		return nil, "", "", fmt.Errorf("unknown %s: %q (available: %s)", kind, nameOrPath, joinSorted(keysOf(builtins)))
	}
	return []byte(raw), "<bundled>/" + nameOrPath, "", nil
}

// isConfigPath reports whether s should be treated as a filesystem path rather
// than a built-in name.
func isConfigPath(s string) bool {
	return strings.HasSuffix(s, ".json") || strings.ContainsRune(s, filepath.Separator) || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../")
}

func keysOf(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
