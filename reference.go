package sia

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// ReferenceKind distinguishes where a target agent's improvable seed comes from.
type ReferenceKind string

const (
	// RefDefault uses the task package's bundled reference/ directory.
	RefDefault ReferenceKind = "default"
	// RefFile uses a single user file, embedded into the meta prompt.
	RefFile ReferenceKind = "file"
	// RefDir uses a multi-file directory the agent reads with its own tools.
	RefDir ReferenceKind = "dir"
)

// AgentReference is a parsed agent_reference spec with paths resolved absolute.
type AgentReference struct {
	Kind       ReferenceKind
	Source     string // abs path to file (RefFile) or dir (RefDir); empty for RefDefault
	Entrypoint string // filename within the directory (RefDir only)
}

// DefaultAgentReference is the historical default: the task's bundled reference.
var DefaultAgentReference = AgentReference{Kind: RefDefault}

// ParseAgentReference parses a raw agent_reference value from a profile. spec is
// the decoded JSON value ("default", or an object with a "source" field). A
// relative source resolves against baseDir, or the current directory if baseDir
// is empty.
func ParseAgentReference(spec any, baseDir string) (AgentReference, error) {
	if spec == nil || spec == "default" {
		return DefaultAgentReference, nil
	}
	obj, ok := spec.(map[string]any)
	if !ok {
		return AgentReference{}, fmt.Errorf(`agent_reference must be "default" or an object with a "source" field`)
	}
	rawSource, ok := obj["source"].(string)
	if !ok {
		return AgentReference{}, fmt.Errorf(`agent_reference must be "default" or an object with a "source" field`)
	}

	source := rawSource
	if !filepath.IsAbs(source) {
		base := baseDir
		if base == "" {
			if cwd, err := os.Getwd(); err == nil {
				base = cwd
			}
		}
		source = filepath.Join(base, source)
	}
	source, err := filepath.Abs(source)
	if err != nil {
		return AgentReference{}, fmt.Errorf("resolve agent_reference source: %w", err)
	}

	info, err := os.Stat(source)
	if err != nil {
		return AgentReference{}, fmt.Errorf("agent_reference source not found: %s", source)
	}
	if info.IsDir() {
		entry, _ := obj["entrypoint"].(string)
		return AgentReference{Kind: RefDir, Source: source, Entrypoint: entry}, nil
	}
	return AgentReference{Kind: RefFile, Source: source}, nil
}

// ResolvedAgentReference is an [AgentReference] resolved against a concrete task.
type ResolvedAgentReference struct {
	InlineSeed   string // entrypoint text to embed in the prompt (default/file); empty for a dir
	RefDir       string // directory copied into each gen working dir (dir only); empty otherwise
	Entrypoint   string // filename the agent treats as the starting point
	Requirements string // requirements.txt to install + carry forward, if present; empty otherwise
}

// Resolve resolves the reference against a concrete task layout into a
// [ResolvedAgentReference].
func (r AgentReference) Resolve(task TaskLayout) (ResolvedAgentReference, error) {
	switch r.Kind {
	case RefDefault:
		refDir := task.ReferenceDir()
		seed, err := os.ReadFile(filepath.Join(refDir, NameReferenceAgentFile))
		if err != nil {
			return ResolvedAgentReference{}, fmt.Errorf("read reference agent: %w", err)
		}
		return ResolvedAgentReference{
			InlineSeed:   string(seed),
			Entrypoint:   NameReferenceAgentFile,
			Requirements: existingFile(filepath.Join(refDir, NameRequirementsTxt)),
		}, nil

	case RefFile:
		seed, err := os.ReadFile(r.Source)
		if err != nil {
			return ResolvedAgentReference{}, fmt.Errorf("read reference file: %w", err)
		}
		return ResolvedAgentReference{
			InlineSeed: string(seed),
			Entrypoint: filepath.Base(r.Source),
		}, nil

	case RefDir:
		entrypoint := r.Entrypoint
		if entrypoint == "" {
			entrypoint = NameReferenceAgentFile
		}
		if !isFile(filepath.Join(r.Source, entrypoint)) {
			return ResolvedAgentReference{}, fmt.Errorf("agent_reference entrypoint %q not found in %s", entrypoint, r.Source)
		}
		return ResolvedAgentReference{
			RefDir:       r.Source,
			Entrypoint:   entrypoint,
			Requirements: existingFile(filepath.Join(r.Source, NameRequirementsTxt)),
		}, nil
	}
	return ResolvedAgentReference{}, fmt.Errorf("unknown agent_reference kind: %q", r.Kind)
}

// CopyInto places reference helper files + requirements.txt into a generation
// working dir. For a directory reference the whole reference is copied so the
// generated target_agent.py can import its helper modules; for default/file
// references only a sibling requirements.txt (if any) is carried in.
func (r ResolvedAgentReference) CopyInto(genDir string) error {
	if r.RefDir != "" {
		return copyTree(r.RefDir, genDir)
	}
	if r.Requirements != "" {
		return copyFile(r.Requirements, filepath.Join(genDir, NameRequirementsTxt))
	}
	return nil
}

// rawAgentReference decodes the agent_reference field while preserving the
// distinction between "default" (string) and an object.
func rawAgentReference(m json.RawMessage) (any, error) {
	if len(m) == 0 {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal(m, &v); err != nil {
		return nil, err
	}
	return v, nil
}

func existingFile(path string) string {
	if isFile(path) {
		return path
	}
	return ""
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(path, target)
	})
}
