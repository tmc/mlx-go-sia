package sia

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Rubric is the frozen, evaluator-owned scoring context for a [PaperEvaluator].
// It lives in a directory outside the agent's gen dir, holds the pristine
// validator, the per-row JSON schema, and known-good positive/negative inputs,
// and is pinned by a content manifest the agent has never seen. Its integrity is
// re-verified at the start of every Evaluate so a tampered frozen input aborts
// scoring (an EvalError) rather than silently passing.
//
// The zero value is unusable; construct one with [LoadRubric].
type Rubric struct {
	// Dir is the read-only frozen directory.
	Dir string
	// Manifest pins every file under Dir to its sha256 at load time.
	Manifest RubricManifest
	// Schema is the frozen per-row structural schema fixtures validate against.
	Schema RowSchema
	// Interpreter is the absolute path to the program that runs the validator
	// and model-backed commands (e.g. "/bin/sh"). It is never PATH-resolved.
	Interpreter string
	// ValidatorRel is the validator script path relative to Dir.
	ValidatorRel string
	// PositiveRel, NegativeRel are validator inputs relative to Dir: the positive
	// must be accepted (exit 0), the negative rejected (exit 1..125).
	PositiveRel string
	NegativeRel string
	// ModelCommand, when set, is the frozen opt-in/model-backed command run in a
	// sealed scratch dir; ModelOutName is its declared output file and
	// ModelMinBytes the minimum plausible artifact size.
	ModelCommand  []string
	ModelOutName  string
	ModelMinBytes int64
	// ScopeRel, when set, is a frozen heavy-skip scope file relative to Dir.
	ScopeRel string

	// oracleFileHashes is the set of sha256 of every frozen file, used to reject
	// an agent that copies an evaluator input verbatim into the gen dir.
	oracleFileHashes map[string]bool
}

// RubricManifest pins the frozen rubric's files to their content hashes.
type RubricManifest struct {
	// Files maps a path relative to the rubric dir to its lowercase hex sha256.
	Files map[string]string `json:"files"`
}

// RowSchema is a hand-rolled structural schema for a coverage-map fixture row.
// It is deliberately not a full JSON Schema implementation: it asserts the
// shape an honest fixture must have (a const schema tag, required fields, exact
// JSON types, enum membership, numeric bounds, sha256-shaped digests) so that
// fabricating a structurally valid row still costs effort, while semantic truth
// is left to the frozen validator. ClaimFields names the fields a falsifier must
// perturb to flip the claim.
type RowSchema struct {
	ConstTag     string              `json:"const_tag"`     // exact value of the schema/schema_version field
	TagField     string              `json:"tag_field"`     // name of that field (e.g. "schema")
	Required     []string            `json:"required"`      // required field names
	StringFields []string            `json:"string_fields"` // must be JSON strings
	IntFields    []string            `json:"int_fields"`    // must be JSON integers
	BoolFields   []string            `json:"bool_fields"`   // must be JSON booleans
	Enums        map[string][]string `json:"enums"`         // field -> allowed string values
	DigestFields []string            `json:"digest_fields"` // must match ^sha256:[0-9a-f]{64}$
	ClaimFields  []string            `json:"claim_fields"`  // perturbing these flips the claim
}

// LoadRubric loads the frozen rubric rooted at dir. It reads rubric.json (the
// configuration), verifies a manifest.json pins every other file, and precomputes
// the oracle hash set. The manifest must already match the on-disk files.
func LoadRubric(dir string) (Rubric, error) {
	cfgPath := filepath.Join(dir, "rubric.json")
	cfgData, err := os.ReadFile(cfgPath)
	if err != nil {
		return Rubric{}, fmt.Errorf("read rubric config: %w", err)
	}
	var r Rubric
	if err := json.Unmarshal(cfgData, &r); err != nil {
		return Rubric{}, fmt.Errorf("parse rubric config: %w", err)
	}
	r.Dir = dir

	manData, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return Rubric{}, fmt.Errorf("read rubric manifest: %w", err)
	}
	if err := json.Unmarshal(manData, &r.Manifest); err != nil {
		return Rubric{}, fmt.Errorf("parse rubric manifest: %w", err)
	}
	if len(r.Manifest.Files) == 0 {
		return Rubric{}, fmt.Errorf("rubric manifest pins no files")
	}

	r.oracleFileHashes = make(map[string]bool, len(r.Manifest.Files))
	for _, h := range r.Manifest.Files {
		r.oracleFileHashes[h] = true
	}
	if err := r.verify(); err != nil {
		return Rubric{}, fmt.Errorf("rubric integrity at load: %w", err)
	}
	return r, nil
}

// verify recomputes the sha256 of every pinned file and fails if any differs
// from the manifest. It is the integrity gate: a mismatch means the agent (or
// disk) altered a frozen input, so scoring must abort rather than trust it.
func (r *Rubric) verify() error {
	if r.Dir == "" {
		return fmt.Errorf("rubric dir not set")
	}
	if len(r.Manifest.Files) == 0 {
		return fmt.Errorf("rubric manifest empty")
	}
	for rel, want := range r.Manifest.Files {
		clean, ok := safeJoin(r.Dir, rel)
		if !ok {
			return fmt.Errorf("pinned file escapes rubric dir: %s", rel)
		}
		data, err := os.ReadFile(clean)
		if err != nil {
			return fmt.Errorf("read pinned file %s: %w", rel, err)
		}
		sum := sha256.Sum256(data)
		if got := hex.EncodeToString(sum[:]); got != want {
			return fmt.Errorf("pinned file %s hash mismatch", rel)
		}
	}
	return nil
}

// frozenPath resolves rel against the rubric dir, returning ok=false if it
// escapes (the frozen files are trusted, but defense-in-depth is cheap).
func (r *Rubric) frozenPath(rel string) (string, bool) {
	if rel == "" {
		return "", false
	}
	return safeJoin(r.Dir, rel)
}

// oracleRowHashes returns the canonical-row hash set for the frozen positive and
// negative fixtures, used to deny credit for replaying an evaluator input row.
func (r *Rubric) oracleRowHashes() map[string]bool {
	out := map[string]bool{}
	for _, rel := range []string{r.PositiveRel, r.NegativeRel} {
		p, ok := r.frozenPath(rel)
		if !ok {
			continue
		}
		f, _, ok := openRegular(r.Dir, rel, maxFixtureBytes)
		if !ok {
			// frozenPath proved containment; read directly for the oracle set.
			data, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			for _, h := range rowHashesOf(data) {
				out[h] = true
			}
			continue
		}
		data := readAllCapped(f, maxFixtureBytes)
		f.Close()
		for _, h := range rowHashesOf(data) {
			out[h] = true
		}
	}
	return out
}

// rowHashesOf returns the canonical-row hash of each well-formed JSON object
// line in data, ignoring malformed lines.
func rowHashesOf(data []byte) []string {
	var out []string
	for _, line := range splitJSONLines(data) {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(line, &obj); err != nil {
			continue
		}
		if h, err := canonicalRowHash(obj); err == nil {
			out = append(out, h)
		}
	}
	sort.Strings(out)
	return out
}
