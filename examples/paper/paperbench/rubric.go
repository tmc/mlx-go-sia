package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	sia "github.com/tmc/mlx-go-sia"
)

// demoValidator accepts a JSONL fixture iff every row carries all five required
// fields (structural check) and "accepted":true (claim check). A missing field
// exits 2 (schema break); "accepted":false exits 1 (claim rejection). This lets
// the evaluator's tamper probe and the frozen negative both be cleanly rejected,
// so an always-exit-0 stub validator cannot satisfy validator_command.
const demoValidator = `#!/bin/sh
f="$1"
[ -f "$f" ] || exit 3
any=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  any=1
  for field in '"schema"' '"run_id"' '"device"' '"accepted"' '"digest"'; do
    case "$line" in
      *"$field"*) : ;;
      *) exit 2 ;;
    esac
  done
  case "$line" in
    *'"accepted":true'*) : ;;
    *'"accepted":false'*) exit 1 ;;
    *) exit 2 ;;
  esac
done < "$f"
[ "$any" = "1" ] || exit 2
exit 0
`

// demoModelCmd writes a deterministic artifact so two evaluator runs are
// byte-identical — a genuine reproducible producer (a printf stub that emitted
// random bytes would fail the reproducibility check).
const demoModelCmd = `#!/bin/sh
printf 'paperbench-demo-model-output-v1\n' > model_out.txt
`

// scaffoldRubric writes the frozen evaluator-owned rubric to dir and pins every
// file in manifest.json so the evaluator's integrity check passes.
func scaffoldRubric(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	schema := sia.RowSchema{
		ConstTag:     demoSchemaTag,
		TagField:     "schema",
		Required:     []string{"schema", "run_id", "device", "accepted", "digest"},
		StringFields: []string{"run_id", "device"},
		BoolFields:   []string{"accepted"},
		Enums:        map[string][]string{"device": {"gpu", "cpu", "ane"}},
		DigestFields: []string{"digest"},
		ClaimFields:  []string{"accepted"},
	}
	schemaJSON, _ := json.Marshal(schema)

	cfg := map[string]any{
		"Schema":        json.RawMessage(schemaJSON),
		"Interpreter":   "/bin/sh",
		"ValidatorRel":  "validator.sh",
		"PositiveRel":   "positive.jsonl",
		"NegativeRel":   "negative.jsonl",
		"ModelCommand":  []string{"/bin/sh", filepath.Join(dir, "model_cmd.sh")},
		"ModelOutName":  "model_out.txt",
		"ModelMinBytes": 4,
		"ScopeRel":      "scope.json",
	}
	cfgJSON, _ := json.MarshalIndent(cfg, "", "  ")

	positive := demoRow("frozen-pos", "gpu", true) + "\n"
	negative := demoRow("frozen-neg", "gpu", false) + "\n"
	scope := `[{"id":"gpu_overlap","predicate":{"jsonl_field":"device","equals":"gpu","min_rows":1}}]`

	pinned := map[string]string{
		"validator.sh":   demoValidator,
		"model_cmd.sh":   demoModelCmd,
		"positive.jsonl": positive,
		"negative.jsonl": negative,
		"scope.json":     scope,
		"rubric.json":    string(cfgJSON),
	}
	files := map[string]string{}
	for rel, content := range pinned {
		mode := os.FileMode(0o644)
		if filepath.Ext(rel) == ".sh" {
			mode = 0o755
		}
		if err := os.WriteFile(filepath.Join(dir, rel), []byte(content), mode); err != nil {
			return err
		}
		sum := sha256.Sum256([]byte(content))
		files[rel] = hex.EncodeToString(sum[:])
	}
	manJSON, _ := json.MarshalIndent(sia.RubricManifest{Files: files}, "", "  ")
	return os.WriteFile(filepath.Join(dir, "manifest.json"), manJSON, 0o644)
}
