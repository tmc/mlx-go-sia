package sia

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The tests build a real frozen rubric and a real gen dir on disk, then assert
// that each evidence boolean is recomputed from ground truth — and, critically,
// that the documented gaming vectors are caught. The frozen validator is a tiny
// shell script that accepts a JSONL fixture iff every row carries
// "accepted":true and a non-"bad" device; it rejects (exit 1) otherwise. This
// lets a negative input and a tampered positive both be cleanly rejected, so an
// always-exit-0 stub cannot pass validator_command.

const testSchemaTag = "paperbench-test-row/v1"

// testRowSchema is the frozen structural schema used across the tests.
func testRowSchema() RowSchema {
	return RowSchema{
		ConstTag:     testSchemaTag,
		TagField:     "schema",
		Required:     []string{"schema", "run_id", "device", "accepted", "digest"},
		StringFields: []string{"run_id", "device"},
		BoolFields:   []string{"accepted"},
		Enums:        map[string][]string{"device": {"gpu", "cpu", "ane", "bad"}},
		DigestFields: []string{"digest"},
		ClaimFields:  []string{"accepted"},
	}
}

// frozenValidatorScript accepts a JSONL file iff every line has accepted:true.
// It distinguishes a schema break (missing fields => exit 2) from a claim
// rejection (accepted:false => exit 1) so falsifier classification can be tested.
const frozenValidatorScript = `#!/bin/sh
# args: <input.jsonl>
f="$1"
[ -f "$f" ] || exit 3
any=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  any=1
  # structural check: every required field must be present (exit 2 if not)
  for field in '"schema"' '"run_id"' '"device"' '"accepted"' '"digest"'; do
    case "$line" in
      *"$field"*) : ;;
      *) exit 2 ;;                     # schema break (missing field)
    esac
  done
  # claim check: the prototype's claim is that the row was accepted
  case "$line" in
    *'"accepted":true'*) : ;;
    *'"accepted":false'*) exit 1 ;;    # claim rejection
    *) exit 2 ;;
  esac
done < "$f"
[ "$any" = "1" ] || exit 2
exit 0
`

// modelCommandScript writes a deterministic artifact (fixed content) so two runs
// are byte-identical — a genuine reproducible producer.
const modelCommandScript = `#!/bin/sh
printf 'model-output-deterministic-v1\n' > model_out.txt
`

// digestOf returns a schema-valid sha256:-prefixed digest derived from s.
func digestOf(s string) string {
	sum := sha256.Sum256([]byte(s))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// row builds a schema-valid test row as a compact JSON line.
func row(runID, device string, accepted bool) string {
	return fmt.Sprintf(`{"schema":%q,"run_id":%q,"device":%q,"accepted":%t,"digest":%q}`,
		testSchemaTag, runID, device, accepted, digestOf(runID+device))
}

// buildRubric writes a frozen rubric to a temp dir and returns a loaded Rubric.
func buildRubric(t *testing.T) Rubric {
	t.Helper()
	dir := t.TempDir()

	write := func(rel, content string, mode os.FileMode) {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), mode); err != nil {
			t.Fatal(err)
		}
	}

	// Frozen positive: one accepted row. Negative: one rejected (accepted:false).
	positive := row("frozen-pos", "gpu", true) + "\n"
	negative := row("frozen-neg", "gpu", false) + "\n"

	write("validator.sh", frozenValidatorScript, 0o755)
	write("model_cmd.sh", modelCommandScript, 0o755)
	write("positive.jsonl", positive, 0o644)
	write("negative.jsonl", negative, 0o644)

	scope := `[{"id":"gpu_overlap","predicate":{"jsonl_field":"device","equals":"gpu","min_rows":1}}]`
	write("scope.json", scope, 0o644)

	schemaJSON, _ := json.Marshal(testRowSchema())
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
	write("rubric.json", string(cfgJSON), 0o644)

	// Build the manifest pinning every file except manifest.json itself.
	files := map[string]string{}
	for _, rel := range []string{"validator.sh", "model_cmd.sh", "positive.jsonl", "negative.jsonl", "scope.json", "rubric.json"} {
		data, err := os.ReadFile(filepath.Join(dir, rel))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		files[rel] = hex.EncodeToString(sum[:])
	}
	manJSON, _ := json.MarshalIndent(RubricManifest{Files: files}, "", "  ")
	write("manifest.json", string(manJSON), 0o644)

	r, err := LoadRubric(dir)
	if err != nil {
		t.Fatalf("LoadRubric: %v", err)
	}
	return r
}

// genWith writes the given files into a fresh gen dir and returns its path.
func genWith(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for rel, content := range files {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// evalGen runs a PaperEvaluator over genDir and returns the parsed results.json.
func evalGen(t *testing.T, rubric Rubric, rowStatus string, genDir string) EvidenceResults {
	t.Helper()
	e := &PaperEvaluator{
		Row: CoverageRow{
			ID:        "test-proto",
			Status:    rowStatus,
			FastCheck: "true", // exit 0
			Examples:  []string{"fixtures/test-proto.jsonl"},
			EvidenceState: map[string]bool{ // self-report; must be IGNORED
				"fixture_row": true, "validator_command": true,
				"artifact_manifest_hash": true, "model_backed_or_opt_in_command": true,
				"control_rows": true, "falsifier_rows": true,
				"heavy_skip_narrowed_or_cleared": true,
			},
		},
		RepoRoot: t.TempDir(),
		Rubric:   rubric,
	}
	res, err := e.Evaluate(context.Background(), genDir)
	if err != nil {
		t.Fatalf("Evaluate returned Go error: %v", err)
	}
	if res.Status != EvalSuccess {
		t.Fatalf("Evaluate status = %q, want success; reason=%q", res.Status, res.Reason)
	}
	data, err := os.ReadFile(res.ResultsPath)
	if err != nil {
		t.Fatalf("read results.json: %v", err)
	}
	var out EvidenceResults
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("parse results.json: %v", err)
	}
	return out
}

// manifestFor builds an evidence manifest for the given gen-dir artifacts.
func manifestFor(t *testing.T, genDir string, artifacts ...[2]string) string {
	t.Helper()
	type art struct {
		ArtifactID   string `json:"artifact_id"`
		ArtifactKind string `json:"artifact_kind"`
		Path         string `json:"path"`
		SHA256       string `json:"sha256"`
		ByteSize     int64  `json:"byte_size"`
	}
	var arts []art
	for i, a := range artifacts {
		id, rel := a[0], a[1]
		data, err := os.ReadFile(filepath.Join(genDir, rel))
		if err != nil {
			t.Fatalf("read artifact %s: %v", rel, err)
		}
		sum := sha256.Sum256(data)
		arts = append(arts, art{
			ArtifactID:   id,
			ArtifactKind: "fixture",
			Path:         rel,
			SHA256:       "sha256:" + hex.EncodeToString(sum[:]),
			ByteSize:     int64(len(data)),
		})
		_ = i
	}
	m := map[string]any{
		"schema_version": manifestSchemaVersion,
		"manifest_id":    "m-1",
		"run_id":         "r-1",
		"artifacts":      arts,
	}
	b, _ := json.MarshalIndent(m, "", "  ")
	return string(b)
}

// --- the table tests ---

func TestPaperEvaluator_HonestPass(t *testing.T) {
	rubric := buildRubric(t)
	// A control (accepted gpu) + a falsifier near-miss (same but accepted:false).
	fixture := row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n"
	artifactBody := row("art", "gpu", true) + "\n"
	gen := genWith(t, map[string]string{
		"fixtures/test-proto.jsonl": fixture,
		"artifact.jsonl":            artifactBody,
		"scope-manifest.json":       `{"covered":["gpu_overlap"],"links":{"gpu_overlap":"a-gpu"}}`,
	})
	gen = withManifest(t, gen, "a-gpu", "artifact.jsonl")

	res := evalGen(t, rubric, "lightweight", gen)

	want := map[string]bool{
		"fixture_row":                    true,
		"validator_command":              true,
		"artifact_manifest_hash":         true,
		"model_backed_or_opt_in_command": true,
		"control_rows":                   true,
		"falsifier_rows":                 true,
		"heavy_skip_narrowed_or_cleared": true, // sole baseline scope id is cleared by a real, hash-verified gpu artifact
	}
	for k, w := range want {
		if res.EvidenceState[k] != w {
			t.Errorf("evidence[%s] = %v, want %v (detail: %s)", k, res.EvidenceState[k], w, res.Detail[k])
		}
	}
	if res.Verdict != "PASS" {
		t.Errorf("verdict = %q, want PASS (blockers=%v)", res.Verdict, res.Blockers)
	}
	if res.AdvisoryScore <= 0 || res.AdvisoryScore > 1.0 {
		t.Errorf("advisory_score = %v, want in (0,1]", res.AdvisoryScore)
	}
}

// withManifest adds an evidence manifest to an existing gen dir.
func withManifest(t *testing.T, genDir string, artifacts ...string) string {
	t.Helper()
	var pairs [][2]string
	for i := 0; i+1 < len(artifacts); i += 2 {
		pairs = append(pairs, [2]string{artifacts[i], artifacts[i+1]})
	}
	man := manifestFor(t, genDir, pairs...)
	if err := os.WriteFile(filepath.Join(genDir, evidenceManifestName), []byte(man), 0o644); err != nil {
		t.Fatal(err)
	}
	return genDir
}

func TestPaperEvaluator_IgnoresSelfReportedEvidence(t *testing.T) {
	rubric := buildRubric(t)
	// Gen dir with NOTHING but a lying results.json claiming everything true.
	gen := genWith(t, map[string]string{
		"results.json": `{"verdict":"PASS","advisory_score":1.0,"evidence_state":{` +
			`"fixture_row":true,"validator_command":true,"artifact_manifest_hash":true,` +
			`"model_backed_or_opt_in_command":true,"control_rows":true,"falsifier_rows":true,` +
			`"heavy_skip_narrowed_or_cleared":true}}`,
	})
	res := evalGen(t, rubric, "lightweight", gen)

	// No real artifacts => fixture_row false, validator_command still true (it
	// uses the FROZEN inputs, not the gen dir), but blockers remain => REVISE.
	if res.EvidenceState["fixture_row"] {
		t.Error("fixture_row true despite no real fixture (self-report was trusted)")
	}
	if res.Verdict != "REVISE" {
		t.Errorf("verdict = %q, want REVISE (pre-seeded all-true results.json must be ignored)", res.Verdict)
	}
}

func TestValidatorCommand_RejectsExitZeroStub(t *testing.T) {
	rubric := buildRubric(t)
	// Swap the frozen validator for an always-exit-0 stub AND re-pin it, to
	// simulate a rubric whose validator never rejects. It must fail the negative
	// and tamper cases.
	stub := "#!/bin/sh\nexit 0\n"
	if err := os.WriteFile(filepath.Join(rubric.Dir, "validator.sh"), []byte(stub), 0o755); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(stub))
	rubric.Manifest.Files["validator.sh"] = hex.EncodeToString(sum[:])
	rubric.oracleFileHashes[hex.EncodeToString(sum[:])] = true

	gen := genWith(t, map[string]string{
		"fixtures/test-proto.jsonl": row("r1", "gpu", true) + "\n",
	})
	res := evalGen(t, rubric, "lightweight", gen)

	if res.EvidenceState["validator_command"] {
		t.Errorf("validator_command true for an always-exit-0 stub (detail: %s)", res.Detail["validator_command"])
	}
	// All gated keys must be forced false and contribute zero weight.
	for _, k := range []string{"artifact_manifest_hash", "model_backed_or_opt_in_command", "control_rows", "falsifier_rows"} {
		if res.EvidenceState[k] {
			t.Errorf("gated key %s true despite validator_command false", k)
		}
	}
	// Score should be at most fixture_row's weight (0.05).
	if res.AdvisoryScore > 0.05+1e-9 {
		t.Errorf("advisory_score = %v, want <= 0.05 when validator_command false", res.AdvisoryScore)
	}
}

func TestFixtureRow_GamingVectors(t *testing.T) {
	rubric := buildRubric(t)
	good := row("r1", "gpu", true)

	cases := []struct {
		name    string
		fixture string
		want    bool
	}{
		{"valid single row", good + "\n", true},
		{"empty file", "", false},
		{"whitespace only", "   \n\t\n", false},
		{"one good many garbage", good + "\nnot json\n{bad\n" + good + "x\n", false},
		{"missing required field", `{"schema":"` + testSchemaTag + `","run_id":"r"}` + "\n", false},
		{"wrong schema tag", strings.Replace(good, testSchemaTag, testSchemaTag+"X", 1) + "\n", false},
		{"bad enum", row("r1", "nope", true) + "\n", false},
		{"string bool", `{"schema":"` + testSchemaTag + `","run_id":"r","device":"gpu","accepted":"true","digest":"` + digestOf("x") + `"}` + "\n", false},
		{"trailing bytes", good + " trailing\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gen := genWith(t, map[string]string{"fixtures/test-proto.jsonl": tc.fixture})
			res := evalGen(t, rubric, "lightweight", gen)
			if got := res.EvidenceState["fixture_row"]; got != tc.want {
				t.Errorf("fixture_row = %v, want %v (detail: %s)", got, tc.want, res.Detail["fixture_row"])
			}
		})
	}
}

func TestFixtureRow_RejectsOracleCopy(t *testing.T) {
	rubric := buildRubric(t)
	// Copy the frozen positive verbatim into the gen dir as the fixture.
	posData, err := os.ReadFile(filepath.Join(rubric.Dir, "positive.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	gen := genWith(t, map[string]string{"fixtures/test-proto.jsonl": string(posData)})
	res := evalGen(t, rubric, "lightweight", gen)
	if res.EvidenceState["fixture_row"] {
		t.Errorf("fixture_row true for a verbatim copy of the frozen oracle (detail: %s)", res.Detail["fixture_row"])
	}
}

func TestFixtureRow_RejectsSymlinkEscape(t *testing.T) {
	rubric := buildRubric(t)
	gen := t.TempDir()
	if err := os.MkdirAll(filepath.Join(gen, "fixtures"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Symlink the fixture at the frozen positive outside the gen dir.
	target := filepath.Join(rubric.Dir, "positive.jsonl")
	link := filepath.Join(gen, "fixtures", "test-proto.jsonl")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink unsupported: %v", err)
	}
	res := evalGen(t, rubric, "lightweight", gen)
	if res.EvidenceState["fixture_row"] {
		t.Errorf("fixture_row true via a symlink escaping the gen dir (detail: %s)", res.Detail["fixture_row"])
	}
}

func TestManifestHash_GamingVectors(t *testing.T) {
	rubric := buildRubric(t)
	validFixture := row("r1", "gpu", true) + "\n"

	t.Run("fabricated random hash no file", func(t *testing.T) {
		man := `{"schema_version":"` + manifestSchemaVersion + `","manifest_id":"m","run_id":"r",` +
			`"artifacts":[{"artifact_id":"a","artifact_kind":"fixture","path":"ghost.jsonl",` +
			`"sha256":"sha256:` + strings.Repeat("a", 64) + `","byte_size":42}]}`
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": validFixture,
			evidenceManifestName:        man,
		})
		res := evalGen(t, rubric, "covered", gen)
		if res.EvidenceState["artifact_manifest_hash"] {
			t.Errorf("manifest hash true for a fabricated hash with no file (detail: %s)", res.Detail["artifact_manifest_hash"])
		}
	})

	t.Run("hash size mismatch", func(t *testing.T) {
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": validFixture,
			"real.txt":                  "actual content here",
		})
		// Manifest with the right path but a wrong (lied) byte_size/hash.
		man := `{"schema_version":"` + manifestSchemaVersion + `","manifest_id":"m","run_id":"r",` +
			`"artifacts":[{"artifact_id":"a","artifact_kind":"fixture","path":"real.txt",` +
			`"sha256":"sha256:` + strings.Repeat("b", 64) + `","byte_size":99999}]}`
		if err := os.WriteFile(filepath.Join(gen, evidenceManifestName), []byte(man), 0o644); err != nil {
			t.Fatal(err)
		}
		res := evalGen(t, rubric, "covered", gen)
		if res.EvidenceState["artifact_manifest_hash"] {
			t.Errorf("manifest hash true despite hash/size disagreement (detail: %s)", res.Detail["artifact_manifest_hash"])
		}
	})

	t.Run("honest manifest passes", func(t *testing.T) {
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": validFixture,
			"real.jsonl":                row("art", "gpu", true) + "\n",
		})
		gen = withManifest(t, gen, "a", "real.jsonl")
		res := evalGen(t, rubric, "covered", gen)
		if !res.EvidenceState["artifact_manifest_hash"] {
			t.Errorf("manifest hash false for an honest manifest (detail: %s)", res.Detail["artifact_manifest_hash"])
		}
	})
}

func TestFalsifierRows_RequiresClaimNearMiss(t *testing.T) {
	rubric := buildRubric(t)

	t.Run("only passing rows", func(t *testing.T) {
		fixture := row("r1", "gpu", true) + "\n" + row("r2", "gpu", true) + "\n"
		gen := genWith(t, map[string]string{"fixtures/test-proto.jsonl": fixture})
		res := evalGen(t, rubric, "lightweight", gen)
		if res.EvidenceState["falsifier_rows"] {
			t.Error("falsifier_rows true with only passing rows")
		}
	})

	t.Run("claim near-miss present", func(t *testing.T) {
		// Same row but accepted flips false: a claim-level near-miss.
		fixture := row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n"
		gen := genWith(t, map[string]string{"fixtures/test-proto.jsonl": fixture})
		res := evalGen(t, rubric, "lightweight", gen)
		if !res.EvidenceState["falsifier_rows"] {
			t.Errorf("falsifier_rows false despite a claim near-miss (detail: %s)", res.Detail["falsifier_rows"])
		}
	})
}

// repinScope rewrites the frozen scope baseline and re-pins its hash so the
// rubric still verifies. It is a test-only helper for exercising heavy_skip.
func repinScope(t *testing.T, rubric *Rubric, scopeJSON string) {
	t.Helper()
	p := filepath.Join(rubric.Dir, "scope.json")
	if err := os.WriteFile(p, []byte(scopeJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(scopeJSON))
	rubric.Manifest.Files["scope.json"] = hex.EncodeToString(sum[:])
}

func TestHeavySkip_NarrowAndGate(t *testing.T) {
	twoIDs := `[{"id":"gpu_overlap","predicate":{"jsonl_field":"device","equals":"gpu","min_rows":1}},` +
		`{"id":"ane_overlap","predicate":{"jsonl_field":"device","equals":"ane","min_rows":1}}]`

	t.Run("strict narrow one of two", func(t *testing.T) {
		rubric := buildRubric(t)
		repinScope(t, &rubric, twoIDs)
		// Cover only gpu_overlap with a real gpu artifact => strict shrink => true.
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n",
			"gpu.jsonl":                 row("g", "gpu", true) + "\n",
			"scope-manifest.json":       `{"covered":["gpu_overlap"],"links":{"gpu_overlap":"a-gpu"}}`,
		})
		gen = withManifest(t, gen, "a-gpu", "gpu.jsonl")
		res := evalGen(t, rubric, "covered", gen)
		if !res.EvidenceState["heavy_skip_narrowed_or_cleared"] {
			t.Errorf("heavy_skip false for a strict one-of-two narrow (detail: %s)", res.Detail["heavy_skip_narrowed_or_cleared"])
		}
	})

	t.Run("wrong-scope artifact does not count", func(t *testing.T) {
		rubric := buildRubric(t)
		repinScope(t, &rubric, twoIDs)
		// Claim to cover ane_overlap but back it with a gpu artifact: predicate
		// device==ane fails => no verified id => false.
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n",
			"gpu.jsonl":                 row("g", "gpu", true) + "\n",
			"scope-manifest.json":       `{"covered":["ane_overlap"],"links":{"ane_overlap":"a-gpu"}}`,
		})
		gen = withManifest(t, gen, "a-gpu", "gpu.jsonl")
		res := evalGen(t, rubric, "covered", gen)
		if res.EvidenceState["heavy_skip_narrowed_or_cleared"] {
			t.Errorf("heavy_skip true when a gpu artifact was credited for an ane scope (detail: %s)", res.Detail["heavy_skip_narrowed_or_cleared"])
		}
	})

	t.Run("gated on manifest", func(t *testing.T) {
		rubric := buildRubric(t)
		repinScope(t, &rubric, twoIDs)
		// Valid fixture + scope-manifest but NO evidence manifest => the gate
		// (artifact_manifest_hash) is false => heavy_skip false.
		gen := genWith(t, map[string]string{
			"fixtures/test-proto.jsonl": row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n",
			"scope-manifest.json":       `{"covered":["gpu_overlap"],"links":{"gpu_overlap":"a-gpu"}}`,
		})
		res := evalGen(t, rubric, "covered", gen)
		if res.EvidenceState["heavy_skip_narrowed_or_cleared"] {
			t.Errorf("heavy_skip true without a verified manifest gate (detail: %s)", res.Detail["heavy_skip_narrowed_or_cleared"])
		}
	})
}

func TestRubricIntegrity_TamperAborts(t *testing.T) {
	rubric := buildRubric(t)
	// Tamper the frozen validator AFTER loading (manifest still pins the old hash).
	if err := os.WriteFile(filepath.Join(rubric.Dir, "validator.sh"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	e := &PaperEvaluator{
		Row:      CoverageRow{ID: "x", Status: "lightweight", FastCheck: "true", EvidenceState: map[string]bool{"fixture_row": false}},
		RepoRoot: t.TempDir(),
		Rubric:   rubric,
	}
	res, err := e.Evaluate(context.Background(), t.TempDir())
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if res.Status != EvalError {
		t.Errorf("status = %q, want EvalError on rubric tamper (a tampered frozen input must abort, not silently score)", res.Status)
	}
}

func TestBlockerTiers_CoveredNeedsMore(t *testing.T) {
	rubric := buildRubric(t)
	// Fixture good (control+falsifier) but NO manifest => covered tier blocks on
	// artifact_manifest_hash even though lightweight would PASS.
	fixture := row("r1", "gpu", true) + "\n" + row("r1", "gpu", false) + "\n"
	gen := genWith(t, map[string]string{"fixtures/test-proto.jsonl": fixture})

	light := evalGen(t, rubric, "lightweight", gen)
	if light.Verdict != "PASS" {
		t.Errorf("lightweight verdict = %q, want PASS (blockers=%v)", light.Verdict, light.Blockers)
	}
	gen2 := genWith(t, map[string]string{"fixtures/test-proto.jsonl": fixture})
	covered := evalGen(t, rubric, "covered", gen2)
	if covered.Verdict != "REVISE" {
		t.Errorf("covered verdict = %q, want REVISE (missing manifest is a blocker)", covered.Verdict)
	}
	if !contains(covered.Blockers, "artifact_manifest_hash") {
		t.Errorf("covered blockers = %v, want to include artifact_manifest_hash", covered.Blockers)
	}
}

func TestLoadCoverageRow_RealRubric(t *testing.T) {
	// Confirm we parse the real coverage-map and that the documented targets have
	// the headroom the spec claims (recompute is independent of these bools, but
	// the loader must read them correctly).
	path := realCoverageMapPath()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("coverage-map.jsonl not present at %s", path)
	}
	for _, id := range []string{"dflash-ddtree", "eagle3-vocab-translation", "gnosis-trace-compression"} {
		row, err := LoadCoverageRow(path, id)
		if err != nil {
			t.Errorf("LoadCoverageRow(%q): %v", id, err)
			continue
		}
		if len(row.EvidenceState) != 7 {
			t.Errorf("row %q has %d evidence keys, want 7", id, len(row.EvidenceState))
		}
		falseCount := 0
		for _, v := range row.EvidenceState {
			if !v {
				falseCount++
			}
		}
		if falseCount == 0 {
			t.Errorf("row %q has no headroom (all evidence true) — not a demo target", id)
		}
	}
	if _, err := LoadCoverageRow(path, "no-such-row-xyz"); err == nil {
		t.Error("LoadCoverageRow accepted a nonexistent id")
	}
}

func realCoverageMapPath() string {
	return filepath.Join("..", "paper-roadmap", "coverage-map.jsonl")
}
