package sia

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// This file holds the seven per-key honest-recompute checks. Each is a method on
// *recomputer, returns a bool (never an error — fail-closed to false on any
// problem), and records a human-readable trace into rc.detail[key] for the
// feedback agent. The guiding rule: ground truth comes from the gen dir or the
// frozen rubric, never from the agent's self-reported evidence_state.

// checkFixtureRow verifies a JSONL trace fixture exists in the gen dir and every
// non-blank line parses and validates against the frozen row schema. A single
// valid row among garbage does not pass; an oracle copy does not pass.
func (rc *recomputer) checkFixtureRow() bool {
	const key = "fixture_row"
	path, ok := rc.fixturePath()
	if !ok {
		rc.detail[key] = "no in-bounds .jsonl fixture found"
		return false
	}
	f, size, ok := openRegular(rc.genDir, path, maxFixtureBytes)
	if !ok {
		rc.detail[key] = "fixture missing, empty, oversized, or path-unsafe"
		return false
	}
	data := readAllCapped(f, maxFixtureBytes)
	f.Close()

	// Reject a verbatim copy of a frozen oracle fixture.
	sum := sha256.Sum256(data)
	if rc.eval.Rubric.oracleFileHashes[hex.EncodeToString(sum[:])] {
		rc.detail[key] = "fixture is a byte-identical copy of a frozen oracle input"
		return false
	}

	counts, rows := parseFixture(data, rc.eval.Rubric.Schema)
	rc.cacheRows(path, rows)

	if counts.valid > 0 && rc.allRowsAreOracle(rows) {
		rc.detail[key] = "every fixture row is a copy of a frozen oracle row"
		return false
	}

	rc.detail[key] = fmt.Sprintf("size=%d total=%d blank=%d parseFail=%d validateFail=%d valid=%d distinct=%d",
		size, counts.total, counts.blank, counts.parseFail, counts.validateFail, counts.valid, counts.distinct)

	return counts.parseFail == 0 && counts.validateFail == 0 && counts.valid >= 1
}

// checkValidatorCommand runs the frozen validator against the frozen positive
// (must exit 0), the frozen negative (must exit in the 1..125 clean-rejection
// band), and a tampered positive (must also reject). A script that ignores its
// input and always exits 0 fails the negative and tamper cases. The verdict is
// read only from the exit code, never from stdout.
func (rc *recomputer) checkValidatorCommand(ctx context.Context) bool {
	const key = "validator_command"
	rb := &rc.eval.Rubric

	valPath, ok := rb.frozenPath(rb.ValidatorRel)
	if !ok {
		rc.detail[key] = "frozen validator path unresolved"
		return false
	}
	posPath, ok1 := rb.frozenPath(rb.PositiveRel)
	negPath, ok2 := rb.frozenPath(rb.NegativeRel)
	if !ok1 || !ok2 {
		rc.detail[key] = "frozen positive/negative path unresolved"
		return false
	}
	if hashOfFile(posPath) == hashOfFile(negPath) {
		rc.detail[key] = "frozen positive and negative inputs are identical"
		return false
	}

	pos := rc.runValidator(ctx, valPath, posPath)
	if pos.killed || pos.launchErr || pos.code != 0 {
		rc.detail[key] = "frozen positive not cleanly accepted: " + pos.String()
		return false
	}
	neg := rc.runValidator(ctx, valPath, negPath)
	if !neg.cleanReject() {
		rc.detail[key] = "frozen negative not cleanly rejected: " + neg.String()
		return false
	}

	// Tamper: corrupt one schema-critical field of the positive and require the
	// validator to reject it. Closes "accepts any present file" stubs.
	tampered, terr := rc.tamperedPositive(posPath)
	if terr != nil {
		rc.detail[key] = "could not build tamper input: " + terr.Error()
		return false
	}
	defer os.Remove(tampered)
	tam := rc.runValidator(ctx, valPath, tampered)
	if !tam.cleanReject() {
		rc.detail[key] = "tampered positive not rejected: " + tam.String()
		return false
	}

	rc.detail[key] = "frozen pos accepted, neg+tamper rejected (exit-code-only)"
	return true
}

// checkManifestHash verifies the gen-dir evidence manifest: it must be schema-
// shaped and EVERY referenced artifact must exist in the gen dir as a regular
// file whose recomputed sha256 and byte size match the manifest, with no empty,
// trivial, duplicate, or oracle-copied artifact. Gated on validator_command.
func (rc *recomputer) checkManifestHash() bool {
	const key = "artifact_manifest_hash"
	m, ok := rc.loadManifest()
	if !ok {
		rc.detail[key] = "manifest missing or structurally invalid"
		return false
	}
	seenID, seenPath, seenHash := map[string]bool{}, map[string]bool{}, map[string]bool{}
	var totalBytes int64
	for _, a := range m.Artifacts {
		if a.ArtifactID == "" || seenID[a.ArtifactID] {
			rc.detail[key] = "missing or duplicate artifact_id"
			return false
		}
		seenID[a.ArtifactID] = true

		clean, ok := safeJoin(rc.genDir, a.Path)
		if !ok || seenPath[clean] {
			rc.detail[key] = "artifact path unsafe or duplicated: " + a.Path
			return false
		}
		seenPath[clean] = true

		if !digestRE.MatchString(a.SHA256) {
			rc.detail[key] = "artifact sha256 not sha256:<64hex>: " + a.ArtifactID
			return false
		}
		f, size, ok := openRegular(rc.genDir, a.Path, maxArtifactBytes)
		if !ok {
			rc.detail[key] = "artifact missing/empty/oversized: " + a.Path
			return false
		}
		sum, n, ok := hashFile(f, maxArtifactBytes)
		f.Close()
		if !ok {
			rc.detail[key] = "artifact unreadable or over cap: " + a.Path
			return false
		}
		if n == 0 || a.ByteSize == 0 || sum == emptySHA256 {
			rc.detail[key] = "artifact empty or trivial: " + a.Path
			return false
		}
		if sum != strings.TrimPrefix(a.SHA256, "sha256:") || a.ByteSize != n || a.ByteSize != size {
			rc.detail[key] = "hash/size disagreement: " + a.Path
			return false
		}
		if seenHash[sum] {
			rc.detail[key] = "duplicate artifact content: " + a.Path
			return false
		}
		seenHash[sum] = true
		if rc.eval.Rubric.oracleFileHashes[sum] {
			rc.detail[key] = "artifact re-presents a frozen oracle file: " + a.Path
			return false
		}
		totalBytes += n
		if totalBytes > maxManifestBytes {
			rc.detail[key] = "manifest artifact bytes exceed budget"
			return false
		}
	}
	rc.manifestHashes = seenHash
	rc.detail[key] = fmt.Sprintf("%d artifact(s) hashed and matched", len(m.Artifacts))
	return true
}

// checkModelBacked runs the frozen model-backed/opt-in command in a sealed
// scratch dir (never the gen dir) and requires it to produce a declared output
// artifact whose hash is reproducible across two independent runs and is not a
// frozen input, a pre-existing file, or empty. A manifest's self-declared
// model_id alone never satisfies this. Gated on validator_command.
func (rc *recomputer) checkModelBacked(ctx context.Context) bool {
	const key = "model_backed_or_opt_in_command"
	rb := &rc.eval.Rubric
	if len(rb.ModelCommand) == 0 || rb.ModelOutName == "" {
		rc.detail[key] = "no frozen model-backed command configured"
		return false
	}
	forbidden := rc.forbiddenHashes()

	h1, ok := rc.produceModelArtifact(ctx, forbidden)
	if !ok {
		rc.detail[key] = "frozen command did not produce a valid artifact"
		return false
	}
	h2, ok := rc.produceModelArtifact(ctx, forbidden)
	if !ok {
		rc.detail[key] = "second model run failed to reproduce artifact"
		return false
	}
	if h1 != h2 {
		rc.detail[key] = "model artifact not reproducible across runs"
		return false
	}
	rc.detail[key] = "frozen command produced a reproducible non-forbidden artifact"
	return true
}

// checkControlRows requires at least one fixture row the frozen validator
// accepts as a positive control, excluding any row that copies a frozen oracle
// positive. Gated on validator_command.
func (rc *recomputer) checkControlRows() bool {
	const key = "control_rows"
	rows, ok := rc.fixtureRows()
	if !ok {
		rc.detail[key] = "no valid fixture rows to score"
		return false
	}
	oracle := rc.eval.Rubric.oracleRowHashes()
	accepted := map[string]bool{}
	for _, r := range rows {
		if oracle[r.hash] {
			continue
		}
		if rc.validatorAcceptsRow(r) {
			accepted[r.hash] = true
		}
	}
	rc.controlHashes = accepted
	rc.detail[key] = fmt.Sprintf("%d distinct accepted control row(s)", len(accepted))
	return len(accepted) >= 1
}

// checkFalsifierRows requires at least one fixture row the frozen validator
// rejects as a claim-level near-miss (a true negative control), distinct from
// the accepted control rows and from any frozen negative. A reject that is only
// a schema break, or an indeterminate timeout, does not count. Gated on
// validator_command.
func (rc *recomputer) checkFalsifierRows() bool {
	const key = "falsifier_rows"
	rows, ok := rc.fixtureRows()
	if !ok {
		rc.detail[key] = "no valid fixture rows to score"
		return false
	}
	if len(rows) < 2 {
		rc.detail[key] = "need >=2 rows (a control and a falsifier)"
		return false
	}
	oracleNeg := rc.eval.Rubric.oracleRowHashes()
	rejected := map[string]bool{}
	for _, r := range rows {
		if oracleNeg[r.hash] {
			continue // agent must author its own falsifier, not alias ours
		}
		if rc.controlHashes[r.hash] {
			continue // a row cannot be both control and falsifier
		}
		if rc.validatorRejectsRowAsClaim(r) {
			rejected[r.hash] = true
		}
	}
	// Require a falsifier that is a near-miss of an accepted control: identical
	// in every non-claim field, differing only in a claim field.
	if !rc.hasNearMiss(rows, rejected) {
		rc.detail[key] = fmt.Sprintf("%d claim-rejected row(s) but none a near-miss of a control", len(rejected))
		return false
	}
	rc.detail[key] = fmt.Sprintf("%d claim-level falsifier near-miss row(s)", len(rejected))
	return true
}

// checkHeavySkip requires the agent's narrowed scope to be backed by a new
// passing artifact for the previously-skipped scope, judged against the frozen
// baseline scope predicates — never by inspecting (shrinking) prose. Gated on
// validator_command and artifact_manifest_hash.
func (rc *recomputer) checkHeavySkip() bool {
	const key = "heavy_skip_narrowed_or_cleared"
	rb := &rc.eval.Rubric
	baseline, ok := rc.loadScopeBaseline()
	if !ok {
		rc.detail[key] = "no frozen heavy-skip baseline configured"
		return false
	}
	claim, ok := rc.loadScopeClaim()
	if !ok {
		rc.detail[key] = "no parseable scope-manifest in gen dir"
		return false
	}
	baseIDs := map[string]scopePredicate{}
	for _, s := range baseline {
		baseIDs[s.ID] = s.Predicate
	}
	verified := map[string]bool{}
	for _, id := range claim.Covered {
		pred, known := baseIDs[id]
		if !known {
			rc.detail[key] = "scope claim references unknown id: " + id
			return false // inventing ids is a hard fail
		}
		artID := claim.Links[id]
		if artID == "" {
			continue
		}
		if rc.scopeArtifactSatisfies(artID, pred) {
			verified[id] = true
		}
	}
	_ = rb
	if len(verified) == 0 {
		rc.detail[key] = "no covered scope backed by a verified artifact"
		return false
	}
	if len(verified) >= len(baseIDs) && len(baseIDs) > 0 {
		// "cleared": every baseline id is verifiably covered — allowed only with
		// real artifacts, which the loop above already required.
		rc.detail[key] = fmt.Sprintf("all %d scope id(s) verifiably covered", len(baseIDs))
		return true
	}
	rc.detail[key] = fmt.Sprintf("%d/%d scope id(s) verifiably narrowed", len(verified), len(baseIDs))
	return len(verified) >= 1 && len(verified) < len(baseIDs)
}

// --- shared recompute helpers ---

// fixturePath returns the gen-dir-relative path to the fixture: the first
// .jsonl entry in row.Examples, else the conventional fixtures/<id>.jsonl.
func (rc *recomputer) fixturePath() (string, bool) {
	for _, ex := range rc.eval.Row.Examples {
		if strings.HasSuffix(ex, ".jsonl") {
			base := filepath.Base(ex)
			cand := filepath.Join("fixtures", base)
			if _, ok := safeJoin(rc.genDir, cand); ok {
				if _, err := os.Lstat(filepath.Join(rc.genDir, cand)); err == nil {
					return cand, true
				}
			}
		}
	}
	cand := filepath.Join("fixtures", rc.eval.Row.ID+".jsonl")
	if _, ok := safeJoin(rc.genDir, cand); ok {
		if _, err := os.Lstat(filepath.Join(rc.genDir, cand)); err == nil {
			return cand, true
		}
	}
	return "", false
}

// cacheRows memoizes validated rows for a fixture path.
func (rc *recomputer) cacheRows(path string, rows []fixtureRow) {
	if rc.rowCache == nil {
		rc.rowCache = map[string][]fixtureRow{}
	}
	rc.rowCache[path] = rows
}

// fixtureRows returns the cached validated rows, parsing the fixture once if the
// fixture_row check has not already populated the cache.
func (rc *recomputer) fixtureRows() ([]fixtureRow, bool) {
	path, ok := rc.fixturePath()
	if !ok {
		return nil, false
	}
	if rc.rowCache != nil {
		if rows, ok := rc.rowCache[path]; ok {
			return rows, len(rows) > 0
		}
	}
	f, _, ok := openRegular(rc.genDir, path, maxFixtureBytes)
	if !ok {
		return nil, false
	}
	data := readAllCapped(f, maxFixtureBytes)
	f.Close()
	_, rows := parseFixture(data, rc.eval.Rubric.Schema)
	rc.cacheRows(path, rows)
	return rows, len(rows) > 0
}

// allRowsAreOracle reports whether every row hashes to a frozen oracle row.
func (rc *recomputer) allRowsAreOracle(rows []fixtureRow) bool {
	if len(rows) == 0 {
		return false
	}
	oracle := rc.eval.Rubric.oracleRowHashes()
	for _, r := range rows {
		if !oracle[r.hash] {
			return false
		}
	}
	return true
}

// validatorAcceptsRow runs the frozen validator on a single-row fixture written
// to a private temp file and reports whether it exits 0.
func (rc *recomputer) validatorAcceptsRow(r fixtureRow) bool {
	v := rc.runValidatorOnRow(r)
	return !v.killed && !v.launchErr && v.code == 0
}

// validatorRejectsRowAsClaim reports whether the validator rejects the row in
// the clean 1..125 band (treated as a claim-level rejection). A kill or launch
// failure is indeterminate and does not count.
func (rc *recomputer) validatorRejectsRowAsClaim(r fixtureRow) bool {
	return rc.runValidatorOnRow(r).cleanReject()
}

// hasNearMiss reports whether some rejected row differs from an accepted control
// only in one or more claim fields (a constrained near-miss), not in structure.
func (rc *recomputer) hasNearMiss(rows []fixtureRow, rejected map[string]bool) bool {
	claimFields := rc.eval.Rubric.Schema.ClaimFields
	if len(claimFields) == 0 {
		// Without declared claim fields, any distinct claim-rejected row counts.
		return len(rejected) >= 1
	}
	var controls, fals []fixtureRow
	for _, r := range rows {
		if rc.controlHashes[r.hash] {
			controls = append(controls, r)
		}
		if rejected[r.hash] {
			fals = append(fals, r)
		}
	}
	for _, f := range fals {
		for _, c := range controls {
			if f.hash == c.hash {
				continue
			}
			if nearMiss(c.obj, f.obj, claimFields) {
				return true
			}
		}
	}
	return false
}

// nearMiss reports whether a and b agree on every non-claim field and differ on
// at least one claim field.
func nearMiss(a, b map[string]json.RawMessage, claimFields []string) bool {
	claim := map[string]bool{}
	for _, c := range claimFields {
		claim[c] = true
	}
	// Every non-claim field present in either must match byte-for-byte.
	keys := map[string]bool{}
	for k := range a {
		keys[k] = true
	}
	for k := range b {
		keys[k] = true
	}
	differsOnClaim := false
	for k := range keys {
		eq := bytesEqualJSON(a[k], b[k])
		if claim[k] {
			if !eq {
				differsOnClaim = true
			}
			continue
		}
		if !eq {
			return false // differs outside the claim fields => not a near-miss
		}
	}
	return differsOnClaim
}

func bytesEqualJSON(a, b json.RawMessage) bool {
	return string(a) == string(b)
}

// --- validator execution ---

// validatorRun is the outcome of one frozen-validator invocation.
type validatorRun struct {
	code      int
	killed    bool
	launchErr bool
}

func (v validatorRun) cleanReject() bool {
	return !v.killed && !v.launchErr && v.code >= 1 && v.code <= 125
}

func (v validatorRun) String() string {
	switch {
	case v.launchErr:
		return "launch failed"
	case v.killed:
		return "killed/timed out"
	default:
		return fmt.Sprintf("exit %d", v.code)
	}
}

// runValidator runs the frozen validator (copied into a private scratch byte for
// byte) against inputPath with a bounded timeout, scrubbed env, and scratch cwd.
func (rc *recomputer) runValidator(ctx context.Context, validatorPath, inputPath string) validatorRun {
	scratch, err := os.MkdirTemp("", "paperbench-val-")
	if err != nil {
		return validatorRun{launchErr: true}
	}
	defer os.RemoveAll(scratch)

	localVal := filepath.Join(scratch, "validator")
	if err := copyFileMode(validatorPath, localVal, 0o500); err != nil {
		return validatorRun{launchErr: true}
	}

	cctx, cancel := context.WithTimeout(ctx, validatorTimeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, rc.eval.Rubric.Interpreter, localVal, inputPath)
	cmd.Dir = scratch
	cmd.Env = scrubbedEnv()
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()

	if cctx.Err() == context.DeadlineExceeded {
		return validatorRun{killed: true}
	}
	if runErr == nil {
		return validatorRun{code: 0}
	}
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		if exitErr.ExitCode() < 0 {
			return validatorRun{killed: true} // signal kill
		}
		return validatorRun{code: exitErr.ExitCode()}
	}
	return validatorRun{launchErr: true}
}

// runValidatorOnRow writes the row to a private temp .jsonl and runs the frozen
// validator on it.
func (rc *recomputer) runValidatorOnRow(r fixtureRow) validatorRun {
	tmp, err := os.CreateTemp("", "paperbench-row-*.jsonl")
	if err != nil {
		return validatorRun{launchErr: true}
	}
	defer os.Remove(tmp.Name())
	row, _ := json.Marshal(r.obj)
	tmp.Write(row)
	tmp.Write([]byte("\n"))
	tmp.Close()

	valPath, ok := rc.eval.Rubric.frozenPath(rc.eval.Rubric.ValidatorRel)
	if !ok {
		return validatorRun{launchErr: true}
	}
	return rc.runValidator(context.Background(), valPath, tmp.Name())
}

// tamperedPositive writes a copy of the frozen positive with one schema-critical
// field corrupted, returning the temp path the caller must remove.
func (rc *recomputer) tamperedPositive(posPath string) (string, error) {
	data, err := os.ReadFile(posPath)
	if err != nil {
		return "", err
	}
	lines := splitJSONLines(stripBOM(data))
	if len(lines) == 0 {
		return "", fmt.Errorf("positive fixture empty")
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(lines[0], &obj); err != nil {
		return "", fmt.Errorf("positive row unparseable")
	}
	// Drop a required field to break the schema/claim.
	tag := rc.eval.Rubric.Schema.TagField
	corrupted := false
	for _, f := range rc.eval.Rubric.Schema.Required {
		if f == tag {
			continue
		}
		if _, ok := obj[f]; ok {
			delete(obj, f)
			corrupted = true
			break
		}
	}
	if !corrupted {
		// Fall back to corrupting an enum field to an out-of-range value.
		for f := range rc.eval.Rubric.Schema.Enums {
			obj[f] = json.RawMessage(`"__paperbench_invalid__"`)
			corrupted = true
			break
		}
	}
	if !corrupted {
		return "", fmt.Errorf("no field available to tamper")
	}
	tmp, err := os.CreateTemp("", "paperbench-tamper-*.jsonl")
	if err != nil {
		return "", err
	}
	row, _ := json.Marshal(obj)
	tmp.Write(row)
	tmp.Write([]byte("\n"))
	tmp.Close()
	return tmp.Name(), nil
}

// --- manifest + model-backed helpers ---

// evidenceManifest mirrors the evidence-manifest.schema.json shape the evaluator
// recomputes against.
type evidenceManifest struct {
	SchemaVersion string             `json:"schema_version"`
	ManifestID    string             `json:"manifest_id"`
	RunID         string             `json:"run_id"`
	Artifacts     []manifestArtifact `json:"artifacts"`
}

type manifestArtifact struct {
	ArtifactID   string `json:"artifact_id"`
	ArtifactKind string `json:"artifact_kind"`
	Path         string `json:"path"`
	SHA256       string `json:"sha256"`
	ByteSize     int64  `json:"byte_size"`
	ModelID      string `json:"model_id"`
	Revision     string `json:"revision"`
}

// loadManifest reads and structurally validates the gen-dir evidence manifest.
func (rc *recomputer) loadManifest() (evidenceManifest, bool) {
	f, _, ok := openRegular(rc.genDir, evidenceManifestName, maxManifestBytes)
	if !ok {
		return evidenceManifest{}, false
	}
	data := readAllCapped(f, maxManifestBytes)
	f.Close()
	var m evidenceManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return evidenceManifest{}, false
	}
	if m.SchemaVersion != manifestSchemaVersion || m.ManifestID == "" || m.RunID == "" {
		return evidenceManifest{}, false
	}
	if len(m.Artifacts) < 1 || len(m.Artifacts) > maxArtifacts {
		return evidenceManifest{}, false
	}
	return m, true
}

// forbiddenHashes is the set of hashes a freshly produced model artifact may not
// equal: every frozen rubric file and every pre-existing gen-dir file, plus the
// empty hash.
func (rc *recomputer) forbiddenHashes() map[string]bool {
	forbidden := map[string]bool{emptySHA256: true}
	for h := range rc.eval.Rubric.oracleFileHashes {
		forbidden[h] = true
	}
	_ = filepath.Walk(rc.genDir, func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi.IsDir() || !fi.Mode().IsRegular() {
			return nil
		}
		if fi.Size() == 0 || fi.Size() > maxArtifactBytes {
			return nil
		}
		f, err := os.Open(p)
		if err != nil {
			return nil
		}
		defer f.Close()
		if sum, _, ok := hashFile(f, maxArtifactBytes); ok {
			forbidden[sum] = true
		}
		return nil
	})
	return forbidden
}

// produceModelArtifact runs the frozen model command in a fresh sealed scratch
// dir and returns the produced artifact's hash, requiring it to be a real,
// non-forbidden, in-bounds file.
func (rc *recomputer) produceModelArtifact(ctx context.Context, forbidden map[string]bool) (string, bool) {
	rb := &rc.eval.Rubric
	scratch, err := os.MkdirTemp("", "paperbench-mb-")
	if err != nil {
		return "", false
	}
	defer os.RemoveAll(scratch)

	outPath := filepath.Join(scratch, rb.ModelOutName)
	if _, err := os.Lstat(outPath); err == nil {
		return "", false // must not pre-exist
	}
	cctx, cancel := context.WithTimeout(ctx, validatorTimeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, rb.ModelCommand[0], rb.ModelCommand[1:]...)
	cmd.Dir = scratch
	cmd.Env = scrubbedEnv()
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil || cctx.Err() != nil {
		return "", false
	}
	f, size, ok := openRegular(scratch, rb.ModelOutName, maxArtifactBytes)
	if !ok {
		return "", false
	}
	sum, n, ok := hashFile(f, maxArtifactBytes)
	f.Close()
	if !ok || n != size || n < rb.ModelMinBytes || forbidden[sum] {
		return "", false
	}
	return sum, true
}

// --- heavy-skip scope helpers ---

type scopeEntry struct {
	ID        string         `json:"id"`
	Predicate scopePredicate `json:"predicate"`
}

// scopePredicate is a frozen per-scope coverage test against a JSONL artifact:
// at least MinRows rows whose Field equals Equals.
type scopePredicate struct {
	JSONLField string `json:"jsonl_field"`
	Equals     string `json:"equals"`
	MinRows    int    `json:"min_rows"`
}

type scopeClaim struct {
	Covered []string          `json:"covered"`
	Links   map[string]string `json:"links"` // scope id -> artifact_id
}

// loadScopeBaseline reads the frozen heavy-skip scope baseline.
func (rc *recomputer) loadScopeBaseline() ([]scopeEntry, bool) {
	rel := rc.eval.Rubric.ScopeRel
	if rel == "" {
		return nil, false
	}
	p, ok := rc.eval.Rubric.frozenPath(rel)
	if !ok {
		return nil, false
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, false
	}
	var entries []scopeEntry
	if err := json.Unmarshal(data, &entries); err != nil || len(entries) == 0 {
		return nil, false
	}
	return entries, true
}

// loadScopeClaim reads the agent's scope-manifest from the gen dir.
func (rc *recomputer) loadScopeClaim() (scopeClaim, bool) {
	f, _, ok := openRegular(rc.genDir, scopeManifestName, maxFixtureBytes)
	if !ok {
		return scopeClaim{}, false
	}
	data := readAllCapped(f, maxFixtureBytes)
	f.Close()
	var c scopeClaim
	if err := json.Unmarshal(data, &c); err != nil || len(c.Covered) == 0 {
		return scopeClaim{}, false
	}
	return c, true
}

// scopeArtifactSatisfies looks up the artifact by id in the gen-dir manifest,
// verifies its hash honestly, and evaluates the frozen predicate against it.
func (rc *recomputer) scopeArtifactSatisfies(artifactID string, pred scopePredicate) bool {
	m, ok := rc.loadManifest()
	if !ok {
		return false
	}
	var art *manifestArtifact
	for i := range m.Artifacts {
		if m.Artifacts[i].ArtifactID == artifactID {
			art = &m.Artifacts[i]
			break
		}
	}
	if art == nil {
		return false
	}
	f, size, ok := openRegular(rc.genDir, art.Path, maxArtifactBytes)
	if !ok {
		return false
	}
	data := readAllCapped(f, maxArtifactBytes)
	f.Close()
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != strings.TrimPrefix(art.SHA256, "sha256:") || art.ByteSize != size {
		return false
	}
	if rc.eval.Rubric.oracleFileHashes[hex.EncodeToString(sum[:])] {
		return false
	}
	return predicateHolds(data, pred)
}

// predicateHolds reports whether at least pred.MinRows JSONL rows in data have
// pred.JSONLField equal to pred.Equals.
func predicateHolds(data []byte, pred scopePredicate) bool {
	if pred.JSONLField == "" {
		return true // a presence-only scope is satisfied by a verified artifact
	}
	min := pred.MinRows
	if min < 1 {
		min = 1
	}
	matched := 0
	for _, line := range splitJSONLines(stripBOM(data)) {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(line, &obj); err != nil {
			continue
		}
		raw, ok := obj[pred.JSONLField]
		if !ok {
			continue
		}
		var v string
		if err := json.Unmarshal(raw, &v); err == nil && v == pred.Equals {
			matched++
		}
	}
	return matched >= min
}

// --- small file helpers ---

// hashOfFile returns the lowercase hex sha256 of the file at path, or "" on
// error (so two unreadable files do not compare equal to each other usefully).
func hashOfFile(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, io.LimitReader(f, maxFixtureBytes)); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// copyFileMode copies src to dst with the given mode.
func copyFileMode(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

var _ = sort.Strings
