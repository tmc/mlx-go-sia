package sia

import (
	"bufio"
	"bytes"
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
	"time"
)

// PaperEvaluator scores one generation's research-prototype implementation
// against a paper-roadmap coverage-map row. It implements [Evaluator].
//
// The contribution is honest recompute: rather than trust the agent's
// self-reported evidence_state (which an agent under optimization pressure will
// game — the "Coupled co-evolutionary Goodhart" failure mode), the evaluator
// independently re-establishes each of the seven evidence booleans from ground
// truth in the generation directory, using a frozen, evaluator-owned rubric the
// agent cannot edit. A self-reported evidence_state in the gen dir is never read.
//
// The verdict is categorical (PASS|REVISE) and gates on fast_check plus the
// blocker keys for the row's status tier. A weighted advisory score is surfaced
// for trend-watching but never decides the verdict, avoiding the
// "pickier reviewer scores lower" failure mode. See [CoverageRow] for the rubric
// row and [EvidenceResults] for the results.json schema.
//
// The zero value is not usable; set Row, RepoRoot, and Rubric.
type PaperEvaluator struct {
	// Row is the target prototype's rubric row (loaded from coverage-map.jsonl).
	Row CoverageRow
	// RepoRoot is the working directory fast_check runs in. It is the operator's
	// trusted checkout, distinct from the agent-writable gen dir.
	RepoRoot string
	// Rubric is the frozen, evaluator-owned context: the pristine validator,
	// schemas, and positive/negative inputs the agent cannot edit. Its integrity
	// is verified against [Rubric.Manifest] at the start of every Evaluate.
	Rubric Rubric
	// FastCheckTimeout bounds the fast_check command (0 uses DefaultFastTimeout).
	FastCheckTimeout time.Duration
	// now supplies the clock for results.json timestamps; nil uses time.Now.
	now func() time.Time
}

// recompute caps and timeouts. Every cap fails closed: a breach makes the key
// false (or skips the artifact), never true-on-error.
const (
	maxFixtureBytes  = 8 << 20  // 8 MiB per JSONL fixture
	maxArtifactBytes = 64 << 20 // 64 MiB per manifest artifact
	maxManifestBytes = 256 << 20
	maxArtifacts     = 256
	maxLineBytes     = 1 << 20 // 1 MiB per JSONL line

	validatorTimeout   = 20 * time.Second
	DefaultFastTimeout = 5 * time.Minute

	// emptySHA256 is sha256 of zero bytes; a "real" artifact never hashes to it.
	emptySHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

	manifestSchemaVersion = "mlx_go_evidence_manifest.v0"
	evidenceManifestName  = "evidence-manifest.json"
	scopeManifestName     = "scope-manifest.json"
)

// evidenceKeys are the seven evidence_state booleans, in score order. They match
// the coverage-map.jsonl schema exactly.
var evidenceKeys = []string{
	"fixture_row",
	"validator_command",
	"artifact_manifest_hash",
	"model_backed_or_opt_in_command",
	"control_rows",
	"falsifier_rows",
	"heavy_skip_narrowed_or_cleared",
}

// keyWeights weights the advisory score. Execution and falsification keys
// dominate the spoofable structural keys. The weights sum to 1.0.
var keyWeights = map[string]float64{
	"fixture_row":                    0.05,
	"validator_command":              0.25,
	"artifact_manifest_hash":         0.20,
	"model_backed_or_opt_in_command": 0.20,
	"control_rows":                   0.10,
	"falsifier_rows":                 0.15,
	"heavy_skip_narrowed_or_cleared": 0.05,
}

// keyGates is the prerequisite chain: a gated key is worth zero (and forced
// false) unless every gate is already true. No credit for hashes or falsifiers
// without a working validator.
var keyGates = map[string][]string{
	"artifact_manifest_hash":         {"validator_command"},
	"model_backed_or_opt_in_command": {"validator_command"},
	"control_rows":                   {"validator_command"},
	"falsifier_rows":                 {"validator_command"},
	"heavy_skip_narrowed_or_cleared": {"validator_command", "artifact_manifest_hash"},
}

// blockerTiers names, per coverage-map status tier, the evidence keys that gate
// the categorical verdict. A row PASSes only when fast_check succeeds and every
// blocker for its tier honest-recomputes to true.
var blockerTiers = map[string][]string{
	"lightweight": {"validator_command", "fixture_row"},
	"covered":     {"validator_command", "fixture_row", "artifact_manifest_hash", "control_rows", "falsifier_rows"},
	"external":    {"validator_command"},
	"source_gap":  {},
}

// CoverageRow is one paper-roadmap coverage-map.jsonl row: the falsifiable claim
// a prototype must support and the machine-checkable evidence rubric. The
// EvidenceState booleans are the agent's self-report and are deliberately not
// trusted by the evaluator; they are loaded only to seed the rubric's key set.
type CoverageRow struct {
	ID            string          `json:"id"`
	Source        string          `json:"source"`
	Claim         string          `json:"claim"`
	Status        string          `json:"status"`
	Coverage      string          `json:"coverage"`
	Examples      []string        `json:"examples"`
	FastCheck     string          `json:"fast_check"`
	HeavySkip     string          `json:"heavy_skip"`
	EvidenceState map[string]bool `json:"evidence_state"`
	Gap           string          `json:"gap"`
}

// LoadCoverageRow reads coverage-map.jsonl at path and returns the row with the
// given id. It is an error if the file cannot be read, contains no such id, or
// the matched row carries no evidence_state (an unscored row has no rubric).
func LoadCoverageRow(path, id string) (CoverageRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return CoverageRow{}, fmt.Errorf("open coverage map: %w", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var row CoverageRow
		if err := json.Unmarshal(line, &row); err != nil {
			continue // tolerate non-row lines; only the target id matters
		}
		if row.ID != id {
			continue
		}
		if len(row.EvidenceState) == 0 {
			return CoverageRow{}, fmt.Errorf("coverage row %q has no evidence_state (unscored)", id)
		}
		return row, nil
	}
	if err := sc.Err(); err != nil {
		return CoverageRow{}, fmt.Errorf("scan coverage map: %w", err)
	}
	return CoverageRow{}, fmt.Errorf("coverage row %q not found in %s", id, path)
}

// EvidenceResults is the results.json the evaluator writes into the gen dir. The
// orchestrator's metric extractor reads advisory_score as a top-level scalar
// (the "number goes up" curve) and renders the whole document into the feedback
// agent's prompt, so blockers and notes drive the next generation's repairs.
type EvidenceResults struct {
	Verdict       string            `json:"verdict"`        // PASS | REVISE
	AdvisoryScore float64           `json:"advisory_score"` // weighted, advisory only
	FastCheckOK   bool              `json:"fast_check_ok"`
	EvidenceState map[string]bool   `json:"evidence_state"` // honest-recomputed
	Blockers      []string          `json:"blockers"`       // unmet blocker keys for the tier
	Notes         string            `json:"notes"`          // human-readable gap summary
	RowID         string            `json:"row_id"`
	Status        string            `json:"status"`
	Detail        map[string]string `json:"evidence_detail"` // per-key recompute trace
}

// Evaluate scores genDir against the rubric row. It returns
// EvalResult{Status: EvalSuccess} with results.json written for any completed
// scoring run — a REVISE verdict is data in results.json, not a Go error. A Go
// EvalError is reserved for the evaluator being unable to run at all: a frozen
// rubric whose integrity check fails, a cancelled context, or an unwritable
// results.json.
func (e *PaperEvaluator) Evaluate(ctx context.Context, genDir string) (EvalResult, error) {
	if err := ctx.Err(); err != nil {
		return EvalResult{Status: EvalError, Reason: "context cancelled before scoring"}, nil
	}
	// Integrity-gate the frozen rubric before trusting any frozen input.
	if err := e.Rubric.verify(); err != nil {
		return EvalResult{Status: EvalError, Reason: fmt.Sprintf("rubric integrity: %v", err)}, nil
	}

	rc := &recomputer{
		eval:   e,
		genDir: genDir,
		state:  make(map[string]bool, len(evidenceKeys)),
		detail: make(map[string]string, len(evidenceKeys)),
	}

	fastOK, fastNote := e.runFastCheck(ctx)
	rc.detail["fast_check"] = fastNote

	// Recompute in dependency order so gates are resolved before gated keys.
	for _, key := range evidenceKeys {
		if blocked := rc.gateUnmet(key); blocked != "" {
			rc.state[key] = false
			rc.detail[key] = "gate not met: " + blocked
			continue
		}
		rc.state[key] = rc.recompute(ctx, key)
	}

	score := advisoryScore(rc.state)
	blockers := unmetBlockers(e.Row.Status, rc.state)
	verdict := "REVISE"
	if fastOK && len(blockers) == 0 {
		verdict = "PASS"
	}

	res := EvidenceResults{
		Verdict:       verdict,
		AdvisoryScore: score,
		FastCheckOK:   fastOK,
		EvidenceState: rc.state,
		Blockers:      blockers,
		Notes:         rc.notes(verdict, blockers),
		RowID:         e.Row.ID,
		Status:        e.Row.Status,
		Detail:        rc.detail,
	}

	resultsPath := filepath.Join(genDir, NameResultsJSON)
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return EvalResult{Status: EvalError, Reason: fmt.Sprintf("marshal results: %v", err)}, nil
	}
	if err := os.WriteFile(resultsPath, append(data, '\n'), 0o644); err != nil {
		return EvalResult{Status: EvalError, Reason: fmt.Sprintf("write results.json: %v", err)}, nil
	}
	return EvalResult{
		Status:      EvalSuccess,
		ResultsPath: resultsPath,
		Output:      fmt.Sprintf("verdict=%s advisory=%.3f blockers=%s", verdict, score, joinSorted(blockers)),
	}, nil
}

// runFastCheck runs the row's fast_check in RepoRoot under a bounded timeout
// with a scrubbed environment, reporting success purely from the exit code. A
// timeout, signal kill, or launch failure is not success.
func (e *PaperEvaluator) runFastCheck(ctx context.Context) (ok bool, note string) {
	if strings.TrimSpace(e.Row.FastCheck) == "" {
		return false, "fast_check empty"
	}
	timeout := e.FastCheckTimeout
	if timeout <= 0 {
		timeout = DefaultFastTimeout
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, "/bin/sh", "-c", e.Row.FastCheck)
	cmd.Dir = e.RepoRoot
	cmd.Env = scrubbedEnv()
	var buf cappedBuffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	runErr := cmd.Run()
	if cctx.Err() == context.DeadlineExceeded {
		return false, "fast_check timed out"
	}
	if runErr == nil {
		return true, "fast_check exit 0"
	}
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		return false, fmt.Sprintf("fast_check exit %d", exitErr.ExitCode())
	}
	return false, "fast_check launch failed: " + runErr.Error()
}

// recomputer carries the per-Evaluate state: the resolved gen dir, the booleans
// recomputed so far (for gate checks), per-key detail strings, and a cache of
// validated fixture rows so dependent keys never re-trust the agent's file.
type recomputer struct {
	eval   *PaperEvaluator
	genDir string
	state  map[string]bool
	detail map[string]string

	// rowCache memoizes parsed+validated fixture rows by fixture path so a
	// dependent key never re-reads (and re-trusts) the agent's file.
	rowCache map[string][]fixtureRow
	// controlHashes is the set of accepted control-row hashes, filled by the
	// control_rows check and read by falsifier_rows to keep the sets disjoint.
	controlHashes map[string]bool
	// manifestHashes is the set of verified artifact hashes from the manifest
	// check, available to later keys.
	manifestHashes map[string]bool
}

// gateUnmet reports the first gate of key that has not recomputed to true, or ""
// when all gates are satisfied.
func (rc *recomputer) gateUnmet(key string) string {
	for _, g := range keyGates[key] {
		if !rc.state[g] {
			return g
		}
	}
	return ""
}

// recompute dispatches to the per-key honest check. Each returns false on any
// error, timeout, or cap breach (fail-closed) and records a detail string.
func (rc *recomputer) recompute(ctx context.Context, key string) bool {
	switch key {
	case "fixture_row":
		return rc.checkFixtureRow()
	case "validator_command":
		return rc.checkValidatorCommand(ctx)
	case "artifact_manifest_hash":
		return rc.checkManifestHash()
	case "model_backed_or_opt_in_command":
		return rc.checkModelBacked(ctx)
	case "control_rows":
		return rc.checkControlRows()
	case "falsifier_rows":
		return rc.checkFalsifierRows()
	case "heavy_skip_narrowed_or_cleared":
		return rc.checkHeavySkip()
	default:
		rc.detail[key] = "unknown key"
		return false
	}
}

// notes summarizes the verdict for the feedback agent.
func (rc *recomputer) notes(verdict string, blockers []string) string {
	if verdict == "PASS" {
		return fmt.Sprintf("PASS: fast_check passed and all blockers for tier %q satisfied.", rc.eval.Row.Status)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "REVISE: close these blockers for tier %q: %s. ", rc.eval.Row.Status, joinSorted(blockers))
	for _, k := range evidenceKeys {
		if d := rc.detail[k]; d != "" {
			fmt.Fprintf(&b, "[%s: %s] ", k, d)
		}
	}
	return strings.TrimSpace(b.String())
}

// advisoryScore is the weighted sum of recomputed-true keys. Gated keys already
// recomputed false when their gate failed, so they contribute zero — no manual
// zeroing is needed here.
func advisoryScore(state map[string]bool) float64 {
	var sum float64
	for k, w := range keyWeights {
		if state[k] {
			sum += w
		}
	}
	// The weights sum to 1.0 in exact arithmetic; clamp to absorb float drift so
	// the surfaced score never reads above 1.0.
	if sum > 1.0 {
		sum = 1.0
	}
	return sum
}

// unmetBlockers returns the blocker keys for status that are not true. An
// unknown status tier defaults to the lightweight blocker set.
func unmetBlockers(status string, state map[string]bool) []string {
	keys, ok := blockerTiers[status]
	if !ok {
		keys = blockerTiers["lightweight"]
	}
	var unmet []string
	for _, k := range keys {
		if !state[k] {
			unmet = append(unmet, k)
		}
	}
	return unmet
}

// scrubbedEnv is the fixed allowlist environment for every subprocess: no
// LD_PRELOAD/DYLD_*/PYTHON*/BASH_ENV inheritance that could divert a frozen
// validator or fast_check.
func scrubbedEnv() []string {
	return []string{"PATH=/usr/bin:/bin", "LC_ALL=C"}
}

// cappedBuffer is an io.Writer that retains at most 64 KiB, discarding the rest,
// so a log-bombing subprocess cannot exhaust memory.
type cappedBuffer struct {
	buf bytes.Buffer
}

func (c *cappedBuffer) Write(p []byte) (int, error) {
	const cap = 64 << 10
	if room := cap - c.buf.Len(); room > 0 {
		if len(p) > room {
			c.buf.Write(p[:room])
		} else {
			c.buf.Write(p)
		}
	}
	return len(p), nil // report full consumption so the process is not blocked
}

func (c *cappedBuffer) String() string { return c.buf.String() }

// safeJoin resolves rel against base and returns the cleaned absolute path only
// when it provably stays inside base: no absolute rel, no ".." escape, and no
// symlink component that resolves outside base. ok is false otherwise, and the
// caller treats a false as the file being absent (fail-closed). The returned
// path is suitable for opening with O_NOFOLLOW on the leaf.
func safeJoin(base, rel string) (string, bool) {
	if rel == "" || filepath.IsAbs(rel) {
		return "", false
	}
	clean := filepath.Clean(filepath.Join(base, rel))
	r, err := filepath.Rel(base, clean)
	if err != nil || r == "." || strings.HasPrefix(r, "..") {
		return "", false
	}
	// Reject any symlink in the component chain that escapes base.
	baseReal, err := filepath.EvalSymlinks(base)
	if err != nil {
		return "", false
	}
	real, err := filepath.EvalSymlinks(clean)
	if err != nil {
		// A not-yet-existing leaf is fine if its parent stays in base.
		parentReal, perr := filepath.EvalSymlinks(filepath.Dir(clean))
		if perr != nil || !within(baseReal, parentReal) {
			return "", false
		}
		return clean, true
	}
	if !within(baseReal, real) {
		return "", false
	}
	return clean, true
}

// within reports whether path is base itself or lies beneath it.
func within(base, path string) bool {
	if base == path {
		return true
	}
	return strings.HasPrefix(path, base+string(os.PathSeparator))
}

// openRegular opens a regular file under base for reading without following a
// symlinked leaf, enforcing the size cap. It returns the open file and its size,
// or ok=false (the caller treats this as absent).
func openRegular(base, rel string, maxBytes int64) (*os.File, int64, bool) {
	clean, ok := safeJoin(base, rel)
	if !ok {
		return nil, 0, false
	}
	fi, err := os.Lstat(clean)
	if err != nil || !fi.Mode().IsRegular() {
		return nil, 0, false
	}
	if fi.Size() == 0 || fi.Size() > maxBytes {
		return nil, 0, false
	}
	f, err := os.OpenFile(clean, os.O_RDONLY|noFollow, 0)
	if err != nil {
		return nil, 0, false
	}
	return f, fi.Size(), true
}

// hashFile reads up to maxBytes from f and returns the lowercase hex sha256 and
// byte count, or ok=false on a read error or cap breach.
func hashFile(f io.Reader, maxBytes int64) (sum string, n int64, ok bool) {
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, maxBytes+1))
	if err != nil || n > maxBytes {
		return "", 0, false
	}
	return hex.EncodeToString(h.Sum(nil)), n, true
}

// canonicalRowHash returns the sha256 of obj re-marshaled with sorted keys, so a
// row is identified by content independent of key order or whitespace.
func canonicalRowHash(obj map[string]json.RawMessage) (string, error) {
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		b.Write(kb)
		b.WriteByte(':')
		b.Write(obj[k])
	}
	b.WriteByte('}')
	sum := sha256.Sum256(b.Bytes())
	return hex.EncodeToString(sum[:]), nil
}
