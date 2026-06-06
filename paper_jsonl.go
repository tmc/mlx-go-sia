package sia

import (
	"bytes"
	"encoding/json"
	"io"
	"regexp"
)

// digestRE matches a sha256-prefixed content digest exactly: lowercase, 64 hex.
var digestRE = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// fixtureRow is one validated JSONL fixture row: its raw object, canonical hash,
// and the validator's later verdict (filled by control/falsifier scoring).
type fixtureRow struct {
	obj  map[string]json.RawMessage
	hash string
}

// jsonlCounts records the outcome of strictly parsing a JSONL fixture: how many
// non-blank lines there were, how many failed to parse or validate, and how many
// distinct valid rows resulted. fixture_row is true only when parseFail and
// validateFail are both zero, so a single good row among garbage cannot launder
// a fixture to true.
type jsonlCounts struct {
	total        int
	blank        int
	parseFail    int
	validateFail int
	valid        int
	distinct     int
}

// parseFixture strictly parses data as JSONL and validates every non-blank line
// against schema. It returns the counts and the distinct valid rows. A leading
// UTF-8 BOM is stripped once and a trailing CR per line is trimmed; beyond that
// no leniency: trailing bytes after a row, duplicate keys, wrong types, or an
// out-of-range enum all count as failures.
func parseFixture(data []byte, schema RowSchema) (jsonlCounts, []fixtureRow) {
	data = stripBOM(data)
	var counts jsonlCounts
	var rows []fixtureRow
	seen := map[string]bool{}

	for _, raw := range splitJSONLines(data) {
		line := bytes.TrimRight(raw, "\r")
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			counts.blank++
			continue
		}
		counts.total++

		obj, err := strictDecodeObject(line)
		if err != nil {
			counts.parseFail++
			continue
		}
		if !schema.validate(obj) {
			counts.validateFail++
			continue
		}
		h, err := canonicalRowHash(obj)
		if err != nil {
			counts.validateFail++
			continue
		}
		counts.valid++
		if !seen[h] {
			seen[h] = true
			counts.distinct++
		}
		rows = append(rows, fixtureRow{obj: obj, hash: h})
	}
	return counts, rows
}

// strictDecodeObject decodes one JSONL line into a JSON object, rejecting
// trailing bytes after the object and duplicate top-level keys.
func strictDecodeObject(line []byte) (map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(line))
	var obj map[string]json.RawMessage
	if err := dec.Decode(&obj); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, errTrailingBytes
	}
	// Duplicate-key detection: re-scan tokens and count object keys at depth 1.
	if dup, err := hasDuplicateKeys(line); err != nil || dup {
		if err != nil {
			return nil, err
		}
		return nil, errDuplicateKey
	}
	return obj, nil
}

var (
	errTrailingBytes = jsonErr("trailing bytes after JSON object")
	errDuplicateKey  = jsonErr("duplicate JSON key")
)

type jsonErr string

func (e jsonErr) Error() string { return string(e) }

// hasDuplicateKeys reports whether the top-level object in line repeats any key.
func hasDuplicateKeys(line []byte) (bool, error) {
	dec := json.NewDecoder(bytes.NewReader(line))
	tok, err := dec.Token()
	if err != nil {
		return false, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return false, nil // not an object; handled elsewhere
	}
	seen := map[string]bool{}
	depth := 1
	expectKey := true
	for depth > 0 {
		t, err := dec.Token()
		if err != nil {
			return false, err
		}
		switch v := t.(type) {
		case json.Delim:
			switch v {
			case '{', '[':
				depth++
				expectKey = false
			case '}', ']':
				depth--
				expectKey = depth == 1
			}
		case string:
			if depth == 1 && expectKey {
				if seen[v] {
					return true, nil
				}
				seen[v] = true
				expectKey = false
			} else if depth == 1 {
				expectKey = true
			}
		default:
			if depth == 1 {
				expectKey = true
			}
		}
	}
	return false, nil
}

// validate reports whether obj satisfies the structural schema: the const tag
// matches byte-for-byte, all required fields are present, declared field types
// hold exactly, enums are in range, and digest fields are sha256-shaped.
func (s RowSchema) validate(obj map[string]json.RawMessage) bool {
	if s.TagField != "" {
		raw, ok := obj[s.TagField]
		if !ok {
			return false
		}
		var tag string
		if err := json.Unmarshal(raw, &tag); err != nil || tag != s.ConstTag {
			return false
		}
	}
	for _, f := range s.Required {
		if _, ok := obj[f]; !ok {
			return false
		}
	}
	for _, f := range s.StringFields {
		if raw, ok := obj[f]; ok && !isJSONString(raw) {
			return false
		}
	}
	for _, f := range s.IntFields {
		if raw, ok := obj[f]; ok && !isJSONInt(raw) {
			return false
		}
	}
	for _, f := range s.BoolFields {
		if raw, ok := obj[f]; ok && !isJSONBool(raw) {
			return false
		}
	}
	for f, allowed := range s.Enums {
		raw, ok := obj[f]
		if !ok {
			continue
		}
		var v string
		if err := json.Unmarshal(raw, &v); err != nil {
			return false
		}
		if !contains(allowed, v) {
			return false
		}
	}
	for _, f := range s.DigestFields {
		raw, ok := obj[f]
		if !ok {
			continue
		}
		var v string
		if err := json.Unmarshal(raw, &v); err != nil || !digestRE.MatchString(v) {
			return false
		}
	}
	return true
}

// isJSONString reports whether raw is a JSON string (not a number or bool).
func isJSONString(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	return len(t) > 0 && t[0] == '"'
}

// isJSONBool reports whether raw is exactly true or false.
func isJSONBool(raw json.RawMessage) bool {
	t := string(bytes.TrimSpace(raw))
	return t == "true" || t == "false"
}

// isJSONInt reports whether raw is a JSON integer (no fraction or exponent,
// no string quoting).
func isJSONInt(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || t[0] == '"' {
		return false
	}
	var n json.Number
	dec := json.NewDecoder(bytes.NewReader(t))
	dec.UseNumber()
	if err := dec.Decode(&n); err != nil {
		return false
	}
	s := n.String()
	return !bytes.ContainsAny([]byte(s), ".eE")
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// splitJSONLines splits data on newlines, preserving line content (a trailing CR
// is trimmed by the caller). Empty trailing segment is dropped.
func splitJSONLines(data []byte) [][]byte {
	lines := bytes.Split(data, []byte("\n"))
	if n := len(lines); n > 0 && len(bytes.TrimSpace(lines[n-1])) == 0 {
		lines = lines[:n-1]
	}
	return lines
}

// stripBOM removes a single leading UTF-8 byte-order mark.
func stripBOM(data []byte) []byte {
	return bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
}

// readAllCapped reads at most maxBytes from r, discarding any excess.
func readAllCapped(r io.Reader, maxBytes int64) []byte {
	data, _ := io.ReadAll(io.LimitReader(r, maxBytes))
	return data
}
