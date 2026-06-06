package sia

import (
	"encoding/json"
	"sort"
	"strings"
)

// missingKeys reports which of keys are absent from the top-level JSON object in
// data. A value of JSON null counts as present (matching Python's key check),
// while a malformed object reports all keys missing.
func missingKeys(data []byte, keys ...string) []string {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return append([]string(nil), keys...)
	}
	var missing []string
	for _, k := range keys {
		if _, ok := obj[k]; !ok {
			missing = append(missing, k)
		}
	}
	return missing
}

// joinSorted sorts s and joins it with ", " for stable error messages.
func joinSorted(s []string) string {
	out := append([]string(nil), s...)
	sort.Strings(out)
	return strings.Join(out, ", ")
}
