// Package jsonutil_port is the anti-leak oracle for the SIA self-port harness.
//
// The PORTING AGENT sees:
//   - jsonutil_reference.py (full Python behavioral reference)
//   - This file's function signatures and test names
//   - The test CONTRACT written as comments here
//
// The PORTING AGENT does NOT see:
//   - The original Go source (jsonutil.go from the sia package)
//   - The expected values in the table cases below (this file stays outside the agent's input)
//
// Grade: `go test ./...` must pass on the candidate jsonutil.go.
package jsonutil

import (
	"sort"
	"strings"
	"testing"
)

// CONTRACT for missingKeys:
//   func missingKeys(data []byte, keys ...string) []string
//   - Parses data as a JSON object and returns keys absent from the top-level object.
//   - A null value counts as present.
//   - If data is malformed or not an object, ALL keys are considered missing.
//   - Return order: same as keys argument order (only the absent ones).

func TestMissingKeys(t *testing.T) {
	tests := []struct {
		name string
		data string
		keys []string
		want []string
	}{
		{
			name: "all_present",
			data: `{"a":1,"b":2,"c":3}`,
			keys: []string{"a", "b", "c"},
			want: nil,
		},
		{
			name: "one_missing",
			data: `{"a":1,"b":2}`,
			keys: []string{"a", "b", "c"},
			want: []string{"c"},
		},
		{
			name: "null_counts_as_present",
			data: `{"a":null,"b":2}`,
			keys: []string{"a", "b"},
			want: nil,
		},
		{
			name: "all_missing_empty_object",
			data: `{}`,
			keys: []string{"x", "y"},
			want: []string{"x", "y"},
		},
		{
			name: "malformed_json_all_missing",
			data: `{not valid json`,
			keys: []string{"x", "y"},
			want: []string{"x", "y"},
		},
		{
			name: "array_not_object_all_missing",
			data: `[1,2,3]`,
			keys: []string{"a"},
			want: []string{"a"},
		},
		{
			name: "no_keys_to_check",
			data: `{"a":1}`,
			keys: []string{},
			want: nil,
		},
		{
			name: "preserves_argument_order",
			data: `{"b":2}`,
			keys: []string{"c", "a", "b"},
			want: []string{"c", "a"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := missingKeys([]byte(tt.data), tt.keys...)
			if !equalStringSlices(got, tt.want) {
				t.Errorf("missingKeys(%q, %v) = %v, want %v", tt.data, tt.keys, got, tt.want)
			}
		})
	}
}

// CONTRACT for joinSorted:
//   func joinSorted(s []string) string
//   - Sorts a copy of s (ascending lexicographic) and joins with ", ".
//   - Does NOT modify the input slice.
//   - Empty slice returns "".

func TestJoinSorted(t *testing.T) {
	tests := []struct {
		name  string
		input []string
		want  string
	}{
		{"empty", []string{}, ""},
		{"single", []string{"missing"}, "missing"},
		{"already_sorted", []string{"a", "b", "c"}, "a, b, c"},
		{"needs_sorting", []string{"c", "a", "b"}, "a, b, c"},
		{"four_elements", []string{"z", "a", "m", "b"}, "a, b, m, z"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputCopy := append([]string(nil), tt.input...)
			got := joinSorted(tt.input)
			if got != tt.want {
				t.Errorf("joinSorted(%v) = %q, want %q", tt.input, got, tt.want)
			}
			// Verify input was not mutated.
			if !equalStringSlices(tt.input, inputCopy) {
				t.Errorf("joinSorted mutated input: was %v, now %v", inputCopy, tt.input)
			}
		})
	}
}

// TestJoinSortedMatchesSort verifies that joinSorted output is equivalent to
// sorting independently and joining — a property-style check.
func TestJoinSortedMatchesSort(t *testing.T) {
	cases := [][]string{
		{"delta", "alpha", "gamma", "beta"},
		{"single"},
		{},
		{"z", "a"},
	}
	for _, c := range cases {
		sorted := append([]string(nil), c...)
		sort.Strings(sorted)
		want := strings.Join(sorted, ", ")
		got := joinSorted(c)
		if got != want {
			t.Errorf("joinSorted(%v) = %q, want %q", c, got, want)
		}
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
