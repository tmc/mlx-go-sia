"""
Python reference implementation of Go sia/jsonutil.go

This file describes the BEHAVIOR of two internal Go functions:
  - missing_keys(data: bytes, *keys: str) -> list[str]
  - join_sorted(s: list[str]) -> str

It is used as the PORTING REFERENCE for a synthetic round-trip SIA task.
The porting agent reads this file and must re-implement equivalent Go code.

=== FUNCTION: missing_keys ===

Signature (Python):
    def missing_keys(data: bytes, *keys: str) -> list[str]:

Behavior:
    Given a JSON-encoded byte string `data` and a variable number of string keys,
    returns a list of keys that are ABSENT from the top-level JSON object.

Rules:
    1. If `data` cannot be parsed as a JSON object (malformed JSON, or the top-level
       value is not an object), ALL keys are considered missing — return all keys
       in their original order as a plain copy.
    2. A key that maps to JSON `null` is considered PRESENT (not missing).
    3. A key that is literally absent from the object's top-level keys is MISSING.
    4. Only top-level keys are checked; nested keys are not traversed.
    5. Return only the keys that are absent. Preserve original argument order in result.
    6. If `data` is valid JSON but no keys are provided, return an empty list.

Examples:
    missing_keys(b'{"a":1,"b":2}', "a", "b", "c")  -> ["c"]
    missing_keys(b'{"a":null}', "a")               -> []      # null = present
    missing_keys(b'{}', "x", "y")                  -> ["x", "y"]
    missing_keys(b'not json', "x", "y")             -> ["x", "y"]  # malformed -> all missing
    missing_keys(b'{"a":1}')                        -> []      # no keys to check
    missing_keys(b'[1,2,3]', "a")                  -> ["a"]   # array is not an object -> all missing

=== FUNCTION: join_sorted ===

Signature (Python):
    def join_sorted(s: list[str]) -> str:

Behavior:
    Sorts the input list (ascending, lexicographic) and joins the elements with
    the string ", " (comma followed by a single space).

Rules:
    1. Do NOT modify the input list — work on a copy.
    2. Sort order is standard lexicographic (Go's sort.Strings order = UTF-8 byte order).
    3. Separator is exactly ", " (no trailing comma or space).
    4. An empty list returns an empty string "".
    5. A single-element list returns just that element with no separator.

Examples:
    join_sorted(["c", "a", "b"])      -> "a, b, c"
    join_sorted(["missing"])          -> "missing"
    join_sorted([])                   -> ""
    join_sorted(["z", "a", "m", "b"]) -> "a, b, m, z"
"""

import json


def missing_keys(data: bytes, *keys: str) -> list[str]:
    """Return keys absent from the top-level JSON object in data."""
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return list(keys)
    if not isinstance(obj, dict):
        return list(keys)
    return [k for k in keys if k not in obj]


def join_sorted(s: list[str]) -> str:
    """Sort s and join with ', ' for stable error messages."""
    return ", ".join(sorted(list(s)))


# Self-tests (run with: python3 jsonutil_reference.py)
if __name__ == "__main__":
    import sys

    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"FAIL {label}: got {got!r}, want {want!r}")
        else:
            print(f"PASS {label}")

    check("present key", missing_keys(b'{"a":1,"b":2}', "a", "b"), [])
    check("absent key", missing_keys(b'{"a":1}', "a", "b", "c"), ["b", "c"])
    check("null is present", missing_keys(b'{"a":null}', "a"), [])
    check("empty obj", missing_keys(b'{}', "x", "y"), ["x", "y"])
    check("bad json all missing", missing_keys(b'not json', "x", "y"), ["x", "y"])
    check("array not obj", missing_keys(b'[1,2]', "a"), ["a"])
    check("no keys", missing_keys(b'{"a":1}'), [])

    check("join sorted abc", join_sorted(["c", "a", "b"]), "a, b, c")
    check("join single", join_sorted(["missing"]), "missing")
    check("join empty", join_sorted([]), "")
    check("join four", join_sorted(["z", "a", "m", "b"]), "a, b, m, z")
    check("no mutation", (lambda s: (join_sorted(s), s))(["c", "a"]), ("a, c", ["c", "a"]))

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        sys.exit(1)
    print("All reference self-tests passed.")
