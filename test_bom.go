package sia

import (
	"fmt"
	"testing"
)

func TestBOMRejection(t *testing.T) {
	// BOM followed by a valid assignment
	bomLine := "\xef\xbb\xbf" + "learning_rate = 1e-5"
	key, val, ok := splitAssignment(bomLine)
	fmt.Printf("BOM line: %q\n", bomLine)
	fmt.Printf("  key=%q, val=%q, ok=%v\n", key, val, ok)
	fmt.Printf("  key runes: %v\n", []rune(key))
	
	// Just BOM
	justBOM := "\xef\xbb\xbf"
	key, val, ok = splitAssignment(justBOM)
	fmt.Printf("Just BOM: %q\n", justBOM)
	fmt.Printf("  key=%q, val=%q, ok=%v\n", key, val, ok)
	
	// Valid line
	validLine := "learning_rate = 1e-5"
	key, val, ok = splitAssignment(validLine)
	fmt.Printf("Valid line: %q\n", validLine)
	fmt.Printf("  key=%q, val=%q, ok=%v\n", key, val, ok)
	fmt.Printf("  key runes: %v\n", []rune(key))
	fmt.Printf("  isIdent(key)=%v\n", isIdent(key))
}
