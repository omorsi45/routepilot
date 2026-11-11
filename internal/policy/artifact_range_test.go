package policy

import (
	"errors"
	"testing"
)

// TestLoadRejectsValuesOutsideTheirRange: Load refuses the calibration values the exporter
// refuses, and the ranges that follow from what the values mean. A value outside its range
// does not crash anything, it quietly clears a quality floor or prices a request wrongly, so
// the manifest is rejected before any request is scored with it. The manifest is re-hashed
// after each edit so that only the range check can fail. The tiny artifact holds the edge
// values 0 and 1 (the first isotonic y, the last lcb upper), so loading it also pins that
// the edges are accepted.
func TestLoadRejectsValuesOutsideTheirRange(t *testing.T) {
	last := func(t *testing.T, d doc, path ...any) int {
		t.Helper()
		return len(objectAt(t, d, path...).([]any)) - 1
	}
	tests := map[string]func(t *testing.T, d doc){
		"isotonic y above 1": func(t *testing.T, d doc) {
			setAt(t, d, 1.4, "models", 0, "calibrator", "y", 3)
		},
		"isotonic y below 0": func(t *testing.T, d doc) {
			setAt(t, d, -0.1, "models", 0, "calibrator", "y", 3)
		},
		"lcb upper does not end at 1": func(t *testing.T, d doc) {
			setAt(t, d, 0.9, "models", 0, "lcb", "upper", last(t, d, "models", 0, "lcb", "upper"))
		},
		"lcb upper ends above 1": func(t *testing.T, d doc) {
			setAt(t, d, 1.5, "models", 0, "lcb", "upper", last(t, d, "models", 0, "lcb", "upper"))
		},
		"lcb lower above 1": func(t *testing.T, d doc) {
			setAt(t, d, 1.2, "models", 0, "lcb", "lower", 2)
		},
		"lcb lower below 0": func(t *testing.T, d doc) {
			setAt(t, d, -0.1, "models", 0, "lcb", "lower", 2)
		},
		"pooled_rate above 1": func(t *testing.T, d doc) {
			setAt(t, d, 1.5, "models", 0, "lcb", "pooled_rate")
		},
		"pooled_rate below 0": func(t *testing.T, d doc) {
			setAt(t, d, -0.5, "models", 0, "lcb", "pooled_rate")
		},
		"pooled_lcb above 1": func(t *testing.T, d doc) {
			setAt(t, d, 2.0, "models", 0, "lcb", "pooled_lcb")
		},
		"pooled_lcb below 0": func(t *testing.T, d doc) {
			setAt(t, d, -0.2, "models", 1, "lcb", "pooled_lcb")
		},
		"smear zero": func(t *testing.T, d doc) {
			setAt(t, d, 0.0, "models", 0, "length", "smear")
		},
		"smear negative": func(t *testing.T, d doc) {
			setAt(t, d, -1.0, "models", 1, "length", "smear")
		},
		"in_ratio zero": func(t *testing.T, d doc) {
			setAt(t, d, 0.0, "models", 0, "length", "in_ratio")
		},
		"in_ratio negative": func(t *testing.T, d doc) {
			setAt(t, d, -0.3, "models", 1, "length", "in_ratio")
		},
	}
	for name, edit := range tests {
		t.Run(name, func(t *testing.T) {
			d := readDoc(t, tinyDir(t))
			edit(t, d)

			_, err := Load(rewrite(t, tinyDir(t), d))

			if !errors.Is(err, ErrManifest) {
				t.Fatalf("Load = %v, want ErrManifest", err)
			}
		})
	}
}
