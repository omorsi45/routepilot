package version

import (
	"runtime"
	"strings"
	"testing"
)

func TestStringIncludesVersionAndGo(t *testing.T) {
	got := String()

	if !strings.Contains(got, Version) {
		t.Errorf("String() = %q, want it to contain Version %q", got, Version)
	}
	if !strings.Contains(got, runtime.Version()) {
		t.Errorf("String() = %q, want it to contain Go version %q", got, runtime.Version())
	}
	if !strings.HasPrefix(got, "routepilot ") {
		t.Errorf("String() = %q, want prefix %q", got, "routepilot ")
	}
}
