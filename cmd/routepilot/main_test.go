package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunVersion(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run([]string{"version"}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(version) = %d, want 0; stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "routepilot dev") {
		t.Errorf("stdout = %q, want it to contain %q", stdout.String(), "routepilot dev")
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunUnknownCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run([]string{"nope"}, &stdout, &stderr)

	if code != 2 {
		t.Fatalf("run(nope) = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "usage") {
		t.Errorf("stderr = %q, want it to contain %q", stderr.String(), "usage")
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty", stdout.String())
	}
}

func TestRunNoArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run(nil, &stdout, &stderr)

	if code != 2 {
		t.Fatalf("run() = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "usage") {
		t.Errorf("stderr = %q, want it to contain %q", stderr.String(), "usage")
	}
}

// TestUsageNamesWhatPolicyCanDo: the top-level usage line for policy lists all three
// things the subcommand does, so gate is not a hidden command.
func TestUsageNamesWhatPolicyCanDo(t *testing.T) {
	var stdout, stderr bytes.Buffer

	run(nil, &stdout, &stderr)

	if want := "verify, inspect or gate a trained policy artifact"; !strings.Contains(stderr.String(), want) {
		t.Errorf("usage = %q, want it to contain %q", stderr.String(), want)
	}
}
