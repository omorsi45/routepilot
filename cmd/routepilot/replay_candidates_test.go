package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReplayCLIEmitCandidates: --emit-candidates adds the learned policy's per-model scores
// to every record, and the default output has none.
func TestReplayCLIEmitCandidates(t *testing.T) {
	plain := filepath.Join(t.TempDir(), "plain.jsonl")
	with := filepath.Join(t.TempDir(), "with.jsonl")
	var stdout, stderr bytes.Buffer
	if code := run(learnedArgs(plain), &stdout, &stderr); code != 0 {
		t.Fatalf("replay learned = %d; stderr = %q", code, stderr.String())
	}
	if code := run(learnedArgs(with, "--emit-candidates"), &stdout, &stderr); code != 0 {
		t.Fatalf("replay learned --emit-candidates = %d; stderr = %q", code, stderr.String())
	}
	rawPlain, err := os.ReadFile(plain)
	if err != nil {
		t.Fatal(err)
	}
	rawWith, err := os.ReadFile(with)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(rawPlain, []byte(`"candidates"`)) {
		t.Error("the default output has candidates")
	}
	lines := strings.Split(strings.TrimSpace(string(rawWith)), "\n")
	if len(lines) != 6 {
		t.Fatalf("%d records, want 6", len(lines))
	}
	for _, l := range lines {
		if !strings.Contains(l, `"candidates":[{"model":"tiny-weak"`) || !strings.Contains(l, `{"model":"tiny-strong"`) {
			t.Errorf("record without both candidates: %s", l)
		}
	}
}

func TestReplayCLIEmitCandidatesNeedsTheLearnedPolicy(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stderr bytes.Buffer
	args := []string{
		"replay", "--config", filepath.Join(learnedFixtureDir, "config.yaml"),
		"--input", filepath.Join(learnedFixtureDir, "items.jsonl"),
		"--out", out, "--split", "all", "--policy", "cheapest", "--emit-candidates",
	}
	if code := run(args, &bytes.Buffer{}, &stderr); code != 2 {
		t.Errorf("exit = %d, want 2; stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "--emit-candidates is only valid with --policy learned") {
		t.Errorf("stderr = %q, want the reason", stderr.String())
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("an output file was created for a usage error")
	}
}
