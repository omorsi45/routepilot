package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var replayFixtureDir = filepath.Join("..", "..", "testdata", "replay")

func replayArgs(t *testing.T, out string, extra ...string) []string {
	t.Helper()
	args := []string{
		"replay",
		"--config", filepath.Join(replayFixtureDir, "config.yaml"),
		"--input", filepath.Join(replayFixtureDir, "items.jsonl"),
		"--out", out,
	}
	return append(args, extra...)
}

func TestReplayCLI(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer

	code := run(replayArgs(t, out, "--split", "test", "--policy", "cheapest", "--seed", "7"), &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(replay) = %d, want 0; stderr = %q", code, stderr.String())
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read out file: %v", err)
	}
	if lines := strings.Count(string(raw), "\n"); lines != 2 {
		t.Errorf("out file has %d lines, want 2 (the test split)", lines)
	}
	if strings.Contains(string(raw), "SENTINEL-PROMPT-7f3a") {
		t.Error("out file carries prompt text")
	}
	// cheap-a recorded on the two test items: ok 250 and ok 120.
	want := "items=2 errors=0 success_rate=1.0000 mean_cost_usd=0.000000000185\n"
	if stdout.String() != want {
		t.Errorf("stdout = %q, want %q", stdout.String(), want)
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q, want empty", stderr.String())
	}
}

func TestReplayCLIFixedModel(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer

	code := run(replayArgs(t, out, "--split", "all", "--policy", "fixed", "--model", "ghost"), &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(replay fixed ghost) = %d, want 0 (routing errors are recorded, not fatal); stderr = %q", code, stderr.String())
	}
	if !strings.HasPrefix(stdout.String(), "items=5 errors=5 success_rate=0.0000 mean_cost_usd=0") {
		t.Errorf("stdout = %q, want 5 errors and no success", stdout.String())
	}
}

func TestReplayCLIUsageErrors(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	cfg := filepath.Join(replayFixtureDir, "config.yaml")
	in := filepath.Join(replayFixtureDir, "items.jsonl")
	cases := map[string][]string{
		"bad flag":            {"replay", "--nope"},
		"no flags":            {"replay"},
		"missing config":      {"replay", "--input", in, "--split", "all", "--policy", "cheapest", "--out", out},
		"missing input":       {"replay", "--config", cfg, "--split", "all", "--policy", "cheapest", "--out", out},
		"missing split":       {"replay", "--config", cfg, "--input", in, "--policy", "cheapest", "--out", out},
		"missing policy":      {"replay", "--config", cfg, "--input", in, "--split", "all", "--out", out},
		"missing out":         {"replay", "--config", cfg, "--input", in, "--split", "all", "--policy", "cheapest"},
		"unknown policy":      {"replay", "--config", cfg, "--input", in, "--split", "all", "--policy", "best", "--out", out},
		"fixed without model": {"replay", "--config", cfg, "--input", in, "--split", "all", "--policy", "fixed", "--out", out},
		"extra argument":      {"replay", "--config", cfg, "--input", in, "--split", "all", "--policy", "cheapest", "--out", out, "extra"},
	}
	for name, args := range cases {
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 2 {
			t.Errorf("%s: run = %d, want 2; stderr = %q", name, code, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Errorf("%s: stdout = %q, want empty", name, stdout.String())
		}
		if !strings.Contains(stderr.String(), "usage") {
			t.Errorf("%s: stderr = %q, want a usage line", name, stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a usage error created the out file")
	}
}

func TestReplayCLIInputErrors(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "missing")
	cfg := filepath.Join(replayFixtureDir, "config.yaml")
	in := filepath.Join(replayFixtureDir, "items.jsonl")
	empty := filepath.Join(dir, "empty.jsonl")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	cases := map[string][]string{
		"config file missing": {"--config", missing, "--input", in},
		"input file missing":  {"--config", cfg, "--input", missing},
		"input file empty":    {"--config", cfg, "--input", empty},
		"out dir missing":     {"--config", cfg, "--input", in, "--out", filepath.Join(dir, "no", "such", "out.jsonl")},
	}
	for name, flags := range cases {
		args := []string{"replay", "--split", "all", "--policy", "cheapest", "--out", filepath.Join(dir, "out.jsonl")}
		args = append(args, flags...) // a later --out wins over the earlier one
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 1 {
			t.Errorf("%s: run = %d, want 1; stderr = %q", name, code, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Errorf("%s: stdout = %q, want empty", name, stdout.String())
		}
		if !strings.Contains(stderr.String(), "routepilot replay:") {
			t.Errorf("%s: stderr = %q, want a routepilot replay error", name, stderr.String())
		}
	}
}
