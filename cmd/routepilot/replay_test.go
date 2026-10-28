package main

import (
	"bytes"
	"encoding/json"
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

const cliLatencyProfiles = `profiles:
  - model: cheap-a
    ttft_median_ms: 400
    ttft_sigma: 0.5
    tpot_median_ms: 10
    tpot_sigma: 0.3
    source: test
    retrieved: "2026-10-28"
  - model: mid-b
    ttft_median_ms: 800
    ttft_sigma: 0.5
    tpot_median_ms: 20
    tpot_sigma: 0.3
    source: test
    retrieved: "2026-10-28"
  - model: top-c
    ttft_median_ms: 1600
    ttft_sigma: 0.5
    tpot_median_ms: 40
    tpot_sigma: 0.3
    source: test
    retrieved: "2026-10-28"
`

func writeCLIProfiles(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "profiles.yaml")
	if err := os.WriteFile(path, []byte(cliLatencyProfiles), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReplayCLILatencyFlags(t *testing.T) {
	profiles := writeCLIProfiles(t)
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer

	code := run(replayArgs(t, out, "--split", "all", "--policy", "cheapest",
		"--latency-profiles", profiles, "--latency-seed", "3", "--slo-e2e-ms", "500"), &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(replay with latency) = %d, want 0; stderr = %q", code, stderr.String())
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var rec struct {
			TTFT     *float64 `json:"ttft_ms_sim"`
			E2E      *float64 `json:"e2e_ms_sim"`
			SLO      *float64 `json:"slo_ms"`
			Violated *bool    `json:"slo_violated"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("decode %q: %v", line, err)
		}
		if rec.TTFT == nil || rec.E2E == nil || rec.SLO == nil || *rec.SLO != 500 || rec.Violated == nil {
			t.Errorf("record %s: want simulated latency and an SLO verdict", line)
		}
	}

	// The same flags give the same file.
	out2 := filepath.Join(t.TempDir(), "out2.jsonl")
	code = run(replayArgs(t, out2, "--split", "all", "--policy", "cheapest",
		"--latency-profiles", profiles, "--latency-seed", "3", "--slo-e2e-ms", "500"), &stdout, &stderr)
	raw2, _ := os.ReadFile(out2)
	if code != 0 || !bytes.Equal(raw, raw2) {
		t.Error("two runs with the same latency flags differ")
	}
}

func TestReplayCLILatencyUsageAndInputErrors(t *testing.T) {
	profiles := writeCLIProfiles(t)
	out := filepath.Join(t.TempDir(), "out.jsonl")
	base := []string{"--split", "all", "--policy", "cheapest"}
	cases := []struct {
		name  string
		flags []string
		want  int
	}{
		{"slo without profiles", []string{"--slo-ttft-ms", "100"}, 2},
		{"both slos", []string{"--latency-profiles", profiles, "--slo-ttft-ms", "100", "--slo-e2e-ms", "200"}, 2},
		{"negative slo", []string{"--latency-profiles", profiles, "--slo-e2e-ms", "-3"}, 2},
		{"missing profile file", []string{"--latency-profiles", filepath.Join(t.TempDir(), "absent.yaml")}, 1},
	}
	for _, tc := range cases {
		var stdout, stderr bytes.Buffer
		args := append(replayArgs(t, out, base...), tc.flags...)
		if code := run(args, &stdout, &stderr); code != tc.want {
			t.Errorf("%s: run = %d, want %d; stderr = %q", tc.name, code, tc.want, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Errorf("%s: stdout = %q, want empty", tc.name, stdout.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a failed latency run left the out file behind")
	}
}
