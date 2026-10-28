package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/policy"
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

var (
	learnedFixtureDir = filepath.Join("..", "..", "testdata", "replay", "learned")
	tinyArtifactDir   = filepath.Join("..", "..", "testdata", "policy", "tiny", "0782d801a5a3")
)

func learnedArgs(out string, extra ...string) []string {
	args := []string{
		"replay",
		"--config", filepath.Join(learnedFixtureDir, "config.yaml"),
		"--input", filepath.Join(learnedFixtureDir, "items.jsonl"),
		"--out", out, "--split", "all", "--policy", "learned",
		"--artifact", tinyArtifactDir,
		"--policy-config", filepath.Join(learnedFixtureDir, "policy.yaml"),
	}
	return append(args, extra...)
}

type learnedLine struct {
	PolicyVersion *string  `json:"policy_version"`
	Model         *string  `json:"model"`
	Floor         *float64 `json:"floor"`
	Policy        string   `json:"policy"`
}

func readLearnedLines(t *testing.T, path string) []learnedLine {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var lines []learnedLine
	for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var rec learnedLine
		if err := json.Unmarshal([]byte(l), &rec); err != nil {
			t.Fatalf("decode %q: %v", l, err)
		}
		lines = append(lines, rec)
	}
	return lines
}

func TestReplayCLILearned(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer
	if code := run(learnedArgs(out), &stdout, &stderr); code != 0 {
		t.Fatalf("run(replay learned) = %d, want 0; stderr = %q", code, stderr.String())
	}
	if !strings.HasPrefix(stdout.String(), "items=6 errors=0 ") {
		t.Errorf("stdout = %q, want 6 items and no errors", stdout.String())
	}
	base := policy.Config{Mode: "min_cost", OODAction: "conservative", OnInfeasible: "best_effort"}
	wantVersion := "0782d801a5a3+" + base.Hash8()
	for _, rec := range readLearnedLines(t, out) {
		if rec.Policy != "learned" || rec.PolicyVersion == nil || *rec.PolicyVersion != wantVersion || rec.Model == nil {
			t.Errorf("record = %+v, want policy learned, version %s and a model", rec, wantVersion)
		}
		if rec.Floor == nil || *rec.Floor != 0 {
			t.Errorf("floor = %v, want 0 from the policy config", rec.Floor)
		}
	}

	// Flags override the policy config, and the version shows the config that ran.
	out2 := filepath.Join(t.TempDir(), "out2.jsonl")
	stdout.Reset()
	args := learnedArgs(out2, "--mode", "dual", "--lambda", "5", "--floor", "0.9", "--epsilon", "0.1", "--seed", "4")
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("run(replay learned with overrides) = %d; stderr = %q", code, stderr.String())
	}
	over := policy.Config{Mode: "dual", Lambda: 5, DefaultFloor: 0.9, Epsilon: 0.1, OODAction: "conservative", OnInfeasible: "best_effort"}
	for _, rec := range readLearnedLines(t, out2) {
		if rec.PolicyVersion == nil || *rec.PolicyVersion != "0782d801a5a3+"+over.Hash8() {
			t.Errorf("version = %v, want 0782d801a5a3+%s", rec.PolicyVersion, over.Hash8())
		}
		if rec.Floor == nil || *rec.Floor != 0.9 {
			t.Errorf("floor = %v, want the 0.9 override", rec.Floor)
		}
	}

	// The same flags give the same file; a flag set to its zero value still overrides.
	out3 := filepath.Join(t.TempDir(), "out3.jsonl")
	if code := run(learnedArgs(out3, "--mode", "dual", "--lambda", "5", "--floor", "0.9", "--epsilon", "0.1", "--seed", "4"), &stdout, &stderr); code != 0 {
		t.Fatal("second run failed")
	}
	a, _ := os.ReadFile(out2)
	b, _ := os.ReadFile(out3)
	if !bytes.Equal(a, b) {
		t.Error("two runs with the same flags differ")
	}
	out4 := filepath.Join(t.TempDir(), "out4.jsonl")
	if code := run(learnedArgs(out4, "--floor", "0"), &stdout, &stderr); code != 0 {
		t.Fatal("run with --floor 0 failed")
	}
}

func TestReplayCLILearnedWithLatency(t *testing.T) {
	prof := filepath.Join(t.TempDir(), "profiles.yaml")
	body := `profiles:
  - model: tiny-strong
    ttft_median_ms: 100
    ttft_sigma: 0.5
    tpot_median_ms: 1
    tpot_sigma: 0.3
    source: test
    retrieved: "2026-10-28"
  - model: tiny-weak
    ttft_median_ms: 10000
    ttft_sigma: 0.5
    tpot_median_ms: 100
    tpot_sigma: 0.3
    source: test
    retrieved: "2026-10-28"
`
	if err := os.WriteFile(prof, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer
	args := learnedArgs(out, "--latency-profiles", prof, "--slo-e2e-ms", "5000")
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("run = %d; stderr = %q", code, stderr.String())
	}
	// Only tiny-strong meets 5000 ms on its simulated profile, so the SLO routes there.
	for _, rec := range readLearnedLines(t, out) {
		if rec.Model == nil || *rec.Model != "tiny-strong" {
			t.Errorf("model = %v, want tiny-strong (the only one inside the SLO)", rec.Model)
		}
	}
}

func TestReplayCLILearnedUsageErrors(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	cfg := filepath.Join(learnedFixtureDir, "config.yaml")
	in := filepath.Join(learnedFixtureDir, "items.jsonl")
	polCfg := filepath.Join(learnedFixtureDir, "policy.yaml")
	static := []string{"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "cheapest"}
	cases := map[string][]string{
		"learned without artifact":      {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "learned", "--policy-config", polCfg},
		"learned without policy config": {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "learned", "--artifact", tinyArtifactDir},
		"artifact with cheapest":        append(append([]string{}, static...), "--artifact", tinyArtifactDir),
		"policy config with cheapest":   append(append([]string{}, static...), "--policy-config", polCfg),
		"mode with cheapest":            append(append([]string{}, static...), "--mode", "dual"),
		"lambda with cheapest":          append(append([]string{}, static...), "--lambda", "1"),
		"floor with cheapest":           append(append([]string{}, static...), "--floor", "0.5"),
		"epsilon with cheapest":         append(append([]string{}, static...), "--epsilon", "0.1"),
		"unknown mode":                  learnedArgs(out, "--mode", "greedy"),
		"negative lambda":               learnedArgs(out, "--lambda", "-1"),
		"floor above one":               learnedArgs(out, "--floor", "1.5"),
		"epsilon of one":                learnedArgs(out, "--epsilon", "1"),
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

func TestReplayCLILearnedInputErrors(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.jsonl")

	// A corrupt artifact: the same bundle with a truncated weights file.
	corrupt := filepath.Join(dir, "corrupt", "0782d801a5a3")
	if err := os.MkdirAll(corrupt, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		b, err := os.ReadFile(filepath.Join(tinyArtifactDir, name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "weights.bin" {
			b = b[:len(b)/2]
		}
		if err := os.WriteFile(filepath.Join(corrupt, name), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	badPolicy := filepath.Join(dir, "bad-policy.yaml")
	if err := os.WriteFile(badPolicy, []byte("mode: sideways\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	cases := map[string][]string{
		"missing artifact":      learnedArgs(out, "--artifact", filepath.Join(dir, "none")),
		"corrupt artifact":      learnedArgs(out, "--artifact", corrupt),
		"missing policy config": learnedArgs(out, "--policy-config", filepath.Join(dir, "none.yaml")),
		"invalid policy config": learnedArgs(out, "--policy-config", badPolicy),
	}
	for name, args := range cases {
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 1 {
			t.Errorf("%s: run = %d, want 1; stderr = %q", name, code, stderr.String())
		}
		if stdout.Len() != 0 || !strings.Contains(stderr.String(), "routepilot replay:") {
			t.Errorf("%s: stdout %q stderr %q", name, stdout.String(), stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("an input error left the out file behind")
	}
}

// writeMFFile writes an MF file for the replay fixture models whose score is
// sigmoid(standardized log1p(bytes)): it grows with the prompt length.
func writeMFFile(t *testing.T, mutate func(map[string]any)) string {
	t.Helper()
	const dim = 8
	u := make([]byte, 4*(dim+12))
	binary.LittleEndian.PutUint32(u[4*dim:], math.Float32bits(1))
	std := make([]float64, 12)
	for i := range std {
		std[i] = 1
	}
	mean := make([]float64, 12)
	mean[0] = math.Log1p(40)
	doc := map[string]any{
		"strong": "top-c", "weak": "cheap-a", "dim": dim,
		"dense_mean": mean, "dense_std": std,
		"u": base64.StdEncoding.EncodeToString(u),
	}
	if mutate != nil {
		mutate(doc)
	}
	body, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "mf.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReplayCLIRouteLLMMF(t *testing.T) {
	mf := writeMFFile(t, nil)
	out := filepath.Join(t.TempDir(), "out.jsonl")
	var stdout, stderr bytes.Buffer

	// The score is sigmoid(log1p(bytes) - log1p(40)): above 0.5 for texts over 40 bytes.
	args := replayArgs(t, out, "--split", "all", "--policy", "routellm_mf", "--mf", mf, "--threshold", "0.5")
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("run(replay routellm_mf) = %d, want 0; stderr = %q", code, stderr.String())
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "SENTINEL-PROMPT-7f3a") {
		t.Error("out file carries prompt text")
	}
	want := map[string]string{ // fixture prompt lengths: 42, 42, 41, 55 and 39 bytes
		"fx-a/1": "top-c", "fx-a/2": "top-c", "fx-b/3": "top-c", "fx-b/4": "top-c", "fx-b/5": "cheap-a",
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(want) {
		t.Fatalf("out file has %d lines, want %d", len(lines), len(want))
	}
	for _, line := range lines {
		var rec struct {
			ItemID string   `json:"item_id"`
			Policy string   `json:"policy"`
			Model  string   `json:"model"`
			P      *float64 `json:"p"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Policy != "routellm_mf" || rec.Model != want[rec.ItemID] || rec.P == nil {
			t.Errorf("%s: policy %q model %q p %v, want routellm_mf, %q and a score", rec.ItemID, rec.Policy, rec.Model, rec.P, want[rec.ItemID])
		}
	}
	if !strings.HasPrefix(stdout.String(), "items=5 errors=0 ") || stderr.Len() != 0 {
		t.Errorf("stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
}

func TestReplayCLIRouteLLMMFUsageErrors(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	mf := writeMFFile(t, nil)
	mfArgs := func(extra ...string) []string {
		return replayArgs(t, out, append([]string{"--split", "all"}, extra...)...)
	}
	cases := map[string][]string{
		"no mf":                mfArgs("--policy", "routellm_mf", "--threshold", "0.5"),
		"no threshold":         mfArgs("--policy", "routellm_mf", "--mf", mf),
		"threshold above one":  mfArgs("--policy", "routellm_mf", "--mf", mf, "--threshold", "1.5"),
		"negative threshold":   mfArgs("--policy", "routellm_mf", "--mf", mf, "--threshold", "-0.5"),
		"nan threshold":        mfArgs("--policy", "routellm_mf", "--mf", mf, "--threshold", "NaN"),
		"mf with cheapest":     mfArgs("--policy", "cheapest", "--mf", mf),
		"threshold w cheapest": mfArgs("--policy", "cheapest", "--threshold", "0.5"),
		"model with mf":        mfArgs("--policy", "routellm_mf", "--mf", mf, "--threshold", "0.5", "--model", "top-c"),
	}
	for name, args := range cases {
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 2 {
			t.Errorf("%s: run = %d, want 2; stderr = %q", name, code, stderr.String())
		}
		if stdout.Len() != 0 || !strings.Contains(stderr.String(), "usage") {
			t.Errorf("%s: stdout %q stderr %q", name, stdout.String(), stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a usage error created the out file")
	}
}

func TestReplayCLIRouteLLMMFInputErrors(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.jsonl")
	cases := map[string]string{
		"missing file":      filepath.Join(dir, "none.json"),
		"truncated weights": writeMFFile(t, func(d map[string]any) { d["u"] = "AAAA" }),
		"unknown model":     writeMFFile(t, func(d map[string]any) { d["strong"] = "ghost-model" }),
	}
	for name, mf := range cases {
		var stdout, stderr bytes.Buffer
		args := replayArgs(t, out, "--split", "all", "--policy", "routellm_mf", "--mf", mf, "--threshold", "0.5")
		if code := run(args, &stdout, &stderr); code != 1 {
			t.Errorf("%s: run = %d, want 1; stderr = %q", name, code, stderr.String())
		}
		if stdout.Len() != 0 || !strings.Contains(stderr.String(), "routepilot replay:") {
			t.Errorf("%s: stdout %q stderr %q", name, stdout.String(), stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("an input error left the out file behind")
	}
}

func relaxedCount(t *testing.T, path string) int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var rec struct {
			Relaxed []string `json:"relaxed"`
		}
		if err := json.Unmarshal([]byte(l), &rec); err != nil {
			t.Fatalf("decode %q: %v", l, err)
		}
		if len(rec.Relaxed) > 0 {
			n++
		}
	}
	return n
}

// TestReplayCLILearnedFloorOn: --floor-on lcb is the default; --floor-on point compares the
// floor with P instead of L, which never relaxes more items and, on the fixture, fewer at
// some floor.
func TestReplayCLILearnedFloorOn(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	run1 := func(name string, extra ...string) string {
		t.Helper()
		out := filepath.Join(dir, name)
		args := learnedArgs(out, append([]string{"--mode", "min_cost"}, extra...)...)
		if code := run(args, &stdout, &stderr); code != 0 {
			t.Fatalf("run(%v) = %d; stderr = %q", extra, code, stderr.String())
		}
		return out
	}

	// The default and an explicit lcb give the same file.
	a, _ := os.ReadFile(run1("default.jsonl", "--floor", "0.5"))
	b, _ := os.ReadFile(run1("lcb.jsonl", "--floor", "0.5", "--floor-on", "lcb"))
	if !bytes.Equal(a, b) {
		t.Error("--floor-on lcb differs from the default")
	}

	// The point arm names itself in the version, the lcb arm does not.
	pointOut := run1("point.jsonl", "--floor", "0.5", "--floor-on", "point")
	cfg := policy.Config{Mode: "min_cost", DefaultFloor: 0.5, OODAction: "conservative", OnInfeasible: "best_effort"}
	for _, rec := range readLearnedLines(t, pointOut) {
		if want := "0782d801a5a3+" + cfg.Hash8() + "+floor-point"; rec.PolicyVersion == nil || *rec.PolicyVersion != want {
			t.Errorf("point version = %v, want %s", rec.PolicyVersion, want)
		}
	}
	for _, rec := range readLearnedLines(t, run1("lcb-version.jsonl", "--floor", "0.5")) {
		if rec.PolicyVersion == nil || strings.Contains(*rec.PolicyVersion, "floor-point") {
			t.Errorf("lcb version = %v, want no floor-point suffix", rec.PolicyVersion)
		}
	}

	fewer := false
	for _, floor := range []string{"0.3", "0.5", "0.7", "0.9"} {
		lcb := relaxedCount(t, run1("l-"+floor+".jsonl", "--floor", floor, "--floor-on", "lcb"))
		point := relaxedCount(t, run1("p-"+floor+".jsonl", "--floor", floor, "--floor-on", "point"))
		if point > lcb {
			t.Errorf("floor %s: point arm relaxed %d items, lcb arm %d; reading P instead of L cannot relax more", floor, point, lcb)
		}
		fewer = fewer || point < lcb
	}
	if !fewer {
		t.Error("the point arm relaxed as many items as the lcb arm at every floor; the flag changes nothing")
	}
}

func TestReplayCLIFloorOnUsageErrors(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	cfg := filepath.Join(learnedFixtureDir, "config.yaml")
	in := filepath.Join(learnedFixtureDir, "items.jsonl")
	cases := map[string][]string{
		"unknown value":       learnedArgs(out, "--floor-on", "median"),
		"empty value":         learnedArgs(out, "--floor-on", ""),
		"with cheapest":       {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "cheapest", "--floor-on", "point"},
		"with cheapest (lcb)": {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "cheapest", "--floor-on", "lcb"},
	}
	for name, args := range cases {
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 2 {
			t.Errorf("%s: run = %d, want 2; stderr = %q", name, code, stderr.String())
		}
		if !strings.Contains(stderr.String(), "floor-on") || !strings.Contains(stderr.String(), "usage") {
			t.Errorf("%s: stderr = %q, want a message naming --floor-on and a usage line", name, stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a usage error created the out file")
	}
}

type oodLine struct {
	ItemID        string   `json:"item_id"`
	OOD           bool     `json:"ood"`
	OODScore      *float64 `json:"ood_score"`
	P             *float64 `json:"p"`
	L             *float64 `json:"l"`
	PolicyVersion *string  `json:"policy_version"`
}

func readOODLines(t *testing.T, path string) []oodLine {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var lines []oodLine
	for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var rec oodLine
		if err := json.Unmarshal([]byte(l), &rec); err != nil {
			t.Fatalf("decode %q: %v", l, err)
		}
		lines = append(lines, rec)
	}
	return lines
}

// TestReplayCLILearnedOODOff: --ood on is the default; --ood off keeps ood and ood_score as the
// artifact computed them but scores the request on the model's own numbers instead of the pooled
// ones, and the policy version names the arm.
func TestReplayCLILearnedOODOff(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	run1 := func(name string, extra ...string) string {
		t.Helper()
		out := filepath.Join(dir, name)
		args := learnedArgs(out, append([]string{"--mode", "min_cost", "--floor", "0.3"}, extra...)...)
		if code := run(args, &stdout, &stderr); code != 0 {
			t.Fatalf("run(%v) = %d; stderr = %q", extra, code, stderr.String())
		}
		return out
	}

	// The default and an explicit on give the same file.
	a, _ := os.ReadFile(run1("default.jsonl"))
	b, _ := os.ReadFile(run1("on.jsonl", "--ood", "on"))
	if !bytes.Equal(a, b) {
		t.Error("--ood on differs from the default")
	}

	shipped := readOODLines(t, filepath.Join(dir, "default.jsonl"))
	off := readOODLines(t, run1("off.jsonl", "--ood", "off"))
	if len(shipped) != len(off) || len(off) == 0 {
		t.Fatalf("%d shipped and %d off records", len(shipped), len(off))
	}
	cfg := policy.Config{Mode: "min_cost", DefaultFloor: 0.3, OODAction: "conservative", OnInfeasible: "best_effort"}
	oodItems, changed := 0, 0
	for i, rec := range off {
		ref := shipped[i]
		// The verdict and the score are those of the artifact whatever the arm.
		if rec.ItemID != ref.ItemID || rec.OOD != ref.OOD || rec.OODScore == nil || ref.OODScore == nil || *rec.OODScore != *ref.OODScore {
			t.Errorf("%s: off ood %v score %v, shipped ood %v score %v; the record must carry them as computed", rec.ItemID, rec.OOD, rec.OODScore, ref.OOD, ref.OODScore)
		}
		if want := "0782d801a5a3+" + cfg.Hash8() + "+ood-off"; rec.PolicyVersion == nil || *rec.PolicyVersion != want {
			t.Errorf("%s: off version = %v, want %s", rec.ItemID, rec.PolicyVersion, want)
		}
		if ref.PolicyVersion == nil || strings.Contains(*ref.PolicyVersion, "ood-off") {
			t.Errorf("%s: shipped version = %v, want no ood-off suffix", rec.ItemID, ref.PolicyVersion)
		}
		if rec.P == nil || rec.L == nil || ref.P == nil || ref.L == nil {
			t.Fatalf("%s: missing p or l", rec.ItemID)
		}
		same := *rec.P == *ref.P && *rec.L == *ref.L
		switch {
		case rec.OOD:
			oodItems++
			if !same {
				changed++
			}
		case !same:
			t.Errorf("%s: in distribution but p/l differ between the arms (%v %v) and (%v %v)", rec.ItemID, *ref.P, *ref.L, *rec.P, *rec.L)
		}
	}
	if oodItems == 0 || changed == 0 {
		t.Errorf("%d out-of-distribution items, %d scored differently with --ood off; the flag changes nothing", oodItems, changed)
	}
}

func TestReplayCLIOODUsageErrors(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.jsonl")
	cfg := filepath.Join(learnedFixtureDir, "config.yaml")
	in := filepath.Join(learnedFixtureDir, "items.jsonl")
	cases := map[string][]string{
		"unknown value":      learnedArgs(out, "--ood", "maybe"),
		"empty value":        learnedArgs(out, "--ood", ""),
		"with cheapest":      {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "cheapest", "--ood", "off"},
		"with cheapest (on)": {"replay", "--config", cfg, "--input", in, "--out", out, "--split", "all", "--policy", "cheapest", "--ood", "on"},
	}
	for name, args := range cases {
		var stdout, stderr bytes.Buffer
		if code := run(args, &stdout, &stderr); code != 2 {
			t.Errorf("%s: run = %d, want 2; stderr = %q", name, code, stderr.String())
		}
		if !strings.Contains(stderr.String(), "--ood") || !strings.Contains(stderr.String(), "usage") {
			t.Errorf("%s: stderr = %q, want a message naming --ood and a usage line", name, stderr.String())
		}
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a usage error created the out file")
	}
}
