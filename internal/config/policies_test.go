package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// learnedYAML is the base config with routing.policy learned and the given policies block
// (already indented under a top-level "policies:" key by the caller).
func learnedYAML(t *testing.T, policies string) string {
	t.Helper()
	y := baseYAML(t, "tenants:", "routing:\n  policy: learned\ntenants:")
	if policies != "" {
		y += "policies:\n" + policies
	}
	return y
}

const stableBlock = `  stable:
    name: flagship
    version: abc123def456
    config: configs/flagship.yaml
`

func TestParseLearnedPolicies(t *testing.T) {
	cfg, err := Parse([]byte(learnedYAML(t, stableBlock)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Routing.Policy != PolicyLearned || PolicyLearned != "learned" {
		t.Errorf("routing.policy = %q, PolicyLearned = %q, want learned", cfg.Routing.Policy, PolicyLearned)
	}
	p := cfg.Policies
	if p.Root != "policies" {
		t.Errorf("policies.root = %q, want the default \"policies\"", p.Root)
	}
	if want := (PolicyRef{Name: "flagship", Version: "abc123def456", Config: "configs/flagship.yaml"}); p.Stable != want {
		t.Errorf("stable = %+v, want %+v", p.Stable, want)
	}
	if p.Canary != nil || len(p.Shadow) != 0 {
		t.Errorf("canary %v shadow %v, want none", p.Canary, p.Shadow)
	}
}

func TestParseLearnedPoliciesFull(t *testing.T) {
	block := `  root: artifacts
` + stableBlock + `  canary:
    name: flagship
    version: 111111111111
    config: configs/canary.yaml
    percent: 5.5
  shadow:
    - name: flagship
      version: 222222222222
      config: configs/shadow-a.yaml
    - name: mixed
      version: 333333333333
      config: configs/shadow-b.yaml
`
	cfg, err := Parse([]byte(learnedYAML(t, block)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	p := cfg.Policies
	if p.Root != "artifacts" {
		t.Errorf("root = %q", p.Root)
	}
	if p.Canary == nil || p.Canary.Name != "flagship" || p.Canary.Version != "111111111111" ||
		p.Canary.Config != "configs/canary.yaml" || p.Canary.Percent != 5.5 {
		t.Errorf("canary = %+v", p.Canary)
	}
	if len(p.Shadow) != 2 || p.Shadow[1].Name != "mixed" || p.Shadow[0].Config != "configs/shadow-a.yaml" {
		t.Errorf("shadow = %+v", p.Shadow)
	}
}

func TestParseLearnedPoliciesRejects(t *testing.T) {
	cases := map[string]struct{ yaml, want string }{
		"learned without policies":  {learnedYAML(t, ""), "policies.stable"},
		"learned with empty stable": {learnedYAML(t, "  root: x\n"), "policies.stable"},
		"stable without name":       {learnedYAML(t, strings.Replace(stableBlock, "name: flagship", "name: \"\"", 1)), "policies.stable.name"},
		"stable without version":    {learnedYAML(t, strings.Replace(stableBlock, "version: abc123def456", "version: \"\"", 1)), "policies.stable.version"},
		"stable without config":     {learnedYAML(t, strings.Replace(stableBlock, "config: configs/flagship.yaml", "config: \"\"", 1)), "policies.stable.config"},
		"name with a separator":     {learnedYAML(t, strings.Replace(stableBlock, "name: flagship", "name: a/b", 1)), "policies.stable.name"},
		"name is dotdot":            {learnedYAML(t, strings.Replace(stableBlock, "name: flagship", "name: ..", 1)), "policies.stable.name"},
		"version with backslash":    {learnedYAML(t, strings.Replace(stableBlock, "version: abc123def456", `version: 'a\b'`, 1)), "policies.stable.version"},
		"canary percent above 100": {learnedYAML(t, stableBlock+"  canary:\n    name: a\n    version: b\n    config: c\n    percent: 100.1\n"),
			"policies.canary.percent"},
		"canary percent negative": {learnedYAML(t, stableBlock+"  canary:\n    name: a\n    version: b\n    config: c\n    percent: -1\n"),
			"policies.canary.percent"},
		"canary percent nan": {learnedYAML(t, stableBlock+"  canary:\n    name: a\n    version: b\n    config: c\n    percent: .nan\n"),
			"policies.canary.percent"},
		"canary without config": {learnedYAML(t, stableBlock+"  canary:\n    name: a\n    version: b\n    percent: 5\n"),
			"policies.canary.config"},
		"shadow without version": {learnedYAML(t, stableBlock+"  shadow:\n    - name: a\n      config: c\n"),
			"policies.shadow[0].version"},
		"unknown policies key": {learnedYAML(t, stableBlock+"  nope: 1\n"), "nope"},
		"unknown ref key":      {learnedYAML(t, stableBlock+"    extra: 1\n"), "extra"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Parse([]byte(tc.yaml))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want one naming %q", err, tc.want)
			}
		})
	}
}

func TestPoliciesWithoutLearnedAreOptional(t *testing.T) {
	// A static policy does not need the section, and a section that is present is still
	// checked.
	if _, err := Parse([]byte(baseYAML(t))); err != nil {
		t.Fatalf("config without policies: %v", err)
	}
	y := baseYAML(t) + "policies:\n" + stableBlock
	if _, err := Parse([]byte(y)); err != nil {
		t.Fatalf("static config with a valid policies section: %v", err)
	}
	bad := baseYAML(t) + "policies:\n  stable:\n    name: only-a-name\n"
	if _, err := Parse([]byte(bad)); err == nil {
		t.Fatal("a partial stable ref was accepted")
	}
}

func TestLoadRecordsTheConfigDirectory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "routepilot.yaml")
	if err := os.WriteFile(path, []byte(baseYAML(t)), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := filepath.Clean(cfg.BaseDir()), filepath.Clean(dir); got != want {
		t.Errorf("BaseDir() = %q, want %q", got, want)
	}
	if got := filepath.Clean(cfg.Path()); got != filepath.Clean(path) {
		t.Errorf("Path() = %q, want %q", got, path)
	}

	parsed, err := Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatal(err)
	}
	if parsed.BaseDir() != "" || parsed.Path() != "" {
		t.Errorf("Parse left BaseDir %q, Path %q, want both empty", parsed.BaseDir(), parsed.Path())
	}
}

func TestLoadBaseDirOfARelativePathIsAbsolute(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "c.yaml"), []byte(baseYAML(t)), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	cfg, err := Load("c.yaml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !filepath.IsAbs(cfg.BaseDir()) {
		t.Errorf("BaseDir() = %q, want an absolute path so a later chdir cannot move it", cfg.BaseDir())
	}
}

func TestResolvePath(t *testing.T) {
	abs := filepath.Join(t.TempDir(), "x.yaml")
	cases := []struct{ base, p, want string }{
		{filepath.Join("a", "b"), "policies", filepath.Join("a", "b", "policies")},
		{filepath.Join("a", "b"), filepath.Join("..", "c.yaml"), filepath.Join("a", "c.yaml")},
		{"", "policies", "policies"},
		{filepath.Join("a", "b"), abs, abs},
	}
	for _, tc := range cases {
		if got := ResolvePath(tc.base, tc.p); got != tc.want {
			t.Errorf("ResolvePath(%q, %q) = %q, want %q", tc.base, tc.p, got, tc.want)
		}
	}
}

func TestParseServerLatencyProfiles(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.LatencyProfiles != "" {
		t.Errorf("latency_profiles = %q by default, want empty", cfg.Server.LatencyProfiles)
	}
	y := baseYAML(t, "server:\n", "server:\n  latency_profiles: bench/latency/profiles.yaml\n")
	cfg, err = Parse([]byte(y))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Server.LatencyProfiles != "bench/latency/profiles.yaml" {
		t.Errorf("latency_profiles = %q", cfg.Server.LatencyProfiles)
	}
}
