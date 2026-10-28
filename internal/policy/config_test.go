package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validPolicyYAML = `mode: dual
lambda: 2.5
default_floor: 0.4
epsilon: 0.05
ood_action: strongest
on_infeasible: reject
`

func writeConfigFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfig(t *testing.T) {
	got, err := LoadConfig(writeConfigFile(t, validPolicyYAML))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	want := Config{Mode: "dual", Lambda: 2.5, DefaultFloor: 0.4, Epsilon: 0.05, OODAction: "strongest", OnInfeasible: "reject"}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}

	// Numeric fields may be left out: zero is valid for all three.
	got, err = LoadConfig(writeConfigFile(t, "mode: min_cost\nood_action: conservative\non_infeasible: best_effort\n"))
	if err != nil || got.Lambda != 0 || got.DefaultFloor != 0 || got.Epsilon != 0 {
		t.Errorf("minimal config: %+v, %v", got, err)
	}
}

func TestLoadConfigRejects(t *testing.T) {
	replace := func(old, new string) string { return strings.Replace(validPolicyYAML, old, new, 1) }
	cases := map[string]struct{ body, want string }{
		"empty file":             {"", "empty"},
		"unknown key":            {validPolicyYAML + "bogus: 1\n", "bogus"},
		"two documents":          {validPolicyYAML + "---\nmode: dual\n", "document"},
		"missing mode":           {replace("mode: dual\n", ""), "mode"},
		"unknown mode":           {replace("mode: dual", "mode: greedy"), "mode"},
		"negative lambda":        {replace("lambda: 2.5", "lambda: -1"), "lambda"},
		"nan lambda":             {replace("lambda: 2.5", "lambda: .nan"), "lambda"},
		"infinite lambda":        {replace("lambda: 2.5", "lambda: .inf"), "lambda"},
		"floor above one":        {replace("default_floor: 0.4", "default_floor: 1.01"), "default_floor"},
		"floor below zero":       {replace("default_floor: 0.4", "default_floor: -0.01"), "default_floor"},
		"nan floor":              {replace("default_floor: 0.4", "default_floor: .nan"), "default_floor"},
		"epsilon of one":         {replace("epsilon: 0.05", "epsilon: 1"), "epsilon"},
		"negative epsilon":       {replace("epsilon: 0.05", "epsilon: -0.1"), "epsilon"},
		"missing ood action":     {replace("ood_action: strongest\n", ""), "ood_action"},
		"unknown ood action":     {replace("ood_action: strongest", "ood_action: ignore"), "ood_action"},
		"missing on_infeasible":  {replace("on_infeasible: reject\n", ""), "on_infeasible"},
		"unknown on_infeasible":  {replace("on_infeasible: reject", "on_infeasible: retry"), "on_infeasible"},
		"wrong type for epsilon": {replace("epsilon: 0.05", "epsilon: lots"), "cannot unmarshal"},
	}
	for name, tc := range cases {
		_, err := LoadConfig(writeConfigFile(t, tc.body))
		if err == nil {
			t.Errorf("%s: want an error", name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %q does not mention %q", name, err, tc.want)
		}
	}
	if _, err := LoadConfig(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Error("a missing file: want an error")
	}
}

func TestConfigValidateCoversOverrides(t *testing.T) {
	base := Config{Mode: ModeDual, OODAction: OODConservative, OnInfeasible: "best_effort"}
	if err := base.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	for name, mutate := range map[string]func(*Config){
		"mode":          func(c *Config) { c.Mode = "x" },
		"lambda":        func(c *Config) { c.Lambda = math.Inf(1) },
		"default_floor": func(c *Config) { c.DefaultFloor = 2 },
		"epsilon":       func(c *Config) { c.Epsilon = 1 },
		"ood_action":    func(c *Config) { c.OODAction = "" },
		"on_infeasible": func(c *Config) { c.OnInfeasible = "x" },
	} {
		c := base
		mutate(&c)
		if err := c.Validate(); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: err = %v, want one naming the field", name, err)
		}
	}
}

func TestConfigHash8(t *testing.T) {
	c := Config{Mode: "dual", Lambda: 2.5, DefaultFloor: 0.4, Epsilon: 0.05, OODAction: "conservative", OnInfeasible: "best_effort"}
	sum := sha256.Sum256([]byte(`{"Mode":"dual","Lambda":2.5,"DefaultFloor":0.4,"Epsilon":0.05,"OODAction":"conservative","OnInfeasible":"best_effort"}`))
	want := hex.EncodeToString(sum[:])[:8]
	if got := c.Hash8(); got != want || len(got) != 8 {
		t.Errorf("Hash8 = %q, want %q (first 8 hex of the SHA-256 of the JSON)", got, want)
	}
	if c.Hash8() != c.Hash8() {
		t.Error("Hash8 is not deterministic")
	}
	seen := map[string]string{"base": c.Hash8()}
	for name, mutate := range map[string]func(*Config){
		"mode":          func(c *Config) { c.Mode = "min_cost" },
		"lambda":        func(c *Config) { c.Lambda = 2.6 },
		"default_floor": func(c *Config) { c.DefaultFloor = 0.5 },
		"epsilon":       func(c *Config) { c.Epsilon = 0.06 },
		"ood_action":    func(c *Config) { c.OODAction = "strongest" },
		"on_infeasible": func(c *Config) { c.OnInfeasible = "reject" },
	} {
		d := c
		mutate(&d)
		h := d.Hash8()
		if other, dup := seen[h]; dup {
			t.Errorf("changing %s gives the same hash as %s", name, other)
		}
		seen[h] = name
	}
}

func TestShippedFlagshipPolicyConfig(t *testing.T) {
	got, err := LoadConfig(filepath.Join("..", "..", "bench", "policies", "flagship.yaml"))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	// lambda is the value bench/scripts/sweeps.py tuned on the calibration split (see
	// bench/results/sweeps); a new tuning changes it here in the same commit.
	want := Config{Mode: ModeDual, Lambda: 6.719579687342048, DefaultFloor: 0, Epsilon: 0, OODAction: OODConservative, OnInfeasible: "best_effort"}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}
