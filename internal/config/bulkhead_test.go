package config

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestExampleConfigBoundsEveryDeployment: the local example caps each deployment at 64 in-flight
// non-streaming attempts, so the example shows the setting.
func TestExampleConfigBoundsEveryDeployment(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "examples", "config.local.yaml"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	n := 0
	for _, m := range cfg.Models {
		for _, d := range m.Deployments {
			n++
			if d.MaxInflight != 64 {
				t.Errorf("deployment %q max_inflight = %d, want 64", d.Name, d.MaxInflight)
			}
		}
	}
	if n == 0 {
		t.Fatal("the example config has no deployments")
	}
}

// TestDeploymentMaxInflight: deployments[].max_inflight caps the in-flight non-streaming
// attempts of one deployment. Absent is 0 (unlimited), and a negative value is rejected at
// its own path.
func TestDeploymentMaxInflight(t *testing.T) {
	t.Run("absent means unlimited", func(t *testing.T) {
		cfg, err := Parse([]byte(baseYAML(t)))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if got := cfg.Models[0].Deployments[0].MaxInflight; got != 0 {
			t.Errorf("MaxInflight = %d, want 0 (unlimited)", got)
		}
	})
	t.Run("a positive limit is read", func(t *testing.T) {
		cfg, err := Parse([]byte(baseYAML(t, "upstream_model: m1", "upstream_model: m1\n        max_inflight: 64")))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if got := cfg.Models[0].Deployments[0].MaxInflight; got != 64 {
			t.Errorf("MaxInflight = %d, want 64", got)
		}
	})
	t.Run("zero is valid", func(t *testing.T) {
		if _, err := Parse([]byte(baseYAML(t, "upstream_model: m1", "upstream_model: m1\n        max_inflight: 0"))); err != nil {
			t.Errorf("Parse: %v", err)
		}
	})
	t.Run("negative is rejected", func(t *testing.T) {
		_, err := Parse([]byte(baseYAML(t, "upstream_model: m1", "upstream_model: m1\n        max_inflight: -1")))
		if err == nil {
			t.Fatal("Parse succeeded, want an error")
		}
		if want := "models[0].deployments[0].max_inflight"; !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not contain %q", err, want)
		}
	})
}
