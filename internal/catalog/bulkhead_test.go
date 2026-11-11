package catalog

import (
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

// TestDeploymentCarriesMaxInflight: the catalog copies deployments[].max_inflight, so the
// gateway can size its per-deployment bulkhead from the catalog alone.
func TestDeploymentCarriesMaxInflight(t *testing.T) {
	cfg := &config.Config{Models: []config.ModelConfig{{
		ID:                 "m",
		InputPricePerMTok:  "1",
		OutputPricePerMTok: "1",
		Deployments: []config.DeploymentConfig{
			{Name: "limited", Provider: "p", UpstreamModel: "u1", MaxInflight: 64},
			{Name: "unlimited", Provider: "p", UpstreamModel: "u2"},
		},
	}}}
	cat, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for name, want := range map[string]int{"limited": 64, "unlimited": 0} {
		_, dep, ok := cat.Deployment(name)
		if !ok {
			t.Fatalf("deployment %q not found", name)
		}
		if dep.MaxInflight != want {
			t.Errorf("deployment %q MaxInflight = %d, want %d", name, dep.MaxInflight, want)
		}
	}
	m, _ := cat.Model("m")
	if m.Deployments[0].MaxInflight != 64 || m.Deployments[1].MaxInflight != 0 {
		t.Errorf("model deployments = %+v, want MaxInflight 64 then 0", m.Deployments)
	}
}
