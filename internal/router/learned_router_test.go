package router

import (
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

func TestNewPolicyLearnedIsNotAStaticPolicy(t *testing.T) {
	p, err := NewPolicy(config.RoutingConfig{Policy: config.PolicyLearned}, fixtureCatalog(t))
	if p != nil || err != nil {
		t.Errorf("NewPolicy(learned) = (%v, %v), want (nil, nil): the learned policy arrives with each request", p, err)
	}
}

func TestNewLearnedRouterHasAuto(t *testing.T) {
	cat := fixtureCatalog(t)
	if New(cat, nil).HasAuto() {
		t.Fatal("a router without a policy reports auto")
	}
	rt := NewLearned(cat)
	if !rt.HasAuto() {
		t.Fatal("NewLearned(...).HasAuto() = false, want true")
	}

	// Without a Learned policy in the input there is nothing to decide with.
	_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto})
	if apiErr == nil || apiErr.Code != "invalid_request" {
		t.Errorf("auto without Input.Learned: err = %v, want invalid_request", apiErr)
	}

	// Pinned requests are unaffected.
	d, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "mid"})
	if apiErr != nil || d.Policy != PolicyPinned {
		t.Errorf("pinned: decision %+v, err %v", d, apiErr)
	}
}
