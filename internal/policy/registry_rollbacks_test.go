package policy

import (
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

// TestRollbacksCountsOnlyRollbacksThatHappened: the counter the metrics export
// (rp_policy_rollbacks_total) moves on a rollback and on nothing else.
func TestRollbacksCountsOnlyRollbacksThatHappened(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("a.yaml", 0.1)
	f.writePolicy("b.yaml", 0.2)
	f.set(config.PoliciesConfig{Root: "policies", Stable: f.ref("a.yaml")})
	reg := f.registry()

	if n := reg.Rollbacks(); n != 0 {
		t.Fatalf("Rollbacks on a new registry = %d, want 0", n)
	}
	if _, _, err := reg.RollbackStable(); err == nil {
		t.Fatal("RollbackStable succeeded with no previous stable slot")
	}
	if n := reg.Rollbacks(); n != 0 {
		t.Errorf("Rollbacks after a refused rollback = %d, want 0", n)
	}

	f.set(config.PoliciesConfig{Root: "policies", Stable: f.ref("b.yaml")})
	if err := reg.Reload(); err != nil {
		t.Fatal(err)
	}
	if n := reg.Rollbacks(); n != 0 {
		t.Errorf("Rollbacks after a reload = %d, want 0", n)
	}
	if _, _, err := reg.RollbackStable(); err != nil {
		t.Fatal(err)
	}
	if n := reg.Rollbacks(); n != 1 {
		t.Errorf("Rollbacks after one rollback = %d, want 1", n)
	}
	if _, _, err := reg.RollbackStable(); err == nil {
		t.Fatal("a second RollbackStable succeeded")
	}
	if n := reg.Rollbacks(); n != 1 {
		t.Errorf("Rollbacks after a second, refused rollback = %d, want still 1", n)
	}
}
