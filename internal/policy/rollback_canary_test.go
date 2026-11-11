package policy

import (
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

// TestRollbackCanaryStopsTheCanaryAndCountsIt: the canary share goes to 0, the old share is
// returned, and the rollback is counted exactly once, in the counter that
// rp_policy_rollbacks_total exports.
func TestRollbackCanaryStopsTheCanaryAndCountsIt(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("canary.yaml", 0.2)
	f.set(config.PoliciesConfig{
		Root: "policies", Stable: f.ref("stable.yaml"),
		Canary: &config.CanaryRef{PolicyRef: f.ref("canary.yaml"), Percent: 25},
	})
	reg := f.registry()
	before := reg.Snapshot()

	if old := reg.RollbackCanary(); old != 25 {
		t.Fatalf("RollbackCanary() = %v, want the old share 25", old)
	}
	after := reg.Snapshot()
	if after.CanaryPercent != 0 || after.Canary == nil || after.Canary.Version != before.Canary.Version {
		t.Errorf("snapshot after = percent %v canary %v, want percent 0 and the canary slot kept", after.CanaryPercent, after.Canary)
	}
	if after.Generation != before.Generation+1 {
		t.Errorf("generation = %d, want %d", after.Generation, before.Generation+1)
	}
	if n := reg.Rollbacks(); n != 1 {
		t.Errorf("Rollbacks() = %d, want 1", n)
	}

	// A second call has nothing to stop: no new snapshot and no second count.
	if old := reg.RollbackCanary(); old != 0 {
		t.Errorf("second RollbackCanary() = %v, want 0", old)
	}
	if got := reg.Snapshot(); got != after {
		t.Error("a rollback with nothing to roll back published a new snapshot")
	}
	if n := reg.Rollbacks(); n != 1 {
		t.Errorf("Rollbacks() after a second call = %d, want still 1", n)
	}
}

func TestRollbackCanaryWithoutACanary(t *testing.T) {
	reg := newFixture(t).registry()
	before := reg.Snapshot()
	if old := reg.RollbackCanary(); old != 0 {
		t.Errorf("RollbackCanary() with no canary = %v, want 0", old)
	}
	if reg.Snapshot() != before || reg.Rollbacks() != 0 {
		t.Error("a rollback with no canary changed the snapshot or the counter")
	}
}
