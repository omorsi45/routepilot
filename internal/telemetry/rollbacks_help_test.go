package telemetry

import (
	"strings"
	"testing"
)

// TestRollbackCounterHelpSaysWhatItCounts: the registry counts a rollback of the stable slot
// and a rollback of the canary, by an operator or by the guardrails, so the help text, which
// dashboards and alert rules quote, must not say stable only.
func TestRollbackCounterHelpSaysWhatItCounts(t *testing.T) {
	m, reg := newTestMetrics()
	m.SetSources(Sources{PolicyRollbacks: func() uint64 { return 1 }})

	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range families {
		if f.GetName() != "rp_policy_rollbacks_total" {
			continue
		}
		help := f.GetHelp()
		for _, want := range []string{"stable policy", "canary", "guardrails"} {
			if !strings.Contains(help, want) {
				t.Errorf("help %q does not mention %q", help, want)
			}
		}
		return
	}
	t.Fatal("rp_policy_rollbacks_total is not exported")
}
