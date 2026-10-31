package main

import (
	"testing"
	"time"
)

// TestReconcileIgnoresTrafficFromBeforeTheBudget: adding budget_usd to a tenant that already
// has ledger rows is the ordinary way budgets are introduced. The rows from before charged no
// counter, so reconcile leaves them out and the tenant matches once its counter and its
// budgeted rows agree.
func TestReconcileIgnoresTrafficFromBeforeTheBudget(t *testing.T) {
	e := newReconcileEnv(t, nil)
	e.ledgerIn("beta", "b-before-1", 2*usd, time.Date(2026, 8, 3, 9, 0, 0, 0, time.UTC), "")
	e.ledgerIn("beta", "b-before-2", usd, time.Date(2026, 9, 20, 9, 0, 0, 0, time.UTC), "")
	e.ledger("beta", "b-after", usd, reconcileAt) // reserved and committed under the budget
	e.counter("beta", betaLim, usd)

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 0 {
		t.Errorf("exit code = %d, want 0; stderr: %s", code, stderr)
	}
	if want := "alpha 0 0 0\nbeta 1 1 0\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
}

// TestReconcileStillSeesALostCommitAmongOldTraffic: leaving the old rows out must not hide a
// budgeted request whose commit never reached the counter.
func TestReconcileStillSeesALostCommitAmongOldTraffic(t *testing.T) {
	e := newReconcileEnv(t, nil)
	e.ledgerIn("beta", "b-before", 2*usd, time.Date(2026, 8, 3, 9, 0, 0, 0, time.UTC), "")
	e.ledger("beta", "b-kept", usd, reconcileAt)
	e.ledger("beta", "b-lost", usd/2, reconcileAt) // its commit was lost
	e.counter("beta", betaLim, usd)

	code, stdout, _ := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 1 || stdout != "alpha 0 0 0\nbeta 1.5 1 0.5\n" {
		t.Errorf("exit %d stdout %q, want exit 1 and the line %q", code, stdout, "beta 1.5 1 0.5")
	}
}

// TestReconcileCountsARequestInTheMonthItReservedIn: a request that reserved on the last
// evening of September and ended after midnight is September's for the counter and for its
// row, so October and September both reconcile.
func TestReconcileCountsARequestInTheMonthItReservedIn(t *testing.T) {
	reserved := time.Date(2026, 9, 30, 23, 59, 59, 0, time.UTC)
	ended := time.Date(2026, 10, 1, 0, 0, 5, 0, time.UTC)
	e := newReconcileEnv(t, nil)
	e.counterAt("alpha", alphaLim, usd, reserved)
	e.ledgerIn("alpha", "a-crossing", usd, ended, "2026-09")
	e.counter("alpha", alphaLim, 2*usd)
	e.ledger("alpha", "a-october", 2*usd, reconcileAt)
	e.counter("beta", betaLim, 0)

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))
	if code != 0 || stdout != "alpha 2 2 0\nbeta 0 0 0\n" {
		t.Errorf("October: exit %d stdout %q stderr %q, want exit 0 and alpha 2 2 0", code, stdout, stderr)
	}

	code, stdout, stderr = e.run("--at", reserved.Format(time.RFC3339))
	if code != 0 || stdout != "alpha 1 1 0\nbeta 0 0 0\n" {
		t.Errorf("September: exit %d stdout %q stderr %q, want exit 0 and alpha 1 1 0", code, stdout, stderr)
	}
}
