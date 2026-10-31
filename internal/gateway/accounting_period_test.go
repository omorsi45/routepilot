package gateway

import (
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/quota"
)

// TestUsageRowsNameTheBudgetPeriodTheyWereChargedTo: every row of a tenant with a budget
// carries the period key its reservation was made in, which routepilot reconcile matches
// against the counter's period; a tenant without a budget writes an empty key.
func TestUsageRowsNameTheBudgetPeriodTheyWereChargedTo(t *testing.T) {
	cases := []struct {
		period string
		want   string
	}{
		{quota.PeriodMonthly, "2026-10"},
		{quota.PeriodTotal, "total"},
	}
	for _, tc := range cases {
		t.Run(tc.period, func(t *testing.T) {
			e := newBudgetEnv(t, budgetOpts{budget: 100 * stdActual, period: tc.period, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
			store := &recordingStore{}
			useLedger(t, e.gw, store)

			if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
				t.Fatalf("tenant-a: status %d: %s", rec.Code, rec.Body.String())
			}
			if rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyB, chatBody("auto", "hi")); rec.Code != http.StatusOK {
				t.Fatalf("tenant-b: status %d: %s", rec.Code, rec.Body.String())
			}
			flushUsage(t, e.gw)

			got := map[string]string{}
			for _, r := range store.got() {
				got[r.Tenant] = r.BudgetPeriod
			}
			if len(got) != 2 || got["tenant-a"] != tc.want || got["tenant-b"] != "" {
				t.Errorf("budget periods by tenant = %q, want tenant-a %q and tenant-b empty", got, tc.want)
			}
		})
	}
}

// TestUsageRowKeepsThePeriodOfTheReservationAcrossAMonthBoundary: a request reserved on
// the last evening of the month and finished after midnight is charged by the ledger to the
// month it reserved in, and its row says the same.
func TestUsageRowKeepsThePeriodOfTheReservationAcrossAMonthBoundary(t *testing.T) {
	var mu sync.Mutex
	now := time.Date(2026, 10, 31, 23, 59, 59, 0, time.UTC)
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	e := newBudgetEnv(t, budgetOpts{budget: 100 * stdActual, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}, now: clock})
	e.setOnUpstream(func(string) { // the request is in flight when the month turns
		mu.Lock()
		now = time.Date(2026, 11, 1, 0, 0, 1, 0, time.UTC)
		mu.Unlock()
	})
	store := &recordingStore{}
	useLedger(t, e.gw, store)

	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	flushUsage(t, e.gw)

	rows := store.got()
	if len(rows) != 1 || rows[0].BudgetPeriod != "2026-10" {
		t.Fatalf("rows = %+v, want one row charged to 2026-10", rows)
	}
}
