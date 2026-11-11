// Package accountingtest is the conformance suite every accounting.Store backend passes.
// A backend's test opens an empty store per subtest and calls Run.
package accountingtest

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/money"
)

// Factory opens an empty store for one subtest and arranges for it to be closed and
// cleaned up when the subtest ends.
type Factory func(t *testing.T) accounting.Store

// base is the instant the fixtures are built around. It has no sub-second part on purpose:
// the window tests probe the instants just before and after it. Backends are compared at
// microsecond resolution.
var base = time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)

// farFuture and the zero time bound a window that holds every row.
var farFuture = time.Date(9999, 12, 31, 0, 0, 0, 0, time.UTC)

func ctx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return c
}

// row is a complete usage row that ended at ended. Its estimate differs from its actual cost
// so a query that sums the wrong column is caught.
func row(request string, attempt int, tenant string, actual money.PicoUSD, ended time.Time) accounting.UsageRow {
	return accounting.UsageRow{
		RequestID: request, Attempt: attempt, Tenant: tenant,
		Model: "model-x", Deployment: "dep-x",
		PromptTokens: 11, CompletionTokens: 7,
		CostEstimate: actual*2 + 5, CostActual: actual,
		PolicyVersion: "pv-1", Status: "ok",
		StartedAt: ended.Add(-250 * time.Millisecond), EndedAt: ended,
	}
}

func mustRecord(t *testing.T, s accounting.Store, r accounting.UsageRow) {
	t.Helper()
	if err := s.RecordAttempt(ctx(t), r); err != nil {
		t.Fatalf("RecordAttempt(%s/%d) error = %v", r.RequestID, r.Attempt, err)
	}
}

func spend(t *testing.T, s accounting.Store, tenant string, from, to time.Time) money.PicoUSD {
	t.Helper()
	got, err := s.TenantSpend(ctx(t), tenant, from, to)
	if err != nil {
		t.Fatalf("TenantSpend(%s, %s, %s) error = %v", tenant, from, to, err)
	}
	return got
}

// Run runs every conformance case against stores from open.
func Run(t *testing.T, open Factory) {
	t.Helper()
	cases := []struct {
		name string
		fn   func(*testing.T, accounting.Store)
	}{
		{"Idempotent", idempotent},
		{"BatchIdempotent", batchIdempotent},
		{"BatchRejectsInvalidRows", batchRejectsInvalidRows},
		{"BatchHonoursContext", batchHonoursContext},
		{"BatchAfterCloseFails", batchAfterClose},
		{"RejectsInvalidRows", rejectsInvalidRows},
		{"SpendSumsActualInWindow", spendSumsActualInWindow},
		{"SpendWindowIsExactToTheMicrosecond", spendWindowMicroseconds},
		{"SpendIgnoresTheZoneOfTheTimestamps", spendIgnoresZone},
		{"SpendInPeriodSumsTheRowsChargedToIt", spendInPeriod},
		{"RolloutsNewestFirst", rolloutsNewestFirst},
		{"RolloutsRoundTrip", rolloutsRoundTrip},
		{"RejectsInvalidRollouts", rejectsInvalidRollouts},
		{"ConcurrentWriters", concurrentWriters},
		{"HonoursContext", honoursContext},
		{"CloseIsIdempotentAndEndsUse", closeEndsUse},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { c.fn(t, open(t)) })
	}
}

// idempotent: a second row with the same (request, attempt) changes nothing, whatever it
// says; another attempt of the same request is a different row.
func idempotent(t *testing.T, s accounting.Store) {
	mustRecord(t, s, row("req-1", 1, "tenant-a", 100, base))
	again := row("req-1", 1, "tenant-b", 999, base.Add(time.Minute))
	again.Status = "http_500"
	mustRecord(t, s, again)

	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 100 {
		t.Errorf("tenant-a spend after a repeated attempt = %d, want the first row's 100", got)
	}
	if got := spend(t, s, "tenant-b", time.Time{}, farFuture); got != 0 {
		t.Errorf("tenant-b spend = %d, want 0: the repeated row must be ignored whole", got)
	}

	mustRecord(t, s, row("req-1", 2, "tenant-a", 50, base))
	mustRecord(t, s, row("req-2", 1, "tenant-a", 7, base))
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 157 {
		t.Errorf("spend with attempt 2 and another request = %d, want 157", got)
	}
}

// batchIdempotent: a batch writes each (request, attempt) once, the first copy winning,
// whether the key is already stored or appears twice in the batch; repeating a batch changes
// nothing, and an empty batch is not an error.
func batchIdempotent(t *testing.T, s accounting.Store) {
	mustRecord(t, s, row("req-1", 1, "tenant-a", 100, base))
	again := row("req-1", 1, "tenant-b", 999, base.Add(time.Minute))
	again.Status = "http_500"
	dupFirst := row("req-2", 1, "tenant-a", 7, base)
	dupSecond := row("req-2", 1, "tenant-b", 555, base)
	batch := []accounting.UsageRow{
		again, row("req-1", 2, "tenant-a", 50, base), dupFirst, dupSecond, row("req-3", 1, "tenant-a", 3, base),
	}

	for pass := 1; pass <= 2; pass++ {
		if err := s.RecordAttempts(ctx(t), batch); err != nil {
			t.Fatalf("RecordAttempts pass %d error = %v", pass, err)
		}
		if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 160 {
			t.Errorf("pass %d: tenant-a spend = %d, want 160 (100 stored + 50 + 7 + 3, each key once)", pass, got)
		}
		if got := spend(t, s, "tenant-b", time.Time{}, farFuture); got != 0 {
			t.Errorf("pass %d: tenant-b spend = %d, want 0: a repeated key must be ignored whole", pass, got)
		}
	}
	if err := s.RecordAttempts(ctx(t), nil); err != nil {
		t.Errorf("RecordAttempts of an empty batch error = %v, want nil", err)
	}
}

// batchRejectsInvalidRows: one bad row rejects the batch and none of its good rows lands.
func batchRejectsInvalidRows(t *testing.T, s accounting.Store) {
	bad := row("req-bad", 1, "tenant-a", 100, base)
	bad.Tenant = ""
	batch := []accounting.UsageRow{row("req-good", 1, "tenant-a", 10, base), bad, row("req-good", 2, "tenant-a", 20, base)}
	if err := s.RecordAttempts(ctx(t), batch); err == nil {
		t.Error("RecordAttempts with an invalid row succeeded, want an error")
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 0 {
		t.Errorf("spend after a rejected batch = %d, want 0: no row of it may land", got)
	}
}

// batchHonoursContext: a canceled context writes nothing.
func batchHonoursContext(t *testing.T, s accounting.Store) {
	done, cancel := context.WithCancel(context.Background())
	cancel()
	batch := []accounting.UsageRow{row("req-ctx", 1, "tenant-a", 100, base), row("req-ctx", 2, "tenant-a", 100, base)}
	if err := s.RecordAttempts(done, batch); err == nil {
		t.Error("RecordAttempts with a canceled context succeeded, want an error")
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 0 {
		t.Errorf("spend = %d, want 0: a canceled batch must not land", got)
	}
}

// batchAfterClose: using a closed store is an error, for a batch as for a row.
func batchAfterClose(t *testing.T, s accounting.Store) {
	if err := s.Close(); err != nil {
		t.Fatalf("Close error = %v", err)
	}
	if err := s.RecordAttempts(ctx(t), []accounting.UsageRow{row("req-closed", 1, "tenant-a", 1, base)}); err == nil {
		t.Error("RecordAttempts after Close succeeded, want an error")
	}
}

func rejectsInvalidRows(t *testing.T, s accounting.Store) {
	for name, mutate := range map[string]func(*accounting.UsageRow){
		"empty request id":  func(r *accounting.UsageRow) { r.RequestID = "" },
		"attempt zero":      func(r *accounting.UsageRow) { r.Attempt = 0 },
		"negative attempt":  func(r *accounting.UsageRow) { r.Attempt = -1 },
		"empty tenant":      func(r *accounting.UsageRow) { r.Tenant = "" },
		"negative cost":     func(r *accounting.UsageRow) { r.CostActual = -1 },
		"negative estimate": func(r *accounting.UsageRow) { r.CostEstimate = -1 },
		"negative tokens":   func(r *accounting.UsageRow) { r.PromptTokens = -1 },
	} {
		r := row("req-bad", 1, "tenant-a", 100, base)
		mutate(&r)
		if err := s.RecordAttempt(ctx(t), r); err == nil {
			t.Errorf("RecordAttempt with %s succeeded, want an error", name)
		}
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 0 {
		t.Errorf("spend after only invalid rows = %d, want 0", got)
	}
}

// spendSumsActualInWindow: the sum is of CostActual, per tenant, over ended_at in [from, to).
func spendSumsActualInWindow(t *testing.T, s accounting.Store) {
	mustRecord(t, s, row("before", 1, "tenant-a", 1, base.Add(-time.Nanosecond)))
	mustRecord(t, s, row("at-from", 1, "tenant-a", 10, base))
	mustRecord(t, s, row("inside", 1, "tenant-a", 100, base.Add(time.Hour)))
	mustRecord(t, s, row("at-to", 1, "tenant-a", 1000, base.Add(2*time.Hour)))
	mustRecord(t, s, row("after", 1, "tenant-a", 10000, base.Add(3*time.Hour)))
	mustRecord(t, s, row("other-tenant", 1, "tenant-b", 100000, base.Add(time.Hour)))

	if got := spend(t, s, "tenant-a", base, base.Add(2*time.Hour)); got != 110 {
		t.Errorf("[from, to) spend = %d, want 110: from is inclusive and to is exclusive", got)
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 11111 {
		t.Errorf("spend over all time = %d, want 11111", got)
	}
	if got := spend(t, s, "tenant-b", base, base.Add(2*time.Hour)); got != 100000 {
		t.Errorf("tenant-b spend = %d, want 100000", got)
	}
	if got := spend(t, s, "nobody", time.Time{}, farFuture); got != 0 {
		t.Errorf("spend of an unknown tenant = %d, want 0", got)
	}
	if got := spend(t, s, "tenant-a", base, base); got != 0 {
		t.Errorf("spend over an empty window = %d, want 0", got)
	}
	if got := spend(t, s, "tenant-a", base.Add(2*time.Hour), base); got != 0 {
		t.Errorf("spend over an inverted window = %d, want 0", got)
	}
}

// spendWindowMicroseconds: instants a microsecond or half a second apart order correctly,
// including a whole second against a fractional one, which a text column holding trimmed
// RFC 3339 would get wrong ("...:00Z" sorts after "...:00.5Z"). The microsecond is the
// resolution every backend keeps: PostgreSQL's timestamptz stores nothing finer, and the
// SQLite tests check separately that its text keeps all nine digits.
func spendWindowMicroseconds(t *testing.T, s accounting.Store) {
	mustRecord(t, s, row("whole", 1, "tenant-a", 1, base))
	mustRecord(t, s, row("half", 1, "tenant-a", 10, base.Add(500*time.Millisecond)))
	mustRecord(t, s, row("micro", 1, "tenant-a", 100, base.Add(500*time.Millisecond+time.Microsecond)))

	for _, c := range []struct {
		name     string
		from, to time.Time
		want     money.PicoUSD
	}{
		{"whole second only", base.Add(-time.Microsecond), base.Add(time.Microsecond), 1},
		{"half second only", base.Add(time.Microsecond), base.Add(500*time.Millisecond + time.Microsecond), 10},
		{"micro only", base.Add(500*time.Millisecond + time.Microsecond), base.Add(time.Second), 100},
		{"from after the whole second", base.Add(500 * time.Millisecond), farFuture, 110},
		{"to before the half second", time.Time{}, base.Add(500 * time.Millisecond), 1},
		{"to is exclusive at the microsecond", time.Time{}, base.Add(500*time.Millisecond + time.Microsecond), 11},
	} {
		if got := spend(t, s, "tenant-a", c.from, c.to); got != c.want {
			t.Errorf("%s: spend = %d, want %d", c.name, got, c.want)
		}
	}
}

// spendIgnoresZone: a row stamped in another zone and a window given in UTC compare as
// the instants they are.
func spendIgnoresZone(t *testing.T, s accounting.Store) {
	zone := time.FixedZone("UTC+5:30", 5*3600+1800)
	mustRecord(t, s, row("zoned", 1, "tenant-a", 5, base.In(zone)))

	if got := spend(t, s, "tenant-a", base, base.Add(time.Second)); got != 5 {
		t.Errorf("UTC window over a row stamped in UTC+5:30: spend = %d, want 5", got)
	}
	if got := spend(t, s, "tenant-a", base.In(zone), base.Add(time.Second).In(zone)); got != 5 {
		t.Errorf("window given in UTC+5:30: spend = %d, want 5", got)
	}
	if got := spend(t, s, "tenant-a", base.Add(time.Second), farFuture); got != 0 {
		t.Errorf("window after the instant: spend = %d, want 0", got)
	}
}

// spendInPeriod: TenantSpendInPeriod sums the rows charged to one budget period of one
// tenant, whatever instant they ended at: a row that ended in October but belongs to a
// September reservation is September's, and a row with no period (a tenant without a
// budget, or a row written before the column existed) is the empty period's.
func spendInPeriod(t *testing.T, s accounting.Store) {
	inPeriod := func(tenant, period string) money.PicoUSD {
		t.Helper()
		got, err := s.TenantSpendInPeriod(ctx(t), tenant, period)
		if err != nil {
			t.Fatalf("TenantSpendInPeriod(%s, %q) error = %v", tenant, period, err)
		}
		return got
	}
	withPeriod := func(r accounting.UsageRow, period string) accounting.UsageRow {
		r.BudgetPeriod = period
		return r
	}
	mustRecord(t, s, withPeriod(row("req-1", 1, "tenant-a", 100, base), "2026-10"))
	mustRecord(t, s, withPeriod(row("req-1", 2, "tenant-a", 30, base), "2026-10"))
	mustRecord(t, s, withPeriod(row("req-2", 1, "tenant-a", 50, base), "2026-09")) // ended in October, reserved in September
	mustRecord(t, s, row("req-3", 1, "tenant-a", 7, base))                         // charged to no budget
	mustRecord(t, s, withPeriod(row("req-4", 1, "tenant-b", 9, base), "2026-10"))
	if err := s.RecordAttempts(ctx(t), []accounting.UsageRow{
		withPeriod(row("req-5", 1, "tenant-a", 4, base), "total"),
		withPeriod(row("req-1", 1, "tenant-a", 999, base), "total"), // a stored key: ignored whole
	}); err != nil {
		t.Fatalf("RecordAttempts error = %v", err)
	}

	for _, c := range []struct {
		tenant, period string
		want           money.PicoUSD
	}{
		{"tenant-a", "2026-10", 130},
		{"tenant-a", "2026-09", 50},
		{"tenant-a", "total", 4},
		{"tenant-a", "", 7},
		{"tenant-a", "2026-11", 0},
		{"tenant-b", "2026-10", 9},
		{"tenant-b", "2026-09", 0},
		{"nobody", "2026-10", 0},
	} {
		if got := inPeriod(c.tenant, c.period); got != c.want {
			t.Errorf("TenantSpendInPeriod(%s, %q) = %d, want %d", c.tenant, c.period, got, c.want)
		}
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 100+30+50+7+4 {
		t.Errorf("TenantSpend over all time = %d, want 191: the period does not change the window sum", got)
	}
}

func event(at time.Time, action, reason string) accounting.RolloutEvent {
	return accounting.RolloutEvent{At: at, Action: action, From: "v1", To: "v2", Percent: 5, Reason: reason, Actor: "ops"}
}

// rolloutsNewestFirst: by time, not by insertion order, and by insertion order for equal
// times (the later insert first).
func rolloutsNewestFirst(t *testing.T, s accounting.Store) {
	if got, err := s.Rollouts(ctx(t), 10); err != nil || len(got) != 0 {
		t.Fatalf("Rollouts on an empty store = %v, %v, want none", got, err)
	}
	for _, e := range []accounting.RolloutEvent{
		event(base.Add(1*time.Minute), accounting.ActionReload, "first"),
		event(base.Add(3*time.Minute), accounting.ActionCanary, "third"),
		event(base.Add(2*time.Minute), accounting.ActionPromote, "second"),
		event(base.Add(3*time.Minute), accounting.ActionRollback, "third again"),
	} {
		if err := s.RecordRollout(ctx(t), e); err != nil {
			t.Fatalf("RecordRollout(%s) error = %v", e.Reason, err)
		}
	}

	got, err := s.Rollouts(ctx(t), 10)
	if err != nil {
		t.Fatalf("Rollouts error = %v", err)
	}
	var reasons []string
	for _, e := range got {
		reasons = append(reasons, e.Reason)
	}
	want := []string{"third again", "third", "second", "first"}
	if fmt.Sprint(reasons) != fmt.Sprint(want) {
		t.Errorf("Rollouts order = %v, want %v", reasons, want)
	}

	if got, _ := s.Rollouts(ctx(t), 2); len(got) != 2 || got[0].Reason != "third again" || got[1].Reason != "third" {
		t.Errorf("Rollouts(limit 2) = %v, want the two newest", got)
	}
}

func rolloutsRoundTrip(t *testing.T, s accounting.Store) {
	in := accounting.RolloutEvent{
		At: base.Add(123456 * time.Microsecond), Action: accounting.ActionAutoRollback,
		From: "abc+1234", To: "def+5678", Percent: 12.5, Reason: "p95 regression: 3 > 2", Actor: "controller",
	}
	if err := s.RecordRollout(ctx(t), in); err != nil {
		t.Fatalf("RecordRollout error = %v", err)
	}
	got, err := s.Rollouts(ctx(t), 1)
	if err != nil || len(got) != 1 {
		t.Fatalf("Rollouts = %v, %v, want one event", got, err)
	}
	e := got[0]
	if !e.At.Equal(in.At) || e.Action != in.Action || e.From != in.From || e.To != in.To ||
		e.Percent != in.Percent || e.Reason != in.Reason || e.Actor != in.Actor {
		t.Errorf("round trip = %+v, want %+v", e, in)
	}
	if e.At.Location() != time.UTC {
		t.Errorf("At location = %v, want UTC", e.At.Location())
	}
}

func rejectsInvalidRollouts(t *testing.T, s accounting.Store) {
	for name, mutate := range map[string]func(*accounting.RolloutEvent){
		"unknown action":   func(e *accounting.RolloutEvent) { e.Action = "explode" },
		"empty action":     func(e *accounting.RolloutEvent) { e.Action = "" },
		"NaN percent":      func(e *accounting.RolloutEvent) { e.Percent = math.NaN() },
		"infinite percent": func(e *accounting.RolloutEvent) { e.Percent = math.Inf(1) },
	} {
		e := event(base, accounting.ActionReload, "x")
		mutate(&e)
		if err := s.RecordRollout(ctx(t), e); err == nil {
			t.Errorf("RecordRollout with %s succeeded, want an error", name)
		}
	}
	for _, limit := range []int{0, -1} {
		if _, err := s.Rollouts(ctx(t), limit); err == nil {
			t.Errorf("Rollouts(limit %d) succeeded, want an error", limit)
		}
	}
	if got, _ := s.Rollouts(ctx(t), 10); len(got) != 0 {
		t.Errorf("Rollouts after only invalid events = %v, want none", got)
	}
}

// concurrentWriters: 32 goroutines write their own rows and one shared (request, attempt)
// while others read; the shared row counts once and nothing is lost.
func concurrentWriters(t *testing.T, s accounting.Store) {
	const (
		writers     = 32
		perWriter   = 10
		sharedCost  = money.PicoUSD(7)
		attemptCost = money.PicoUSD(3)
	)
	var wg sync.WaitGroup
	errs := make(chan error, writers*(perWriter*2+2))
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := ctx(t)
			shared := row("shared", 1, "tenant-a", sharedCost, base)
			if err := s.RecordAttempt(c, shared); err != nil {
				errs <- fmt.Errorf("writer %d shared: %w", w, err)
			}
			for i := range perWriter {
				req := fmt.Sprintf("w%d-r%d", w, i)
				for attempt := 1; attempt <= 2; attempt++ {
					if err := s.RecordAttempt(c, row(req, attempt, "tenant-a", attemptCost, base)); err != nil {
						errs <- fmt.Errorf("writer %d %s/%d: %w", w, req, attempt, err)
					}
				}
				if _, err := s.TenantSpend(c, "tenant-a", time.Time{}, farFuture); err != nil {
					errs <- fmt.Errorf("writer %d read: %w", w, err)
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	want := sharedCost + writers*perWriter*2*attemptCost
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != want {
		t.Errorf("spend after %d concurrent writers = %d, want %d", writers, got, want)
	}
}

func honoursContext(t *testing.T, s accounting.Store) {
	done, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.RecordAttempt(done, row("req-ctx", 1, "tenant-a", 100, base)); err == nil {
		t.Error("RecordAttempt with a canceled context succeeded, want an error")
	}
	if got := spend(t, s, "tenant-a", time.Time{}, farFuture); got != 0 {
		t.Errorf("spend = %d, want 0: a canceled write must not land", got)
	}
}

func closeEndsUse(t *testing.T, s accounting.Store) {
	if err := s.Close(); err != nil {
		t.Fatalf("Close error = %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close error = %v, want nil", err)
	}
	if err := s.RecordAttempt(ctx(t), row("req-closed", 1, "tenant-a", 1, base)); err == nil {
		t.Error("RecordAttempt after Close succeeded, want an error")
	}
	if _, err := s.TenantSpend(ctx(t), "tenant-a", time.Time{}, farFuture); err == nil {
		t.Error("TenantSpend after Close succeeded, want an error")
	}
	if _, err := s.TenantSpendInPeriod(ctx(t), "tenant-a", "total"); err == nil {
		t.Error("TenantSpendInPeriod after Close succeeded, want an error")
	}
}
