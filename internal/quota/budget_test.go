package quota_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"math"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/quota/quotatest"
)

func TestPeriodKey(t *testing.T) {
	est := time.FixedZone("EST", -5*3600)
	tests := []struct {
		name   string
		period string
		now    time.Time
		want   string
	}{
		{"monthly is the UTC year and month", quota.PeriodMonthly, time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC), "2026-10"},
		{"monthly pads the month", quota.PeriodMonthly, time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC), "2026-03"},
		{"the last second of a month", quota.PeriodMonthly, time.Date(2026, 10, 31, 23, 59, 59, 999_999_999, time.UTC), "2026-10"},
		{"the first instant of the next month", quota.PeriodMonthly, time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC), "2026-11"},
		{"a local time is converted to UTC first", quota.PeriodMonthly, time.Date(2026, 10, 31, 20, 0, 0, 0, est), "2026-11"},
		{"a December rolls into the next year", quota.PeriodMonthly, time.Date(2026, 12, 31, 23, 0, 0, 0, time.UTC), "2026-12"},
		{"total does not depend on the time", quota.PeriodTotal, time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC), "total"},
		{"total in another year", quota.PeriodTotal, time.Date(2031, 1, 1, 0, 0, 0, 0, time.UTC), "total"},
		{"an empty period is monthly", "", time.Date(2026, 10, 5, 0, 0, 0, 0, time.UTC), "2026-10"},
	}
	for _, tc := range tests {
		if got := quota.PeriodKey(tc.period, tc.now); got != tc.want {
			t.Errorf("%s: PeriodKey(%q, %v) = %q, want %q", tc.name, tc.period, tc.now, got, tc.want)
		}
	}
}

func TestBudgetConformanceMemory(t *testing.T) {
	quotatest.RunBudget(t, func(*testing.T) quota.BudgetLedger { return quota.NewMemoryBudget() })
}

func TestBudgetConformanceMiniredis(t *testing.T) {
	quotatest.RunBudget(t, func(t *testing.T) quota.BudgetLedger {
		_, c := newMiniredisClient(t)
		return quota.NewRedisBudget(c, "rp-test")
	})
}

func TestBudgetConformanceReal(t *testing.T) {
	url := os.Getenv("ROUTEPILOT_TEST_REDIS_URL")
	if url == "" {
		t.Skip("ROUTEPILOT_TEST_REDIS_URL is not set; skipping the budget conformance against a real Redis")
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		t.Fatalf("parse ROUTEPILOT_TEST_REDIS_URL: %v", err)
	}
	c := redis.NewClient(opts)
	t.Cleanup(func() { _ = c.Close() })

	// Every subtest gets a ledger of its own under a prefix of its own, like the fresh
	// miniredis of the default suite: request IDs are global in the ledger and Sweep counts
	// every tenant under its prefix, so subtests sharing one prefix would see each other.
	quotatest.RunBudget(t, func(t *testing.T) quota.BudgetLedger {
		var b [8]byte
		if _, err := rand.Read(b[:]); err != nil {
			t.Fatal(err)
		}
		prefix := "routepilot-test-" + hex.EncodeToString(b[:])
		t.Cleanup(func() {
			ctx := context.Background()
			iter := c.Scan(ctx, 0, prefix+":*", 500).Iterator()
			for iter.Next(ctx) {
				c.Del(ctx, iter.Val())
			}
		})
		return quota.NewRedisBudget(c, prefix)
	})
}

func TestLimitersCarryABudgetLedger(t *testing.T) {
	if quota.NewMemory().Budget == nil {
		t.Error("NewMemory().Budget is nil")
	}
	_, c := newMiniredisClient(t)
	if quota.NewRedis(c, "rp").Budget == nil {
		t.Error("NewRedis().Budget is nil")
	}
}

// TestMemoryBudgetSaturates: amounts at the top of the int64 range never wrap.
func TestMemoryBudgetSaturates(t *testing.T) {
	ctx, now := context.Background(), time.Unix(1_700_000_000, 0)
	lim := quota.BudgetLimit{Limit: money.PicoUSD(math.MaxInt64), Period: quota.PeriodTotal}
	l := quota.NewMemoryBudget()

	if ok, rem, err := l.Reserve(ctx, "t", "r1", lim, money.PicoUSD(math.MaxInt64), time.Hour, now); err != nil || !ok || rem != 0 {
		t.Fatalf("reserve the whole range = (%v, %d, %v), want (true, 0, nil)", ok, rem, err)
	}
	if ok, _, _ := l.Reserve(ctx, "t", "r2", lim, 1, time.Hour, now); ok {
		t.Fatal("one more picodollar fit after the whole range was reserved")
	}
	for _, id := range []string{"r1", "r3"} {
		if _, err := l.Commit(ctx, "t", id, money.PicoUSD(math.MaxInt64), now); err != nil {
			t.Fatalf("Commit(%s): %v", id, err)
		}
	}
	if got, _ := l.Spent(ctx, "t", lim, now); got != money.PicoUSD(math.MaxInt64) {
		t.Errorf("Spent = %d, want it saturated at MaxInt64", got)
	}
	if got, _ := l.Remaining(ctx, "t", lim, now); got != 0 {
		t.Errorf("Remaining = %d, want 0", got)
	}
}

// TestBudgetNegativeAmountsAreZero: a negative amount can never add credit.
func TestBudgetNegativeAmountsAreZero(t *testing.T) {
	_, c := newMiniredisClient(t)
	ledgers := map[string]quota.BudgetLedger{
		"memory": quota.NewMemoryBudget(),
		"redis":  quota.NewRedisBudget(c, "rp"),
	}
	ctx, now := context.Background(), time.Unix(1_700_000_000, 0)
	lim := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodTotal}
	for name, l := range ledgers {
		t.Run(name, func(t *testing.T) {
			if ok, rem, err := l.Reserve(ctx, "t", "r1", lim, -50, time.Hour, now); err != nil || !ok || rem != 1000 {
				t.Fatalf("Reserve(-50) = (%v, %d, %v), want (true, 1000, nil)", ok, rem, err)
			}
			if over, err := l.Commit(ctx, "t", "r1", -50, now); err != nil || over != 0 {
				t.Fatalf("Commit(-50) = (%d, %v), want (0, nil)", over, err)
			}
			if got, _ := l.Spent(ctx, "t", lim, now); got != 0 {
				t.Errorf("Spent = %d, want 0", got)
			}
		})
	}
}

// TestRedisBudgetKeyLayout pins the Redis keys. The braces are hash tags: the three keys
// of one tenant and period share a slot.
func TestRedisBudgetKeyLayout(t *testing.T) {
	mr, c := newMiniredisClient(t)
	l := quota.NewRedisBudget(c, "rp")
	ctx := context.Background()
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	lim := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodMonthly}

	if ok, rem, err := l.Reserve(ctx, "tenant-a", "req-1", lim, 300, 10*time.Second, now); err != nil || !ok || rem != 700 {
		t.Fatalf("Reserve = (%v, %d, %v), want (true, 700, nil)", ok, rem, err)
	}
	if got := mr.HGet("rp:budget:{tenant-a}:2026-10:leases", "req-1"); got != "300" {
		t.Errorf("leases[req-1] = %q, want 300", got)
	}
	score, err := mr.ZScore("rp:budget:{tenant-a}:2026-10:expiry", "req-1")
	if err != nil || int64(score) != now.Add(10*time.Second).UnixMicro() {
		t.Errorf("expiry[req-1] = %v, %v, want %d", score, err, now.Add(10*time.Second).UnixMicro())
	}
	rec := "rp:budget:lease:{req-1}"
	for field, want := range map[string]string{"tenant": "tenant-a", "period": "2026-10", "amount": "300"} {
		if got := mr.HGet(rec, field); got != want {
			t.Errorf("%s[%s] = %q, want %q", rec, field, got, want)
		}
	}
	if got := mr.TTL(rec); got != 10*time.Second+48*time.Hour {
		t.Errorf("lease record ttl = %v, want the lease ttl plus 48h", got)
	}

	if over, err := l.Commit(ctx, "tenant-a", "req-1", 250, now); err != nil || over != 0 {
		t.Fatalf("Commit = (%d, %v), want (0, nil)", over, err)
	}
	if got, err := mr.Get("rp:budget:{tenant-a}:2026-10:spent"); err != nil || got != "250" {
		t.Errorf("spent = %q, %v, want 250", got, err)
	}
	if got, err := mr.Get("rp:budget:commit:{req-1}"); err != nil || got != "0" {
		t.Errorf("commit result = %q, %v, want 0", got, err)
	}
	if got := mr.TTL("rp:budget:commit:{req-1}"); got != 48*time.Hour {
		t.Errorf("commit result ttl = %v, want 48h", got)
	}
	for _, gone := range []string{rec, "rp:budget:{tenant-a}:2026-10:leases", "rp:budget:{tenant-a}:2026-10:expiry"} {
		if mr.Exists(gone) {
			t.Errorf("key %s still exists after the commit", gone)
		}
	}
}

// TestRedisBudgetClampsToExactRange: Lua numbers are doubles, exact below 2^53, so the Redis
// ledger treats a limit above 9e15 picodollars (9,000 USD) as 9e15. That is stricter, never
// looser, and config validation refuses a larger budget_usd for the Redis backend.
func TestRedisBudgetClampsToExactRange(t *testing.T) {
	_, c := newMiniredisClient(t)
	l := quota.NewRedisBudget(c, "rp")
	ctx, now := context.Background(), time.Unix(1_700_000_000, 0)
	const exact = 9_000_000_000_000_000
	lim := quota.BudgetLimit{Limit: money.PicoUSD(math.MaxInt64), Period: quota.PeriodTotal}

	if got, err := l.Remaining(ctx, "t", lim, now); err != nil || got != exact {
		t.Fatalf("Remaining of a vast limit = %d, %v, want it clamped to %d", got, err, int64(exact))
	}
	if ok, rem, err := l.Reserve(ctx, "t", "huge", lim, money.PicoUSD(math.MaxInt64), time.Hour, now); err != nil || ok || rem != exact {
		t.Fatalf("Reserve(MaxInt64) = (%v, %d, %v), want a refusal with %d left", ok, rem, err, int64(exact))
	}
	if ok, rem, err := l.Reserve(ctx, "t", "r1", lim, exact, time.Hour, now); err != nil || !ok || rem != 0 {
		t.Fatalf("Reserve(%d) = (%v, %d, %v), want (true, 0, nil)", int64(exact), ok, rem, err)
	}
	if ok, _, _ := l.Reserve(ctx, "t", "r2", lim, 1, time.Hour, now); ok {
		t.Fatal("one more picodollar fit above the exact range")
	}
	// A saturated actual (a price times a token count that overflowed) is recorded as a
	// spend far above any limit, not an error.
	if over, err := l.Commit(ctx, "t", "r1", money.PicoUSD(math.MaxInt64), now); err != nil || over <= 0 {
		t.Fatalf("Commit(MaxInt64) = (%d, %v), want a positive overshoot", over, err)
	}
	if got, err := l.Remaining(ctx, "t", lim, now); err != nil || got != 0 {
		t.Errorf("Remaining after a saturated commit = %d, %v, want 0", got, err)
	}
}

func TestRedisBudgetErrorsPropagate(t *testing.T) {
	l := quota.NewRedisBudget(newClosedClient(t), "rp-test")
	ctx, now := context.Background(), time.Unix(1_700_000_000, 0)
	lim := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodTotal}

	if _, _, err := l.Reserve(ctx, "t", "r", lim, 1, time.Second, now); err == nil {
		t.Error("Reserve: want an error from a closed server")
	}
	if _, err := l.Commit(ctx, "t", "r", 1, now); err == nil {
		t.Error("Commit: want an error from a closed server")
	}
	if err := l.Release(ctx, "t", "r"); err == nil {
		t.Error("Release: want an error from a closed server")
	}
	if _, err := l.Remaining(ctx, "t", lim, now); err == nil {
		t.Error("Remaining: want an error from a closed server")
	}
	if _, err := l.Spent(ctx, "t", lim, now); err == nil {
		t.Error("Spent: want an error from a closed server")
	}
	if _, err := l.Sweep(ctx, now); err == nil {
		t.Error("Sweep: want an error from a closed server")
	}
}

// TestRedisBudgetLedgersShareState: two ledgers on one Redis (two gateway instances) see one
// budget.
func TestRedisBudgetLedgersShareState(t *testing.T) {
	_, c := newMiniredisClient(t)
	a, b := quota.NewRedisBudget(c, "rp"), quota.NewRedisBudget(c, "rp")
	ctx, now := context.Background(), time.Unix(1_700_000_000, 0)
	lim := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodTotal}

	if ok, _, err := a.Reserve(ctx, "t", "r1", lim, 700, time.Hour, now); err != nil || !ok {
		t.Fatalf("a.Reserve: %v %v", ok, err)
	}
	if ok, _, err := b.Reserve(ctx, "t", "r2", lim, 400, time.Hour, now); err != nil || ok {
		t.Fatalf("b.Reserve(400) = (%v, %v), want a refusal: instance a holds 700", ok, err)
	}
	if over, err := b.Commit(ctx, "t", "r1", 600, now); err != nil || over != 0 {
		t.Fatalf("b.Commit of a's reservation = (%d, %v), want (0, nil)", over, err)
	}
	if got, _ := a.Spent(ctx, "t", lim, now); got != 600 {
		t.Errorf("a.Spent = %d, want 600", got)
	}
}
