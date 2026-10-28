package quotatest

import (
	"context"
	"math/rand/v2"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
)

// RunBudget runs the BudgetLedger conformance cases. Like the other suites it uses explicit
// clock values and derives its tenants and request IDs from t.Name().
func RunBudget(t *testing.T, newL func(t *testing.T) quota.BudgetLedger) {
	const ttl = 10 * time.Second
	ctx := context.Background()
	monthly := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodMonthly}
	total := quota.BudgetLimit{Limit: 1000, Period: quota.PeriodTotal}

	reserve := func(t *testing.T, l quota.BudgetLedger, tenant, id string, lim quota.BudgetLimit, amount money.PicoUSD, now time.Time) (bool, money.PicoUSD) {
		t.Helper()
		ok, rem, err := l.Reserve(ctx, tenant, id, lim, amount, ttl, now)
		if err != nil {
			t.Fatalf("Reserve(%q, %d): unexpected error: %v", id, amount, err)
		}
		return ok, rem
	}
	wantReserve := func(t *testing.T, l quota.BudgetLedger, tenant, id string, lim quota.BudgetLimit, amount money.PicoUSD, now time.Time, wantOK bool, wantRem money.PicoUSD) {
		t.Helper()
		if ok, rem := reserve(t, l, tenant, id, lim, amount, now); ok != wantOK || rem != wantRem {
			t.Fatalf("Reserve(%q, %d) = (%v, remaining %d), want (%v, remaining %d)", id, amount, ok, rem, wantOK, wantRem)
		}
	}
	commit := func(t *testing.T, l quota.BudgetLedger, tenant, id string, actual money.PicoUSD, now time.Time) money.PicoUSD {
		t.Helper()
		over, err := l.Commit(ctx, tenant, id, actual, now)
		if err != nil {
			t.Fatalf("Commit(%q, %d): unexpected error: %v", id, actual, err)
		}
		return over
	}
	remaining := func(t *testing.T, l quota.BudgetLedger, tenant string, lim quota.BudgetLimit, now time.Time) money.PicoUSD {
		t.Helper()
		r, err := l.Remaining(ctx, tenant, lim, now)
		if err != nil {
			t.Fatalf("Remaining: unexpected error: %v", err)
		}
		return r
	}
	spent := func(t *testing.T, l quota.BudgetLedger, tenant string, lim quota.BudgetLimit, now time.Time) money.PicoUSD {
		t.Helper()
		s, err := l.Spent(ctx, tenant, lim, now)
		if err != nil {
			t.Fatalf("Spent: unexpected error: %v", err)
		}
		return s
	}
	sweep := func(t *testing.T, l quota.BudgetLedger, now time.Time) int {
		t.Helper()
		n, err := l.Sweep(ctx, now)
		if err != nil {
			t.Fatalf("Sweep: unexpected error: %v", err)
		}
		return n
	}

	t.Run("ReserveWithinLimit", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 600, t0, true, 400)
		wantReserve(t, l, tn, "r2", monthly, 400, t0, true, 0)
		if got := remaining(t, l, tn, monthly, t0); got != 0 {
			t.Errorf("Remaining = %d, want 0", got)
		}
		if got := spent(t, l, tn, monthly, t0); got != 0 {
			t.Errorf("Spent = %d, want 0: a reservation is not spend", got)
		}
	})

	t.Run("ReserveOverLimitFails", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 1001, t0, false, 1000)
		if got := remaining(t, l, tn, monthly, t0); got != 1000 {
			t.Fatalf("Remaining after a refused reserve = %d, want 1000: a refusal holds nothing", got)
		}
		wantReserve(t, l, tn, "r1", monthly, 1000, t0, true, 0) // exactly the limit fits
		wantReserve(t, l, tn, "r2", monthly, 1, t0, false, 0)
	})

	t.Run("RaiseFitsOrFails", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 300, t0, true, 700)
		wantReserve(t, l, tn, "r2", monthly, 500, t0, true, 200)
		wantReserve(t, l, tn, "r1", monthly, 500, t0, true, 0) // raised: 500 + 500 = 1000
		wantReserve(t, l, tn, "r1", monthly, 501, t0, false, 0)
		if got := remaining(t, l, tn, monthly, t0); got != 0 {
			t.Fatalf("Remaining after a refused raise = %d, want 0: r1 keeps 500", got)
		}
		// A reservation is never lowered: asking for less succeeds and changes nothing.
		wantReserve(t, l, tn, "r1", monthly, 100, t0, true, 0)
		if over := commit(t, l, tn, "r1", 500, t0); over != 0 {
			t.Errorf("Commit(500) of a reservation raised to 500: overshoot = %d, want 0", over)
		}
	})

	t.Run("RaiseExtendsTheLease", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 300, t0, true, 700) // expires at t0+10s
		raised := t0.Add(5 * time.Second)
		wantReserve(t, l, tn, "r1", monthly, 400, raised, true, 600) // expires at t0+15s
		if got := remaining(t, l, tn, monthly, t0.Add(12*time.Second)); got != 600 {
			t.Errorf("Remaining at t0+12s = %d, want 600: the raise renewed the lease", got)
		}
		if got := remaining(t, l, tn, monthly, t0.Add(15*time.Second)); got != 1000 {
			t.Errorf("Remaining at t0+15s = %d, want 1000: the renewed lease has expired", got)
		}
	})

	t.Run("CommitIdempotent", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 500, t0, true, 500)
		if over := commit(t, l, tn, "r1", 300, t0); over != 0 {
			t.Fatalf("Commit(300) of a 500 reservation: overshoot = %d, want 0", over)
		}
		if got := spent(t, l, tn, monthly, t0); got != 300 {
			t.Fatalf("Spent = %d, want 300", got)
		}
		if got := remaining(t, l, tn, monthly, t0); got != 700 {
			t.Fatalf("Remaining = %d, want 700: the reservation is gone, 300 is spent", got)
		}
		if over := commit(t, l, tn, "r1", 999, t0); over != 0 {
			t.Errorf("second Commit(999): overshoot = %d, want the first result 0", over)
		}
		if got := spent(t, l, tn, monthly, t0); got != 300 {
			t.Errorf("Spent after a second commit = %d, want 300 (unchanged)", got)
		}

		// A repeated commit returns the first overshoot, not a new one.
		wantReserve(t, l, tn, "r2", monthly, 100, t0, true, 600)
		if over := commit(t, l, tn, "r2", 250, t0); over != 150 {
			t.Fatalf("Commit(250) of a 100 reservation: overshoot = %d, want 150", over)
		}
		if over := commit(t, l, tn, "r2", 1, t0); over != 150 {
			t.Errorf("repeated commit: overshoot = %d, want the first result 150", over)
		}
		if got := spent(t, l, tn, monthly, t0); got != 550 {
			t.Errorf("Spent = %d, want 550", got)
		}
	})

	t.Run("CommitOvershootReported", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 100, t0, true, 900)
		if over := commit(t, l, tn, "r1", 250, t0); over != 150 {
			t.Fatalf("overshoot = %d, want 150", over)
		}
		if got := spent(t, l, tn, monthly, t0); got != 250 {
			t.Fatalf("Spent = %d, want the full actual 250", got)
		}
		// A commit without a reservation adds the actual and reports all of it.
		if over := commit(t, l, tn, "ghost", 70, t0); over != 70 {
			t.Fatalf("commit without a reservation: overshoot = %d, want 70", over)
		}
		if got := spent(t, l, tn, monthly, t0); got != 320 {
			t.Fatalf("Spent = %d, want 320", got)
		}
		if got := remaining(t, l, tn, monthly, t0); got != 680 {
			t.Errorf("Remaining = %d, want 680", got)
		}
		// Spend past the limit leaves nothing, never a negative amount.
		commit(t, l, tn, "big", 5000, t0)
		if got := remaining(t, l, tn, monthly, t0); got != 0 {
			t.Errorf("Remaining after overspend = %d, want 0", got)
		}
		wantReserve(t, l, tn, "after", monthly, 1, t0, false, 0)
	})

	t.Run("ReleaseFrees", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 800, t0, true, 200)
		wantReserve(t, l, tn, "r2", monthly, 300, t0, false, 200)
		if err := l.Release(ctx, tn, "r1"); err != nil {
			t.Fatalf("Release: %v", err)
		}
		wantReserve(t, l, tn, "r2", monthly, 300, t0, true, 700)
		if got := spent(t, l, tn, monthly, t0); got != 0 {
			t.Errorf("Spent after a release = %d, want 0", got)
		}

		// An absent reservation, a repeat, and a committed one are all no-ops.
		for _, id := range []string{"r1", "never-reserved"} {
			if err := l.Release(ctx, tn, id); err != nil {
				t.Errorf("Release(%q) of an absent reservation: %v", id, err)
			}
		}
		commit(t, l, tn, "r2", 100, t0)
		if err := l.Release(ctx, tn, "r2"); err != nil {
			t.Errorf("Release after Commit: %v", err)
		}
		if got := spent(t, l, tn, monthly, t0); got != 100 {
			t.Errorf("Spent = %d, want 100: a release must not undo a commit", got)
		}
		if over := commit(t, l, tn, "r2", 5, t0); over != 0 {
			t.Errorf("Commit after Release after Commit: overshoot = %d, want the first result 0", over)
		}
	})

	t.Run("ExpiredLeaseSweptAndIgnored", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "a", monthly, 400, t0, true, 600) // expires at t0+10s
		if ok, _, err := l.Reserve(ctx, tn, "c", monthly, 300, time.Minute, t0); err != nil || !ok {
			t.Fatalf("Reserve c: ok=%v err=%v", ok, err)
		}
		if n := sweep(t, l, t0.Add(9*time.Second)); n != 0 {
			t.Fatalf("Sweep before any expiry = %d, want 0", n)
		}
		wantReserve(t, l, tn, "d", monthly, 400, t0.Add(9*time.Second), false, 300)

		at := t0.Add(ttl) // a's expiry is at or before now: dropped
		if got := remaining(t, l, tn, monthly, at); got != 700 {
			t.Fatalf("Remaining at expiry = %d, want 700: the expired lease is ignored", got)
		}
		wantReserve(t, l, tn, "d", monthly, 700, at, true, 0)
		if n := sweep(t, l, at); n != 1 {
			t.Fatalf("Sweep at expiry = %d, want 1 (a)", n)
		}
		if n := sweep(t, l, at); n != 0 {
			t.Fatalf("second Sweep = %d, want 0", n)
		}
		wantReserve(t, l, tn, "e", monthly, 1, at, false, 0) // c and d still hold
	})

	t.Run("ExpiredReservationIsNotExisting", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", monthly, 800, t0, true, 200)
		later := t0.Add(ttl + time.Second)
		// The old 800 expired, so this is a fresh reservation of 100, not a raise.
		wantReserve(t, l, tn, "r1", monthly, 100, later, true, 900)
	})

	t.Run("LateCommitAfterExpiryIsCharged", func(t *testing.T) {
		for _, swept := range []bool{false, true} {
			name := "unswept"
			if swept {
				name = "swept"
			}
			l, tn := newL(t), key(t, name)
			wantReserve(t, l, tn, "r1", total, 400, t0, true, 600)
			late := t0.Add(ttl + time.Second)
			if swept {
				if n := sweep(t, l, late); n != 1 {
					t.Fatalf("%s: Sweep = %d, want 1", name, n)
				}
			}
			// The lease no longer protected the budget, so the whole actual is overshoot,
			// and it is charged to the period the reservation was made in.
			if over := commit(t, l, tn, "r1", 100, late); over != 100 {
				t.Errorf("%s: late commit overshoot = %d, want 100", name, over)
			}
			if got := spent(t, l, tn, total, late); got != 100 {
				t.Errorf("%s: Spent = %d, want 100", name, got)
			}
		}
	})

	t.Run("MonthlyPeriodRollsOver", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		oct := time.Date(2026, 10, 31, 23, 59, 59, 0, time.UTC)
		nov := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)

		wantReserve(t, l, tn, "r1", monthly, 1000, oct, true, 0)
		commit(t, l, tn, "r1", 1000, oct)
		wantReserve(t, l, tn, "r2", monthly, 1, oct, false, 0)
		if got := spent(t, l, tn, monthly, oct); got != 1000 {
			t.Fatalf("October Spent = %d, want 1000", got)
		}

		if got := remaining(t, l, tn, monthly, nov); got != 1000 {
			t.Fatalf("November Remaining = %d, want 1000", got)
		}
		if got := spent(t, l, tn, monthly, nov); got != 0 {
			t.Fatalf("November Spent = %d, want 0", got)
		}
		wantReserve(t, l, tn, "r2", monthly, 1000, nov, true, 0)
		commit(t, l, tn, "r2", 1000, nov)
		if got := spent(t, l, tn, monthly, nov); got != 1000 {
			t.Errorf("November Spent = %d, want 1000", got)
		}
		if got := spent(t, l, tn, monthly, oct); got != 1000 {
			t.Errorf("October Spent changed to %d, want 1000", got)
		}
	})

	t.Run("ReservationKeepsItsPeriod", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		oct := time.Date(2026, 10, 31, 23, 0, 0, 0, time.UTC)
		nov := time.Date(2026, 11, 2, 12, 0, 0, 0, time.UTC)
		long := func(id string, amount money.PicoUSD, now time.Time) (bool, money.PicoUSD) {
			t.Helper()
			ok, rem, err := l.Reserve(ctx, tn, id, monthly, amount, 72*time.Hour, now)
			if err != nil {
				t.Fatalf("Reserve(%q): %v", id, err)
			}
			return ok, rem
		}

		// Committed in November, charged to October.
		if ok, _ := long("r1", 400, oct); !ok {
			t.Fatal("reserve r1 in October refused")
		}
		if over := commit(t, l, tn, "r1", 300, nov); over != 0 {
			t.Errorf("overshoot = %d, want 0", over)
		}
		if got := spent(t, l, tn, monthly, oct); got != 300 {
			t.Errorf("October Spent = %d, want 300", got)
		}
		if got := spent(t, l, tn, monthly, nov); got != 0 {
			t.Errorf("November Spent = %d, want 0", got)
		}

		// A raise after the rollover is checked against the reservation's own period:
		// 300 spent + 600 fits October, but 300 + 701 does not, though November is empty.
		if ok, _ := long("r2", 400, oct); !ok {
			t.Fatal("reserve r2 in October refused")
		}
		if ok, rem := long("r2", 600, nov); !ok || rem != 100 {
			t.Fatalf("raise to 600 after the rollover = (%v, %d), want (true, 100)", ok, rem)
		}
		if ok, _ := long("r2", 701, nov); ok {
			t.Fatal("raise to 701 fit, but October has only 700 left for r2")
		}
		if got := remaining(t, l, tn, monthly, nov); got != 1000 {
			t.Errorf("November Remaining = %d, want 1000: r2 belongs to October", got)
		}
		commit(t, l, tn, "r2", 600, nov)
		if got := spent(t, l, tn, monthly, oct); got != 900 {
			t.Errorf("October Spent = %d, want 900", got)
		}
	})

	t.Run("TotalPeriodNeverRollsOver", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		wantReserve(t, l, tn, "r1", total, 700, t0, true, 300)
		commit(t, l, tn, "r1", 700, t0)
		for _, years := range []int{0, 1, 5} {
			at := t0.AddDate(years, 3, 0)
			if got := spent(t, l, tn, total, at); got != 700 {
				t.Errorf("+%dy: Spent = %d, want 700", years, got)
			}
			if got := remaining(t, l, tn, total, at); got != 300 {
				t.Errorf("+%dy: Remaining = %d, want 300", years, got)
			}
		}
	})

	t.Run("TenantsIndependent", func(t *testing.T) {
		l, a, b := newL(t), key(t, "a"), key(t, "b")
		wantReserve(t, l, a, "ra", monthly, 1000, t0, true, 0)
		wantReserve(t, l, b, "rb", monthly, 1000, t0, true, 0)
		commit(t, l, a, "ra", 1000, t0)
		if got := remaining(t, l, b, monthly, t0); got != 0 {
			t.Errorf("tenant b Remaining = %d, want 0 (its own reservation)", got)
		}
		if got := spent(t, l, b, monthly, t0); got != 0 {
			t.Errorf("tenant b Spent = %d, want 0", got)
		}
		unknown := key(t, "unknown")
		if got := remaining(t, l, unknown, monthly, t0); got != 1000 {
			t.Errorf("unknown tenant Remaining = %d, want the whole limit", got)
		}
	})

	t.Run("ZeroAmountAndZeroLimit", func(t *testing.T) {
		l, tn := newL(t), key(t, "tenant")
		none := quota.BudgetLimit{Limit: 0, Period: quota.PeriodMonthly}
		wantReserve(t, l, tn, "free", none, 0, t0, true, 0) // a free model under a zero budget
		wantReserve(t, l, tn, "paid", none, 1, t0, false, 0)
		if over := commit(t, l, tn, "free", 0, t0); over != 0 {
			t.Errorf("Commit(0): overshoot = %d, want 0", over)
		}
	})

	t.Run("SweepCountsEveryTenant", func(t *testing.T) {
		l := newL(t)
		for _, name := range []string{"x", "y", "z"} {
			tn := key(t, name)
			wantReserve(t, l, tn, name+"-1", monthly, 100, t0, true, 900)
			wantReserve(t, l, tn, name+"-2", monthly, 100, t0, true, 800)
		}
		long := key(t, "x")
		if ok, _, err := l.Reserve(ctx, long, "x-long", monthly, 100, time.Hour, t0); err != nil || !ok {
			t.Fatalf("Reserve x-long: ok=%v err=%v", ok, err)
		}
		if n := sweep(t, l, t0.Add(ttl)); n != 6 {
			t.Errorf("Sweep = %d, want 6 (two per tenant; x-long lives on)", n)
		}
		if got := remaining(t, l, long, monthly, t0.Add(ttl)); got != 900 {
			t.Errorf("Remaining after the sweep = %d, want 900", got)
		}
	})

	// Phase 1: 64 goroutines reserve at once; the granted amounts never exceed the limit.
	// Phase 2: they commit at once, each with an actual no larger than its reservation;
	// the committed spend stays within the limit and nothing is reported as overshoot.
	t.Run("ConcurrentNeverOvershoots", func(t *testing.T) {
		const limit, calls = 10_000, 1000
		lim := quota.BudgetLimit{Limit: limit, Period: quota.PeriodTotal}
		l, tn := newL(t), key(t, "tenant")

		rng := rand.New(rand.NewPCG(1, 2))
		amounts := make([]money.PicoUSD, calls)
		for i := range amounts {
			amounts[i] = money.PicoUSD(1 + rng.IntN(500))
		}
		granted := make([]atomic.Bool, calls)
		id := func(i int) string { return key(t, "req-"+strconv.Itoa(i)) }

		var grantedSum atomic.Int64
		parallel(64, calls, func(i int) {
			ok, _, err := l.Reserve(ctx, tn, id(i), lim, amounts[i], time.Hour, t0)
			if err != nil {
				t.Errorf("Reserve: unexpected error: %v", err)
				return
			}
			if ok {
				granted[i].Store(true)
				grantedSum.Add(int64(amounts[i]))
			}
		})
		sum := grantedSum.Load()
		if sum > limit {
			t.Fatalf("granted %d across concurrent reservers, above the limit %d", sum, limit)
		}
		if sum == 0 || sum < limit-500 {
			t.Fatalf("granted only %d of %d: the test did not keep the ledger busy", sum, limit)
		}
		if got := remaining(t, l, tn, lim, t0); int64(got) != limit-sum {
			t.Fatalf("Remaining = %d, want %d", got, limit-sum)
		}

		var committedSum, overshoot atomic.Int64
		parallel(64, calls, func(i int) {
			if !granted[i].Load() {
				return
			}
			actual := amounts[i] * money.PicoUSD(i%5+1) / 5 // never above the reservation
			over, err := l.Commit(ctx, tn, id(i), actual, t0)
			if err != nil {
				t.Errorf("Commit: unexpected error: %v", err)
				return
			}
			overshoot.Add(int64(over))
			committedSum.Add(int64(actual))
		})
		if got := overshoot.Load(); got != 0 {
			t.Errorf("total overshoot = %d, want 0 when no actual exceeds its reservation", got)
		}
		got := spent(t, l, tn, lim, t0)
		if int64(got) != committedSum.Load() || got > limit {
			t.Errorf("Spent = %d, want the committed %d and no more than %d", got, committedSum.Load(), limit)
		}
		if rem := remaining(t, l, tn, lim, t0); int64(rem) != limit-int64(got) {
			t.Errorf("Remaining = %d, want %d", rem, limit-int64(got))
		}
	})

	// 64 goroutines each hold a small reservation and raise it; the sum of what is granted
	// never passes the limit, whatever the interleaving.
	t.Run("ConcurrentRaisesNeverOvershoot", func(t *testing.T) {
		const limit, calls = 5_000, 64
		lim := quota.BudgetLimit{Limit: limit, Period: quota.PeriodTotal}
		l, tn := newL(t), key(t, "tenant")
		final := make([]atomic.Int64, calls)

		parallel(64, calls, func(i int) {
			rid := key(t, "req-"+strconv.Itoa(i))
			for _, amount := range []money.PicoUSD{20, 60, 120, 200} {
				ok, _, err := l.Reserve(ctx, tn, rid, lim, amount, time.Hour, t0)
				if err != nil {
					t.Errorf("Reserve: unexpected error: %v", err)
					return
				}
				if !ok {
					return // a refused raise leaves the earlier reservation
				}
				final[i].Store(int64(amount))
			}
		})
		var sum int64
		for i := range final {
			sum += final[i].Load()
		}
		if sum > limit {
			t.Fatalf("reservations total %d, above the limit %d", sum, limit)
		}
		if got := remaining(t, l, tn, lim, t0); int64(got) != limit-sum {
			t.Fatalf("Remaining = %d, want %d", got, limit-sum)
		}
	})
}
