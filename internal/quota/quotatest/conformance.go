// Package quotatest holds conformance suites shared by every quota backend
// (memory, Redis). Each Run* function takes a constructor and exercises the
// limiter contract with explicit clock values: no time.Now, no sleeps.
//
// Every subtest builds a fresh limiter and derives its keys from t.Name(), so a
// backend that shares storage across constructor calls does not leak state
// between subtests.
package quotatest

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/quota"
)

// t0 is the fixed base instant used by every case.
var t0 = time.Unix(1_700_000_000, 0)

func key(t *testing.T, name string) string { return t.Name() + "/" + name }

// parallel runs fn(0..calls-1) spread over workers goroutines. Workers wait on
// a start barrier that is closed once all of them are spawned, so the calls
// overlap as much as possible instead of draining in the first worker.
func parallel(workers, calls int, fn func(i int)) {
	var wg sync.WaitGroup
	var next atomic.Int64
	start := make(chan struct{})
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for {
				i := int(next.Add(1)) - 1
				if i >= calls {
					return
				}
				fn(i)
			}
		}()
	}
	close(start)
	wg.Wait()
}

// RunRequestLimiter runs the RequestLimiter conformance cases.
func RunRequestLimiter(t *testing.T, newL func(t *testing.T) quota.RequestLimiter) {
	lim := quota.RateLimit{PerMinute: 60, Burst: 10} // T = 1s, tau = 9s
	ctx := context.Background()

	allow := func(t *testing.T, l quota.RequestLimiter, k string, now time.Time) quota.Result {
		t.Helper()
		r, err := l.Allow(ctx, k, lim, now)
		if err != nil {
			t.Fatalf("Allow(%q): unexpected error: %v", k, err)
		}
		return r
	}

	t.Run("BurstThenReject", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		for i := 0; i < 10; i++ {
			if r := allow(t, l, k, t0); !r.Allowed {
				t.Fatalf("call %d: want allowed within burst, got %+v", i+1, r)
			}
		}
		r := allow(t, l, k, t0)
		if r.Allowed || r.RetryAfter != time.Second {
			t.Fatalf("11th call: want {Allowed:false RetryAfter:1s}, got %+v", r)
		}
		// A rejection must not consume capacity.
		if r2 := allow(t, l, k, t0); r2.Allowed || r2.RetryAfter != time.Second {
			t.Fatalf("12th call: want {Allowed:false RetryAfter:1s}, got %+v", r2)
		}
	})

	t.Run("RefillsAtRate", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		for i := 0; i < 10; i++ {
			allow(t, l, k, t0)
		}
		just := t0.Add(time.Second - time.Microsecond)
		if r := allow(t, l, k, just); r.Allowed || r.RetryAfter != time.Microsecond {
			t.Fatalf("1us early: want {Allowed:false RetryAfter:1us}, got %+v", r)
		}
		at := t0.Add(time.Second)
		if r := allow(t, l, k, at); !r.Allowed {
			t.Fatalf("after one interval: want allowed, got %+v", r)
		}
		if r := allow(t, l, k, at); r.Allowed || r.RetryAfter != time.Second {
			t.Fatalf("second call at the same instant: want {Allowed:false RetryAfter:1s}, got %+v", r)
		}
	})

	t.Run("KeysIndependent", func(t *testing.T) {
		l, ka, kb := newL(t), key(t, "a"), key(t, "b")
		for i := 0; i < 10; i++ {
			allow(t, l, ka, t0)
		}
		if r := allow(t, l, ka, t0); r.Allowed {
			t.Fatalf("key a should be exhausted, got %+v", r)
		}
		for i := 0; i < 10; i++ {
			if r := allow(t, l, kb, t0); !r.Allowed {
				t.Fatalf("key b call %d: want allowed, got %+v", i+1, r)
			}
		}
	})

	t.Run("ClockBackwards", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		later := t0.Add(5 * time.Second)
		for i := 0; i < 10; i++ {
			if r := allow(t, l, k, later); !r.Allowed {
				t.Fatalf("call %d at later: want allowed, got %+v", i+1, r)
			}
		}
		atLater := allow(t, l, k, later)
		if atLater.Allowed || atLater.RetryAfter != time.Second {
			t.Fatalf("exhausted at later: want {Allowed:false RetryAfter:1s}, got %+v", atLater)
		}
		// The key's clock is the latest instant seen, so an earlier now is
		// evaluated as `later`: the wait is 1s, not 6s.
		earlier := allow(t, l, k, t0)
		if earlier.Allowed || earlier.RetryAfter != time.Second {
			t.Fatalf("earlier now: want {Allowed:false RetryAfter:1s}, got %+v", earlier)
		}
	})

	t.Run("DenyRecordsLatestInstant", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		for i := 0; i < 10; i++ {
			allow(t, l, k, t0)
		}
		// A rejection at a later instant still advances the key's clock.
		mid := t0.Add(500 * time.Millisecond)
		if r := allow(t, l, k, mid); r.Allowed || r.RetryAfter != 500*time.Millisecond {
			t.Fatalf("deny at +500ms: want {Allowed:false RetryAfter:500ms}, got %+v", r)
		}
		// An earlier now is evaluated at +500ms, not at t0 (which would give 1s).
		if r := allow(t, l, k, t0); r.Allowed || r.RetryAfter != 500*time.Millisecond {
			t.Fatalf("earlier now after a later deny: want {Allowed:false RetryAfter:500ms}, got %+v", r)
		}
	})

	// A single request must fit an idle key even when the emission interval
	// (60_000_000 / 7 us) is not a whole number of microseconds.
	t.Run("SingleBurstFitsWithInexactInterval", func(t *testing.T) {
		inexact := quota.RateLimit{PerMinute: 7, Burst: 1}
		l, k := newL(t), key(t, "k")
		if r, err := l.Allow(ctx, k, inexact, t0); err != nil || !r.Allowed {
			t.Fatalf("first call at 7/min burst 1: want allowed, got %+v err=%v", r, err)
		}
		r, err := l.Allow(ctx, k, inexact, t0)
		if err != nil || r.Allowed || r.RetryAfter <= 0 {
			t.Fatalf("second call: want rejected with a wait, got %+v err=%v", r, err)
		}
		if r, err := l.Allow(ctx, k, inexact, t0.Add(r.RetryAfter)); err != nil || !r.Allowed {
			t.Fatalf("call after the advised wait: want allowed, got %+v err=%v", r, err)
		}
	})

	t.Run("Concurrent", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		var allowed atomic.Int64
		parallel(64, 1000, func(int) {
			r, err := l.Allow(ctx, k, lim, t0)
			if err != nil {
				t.Errorf("Allow: unexpected error: %v", err)
				return
			}
			if r.Allowed {
				allowed.Add(1)
			}
		})
		if got := allowed.Load(); got != 10 {
			t.Fatalf("allowed %d of 1000 concurrent calls, want exactly 10", got)
		}
	})
}

// RunTokenLimiter runs the TokenLimiter conformance cases.
func RunTokenLimiter(t *testing.T, newL func(t *testing.T) quota.TokenLimiter) {
	lim := quota.RateLimit{PerMinute: 600, Burst: 100} // T = 100ms, tau = 9.9s
	ctx := context.Background()

	reserve := func(t *testing.T, l quota.TokenLimiter, k string, n int64, now time.Time) quota.Result {
		t.Helper()
		r, err := l.Reserve(ctx, k, lim, n, now)
		if err != nil {
			t.Fatalf("Reserve(%q, %d): unexpected error: %v", k, n, err)
		}
		return r
	}
	refund := func(t *testing.T, l quota.TokenLimiter, k string, n int64, now time.Time) {
		t.Helper()
		if err := l.Refund(ctx, k, lim, n, now); err != nil {
			t.Fatalf("Refund(%q, %d): unexpected error: %v", k, n, err)
		}
	}

	t.Run("ReserveUpToBurst", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		if r := reserve(t, l, k, 40, t0); !r.Allowed {
			t.Fatalf("reserve 40: want allowed, got %+v", r)
		}
		if r := reserve(t, l, k, 60, t0); !r.Allowed {
			t.Fatalf("reserve 60 (reaching burst): want allowed, got %+v", r)
		}
		r := reserve(t, l, k, 1, t0)
		if r.Allowed || r.RetryAfter != 100*time.Millisecond {
			t.Fatalf("reserve 1 past burst: want {Allowed:false RetryAfter:100ms}, got %+v", r)
		}
		if r := reserve(t, l, k, 1, t0.Add(100*time.Millisecond)); !r.Allowed {
			t.Fatalf("reserve 1 after one interval: want allowed, got %+v", r)
		}
	})

	t.Run("OverBurstNeverAllowed", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		for _, now := range []time.Time{t0, t0.Add(time.Hour)} {
			r := reserve(t, l, k, 101, now)
			if r.Allowed || r.RetryAfter != 0 {
				t.Fatalf("reserve 101 at %v: want {Allowed:false RetryAfter:0}, got %+v", now, r)
			}
		}
		// The over-burst attempts stored nothing, so a full burst still fits.
		if r := reserve(t, l, k, 100, t0); !r.Allowed {
			t.Fatalf("reserve 100 after over-burst attempts: want allowed, got %+v", r)
		}
	})

	t.Run("RefundRestoresCapacity", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		if r := reserve(t, l, k, 100, t0); !r.Allowed {
			t.Fatalf("reserve 100: want allowed, got %+v", r)
		}
		if r := reserve(t, l, k, 1, t0); r.Allowed {
			t.Fatalf("reserve 1 on a full bucket: want rejected, got %+v", r)
		}
		refund(t, l, k, 30, t0)
		if r := reserve(t, l, k, 30, t0); !r.Allowed {
			t.Fatalf("reserve 30 after refund 30: want allowed, got %+v", r)
		}
		if r := reserve(t, l, k, 1, t0); r.Allowed {
			t.Fatalf("reserve 1 after re-filling: want rejected, got %+v", r)
		}
	})

	t.Run("RefundNeverBeforeNow", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		if r := reserve(t, l, k, 10, t0); !r.Allowed {
			t.Fatalf("reserve 10: want allowed, got %+v", r)
		}
		refund(t, l, k, 50, t0) // far more than was reserved
		if r := reserve(t, l, k, 100, t0); !r.Allowed {
			t.Fatalf("reserve 100 after over-refund: want allowed, got %+v", r)
		}
		if r := reserve(t, l, k, 1, t0); r.Allowed {
			t.Fatalf("over-refund granted more than Burst, got %+v", r)
		}

		// Refunding a key that holds no state must not create credit.
		fresh := key(t, "fresh")
		refund(t, l, fresh, 50, t0)
		if r := reserve(t, l, fresh, 100, t0); !r.Allowed {
			t.Fatalf("reserve 100 on refunded fresh key: want allowed, got %+v", r)
		}
		if r := reserve(t, l, fresh, 1, t0); r.Allowed {
			t.Fatalf("refund on a fresh key granted more than Burst, got %+v", r)
		}
	})

	t.Run("NoOpsDoNotAdvanceClock", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		if r := reserve(t, l, k, 100, t0); !r.Allowed {
			t.Fatalf("reserve 100: want allowed, got %+v", r)
		}
		// None of these writes state, so none may move the key's clock to
		// `later`: that would refill the bucket.
		later := t0.Add(time.Hour)
		unlimited := quota.RateLimit{PerMinute: 0, Burst: 1}
		if r := reserve(t, l, k, 101, later); r.Allowed {
			t.Fatalf("over-burst reserve: want rejected, got %+v", r)
		}
		if r := reserve(t, l, k, 0, later); !r.Allowed {
			t.Fatalf("reserve 0: want allowed no-op, got %+v", r)
		}
		if r, err := l.Reserve(ctx, k, unlimited, 5, later); err != nil || !r.Allowed {
			t.Fatalf("unlimited reserve: want allowed, got %+v err=%v", r, err)
		}
		refund(t, l, k, 0, later)
		if err := l.Refund(ctx, k, unlimited, 5, later); err != nil {
			t.Fatalf("unlimited refund: %v", err)
		}
		if err := l.Refund(ctx, key(t, "no-state"), lim, 5, later); err != nil {
			t.Fatalf("refund without state: %v", err)
		}
		just := t0.Add(100*time.Millisecond - time.Microsecond)
		if r := reserve(t, l, k, 1, just); r.Allowed || r.RetryAfter != time.Microsecond {
			t.Fatalf("1us before refill: want {Allowed:false RetryAfter:1us}, got %+v", r)
		}
		if r := reserve(t, l, k, 1, t0.Add(100*time.Millisecond)); !r.Allowed {
			t.Fatalf("at refill: want allowed, got %+v", r)
		}
	})

	// At 40M tokens per minute one token takes 1.5 us, which a floored
	// emission interval would round down to 1 us and refill 50% too fast.
	t.Run("RefillAtHighRateIsExact", func(t *testing.T) {
		high := quota.RateLimit{PerMinute: 40_000_000, Burst: 40_000_000}
		const refilled = 20_000_000 // 30 s of a 40M/min rate
		fill := func(t *testing.T, l quota.TokenLimiter, k string) {
			t.Helper()
			r, err := l.Reserve(ctx, k, high, 40_000_000, t0)
			if err != nil || !r.Allowed {
				t.Fatalf("reserve the full burst: want allowed, got %+v err=%v", r, err)
			}
		}
		at := t0.Add(30 * time.Second)

		l, k := newL(t), key(t, "exact")
		fill(t, l, k)
		if r, err := l.Reserve(ctx, k, high, refilled, at); err != nil || !r.Allowed {
			t.Fatalf("reserve %d after 30s: want allowed, got %+v err=%v", refilled, r, err)
		}
		if r, err := l.Reserve(ctx, k, high, 2, at); err != nil || r.Allowed {
			t.Fatalf("reserve 2 more after the refill is used: want rejected, got %+v err=%v", r, err)
		}

		// One more than refilled+1 must not fit, so the grant is exact to +/- 1.
		k2 := key(t, "over")
		fill(t, l, k2)
		if r, err := l.Reserve(ctx, k2, high, refilled+2, at); err != nil || r.Allowed {
			t.Fatalf("reserve %d after 30s: want rejected, got %+v err=%v", refilled+2, r, err)
		}
	})

	// A reservation of exactly Burst must fit an idle key even when
	// Burst*60_000_000/PerMinute is not a whole number of microseconds.
	t.Run("FullBurstFitsWithInexactInterval", func(t *testing.T) {
		inexact := quota.RateLimit{PerMinute: 7, Burst: 3}
		l, k := newL(t), key(t, "k")
		if r, err := l.Reserve(ctx, k, inexact, 3, t0); err != nil || !r.Allowed {
			t.Fatalf("reserve a full burst of 3 at 7/min: want allowed, got %+v err=%v", r, err)
		}
		if r, err := l.Reserve(ctx, k, inexact, 1, t0); err != nil || r.Allowed || r.RetryAfter <= 0 {
			t.Fatalf("reserve 1 on the full bucket: want rejected with a wait, got %+v err=%v", r, err)
		}
	})

	t.Run("ConcurrentReservationsNeverExceedBurst", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		var allowed atomic.Int64
		parallel(64, 1000, func(int) {
			r, err := l.Reserve(ctx, k, lim, 1, t0)
			if err != nil {
				t.Errorf("Reserve: unexpected error: %v", err)
				return
			}
			if r.Allowed {
				allowed.Add(1)
			}
		})
		if got := allowed.Load(); got != 100 {
			t.Fatalf("reserved %d tokens concurrently, want exactly Burst (100)", got)
		}
	})
}

// RunConcurrencyLimiter runs the ConcurrencyLimiter conformance cases.
func RunConcurrencyLimiter(t *testing.T, newL func(t *testing.T) quota.ConcurrencyLimiter) {
	const ttl = 10 * time.Second
	ctx := context.Background()

	acquire := func(t *testing.T, l quota.ConcurrencyLimiter, k, lease string, limit int, now time.Time) bool {
		t.Helper()
		ok, err := l.Acquire(ctx, k, lease, limit, ttl, now)
		if err != nil {
			t.Fatalf("Acquire(%q, %q): unexpected error: %v", k, lease, err)
		}
		return ok
	}
	release := func(t *testing.T, l quota.ConcurrencyLimiter, k, lease string) {
		t.Helper()
		if err := l.Release(ctx, k, lease); err != nil {
			t.Fatalf("Release(%q, %q): unexpected error: %v", k, lease, err)
		}
	}

	t.Run("LimitEnforced", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		if !acquire(t, l, k, "a", 2, t0) || !acquire(t, l, k, "b", 2, t0) {
			t.Fatal("first two leases: want acquired")
		}
		if acquire(t, l, k, "c", 2, t0) {
			t.Fatal("third lease over limit 2: want rejected")
		}
	})

	t.Run("ReleaseFreesSlot", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		acquire(t, l, k, "a", 2, t0)
		acquire(t, l, k, "b", 2, t0)
		release(t, l, k, "a")
		if !acquire(t, l, k, "c", 2, t0) {
			t.Fatal("lease after release: want acquired")
		}
		if acquire(t, l, k, "d", 2, t0) {
			t.Fatal("lease over limit after refill: want rejected")
		}
	})

	t.Run("LeaseExpires", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		acquire(t, l, k, "a", 2, t0)
		acquire(t, l, k, "b", 2, t0)
		if acquire(t, l, k, "c", 2, t0.Add(ttl-time.Microsecond)) {
			t.Fatal("just before ttl: leases still held, want rejected")
		}
		at := t0.Add(ttl)
		if !acquire(t, l, k, "c", 2, at) || !acquire(t, l, k, "d", 2, at) {
			t.Fatal("at ttl: both expired slots should be free")
		}
		if acquire(t, l, k, "e", 2, at) {
			t.Fatal("over limit after expiry refill: want rejected")
		}
	})

	t.Run("SameLeaseRefreshes", func(t *testing.T) {
		// Re-acquiring a held lease succeeds and does not count twice.
		l, k := newL(t), key(t, "twice")
		if !acquire(t, l, k, "a", 2, t0) || !acquire(t, l, k, "a", 2, t0) {
			t.Fatal("same lease twice: want acquired both times")
		}
		if !acquire(t, l, k, "b", 2, t0) {
			t.Fatal("second distinct lease: want acquired (a counted once)")
		}
		if acquire(t, l, k, "c", 2, t0) {
			t.Fatal("third distinct lease over limit 2: want rejected")
		}

		// Re-acquiring pushes the expiry out.
		k2 := key(t, "extend")
		acquire(t, l, k2, "a", 1, t0)
		acquire(t, l, k2, "a", 1, t0.Add(5*time.Second)) // now expires at t0+15s
		if acquire(t, l, k2, "b", 1, t0.Add(12*time.Second)) {
			t.Fatal("refreshed lease must still hold at t0+12s")
		}
		if !acquire(t, l, k2, "b", 1, t0.Add(15*time.Second)) {
			t.Fatal("refreshed lease must be free at t0+15s")
		}
	})

	t.Run("ReleaseIdempotent", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		release(t, l, key(t, "never-used"), "ghost") // unknown key and lease
		acquire(t, l, k, "a", 1, t0)
		release(t, l, k, "a")
		release(t, l, k, "a")
		if !acquire(t, l, k, "b", 1, t0) {
			t.Fatal("lease after double release: want acquired")
		}
		release(t, l, k, "a") // stale release must not free b's slot
		if acquire(t, l, k, "c", 1, t0) {
			t.Fatal("stale release freed another lease's slot")
		}
	})

	t.Run("ConcurrentAcquireNeverExceedsLimit", func(t *testing.T) {
		l, k := newL(t), key(t, "k")
		var held atomic.Int64
		parallel(64, 1000, func(i int) {
			lease := "lease-" + strconv.Itoa(i)
			ok, err := l.Acquire(ctx, k, lease, 10, ttl, t0)
			if err != nil {
				t.Errorf("Acquire: unexpected error: %v", err)
				return
			}
			if ok {
				held.Add(1)
			}
		})
		if got := held.Load(); got != 10 {
			t.Fatalf("acquired %d leases concurrently, want exactly the limit (10)", got)
		}
	})
}
