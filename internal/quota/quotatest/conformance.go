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

// parallel runs fn(0..calls-1) spread over workers goroutines.
func parallel(workers, calls int, fn func(i int)) {
	var wg sync.WaitGroup
	var next atomic.Int64
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= calls {
					return
				}
				fn(i)
			}
		}()
	}
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
		if atLater.Allowed {
			t.Fatalf("exhausted at later: want rejected, got %+v", atLater)
		}
		earlier := allow(t, l, k, t0)
		if earlier.Allowed {
			t.Fatalf("earlier clock must not be more permissive, got %+v", earlier)
		}
		if earlier.RetryAfter < atLater.RetryAfter {
			t.Fatalf("earlier RetryAfter %v is shorter than at later %v", earlier.RetryAfter, atLater.RetryAfter)
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
