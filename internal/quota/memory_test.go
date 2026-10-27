package quota_test

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/quota/quotatest"
)

var base = time.Unix(1_700_000_000, 0)

func TestMemoryConformance(t *testing.T) {
	t.Run("RequestLimiter", func(t *testing.T) {
		quotatest.RunRequestLimiter(t, func(*testing.T) quota.RequestLimiter { return quota.NewMemory().Requests })
	})
	t.Run("TokenLimiter", func(t *testing.T) {
		quotatest.RunTokenLimiter(t, func(*testing.T) quota.TokenLimiter { return quota.NewMemory().Tokens })
	})
	t.Run("ConcurrencyLimiter", func(t *testing.T) {
		quotatest.RunConcurrencyLimiter(t, func(*testing.T) quota.ConcurrencyLimiter { return quota.NewMemory().Concurrency })
	})
}

func TestNewMemoryPopulatesAllLimiters(t *testing.T) {
	l := quota.NewMemory()
	if l.Requests == nil || l.Tokens == nil || l.Concurrency == nil {
		t.Fatalf("NewMemory left a limiter nil: %+v", l)
	}
}

func TestUnlimitedRateAlwaysAllowsAndStoresNothing(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	for _, lim := range []quota.RateLimit{{PerMinute: 0, Burst: 1}, {PerMinute: -5, Burst: 1}} {
		for i := 0; i < 100; i++ {
			r, err := l.Requests.Allow(ctx, "k", lim, base)
			if err != nil || !r.Allowed {
				t.Fatalf("PerMinute %d call %d: want allowed, got %+v err=%v", lim.PerMinute, i, r, err)
			}
		}
		r, err := l.Tokens.Reserve(ctx, "k", lim, 1_000_000, base)
		if err != nil || !r.Allowed {
			t.Fatalf("PerMinute %d reserve: want allowed, got %+v err=%v", lim.PerMinute, r, err)
		}
	}
	// Nothing was stored: unlimited calls at a much later instant must not
	// move the key's clock, or this limited key would look fully refilled.
	lim := quota.RateLimit{PerMinute: 60, Burst: 1} // T = 1s
	if r, _ := l.Requests.Allow(ctx, "limited", lim, base); !r.Allowed {
		t.Fatalf("first limited call: want allowed, got %+v", r)
	}
	later := base.Add(time.Hour)
	unlimited := quota.RateLimit{PerMinute: 0, Burst: 1}
	if r, _ := l.Requests.Allow(ctx, "limited", unlimited, later); !r.Allowed {
		t.Fatalf("unlimited call at a later instant: want allowed, got %+v", r)
	}
	if r, _ := l.Tokens.Reserve(ctx, "limited", unlimited, 5, later); !r.Allowed {
		t.Fatalf("unlimited reserve at a later instant: want allowed, got %+v", r)
	}
	just := base.Add(time.Second - time.Microsecond)
	if r, _ := l.Requests.Allow(ctx, "limited", lim, just); r.Allowed || r.RetryAfter != time.Microsecond {
		t.Fatalf("1us before refill: want {false 1us}, got %+v", r)
	}
	if r, _ := l.Requests.Allow(ctx, "limited", lim, base.Add(time.Second)); !r.Allowed {
		t.Fatalf("at base+T: want allowed, got %+v", r)
	}
}

func TestNonPositiveBurstIsOne(t *testing.T) {
	ctx := context.Background()
	for _, burst := range []int64{0, -3} {
		l := quota.NewMemory()
		lim := quota.RateLimit{PerMinute: 60, Burst: burst}
		if r, _ := l.Requests.Allow(ctx, "k", lim, base); !r.Allowed {
			t.Fatalf("Burst %d first call: want allowed", burst)
		}
		r, _ := l.Requests.Allow(ctx, "k", lim, base)
		if r.Allowed || r.RetryAfter != time.Second {
			t.Fatalf("Burst %d second call: want {false 1s}, got %+v", burst, r)
		}
	}
}

func TestNonPositiveCostIsNoOp(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	lim := quota.RateLimit{PerMinute: 60, Burst: 2}
	for _, n := range []int64{0, -4} {
		r, err := l.Tokens.Reserve(ctx, "k", lim, n, base)
		if err != nil || !r.Allowed || r.RetryAfter != 0 {
			t.Fatalf("Reserve(%d): want allowed no-op, got %+v err=%v", n, r, err)
		}
		if err := l.Tokens.Refund(ctx, "k", lim, n, base); err != nil {
			t.Fatalf("Refund(%d): %v", n, err)
		}
	}
	if r, _ := l.Tokens.Reserve(ctx, "k", lim, 2, base); !r.Allowed {
		t.Fatalf("no-ops must leave the bucket full, got %+v", r)
	}
}

func TestNonPositiveRefundIsNoOp(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	lim := quota.RateLimit{PerMinute: 60, Burst: 2}
	if r, _ := l.Tokens.Reserve(ctx, "k", lim, 1, base); !r.Allowed {
		t.Fatal("first reserve: want allowed")
	}
	// A negative refund must not push the TAT forward (which would charge).
	if err := l.Tokens.Refund(ctx, "k", lim, -4, base); err != nil {
		t.Fatalf("Refund(-4): %v", err)
	}
	if r, _ := l.Tokens.Reserve(ctx, "k", lim, 1, base); !r.Allowed {
		t.Fatalf("reserve after Refund(-4): want allowed (Burst 2), got %+v", r)
	}
}

func TestCostRoundsUpToWholeMicroseconds(t *testing.T) {
	ctx := context.Background()
	// 7 per minute: one request costs ceil(60_000_000 / 7) = 8_571_429us. The
	// wait is rounded up so a retry after it is never early.
	l := quota.NewMemory()
	lim := quota.RateLimit{PerMinute: 7, Burst: 1}
	l.Requests.Allow(ctx, "k", lim, base)
	r, _ := l.Requests.Allow(ctx, "k", lim, base)
	if r.Allowed || r.RetryAfter != 8_571_429*time.Microsecond {
		t.Fatalf("want RetryAfter 8571429us, got %+v", r)
	}

	// Rates above one per microsecond cost 1us per request instead of 0.
	fast := quota.RateLimit{PerMinute: 600_000_000, Burst: 1}
	l.Requests.Allow(ctx, "fast", fast, base)
	r, _ = l.Requests.Allow(ctx, "fast", fast, base)
	if r.Allowed || r.RetryAfter != time.Microsecond {
		t.Fatalf("want RetryAfter 1us for the minimum cost, got %+v", r)
	}
}

func TestLargeBurstIsExactBelowOverflow(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	// T = 60s, tau = 60s * (1e11 - 1) = 6e18us: just under the int64 limit.
	lim := quota.RateLimit{PerMinute: 1, Burst: 100_000_000_000}
	if r, _ := l.Tokens.Reserve(ctx, "k", lim, lim.Burst, base); !r.Allowed {
		t.Fatalf("reserve Burst on an empty bucket: want allowed, got %+v", r)
	}
	r, _ := l.Tokens.Reserve(ctx, "k", lim, 1, base)
	if r.Allowed || r.RetryAfter != time.Minute {
		t.Fatalf("one past Burst: want {false 1m}, got %+v", r)
	}
}

func TestSaturatingArithmeticDoesNotWrap(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	// n*T and tau both exceed int64 microseconds. They must saturate rather
	// than wrap to small or negative values.
	huge := quota.RateLimit{PerMinute: 1, Burst: math.MaxInt64}
	for i := 0; i < 3; i++ {
		r, err := l.Tokens.Reserve(ctx, "k", huge, math.MaxInt64, base)
		if err != nil {
			t.Fatal(err)
		}
		if r.RetryAfter < 0 {
			t.Fatalf("call %d: RetryAfter must not be negative, got %v", i, r.RetryAfter)
		}
	}
	// Zero time has a hugely negative UnixMicro and must not overflow either.
	zl := quota.RateLimit{PerMinute: 60, Burst: 1}
	if r, _ := l.Requests.Allow(ctx, "z", zl, time.Time{}); !r.Allowed {
		t.Fatalf("zero time first call: want allowed, got %+v", r)
	}
	if r, _ := l.Requests.Allow(ctx, "z", zl, time.Time{}); r.Allowed || r.RetryAfter != time.Second {
		t.Fatalf("zero time second call: want {false 1s}, got %+v", r)
	}
}

func TestUnlimitedConcurrencyStoresNothing(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory().Concurrency
	for i := 0; i < 5; i++ {
		ok, err := l.Acquire(ctx, "k", string(rune('a'+i)), 0, time.Second, base)
		if err != nil || !ok {
			t.Fatalf("limit 0 acquire %d: want true, got %v err=%v", i, ok, err)
		}
	}
	if err := l.Release(ctx, "k", "a"); err != nil {
		t.Fatal(err)
	}
	// Unlimited acquires recorded nothing, so a real limit starts empty.
	if ok, _ := l.Acquire(ctx, "k", "x", 1, time.Second, base); !ok {
		t.Fatal("unlimited acquires must not occupy slots")
	}
}

func TestNonPositiveTTLIsOneSecond(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory().Concurrency
	if ok, _ := l.Acquire(ctx, "k", "a", 1, 0, base); !ok {
		t.Fatal("first acquire with ttl 0: want true")
	}
	if ok, _ := l.Acquire(ctx, "k", "b", 1, time.Second, base.Add(time.Second-time.Microsecond)); ok {
		t.Fatal("ttl 0 must hold the slot just under 1s")
	}
	if ok, _ := l.Acquire(ctx, "k", "b", 1, time.Second, base.Add(time.Second)); !ok {
		t.Fatal("ttl 0 must expire at 1s")
	}
}

func TestBackwardsClockUsesLatestSeenInstant(t *testing.T) {
	ctx := context.Background()
	l := quota.NewMemory()
	lim := quota.RateLimit{PerMinute: 60, Burst: 10}
	later := base.Add(5 * time.Second)
	for i := 0; i < 10; i++ {
		l.Requests.Allow(ctx, "k", lim, later)
	}
	// Evaluated as if at `later`, not at base: the wait is 1s, not 6s.
	r, _ := l.Requests.Allow(ctx, "k", lim, base)
	if r.Allowed || r.RetryAfter != time.Second {
		t.Fatalf("earlier now: want {false 1s}, got %+v", r)
	}
	// And an earlier now never lets a refill happen "in the past".
	if r, _ := l.Requests.Allow(ctx, "k", lim, later.Add(time.Second)); !r.Allowed {
		t.Fatalf("one interval after the latest instant: want allowed, got %+v", r)
	}
}
