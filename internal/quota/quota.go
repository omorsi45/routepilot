// Package quota defines the limiter contracts the gateway uses to admit
// requests: request rate, token rate, and concurrency. Memory and Redis
// backends implement the same interfaces and pass the same conformance suite
// (see the quotatest package).
//
// Rate limits use GCRA (generic cell rate algorithm) over integer
// microseconds, identically in every backend:
//
//	T   = max(60_000_000 / PerMinute, 1)   emission interval, floor division
//	tau = T * (Burst - 1)                  burst tolerance
//	now = now.UnixMicro()
//
// Each key keeps a stored TAT (theoretical arrival time) and a last-seen
// instant, the latest effective clock it was evaluated at. A request of cost n
// (n is 1 for RequestLimiter) is evaluated as
//
//	eff     = max(now, last seen)          effective clock, never goes backwards
//	tat0    = max(stored TAT, eff)
//	newTAT  = tat0 + n*T
//	allowed = newTAT - eff <= tau + T
//
// On allow, newTAT is stored. On deny the stored TAT is unchanged and
// RetryAfter is (newTAT - eff) - (tau + T) microseconds. Last seen is set to
// eff on allow, on deny, and on a refund of a key that has state. It is not
// written, and nothing else is stored, for an unlimited limit, n <= 0, a
// reservation over the burst, or a refund of a key with no state. A key with
// no state is evaluated as if its TAT and last seen were now. All products and
// sums saturate at math.MaxInt64 rather than wrapping; the Redis backend
// documents the narrower range in which it is exact.
package quota

import (
	"context"
	"math"
	"time"
)

// Result is the outcome of a rate-limit decision.
type Result struct {
	Allowed bool
	// RetryAfter is how long until the same request would be allowed. It is
	// zero when Allowed, and also zero for a token reservation that can never
	// be allowed because it exceeds the burst.
	RetryAfter time.Duration
}

// RateLimit configures a GCRA limiter. PerMinute <= 0 means unlimited: every
// call is allowed and nothing is stored. Burst <= 0 is treated as 1.
type RateLimit struct {
	PerMinute, Burst int64
}

// RequestLimiter limits how often a key may make a request.
//
// The caller supplies now explicitly. Implementations never become more
// permissive when now moves backwards: a key's effective clock is the latest
// instant it has been evaluated at, so an earlier now is treated as that
// latest instant.
type RequestLimiter interface {
	Allow(ctx context.Context, key string, lim RateLimit, now time.Time) (Result, error)
}

// TokenLimiter limits token throughput for a key. The same clock rule as
// RequestLimiter applies.
type TokenLimiter interface {
	// Reserve takes n tokens, costing n*T. A reservation with n > Burst can
	// never succeed: it returns Result{Allowed: false, RetryAfter: 0} and
	// stores nothing, so the caller can answer 429 without a Retry-After.
	// n <= 0 is an allowed no-op.
	Reserve(ctx context.Context, key string, lim RateLimit, n int64, now time.Time) (Result, error)
	// Refund returns n tokens by moving the stored TAT back by n*T, never to
	// before now. A key with no stored state, an unlimited limit, or n <= 0
	// is a no-op.
	Refund(ctx context.Context, key string, lim RateLimit, n int64, now time.Time) error
}

// ConcurrencyLimiter caps in-flight work per key using expiring leases.
type ConcurrencyLimiter interface {
	// Acquire takes a slot for lease. Leases whose expiry is at or before now
	// are dropped first. If lease is already held its expiry is refreshed to
	// now+ttl and it is not counted twice. Otherwise it is added when fewer
	// than limit leases are held. limit <= 0 means unlimited: it returns true
	// and stores nothing. ttl <= 0 is treated as one second.
	Acquire(ctx context.Context, key, lease string, limit int, ttl time.Duration, now time.Time) (bool, error)
	// Release frees lease. Releasing an unknown lease or key is a no-op.
	Release(ctx context.Context, key, lease string) error
}

// Limiters bundles the three limiter kinds a gateway needs.
type Limiters struct {
	Requests    RequestLimiter
	Tokens      TokenLimiter
	Concurrency ConcurrencyLimiter
}

// gcraParams returns the emission interval and burst tolerance in
// microseconds. unlimited reports PerMinute <= 0.
func gcraParams(lim RateLimit) (t, tau int64, unlimited bool) {
	if lim.PerMinute <= 0 {
		return 0, 0, true
	}
	t = 60_000_000 / lim.PerMinute
	if t < 1 {
		t = 1
	}
	return t, satMul(t, max(lim.Burst, 1)-1), false
}

// satMul multiplies non-negative values, saturating at math.MaxInt64.
func satMul(a, b int64) int64 {
	if a == 0 || b == 0 {
		return 0
	}
	if a > math.MaxInt64/b {
		return math.MaxInt64
	}
	return a * b
}

// satAdd adds non-negative values, saturating at math.MaxInt64.
func satAdd(a, b int64) int64 {
	if a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}

// satSub returns a-b for a >= b, saturating at math.MaxInt64 when b is so
// negative that the difference does not fit.
func satSub(a, b int64) int64 {
	if b < 0 && a > math.MaxInt64+b {
		return math.MaxInt64
	}
	return a - b
}

// microsToDuration converts non-negative microseconds to a Duration,
// saturating at the largest representable Duration.
func microsToDuration(us int64) time.Duration {
	if us > int64(math.MaxInt64/time.Microsecond) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(us) * time.Microsecond
}
