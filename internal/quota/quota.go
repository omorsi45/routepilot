// Package quota defines the limiter contracts the gateway uses to admit
// requests: request rate, token rate, and concurrency. Memory and Redis
// backends implement the same interfaces and pass the same conformance suite
// (see the quotatest package).
//
// Rate limits use GCRA (generic cell rate algorithm) over integer
// microseconds, identically in every backend. Costs and the burst window are
// computed exactly, with 128-bit intermediates, so no rate is rounded to a
// whole emission interval:
//
//	cost(n) = ceil(n * 60_000_000 / PerMinute)           time n units take
//	limit   = floor(Burst * 60_000_000 / PerMinute)      burst window
//	now     = now.UnixMicro()
//
// with Burst <= 0 treated as 1, limit raised to cost(Burst) when rounding
// leaves it lower (so a reservation within the burst fits an idle key), and
// both saturating at math.MaxInt64. Each key keeps a stored TAT (theoretical
// arrival time) and a last-seen instant, the latest effective clock it was
// evaluated at. A request of n units (n is 1 for RequestLimiter) is evaluated
// as
//
//	eff     = max(now, last seen)          effective clock, never goes backwards
//	tat0    = max(stored TAT, eff)
//	newTAT  = tat0 + cost(n)
//	allowed = newTAT - eff <= limit
//
// On allow, newTAT is stored. On deny the stored TAT is unchanged and
// RetryAfter is (newTAT - eff) - limit microseconds. Last seen is set to
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
	"math/bits"
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
	// Reserve takes n tokens, costing cost(n) (see the package comment). A
	// reservation with n > Burst can
	// never succeed: it returns Result{Allowed: false, RetryAfter: 0} and
	// stores nothing, so the caller can answer 429 without a Retry-After.
	// n <= 0 is an allowed no-op.
	Reserve(ctx context.Context, key string, lim RateLimit, n int64, now time.Time) (Result, error)
	// Refund returns n tokens by moving the stored TAT back by cost(n), never to
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

// Limiters bundles the limiters a gateway needs: the three rate kinds and the tenant
// budget ledger.
type Limiters struct {
	Requests    RequestLimiter
	Tokens      TokenLimiter
	Concurrency ConcurrencyLimiter
	Budget      BudgetLedger
}

// gcraParams returns the cost of n units and the burst limit for lim, both in
// microseconds. unlimited reports PerMinute <= 0.
//
// The limit is gcraLimit raised to the cost of a full burst. Cost rounds up
// and the limit rounds down, so when Burst*60_000_000/PerMinute is not a whole
// number the floor alone would sit one microsecond below the cost of Burst
// units, and a reservation within the burst could never fit an idle key.
func gcraParams(lim RateLimit, n int64) (cost, limit int64, unlimited bool) {
	if lim.PerMinute <= 0 {
		return 0, 0, true
	}
	burst := max(lim.Burst, 1)
	limit = max(gcraLimit(burst, lim.PerMinute), gcraCost(burst, lim.PerMinute))
	return gcraCost(n, lim.PerMinute), limit, false
}

// microsPerMinute converts a per-minute rate to microseconds per unit.
const microsPerMinute = 60_000_000

// gcraCost is the time n units take at perMinute units per minute, in
// microseconds: ceil(n * 60_000_000 / perMinute), computed exactly with a
// 128-bit product and saturating at math.MaxInt64. It is 0 for n <= 0 or
// perMinute <= 0.
func gcraCost(n, perMinute int64) int64 {
	return mulDivSat(n, perMinute, true)
}

// gcraLimit is the burst window of burst units (at least 1) at perMinute units
// per minute, in microseconds: floor(burst * 60_000_000 / perMinute), computed
// exactly and saturating at math.MaxInt64. It is 0 for perMinute <= 0.
func gcraLimit(burst, perMinute int64) int64 {
	return mulDivSat(max(burst, 1), perMinute, false)
}

// WindowMicros is the GCRA burst window of a limit, in microseconds: the span
// of time the full burst takes at the limit's rate. The config check that
// bounds a stored arrival time to a day uses it, so it and the limiters agree.
func WindowMicros(burst, perMinute int64) int64 {
	return gcraLimit(burst, perMinute)
}

// mulDivSat returns n * 60_000_000 / perMinute, rounded up or down, using the
// full 128-bit product. A result that does not fit an int64 saturates at
// math.MaxInt64. It returns 0 for n <= 0 or perMinute <= 0.
func mulDivSat(n, perMinute int64, roundUp bool) int64 {
	if n <= 0 || perMinute <= 0 {
		return 0
	}
	d := uint64(perMinute)
	hi, lo := bits.Mul64(uint64(n), microsPerMinute)
	if hi >= d { // Div64 would overflow: the quotient needs more than 64 bits
		return math.MaxInt64
	}
	q, r := bits.Div64(hi, lo, d)
	if q > math.MaxInt64 {
		return math.MaxInt64
	}
	if roundUp && r != 0 {
		if q == math.MaxInt64 {
			return math.MaxInt64
		}
		q++
	}
	return int64(q)
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
