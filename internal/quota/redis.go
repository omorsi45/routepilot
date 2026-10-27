package quota

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// redisMaxExact bounds every microsecond value sent to a Lua script. Lua
// numbers are doubles, exact for integers up to 2^53, and the scripts only add
// and compare. With |now|, cost and limit each held to at most 2^52
// microseconds (about 142 years), every stored TAT and every sum the scripts
// form stays within 2^53, so decisions are exact and match the memory backend.
//
// Values beyond the bound are clamped in Go, never wrapped:
//   - now is clamped to [-2^52, 2^52] (roughly years 1827 to 2112). The clamp
//     is monotonic, so a backwards clock is still never more permissive.
//   - cost and limit are clamped to 2^52, which only makes the limiter stricter
//     for bursts above 142 years of emission intervals.
//   - a lease TTL is clamped to 2^52 microseconds.
//
// Inside the bound the Redis and memory backends return identical results.
const redisMaxExact = int64(1) << 52

// NewRedis returns limiters backed by Redis, sharing client. Each operation is
// a single atomic Lua script (EVALSHA, falling back to EVAL when the script is
// not cached), with the caller's now passed in as an argument so behavior is
// deterministic and matches the memory backend.
//
// Keys are <prefix>:rpm:{key}, <prefix>:tpm:{key} and <prefix>:conc:{key}. The
// braces are a Redis Cluster hash tag, so every key for one tenant lands in the
// same slot. Keys expire on their own once their state no longer matters.
// Redis errors are returned wrapped; they never include the key.
func NewRedis(client redis.UniversalClient, prefix string) Limiters {
	b := &redisBackend{client: client, prefix: prefix}
	return Limiters{
		Requests:    redisRequests{b},
		Tokens:      redisTokens{b},
		Concurrency: redisConcurrency{b},
	}
}

type redisBackend struct {
	client redis.UniversalClient
	prefix string
}

func (b *redisBackend) key(kind, key string) string {
	return b.prefix + ":" + kind + ":{" + key + "}"
}

// clampNow converts now to microseconds within the exact range.
func clampNow(now time.Time) int64 {
	return min(max(now.UnixMicro(), -redisMaxExact), redisMaxExact)
}

// reserve is the shared GCRA decision for requests (n = 1) and tokens.
func (b *redisBackend) reserve(ctx context.Context, kind, key string, lim RateLimit, n int64, now time.Time) (Result, error) {
	cost, limit, unlimited := gcraParams(lim, n)
	if unlimited || n <= 0 {
		return Result{Allowed: true}, nil
	}
	if n > max(lim.Burst, 1) {
		// Can never be allowed (M2-R8). Decided here: no round trip, no state.
		return Result{}, nil
	}
	cost, limit = min(cost, redisMaxExact), min(limit, redisMaxExact)

	res, err := reserveScript.Run(ctx, b.client, []string{b.key(kind, key)}, clampNow(now), cost, limit).Int64Slice()
	if err != nil {
		return Result{}, fmt.Errorf("quota: redis %s reserve: %w", kind, err)
	}
	if len(res) != 2 {
		return Result{}, fmt.Errorf("quota: redis %s reserve: unexpected script reply of %d values", kind, len(res))
	}
	if res[0] == 1 {
		return Result{Allowed: true}, nil
	}
	return Result{RetryAfter: microsToDuration(max(res[1], 0))}, nil
}

type redisRequests struct{ b *redisBackend }

func (r redisRequests) Allow(ctx context.Context, key string, lim RateLimit, now time.Time) (Result, error) {
	return r.b.reserve(ctx, "rpm", key, lim, 1, now)
}

type redisTokens struct{ b *redisBackend }

func (r redisTokens) Reserve(ctx context.Context, key string, lim RateLimit, n int64, now time.Time) (Result, error) {
	return r.b.reserve(ctx, "tpm", key, lim, n, now)
}

func (r redisTokens) Refund(ctx context.Context, key string, lim RateLimit, n int64, now time.Time) error {
	cost, _, unlimited := gcraParams(lim, n)
	if unlimited || n <= 0 {
		return nil
	}
	cost = min(cost, redisMaxExact)
	if err := refundScript.Run(ctx, r.b.client, []string{r.b.key("tpm", key)}, clampNow(now), cost).Err(); err != nil {
		return fmt.Errorf("quota: redis tpm refund: %w", err)
	}
	return nil
}

type redisConcurrency struct{ b *redisBackend }

func (c redisConcurrency) Acquire(ctx context.Context, key, lease string, limit int, ttl time.Duration, now time.Time) (bool, error) {
	if limit <= 0 {
		return true, nil
	}
	if ttl <= 0 {
		ttl = time.Second
	}
	nowUS := clampNow(now)
	expiry := nowUS + min(ttl.Microseconds(), redisMaxExact)

	got, err := acquireScript.Run(ctx, c.b.client, []string{c.b.key("conc", key)},
		nowUS, expiry, lease, strconv.Itoa(limit)).Int64()
	if err != nil {
		return false, fmt.Errorf("quota: redis conc acquire: %w", err)
	}
	return got == 1, nil
}

func (c redisConcurrency) Release(ctx context.Context, key, lease string) error {
	if err := c.b.client.ZRem(ctx, c.b.key("conc", key), lease).Err(); err != nil {
		return fmt.Errorf("quota: redis conc release: %w", err)
	}
	return nil
}
