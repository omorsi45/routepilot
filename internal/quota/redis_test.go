package quota_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/quota/quotatest"
)

// runRedisConformance runs the three shared suites. newLimiters is called once
// per subtest and must return limiters that share no state with other calls,
// or use keys that do not collide (the suites derive keys from t.Name()).
func runRedisConformance(t *testing.T, newLimiters func(t *testing.T) quota.Limiters) {
	t.Run("RequestLimiter", func(t *testing.T) {
		quotatest.RunRequestLimiter(t, func(t *testing.T) quota.RequestLimiter { return newLimiters(t).Requests })
	})
	t.Run("TokenLimiter", func(t *testing.T) {
		quotatest.RunTokenLimiter(t, func(t *testing.T) quota.TokenLimiter { return newLimiters(t).Tokens })
	})
	t.Run("ConcurrencyLimiter", func(t *testing.T) {
		quotatest.RunConcurrencyLimiter(t, func(t *testing.T) quota.ConcurrencyLimiter { return newLimiters(t).Concurrency })
	})
}

func newMiniredisClient(t *testing.T) (*miniredis.Miniredis, *redis.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	c := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = c.Close() })
	return mr, c
}

// discardLogger silences go-redis's own dial-failure logging, which the
// closed-server tests would otherwise print on every operation.
type discardLogger struct{}

func (discardLogger) Printf(context.Context, string, ...interface{}) {}

// newClosedClient returns a client whose server is already gone, with retries
// trimmed so each operation fails fast.
func newClosedClient(t *testing.T) *redis.Client {
	t.Helper()
	redis.SetLogger(discardLogger{})
	mr := miniredis.RunT(t)
	c := redis.NewClient(&redis.Options{Addr: mr.Addr(), MaxRetries: -1, DialerRetries: 1})
	t.Cleanup(func() { _ = c.Close() })
	mr.Close()
	return c
}

func TestRedisConformanceMiniredis(t *testing.T) {
	runRedisConformance(t, func(t *testing.T) quota.Limiters {
		_, c := newMiniredisClient(t)
		return quota.NewRedis(c, "rp-test")
	})
}

func TestRedisConformanceReal(t *testing.T) {
	url := os.Getenv("ROUTEPILOT_TEST_REDIS_URL")
	if url == "" {
		t.Skip("ROUTEPILOT_TEST_REDIS_URL is not set; skipping conformance against a real Redis")
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		t.Fatalf("parse ROUTEPILOT_TEST_REDIS_URL: %v", err)
	}
	c := redis.NewClient(opts)
	t.Cleanup(func() { _ = c.Close() })

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

	runRedisConformance(t, func(*testing.T) quota.Limiters { return quota.NewRedis(c, prefix) })
}

func TestRedisErrorsPropagate(t *testing.T) {
	l := quota.NewRedis(newClosedClient(t), "rp-test")

	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	lim := quota.RateLimit{PerMinute: 60, Burst: 10}

	if _, err := l.Requests.Allow(ctx, "k", lim, now); err == nil {
		t.Error("Allow: want error from a closed server")
	}
	if _, err := l.Tokens.Reserve(ctx, "k", lim, 1, now); err == nil {
		t.Error("Reserve: want error from a closed server")
	}
	if err := l.Tokens.Refund(ctx, "k", lim, 1, now); err == nil {
		t.Error("Refund: want error from a closed server")
	}
	if _, err := l.Concurrency.Acquire(ctx, "k", "lease", 2, time.Second, now); err == nil {
		t.Error("Acquire: want error from a closed server")
	}
	if err := l.Concurrency.Release(ctx, "k", "lease"); err == nil {
		t.Error("Release: want error from a closed server")
	}
}

func TestRedisErrorsDoNotLeakKeys(t *testing.T) {
	l := quota.NewRedis(newClosedClient(t), "rp-test")

	_, err := l.Requests.Allow(context.Background(), "tenant-secret-key", quota.RateLimit{PerMinute: 60, Burst: 1}, time.Unix(1_700_000_000, 0))
	if err == nil {
		t.Fatal("want error from a closed server")
	}
	if got := err.Error(); strings.Contains(got, "tenant-secret-key") {
		t.Fatalf("error text must not include the key: %q", got)
	}
}

// A reservation over the burst is decided in Go (M2-R8): it succeeds with a
// zero RetryAfter even when Redis is unreachable, proving no round trip.
func TestRedisOverBurstReserveDoesNotCallRedis(t *testing.T) {
	l := quota.NewRedis(newClosedClient(t), "rp-test")

	lim := quota.RateLimit{PerMinute: 600, Burst: 100}
	r, err := l.Tokens.Reserve(context.Background(), "k", lim, 101, time.Unix(1_700_000_000, 0))
	if err != nil || r.Allowed || r.RetryAfter != 0 {
		t.Fatalf("over-burst reserve: want {false 0} and no error, got %+v err=%v", r, err)
	}
}

// Calls that are no-ops by contract never reach Redis either.
func TestRedisNoOpCallsDoNotCallRedis(t *testing.T) {
	l := quota.NewRedis(newClosedClient(t), "rp-test")

	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	lim := quota.RateLimit{PerMinute: 60, Burst: 2}
	unlimited := quota.RateLimit{PerMinute: 0, Burst: 2}

	if r, err := l.Requests.Allow(ctx, "k", unlimited, now); err != nil || !r.Allowed {
		t.Errorf("unlimited Allow: got %+v err=%v", r, err)
	}
	if r, err := l.Tokens.Reserve(ctx, "k", lim, 0, now); err != nil || !r.Allowed {
		t.Errorf("Reserve(0): got %+v err=%v", r, err)
	}
	if err := l.Tokens.Refund(ctx, "k", lim, -1, now); err != nil {
		t.Errorf("Refund(-1): %v", err)
	}
	if err := l.Tokens.Refund(ctx, "k", unlimited, 3, now); err != nil {
		t.Errorf("unlimited Refund: %v", err)
	}
	if ok, err := l.Concurrency.Acquire(ctx, "k", "a", 0, time.Second, now); err != nil || !ok {
		t.Errorf("limit 0 Acquire: got %v err=%v", ok, err)
	}
}

func TestRedisKeyLayoutAndState(t *testing.T) {
	mr, c := newMiniredisClient(t)
	l := quota.NewRedis(c, "rp")
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	nowUS := now.UnixMicro()

	// 60/min, Burst 10: T = 1s. One request stores TAT = now + T.
	rl := quota.RateLimit{PerMinute: 60, Burst: 10}
	if r, err := l.Requests.Allow(ctx, "tenant-a", rl, now); err != nil || !r.Allowed {
		t.Fatalf("Allow: %+v err=%v", r, err)
	}
	if got, want := mr.HGet("rp:rpm:{tenant-a}", "tat"), strconv.FormatInt(nowUS+1_000_000, 10); got != want {
		t.Errorf("rpm tat = %q, want %q", got, want)
	}
	if got, want := mr.HGet("rp:rpm:{tenant-a}", "seen"), strconv.FormatInt(nowUS, 10); got != want {
		t.Errorf("rpm seen = %q, want %q", got, want)
	}
	// TTL = (tat - now) + 1s = 2s.
	if got := mr.TTL("rp:rpm:{tenant-a}"); got != 2*time.Second {
		t.Errorf("rpm ttl = %v, want 2s", got)
	}

	// 600/min, Burst 100: T = 100ms. Reserve 50 stores TAT = now + 5s.
	tl := quota.RateLimit{PerMinute: 600, Burst: 100}
	if r, err := l.Tokens.Reserve(ctx, "tenant-a", tl, 50, now); err != nil || !r.Allowed {
		t.Fatalf("Reserve: %+v err=%v", r, err)
	}
	if got, want := mr.HGet("rp:tpm:{tenant-a}", "tat"), strconv.FormatInt(nowUS+5_000_000, 10); got != want {
		t.Errorf("tpm tat = %q, want %q", got, want)
	}
	if got := mr.TTL("rp:tpm:{tenant-a}"); got != 6*time.Second {
		t.Errorf("tpm ttl = %v, want 6s", got)
	}

	// A refund shrinks the TAT and the TTL with it: 20 tokens back = 2s.
	if err := l.Tokens.Refund(ctx, "tenant-a", tl, 20, now); err != nil {
		t.Fatalf("Refund: %v", err)
	}
	if got, want := mr.HGet("rp:tpm:{tenant-a}", "tat"), strconv.FormatInt(nowUS+3_000_000, 10); got != want {
		t.Errorf("tpm tat after refund = %q, want %q", got, want)
	}
	if got := mr.TTL("rp:tpm:{tenant-a}"); got != 4*time.Second {
		t.Errorf("tpm ttl after refund = %v, want 4s", got)
	}

	// Concurrency: sorted set scored by lease expiry in microseconds.
	if ok, err := l.Concurrency.Acquire(ctx, "tenant-a", "lease-1", 2, 10*time.Second, now); err != nil || !ok {
		t.Fatalf("Acquire: %v err=%v", ok, err)
	}
	score, err := mr.ZScore("rp:conc:{tenant-a}", "lease-1")
	if err != nil || int64(score) != nowUS+10_000_000 {
		t.Errorf("lease score = %v err=%v, want %d", score, err, nowUS+10_000_000)
	}
	if got := mr.TTL("rp:conc:{tenant-a}"); got != 11*time.Second {
		t.Errorf("conc ttl = %v, want 11s", got)
	}
	if err := l.Concurrency.Release(ctx, "tenant-a", "lease-1"); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if mr.Exists("rp:conc:{tenant-a}") {
		t.Error("releasing the last lease should leave no key behind")
	}
}

func TestRedisDenyDoesNotStoreTATButRecordsSeen(t *testing.T) {
	mr, c := newMiniredisClient(t)
	l := quota.NewRedis(c, "rp")
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	lim := quota.RateLimit{PerMinute: 60, Burst: 1}

	l.Requests.Allow(ctx, "k", lim, now)
	later := now.Add(300 * time.Millisecond)
	if r, _ := l.Requests.Allow(ctx, "k", lim, later); r.Allowed {
		t.Fatalf("want deny, got %+v", r)
	}
	if got, want := mr.HGet("rp:rpm:{k}", "tat"), strconv.FormatInt(now.UnixMicro()+1_000_000, 10); got != want {
		t.Errorf("tat after deny = %q, want %q (unchanged)", got, want)
	}
	if got, want := mr.HGet("rp:rpm:{k}", "seen"), strconv.FormatInt(later.UnixMicro(), 10); got != want {
		t.Errorf("seen after deny = %q, want %q", got, want)
	}
}

func TestRedisOutOfRangeValuesSaturateInsteadOfFailing(t *testing.T) {
	_, c := newMiniredisClient(t)
	l := quota.NewRedis(c, "rp")
	ctx := context.Background()

	// Burst and cost far beyond what doubles can hold exactly.
	huge := quota.RateLimit{PerMinute: 1, Burst: math.MaxInt64}
	for i := 0; i < 3; i++ {
		r, err := l.Tokens.Reserve(ctx, "huge", huge, math.MaxInt64, time.Unix(1_700_000_000, 0))
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if r.RetryAfter < 0 {
			t.Fatalf("call %d: RetryAfter must not be negative, got %v", i, r.RetryAfter)
		}
		if i == 0 && !r.Allowed {
			t.Fatalf("first full-burst reserve: want allowed, got %+v", r)
		}
		if i > 0 && (r.Allowed || r.RetryAfter <= 0) {
			t.Fatalf("call %d: want a positive RetryAfter, got %+v", i, r)
		}
	}

	// Instants outside the exact range (zero time, far future) are clamped.
	zl := quota.RateLimit{PerMinute: 60, Burst: 1}
	for _, at := range []time.Time{{}, time.Unix(1<<40, 0)} {
		key := "clock-" + at.UTC().Format(time.RFC3339)
		if r, err := l.Requests.Allow(ctx, key, zl, at); err != nil || !r.Allowed {
			t.Fatalf("%v first call: want allowed, got %+v err=%v", at, r, err)
		}
		if r, err := l.Requests.Allow(ctx, key, zl, at); err != nil || r.Allowed || r.RetryAfter != time.Second {
			t.Fatalf("%v second call: want {false 1s}, got %+v err=%v", at, r, err)
		}
	}

	// A huge TTL cannot overflow the lease expiry.
	if ok, err := l.Concurrency.Acquire(ctx, "k", "a", 1, time.Duration(math.MaxInt64), time.Unix(1_700_000_000, 0)); err != nil || !ok {
		t.Fatalf("Acquire with huge ttl: %v err=%v", ok, err)
	}
	if ok, _ := l.Concurrency.Acquire(ctx, "k", "b", 1, time.Second, time.Unix(1_700_000_000, 0)); ok {
		t.Fatal("huge-ttl lease must still hold")
	}
}
