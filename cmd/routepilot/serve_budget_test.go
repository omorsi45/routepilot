package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
)

// sweepCounter is a BudgetLedger that only counts Sweep calls.
type sweepCounter struct {
	quota.BudgetLedger
	sweeps atomic.Int64
	fail   bool
}

func (c *sweepCounter) Sweep(context.Context, time.Time) (int, error) {
	c.sweeps.Add(1)
	if c.fail {
		return 0, fmt.Errorf("redis is down")
	}
	return 0, nil
}

func TestSweepBudgetsRunsUntilCanceled(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprintf("failing=%v", fail), func(t *testing.T) {
			c := &sweepCounter{fail: fail}
			var logs strings.Builder
			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan struct{})
			go func() {
				sweepBudgets(ctx, c, 5*time.Millisecond, slog.New(slog.NewTextHandler(&logs, nil)))
				close(done)
			}()

			deadline := time.Now().Add(5 * time.Second)
			for c.sweeps.Load() < 3 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if got := c.sweeps.Load(); got < 3 {
				t.Fatalf("Sweep called %d times, want a call every interval", got)
			}
			cancel()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("sweepBudgets did not return after cancel")
			}
			after := c.sweeps.Load()
			time.Sleep(30 * time.Millisecond)
			if c.sweeps.Load() != after {
				t.Error("Sweep kept running after cancel")
			}
		})
	}
}

// budgetServeConfig is a valid config with a priced model and two tenants: "capped" with the
// given budget and "free" with none.
func budgetServeConfig(t *testing.T, upstreamURL, backend, capped string) *config.Config {
	t.Helper()
	quotaSection := ""
	if backend == config.QuotaRedis {
		quotaSection = "quota:\n  backend: redis\n  redis_url_env: ROUTEPILOT_TEST_REDIS_URL\n"
	}
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: m
    input_price_per_mtok: "1"
    output_price_per_mtok: "2"
    deployments:
      - provider: mockp
        upstream_model: m
tenants:
  - id: capped
    api_key_sha256: %s
    allowed_models: ["*"]
    budget_usd: %q
  - id: free
    api_key_sha256: %s
    allowed_models: ["*"]
%s`, upstreamURL+"/v1", sha256Hex("test-key-capped"), capped, sha256Hex("test-key-free"), quotaSection)))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}
	return cfg
}

func startBudgetServe(t *testing.T, cfg *config.Config) (url string, stop func()) {
	t.Helper()
	ln := listenLocal(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil))) }()
	stop = func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("serve() = %v, want nil after a clean shutdown", err)
			}
		case <-time.After(10 * time.Second):
			t.Error("serve() did not return within 10s of cancellation")
		}
	}
	t.Cleanup(cancel)
	return "http://" + ln.Addr().String(), stop
}

func redisClientFor(t *testing.T, mr *miniredis.Miniredis) *redis.Client {
	t.Helper()
	c := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func postChat(t *testing.T, url, key string) (int, string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST error = %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}

// TestServeEnforcesTheBudgetWithTheMemoryLedger: without quota.backend the budget ledger is
// the in-process one, and a tenant with a budget too small for any request is refused while a
// tenant without one is served.
func TestServeEnforcesTheBudgetWithTheMemoryLedger(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	url, stop := startBudgetServe(t, budgetServeConfig(t, upstream.URL, config.QuotaMemory, "0.000001"))

	if code, body := postChat(t, url, "test-key-capped"); code != http.StatusTooManyRequests || !strings.Contains(body, "insufficient_quota") {
		t.Errorf("capped tenant: status %d body %s, want 429 insufficient_quota", code, body)
	}
	if code, body := postChat(t, url, "test-key-free"); code != http.StatusOK {
		t.Errorf("tenant without a budget: status %d body %s, want 200", code, body)
	}
	stop()
}

// TestServeUsesTheRedisBudgetLedgerAndSweepsIt: with quota.backend redis the gateway keeps its
// budgets in Redis under the quota key prefix, and the sweeper drops the expired leases there.
func TestServeUsesTheRedisBudgetLedgerAndSweepsIt(t *testing.T) {
	old := budgetSweepEvery
	budgetSweepEvery = 10 * time.Millisecond
	t.Cleanup(func() { budgetSweepEvery = old })

	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+mr.Addr())

	// A lease left behind by a gateway that died long ago.
	seed := quota.NewRedisBudget(redisClientFor(t, mr), redisKeyPrefix)
	lim := quota.BudgetLimit{Limit: money.PicoUSD(1_000_000_000_000), Period: quota.PeriodTotal}
	if ok, _, err := seed.Reserve(context.Background(), "capped", "dead-request", lim, 5, time.Second, time.Now().Add(-time.Hour)); err != nil || !ok {
		t.Fatalf("seed reserve: ok=%v err=%v", ok, err)
	}
	const staleLeases = "routepilot:budget:{capped}:total:leases"
	if !mr.Exists(staleLeases) {
		t.Fatalf("seed did not write %s; keys: %v", staleLeases, mr.Keys())
	}

	url, stop := startBudgetServe(t, budgetServeConfig(t, upstream.URL, config.QuotaRedis, "1"))
	if code, body := postChat(t, url, "test-key-capped"); code != http.StatusOK {
		t.Fatalf("status %d body %s, want 200", code, body)
	}
	// The response is flushed before the budget commit (Ruling V1-R30), so the spent key
	// appears a moment after the client has its answer.
	var spent string
	for deadline := time.Now().Add(5 * time.Second); spent == "" && time.Now().Before(deadline); {
		for _, k := range mr.Keys() {
			if strings.HasPrefix(k, "routepilot:budget:{capped}:") && strings.HasSuffix(k, ":spent") {
				spent = k
			}
		}
		if spent == "" {
			time.Sleep(10 * time.Millisecond)
		}
	}
	if spent == "" {
		t.Errorf("no spent key under routepilot:budget:{capped}: after a request; keys: %v", mr.Keys())
	}

	deadline := time.Now().Add(5 * time.Second)
	for mr.Exists(staleLeases) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if mr.Exists(staleLeases) {
		t.Errorf("the sweeper did not drop the expired lease; keys: %v", mr.Keys())
	}
	stop()
}

// TestHasBudget: serve runs the sweeper only when some tenant has a budget, because the Redis
// sweep scans the keyspace.
func TestHasBudget(t *testing.T) {
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: mockp
    type: openaicompat
    base_url: "http://127.0.0.1:1/v1"
models:
  - id: m
    deployments:
      - provider: mockp
        upstream_model: m
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
`, sha256Hex("k"))))
	if err != nil {
		t.Fatal(err)
	}
	if hasBudget(cfg) {
		t.Error("hasBudget = true for a config without budget_usd")
	}
	capped := budgetServeConfig(t, "http://127.0.0.1:1", config.QuotaMemory, "1")
	if !hasBudget(capped) {
		t.Error("hasBudget = false for a config with budget_usd")
	}
}
