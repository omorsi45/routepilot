package main

import (
	"context"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

// serveLogs starts serve with cfg, waits for it to report that it is listening, stops it and
// returns what it logged.
func serveLogs(t *testing.T, cfg *config.Config) string {
	t.Helper()
	logs := &syncBuffer{}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, listenLocal(t), slog.New(slog.NewJSONHandler(logs, nil))) }()
	for deadline := time.Now().Add(5 * time.Second); !strings.Contains(logs.String(), `"listening"`) && time.Now().Before(deadline); {
		time.Sleep(5 * time.Millisecond)
	}
	if !strings.Contains(logs.String(), `"listening"`) {
		t.Fatalf("serve did not start; log: %s", logs.String())
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("serve() = %v, want nil after a clean shutdown", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("serve() did not return within 10s of cancellation")
	}
	return logs.String()
}

// warnLines are the log lines that warn about a budget held in one process's memory.
func warnLines(logs string) []string {
	var out []string
	for _, l := range strings.Split(logs, "\n") {
		if strings.Contains(l, `"level":"WARN"`) && strings.Contains(l, "resets on restart") {
			out = append(out, l)
		}
	}
	return out
}

// TestServeWarnsOnceForABudgetOnTheMemoryBackend: with the default quota backend the budget
// counter lives in the process, so a restart (or a second replica) forgets what was spent.
// Start says so, once for each tenant with a budget and naming it.
func TestServeWarnsOnceForABudgetOnTheMemoryBackend(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)

	lines := warnLines(serveLogs(t, budgetServeConfig(t, upstream.URL, config.QuotaMemory, "1")))

	if len(lines) != 1 {
		t.Fatalf("got %d warning lines %q, want exactly 1 (the tenant without a budget gets none)", len(lines), lines)
	}
	for _, want := range []string{`"tenant":"capped"`, "per process", "resets on restart"} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("warning %s does not contain %q", lines[0], want)
		}
	}
	if strings.Contains(lines[0], "free") {
		t.Errorf("warning %s names the tenant without a budget", lines[0])
	}
}

// TestServeDoesNotWarnWhenNoTenantHasABudget: nothing to lose, nothing to say.
func TestServeDoesNotWarnWhenNoTenantHasABudget(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)

	if lines := warnLines(serveLogs(t, accountingConfig(t, upstream.URL, ""))); len(lines) != 0 {
		t.Errorf("got warning lines %q for a config without budgets, want none", lines)
	}
}

// TestServeDoesNotWarnForABudgetOnRedis: the Redis counter outlives the process and is shared
// by every replica.
func TestServeDoesNotWarnForABudgetOnRedis(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+mr.Addr())

	if lines := warnLines(serveLogs(t, budgetServeConfig(t, upstream.URL, config.QuotaRedis, "1"))); len(lines) != 0 {
		t.Errorf("got warning lines %q for a budget on Redis, want none", lines)
	}
}
