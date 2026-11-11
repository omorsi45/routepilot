package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

// TestServeExposesMetrics: GET /metrics on the main listener needs no key and carries the
// RoutePilot series of the requests served and, because serve adds them to the registry, the
// Go runtime and process series.
func TestServeExposesMetrics(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	url, stop := startBudgetServe(t, budgetServeConfig(t, upstream.URL, config.QuotaMemory, "1"))

	if code, body := postChat(t, url, "test-key-free"); code != http.StatusOK {
		t.Fatalf("chat: status %d body %s", code, body)
	}
	resp, err := http.Get(url + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	body := string(b)
	stop()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /metrics without a key: status %d: %s", resp.StatusCode, body)
	}
	for _, want := range []string{
		`rp_requests_total{model="m",policy_role="pinned",status="200",tenant="free"} 1`,
		`rp_breaker_state{deployment="mockp/m"} 0`,
		"# TYPE go_goroutines gauge",
		"# TYPE process_cpu_seconds_total counter",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("/metrics lacks %q\n%s", want, firstLines(body, 60))
		}
	}
}

func firstLines(s string, n int) string {
	lines := strings.SplitN(s, "\n", n+1)
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, "\n")
}
