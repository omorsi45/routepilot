package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/quota"
)

// TestServeForcedShutdownKeepsTheRowsAndTheBudgetCommit: a stream still open when the grace
// period ends is cut by the server's Close, and the request still writes its usage row and
// commits its budget before serve returns. The deferred closes of serve must wait for it:
// closing the usage queue first would refuse the row, and closing the budget's Redis client
// first would lose the commit.
func TestServeForcedShutdownKeepsTheRowsAndTheBudgetCommit(t *testing.T) {
	var once sync.Once
	streaming := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}`+"\n\n")
		_ = http.NewResponseController(w).Flush()
		once.Do(func() { close(streaming) })
		<-r.Context().Done() // a stream that never ends, until the gateway cuts it
	}))
	t.Cleanup(upstream.Close)

	mr := miniredis.RunT(t)
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+mr.Addr())
	db := filepath.Join(t.TempDir(), "usage.db")
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
server:
  shutdown_grace_ms: 50
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
    budget_usd: "1"
    budget_period: total
quota:
  backend: redis
  redis_url_env: ROUTEPILOT_TEST_REDIS_URL
accounting:
  driver: sqlite
  sqlite_path: %q
`, upstream.URL+"/v1", sha256Hex("test-key-capped"), filepath.ToSlash(db))))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}

	ln := listenLocal(t)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil))) }()

	req, err := http.NewRequest(http.MethodPost, "http://"+ln.Addr().String()+"/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer test-key-capped")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST error = %v", err)
	}
	defer resp.Body.Close()
	// The first content event reaching the client means the gateway has committed the stream.
	if line, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil || !strings.HasPrefix(line, "data:") {
		t.Fatalf("first stream line = %q, %v, want a data event", line, err)
	}
	<-streaming

	cancel() // the signal: the stream is still open and the grace period is 50 ms
	var serveErr error
	select {
	case serveErr = <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("serve() did not return within 30s of the signal")
	}
	if !errors.Is(serveErr, context.DeadlineExceeded) {
		t.Fatalf("serve() = %v, want the forced-shutdown error, or the test does not reach the path it pins", serveErr)
	}

	store, err := accounting.OpenSQLite(db)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	spentRows, err := store.TenantSpend(context.Background(), "capped", time.Time{}, farFuture())
	if err != nil || spentRows <= 0 {
		t.Errorf("TenantSpend(capped) = %d, %v, want the cut stream's usage row", spentRows, err)
	}
	counter := quota.NewRedisBudget(redisClientFor(t, mr), redisKeyPrefix)
	spentBudget, err := counter.Spent(context.Background(), "capped", quota.BudgetLimit{Period: quota.PeriodTotal}, time.Now())
	if err != nil || spentBudget <= 0 {
		t.Errorf("budget counter = %d, %v, want the cut stream's commit", spentBudget, err)
	}
	if spentRows != spentBudget {
		t.Errorf("ledger spend %d != budget counter %d, want the same amount in both", spentRows, spentBudget)
	}
}
