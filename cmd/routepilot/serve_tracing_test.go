package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

// otlpCollector is a loopback OTLP/HTTP endpoint that keeps the bodies it is sent.
type otlpCollector struct {
	srv    *httptest.Server
	mu     sync.Mutex
	bodies []string
}

func newOTLPCollector(t *testing.T) *otlpCollector {
	t.Helper()
	c := &otlpCollector{}
	c.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.bodies = append(c.bodies, string(b))
		c.mu.Unlock()
		w.Header().Set("Content-Type", "application/x-protobuf")
	}))
	t.Cleanup(c.srv.Close)
	return c
}

func (c *otlpCollector) all() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.Join(c.bodies, "")
}

// TestServeExportsTraces: with tracing.otlp_endpoint set, the spans of a request reach the
// collector by the time serve has shut down, because serve flushes the exporter after the
// HTTP server has drained.
func TestServeExportsTraces(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	col := newOTLPCollector(t)
	cfg := budgetServeConfig(t, upstream.URL, config.QuotaMemory, "1")
	cfg.Tracing = config.TracingConfig{OTLPEndpoint: col.srv.URL}
	url, stop := startBudgetServe(t, cfg)

	if code, body := postChat(t, url, "test-key-free"); code != http.StatusOK {
		t.Fatalf("chat: status %d body %s", code, body)
	}
	stop() // drains the server, then flushes the exporter

	got := col.all()
	for _, want := range []string{"chat.completions", "upstream.attempt", "decide", "routepilot"} {
		if !strings.Contains(got, want) {
			t.Errorf("the collector never got %q", want)
		}
	}
	for _, secret := range []string{"test-key-free", sha256Hex("test-key-free")} {
		if strings.Contains(got, secret) {
			t.Errorf("the exported traces contain %q", secret)
		}
	}
}

// TestServeWithoutTracingSendsNothing: no endpoint, no exporter.
func TestServeWithoutTracingSendsNothing(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	col := newOTLPCollector(t)
	url, stop := startBudgetServe(t, budgetServeConfig(t, upstream.URL, config.QuotaMemory, "1"))
	postChat(t, url, "test-key-free")
	stop()
	if got := col.all(); got != "" {
		t.Errorf("a collector that was never configured got %d bytes", len(got))
	}
}

// TestServeFailsOnABadTracingEndpoint: a collector address serve cannot use is a startup
// error that names tracing, not a silent noop.
func TestServeFailsOnABadTracingEndpoint(t *testing.T) {
	cfg := budgetServeConfig(t, "http://127.0.0.1:1", config.QuotaMemory, "1")
	cfg.Tracing = config.TracingConfig{OTLPEndpoint: "localhost:4318"}
	// A serve that ignores the endpoint would run until the context ends.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := serve(ctx, cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err == nil || !strings.Contains(err.Error(), "tracing") || ctx.Err() != nil {
		t.Errorf("serve() = %v (context done: %v), want a startup error about tracing", err, ctx.Err() != nil)
	}
}
