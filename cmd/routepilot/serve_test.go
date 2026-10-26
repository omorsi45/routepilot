package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

func TestServeRequiresConfig(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run([]string{"serve"}, &stdout, &stderr)

	if code != 2 {
		t.Fatalf("run(serve) = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "-config") {
		t.Errorf("stderr = %q, want it to mention -config", stderr.String())
	}
}

func TestServeMissingConfigFileExits1(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run([]string{"serve", "-config", "/nonexistent/routepilot.yaml"}, &stdout, &stderr)

	if code != 1 {
		t.Fatalf("run(serve) = %d, want 1; stderr = %q", code, stderr.String())
	}
}

func TestServeStartsAndShutsDown(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: m
    deployments:
      - provider: mockp
        upstream_model: m
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
`, upstream.URL+"/v1", strings.Repeat("a", 64))))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil))) }()

	resp, err := http.Get("http://" + ln.Addr().String() + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz error = %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("GET /healthz status = %d, want 200", resp.StatusCode)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("serve() = %v, want nil after a clean shutdown", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("serve() did not return within 2s of cancellation")
	}
}
