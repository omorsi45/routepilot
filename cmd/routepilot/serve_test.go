package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/redis/go-redis/v9/logging"

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

// redisServeConfig is a valid config that selects the Redis quota backend and
// limits tenant "t" to rpm 60 with a burst of 1.
func redisServeConfig(t *testing.T, upstreamURL string) *config.Config {
	t.Helper()
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
    rpm: 60
    rpm_burst: 1
quota:
  backend: redis
  redis_url_env: ROUTEPILOT_TEST_REDIS_URL
`, upstreamURL+"/v1", sha256Hex("test-key-redis"))))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}
	return cfg
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func listenLocal(t *testing.T) net.Listener {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() error = %v", err)
	}
	return ln
}

func TestServeRedisBackendMissingURLEnv(t *testing.T) {
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "")
	cfg := redisServeConfig(t, "http://127.0.0.1:1")

	err := serve(context.Background(), cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil)))

	if err == nil || !strings.Contains(err.Error(), "ROUTEPILOT_TEST_REDIS_URL") {
		t.Fatalf("serve() error = %v, want one naming ROUTEPILOT_TEST_REDIS_URL", err)
	}
}

func TestServeRedisBackendBadURL(t *testing.T) {
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "not a redis url")
	cfg := redisServeConfig(t, "http://127.0.0.1:1")

	err := serve(context.Background(), cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil)))

	if err == nil || !strings.Contains(err.Error(), "ROUTEPILOT_TEST_REDIS_URL") {
		t.Fatalf("serve() error = %v, want one naming ROUTEPILOT_TEST_REDIS_URL", err)
	}
}

// discardRedisLogger silences go-redis's dial-failure logging for the test and
// restores the default afterwards.
type discardRedisLogger struct{}

func (discardRedisLogger) Printf(context.Context, string, ...interface{}) {}

func TestServeRedisBackendUnreachableFailsAtStartup(t *testing.T) {
	redis.SetLogger(discardRedisLogger{})
	t.Cleanup(logging.Enable)
	mr := miniredis.RunT(t)
	addr := mr.Addr()
	mr.Close()
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+addr)
	cfg := redisServeConfig(t, "http://127.0.0.1:1")

	err := serve(context.Background(), cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil)))

	if err == nil || !strings.Contains(err.Error(), "ping") {
		t.Fatalf("serve() error = %v, want a startup ping failure", err)
	}
}

func TestServeRedisBackendEnforcesRPM(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+mr.Addr())
	cfg := redisServeConfig(t, upstream.URL)
	ln := listenLocal(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil))) }()

	post := func() int {
		req, err := http.NewRequest(http.MethodPost, "http://"+ln.Addr().String()+"/v1/chat/completions",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		req.Header.Set("Authorization", "Bearer test-key-redis")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("POST error = %v", err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if got := post(); got != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", got)
	}
	if got := post(); got != http.StatusTooManyRequests {
		t.Errorf("second request status = %d, want 429 from the rpm burst of 1", got)
	}
	if keys := mr.Keys(); len(keys) == 0 {
		t.Error("miniredis holds no keys, the limiter did not use Redis")
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("serve() = %v, want nil after a clean shutdown", err)
		}
	// The 10 s bound is tolerance for a loaded machine (make verify runs
	// packages in parallel), not an expected shutdown time: shutdown takes
	// milliseconds when idle.
	case <-time.After(10 * time.Second):
		t.Fatal("serve() did not return within 10s of cancellation")
	}
}
