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
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
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

// syncBuffer is a bytes.Buffer safe for a writer goroutine and a polling test.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// TestServeSecondSignalExits checks the stop wiring: once the first signal
// has cancelled the context, runServe calls the stop function from
// signal.NotifyContext, which restores the default handling so a second
// SIGINT or SIGTERM terminates the process. The signal source is injected, and
// a request stalled upstream keeps the graceful shutdown pending so the test
// can see stop was called while shutdown was still in progress.
func TestServeSecondSignalExits(t *testing.T) {
	h := mock.NewHandler(mock.Config{})
	upstream := httptest.NewServer(h)
	t.Cleanup(upstream.Close)
	cfgPath := filepath.Join(t.TempDir(), "routepilot.yaml")
	yaml := fmt.Sprintf(`
server:
  listen: 127.0.0.1:0
  shutdown_grace_ms: 1000
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: m
    deployments:
      - provider: mockp
        upstream_model: "m;fail=stall"
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
`, upstream.URL+"/v1", sha256Hex("test-key-signal"))
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	sigCtx, firstSignal := context.WithCancel(context.Background())
	defer firstSignal()
	var stopCalls atomic.Int32
	stopped := make(chan struct{})
	orig := notifyContext
	notifyContext = func(context.Context, ...os.Signal) (context.Context, context.CancelFunc) {
		return sigCtx, func() {
			if stopCalls.Add(1) == 1 {
				close(stopped)
			}
		}
	}
	t.Cleanup(func() { notifyContext = orig })

	stderr := &syncBuffer{}
	exit := make(chan int, 1)
	go func() { exit <- runServe([]string{"-config", cfgPath}, io.Discard, stderr) }()

	addrRE := regexp.MustCompile(`"addr":"([^"]+)"`)
	var addr string
	for deadline := time.Now().Add(2 * time.Second); addr == "" && time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if m := addrRE.FindStringSubmatch(stderr.String()); m != nil {
			addr = m[1]
		}
	}
	if addr == "" {
		t.Fatalf("server did not report its address; stderr: %s", stderr.String())
	}

	// A request stalled upstream keeps the shutdown waiting out its grace.
	go func() {
		req, err := http.NewRequest(http.MethodPost, "http://"+addr+"/v1/chat/completions",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			return
		}
		req.Header.Set("Authorization", "Bearer test-key-signal")
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	}()
	for deadline := time.Now().Add(2 * time.Second); h.Stats().Requests == 0 && time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
	}
	if h.Stats().Requests == 0 {
		t.Fatal("the stalled request never reached the upstream")
	}

	if n := stopCalls.Load(); n != 0 {
		t.Fatalf("stop called %d times before the first signal, want 0", n)
	}
	firstSignal()

	select {
	case <-stopped:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("stop was not called after the first signal")
	}
	select {
	case code := <-exit:
		t.Fatalf("runServe returned %d before the stalled request ended: stop was not called ahead of shutdown", code)
	default:
	}

	select {
	case <-exit:
	case <-time.After(5 * time.Second):
		t.Fatal("runServe did not return after the shutdown grace")
	}
	if n := stopCalls.Load(); n != 1 {
		t.Errorf("stop called %d times in total, want exactly 1", n)
	}
}
