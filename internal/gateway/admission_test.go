package gateway

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/quota"
)

// admissionNow is the fixed gateway clock, so GCRA state never drifts between
// requests inside a test.
var admissionNow = time.Unix(1700000000, 0)

// newAdmissionGateway builds a gateway whose config is adjusted by mutate
// before New, with the given limiters (nil fields use the memory backend).
func newAdmissionGateway(t *testing.T, mutate func(*config.Config), p provider.Provider, lim quota.Limiters) (*Server, *bytes.Buffer) {
	t.Helper()
	cfg := buildConfig(t, cfgOpts{})
	if mutate != nil {
		mutate(cfg)
	}
	logs := &bytes.Buffer{}
	gw, err := New(cfg, map[string]provider.Provider{"mockp": p}, Options{
		Logger:   slog.New(slog.NewTextHandler(logs, nil)),
		Now:      func() time.Time { return admissionNow },
		Limiters: lim,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return gw, logs
}

// tenantA returns tenant-a's config for mutation.
func tenantA(cfg *config.Config) *config.TenantConfig { return &cfg.Tenants[0] }

// tenantB returns tenant-b's config for mutation.
func tenantB(cfg *config.Config) *config.TenantConfig { return &cfg.Tenants[1] }

// completionWithUsage is a provider that answers every request with the usage
// returned by usage (nil usage means the response carries none).
func completionWithUsage(usage func() *openai.Usage) *fakeProvider {
	return &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		return &openai.ChatCompletion{Usage: usage()}, nil
	}}
}

func okProvider() *fakeProvider {
	return completionWithUsage(func() *openai.Usage { return nil })
}

// inUpperHi is InUpper of the one-message "hi" request the tests send.
func inUpperHi(t *testing.T) int {
	t.Helper()
	req, apiErr := openai.ParseChatCompletionRequest([]byte(chatBody("model-a", "hi")))
	if apiErr != nil {
		t.Fatalf("parse: %v", apiErr)
	}
	return catalog.Derive(req).InUpper
}

// maxTokensFor returns the max_tokens that makes InUpper+K equal n.
func maxTokensFor(t *testing.T, n int) int {
	t.Helper()
	k := n - inUpperHi(t)
	if k <= 0 {
		t.Fatalf("n=%d is not above InUpper", n)
	}
	return k
}

func bodyWithMaxTokens(model string, maxTokens int, stream bool) string {
	return fmt.Sprintf(`{"model":%q,"stream":%t,"max_tokens":%d,"messages":[{"role":"user","content":"hi"}]}`, model, stream, maxTokens)
}

func chat(gw *Server, key, body string) *httptest.ResponseRecorder {
	return do(gw.Handler(), http.MethodPost, "/v1/chat/completions", key, body)
}

func wantRateLimited(t *testing.T, rec *httptest.ResponseRecorder, message, retryAfter string) {
	t.Helper()
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429: %s", rec.Code, rec.Body.String())
	}
	e := decodeErr(t, rec).Error
	if e.Code != "rate_limit_exceeded" || e.Message != message {
		t.Errorf("error = %q / %q, want rate_limit_exceeded / %q", e.Code, e.Message, message)
	}
	got, present := rec.Header()["Retry-After"]
	switch {
	case retryAfter == "" && present:
		t.Errorf("Retry-After = %v, want absent", got)
	case retryAfter != "" && (!present || got[0] != retryAfter):
		t.Errorf("Retry-After = %v, want %q", got, retryAfter)
	}
}

func TestRPMLimit(t *testing.T) {
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		for _, tn := range []*config.TenantConfig{tenantA(cfg), tenantB(cfg)} {
			tn.RPM, tn.RPMBurst = 60, 2
		}
	}, okProvider(), quota.Limiters{})

	body := chatBody("model-a", "hi")
	for i := 0; i < 2; i++ {
		if rec := chat(gw, keyA, body); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d, want 200: %s", i+1, rec.Code, rec.Body.String())
		}
	}
	wantRateLimited(t, chat(gw, keyA, body), "tenant request rate exceeded", "1")

	if rec := chat(gw, keyB, body); rec.Code != http.StatusOK {
		t.Errorf("other tenant: status %d, want 200 (limits are per tenant): %s", rec.Code, rec.Body.String())
	}
}

func TestTPMLimitAndRefund(t *testing.T) {
	var used atomic.Pointer[openai.Usage]
	p := completionWithUsage(func() *openai.Usage { return used.Load() })
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 1000
	}, p, quota.Limiters{})

	req := func(n int) *httptest.ResponseRecorder {
		return chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, n), false))
	}

	// The first request uses everything it reserved, so nothing is refunded.
	used.Store(&openai.Usage{PromptTokens: 100, CompletionTokens: 500})
	if rec := req(600); rec.Code != http.StatusOK {
		t.Fatalf("first 600: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	// 400 tokens are left, so a second 600 waits for 200 tokens at 1000/min: 12 s.
	wantRateLimited(t, req(600), "tenant token rate exceeded", "12")

	// 400 fits exactly. It then uses 100, so 300 are refunded...
	used.Store(&openai.Usage{PromptTokens: 40, CompletionTokens: 60})
	if rec := req(400); rec.Code != http.StatusOK {
		t.Fatalf("400: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	// ...which admits a third request of 300 and nothing more.
	used.Store(&openai.Usage{PromptTokens: 150, CompletionTokens: 150})
	if rec := req(300); rec.Code != http.StatusOK {
		t.Fatalf("300 after the refund: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	// The bucket is empty: one more token (n = InUpper+1 = 27 at 60 ms each)
	// waits 1.62 s, which rounds up to 2.
	wantRateLimited(t, req(inUpperHi(t)+1), "tenant token rate exceeded", "2")
}

func TestTPMNoRefundWhenUsageUnknown(t *testing.T) {
	// The upstream was tried and failed without reporting usage: the tokens
	// may have been spent, so the reservation stands.
	p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		return nil, &provider.Error{Kind: provider.KindHTTP, Status: http.StatusBadRequest, Message: "bad"}
	}}
	lim := quota.NewMemory()
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 1000
	}, p, lim)

	if rec := chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, 1000), false)); rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
	}
	res, err := lim.Tokens.Reserve(context.Background(), "tenant-a", quota.RateLimit{PerMinute: 1000, Burst: 1000}, 1, admissionNow)
	if err != nil || res.Allowed {
		t.Errorf("Reserve(1) = %+v, %v; want denied because the reservation was kept", res, err)
	}
}

func TestRefundAmount(t *testing.T) {
	tests := []struct {
		name     string
		n        int64
		attempts int
		usage    *openai.Usage
		want     int64
	}{
		{"no attempt sent refunds everything", 600, 0, nil, 600},
		{"no attempt sent ignores stale usage", 600, 0, &openai.Usage{PromptTokens: 1}, 600},
		{"attempted without usage keeps the reservation", 600, 1, nil, 0},
		{"usage below the reservation refunds the difference", 600, 1, &openai.Usage{PromptTokens: 50, CompletionTokens: 50}, 500},
		{"usage equal to the reservation refunds nothing", 600, 2, &openai.Usage{PromptTokens: 100, CompletionTokens: 500}, 0},
		{"usage above the reservation refunds nothing", 600, 1, &openai.Usage{PromptTokens: 400, CompletionTokens: 400}, 0},
		{"zero usage refunds everything", 600, 1, &openai.Usage{}, 600},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rs := &reqState{attempts: tt.attempts, usage: tt.usage}
			if got := refundAmount(tt.n, rs); got != tt.want {
				t.Errorf("refundAmount = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestRetryAfterSeconds(t *testing.T) {
	tests := map[time.Duration]string{
		time.Microsecond:         "1",
		time.Second:              "1",
		time.Second + 1:          "2",
		12 * time.Second:         "12",
		12*time.Second + 1:       "13",
		time.Duration(1<<63 - 1): "9223372037",
	}
	for d, want := range tests {
		if got := retryAfterSeconds(d); got != want {
			t.Errorf("retryAfterSeconds(%v) = %q, want %q", d, got, want)
		}
	}
}

func TestTokenBurstExceeded(t *testing.T) {
	called := false
	p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		called = true
		return &openai.ChatCompletion{}, nil
	}}
	lim := quota.NewMemory()
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		tn := tenantA(cfg)
		tn.TPM, tn.TPMBurst, tn.MaxConcurrency = 1000, 500, 1
	}, p, lim)

	rec := chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, 600), false))

	wantRateLimited(t, rec, "request exceeds tenant token burst; lower max_tokens", "")
	if called {
		t.Error("the upstream was called for a request over the token burst")
	}
	// The rejected request must not keep its concurrency lease.
	ok, err := lim.Concurrency.Acquire(context.Background(), "tenant-a", "probe", 1, time.Minute, admissionNow)
	if err != nil || !ok {
		t.Errorf("Acquire after the rejection = %v, %v; want the slot free", ok, err)
	}
}

// gatedStream serves chunks, then blocks until release closes (clean end) or
// the stream is closed (canceled).
type gatedStream struct {
	chunks  []*openai.ChatCompletionChunk
	release <-chan struct{}
	closed  chan struct{}
	once    sync.Once
}

func newGatedStream(release <-chan struct{}, chunks ...*openai.ChatCompletionChunk) *gatedStream {
	return &gatedStream{chunks: chunks, release: release, closed: make(chan struct{})}
}

func (g *gatedStream) Recv() (*openai.ChatCompletionChunk, error) {
	if len(g.chunks) > 0 {
		c := g.chunks[0]
		g.chunks = g.chunks[1:]
		return c, nil
	}
	select {
	case <-g.release:
		return nil, io.EOF
	case <-g.closed:
		return nil, context.Canceled
	}
}

func (g *gatedStream) Close() error {
	g.once.Do(func() { close(g.closed) })
	return nil
}

// stalledProvider streams one chunk and then stalls until release closes.
// started receives one value per stream opened.
func stalledProvider(release <-chan struct{}, started chan<- struct{}, first *openai.ChatCompletionChunk) *fakeProvider {
	return &fakeProvider{
		complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			return &openai.ChatCompletion{}, nil
		},
		stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
			started <- struct{}{}
			return newGatedStream(release, first), nil
		},
	}
}

// startStream posts a streaming request on a real connection and returns the
// response once its headers arrive.
func startStream(t *testing.T, ctx context.Context, url, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+keyA)
	resp, err := (&http.Client{Transport: &http.Transport{}}).Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func TestConcurrencyLimit(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 4)
	p := stalledProvider(release, started, textChunk("hello"))
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).MaxConcurrency = 1 }, p, quota.Limiters{})
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)

	resp := startStream(t, context.Background(), srv.URL, streamReqBody("model-a", false))
	<-started

	wantRateLimited(t, chat(gw, keyA, chatBody("model-a", "hi")), "tenant concurrency limit reached", "1")
	if rec := chat(gw, keyB, chatBody("model-a", "hi")); rec.Code == http.StatusTooManyRequests {
		t.Errorf("other tenant was limited by tenant-a's concurrency: %s", rec.Body.String())
	}

	close(release)
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("read first stream: %v", err)
	}
	// The slot is released just after the response ends, so poll for it.
	waitFor(t, time.Second, "the slot to be released", func() bool {
		return chat(gw, keyA, chatBody("model-a", "hi")).Code == http.StatusOK
	})
}

// startTrickle posts a chat request whose body arrives only when the test
// writes to the returned pipe, so the handler sits in its body read. The
// recorder it served arrives on the channel once the handler returns.
func startTrickle(gw *Server, key string) (*io.PipeWriter, <-chan *httptest.ResponseRecorder) {
	pr, pw := io.Pipe()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", pr)
	req.Header.Set("Authorization", "Bearer "+key)
	rec := httptest.NewRecorder()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		gw.Handler().ServeHTTP(rec, req)
		done <- rec
	}()
	return pw, done
}

// TestTenantConcurrencyIsTakenBeforeTheBodyRead: the tenant's concurrency
// lease is taken right after the global slot, before the body is read, so a
// tenant whose bodies trickle in cannot hold every global slot. Here the
// gateway has two slots and tenant-a may run one request.
func TestTenantConcurrencyIsTakenBeforeTheBodyRead(t *testing.T) {
	lim := quota.NewMemory()
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		cfg.Server.MaxInflight, cfg.Server.QueueWaitMs = 2, 50
		tenantA(cfg).MaxConcurrency = 1
	}, okProvider(), lim)
	bg := context.Background()

	// Tenant-a's first request trickles its body, holding a slot and its lease.
	pw1, done1 := startTrickle(gw, keyA)
	defer pw1.Close()
	waitFor(t, time.Second, "tenant-a's first request to hold its lease while its body is unsent", func() bool {
		ok, err := lim.Concurrency.Acquire(bg, "tenant-a", "probe", 1, time.Minute, admissionNow)
		if err == nil && ok {
			_ = lim.Concurrency.Release(bg, "tenant-a", "probe")
			return false
		}
		return err == nil
	})

	// Its second request is over the tenant limit: answered before any body.
	pw2, done2 := startTrickle(gw, keyA)
	defer pw2.Close()
	select {
	case rec := <-done2:
		wantRateLimited(t, rec, "tenant concurrency limit reached", "1")
	case <-time.After(2 * time.Second):
		t.Fatal("tenant-a's second request is still waiting for its body, holding a global slot")
	}

	// So tenant-b is admitted: tenant-a holds one global slot, not both.
	if rec := chat(gw, keyB, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("tenant-b: status %d, want 200: %s", rec.Code, rec.Body.String())
	}

	// The first request completes once its body arrives and frees the lease.
	if _, err := io.WriteString(pw1, chatBody("model-a", "hi")); err != nil {
		t.Fatalf("write body: %v", err)
	}
	_ = pw1.Close()
	if rec := <-done1; rec.Code != http.StatusOK {
		t.Fatalf("first request: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	waitFor(t, time.Second, "the lease to be released", func() bool {
		ok, err := lim.Concurrency.Acquire(bg, "tenant-a", "probe", 1, time.Minute, admissionNow)
		return err == nil && ok
	})
}

// TestBodyFailuresReleaseTheEarlyLease: a request that fails after taking its
// lease but before admission (bad body, unknown model) hands the lease back
// before answering, so the tenant's next request is not refused.
func TestBodyFailuresReleaseTheEarlyLease(t *testing.T) {
	for name, body := range map[string]string{
		"unparsable body": "not json",
		"unknown model":   chatBody("no-such-model", "hi"),
	} {
		t.Run(name, func(t *testing.T) {
			stub := &errLimiters{mem: quota.NewMemory(), which: "none"}
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).MaxConcurrency = 1 }, okProvider(), stub.limiters())

			rec := chat(gw, keyA, body)
			if rec.Code < 400 || rec.Code >= 500 {
				t.Fatalf("status %d, want a 4xx: %s", rec.Code, rec.Body.String())
			}

			id := rec.Header().Get(requestIDHeader)
			if got := stub.releasedLeases(); len(got) != 1 || got[0] != id || id == "" {
				t.Errorf("Release leases = %v, want exactly the request ID %q", got, id)
			}
			if rec := chat(gw, keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
				t.Errorf("next request: status %d, want 200: %s", rec.Code, rec.Body.String())
			}
		})
	}
}

// acquireCall is one Acquire seen by acquireRecorder.
type acquireCall struct {
	lease string
	ttl   time.Duration
}

// acquireRecorder records the lease ID and TTL of every Acquire.
type acquireRecorder struct {
	quota.ConcurrencyLimiter
	mu    sync.Mutex
	calls []acquireCall
}

func (a *acquireRecorder) Acquire(ctx context.Context, key, lease string, limit int, ttl time.Duration, now time.Time) (bool, error) {
	a.mu.Lock()
	a.calls = append(a.calls, acquireCall{lease, ttl})
	a.mu.Unlock()
	return a.ConcurrencyLimiter.Acquire(ctx, key, lease, limit, ttl, now)
}

// TestConcurrencyLeaseIsRefreshedToTheRealDeadline: the lease taken before the
// body is read lives for the longest deadline the tenant may be given plus 60
// s. Once the request is parsed the same lease is refreshed to the request's
// real deadline plus 60 s.
func TestConcurrencyLeaseIsRefreshedToTheRealDeadline(t *testing.T) {
	withTimeout := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":5000}}`
	tests := []struct {
		name        string
		tenantMaxMs int
		body        string
		early, real time.Duration
	}{
		{"tenant cap and client timeout", 20_000, withTimeout, 80 * time.Second, 65 * time.Second},
		{"server defaults", 0, chatBody("model-a", "hi"), 660 * time.Second, 180 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &acquireRecorder{ConcurrencyLimiter: quota.NewMemory().Concurrency}
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				tn := tenantA(cfg)
				tn.MaxConcurrency, tn.MaxTimeoutMs = 1, tt.tenantMaxMs
			}, okProvider(), quota.Limiters{Concurrency: rec})

			resp := chat(gw, keyA, tt.body)
			if resp.Code != http.StatusOK {
				t.Fatalf("status %d, want 200: %s", resp.Code, resp.Body.String())
			}

			id := resp.Header().Get(requestIDHeader)
			want := []acquireCall{{id, tt.early}, {id, tt.real}}
			rec.mu.Lock()
			defer rec.mu.Unlock()
			if id == "" || len(rec.calls) != len(want) || rec.calls[0] != want[0] || rec.calls[1] != want[1] {
				t.Errorf("Acquire calls = %+v, want %+v (same lease, early TTL then refreshed TTL)", rec.calls, want)
			}
		})
	}
}

func TestGlobalSemaphoreOverload(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 4)
	p := &fakeProvider{complete: func(ctx context.Context, _ *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		started <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
		}
		return &openai.ChatCompletion{}, nil
	}}
	gw, logs := newAdmissionGateway(t, func(cfg *config.Config) {
		cfg.Server.MaxInflight, cfg.Server.QueueWaitMs = 1, 50
	}, p, quota.Limiters{})

	first := make(chan int, 1)
	go func() { first <- chat(gw, keyA, chatBody("model-a", "hi")).Code }()
	<-started

	t0 := time.Now()
	rec := chat(gw, keyB, chatBody("model-a", "hi"))
	elapsed := time.Since(t0)
	if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
		t.Fatalf("status %d body %s, want 503 overloaded", rec.Code, rec.Body.String())
	}
	if elapsed < 40*time.Millisecond || elapsed > 200*time.Millisecond {
		t.Errorf("rejected after %v, want about the 50ms queue wait (under 200ms)", elapsed)
	}
	if !strings.Contains(logs.String(), "error_code=overloaded") {
		t.Errorf("log lacks error_code=overloaded:\n%s", logs.String())
	}

	// Authentication comes first: a bad key is 401 even when the gateway is full.
	if rec := chat(gw, "test-key-nobody", chatBody("model-a", "hi")); rec.Code != http.StatusUnauthorized {
		t.Errorf("bad key while full: status %d, want 401", rec.Code)
	}
	// The slot is taken before the body is read: an unparsable body waits too.
	if rec := chat(gw, keyB, "not json"); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("bad body while full: status %d, want 503 (slot is taken before parsing)", rec.Code)
	}

	close(release)
	if code := <-first; code != http.StatusOK {
		t.Fatalf("first request status %d, want 200", code)
	}
	if rec := chat(gw, keyB, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
		t.Errorf("after the slot freed: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
}

func TestSemaphoreQueuedClientLeaving(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 4)
	p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		started <- struct{}{}
		<-release
		return &openai.ChatCompletion{}, nil
	}}
	gw, logs := newAdmissionGateway(t, func(cfg *config.Config) {
		cfg.Server.MaxInflight, cfg.Server.QueueWaitMs = 1, 5000
	}, p, quota.Limiters{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		chat(gw, keyA, chatBody("model-a", "hi"))
	}()
	defer func() {
		close(release)
		<-done // join the first request before the test returns
	}()
	<-started

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("model-a", "hi"))).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyB)
	rec := httptest.NewRecorder()
	time.AfterFunc(30*time.Millisecond, cancel)
	gw.Handler().ServeHTTP(rec, req)

	if rec.Body.Len() != 0 {
		t.Errorf("wrote %q for a client that left", rec.Body.String())
	}
	if !strings.Contains(logs.String(), "status=499") {
		t.Errorf("log lacks status=499:\n%s", logs.String())
	}
}

func TestDisconnectReleasesQuota(t *testing.T) {
	started := make(chan struct{}, 1)
	// One chunk carries content and usage; the usage must be captured even
	// though this client did not ask for it. Then the upstream stalls.
	first := textChunk("hello")
	first.Usage = &openai.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	p := stalledProvider(nil, started, first)

	lim := quota.NewMemory()
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		tn := tenantA(cfg)
		tn.MaxConcurrency, tn.TPM, tn.TPMBurst = 1, 1000, 1000
	}, p, lim)
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// n is the whole burst, so the bucket is empty while the stream is open.
	resp := startStream(t, ctx, srv.URL, bodyWithMaxTokens("model-a", maxTokensFor(t, 1000), true))
	<-started
	if _, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil {
		t.Fatalf("read first event: %v", err)
	}
	tpm := quota.RateLimit{PerMinute: 1000, Burst: 1000}
	bg := context.Background()
	if res, err := lim.Tokens.Reserve(bg, "tenant-a", tpm, 1, admissionNow); err != nil || res.Allowed {
		t.Fatalf("Reserve(1) while streaming = %+v, %v; want denied (bucket empty)", res, err)
	}
	if ok, err := lim.Concurrency.Acquire(bg, "tenant-a", "probe", 1, time.Minute, admissionNow); err != nil || ok {
		t.Fatalf("Acquire while streaming = %v, %v; want denied (slot held)", ok, err)
	}

	cancel()

	waitFor(t, time.Second, "the concurrency slot to be released", func() bool {
		ok, err := lim.Concurrency.Acquire(bg, "tenant-a", "probe", 1, time.Minute, admissionNow)
		return err == nil && ok
	})
	_ = lim.Concurrency.Release(bg, "tenant-a", "probe")
	// 1000 reserved, 15 used: 985 come back, so exactly 985 fit again.
	waitFor(t, time.Second, "the unused tokens to be refunded", func() bool {
		res, err := lim.Tokens.Reserve(bg, "tenant-a", tpm, 985, admissionNow)
		return err == nil && res.Allowed
	})
	if res, _ := lim.Tokens.Reserve(bg, "tenant-a", tpm, 1, admissionNow); res.Allowed {
		t.Error("more than the unused tokens were refunded")
	}
}

// errLimiters fails the selected limiter and counts every limiter call.
type errLimiters struct {
	mem   quota.Limiters
	err   error
	which string // "requests", "tokens", "concurrency", "refund", "release"; anything else never fails
	hang  string // "requests", "tokens" or "concurrency": that call blocks until its context is done
	calls atomic.Int32

	mu       sync.Mutex
	released []string // lease IDs passed to Release, in call order
}

func (e *errLimiters) releasedLeases() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.released...)
}

func (e *errLimiters) limiters() quota.Limiters {
	return quota.Limiters{Requests: e, Tokens: e, Concurrency: e}
}

func (e *errLimiters) fail(kind string) error {
	e.calls.Add(1)
	if kind == e.which {
		return e.err
	}
	return nil
}

// block waits for ctx when kind is the hanging call, as a stalled backend would.
func (e *errLimiters) block(ctx context.Context, kind string) error {
	if kind != e.hang {
		return nil
	}
	<-ctx.Done()
	return ctx.Err()
}

func (e *errLimiters) Allow(ctx context.Context, key string, lim quota.RateLimit, now time.Time) (quota.Result, error) {
	if err := e.block(ctx, "requests"); err != nil {
		return quota.Result{}, err
	}
	if err := e.fail("requests"); err != nil {
		return quota.Result{}, err
	}
	return e.mem.Requests.Allow(ctx, key, lim, now)
}

func (e *errLimiters) Reserve(ctx context.Context, key string, lim quota.RateLimit, n int64, now time.Time) (quota.Result, error) {
	if err := e.block(ctx, "tokens"); err != nil {
		return quota.Result{}, err
	}
	if err := e.fail("tokens"); err != nil {
		return quota.Result{}, err
	}
	return e.mem.Tokens.Reserve(ctx, key, lim, n, now)
}

func (e *errLimiters) Refund(ctx context.Context, key string, lim quota.RateLimit, n int64, now time.Time) error {
	if err := e.fail("refund"); err != nil {
		return err
	}
	return e.mem.Tokens.Refund(ctx, key, lim, n, now)
}

func (e *errLimiters) Acquire(ctx context.Context, key, lease string, limit int, ttl time.Duration, now time.Time) (bool, error) {
	if err := e.block(ctx, "concurrency"); err != nil {
		return false, err
	}
	if err := e.fail("concurrency"); err != nil {
		return false, err
	}
	return e.mem.Concurrency.Acquire(ctx, key, lease, limit, ttl, now)
}

func (e *errLimiters) Release(ctx context.Context, key, lease string) error {
	e.mu.Lock()
	e.released = append(e.released, lease)
	e.mu.Unlock()
	if err := e.fail("release"); err != nil {
		return err
	}
	return e.mem.Concurrency.Release(ctx, key, lease)
}

func TestLimiterErrorIs503(t *testing.T) {
	for _, which := range []string{"concurrency", "requests", "tokens"} {
		t.Run(which, func(t *testing.T) {
			stub := &errLimiters{mem: quota.NewMemory(), err: errors.New("limiter backend down"), which: which}
			called := false
			p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
				called = true
				return &openai.ChatCompletion{}, nil
			}}
			gw, logs := newAdmissionGateway(t, func(cfg *config.Config) {
				tn := tenantA(cfg)
				tn.MaxConcurrency, tn.RPM, tn.RPMBurst, tn.TPM, tn.TPMBurst = 1, 60, 2, 1000, 1000
			}, p, stub.limiters())

			rec := chat(gw, keyA, chatBody("model-a", "hi"))

			if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
				t.Fatalf("status %d body %s, want 503 overloaded", rec.Code, rec.Body.String())
			}
			if called {
				t.Error("the upstream was called after a limiter error")
			}
			out := logs.String()
			if !strings.Contains(out, "level=WARN") || !strings.Contains(out, "limiter backend down") {
				t.Errorf("limiter error not logged at warn with its cause:\n%s", out)
			}
			for _, secret := range []string{keyA, keyHash(keyA)} {
				if strings.Contains(out, secret) {
					t.Errorf("log leaks a credential:\n%s", out)
				}
			}
			// Whatever an earlier check took was handed back.
			ok, err := stub.mem.Concurrency.Acquire(context.Background(), "tenant-a", "probe", 1, time.Minute, admissionNow)
			if err != nil || !ok {
				t.Errorf("Acquire after the failure = %v, %v; want the slot free", ok, err)
			}
		})
	}
}

// TestLimiterCallsAreTimeBounded: a limiter that never answers must not hold
// the request for its whole deadline; each call is cut off after 250 ms and
// answered as a limiter error.
func TestLimiterCallsAreTimeBounded(t *testing.T) {
	for _, which := range []string{"concurrency", "requests", "tokens"} {
		t.Run(which, func(t *testing.T) {
			stub := &errLimiters{mem: quota.NewMemory(), hang: which}
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				tn := tenantA(cfg)
				tn.MaxConcurrency, tn.RPM, tn.RPMBurst, tn.TPM, tn.TPMBurst = 1, 60, 2, 1000, 1000
			}, okProvider(), stub.limiters())

			start := time.Now()
			rec := chat(gw, keyA, chatBody("model-a", "hi"))

			if elapsed := time.Since(start); elapsed > time.Second {
				t.Errorf("took %v, want a response within 1s", elapsed)
			}
			if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
				t.Fatalf("status %d body %s, want 503 overloaded", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestConcurrencyAcquireErrorReleasesLease: a reply timeout can follow a script
// that already took the lease, so the failed Acquire is followed by a Release
// of the request's lease (absent leases are a no-op in both backends).
func TestConcurrencyAcquireErrorReleasesLease(t *testing.T) {
	stub := &errLimiters{mem: quota.NewMemory(), err: errors.New("reply timeout"), which: "concurrency"}
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).MaxConcurrency = 1 }, okProvider(), stub.limiters())

	rec := chat(gw, keyA, chatBody("model-a", "hi"))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503: %s", rec.Code, rec.Body.String())
	}
	id := rec.Header().Get(requestIDHeader)
	if got := stub.releasedLeases(); len(got) != 1 || got[0] != id || id == "" {
		t.Errorf("Release leases = %v, want exactly the request ID %q", got, id)
	}
}

// TestFinishFailuresAreLoggedAtWarn: a refund or release that fails after the
// answer was sent does not change the answer, is logged at warn with its
// cause, and never logs the API key or its hash.
func TestFinishFailuresAreLoggedAtWarn(t *testing.T) {
	for _, which := range []string{"refund", "release"} {
		t.Run(which, func(t *testing.T) {
			stub := &errLimiters{mem: quota.NewMemory(), err: errors.New(which + " backend down"), which: which}
			// Small usage so the refund path runs.
			p := completionWithUsage(func() *openai.Usage { return &openai.Usage{PromptTokens: 1, CompletionTokens: 1} })
			gw, logs := newAdmissionGateway(t, func(cfg *config.Config) {
				tn := tenantA(cfg)
				tn.MaxConcurrency, tn.TPM, tn.TPMBurst = 1, 1000, 1000
			}, p, stub.limiters())

			rec := chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, 100), false))

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d, want 200 (cleanup failures do not change the answer): %s", rec.Code, rec.Body.String())
			}
			out := logs.String()
			if !strings.Contains(out, "level=WARN") || !strings.Contains(out, which+" backend down") {
				t.Errorf("%s failure not logged at warn with its cause:\n%s", which, out)
			}
			for _, secret := range []string{keyA, keyHash(keyA)} {
				if strings.Contains(out, secret) {
					t.Errorf("log leaks a credential:\n%s", out)
				}
			}
		})
	}
}

func TestUnlimitedTenantSkipsLimiters(t *testing.T) {
	stub := &errLimiters{mem: quota.NewMemory(), err: errors.New("must not be called"), which: "none"}
	gw, _ := newAdmissionGateway(t, nil, okProvider(), stub.limiters())

	if rec := chat(gw, keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if n := stub.calls.Load(); n != 0 {
		t.Errorf("limiters were called %d times for a tenant with no limits, want 0", n)
	}
}

func TestRedisBackendEndToEnd(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })

	newServer := func() *Server {
		gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
			tenantA(cfg).RPM, tenantA(cfg).RPMBurst = 60, 2
		}, okProvider(), quota.NewRedis(client, "routepilot"))
		return gw
	}
	one, two := newServer(), newServer()
	body := chatBody("model-a", "hi")

	if rec := chat(one, keyA, body); rec.Code != http.StatusOK {
		t.Fatalf("server one: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if rec := chat(two, keyA, body); rec.Code != http.StatusOK {
		t.Fatalf("server two: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	// The burst of 2 is spent across both servers.
	wantRateLimited(t, chat(one, keyA, body), "tenant request rate exceeded", "1")
	wantRateLimited(t, chat(two, keyA, body), "tenant request rate exceeded", "1")
}

// TestStreamFailedAttemptUsageIsNotRefunded: a usage-only chunk from the only
// attempt, which then ends without content, must not be treated as the usage
// of an answer. No attempt produced a response and the usage is unknown, so
// the reservation stands (M2-R18).
func TestStreamFailedAttemptUsageIsNotRefunded(t *testing.T) {
	released := make(chan struct{})
	close(released) // the stream ends cleanly right after its chunks
	p := &fakeProvider{
		complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			return &openai.ChatCompletion{}, nil
		},
		stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
			usageOnly := &openai.ChatCompletionChunk{Usage: &openai.Usage{PromptTokens: 10, TotalTokens: 10}}
			return newGatedStream(released, usageOnly), nil
		},
	}
	lim := quota.NewMemory()
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 1000
	}, p, lim)

	rec := chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, 1000), true))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	res, err := lim.Tokens.Reserve(context.Background(), "tenant-a", quota.RateLimit{PerMinute: 1000, Burst: 1000}, 1, admissionNow)
	if err != nil || res.Allowed {
		t.Errorf("Reserve(1) = %+v, %v; want denied because no refund happens for a failed attempt", res, err)
	}
}
