package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
	"github.com/omorsi45/routepilot/internal/reliability"
	"github.com/omorsi45/routepilot/internal/router"
)

// depSpec is one deployment: its unique name and the upstream model string,
// which carries the mock directives and identifies the deployment upstream.
type depSpec struct{ name, upstream string }

// modelSpec is one catalog model of an attempt test.
type modelSpec struct {
	id      string
	in, out string
	rank    int
	maxOut  int      // max_output_tokens; 0 keeps the config default
	caps    []string // nil keeps the streaming-only default
	deps    []depSpec
}

type attemptOpts struct {
	models      []modelSpec
	policy      string // routing policy; empty disables auto
	maxAttempts int
	upstream    http.Handler // nil: the mock handler
	clock       func() time.Time
	realSleep   bool
	// sleep replaces the recording fake Sleep when set.
	sleep func(ctx context.Context, d time.Duration) error
	// defaultTimeoutMs overrides server.default_timeout_ms when positive.
	defaultTimeoutMs int
	// ttftMs overrides the provider's ttft_timeout_ms when positive.
	ttftMs int
	// tenantMaxTimeoutMs sets the tenant's max_timeout_ms when positive.
	tenantMaxTimeoutMs int
}

// attemptEnv is a gateway over one provider whose upstream counts requests by
// upstream model string, with a fake Sleep and a fixed Rand.
type attemptEnv struct {
	gw     *Server
	logs   *strings.Builder
	mu     sync.Mutex
	hits   map[string]int
	sleeps []time.Duration
}

func (e *attemptEnv) count(upstream string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.hits[upstream]
}

func (e *attemptEnv) total() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := 0
	for _, c := range e.hits {
		n += c
	}
	return n
}

func (e *attemptEnv) slept() []time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]time.Duration(nil), e.sleeps...)
}

func newAttemptEnv(t *testing.T, o attemptOpts) *attemptEnv {
	t.Helper()
	e := &attemptEnv{hits: map[string]int{}, logs: &strings.Builder{}}
	next := o.upstream
	if next == nil {
		next = mock.NewHandler(mock.Config{})
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		var body struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(b, &body)
		e.mu.Lock()
		e.hits[body.Model]++
		e.mu.Unlock()
		next.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	var sb strings.Builder
	fmt.Fprintf(&sb, "providers:\n  - name: mockp\n    type: openaicompat\n    base_url: %q\n", srv.URL+"/v1")
	if o.ttftMs > 0 {
		fmt.Fprintf(&sb, "    ttft_timeout_ms: %d\n", o.ttftMs)
	}
	if o.defaultTimeoutMs > 0 {
		fmt.Fprintf(&sb, "server:\n  default_timeout_ms: %d\n", o.defaultTimeoutMs)
	}
	if o.maxAttempts > 0 {
		fmt.Fprintf(&sb, "reliability:\n  max_attempts: %d\n", o.maxAttempts)
	}
	if o.policy != "" {
		fmt.Fprintf(&sb, "routing:\n  policy: %s\n", o.policy)
	}
	sb.WriteString("models:\n")
	for _, m := range o.models {
		fmt.Fprintf(&sb, "  - id: %s\n    rank: %d\n", m.id, m.rank)
		if m.in != "" {
			fmt.Fprintf(&sb, "    input_price_per_mtok: %q\n    output_price_per_mtok: %q\n", m.in, m.out)
		}
		if m.maxOut > 0 {
			fmt.Fprintf(&sb, "    max_output_tokens: %d\n", m.maxOut)
		}
		if m.caps != nil {
			fmt.Fprintf(&sb, "    capabilities: [%s]\n", strings.Join(m.caps, ", "))
		}
		sb.WriteString("    deployments:\n")
		for _, d := range m.deps {
			fmt.Fprintf(&sb, "      - name: %s\n        provider: mockp\n        upstream_model: %q\n", d.name, d.upstream)
		}
	}
	fmt.Fprintf(&sb, "tenants:\n  - id: tenant-a\n    api_key_sha256: %s\n    allowed_models: [\"*\"]\n", keyHash(keyA))
	if o.tenantMaxTimeoutMs > 0 {
		fmt.Fprintf(&sb, "    max_timeout_ms: %d\n", o.tenantMaxTimeoutMs)
	}
	cfg, err := config.Parse([]byte(sb.String()))
	if err != nil {
		t.Fatalf("parse config: %v\n%s", err, sb.String())
	}

	opts := Options{
		Logger: slog.New(slog.NewTextHandler(e.logs, nil)),
		Now:    o.clock,
		Rand:   func() float64 { return 1 },
	}
	if o.sleep != nil {
		opts.Sleep = o.sleep
	} else if !o.realSleep {
		opts.Sleep = func(_ context.Context, d time.Duration) error {
			e.mu.Lock()
			e.sleeps = append(e.sleeps, d)
			e.mu.Unlock()
			return nil
		}
	}
	prov := openaicompat.New(cfg.Providers[0], "", srv.Client())
	e.gw, err = New(cfg, map[string]provider.Provider{"mockp": prov}, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func (e *attemptEnv) post(body string) *httptest.ResponseRecorder {
	return do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
}

func modelOf(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var resp openai.ChatCompletion
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	return resp.Model
}

func TestFailoverToSecondDeployment(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub"}}},
	}})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	if got := rec.Header().Get("x-routepilot-model"); got != "m" {
		t.Errorf("model header %q, want m", got)
	}
	if a, b := e.count("ua;fail=http500"), e.count("ub"); a != 1 || b != 1 {
		t.Errorf("upstream hits a=%d b=%d, want 1 each", a, b)
	}
	logs := e.logs.String()
	for _, want := range []string{"attempts=2", "policy=pinned", "deployment=dep-b"} {
		if !strings.Contains(logs, want) {
			t.Errorf("log line missing %q: %s", want, logs)
		}
	}
}

func TestFailoverToNextModel(t *testing.T) {
	ok := mock.NewHandler(mock.Config{})
	up503 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if strings.Contains(string(b), `"model":"uc"`) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"overloaded","type":"server_error","code":"overloaded"}}`)
			return
		}
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		ok.ServeHTTP(w, r)
	})
	e := newAttemptEnv(t, attemptOpts{
		policy:   config.PolicyCheapest,
		upstream: up503,
		models: []modelSpec{
			{id: "cheap", in: "0.1", out: "0.1", deps: []depSpec{{"cheap-a", "uc"}}},
			{id: "pricey", in: "5", out: "5", deps: []depSpec{{"pricey-a", "up"}}},
		},
	})
	rec := e.post(chatBody("auto", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := modelOf(t, rec); got != "pricey" {
		t.Errorf("body model %q, want pricey", got)
	}
	if got := rec.Header().Get("x-routepilot-model"); got != "pricey" {
		t.Errorf("model header %q, want pricey", got)
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "pricey-a" {
		t.Errorf("deployment header %q, want pricey-a", got)
	}
	if !strings.Contains(e.logs.String(), "policy=cheapest") {
		t.Errorf("log line missing policy=cheapest: %s", e.logs.String())
	}
}

func TestNonRetryable4xxReturnsImmediately(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http400"}, {"dep-b", "ub"}}},
	}})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
	}
	if n := e.count("ub"); n != 0 {
		t.Errorf("second deployment saw %d requests, want 0", n)
	}
	// A rejected request is not a deployment fault: it counts as a success.
	if h := e.gw.health.Snapshot()["dep-a"]; h.State != reliability.Closed || h.ErrorRate != 0 {
		t.Errorf("dep-a health = %+v, want closed with error rate 0", h)
	}
}

func TestMaxAttemptsCap(t *testing.T) {
	fail := func(n string) depSpec { return depSpec{"dep-" + n, "u" + n + ";fail=http500"} }
	e := newAttemptEnv(t, attemptOpts{
		maxAttempts: 3,
		models:      []modelSpec{{id: "m", deps: []depSpec{fail("1"), fail("2"), fail("3"), fail("4")}}},
	})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	if n := e.total(); n != 3 {
		t.Errorf("upstream requests = %d, want exactly 3", n)
	}
	if !strings.Contains(e.logs.String(), "attempts=3") {
		t.Errorf("log line missing attempts=3: %s", e.logs.String())
	}
}

func TestRetryAfterShortIsWaited(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http429;retry_after=1"}, {"dep-b", "ub"}}},
	}})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := e.slept(); len(got) != 1 || got[0] != time.Second {
		t.Errorf("sleeps = %v, want [1s]", got)
	}
}

func TestRetryAfterLongTripsBreaker(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http429;retry_after=30"}, {"dep-b", "ub"}}},
	}})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	for _, d := range e.slept() {
		if d >= 2*time.Second {
			t.Errorf("slept %v for a 30s Retry-After, want the breaker trip instead", d)
		}
	}
	if e.gw.health.Available("dep-a") {
		t.Error("dep-a is still available after a 30s Retry-After")
	}
	rec = e.post(chatBody("m", "hi again"))
	if rec.Code != http.StatusOK || rec.Header().Get("x-routepilot-deployment") != "dep-b" {
		t.Fatalf("second request: status %d deployment %q", rec.Code, rec.Header().Get("x-routepilot-deployment"))
	}
	if n := e.count("ua;fail=http429;retry_after=30"); n != 1 {
		t.Errorf("dep-a saw %d requests, want 1 (skipped on the second request)", n)
	}
}

func TestRetryAfterQuarantineIsCapped(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	advance := func(d time.Duration) { mu.Lock(); now = now.Add(d); mu.Unlock() }
	e := newAttemptEnv(t, attemptOpts{
		clock: clock,
		models: []modelSpec{
			{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http429;retry_after=86400"}, {"dep-b", "ub"}}},
		},
	})
	if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	advance(reliability.MaxOpenFor - time.Second)
	if e.gw.health.Available("dep-a") {
		t.Error("dep-a is available at 119s, want still quarantined")
	}
	advance(time.Second)
	if !e.gw.health.Available("dep-a") {
		t.Error("dep-a is unavailable at 120s after a 86400s Retry-After, want the trip capped at MaxOpenFor")
	}
}

// The blame budget is the deadline a request gets without a client timeout_ms,
// so a tenant whose max_timeout_ms is below the server default still feeds
// deadline expiries to the breaker (M2-R14c).
func TestDeadlineBlameUsesTenantEffectiveDefault(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		defaultTimeoutMs:   400,
		tenantMaxTimeoutMs: 300,
		models:             []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=stall"}, {"dep-b", "ub"}}}},
	})
	for i := range reliability.ConsecutiveFailures {
		rec := e.post(chatBody("m", "hi"))
		if rec.Code != http.StatusGatewayTimeout {
			t.Fatalf("request %d: status %d, want 504: %s", i, rec.Code, rec.Body.String())
		}
	}
	if e.gw.health.Available("dep-a") {
		t.Fatal("dep-a is still available after 5 deadline expiries, want its breaker open")
	}
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusOK || rec.Header().Get("x-routepilot-deployment") != "dep-b" {
		t.Fatalf("request after the trip: status %d deployment %q: %s", rec.Code, rec.Header().Get("x-routepilot-deployment"), rec.Body.String())
	}
}

func TestUpstream401Quarantines(t *testing.T) {
	ok := mock.NewHandler(mock.Config{})
	up401 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if strings.Contains(string(b), `"model":"ua"`) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"message":"bad key","type":"authentication_error","code":"invalid_api_key"}}`)
			return
		}
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		ok.ServeHTTP(w, r)
	})
	e := newAttemptEnv(t, attemptOpts{
		upstream: up401,
		models:   []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}, {"dep-b", "ub"}}}},
	})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusOK || rec.Header().Get("x-routepilot-deployment") != "dep-b" {
		t.Fatalf("first request: status %d deployment %q: %s", rec.Code, rec.Header().Get("x-routepilot-deployment"), rec.Body.String())
	}
	if e.gw.health.Available("dep-a") {
		t.Error("dep-a is available after an upstream 401, want quarantined")
	}
	rec = e.post(chatBody("m", "hi again"))
	if rec.Code != http.StatusOK {
		t.Fatalf("second request: status %d: %s", rec.Code, rec.Body.String())
	}
	if n := e.count("ua"); n != 1 {
		t.Errorf("dep-a saw %d requests, want 1", n)
	}
}

func TestAllDeploymentsOpenIs503(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua"}, {"dep-b", "ub"}}},
	}})
	e.gw.health.Trip("dep-a", time.Minute)
	e.gw.health.Trip("dep-b", time.Minute)
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "no_healthy_deployment" {
		t.Errorf("code %q, want no_healthy_deployment", got)
	}
	if n := e.total(); n != 0 {
		t.Errorf("upstream saw %d requests, want 0", n)
	}
}

func TestBackoffNeverCrossesDeadline(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		realSleep: true,
		models: []modelSpec{
			{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub;fail=http500"}}},
		},
	})
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":50}}`
	start := time.Now()
	rec := e.post(body)
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond {
		t.Errorf("took %v, want under 200ms", elapsed)
	}
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "upstream_timeout" {
		t.Errorf("code %q, want upstream_timeout", got)
	}
	if n := e.total(); n != 1 {
		t.Errorf("upstream requests = %d, want 1 (the backoff would pass the deadline)", n)
	}
}

func TestBreakerOpensAfterConsecutiveFailures(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}}},
	}})
	for i := 0; i < reliability.ConsecutiveFailures; i++ {
		rec := e.post(chatBody("m", "hi"))
		if rec.Code != http.StatusBadGateway {
			t.Fatalf("request %d: status %d, want 502", i+1, rec.Code)
		}
	}
	before := e.total()
	rec := e.post(chatBody("m", "hi"))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "no_healthy_deployment" {
		t.Errorf("code %q, want no_healthy_deployment", got)
	}
	if after := e.total(); after != before {
		t.Errorf("open breaker still sent %d upstream requests", after-before)
	}
}

func TestCapabilityRoutingEndToEnd(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		policy: config.PolicyCheapest,
		models: []modelSpec{
			{id: "plain", in: "0.1", out: "0.1", deps: []depSpec{{"plain-a", "uplain"}}},
			{id: "tooled", in: "5", out: "5", caps: []string{"streaming", "tools"}, deps: []depSpec{{"tooled-a", "utooled"}}},
		},
	})
	body := `{"model":"auto","messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	rec := e.post(body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := modelOf(t, rec); got != "tooled" {
		t.Errorf("body model %q, want tooled", got)
	}
	if n := e.count("uplain"); n != 0 {
		t.Errorf("the model without tools saw %d requests, want 0", n)
	}
}

func TestClientDisconnectReleasesProbe(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	e := newAttemptEnv(t, attemptOpts{
		clock:  clock,
		models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=stall"}}}},
	})
	e.gw.health.Trip("dep-a", time.Second)
	mu.Lock()
	now = now.Add(2 * time.Second) // the cooldown elapsed: the next attempt is the probe
	mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("m", "hi"))).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyA)
	done := make(chan struct{})
	go func() {
		e.gw.Handler().ServeHTTP(httptest.NewRecorder(), req)
		close(done)
	}()
	waitFor(t, 2*time.Second, "the probe to reach the upstream", func() bool { return e.total() == 1 })
	if e.gw.health.Available("dep-a") {
		t.Fatal("dep-a admits a second attempt while the probe is in flight")
	}
	cancel()
	<-done
	if !e.gw.health.Available("dep-a") {
		t.Error("a disconnected client left the half-open probe claimed")
	}
}

func TestNewRejectsNonPositiveProviderTimeouts(t *testing.T) {
	for name, mutate := range map[string]func(*config.ProviderConfig){
		"ttft": func(p *config.ProviderConfig) { p.TTFTTimeoutMs = 0 },
		"idle": func(p *config.ProviderConfig) { p.IdleTimeoutMs = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			cfg := buildConfig(t, cfgOpts{})
			mutate(&cfg.Providers[0])
			_, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{})
			if err == nil {
				t.Fatal("New succeeded, want error")
			}
			if !strings.Contains(err.Error(), "mockp") {
				t.Errorf("error %q should name the provider", err)
			}
		})
	}
}

type retryableCase struct {
	name string
	err  error
	ctx  context.Context
	want bool
}

func TestRetryable(t *testing.T) {
	httpErr := func(status int) error { return &provider.Error{Kind: provider.KindHTTP, Status: status} }
	live := context.Background()
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()
	ttft, cancelTTFT := context.WithCancelCause(context.Background())
	cancelTTFT(errTTFTTimeout)
	idle, cancelIdle := context.WithCancelCause(context.Background())
	cancelIdle(errIdleTimeout)
	gone, cancelGone := context.WithCancelCause(context.Background())
	cancelGone(context.Canceled)

	tests := []retryableCase{
		{"nil", nil, live, false},
		{"plain error", errors.New("boom"), live, false},
		{"translation error", openai.ErrInvalidRequest("x", ""), live, false},
		{"wrapped translation error", fmt.Errorf("wrap: %w", openai.ErrInternal()), live, false},
		{"connect", &provider.Error{Kind: provider.KindConnect}, live, true},
		{"protocol", &provider.Error{Kind: provider.KindProtocol}, live, true},
		{"timeout while time remains", &provider.Error{Kind: provider.KindTimeout}, live, true},
		{"timeout after the deadline", &provider.Error{Kind: provider.KindTimeout}, expired, false},
		{"canceled by the TTFT watchdog", &provider.Error{Kind: provider.KindCanceled}, ttft, true},
		{"canceled by the idle watchdog", &provider.Error{Kind: provider.KindCanceled}, idle, false},
		{"client disconnect", &provider.Error{Kind: provider.KindCanceled}, gone, false},
		{"wrapped provider error", fmt.Errorf("wrap: %w", httpErr(503)), live, true},
		// The status list decides, not provider.Error.Retryable.
		{"529 without the Retryable flag", &provider.Error{Kind: provider.KindHTTP, Status: 529}, live, true},
		{"501 with the Retryable flag", &provider.Error{Kind: provider.KindHTTP, Status: 501, Retryable: true}, live, false},
	}
	for _, status := range []int{408, 409, 429, 500, 502, 503, 504, 529, 401, 403} {
		tests = append(tests, retryableCase{fmt.Sprintf("http %d", status), httpErr(status), live, true})
	}
	for _, status := range []int{400, 404, 413, 422, 501, 505} {
		tests = append(tests, retryableCase{fmt.Sprintf("http %d", status), httpErr(status), live, false})
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := retryable(tc.err, tc.ctx); got != tc.want {
				t.Errorf("retryable = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAttemptPlan(t *testing.T) {
	cat, err := catalog.New(buildConfig(t, cfgOpts{}))
	if err != nil {
		t.Fatalf("catalog.New: %v", err)
	}
	ma, _ := cat.Model("model-a")
	mb, _ := cat.Model("model-b")
	twoDeps := func(m *catalog.Model) router.Candidate {
		deps := []catalog.Deployment{m.Deployments[0], {Name: m.ID + "-2", Provider: "mockp", UpstreamModel: "x"}}
		return router.Candidate{Model: m, Deployments: deps}
	}
	dec := router.Decision{Candidates: []router.Candidate{twoDeps(ma), twoDeps(mb)}}

	walk := func(plan *attemptPlan) string {
		var got []string
		for {
			m, d, _, ok := plan.next()
			if !ok {
				return strings.Join(got, ",")
			}
			got = append(got, m.ID+"/"+d.Name)
		}
	}
	newServer := func(max int) *Server {
		cfg := buildConfig(t, cfgOpts{})
		cfg.Reliability.MaxAttempts = max
		s, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return s
	}

	t.Run("walks models then deployments in order", func(t *testing.T) {
		s := newServer(5)
		want := "model-a/dep-a,model-a/model-a-2,model-b/dep-b,model-b/model-b-2"
		if got := walk(s.newAttemptPlan(dec)); got != want {
			t.Errorf("order = %s, want %s", got, want)
		}
	})
	t.Run("stops at max attempts", func(t *testing.T) {
		s := newServer(3)
		want := "model-a/dep-a,model-a/model-a-2,model-b/dep-b"
		if got := walk(s.newAttemptPlan(dec)); got != want {
			t.Errorf("pairs = %s, want %s", got, want)
		}
	})
	t.Run("skipped pairs do not count as attempts", func(t *testing.T) {
		s := newServer(2)
		s.health.Trip("dep-a", time.Minute)
		s.health.Trip("model-a-2", time.Minute)
		want := "model-b/dep-b,model-b/model-b-2"
		if got := walk(s.newAttemptPlan(dec)); got != want {
			t.Errorf("pairs = %s, want %s", got, want)
		}
	})
	t.Run("more looks ahead without claiming", func(t *testing.T) {
		s := newServer(5)
		plan := s.newAttemptPlan(router.Decision{Candidates: []router.Candidate{twoDeps(ma)}})
		if !plan.more() {
			t.Fatal("more = false on a fresh plan")
		}
		plan.next()
		plan.next()
		if plan.more() {
			t.Error("more = true after every pair was issued")
		}
	})
}

func TestWaitBeforeRetry(t *testing.T) {
	newServer := func() (*Server, *[]time.Duration) {
		var slept []time.Duration
		s, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{
			Rand: func() float64 { return 1 },
			Sleep: func(_ context.Context, d time.Duration) error {
				slept = append(slept, d)
				return nil
			},
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return s, &slept
	}
	ctx := context.Background()
	rateLimited := func(ra time.Duration) error {
		return &provider.Error{Kind: provider.KindHTTP, Status: 429, RetryAfter: ra}
	}

	t.Run("short Retry-After is waited", func(t *testing.T) {
		s, slept := newServer()
		err := s.waitBeforeRetry(ctx, 0, rateLimited(2*time.Second))
		if err != nil || len(*slept) != 1 || (*slept)[0] != 2*time.Second {
			t.Errorf("err=%v slept=%v, want one 2s sleep", err, *slept)
		}
	})
	t.Run("long Retry-After does not sleep", func(t *testing.T) {
		s, slept := newServer()
		err := s.waitBeforeRetry(ctx, 0, rateLimited(3*time.Second))
		if err != nil || len(*slept) != 0 {
			t.Errorf("err=%v slept=%v, want no sleep", err, *slept)
		}
	})
	t.Run("a zero wait after the deadline fails", func(t *testing.T) {
		s, slept := newServer()
		dctx, cancel := context.WithDeadline(ctx, time.Now().Add(-time.Second))
		defer cancel()
		err := s.waitBeforeRetry(dctx, 0, rateLimited(3*time.Second))
		if !errors.Is(err, context.DeadlineExceeded) || len(*slept) != 0 {
			t.Errorf("err=%v slept=%v, want DeadlineExceeded and no sleep", err, *slept)
		}
	})
	t.Run("no Retry-After uses backoff", func(t *testing.T) {
		s, slept := newServer()
		err := s.waitBeforeRetry(ctx, 1, errors.New("boom"))
		want := reliability.Backoff(1, func() float64 { return 1 })
		if err != nil || len(*slept) != 1 || (*slept)[0] != want {
			t.Errorf("err=%v slept=%v, want one %v sleep", err, *slept, want)
		}
	})
	t.Run("a wait past the deadline fails without sleeping", func(t *testing.T) {
		s, slept := newServer()
		dctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
		defer cancel()
		err := s.waitBeforeRetry(dctx, 0, rateLimited(time.Second))
		if !errors.Is(err, context.DeadlineExceeded) || len(*slept) != 0 {
			t.Errorf("err=%v slept=%v, want DeadlineExceeded and no sleep", err, *slept)
		}
	})
}

func streamBody(model string) string {
	return fmt.Sprintf(`{"model":%q,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, model)
}

func TestStreamOutcomesFeedHealth(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
		{id: "good", deps: []depSpec{{"dep-good", "ug;tokens=2"}}},
		{id: "bad", deps: []depSpec{{"dep-bad", "ub;fail=http500"}}},
	}})

	rec := e.post(streamBody("good"))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("good stream: status %d: %s", rec.Code, rec.Body.String())
	}
	if h, seen := e.gw.health.Snapshot()["dep-good"]; !seen {
		t.Error("a completed stream did not report an outcome for dep-good")
	} else if h.State != reliability.Closed || h.ErrorRate != 0 {
		t.Errorf("dep-good health = %+v, want closed with error rate 0", h)
	}

	for i := 0; i < reliability.ConsecutiveFailures; i++ {
		if rec := e.post(streamBody("bad")); rec.Code != http.StatusBadGateway {
			t.Fatalf("bad stream %d: status %d, want 502: %s", i+1, rec.Code, rec.Body.String())
		}
	}
	before := e.count("ub;fail=http500")
	rec = e.post(streamBody("bad"))
	if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "no_healthy_deployment" {
		t.Fatalf("status %d, want 503 no_healthy_deployment: %s", rec.Code, rec.Body.String())
	}
	if after := e.count("ub;fail=http500"); after != before {
		t.Errorf("open breaker still sent %d stream requests upstream", after-before)
	}
	if !strings.Contains(e.logs.String(), "attempts=1") {
		t.Errorf("log line missing attempts=1: %s", e.logs.String())
	}
}

func TestLongRetryAfterTripsWithoutAFollowUpAttempt(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	const upstream = "ua;fail=http429;retry_after=30"
	e := newAttemptEnv(t, attemptOpts{
		clock:  clock,
		models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", upstream}}}},
	})
	rec := e.post(chatBody("m", "hi"))
	if rec.Code < 400 {
		t.Fatalf("first request: status %d, want an error: %s", rec.Code, rec.Body.String())
	}
	if e.gw.health.Available("dep-a") {
		t.Error("dep-a is available after a 30s Retry-After on the only deployment")
	}
	rec = e.post(chatBody("m", "hi again"))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("second request: status %d, want 503: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "no_healthy_deployment" {
		t.Errorf("code %q, want no_healthy_deployment", got)
	}
	if n := e.count(upstream); n != 1 {
		t.Errorf("dep-a saw %d requests, want 1 (none for the second request)", n)
	}
}

func TestShortClientDeadlineDoesNotFeedBreaker(t *testing.T) {
	for name, body := range map[string]string{
		"complete": `{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":50}}`,
		"stream":   `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":50}}`,
	} {
		t.Run(name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
				{id: "m", deps: []depSpec{{"dep-a", "ua;fail=stall"}}},
			}})
			for i := 0; i < reliability.ConsecutiveFailures; i++ {
				if rec := e.post(body); rec.Code != http.StatusGatewayTimeout {
					t.Fatalf("request %d: status %d, want 504: %s", i+1, rec.Code, rec.Body.String())
				}
			}
			if !e.gw.health.Available("dep-a") {
				t.Error("dep-a is unavailable after short client deadlines, want the breaker closed")
			}
			if h := e.gw.health.Snapshot()["dep-a"]; h.State != reliability.Closed || h.ErrorRate != 0 {
				t.Errorf("dep-a health = %+v, want closed with error rate 0", h)
			}
		})
	}
}

func TestServerDefaultDeadlineFeedsBreaker(t *testing.T) {
	for name, body := range map[string]string{
		"complete": chatBody("m", "hi"),
		"stream":   streamBody("m"),
	} {
		t.Run(name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{
				defaultTimeoutMs: 100,
				models:           []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=stall"}}}},
			})
			for i := 0; i < reliability.ConsecutiveFailures; i++ {
				if rec := e.post(body); rec.Code != http.StatusGatewayTimeout {
					t.Fatalf("request %d: status %d, want 504: %s", i+1, rec.Code, rec.Body.String())
				}
			}
			if e.gw.health.Available("dep-a") {
				t.Error("dep-a is available after repeated server-default timeouts, want the breaker open")
			}
		})
	}
}

func TestNoRetryClaimedAfterDeadlinePassedDuringWait(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		// The backoff (50ms with Rand=1) fits the 100ms deadline, but the
		// sleep itself overruns it.
		sleep: func(ctx context.Context, _ time.Duration) error {
			<-ctx.Done()
			return nil
		},
		models: []modelSpec{
			{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub"}}},
		},
	})
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":100}}`
	rec := e.post(body)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
	}
	if n := e.count("ub"); n != 0 {
		t.Errorf("second deployment saw %d requests, want 0", n)
	}
	if _, claimed := e.gw.health.Snapshot()["dep-b"]; claimed {
		t.Error("dep-b was claimed after the deadline passed")
	}
	if !strings.Contains(e.logs.String(), "attempts=1") {
		t.Errorf("log line missing attempts=1: %s", e.logs.String())
	}
}

// expiredCtx returns a context whose deadline has passed.
func expiredCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Millisecond))
	t.Cleanup(cancel)
	return ctx
}

func TestSettleDeadlineBlameThreshold(t *testing.T) {
	timeout := &provider.Error{Kind: provider.KindTimeout, Message: "deadline"}
	for name, tc := range map[string]struct {
		elapsed time.Duration
		blamed  bool
	}{
		"just under 90%":  {899 * time.Millisecond, false},
		"exactly 90%":     {900 * time.Millisecond, true},
		"the full budget": {time.Second, true},
		"far shorter":     {50 * time.Millisecond, false},
	} {
		t.Run(name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{
				defaultTimeoutMs: 1000,
				models:           []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
			})
			if !e.gw.health.Allow("dep-a") {
				t.Fatal("Allow refused a fresh deployment")
			}
			e.gw.settle(expiredCtx(t), &reqState{blameBudget: time.Second}, "dep-a", timeout, tc.elapsed)
			// A recorded failure sets the error rate; an abort leaves it 0.
			if blamed := e.gw.health.Snapshot()["dep-a"].ErrorRate > 0; blamed != tc.blamed {
				t.Errorf("failure recorded = %v, want %v", blamed, tc.blamed)
			}
		})
	}
}

// A 401/403 or a long Retry-After that arrives just before the deadline is
// still the deployment's fault: only timeout and cancel failures are excused.
func TestSettleDeadlineBranchOnlyForTimeoutKinds(t *testing.T) {
	for name, err := range map[string]*provider.Error{
		"401":              {Kind: provider.KindHTTP, Status: 401, Message: "bad key"},
		"long retry-after": {Kind: provider.KindHTTP, Status: 429, Retryable: true, RetryAfter: 30 * time.Second},
	} {
		t.Run(name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{
				defaultTimeoutMs: 1000,
				models:           []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
			})
			if !e.gw.health.Allow("dep-a") {
				t.Fatal("Allow refused a fresh deployment")
			}
			e.gw.settle(expiredCtx(t), &reqState{blameBudget: time.Second}, "dep-a", err, 10*time.Millisecond)
			if e.gw.health.Snapshot()["dep-a"].ErrorRate == 0 {
				t.Error("no failure recorded: the claim was released as a short deadline")
			}
			if e.gw.health.Available("dep-a") {
				t.Error("dep-a is still available, want it tripped")
			}
		})
	}
}

// claimFixture is a server and a one-pair plan for driving claimNext and
// failAttempts directly. sleep replaces the server's Sleep when set.
func claimFixture(t *testing.T, sleep func(ctx context.Context, d time.Duration) error) (*Server, *attemptPlan, *reqState) {
	t.Helper()
	cfg := buildConfig(t, cfgOpts{})
	s, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{
		Rand:  func() float64 { return 1 },
		Sleep: sleep,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ma, _ := s.cat.Model("model-a")
	plan := s.newAttemptPlan(router.Decision{Candidates: []router.Candidate{{Model: ma, Deployments: ma.Deployments}}})
	return s, plan, &reqState{attempts: 1}
}

func TestClaimNextStopsOnContextError(t *testing.T) {
	req := &openai.ChatCompletionRequest{}
	// A Retry-After over the inline limit gives a zero wait.
	zeroWait := &provider.Error{Kind: provider.KindHTTP, Status: 429, RetryAfter: 3 * time.Second}
	noSleep := func(context.Context, time.Duration) error { return nil }

	check := func(t *testing.T, s *Server, rs *reqState, outcome, want claimOutcome, wantCanceled bool) {
		t.Helper()
		if outcome != want {
			t.Errorf("outcome = %d, want %d", outcome, want)
		}
		if rs.canceled != wantCanceled {
			t.Errorf("canceled = %v, want %v", rs.canceled, wantCanceled)
		}
		if rs.attempts != 1 {
			t.Errorf("attempts = %d, want 1 (nothing claimed)", rs.attempts)
		}
		if _, claimed := s.health.Snapshot()["dep-a"]; claimed {
			t.Error("dep-a was claimed")
		}
	}

	t.Run("client gone before a zero wait returns", func(t *testing.T) {
		s, plan, rs := claimFixture(t, noSleep)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, _, _, outcome := s.claimNext(ctx, rs, plan, req, zeroWait)
		check(t, s, rs, outcome, claimClientGone, true)
	})
	t.Run("client gone during the sleep", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		s, plan, rs := claimFixture(t, func(context.Context, time.Duration) error {
			cancel()
			return nil
		})
		_, _, _, outcome := s.claimNext(ctx, rs, plan, req, errors.New("boom"))
		check(t, s, rs, outcome, claimClientGone, true)
	})
	t.Run("deadline passed during the sleep", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		s, plan, rs := claimFixture(t, func(ctx context.Context, _ time.Duration) error {
			<-ctx.Done()
			return nil
		})
		_, _, _, outcome := s.claimNext(ctx, rs, plan, req, errors.New("boom"))
		check(t, s, rs, outcome, claimDeadline, false)
	})
	t.Run("a live context claims the pair", func(t *testing.T) {
		s, plan, rs := claimFixture(t, noSleep)
		_, dep, _, outcome := s.claimNext(context.Background(), rs, plan, req, errors.New("boom"))
		if outcome != claimOK || dep.Name != "dep-a" || rs.attempts != 2 {
			t.Errorf("outcome=%d dep=%q attempts=%d, want claimOK dep-a 2", outcome, dep.Name, rs.attempts)
		}
	})
}

func TestFailAttemptsWritesNothingWhenClientGone(t *testing.T) {
	lastDep := catalog.Deployment{Name: "dep-a"}
	cases := []struct {
		name     string
		attempts int
		err      error
		timedOut bool
	}{
		{"503 branch", 0, nil, false},
		{"504 branch", 1, &provider.Error{Kind: provider.KindTimeout}, true},
		{"mapped error branch", 1, &provider.Error{Kind: provider.KindConnect}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, _, rs := claimFixture(t, nil)
			rs.attempts = tc.attempts
			clientCtx, cancel := context.WithCancel(context.Background())
			cancel()
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(clientCtx)
			rec := httptest.NewRecorder()

			s.failAttempts(context.Background(), rec, r, rs, lastDep, tc.err, tc.timedOut)

			if rec.Body.Len() != 0 || len(rec.Header()) != 0 {
				t.Errorf("wrote to a gone client: header %v body %q", rec.Header(), rec.Body.String())
			}
			if !rs.canceled {
				t.Error("canceled = false, want true")
			}
			if rs.errCode != "" {
				t.Errorf("errCode = %q, want none", rs.errCode)
			}
		})
	}
}
