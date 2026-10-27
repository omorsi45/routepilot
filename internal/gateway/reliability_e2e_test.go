package gateway

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
	"github.com/omorsi45/routepilot/internal/quota"
)

// These tests characterize the guarantees of the failover and quota code:
// every failure that happens before a response is committed is recovered, no
// upstream call follows a commit, and the tenant limits are never exceeded
// under concurrency. They issue many real HTTP requests, so the retry wait is
// a no-op; the timeouts keep their defaults.

const (
	e2eWorkers      = 16 // concurrent clients of the failover tests
	e2eUnaryCount   = 1000
	e2eStreamCount  = 500
	e2eMidstreamCnt = 200
	e2eModel        = "m"
)

// syncBuffer is a log sink that is safe to read while the gateway writes.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// failoverEnv is a gateway over one model with two deployments on separate
// mock handlers: dep-a is configured to fail, dep-b is healthy.
type failoverEnv struct {
	gw     *httptest.Server
	client *http.Client
	logs   *syncBuffer
	a, b   *mock.Handler
}

// newFailoverEnv serves model "m" from dep-a (upstream model aModel, handled
// by a with its fail rates) and dep-b (a healthy mock). The retry wait is a
// no-op and the jitter source is fixed.
func newFailoverEnv(t *testing.T, aCfg mock.Config, aModel string) *failoverEnv {
	t.Helper()
	e := &failoverEnv{
		a:    mock.NewHandler(aCfg),
		b:    mock.NewHandler(mock.Config{Seed: 2}),
		logs: &syncBuffer{},
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))
		// The upstream model names the deployment: "ub" is dep-b's.
		if strings.Contains(string(body), `"model":"ub"`) {
			e.b.ServeHTTP(w, r)
			return
		}
		e.a.ServeHTTP(w, r)
	}))
	t.Cleanup(up.Close)

	yaml := fmt.Sprintf(`providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: %s
    deployments:
      - name: dep-a
        provider: mockp
        upstream_model: %q
      - name: dep-b
        provider: mockp
        upstream_model: "ub"
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
`, up.URL+"/v1", e2eModel, aModel, keyHash(keyA))
	cfg, err := config.Parse([]byte(yaml))
	if err != nil {
		t.Fatalf("parse config: %v\n%s", err, yaml)
	}
	gw, err := New(cfg, map[string]provider.Provider{
		"mockp": openaicompat.New(cfg.Providers[0], "", up.Client()),
	}, Options{
		Logger: slog.New(slog.NewTextHandler(e.logs, nil)),
		Rand:   func() float64 { return 0 },
		Sleep: func(ctx context.Context, _ time.Duration) error {
			return ctx.Err()
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	e.gw = httptest.NewServer(gw.Handler())
	t.Cleanup(e.gw.Close)
	tr := &http.Transport{MaxIdleConnsPerHost: e2eWorkers}
	t.Cleanup(tr.CloseIdleConnections)
	e.client = &http.Client{Transport: tr, Timeout: 30 * time.Second}
	return e
}

// reply is what one client saw.
type reply struct {
	status int
	dep    string // x-routepilot-deployment
	body   string
	err    error
}

func (e *failoverEnv) post(body string) reply {
	req, err := http.NewRequest(http.MethodPost, e.gw.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		return reply{err: err}
	}
	req.Header.Set("Authorization", "Bearer "+keyA)
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return reply{err: err}
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return reply{status: resp.StatusCode, dep: resp.Header.Get("x-routepilot-deployment"), body: string(b), err: err}
}

// runPool calls f for 0..n-1 on a fixed number of workers and returns the
// results in index order.
func runPool(n, workers int, f func(i int) reply) []reply {
	out := make([]reply, n)
	idx := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idx {
				out[i] = f(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		idx <- i
	}
	close(idx)
	wg.Wait()
	return out
}

// logEntry is one request log line.
type logEntry struct {
	dep      string
	status   int
	attempts int
}

var (
	reLogDep      = regexp.MustCompile(`\bdeployment=(\S*)`)
	reLogStatus   = regexp.MustCompile(`\bstatus=(\d+)`)
	reLogAttempts = regexp.MustCompile(`\battempts=(\d+)`)
)

// parseRequestLogs extracts the request lines from the captured gateway log.
func parseRequestLogs(t *testing.T, logs string) []logEntry {
	t.Helper()
	var out []logEntry
	for _, line := range strings.Split(logs, "\n") {
		if !strings.Contains(line, "msg=request ") {
			continue
		}
		var m [][]string
		for _, re := range []*regexp.Regexp{reLogDep, reLogStatus, reLogAttempts} {
			sub := re.FindStringSubmatch(line)
			if sub == nil {
				t.Fatalf("request log line lacks %v: %s", re, line)
			}
			m = append(m, sub)
		}
		status, _ := strconv.Atoi(m[1][1])
		attempts, _ := strconv.Atoi(m[2][1])
		out = append(out, logEntry{dep: m[0][1], status: status, attempts: attempts})
	}
	return out
}

// finish closes the gateway server, which blocks until every handler has
// returned and so every request log line has been written, and returns the
// parsed log.
func (e *failoverEnv) finish(t *testing.T, want int) []logEntry {
	t.Helper()
	e.gw.Close()
	entries := parseRequestLogs(t, e.logs.String())
	if len(entries) != want {
		t.Fatalf("request log has %d lines, want %d", len(entries), want)
	}
	return entries
}

// assertFailoverRecovered checks the guarantee for a model whose first
// deployment fails some requests before any response is committed and whose
// second deployment is healthy: every request succeeds, and every failure
// injected at the first deployment was followed by exactly one attempt at the
// second. The breaker may open on dep-a mid-run, after which requests skip it,
// which is correct and shows up as "skips".
func assertFailoverRecovered(t *testing.T, e *failoverEnv, replies []reply, stream bool) {
	t.Helper()
	n := len(replies)
	entries := e.finish(t, n)

	servedA := 0
	for i, r := range replies {
		switch {
		case r.err != nil:
			t.Errorf("request %d: client error: %v", i, r.err)
		case r.status != http.StatusOK:
			t.Errorf("request %d: status %d, want 200 (an eligible failure must be recovered): %s", i, r.status, r.body)
		case stream && !strings.Contains(r.body, "data: [DONE]"):
			t.Errorf("request %d: stream ended without [DONE]: %s", i, r.body)
		case stream && strings.Contains(r.body, `"error"`):
			t.Errorf("request %d: stream carries an error event: %s", i, r.body)
		}
		if r.dep == "dep-a" {
			servedA++
		}
	}
	if t.Failed() {
		return
	}

	// A mock counts a completion just after it writes the last byte, which
	// can land after the client has the whole reply.
	waitFor(t, 3*time.Second, "dep-a's completed count to reach the number of requests it served", func() bool {
		return e.a.Stats().Completed == int64(servedA)
	})
	a, b := e.a.Stats(), e.b.Stats()

	injected := a.Requests - a.Completed
	recovered, sumAttempts, maxAttempts := int64(0), int64(0), 0
	for _, en := range entries {
		sumAttempts += int64(en.attempts)
		maxAttempts = max(maxAttempts, en.attempts)
		if en.status == http.StatusOK && en.attempts == 2 {
			recovered++
		}
	}
	skips := int64(n) - a.Requests
	rate := 100.0
	if injected > 0 {
		rate = 100 * float64(recovered) / float64(injected)
	}
	t.Logf("requests=%d dep-a requests=%d completed=%d dep-b requests=%d; failures injected at dep-a=%d recovered=%d (%.1f%%); breaker skips of dep-a=%d",
		n, a.Requests, a.Completed, b.Requests, injected, recovered, rate, skips)

	if injected == 0 {
		t.Fatal("no failure was injected at dep-a, so nothing was measured")
	}
	if recovered != injected {
		t.Errorf("recovery rate %.1f%% (%d of %d injected failures recovered), want 100%%", rate, recovered, injected)
	}
	if want := int64(n) - a.Completed; b.Requests != want {
		t.Errorf("dep-b saw %d requests, want %d (requests minus dep-a completions: one retry per injected failure plus every skip of dep-a)", b.Requests, want)
	}
	if maxAttempts > 2 {
		t.Errorf("a request made %d attempts, want at most 2 against one failing and one healthy deployment", maxAttempts)
	}
	if want := a.Requests + b.Requests; sumAttempts != want {
		t.Errorf("logged attempts sum to %d but the upstreams saw %d requests: an upstream call was made that the gateway did not count", sumAttempts, want)
	}
}

func TestEligibleFailoverRate(t *testing.T) {
	e := newFailoverEnv(t, mock.Config{
		Seed:      7,
		FailRates: map[string]float64{"http500": 0.3, "reset": 0.1},
	}, "ua")

	replies := runPool(e2eUnaryCount, e2eWorkers, func(int) reply {
		return e.post(chatBody(e2eModel, "hi"))
	})

	assertFailoverRecovered(t, e, replies, false)
}

func TestStreamFailoverRate(t *testing.T) {
	e := newFailoverEnv(t, mock.Config{
		Seed:      7,
		FailRates: map[string]float64{"http500": 0.3, "reset": 0.1},
	}, "ua")

	replies := runPool(e2eStreamCount, e2eWorkers, func(int) reply {
		return e.post(streamReqBody(e2eModel, false))
	})

	assertFailoverRecovered(t, e, replies, true)
	// assertFailoverRecovered closed the server, so the log is complete: no
	// stream made more than the one attempt that failed and the one that
	// committed, so none made an upstream call after its commit.
	for i, en := range parseRequestLogs(t, e.logs.String()) {
		if en.status == http.StatusOK && en.attempts > 2 {
			t.Errorf("log line %d: stream returned 200 after %d attempts, want at most 2", i, en.attempts)
		}
	}
}

func TestMidstreamNeverRetried(t *testing.T) {
	// fail_after=3 sends the role chunk and two content chunks, so the
	// gateway has committed the response when the connection breaks.
	e := newFailoverEnv(t, mock.Config{
		Seed:      7,
		FailRates: map[string]float64{"midstream": 1},
	}, "ua;fail_after=3")

	// Fewer than the breaker's consecutive-failure threshold: dep-a is
	// certain to be tried, and dep-b must stay untouched.
	var replies []reply
	for i := 0; i < 4; i++ {
		replies = append(replies, e.post(streamReqBody(e2eModel, false)))
	}
	if got := e.b.Stats().Requests; got != 0 {
		t.Fatalf("dep-b saw %d requests after 4 mid-stream failures at dep-a, want 0: a committed stream was retried", got)
	}
	// The rest run concurrently. dep-a's breaker opens after five failures,
	// so later requests start at dep-b and finish normally; that is not a
	// retry, and the checks below tell the two apart.
	replies = append(replies, runPool(e2eMidstreamCnt-4, e2eWorkers, func(int) reply {
		return e.post(streamReqBody(e2eModel, false))
	})...)
	entries := e.finish(t, e2eMidstreamCnt)

	failedAtA, servedByB := 0, 0
	for i, r := range replies {
		if r.err != nil {
			t.Errorf("request %d: client error: %v", i, r.err)
			continue
		}
		if r.status != http.StatusOK {
			t.Errorf("request %d: status %d, want 200 (the response was committed)", i, r.status)
			continue
		}
		hasDone, hasError := strings.Contains(r.body, "data: [DONE]"), strings.Contains(r.body, `"error"`)
		switch r.dep {
		case "dep-a":
			failedAtA++
			if hasDone || !hasError {
				t.Errorf("request %d: broken stream at dep-a has [DONE]=%v error event=%v, want an error event and no [DONE]:\n%s", i, hasDone, hasError, r.body)
			}
		case "dep-b":
			servedByB++
			if !hasDone || hasError {
				t.Errorf("request %d: stream served by dep-b has [DONE]=%v error event=%v, want a clean stream", i, hasDone, hasError)
			}
		default:
			t.Errorf("request %d: deployment header %q, want dep-a or dep-b", i, r.dep)
		}
	}
	a, b := e.a.Stats(), e.b.Stats()
	t.Logf("requests=%d dep-a requests=%d (all broken mid-stream) dep-b requests=%d (started there after the breaker opened)", e2eMidstreamCnt, a.Requests, b.Requests)

	if a.Completed != 0 {
		t.Errorf("dep-a completed %d streams, want 0", a.Completed)
	}
	if got := a.Requests; got != int64(failedAtA) {
		t.Errorf("dep-a saw %d requests but %d replies came from it", got, failedAtA)
	}
	if got := b.Requests; got != int64(servedByB) {
		t.Errorf("dep-b saw %d requests but %d replies came from it", got, servedByB)
	}
	// The core property: one upstream request per client request. A retry
	// after the commit would make the upstream total exceed the request count.
	if got := a.Requests + b.Requests; got != e2eMidstreamCnt {
		t.Errorf("upstreams saw %d requests for %d client requests: a request made a second upstream call", got, e2eMidstreamCnt)
	}
	for i, en := range entries {
		if en.attempts != 1 {
			t.Errorf("log line %d: %d attempts, want 1 (no request is retried after a commit)", i, en.attempts)
		}
	}
}

// startTogether runs f for 0..n-1 on n goroutines released at the same
// moment and returns the results in index order.
func startTogether(n int, f func(i int) *httptest.ResponseRecorder) []*httptest.ResponseRecorder {
	out := make([]*httptest.ResponseRecorder, n)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			out[i] = f(i)
		}()
	}
	close(start)
	wg.Wait()
	return out
}

// tally counts 200s and 429s and fails on any other status or on a 429 that
// is not a rate_limit_exceeded error.
func tally(t *testing.T, recs []*httptest.ResponseRecorder) (ok, limited int) {
	t.Helper()
	for i, rec := range recs {
		switch rec.Code {
		case http.StatusOK:
			ok++
		case http.StatusTooManyRequests:
			limited++
			if code := decodeErr(t, rec).Error.Code; code != "rate_limit_exceeded" {
				t.Errorf("request %d: 429 with code %q, want rate_limit_exceeded", i, code)
			}
		default:
			t.Errorf("request %d: status %d, want 200 or 429: %s", i, rec.Code, rec.Body.String())
		}
	}
	return ok, limited
}

// trackedTokens wraps a TokenLimiter and tracks the tokens reserved but not
// refunded. A reservation is counted after the limiter allows it and a refund
// before the limiter applies it, so the count can only run low, never high:
// the tracker may miss a momentary overshoot but never invents one.
type trackedTokens struct {
	quota.TokenLimiter
	cur, peak atomic.Int64
}

func (tt *trackedTokens) Reserve(ctx context.Context, key string, lim quota.RateLimit, n int64, now time.Time) (quota.Result, error) {
	res, err := tt.TokenLimiter.Reserve(ctx, key, lim, n, now)
	if err == nil && res.Allowed && n > 0 {
		v := tt.cur.Add(n)
		for {
			p := tt.peak.Load()
			if v <= p || tt.peak.CompareAndSwap(p, v) {
				break
			}
		}
	}
	return res, err
}

func (tt *trackedTokens) Refund(ctx context.Context, key string, lim quota.RateLimit, n int64, now time.Time) error {
	if n > 0 {
		tt.cur.Add(-n)
	}
	return tt.TokenLimiter.Refund(ctx, key, lim, n, now)
}

func TestQuotaNeverOvershoots(t *testing.T) {
	const clients = 64
	backends := []struct {
		name string
		make func(t *testing.T) quota.Limiters
	}{
		{"memory", func(*testing.T) quota.Limiters { return quota.NewMemory() }},
		{"redis", func(t *testing.T) quota.Limiters {
			mr := miniredis.RunT(t)
			client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
			t.Cleanup(func() { _ = client.Close() })
			return quota.NewRedis(client, "routepilot")
		}},
	}

	for _, be := range backends {
		t.Run(be.name+"/rpm", func(t *testing.T) {
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				tn := tenantA(cfg)
				tn.RPM, tn.RPMBurst = 120, 20
				tn.TPM, tn.TPMBurst = 1_000_000, 1_000_000
			}, okProvider(), be.make(t))

			recs := startTogether(clients, func(int) *httptest.ResponseRecorder {
				return chat(gw, keyA, chatBody("model-a", "hi"))
			})

			ok, limited := tally(t, recs)
			t.Logf("%d clients at once, rpm 120 burst 20, clock fixed: admitted=%d limited=%d", clients, ok, limited)
			if ok != 20 || limited != clients-20 {
				t.Errorf("admitted %d and limited %d of %d concurrent requests, want exactly 20 admitted (the burst) and %d limited", ok, limited, clients, clients-20)
			}
		})

		t.Run(be.name+"/tpm no refunds", func(t *testing.T) {
			lim := be.make(t)
			tracked := &trackedTokens{TokenLimiter: lim.Tokens}
			lim.Tokens = tracked
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 1000
			}, okProvider(), lim) // no usage reported, so no reservation is refunded

			body := bodyWithMaxTokens("model-a", maxTokensFor(t, 300), false)
			recs := startTogether(clients, func(int) *httptest.ResponseRecorder {
				return chat(gw, keyA, body)
			})

			ok, limited := tally(t, recs)
			t.Logf("%d clients at once, tpm_burst 1000, 300 reserved each: admitted=%d limited=%d peak reserved=%d", clients, ok, limited, tracked.peak.Load())
			if p := tracked.peak.Load(); p > 1000 {
				t.Errorf("peak reserved tokens %d exceeds tpm_burst 1000", p)
			}
			if ok != 3 || limited != clients-3 {
				t.Errorf("admitted %d and limited %d, want exactly 3 admitted (900 of 1000 tokens; a fourth would need 1200)", ok, limited)
			}
		})

		t.Run(be.name+"/tpm with refunds", func(t *testing.T) {
			lim := be.make(t)
			tracked := &trackedTokens{TokenLimiter: lim.Tokens}
			lim.Tokens = tracked
			// Each admitted request uses 100 of its 300 reserved tokens and
			// refunds 200, so admissions and refunds interleave.
			p := completionWithUsage(func() *openai.Usage {
				time.Sleep(time.Millisecond)
				return &openai.Usage{PromptTokens: 40, CompletionTokens: 60}
			})
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 1000
			}, p, lim)

			body := bodyWithMaxTokens("model-a", maxTokensFor(t, 300), false)
			recs := startTogether(clients, func(int) *httptest.ResponseRecorder {
				return chat(gw, keyA, body)
			})

			ok, limited := tally(t, recs)
			t.Logf("%d clients at once, tpm_burst 1000, 300 reserved and 200 refunded each: admitted=%d limited=%d peak reserved=%d", clients, ok, limited, tracked.peak.Load())
			if p := tracked.peak.Load(); p > 1000 {
				t.Errorf("peak outstanding reserved tokens %d exceeds tpm_burst 1000", p)
			}
			if ok < 3 {
				t.Errorf("admitted %d, want at least the 3 that fit before any refund", ok)
			}
		})
	}
}

func TestConcurrencyNeverOvershoots(t *testing.T) {
	const (
		limit   = 8
		clients = 64
	)
	var inflight, peak atomic.Int64
	slow := mock.NewHandler(mock.Config{TTFT: mock.Dist{MedianMs: 100}}) // 100 ms per unary request
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v := inflight.Add(1)
		defer inflight.Add(-1)
		for {
			p := peak.Load()
			if v <= p || peak.CompareAndSwap(p, v) {
				break
			}
		}
		slow.ServeHTTP(w, r)
	}))
	t.Cleanup(up.Close)

	cfg := buildConfig(t, cfgOpts{baseURL: up.URL + "/v1"})
	tenantA(cfg).MaxConcurrency = limit
	gw, err := New(cfg, map[string]provider.Provider{
		"mockp": openaicompat.New(cfg.Providers[0], "", up.Client()),
	}, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	recs := startTogether(clients, func(int) *httptest.ResponseRecorder {
		return chat(gw, keyA, chatBody("model-a", "hi"))
	})
	ok, limited := tally(t, recs)
	t.Logf("%d clients at once, max_concurrency %d, 100 ms upstream: admitted=%d limited=%d peak in-flight upstream requests=%d", clients, limit, ok, limited, peak.Load())
	if p := peak.Load(); p > limit {
		t.Errorf("observed %d upstream requests in flight, want at most max_concurrency %d", p, limit)
	}
	if ok < 1 {
		t.Error("no request was admitted")
	}

	// Every slot was released: a full wave of 8 is admitted again.
	recs = startTogether(limit, func(int) *httptest.ResponseRecorder {
		return chat(gw, keyA, chatBody("model-a", "hi"))
	})
	if ok, limited := tally(t, recs); ok != limit {
		t.Errorf("second wave of %d: admitted %d, limited %d, want all admitted after the slots were released", limit, ok, limited)
	}
	if p := peak.Load(); p > limit {
		t.Errorf("observed %d upstream requests in flight after the second wave, want at most %d", p, limit)
	}
}
