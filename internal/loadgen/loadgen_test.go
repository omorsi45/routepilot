package loadgen

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testKey = "test-key-loadgen"

// testConfig is a short, quiet run against url: 100 requests a second for half a second.
func testConfig(url string) Config {
	return Config{
		URL: url, APIKey: testKey, Model: "m", RPS: 100,
		Duration: 500 * time.Millisecond, Seed: 1, MaxInflight: 64, MaxTokens: 16,
	}
}

func run(t *testing.T, cfg Config) Result {
	t.Helper()
	res, err := Run(context.Background(), NewClient(cfg.MaxInflight), cfg)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return res
}

// okBody is a minimal non-streaming chat completion.
const okBody = `{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`

func okHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, okBody)
}

// sseHandler answers a stream: an empty role chunk at once, the first content chunk after
// firstContent, four more content chunks 5 ms apart, then [DONE].
func sseHandler(firstContent time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		rc := http.NewResponseController(w)
		_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`+"\n\n")
		_ = rc.Flush()
		time.Sleep(firstContent)
		for i := range 5 {
			_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"tok"}}]}`+"\n\n")
			_ = rc.Flush()
			if i < 4 {
				time.Sleep(5 * time.Millisecond)
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		_ = rc.Flush()
	}
}

func TestArrivalsArePoissonAndSeeded(t *testing.T) {
	const rps = 1000.0
	horizon := 100 * time.Second
	a := arrivals(7, rps, horizon)

	if want := rps * horizon.Seconds(); math.Abs(float64(len(a))-want)/want > 0.03 {
		t.Fatalf("got %d arrivals, want about %.0f", len(a), want)
	}
	if !slices.IsSorted(a) || a[len(a)-1] >= horizon {
		t.Fatalf("arrivals must be sorted and inside the horizon, last = %v", a[len(a)-1])
	}

	var sum, sumSq float64
	prev := 0.0
	for _, d := range a {
		gap := d.Seconds() - prev
		prev = d.Seconds()
		sum += gap
		sumSq += gap * gap
	}
	n := float64(len(a))
	mean := sum / n
	if want := 1 / rps; math.Abs(mean-want)/want > 0.03 {
		t.Errorf("mean gap = %.6fs, want %.6fs within 3%%", mean, want)
	}
	// Exponential gaps have a standard deviation equal to their mean.
	if cv := math.Sqrt(sumSq/n-mean*mean) / mean; math.Abs(cv-1) > 0.03 {
		t.Errorf("coefficient of variation = %.3f, want 1 within 3%% (exponential gaps)", cv)
	}

	if !slices.Equal(a, arrivals(7, rps, horizon)) {
		t.Error("the same seed must give the same schedule")
	}
	if slices.Equal(a, arrivals(8, rps, horizon)) {
		t.Error("a different seed must give a different schedule")
	}
}

func TestLatencyFromScheduledTime(t *testing.T) {
	// The first request starts a 200 ms stall that holds every request that arrives during
	// it. A closed loop would see one slow request; an open loop that measures from the
	// scheduled time sees every request that should have been sent during the stall.
	var mu sync.Mutex
	var until time.Time
	started := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		if !started {
			started = true
			until = time.Now().Add(200 * time.Millisecond)
		}
		wait := time.Until(until)
		mu.Unlock()
		time.Sleep(wait)
		okHandler(w, r)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.RPS, cfg.Duration, cfg.MaxInflight = 200, time.Second, 256
	res := run(t, cfg)

	if res.Errors != 0 || res.Dropped != 0 || res.Completed != res.Scheduled {
		t.Fatalf("scheduled=%d completed=%d errors=%d dropped=%d, want every request completed",
			res.Scheduled, res.Completed, res.Errors, res.Dropped)
	}
	ms := func(q float64) float64 { return float64(res.Latency.ValueAtQuantile(q)) / 1000 }
	if got := float64(res.Latency.Max()) / 1000; got < 150 {
		t.Errorf("max latency = %.1f ms, want the stall (>= 150 ms)", got)
	}
	// About 20% of the requests were scheduled inside the stall, so the 85th percentile
	// already lies in it and the 99th is near its full length.
	if got := ms(85); got < 20 {
		t.Errorf("p85 = %.1f ms, want >= 20 ms: requests queued behind the stall must show it", got)
	}
	if got := ms(99); got < 150 {
		t.Errorf("p99 = %.1f ms, want >= 150 ms", got)
	}
	if got := ms(50); got > 100 {
		t.Errorf("p50 = %.1f ms, want the unstalled requests to be fast", got)
	}
}

func TestOneMeasuresFromScheduledTime(t *testing.T) {
	// A request whose scheduled time is 300 ms in the past was launched late by the
	// scheduler. The server answers at once, yet the latency must include the lateness:
	// it runs from the scheduled time, not from the moment of sending.
	srv := httptest.NewServer(http.HandlerFunc(okHandler))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	body, err := requestBody(cfg)
	if err != nil {
		t.Fatal(err)
	}
	g := &generator{client: NewClient(1), cfg: cfg, body: body, timeout: requestTimeout, rec: newRecorder()}
	g.one(context.Background(), time.Now().Add(-300*time.Millisecond), true)

	if g.rec.completed != 1 {
		t.Fatalf("completed = %d, errors = %v, want 1 completed request", g.rec.completed, g.rec.byKind)
	}
	if got := g.rec.latency.Min(); got < 300_000 {
		t.Errorf("latency = %d us, want >= 300000 us (measured from the scheduled time)", got)
	}
}

func TestDropWhenInflightFull(t *testing.T) {
	var cur, peak, seen atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Add(1)
		n := cur.Add(1)
		for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
		}
		time.Sleep(100 * time.Millisecond)
		cur.Add(-1)
		okHandler(w, r)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.RPS, cfg.Duration, cfg.MaxInflight = 200, 500*time.Millisecond, 3
	begin := time.Now()
	res := run(t, cfg)
	took := time.Since(begin)

	if res.Dropped == 0 || res.Completed < 3 {
		t.Fatalf("dropped=%d completed=%d, want drops once 3 requests are in flight", res.Dropped, res.Completed)
	}
	if res.Completed+res.Errors+res.Dropped != res.Scheduled {
		t.Errorf("completed %d + errors %d + dropped %d != scheduled %d", res.Completed, res.Errors, res.Dropped, res.Scheduled)
	}
	if p := peak.Load(); p > 3 {
		t.Errorf("server saw %d requests at once, want at most MaxInflight = 3", p)
	}
	if got := seen.Load(); got != int64(res.Completed+res.Errors) {
		t.Errorf("server saw %d requests, want %d (a dropped request is never sent)", got, res.Completed+res.Errors)
	}
	if got := res.Latency.TotalCount(); got != int64(res.Completed) {
		t.Errorf("latency samples = %d, want %d: a dropped request records no latency", got, res.Completed)
	}
	// Dropped requests are not delayed: the run lasts the window plus one request.
	if took > time.Second+500*time.Millisecond {
		t.Errorf("run took %v, want about 600 ms: a full pool must drop, not wait", took)
	}
}

func TestWarmupRequestsAreSentButNotRecorded(t *testing.T) {
	var seen atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Add(1)
		okHandler(w, r)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Warmup, cfg.Duration = 300*time.Millisecond, 500*time.Millisecond
	res := run(t, cfg)

	want := 0
	for _, off := range arrivals(cfg.Seed, cfg.RPS, cfg.Warmup+cfg.Duration) {
		if off >= cfg.Warmup {
			want++
		}
	}
	if res.Scheduled != want || res.Completed != want {
		t.Errorf("scheduled=%d completed=%d, want %d (only the measurement window counts)", res.Scheduled, res.Completed, want)
	}
	if got := res.Latency.TotalCount(); got != int64(want) {
		t.Errorf("latency samples = %d, want %d", got, want)
	}
	if got := seen.Load(); got <= int64(want) {
		t.Errorf("server saw %d requests, want more than %d: warmup requests are still sent", got, want)
	}
}

func TestRequestShape(t *testing.T) {
	type seenReq struct {
		method, path, auth, ctype, accept string
		body                              []byte
	}
	got := make(chan seenReq, 64)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		select {
		case got <- seenReq{r.Method, r.URL.Path, r.Header.Get("Authorization"), r.Header.Get("Content-Type"), r.Header.Get("Accept"), b}:
		default:
		}
		okHandler(w, r)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL + "/v1/chat/completions")
	run(t, cfg)

	req := <-got
	if req.method != http.MethodPost || req.path != "/v1/chat/completions" {
		t.Errorf("request = %s %s, want POST /v1/chat/completions", req.method, req.path)
	}
	if req.auth != "Bearer "+testKey {
		t.Errorf("Authorization header = %q, want the bearer key", req.auth)
	}
	if req.ctype != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", req.ctype)
	}
	var body struct {
		Model     string `json:"model"`
		Stream    bool   `json:"stream"`
		MaxTokens int    `json:"max_tokens"`
		Messages  []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(req.body, &body); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	if body.Model != "m" || body.Stream || body.MaxTokens != 16 {
		t.Errorf("body model=%q stream=%v max_tokens=%d, want m/false/16", body.Model, body.Stream, body.MaxTokens)
	}
	if len(body.Messages) != 1 || body.Messages[0].Role != "user" {
		t.Fatalf("messages = %+v, want one user message", body.Messages)
	}
	if n := len(body.Messages[0].Content); n < 150 || n > 250 {
		t.Errorf("message is %d bytes, want about 200", n)
	}
}

func TestStreamingMeasuresFirstContentAndDone(t *testing.T) {
	var streamed atomic.Bool
	handler := sseHandler(30 * time.Millisecond)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if strings.Contains(string(b), `"stream":true`) && r.Header.Get("Accept") == "text/event-stream" {
			streamed.Store(true)
		}
		handler(w, r)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Stream = true
	res := run(t, cfg)

	if !streamed.Load() {
		t.Error("a streaming run must send stream:true and Accept: text/event-stream")
	}
	if res.Errors != 0 || res.Completed != res.Scheduled || res.Completed == 0 {
		t.Fatalf("scheduled=%d completed=%d errors=%d kinds=%v", res.Scheduled, res.Completed, res.Errors, res.ErrorsByKind)
	}
	if got := res.TTFT.TotalCount(); got != int64(res.Completed) {
		t.Fatalf("ttft samples = %d, want %d", got, res.Completed)
	}
	// The role chunk has empty content and arrives at once: TTFT waits for real content.
	if min := float64(res.TTFT.Min()) / 1000; min < 25 {
		t.Errorf("min TTFT = %.1f ms, want >= 25 ms (the empty role chunk is not the first token)", min)
	}
	// Latency runs to [DONE], 20 ms of inter-token gaps after the first content.
	if min := float64(res.Latency.Min()) / 1000; min < 45 {
		t.Errorf("min latency = %.1f ms, want >= 45 ms", min)
	}
	if res.Latency.Min() <= res.TTFT.Min() {
		t.Errorf("latency %d us must exceed TTFT %d us", res.Latency.Min(), res.TTFT.Min())
	}
}

func TestNonStreamingHasNoTTFT(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(okHandler))
	defer srv.Close()
	res := run(t, testConfig(srv.URL))
	if res.TTFT.TotalCount() != 0 {
		t.Errorf("non-streaming run recorded %d TTFT samples, want 0", res.TTFT.TotalCount())
	}
	if res.Latency.TotalCount() != int64(res.Completed) || res.Completed == 0 {
		t.Errorf("latency samples = %d, completed = %d", res.Latency.TotalCount(), res.Completed)
	}
}

func TestErrorKinds(t *testing.T) {
	closedPort := func() string {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		addr := ln.Addr().String()
		_ = ln.Close()
		return "http://" + addr
	}
	tests := []struct {
		name    string
		stream  bool
		timeout time.Duration
		handler http.HandlerFunc // nil: nothing listens
		want    string
	}{
		{name: "http 500", handler: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"error":{"message":"boom"}}`, http.StatusInternalServerError)
		}, want: "http_500"},
		{name: "http 429", handler: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "slow down", http.StatusTooManyRequests)
		}, want: "http_429"},
		{name: "connect", want: "connect"},
		{name: "timeout", timeout: 50 * time.Millisecond, handler: func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(300 * time.Millisecond)
			okHandler(w, r)
		}, want: "timeout"},
		{name: "short body", handler: func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = io.WriteString(w, "short")
		}, want: "read"},
		{name: "stream without done", stream: true, handler: func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"tok"}}]}`+"\n\n")
		}, want: "read"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := closedPort()
			if tt.handler != nil {
				srv := httptest.NewServer(tt.handler)
				defer srv.Close()
				url = srv.URL
			}
			cfg := testConfig(url)
			cfg.Stream, cfg.timeout = tt.stream, tt.timeout
			res := run(t, cfg)

			if res.Scheduled == 0 || res.Errors != res.Scheduled || res.Completed != 0 {
				t.Fatalf("scheduled=%d completed=%d errors=%d dropped=%d kinds=%v, want every request an error",
					res.Scheduled, res.Completed, res.Errors, res.Dropped, res.ErrorsByKind)
			}
			if len(res.ErrorsByKind) != 1 || res.ErrorsByKind[tt.want] != res.Errors {
				t.Errorf("ErrorsByKind = %v, want only %s = %d", res.ErrorsByKind, tt.want, res.Errors)
			}
			if res.Latency.TotalCount() != 0 || res.TTFT.TotalCount() != 0 {
				t.Errorf("errors must stay out of the histograms, got %d latency and %d ttft samples",
					res.Latency.TotalCount(), res.TTFT.TotalCount())
			}
		})
	}
}

func TestConnectionsAreReused(t *testing.T) {
	var conns atomic.Int64
	srv := httptest.NewUnstartedServer(http.HandlerFunc(okHandler))
	srv.Config.ConnState = func(_ net.Conn, s http.ConnState) {
		if s == http.StateNew {
			conns.Add(1)
		}
	}
	srv.Start()
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.RPS, cfg.Duration = 50, time.Second
	res := run(t, cfg)

	if res.Completed < 30 {
		t.Fatalf("completed %d, want at least 30", res.Completed)
	}
	if got := conns.Load(); got > 6 {
		t.Errorf("%d connections for %d requests, want the keep-alive pool to be reused", got, res.Completed)
	}
}

func TestNewClientTransport(t *testing.T) {
	c := NewClient(321)
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport is %T, want a dedicated *http.Transport", c.Transport)
	}
	if tr == http.DefaultTransport {
		t.Error("the client must not share http.DefaultTransport")
	}
	if tr.MaxIdleConnsPerHost != 321 || tr.MaxConnsPerHost != 0 || tr.DisableKeepAlives {
		t.Errorf("MaxIdleConnsPerHost=%d MaxConnsPerHost=%d DisableKeepAlives=%v, want 321/0/false",
			tr.MaxIdleConnsPerHost, tr.MaxConnsPerHost, tr.DisableKeepAlives)
	}
}

func TestRunStopsWhenContextCanceled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(okHandler))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Duration = 30 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	begin := time.Now()
	_, err := Run(ctx, NewClient(cfg.MaxInflight), cfg)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
	if took := time.Since(begin); took > 3*time.Second {
		t.Errorf("Run took %v after cancellation, want it to stop promptly", took)
	}
}

func TestRunRejectsBadConfig(t *testing.T) {
	good := testConfig("http://127.0.0.1:1/v1/chat/completions")
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{"empty url", func(c *Config) { c.URL = "" }},
		{"relative url", func(c *Config) { c.URL = "/v1/chat/completions" }},
		{"empty model", func(c *Config) { c.Model = "" }},
		{"zero rps", func(c *Config) { c.RPS = 0 }},
		{"nan rps", func(c *Config) { c.RPS = math.NaN() }},
		{"zero duration", func(c *Config) { c.Duration = 0 }},
		{"negative warmup", func(c *Config) { c.Warmup = -time.Second }},
		{"zero max inflight", func(c *Config) { c.MaxInflight = 0 }},
		{"negative max tokens", func(c *Config) { c.MaxTokens = -1 }},
		{"more arrivals than memory allows", func(c *Config) { c.RPS, c.Duration = 1e6, time.Hour }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := good
			tt.mutate(&cfg)
			if _, err := Run(context.Background(), NewClient(1), cfg); err == nil {
				t.Error("want an error for an invalid config")
			}
		})
	}
}
