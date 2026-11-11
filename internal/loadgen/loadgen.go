// Package loadgen is an open-loop, coordinated-omission-safe HTTP load generator for
// OpenAI-compatible chat completion endpoints.
//
// Open loop: the arrival times are drawn up front from a seeded Poisson process, a single
// scheduler goroutine launches each request at its scheduled time and never waits for a
// response, and a request's latency runs from its scheduled time, not from the moment it
// was sent. A server that stalls therefore shows the stall in every request that should
// have been sent during it, instead of hiding it by slowing the generator down (coordinated
// omission). A request that cannot start because MaxInflight requests are already in flight
// is dropped and counted, never delayed.
package loadgen

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/HdrHistogram/hdrhistogram-go"
)

const (
	// requestTimeout bounds one request, from the moment it is sent to the last byte read.
	requestTimeout = 30 * time.Second

	// The histograms hold microseconds from 1 us to 120 s with 3 significant digits.
	histMin    = 1
	histMax    = 120_000_000
	histDigits = 3

	// maxArrivals bounds the schedule so a typo in the rate or the duration cannot exhaust
	// memory before the first request is sent.
	maxArrivals = 20_000_000

	// maxErrorBody is how much of an error response is read so the connection can be reused.
	maxErrorBody = 1 << 20

	// arrivalStream is the second PCG word of the arrival generator: any constant works, it
	// only separates this generator's stream from other uses of the same seed.
	arrivalStream = 0x9e3779b97f4a7c15
)

// prompt is the invented user message every request carries (about 200 bytes). It is fixed
// so that runs differ only in the system under test.
const prompt = "Explain in two short sentences why a small online shop should keep regular " +
	"backups of its order data, and name one common mistake teams make when they first " +
	"automate nightly backups of customer records."

// Config describes one load run.
type Config struct {
	// URL is the full chat completions endpoint, for example
	// http://127.0.0.1:8080/v1/chat/completions.
	URL string
	// APIKey is sent as a bearer token. Empty sends no Authorization header.
	APIKey string
	// Model is the model field of every request.
	Model string
	// Stream asks for a server-sent event stream and records the time to first token.
	Stream bool
	// RPS is the mean arrival rate in requests per second.
	RPS float64
	// Warmup is the time at the start of the run whose requests are sent but not recorded.
	Warmup time.Duration
	// Duration is the length of the measurement window that follows the warmup.
	Duration time.Duration
	// Seed seeds the arrival schedule, so two runs with the same seed send at the same times.
	Seed int64
	// MaxInflight is the most requests in flight at once; further arrivals are dropped.
	MaxInflight int
	// MaxTokens is the max_tokens field of every request; zero omits it.
	MaxTokens int

	// timeout bounds one request; zero means requestTimeout. Tests shorten it.
	timeout time.Duration
}

// Result is the outcome of one load run. Every count and histogram covers the measurement
// window only; requests of the warmup are sent but appear nowhere in it.
type Result struct {
	// Scheduled is the number of arrivals in the measurement window. It always equals
	// Completed + Errors + Dropped for a run that was not canceled.
	Scheduled int
	// Completed counts requests that finished with a 2xx response and a complete body.
	Completed int
	// Errors counts requests that failed: any non-2xx status or transport error.
	Errors int
	// Dropped counts arrivals that found MaxInflight requests in flight.
	Dropped int
	// Latency is the time from the scheduled arrival to the full body (non-streaming) or to
	// [DONE] (streaming), in microseconds, for completed requests.
	Latency *hdrhistogram.Histogram
	// TTFT is the time from the scheduled arrival to the first stream event with content,
	// in microseconds. It is empty for non-streaming runs.
	TTFT *hdrhistogram.Histogram
	// LaunchLag is how late the scheduler launched each request, in microseconds. The lag is
	// part of every latency; it shows how much of it the generator itself added.
	LaunchLag *hdrhistogram.Histogram
	// ErrorsByKind counts Errors by kind: http_<status>, timeout, connect or read.
	ErrorsByKind map[string]int
	// Elapsed is the time from the start of the measurement window to the end of the last
	// request, at least Duration for a run that was not canceled.
	Elapsed time.Duration
}

// NewClient returns an HTTP client with a transport of its own, sized for maxInflight
// concurrent requests: keep-alives on and no cap on connections per host. Callers create one
// client per run so that runs do not share connections.
func NewClient(maxInflight int) *http.Client {
	idle := max(maxInflight, 1)
	return &http.Client{Transport: &http.Transport{
		Proxy:               nil, // load goes straight to the target, never through a proxy
		DialContext:         (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:        idle,
		MaxIdleConnsPerHost: idle,
		MaxConnsPerHost:     0,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true, // measure the bytes the server sends, not a gzip layer
	}}
}

// Validate reports why cfg cannot run. Messages never repeat the API key.
func (cfg Config) Validate() error {
	u, err := url.Parse(cfg.URL)
	switch {
	case err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "":
		return errors.New("loadgen: URL must be an absolute http or https URL")
	case cfg.Model == "":
		return errors.New("loadgen: Model must not be empty")
	case !(cfg.RPS > 0) || math.IsInf(cfg.RPS, 0):
		return fmt.Errorf("loadgen: RPS must be positive and finite, got %v", cfg.RPS)
	case cfg.Warmup < 0:
		return fmt.Errorf("loadgen: Warmup must not be negative, got %v", cfg.Warmup)
	case cfg.Duration <= 0:
		return fmt.Errorf("loadgen: Duration must be positive, got %v", cfg.Duration)
	case cfg.MaxInflight <= 0:
		return fmt.Errorf("loadgen: MaxInflight must be positive, got %d", cfg.MaxInflight)
	case cfg.MaxTokens < 0:
		return fmt.Errorf("loadgen: MaxTokens must not be negative, got %d", cfg.MaxTokens)
	case cfg.RPS*(cfg.Warmup+cfg.Duration).Seconds() > maxArrivals:
		return fmt.Errorf("loadgen: %v for %v is more than %d requests", cfg.RPS, cfg.Warmup+cfg.Duration, maxArrivals)
	}
	return nil
}

// arrivals returns the offsets from the start of the run at which requests are scheduled:
// the points of a Poisson process of rate rps, that is, exponential gaps with mean 1/rps,
// up to horizon. The same seed always gives the same offsets.
func arrivals(seed int64, rps float64, horizon time.Duration) []time.Duration {
	rng := rand.New(rand.NewPCG(uint64(seed), arrivalStream))
	end := horizon.Seconds()
	out := make([]time.Duration, 0, int(rps*end*1.05)+16)
	for t := 0.0; ; {
		t += rng.ExpFloat64() / rps
		if t >= end {
			return out
		}
		out = append(out, time.Duration(t*float64(time.Second)))
	}
}

// requestBody is the JSON body every request sends.
func requestBody(cfg Config) ([]byte, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type chatRequest struct {
		Model     string    `json:"model"`
		Messages  []message `json:"messages"`
		MaxTokens int       `json:"max_tokens,omitempty"`
		Stream    bool      `json:"stream"`
	}
	b, err := json.Marshal(chatRequest{
		Model:     cfg.Model,
		Messages:  []message{{Role: "user", Content: prompt}},
		MaxTokens: cfg.MaxTokens,
		Stream:    cfg.Stream,
	})
	if err != nil {
		return nil, fmt.Errorf("loadgen: encode the request body: %w", err)
	}
	return b, nil
}

func newHistogram() *hdrhistogram.Histogram {
	return hdrhistogram.New(histMin, histMax, histDigits)
}

// recorder collects the outcome of the measured requests. Workers call it concurrently.
type recorder struct {
	mu        sync.Mutex
	completed int
	errors    int
	byKind    map[string]int
	latency   *hdrhistogram.Histogram
	ttft      *hdrhistogram.Histogram
	lastEnd   time.Time
}

func newRecorder() *recorder {
	return &recorder{byKind: map[string]int{}, latency: newHistogram(), ttft: newHistogram()}
}

// micros converts d to histogram units, clamped to the range the histograms track.
func micros(d time.Duration) int64 {
	return min(max(d.Microseconds(), histMin), histMax)
}

func (r *recorder) success(latency, ttft time.Duration, stream bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.completed++
	_ = r.latency.RecordValue(micros(latency)) // clamped into range, so it cannot fail
	if stream {
		_ = r.ttft.RecordValue(micros(ttft))
	}
}

func (r *recorder) failure(kind string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.errors++
	r.byKind[kind]++
}

func (r *recorder) finished(at time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if at.After(r.lastEnd) {
		r.lastEnd = at
	}
}

// generator is the state shared by the requests of one run.
type generator struct {
	client  *http.Client
	cfg     Config
	body    []byte
	timeout time.Duration
	rec     *recorder
}

// Run sends requests at cfg.RPS for cfg.Warmup + cfg.Duration through c and returns what the
// measurement window saw. c should come from NewClient, so that its transport is sized for
// cfg.MaxInflight. Run returns after every request in flight has finished. When ctx ends
// first, it stops scheduling, drops the requests it cut short from the counts, and returns
// the partial result with ctx.Err().
func Run(ctx context.Context, c *http.Client, cfg Config) (Result, error) {
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}
	body, err := requestBody(cfg)
	if err != nil {
		return Result{}, err
	}
	g := &generator{
		client: c, cfg: cfg, body: body, timeout: cmp.Or(cfg.timeout, requestTimeout), rec: newRecorder(),
	}
	schedule := arrivals(cfg.Seed, cfg.RPS, cfg.Warmup+cfg.Duration)
	lag := newHistogram()

	var (
		scheduled, dropped int
		inflight           atomic.Int64
		wg                 sync.WaitGroup
	)
	timer := time.NewTimer(time.Hour)
	timer.Stop()

	t0 := time.Now() // carries a monotonic reading, so every offset below is monotonic too
	for _, off := range schedule {
		at := t0.Add(off)
		if !sleepUntil(ctx, timer, at) {
			break
		}
		measured := off >= cfg.Warmup
		if measured {
			scheduled++
			_ = lag.RecordValue(micros(time.Since(at)))
		}
		if inflight.Add(1) > int64(cfg.MaxInflight) {
			inflight.Add(-1)
			if measured {
				dropped++
			}
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer inflight.Add(-1)
			g.one(ctx, at, measured)
		}()
	}
	wg.Wait()

	start := t0.Add(cfg.Warmup)
	end := g.rec.lastEnd
	if ctx.Err() == nil {
		end = latest(end, start.Add(cfg.Duration))
	}
	res := Result{
		Scheduled: scheduled, Completed: g.rec.completed, Errors: g.rec.errors, Dropped: dropped,
		Latency: g.rec.latency, TTFT: g.rec.ttft, LaunchLag: lag, ErrorsByKind: g.rec.byKind,
		Elapsed: max(end.Sub(start), 0),
	}
	return res, ctx.Err()
}

func latest(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}

// sleepUntil waits for at, or returns false at once when ctx ends. An arrival that is already
// due returns without waiting, so a scheduler that fell behind catches up in a burst, and its
// lateness shows in the latency of those requests.
func sleepUntil(ctx context.Context, timer *time.Timer, at time.Time) bool {
	d := time.Until(at)
	if d <= 0 {
		return ctx.Err() == nil
	}
	timer.Reset(d)
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		timer.Stop()
		return false
	}
}

// one sends a single request scheduled for at and records its outcome when measured.
func (g *generator) one(ctx context.Context, at time.Time, measured bool) {
	latency, ttft, kind := g.send(ctx, at)
	end := time.Now()
	if ctx.Err() != nil || !measured {
		return // cut short by the end of the run, or a warmup request
	}
	g.rec.finished(end)
	if kind != "" {
		g.rec.failure(kind)
		return
	}
	g.rec.success(latency, ttft, g.cfg.Stream)
}

// send performs the request and returns its latency and time to first token, both measured
// from at, or the kind of error that ended it.
func (g *generator) send(ctx context.Context, at time.Time) (latency, ttft time.Duration, kind string) {
	rctx, cancel := context.WithTimeout(ctx, g.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, g.cfg.URL, bytes.NewReader(g.body))
	if err != nil {
		return 0, 0, "connect" // the URL was validated, so this is not expected
	}
	req.Header.Set("Content-Type", "application/json")
	if g.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.APIKey)
	}
	if g.cfg.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return 0, 0, errorKind(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody)) // lets the connection be reused
		return 0, 0, "http_" + strconv.Itoa(resp.StatusCode)
	}

	if !g.cfg.Stream {
		if _, err := io.Copy(io.Discard, resp.Body); err != nil {
			return 0, 0, errorKind(err)
		}
		return time.Since(at), 0, ""
	}
	first, done, err := readStream(resp.Body)
	if err != nil {
		return 0, 0, errorKind(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body) // the chunked terminator, so the connection is reused
	return done.Sub(at), first.Sub(at), ""
}

// errNoContent is a stream that reached [DONE] without a single event with content.
var errNoContent = errors.New("stream ended without any content")

// readStream reads server-sent events until [DONE]. It returns when the first event with
// content arrived and when [DONE] did.
func readStream(body io.Reader) (first, done time.Time, err error) {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 4096), 1<<20)
	for sc.Scan() {
		payload, ok := bytes.CutPrefix(sc.Bytes(), []byte("data:"))
		if !ok {
			continue
		}
		payload = bytes.TrimSpace(payload)
		if string(payload) == "[DONE]" {
			if first.IsZero() {
				return first, done, errNoContent
			}
			return first, time.Now(), nil
		}
		if first.IsZero() && hasContent(payload) {
			first = time.Now()
		}
	}
	if err := sc.Err(); err != nil {
		return first, done, err
	}
	return first, done, io.ErrUnexpectedEOF
}

// hasContent reports whether a chat completion chunk carries a non-empty delta content.
func hasContent(payload []byte) bool {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(payload, &chunk) != nil {
		return false
	}
	for _, c := range chunk.Choices {
		if c.Delta.Content != "" {
			return true
		}
	}
	return false
}

// errorKind names a transport error: timeout (the request deadline or a network timeout),
// connect (the dial failed) or read (anything else: a reset, a truncated body, a broken
// stream).
func errorKind(err error) string {
	var ne net.Error
	var oe *net.OpError
	switch {
	case errors.Is(err, context.DeadlineExceeded), errors.As(err, &ne) && ne.Timeout():
		return "timeout"
	case errors.As(err, &oe) && oe.Op == "dial":
		return "connect"
	}
	return "read"
}
