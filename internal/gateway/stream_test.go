package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
	"github.com/omorsi45/routepilot/internal/reliability"
)

// streamClient is an HTTP client with its own transport so a test can drop its
// idle connections before counting goroutines.
type streamClient struct {
	*http.Client
	tr *http.Transport
}

func newStreamClient() *streamClient {
	tr := &http.Transport{}
	return &streamClient{Client: &http.Client{Transport: tr}, tr: tr}
}

// startGateway serves the env's gateway on a real loopback listener.
func startGateway(t *testing.T, e *env) string {
	t.Helper()
	srv := httptest.NewServer(e.gw.Handler())
	t.Cleanup(srv.Close)
	return srv.URL
}

func streamReqBody(model string, includeUsage bool) string {
	opts := ""
	if includeUsage {
		opts = `,"stream_options":{"include_usage":true}`
	}
	return fmt.Sprintf(`{"model":%q,"stream":true,"messages":[{"role":"user","content":"hi"}]%s}`, model, opts)
}

func postStream(t *testing.T, c *streamClient, url, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+keyA)
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

// readEvent returns the data of the next SSE event, or the read error.
func readEvent(br *bufio.Reader) (string, error) {
	var data []string
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return "", err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if len(data) > 0 {
				return strings.Join(data, "\n"), nil
			}
			continue
		}
		if v, ok := strings.CutPrefix(line, "data: "); ok {
			data = append(data, v)
		}
	}
}

// readAll returns every event up to the end of the body.
func readAll(t *testing.T, resp *http.Response) []string {
	t.Helper()
	br := bufio.NewReader(resp.Body)
	var events []string
	for {
		ev, err := readEvent(br)
		if err != nil {
			return events
		}
		events = append(events, ev)
	}
}

func decodeChunk(t *testing.T, ev string) openai.ChatCompletionChunk {
	t.Helper()
	var c openai.ChatCompletionChunk
	if err := json.Unmarshal([]byte(ev), &c); err != nil {
		t.Fatalf("event is not a chunk: %v\n%s", err, ev)
	}
	return c
}

func TestStreamHappyPath(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=4"})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	// The Go client keeps Connection in resp.Header (verified: the assertion
	// below passes), so keep-alive is asserted like the other headers.
	for k, want := range map[string]string{
		"Content-Type":            "text/event-stream",
		"Cache-Control":           "no-cache",
		"X-Accel-Buffering":       "no",
		"Connection":              "keep-alive",
		"x-routepilot-model":      "model-a",
		"x-routepilot-deployment": "dep-a",
	} {
		if got := resp.Header.Get(k); got != want {
			t.Errorf("header %s = %q, want %q", k, got, want)
		}
	}
	reqID := resp.Header.Get("x-routepilot-request-id")
	if reqID == "" {
		t.Fatal("missing x-routepilot-request-id")
	}

	events := readAll(t, resp)
	// role + 4 content + finish + [DONE]; the usage chunk is withheld.
	if len(events) != 7 {
		t.Fatalf("got %d events, want 7: %q", len(events), events)
	}
	if events[len(events)-1] != "[DONE]" {
		t.Errorf("last event %q, want [DONE]", events[len(events)-1])
	}
	var content strings.Builder
	for _, ev := range events[:6] {
		if strings.Contains(ev, `"usage"`) {
			t.Errorf("usage leaked to a client that did not ask: %s", ev)
		}
		c := decodeChunk(t, ev)
		if c.ID != "chatcmpl-"+reqID || c.Model != "model-a" {
			t.Errorf("id/model %q/%q, want chatcmpl-%s/model-a", c.ID, c.Model, reqID)
		}
		if c.Created == 0 {
			t.Error("created is zero")
		}
		if len(c.Choices) != 1 {
			t.Fatalf("chunk has %d choices: %s", len(c.Choices), ev)
		}
		if d := c.Choices[0].Delta.Content; d != nil {
			content.WriteString(*d)
		}
	}
	if got := content.String(); got != "tok0 tok1 tok2 tok3" {
		t.Errorf("content %q, want %q", got, "tok0 tok1 tok2 tok3")
	}
	if first := decodeChunk(t, events[0]); first.Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk role %q, want assistant", first.Choices[0].Delta.Role)
	}
	if last := decodeChunk(t, events[5]); last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "stop" {
		t.Errorf("last chunk finish_reason %v, want stop", last.Choices[0].FinishReason)
	}
	if got := e.mock.Stats().Requests; got != 1 {
		t.Errorf("upstream requests %d, want 1", got)
	}
}

func TestStreamUsageWhenRequested(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=4"})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", true))
	reqID := resp.Header.Get("x-routepilot-request-id")

	events := readAll(t, resp)
	// role + 4 content + finish + usage + [DONE]
	if len(events) != 8 {
		t.Fatalf("got %d events, want 8: %q", len(events), events)
	}
	if events[7] != "[DONE]" {
		t.Errorf("last event %q, want [DONE]", events[7])
	}
	u := decodeChunk(t, events[6])
	if len(u.Choices) != 0 || u.Usage == nil || u.Usage.CompletionTokens != 4 {
		t.Fatalf("usage chunk wrong: %s", events[6])
	}
	if !strings.Contains(events[6], `"choices":[]`) {
		t.Errorf("usage chunk must carry empty choices: %s", events[6])
	}
	if u.ID != "chatcmpl-"+reqID || u.Model != "model-a" {
		t.Errorf("usage chunk id/model %q/%q not rewritten", u.ID, u.Model)
	}
}

func TestStreamPreFirstByteErrorIsJSON(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=http500"})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))

	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	var eb errBody
	if err := json.Unmarshal(body, &eb); err != nil {
		t.Fatalf("body is not an error envelope: %v\n%s", err, body)
	}
	if eb.Error.Type != "upstream_error" {
		t.Errorf("type %q, want upstream_error", eb.Error.Type)
	}
	if got := e.mock.Stats().Requests; got != 1 {
		t.Errorf("upstream requests %d, want 1 (no retry)", got)
	}
}

func TestStreamTTFTTimeout(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;ttft_ms=2000", ttftMs: 100})
	start := time.Now()
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))

	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	var eb errBody
	if err := json.Unmarshal(body, &eb); err != nil || eb.Error.Code != "upstream_timeout" {
		t.Errorf("want upstream_timeout envelope, got %s (%v)", body, err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("took %v, want under 1s", d)
	}
	waitFor(t, time.Second, "mock sees the cancel", func() bool { return e.mock.Stats().Canceled == 1 })
}

func TestStreamMidstreamErrorEvent(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=midstream;fail_after=2"})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200 (headers were already sent)", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 3 {
		t.Fatalf("got %d events, want 2 chunks + 1 error: %q", len(events), events)
	}
	for _, ev := range events {
		if ev == "[DONE]" {
			t.Error("[DONE] sent after a failed stream")
		}
	}
	var eb errBody
	if err := json.Unmarshal([]byte(events[2]), &eb); err != nil {
		t.Fatalf("last event is not an error object: %v\n%s", err, events[2])
	}
	if eb.Error.Type != "upstream_error" || eb.Error.Code != "upstream_error" {
		t.Errorf("type/code %q/%q, want upstream_error/upstream_error", eb.Error.Type, eb.Error.Code)
	}
	if eb.Error.Message != "upstream dep-a failed mid-stream" {
		t.Errorf("message %q", eb.Error.Message)
	}
	if got := e.mock.Stats().Requests; got != 1 {
		t.Errorf("upstream requests %d, want exactly 1 (no retry after first byte)", got)
	}
	waitFor(t, time.Second, "request log line", func() bool { return strings.Contains(e.logs.String(), "error_code=upstream_error") })
	// The status line was sent before the failure, so the single request line
	// for this request records status 200 together with the error code.
	lines := strings.Split(strings.TrimSpace(e.logs.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("got %d log lines, want exactly 1: %q", len(lines), lines)
	}
	if !strings.Contains(lines[0], "status=200") || !strings.Contains(lines[0], "error_code=upstream_error") {
		t.Errorf("log line %q, want status=200 and error_code=upstream_error", lines[0])
	}
}

func TestStreamIdleTimeout(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=5;tpot_ms=1000", idleMs: 100})
	start := time.Now()
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 3 {
		t.Fatalf("got %d events, want 2 chunks + 1 error: %q", len(events), events)
	}
	var eb errBody
	if err := json.Unmarshal([]byte(events[2]), &eb); err != nil {
		t.Fatalf("last event is not an error object: %v\n%s", err, events[2])
	}
	if eb.Error.Type != "upstream_error" || eb.Error.Code != "upstream_timeout" {
		t.Errorf("type/code %q/%q, want upstream_error/upstream_timeout", eb.Error.Type, eb.Error.Code)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("took %v, want under 1s", d)
	}
	waitFor(t, time.Second, "mock sees the cancel", func() bool { return e.mock.Stats().Canceled == 1 })
	if got := e.mock.Stats().Requests; got != 1 {
		t.Errorf("upstream requests %d, want 1", got)
	}
}

func TestStreamSlowClientIsNotUpstreamIdle(t *testing.T) {
	// The upstream is never idle (tpot_ms=0), but the client stops reading for
	// longer than the idle timeout. Only upstream waits count toward the idle
	// watchdog, so the stream must still finish cleanly.
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=20000;tpot_ms=0", idleMs: 300})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	br := bufio.NewReader(resp.Body)
	if _, err := readEvent(br); err != nil {
		t.Fatalf("read first event: %v", err)
	}
	time.Sleep(1500 * time.Millisecond)

	var last string
	n := 1
	for {
		ev, err := readEvent(br)
		if err != nil {
			break
		}
		n++
		last = ev
		if strings.Contains(ev, `"error"`) {
			t.Fatalf("error event after %d events: %s", n, ev)
		}
	}
	if last != "[DONE]" {
		t.Errorf("last event %q after %d events, want [DONE]", last, n)
	}
	if got := e.mock.Stats().Canceled; got != 0 {
		t.Errorf("mock canceled %d, want 0 (a slow client is not an upstream fault)", got)
	}
}

func TestStreamRequestDeadlineMidStream(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=100;tpot_ms=50"})
	body := `{"model":"model-a","stream":true,"messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":200}}`
	resp := postStream(t, newStreamClient(), startGateway(t, e), body)

	events := readAll(t, resp)
	if len(events) < 2 {
		t.Fatalf("got %d events, want chunks then an error: %q", len(events), events)
	}
	last := events[len(events)-1]
	var eb errBody
	if err := json.Unmarshal([]byte(last), &eb); err != nil || eb.Error.Code != "upstream_timeout" {
		t.Errorf("last event should be an upstream_timeout error, got %s (%v)", last, err)
	}
	for _, ev := range events {
		if ev == "[DONE]" {
			t.Error("[DONE] sent after a deadline")
		}
	}
	waitFor(t, time.Second, "mock sees the cancel", func() bool { return e.mock.Stats().Canceled == 1 })
}

func TestStreamClientDisconnectCancelsUpstream(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=1000;tpot_ms=10"})
	url := startGateway(t, e)
	c := newStreamClient()

	// Warm up with a complete stream (model-b has no directives) so lazily
	// started goroutines exist before the baseline is taken.
	readAll(t, postStream(t, c, url, streamReqBody("model-b", false)))
	c.tr.CloseIdleConnections()
	baseline := runtime.NumGoroutine()

	resp := postStream(t, c, url, streamReqBody("model-a", false))
	br := bufio.NewReader(resp.Body)
	for i := 0; i < 3; i++ {
		if _, err := readEvent(br); err != nil {
			t.Fatalf("read event %d: %v", i, err)
		}
	}
	_ = resp.Body.Close()

	waitFor(t, time.Second, "upstream canceled", func() bool { return e.mock.Stats().Canceled == 1 })
	c.tr.CloseIdleConnections()
	waitFor(t, 2*time.Second, fmt.Sprintf("goroutines back to baseline %d", baseline), func() bool {
		return runtime.NumGoroutine() <= baseline+2
	})
	if got := e.mock.Stats().Requests; got != 2 {
		t.Errorf("upstream requests %d, want 2 (warm-up and test)", got)
	}
}

func TestStreamBackpressure(t *testing.T) {
	// The mock caps a stream at 100000 tokens, so that is the total here.
	const total = 100000
	e := newMockEnv(t, cfgOpts{upstreamA: fmt.Sprintf("m;tokens=%d;tpot_ms=0", total)})
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	br := bufio.NewReader(resp.Body)
	if _, err := readEvent(br); err != nil {
		t.Fatalf("read first event: %v", err)
	}

	// The client now stops reading. Socket buffers fill and the gateway must
	// stop pulling from the upstream instead of buffering without bound. How
	// long the buffers take to fill depends on machine load, so wait until the
	// producer has stalled: two samples 300ms apart are equal.
	stalled := waitForStall(t, e, 100*time.Millisecond, 300*time.Millisecond, 5*time.Second)

	// Then it must stay stalled while the client keeps not reading.
	time.Sleep(500 * time.Millisecond)
	after := e.mock.Stats().EventsWritten

	t.Logf("events written: %d when stalled, %d 500ms later, of %d", stalled, after, total)
	if after != stalled {
		t.Errorf("upstream kept writing after it stalled: %d -> %d", stalled, after)
	}
	if after >= total/2 {
		t.Errorf("upstream wrote %d of %d events while the client was paused, want under half", after, total)
	}

	_ = resp.Body.Close()
	waitFor(t, 2*time.Second, "upstream canceled", func() bool { return e.mock.Stats().Canceled == 1 })
}

// waitForStall polls the mock's written-event count every interval until the
// count is unchanged across a window, and returns that count. It fails the test
// if the producer never stalls within the deadline.
func waitForStall(t *testing.T, e *env, interval, window, deadline time.Duration) int64 {
	t.Helper()
	type sample struct {
		at time.Time
		n  int64
	}
	var history []sample
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		now := sample{time.Now(), e.mock.Stats().EventsWritten}
		history = append(history, now)
		for _, old := range history {
			if now.at.Sub(old.at) >= window && old.n == now.n {
				return now.n
			}
		}
		time.Sleep(interval)
	}
	t.Fatalf("upstream never stalled within %v (samples: %v)", deadline, history)
	return 0
}

// fakeStream replays chunks, then returns err.
type fakeStream struct {
	chunks []*openai.ChatCompletionChunk
	err    error
}

func (f *fakeStream) Recv() (*openai.ChatCompletionChunk, error) {
	if len(f.chunks) == 0 {
		return nil, f.err
	}
	c := f.chunks[0]
	f.chunks = f.chunks[1:]
	return c, nil
}

func (f *fakeStream) Close() error { return nil }

func TestStreamEmptyUpstreamIsJSONError(t *testing.T) {
	p := &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
		return &fakeStream{err: io.EOF}, nil
	}}
	gw, _ := newFakeGateway(t, cfgOpts{}, p)
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamReqBody("model-a", false))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	if got := decodeErr(t, rec).Error.Code; got != "upstream_error" {
		t.Errorf("code %q, want upstream_error", got)
	}
}

// waitFor polls cond until it holds or d elapses.
func waitFor(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %v waiting for: %s", d, what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// sseEvents returns the data of every event in a recorded SSE body.
func sseEvents(body string) []string {
	var events []string
	for _, block := range strings.Split(body, "\n\n") {
		if data, ok := strings.CutPrefix(strings.TrimSpace(block), "data: "); ok {
			events = append(events, data)
		}
	}
	return events
}

// twoDepModel is one model served by two deployments on the same upstream.
func twoDepModel(a, b string) []modelSpec {
	return []modelSpec{{id: "m", deps: []depSpec{{"dep-a", a}, {"dep-b", b}}}}
}

func TestStreamFailoverBeforeCommit(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: twoDepModel("ua;fail=http500", "ub;tokens=3")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	events := sseEvents(rec.Body.String())
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("stream does not end with [DONE]: %q", events)
	}
	if !strings.Contains(rec.Body.String(), "tok") {
		t.Errorf("dep-b content missing from the stream: %s", rec.Body.String())
	}
	if a, b := e.count("ua;fail=http500"), e.count("ub;tokens=3"); a != 1 || b != 1 {
		t.Errorf("upstream hits a=%d b=%d, want 1 and 1", a, b)
	}
	if !strings.Contains(e.logs.String(), "attempts=2") {
		t.Errorf("log line missing attempts=2: %s", e.logs.String())
	}
	if !strings.Contains(e.logs.String(), "deployment=dep-b") {
		t.Errorf("log line should name the committing deployment dep-b: %s", e.logs.String())
	}
}

func TestStreamTTFTTimeoutFailsOver(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{ttftMs: 100, models: twoDepModel("ua;ttft_ms=2000", "ub;tokens=3")})
	start := time.Now()
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("took %v, want under 1s", d)
	}
	if h := e.gw.health.Snapshot()["dep-a"]; h.ErrorRate == 0 {
		t.Errorf("dep-a health = %+v, want the TTFT timeout recorded as a failure", h)
	}
}

func TestStreamFailoverDiscardsHeldChunks(t *testing.T) {
	mockH := mock.NewHandler(mock.Config{})
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		if !strings.Contains(string(b), `"ua"`) {
			mockH.ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"ua","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"error":{"message":"boom"}}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	e := newAttemptEnv(t, attemptOpts{upstream: upstream, models: twoDepModel("ua", "ub;tokens=3")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	var roles, firstContent int
	firstContent = -1
	for i, ev := range sseEvents(rec.Body.String()) {
		if ev == "[DONE]" {
			continue
		}
		// created:1 is the marker of dep-a's held chunk; dep-b stamps its own.
		if strings.Contains(ev, `"created":1,`) || strings.HasSuffix(ev, `"created":1}`) {
			t.Errorf("event %d carries dep-a's held chunk: %s", i, ev)
		}
		c := decodeChunk(t, ev)
		for _, ch := range c.Choices {
			if ch.Delta.Role != "" {
				roles++
				if firstContent >= 0 {
					t.Errorf("role chunk at event %d follows content at event %d", i, firstContent)
				}
			}
			if ch.Delta.Content != nil && *ch.Delta.Content != "" && firstContent < 0 {
				firstContent = i
			}
		}
	}
	if roles != 1 {
		t.Errorf("client saw %d role chunks, want exactly 1 (dep-a's held chunk must be discarded)", roles)
	}
	if firstContent < 0 {
		t.Error("dep-b content never reached the client")
	}
}

// An upstream that sends only non-committing chunks cannot make the gateway
// buffer without bound: held chunks are capped in bytes as well as in count,
// and exceeding the byte cap is the same retryable protocol failure.
func TestStreamHeldBytesCapFailsOver(t *testing.T) {
	const (
		padBytes = 128 << 10 // per chunk, kept as an unknown top-level field
		nChunks  = 12        // 1.5 MiB in total, far fewer than maxHeldChunks
	)
	if nChunks >= maxHeldChunks || nChunks*padBytes <= maxHeldBytes {
		t.Fatalf("test setup must exceed only the byte cap: %d chunks of %d bytes", nChunks, padBytes)
	}
	heldChunk := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"ua","choices":[{"index":0,"delta":{"role":"assistant","content":""}}],"x_pad":"` +
		strings.Repeat("p", padBytes) + `"}`

	mockH := mock.NewHandler(mock.Config{})
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		if !strings.Contains(string(b), `"ua"`) {
			mockH.ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i := 0; i < nChunks; i++ {
			fmt.Fprint(w, "data: "+heldChunk+"\n\n")
		}
		fmt.Fprint(w, "data: "+contentChunk("late")+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	e := newAttemptEnv(t, attemptOpts{upstream: upstream, models: twoDepModel("ua", "ub;tokens=3")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 from dep-b: %.200s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	if strings.Contains(rec.Body.String(), "x_pad") {
		t.Error("dep-a's held chunks reached the client")
	}
	if a, b := e.count("ua"), e.count("ub;tokens=3"); a != 1 || b != 1 {
		t.Errorf("upstream hits a=%d b=%d, want 1 and 1", a, b)
	}
}

// An upstream that ends without any committing chunk is a retryable failure,
// not an empty 200 stream.
func TestStreamEmptyUpstreamFailsOver(t *testing.T) {
	cfg := twoProviderModelConfig(t)
	p := &fakeProvider{stream: func(_ context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error) {
		if req.Model == "ua" {
			return &fakeStream{err: io.EOF}, nil
		}
		return &fakeStream{chunks: []*openai.ChatCompletionChunk{textChunk("from-b")}, err: io.EOF}, nil
	}}
	gw, err := New(cfg, map[string]provider.Provider{"mockp": p}, Options{
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Sleep:  func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	if !strings.Contains(rec.Body.String(), "from-b") {
		t.Errorf("dep-b content missing: %s", rec.Body.String())
	}
	if h := gw.health.Snapshot()["dep-a"]; h.ErrorRate == 0 {
		t.Errorf("dep-a health = %+v, want the empty stream recorded as a failure", h)
	}
}

func TestStreamNoRetryAfterCommit(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: twoDepModel("ua;fail=midstream;fail_after=2", "ub;tokens=3")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 (headers were already sent): %s", rec.Code, rec.Body.String())
	}
	events := sseEvents(rec.Body.String())
	if len(events) != 3 {
		t.Fatalf("got %d events, want 2 chunks + 1 error: %q", len(events), events)
	}
	for _, ev := range events {
		if ev == "[DONE]" {
			t.Error("[DONE] sent after a failed stream")
		}
	}
	if !strings.Contains(events[2], "failed mid-stream") {
		t.Errorf("last event %q is not the mid-stream error", events[2])
	}
	if n := e.count("ub;tokens=3"); n != 0 {
		t.Errorf("dep-b saw %d requests, want 0 (no retry after commit)", n)
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-a" {
		t.Errorf("deployment header %q, want dep-a", got)
	}
}

func TestStreamAllAttemptsFailIsJSON(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: twoDepModel("ua;fail=http500", "ub;fail=http500")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	if got := decodeErr(t, rec).Error.Code; got != "upstream_error" {
		t.Errorf("code %q, want upstream_error", got)
	}
	if a, b := e.count("ua;fail=http500"), e.count("ub;fail=http500"); a != 1 || b != 1 {
		t.Errorf("upstream hits a=%d b=%d, want 1 and 1", a, b)
	}
	if strings.Contains(rec.Body.String(), "data: ") {
		t.Errorf("SSE written before commit: %s", rec.Body.String())
	}
}

func TestStreamClientDisconnectReleasesProbe(t *testing.T) {
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
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamBody("m"))).WithContext(ctx)
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

func TestStreamDeadlineAfterCommitRecordsSuccess(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	e := newAttemptEnv(t, attemptOpts{
		clock:  clock,
		models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;tokens=100;tpot_ms=50"}}}},
	})
	e.gw.health.Trip("dep-a", time.Second)
	mu.Lock()
	now = now.Add(2 * time.Second) // the cooldown elapsed: this request is the probe
	mu.Unlock()

	// The 200ms client deadline is far under the server default, so it is
	// short. After commit that still counts as a success, which closes the
	// breaker; releasing the claim as an abort would leave it open.
	body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":200}}`
	rec := e.post(body)

	if !strings.Contains(rec.Body.String(), "upstream_timeout") {
		t.Fatalf("stream did not end in the deadline error event: %s", rec.Body.String())
	}
	if h := e.gw.health.Snapshot()["dep-a"]; h.State != reliability.Closed || h.ErrorRate != 0 {
		t.Errorf("dep-a health = %+v, want closed with error rate 0 (success recorded)", h)
	}
}

// ignoringStream returns its chunks after the context is canceled, as a
// provider that does not honor cancellation promptly would.
type ignoringStream struct {
	ctx    context.Context
	chunks []*openai.ChatCompletionChunk
}

func (s *ignoringStream) Recv() (*openai.ChatCompletionChunk, error) {
	<-s.ctx.Done()
	if len(s.chunks) == 0 {
		return nil, io.EOF
	}
	c := s.chunks[0]
	s.chunks = s.chunks[1:]
	return c, nil
}

func (s *ignoringStream) Close() error { return nil }

func textChunk(text string) *openai.ChatCompletionChunk {
	return &openai.ChatCompletionChunk{Choices: []openai.ChunkChoice{{Delta: openai.Delta{Content: &text}}}}
}

// TestStreamCommitLosesRaceToTTFTWatchdog makes the TTFT watchdog fire before
// the first committing chunk is handled, deterministically: the first
// attempt's Recv blocks until the watchdog has canceled its context and then
// returns a content chunk anyway. The attempt must be a pre-commit timeout,
// so the request fails over instead of committing to dep-a.
func TestStreamCommitLosesRaceToTTFTWatchdog(t *testing.T) {
	cfg := twoProviderModelConfig(t)
	p := &fakeProvider{stream: func(ctx context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error) {
		if req.Model == "ua" {
			return &ignoringStream{ctx: ctx, chunks: []*openai.ChatCompletionChunk{textChunk("from-a")}}, nil
		}
		return &fakeStream{chunks: []*openai.ChatCompletionChunk{textChunk("from-b")}, err: io.EOF}, nil
	}}
	logs := &strings.Builder{}
	gw, err := New(cfg, map[string]provider.Provider{"mockp": p}, Options{
		Logger: slog.New(slog.NewTextHandler(logs, nil)),
		Sleep:  func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b (dep-a lost the TTFT race)", got)
	}
	if body := rec.Body.String(); strings.Contains(body, "from-a") || !strings.Contains(body, "from-b") {
		t.Errorf("body should carry only dep-b's content: %s", body)
	}
	if h := gw.health.Snapshot()["dep-a"]; h.ErrorRate == 0 {
		t.Errorf("dep-a health = %+v, want the TTFT timeout recorded as a failure", h)
	}
}

// twoProviderModelConfig is model m with deployments dep-a and dep-b on one
// provider with a 50 ms TTFT timeout, for tests that script the provider.
func twoProviderModelConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(fmt.Sprintf(`providers:
  - name: mockp
    type: openaicompat
    base_url: "http://127.0.0.1:1/v1"
    ttft_timeout_ms: 50
models:
  - id: m
    deployments:
      - name: dep-a
        provider: mockp
        upstream_model: ua
      - name: dep-b
        provider: mockp
        upstream_model: ub
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
`, keyHash(keyA))))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	return cfg
}

// A non-retryable pre-commit error ends the request with its own JSON error:
// the second deployment is never tried.
func TestStreamNonRetryableErrorBeforeCommitStops(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: twoDepModel("ua;fail=http400", "ub;tokens=3")})
	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	if strings.Contains(rec.Body.String(), "data: ") {
		t.Errorf("SSE written before commit: %s", rec.Body.String())
	}
	if n := e.count("ub;tokens=3"); n != 0 {
		t.Errorf("dep-b saw %d requests, want 0 (a 400 is not retried)", n)
	}
}

// Each attempt runs under its own provider's TTFT timeout: dep-a's provider
// gives up after 100 ms while dep-b's would wait 5 s.
func TestStreamPerProviderTTFTTimeout(t *testing.T) {
	srv := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(srv.Close)
	cfg, err := config.Parse([]byte(fmt.Sprintf(`providers:
  - name: fastp
    type: openaicompat
    base_url: %q
    ttft_timeout_ms: 100
  - name: slowp
    type: openaicompat
    base_url: %q
    ttft_timeout_ms: 5000
models:
  - id: m
    deployments:
      - name: dep-a
        provider: fastp
        upstream_model: "ua;ttft_ms=2000"
      - name: dep-b
        provider: slowp
        upstream_model: "ub;tokens=3"
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
`, srv.URL+"/v1", srv.URL+"/v1", keyHash(keyA))))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	gw, err := New(cfg, map[string]provider.Provider{
		"fastp": openaicompat.New(cfg.Providers[0], "", srv.Client()),
		"slowp": openaicompat.New(cfg.Providers[1], "", srv.Client()),
	}, Options{
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Sleep:  func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	start := time.Now()
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
		t.Errorf("deployment header %q, want dep-b", got)
	}
	if d := time.Since(start); d < 90*time.Millisecond || d > time.Second {
		t.Errorf("took %v, want about dep-a's 100 ms TTFT timeout and well under dep-b's 5 s", d)
	}
	if h := gw.health.Snapshot()["dep-a"]; h.ErrorRate == 0 {
		t.Errorf("dep-a health = %+v, want the TTFT timeout recorded as a failure", h)
	}
}

// lateStream blocks until the attempt context is canceled, waits a further
// delay, then returns content anyway, like a provider that ignores cancel.
type lateStream struct {
	ctx   context.Context
	delay time.Duration
	sent  bool
}

func (s *lateStream) Recv() (*openai.ChatCompletionChunk, error) {
	<-s.ctx.Done()
	time.Sleep(s.delay)
	if s.sent {
		return nil, io.EOF
	}
	s.sent = true
	return textChunk("late"), nil
}

func (s *lateStream) Close() error { return nil }

// TestStreamDeadlineBeforeTTFTThenLateContentIsTimeout covers Ruling M2-R15.
// The request deadline (100 ms) is shorter than the TTFT budget (300 ms), so
// the deadline cancels the attempt first. The provider then returns content
// after both have expired. The watchdog could not be stopped, so the TTFT
// budget ran out: this is a pre-commit timeout (504 JSON), not a 200 stream,
// and the deadline was short, so the half-open probe is released, not
// recorded as a success.
func TestStreamDeadlineBeforeTTFTThenLateContentIsTimeout(t *testing.T) {
	cfg, err := config.Parse([]byte(fmt.Sprintf(`providers:
  - name: mockp
    type: openaicompat
    base_url: "http://127.0.0.1:1/v1"
    ttft_timeout_ms: 300
models:
  - id: m
    deployments:
      - name: dep-a
        provider: mockp
        upstream_model: ua
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
`, keyHash(keyA))))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	p := &fakeProvider{stream: func(ctx context.Context, _ *openai.ChatCompletionRequest) (provider.Stream, error) {
		return &lateStream{ctx: ctx, delay: 350 * time.Millisecond}, nil
	}}
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	gw, err := New(cfg, map[string]provider.Provider{"mockp": p}, Options{
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Now:    clock,
		Sleep:  func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	gw.health.Trip("dep-a", time.Second)
	mu.Lock()
	now = now.Add(2 * time.Second) // the cooldown elapsed: this request is the probe
	mu.Unlock()

	body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":100}}`
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)

	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	if got := decodeErr(t, rec).Error.Code; got != "upstream_timeout" {
		t.Errorf("code %q, want upstream_timeout", got)
	}
	if strings.Contains(rec.Body.String(), "late") {
		t.Errorf("late content reached the client: %s", rec.Body.String())
	}
	h := gw.health.Snapshot()["dep-a"]
	if h.State == reliability.Closed {
		t.Errorf("dep-a health = %+v, want not closed (a success would have closed the breaker)", h)
	}
	if !gw.health.Available("dep-a") {
		t.Error("the half-open probe was left claimed")
	}
}
