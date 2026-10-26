package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
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
	for k, want := range map[string]string{
		"Content-Type":            "text/event-stream",
		"Cache-Control":           "no-cache",
		"X-Accel-Buffering":       "no",
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
	// stop pulling from the upstream instead of buffering without bound.
	time.Sleep(500 * time.Millisecond)
	at500 := e.mock.Stats().EventsWritten
	time.Sleep(500 * time.Millisecond)
	at1000 := e.mock.Stats().EventsWritten

	t.Logf("events written: %d at 500ms, %d at 1000ms of %d", at500, at1000, total)
	if grew := at1000 - at500; grew >= 1000 {
		t.Errorf("upstream kept writing during the pause: %d events in 500ms (%d -> %d)", grew, at500, at1000)
	}
	if at1000 >= total/2 {
		t.Errorf("upstream wrote %d of %d events while the client was paused, want under half", at1000, total)
	}

	_ = resp.Body.Close()
	waitFor(t, 2*time.Second, "upstream canceled", func() bool { return e.mock.Stats().Canceled == 1 })
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
