package mock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
)

func chatBody(model, extra string) string {
	return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]%s}`, model, extra)
}

func newServer(t *testing.T, cfg Config) (*Handler, *httptest.Server) {
	t.Helper()
	h := NewHandler(cfg)
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return h, srv
}

func post(t *testing.T, ctx context.Context, url, body string, hdr map[string]string) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	return http.DefaultClient.Do(req)
}

func mustPost(t *testing.T, url, body string, hdr map[string]string) *http.Response {
	t.Helper()
	resp, err := post(t, context.Background(), url, body, hdr)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestMockCompletionUsage(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	body := `{"model":"m;tokens=5;ttft_ms=0;tpot_ms=0","messages":[` +
		`{"role":"system","content":"abcd"},{"role":"user","content":"hello world"}]}`
	resp := mustPost(t, srv.URL, body, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q", ct)
	}
	var got openai.ChatCompletion
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Object != openai.ObjectChatCompletion || got.Model != "m" {
		t.Fatalf("object=%q model=%q", got.Object, got.Model)
	}
	if !strings.HasPrefix(got.ID, "chatcmpl-mock-") || got.Created == 0 {
		t.Fatalf("id=%q created=%d", got.ID, got.Created)
	}
	if len(got.Choices) != 1 || got.Choices[0].FinishReason != "stop" || got.Choices[0].Message.Role != "assistant" {
		t.Fatalf("choices = %+v", got.Choices)
	}
	if text := got.Choices[0].Message.Content.PlainText(); text != "tok0 tok1 tok2 tok3 tok4" {
		t.Fatalf("content = %q", text)
	}
	// 15 content bytes -> ceil(15/4) = 4, plus 4 per message (2) = 12.
	want := openai.Usage{PromptTokens: 12, CompletionTokens: 5, TotalTokens: 17}
	if got.Usage == nil || *got.Usage != want {
		t.Fatalf("usage = %+v, want %+v", got.Usage, want)
	}
	if s := h.Stats(); s.Requests != 1 || s.Completed != 1 || s.Canceled != 0 {
		t.Fatalf("stats = %+v", s)
	}
}

func readEvents(t *testing.T, r io.Reader) ([]openai.Event, error) {
	t.Helper()
	sr := openai.NewSSEReader(r, 0)
	var events []openai.Event
	for {
		ev, err := sr.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return events, nil
			}
			return events, err
		}
		events = append(events, ev)
	}
}

func TestMockStreamEvents(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	body := chatBody("m;tokens=5;ttft_ms=0;tpot_ms=0", `,"stream":true,"stream_options":{"include_usage":true}`)
	resp := mustPost(t, srv.URL, body, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q", ct)
	}
	events, err := readEvents(t, resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 9 || !events[8].IsDone() {
		t.Fatalf("got %d events, want 8 chunks + [DONE]", len(events))
	}
	chunks := make([]openai.ChatCompletionChunk, 8)
	for i := range chunks {
		if err := json.Unmarshal(events[i].Data, &chunks[i]); err != nil {
			t.Fatalf("event %d: %v: %s", i, err, events[i].Data)
		}
		if chunks[i].Object != openai.ObjectChatCompletionChunk || chunks[i].Model != "m" || chunks[i].ID != chunks[0].ID {
			t.Fatalf("event %d header = %+v", i, chunks[i])
		}
	}
	if d := chunks[0].Choices[0].Delta; d.Role != "assistant" {
		t.Fatalf("first delta = %+v", d)
	}
	var text strings.Builder
	for i := 1; i <= 5; i++ {
		c := chunks[i].Choices[0].Delta.Content
		if c == nil {
			t.Fatalf("chunk %d has no content", i)
		}
		text.WriteString(*c)
	}
	if text.String() != "tok0 tok1 tok2 tok3 tok4" {
		t.Fatalf("streamed text = %q", text.String())
	}
	if fr := chunks[6].Choices[0].FinishReason; fr == nil || *fr != "stop" {
		t.Fatalf("finish chunk = %+v", chunks[6])
	}
	if chunks[7].Usage == nil || chunks[7].Usage.CompletionTokens != 5 || chunks[7].Choices == nil || len(chunks[7].Choices) != 0 {
		t.Fatalf("usage chunk = %s", events[7].Data)
	}
	if !bytes.Contains(events[7].Data, []byte(`"choices":[]`)) {
		t.Fatalf("usage chunk choices must be an empty array: %s", events[7].Data)
	}
	waitFor(t, "stream completion", func() bool { return h.Stats().Completed == 1 })
	if s := h.Stats(); s.EventsWritten != 8 || s.Requests != 1 || s.Canceled != 0 {
		t.Fatalf("stats = %+v", s)
	}
}

func TestMockStreamWithoutUsage(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	resp := mustPost(t, srv.URL, chatBody("m;tokens=2;ttft_ms=0;tpot_ms=0", `,"stream":true`), nil)
	events, err := readEvents(t, resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	// role + 2 tokens + finish + [DONE]
	if len(events) != 5 || !events[4].IsDone() {
		t.Fatalf("got %d events", len(events))
	}
}

func TestMockStreamHeadersBeforeTTFT(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	start := time.Now()
	resp := mustPost(t, srv.URL, chatBody("m;tokens=2;ttft_ms=300;tpot_ms=0", `,"stream":true`), nil)
	if d := time.Since(start); d > 200*time.Millisecond {
		t.Fatalf("headers took %v, want them before TTFT", d)
	}
	sr := openai.NewSSEReader(resp.Body, 0)
	if _, err := sr.Next(); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < 300*time.Millisecond {
		t.Fatalf("first event after %v, want at least the 300ms TTFT", d)
	}
}

func TestMockNonStreamSleepsTTFTPlusTokens(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	start := time.Now()
	resp := mustPost(t, srv.URL, chatBody("m;tokens=4;ttft_ms=100;tpot_ms=50", ""), nil)
	_, _ = io.Copy(io.Discard, resp.Body)
	if d := time.Since(start); d < 300*time.Millisecond {
		t.Fatalf("took %v, want at least 100ms + 4*50ms", d)
	}
}

func TestMockDirectivesFromModelAndHeader(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})

	resp := mustPost(t, srv.URL, chatBody("m;fail=http500", ""), nil)
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("model directive: status = %d", resp.StatusCode)
	}
	var env struct {
		Error struct{ Type, Code string }
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil || env.Error.Type != "server_error" {
		t.Fatalf("500 body: %+v err=%v", env, err)
	}

	resp = mustPost(t, srv.URL, chatBody("m", ""), map[string]string{"x-mock-fail": "http429"})
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("header directive: status = %d", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "1" {
		t.Fatalf("Retry-After = %q, want 1", ra)
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil || env.Error.Type != "rate_limit_error" {
		t.Fatalf("429 body: %+v err=%v", env, err)
	}

	// The model directive beats the header.
	resp = mustPost(t, srv.URL, chatBody("m;fail=http400", ""), map[string]string{"x-mock-fail": "http500"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("precedence: status = %d, want 400", resp.StatusCode)
	}

	// retry_after is configurable.
	resp = mustPost(t, srv.URL, chatBody("m;fail=http429;retry_after=7", ""), nil)
	if ra := resp.Header.Get("Retry-After"); resp.StatusCode != 429 || ra != "7" {
		t.Fatalf("status=%d Retry-After=%q", resp.StatusCode, ra)
	}

	// Header-supplied tokens apply when the model has none.
	resp = mustPost(t, srv.URL, chatBody("m", ""), map[string]string{"x-mock-tokens": "2", "x-mock-ttft_ms": "0", "x-mock-tpot_ms": "0"})
	var got openai.ChatCompletion
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Usage.CompletionTokens != 2 {
		t.Fatalf("completion tokens = %d, want 2", got.Usage.CompletionTokens)
	}
}

func TestMockHTMLBadGateway(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	resp := mustPost(t, srv.URL, chatBody("m;fail=html502", ""), nil)
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadGateway || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		t.Fatalf("status=%d content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if string(b) != "<html><body>502 Bad Gateway</body></html>" {
		t.Fatalf("body = %q", b)
	}
}

func TestMockBadDirectives(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	tests := []struct {
		name, model string
		hdr         map[string]string
		wantInMsg   string
	}{
		{"unknown key", "m;bogus=1", nil, "bogus"},
		{"bad fail value", "m;fail=nope", nil, "fail"},
		{"bad tokens", "m;tokens=abc", nil, "tokens"},
		{"negative tokens", "m;tokens=-1", nil, "tokens"},
		{"bad float", "m;ttft_ms=fast", nil, "ttft_ms"},
		{"missing equals", "m;fail", nil, "fail"},
		{"unknown header key", "m", map[string]string{"x-mock-wat": "1"}, "wat"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := mustPost(t, srv.URL, chatBody(tc.model, ""), tc.hdr)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d", resp.StatusCode)
			}
			var env struct {
				Error struct{ Message, Type string }
			}
			if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
				t.Fatal(err)
			}
			if env.Error.Type != "invalid_request_error" || !strings.Contains(env.Error.Message, tc.wantInMsg) {
				t.Fatalf("error = %+v", env.Error)
			}
		})
	}
	if s := h.Stats(); s.Requests != int64(len(tests)) || s.Completed != 0 {
		t.Fatalf("stats = %+v", s)
	}
}

func TestMockInvalidRequestBody(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	resp := mustPost(t, srv.URL, `{"model":`, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestMockRoutes(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	want := `{"object":"list","data":[{"id":"mock","object":"model","created":0,"owned_by":"rp-mock"}]}`
	if resp.StatusCode != 200 || strings.TrimSpace(string(b)) != want {
		t.Fatalf("models: status=%d body=%s", resp.StatusCode, b)
	}

	r2, err := http.Get(srv.URL + "/nope")
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Body.Close()
	if r2.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown path status = %d", r2.StatusCode)
	}

	r3, err := http.Get(srv.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer r3.Body.Close()
	if r3.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET chat status = %d", r3.StatusCode)
	}
}

func TestMockStallCanceled(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	resp, err := post(t, ctx, srv.URL, chatBody("m;fail=stall", ""), nil)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatal("expected the client timeout")
	}
	waitFor(t, "Canceled == 1", func() bool { return h.Stats().Canceled == 1 })
	if s := h.Stats(); s.Completed != 0 {
		t.Fatalf("stats = %+v", s)
	}
}

func TestMockStallStreamSendsHeadersFirst(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp, err := post(t, ctx, srv.URL, chatBody("m;fail=stall", `,"stream":true`), nil)
	if err != nil {
		t.Fatalf("headers should arrive before the stall: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	cancel()
	waitFor(t, "Canceled == 1", func() bool { return h.Stats().Canceled == 1 })
}

func TestMockCancelDuringTTFT(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	resp, err := post(t, ctx, srv.URL, chatBody("m;ttft_ms=1500", ""), nil)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatal("expected the client timeout")
	}
	waitFor(t, "Canceled == 1", func() bool { return h.Stats().Canceled == 1 })
}

func TestMockMidstream(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	resp := mustPost(t, srv.URL, chatBody("m;fail=midstream;fail_after=2;tokens=5;ttft_ms=0;tpot_ms=0", `,"stream":true`), nil)
	events, err := readEvents(t, resp.Body)
	if err == nil {
		t.Fatal("expected the connection to break mid-stream")
	}
	if len(events) != 2 {
		t.Fatalf("got %d events, want exactly 2", len(events))
	}
	for _, ev := range events {
		if ev.IsDone() {
			t.Fatal("must not send [DONE]")
		}
	}
	if s := h.Stats(); s.Completed != 0 || s.EventsWritten != 2 {
		t.Fatalf("stats = %+v", s)
	}
}

func TestMockReset(t *testing.T) {
	_, srv := newServer(t, Config{Seed: 1})
	resp, err := post(t, context.Background(), srv.URL, chatBody("m;fail=reset", ""), nil)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatalf("expected a transport error, got status %d", resp.StatusCode)
	}
}

func countStatus500(t *testing.T, seed int64, n int) int {
	t.Helper()
	h := NewHandler(Config{Seed: seed, FailRates: map[string]float64{"http500": 0.3}})
	body := chatBody("m;ttft_ms=0;tpot_ms=0;tokens=1", "")
	count := 0
	for i := 0; i < n; i++ {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		h.ServeHTTP(rec, req)
		switch rec.Code {
		case http.StatusInternalServerError:
			count++
		case http.StatusOK:
		default:
			t.Fatalf("unexpected status %d", rec.Code)
		}
	}
	return count
}

func TestMockFailRatesSeeded(t *testing.T) {
	a := countStatus500(t, 7, 1000)
	b := countStatus500(t, 7, 1000)
	if a != b {
		t.Fatalf("same seed gave %d and %d failures", a, b)
	}
	if a < 250 || a > 350 {
		t.Fatalf("failures = %d, want within [250,350]", a)
	}
}

func TestMockExplicitDirectiveSkipsFailRates(t *testing.T) {
	h := NewHandler(Config{Seed: 7, FailRates: map[string]float64{"http500": 1}})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(chatBody("m;fail=http429", "")))
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want the explicit 429", rec.Code)
	}
}

func TestDistSampleDeterministic(t *testing.T) {
	d := Dist{MedianMs: 100, Sigma: 0.5}
	r1 := rand.New(rand.NewPCG(1, 2))
	r2 := rand.New(rand.NewPCG(1, 2))
	for i := 0; i < 20; i++ {
		a, b := d.Sample(r1), d.Sample(r2)
		if a != b {
			t.Fatalf("sample %d differs: %v vs %v", i, a, b)
		}
		if a <= 0 {
			t.Fatalf("sample %d = %v, want positive", i, a)
		}
	}

	c := Dist{MedianMs: 40}
	if got := c.Sample(r1); got != 40*time.Millisecond {
		t.Fatalf("Sigma 0 sample = %v, want exactly 40ms", got)
	}
	if got := (Dist{}).Sample(r1); got != 0 {
		t.Fatalf("zero dist sample = %v", got)
	}
}

func TestDistSampleMedian(t *testing.T) {
	d := Dist{MedianMs: 100, Sigma: 0.4}
	r := rand.New(rand.NewPCG(3, 4))
	below := 0
	const n = 4000
	for i := 0; i < n; i++ {
		if d.Sample(r) < 100*time.Millisecond {
			below++
		}
	}
	if frac := float64(below) / n; math.Abs(frac-0.5) > 0.05 {
		t.Fatalf("fraction below median = %.3f, want about 0.5", frac)
	}
}
