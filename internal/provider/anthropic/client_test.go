package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/anthropic"
)

func newReq(model string) *openai.ChatCompletionRequest {
	text := "hello"
	return &openai.ChatCompletionRequest{
		Model:    model,
		Messages: []openai.Message{{Role: "user", Content: openai.Content{Text: &text}}},
	}
}

func newClient(baseURL, key string) *anthropic.Client {
	return anthropic.New(config.ProviderConfig{Name: "p", Type: config.ProviderAnthropic, BaseURL: baseURL}, key,
		&http.Client{Transport: provider.NewTransport()})
}

func mustProviderErr(t *testing.T, err error) *provider.Error {
	t.Helper()
	var pe *provider.Error
	if !errors.As(err, &pe) {
		t.Fatalf("error %v (%T) is not a *provider.Error", err, err)
	}
	return pe
}

func readFixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// sseServer serves body as an SSE response.
func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
}

func drain(t *testing.T, st provider.Stream) []*openai.ChatCompletionChunk {
	t.Helper()
	var chunks []*openai.ChatCompletionChunk
	for {
		ch, err := st.Recv()
		if errors.Is(err, io.EOF) {
			return chunks
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		chunks = append(chunks, ch)
	}
}

func TestCompleteSendsHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "test-key-a" {
			t.Errorf("x-api-key = %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q", got)
		}
		if got := r.Header.Get("content-type"); got != "application/json" {
			t.Errorf("content-type = %q", got)
		}
		if _, ok := r.Header["Authorization"]; ok {
			t.Error("Authorization header must not be sent")
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "claude-fixture-1" || body["max_tokens"] != float64(4096) {
			t.Errorf("body = %v", body)
		}
		if _, ok := body["stream"]; ok {
			t.Errorf("stream should not be set for Complete: %v", body["stream"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, readFixture(t, "message_tool.json"))
	}))
	defer srv.Close()

	// A trailing slash in base_url must not produce a double slash.
	if _, err := newClient(srv.URL+"/", "test-key-a").Complete(context.Background(), newReq("claude-fixture-1")); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteNoKeyNoHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := r.Header["X-Api-Key"]; ok {
			t.Errorf("x-api-key sent without a key: %q", r.Header.Get("x-api-key"))
		}
		_, _ = io.WriteString(w, readFixture(t, "message_tool.json"))
	}))
	defer srv.Close()
	if _, err := newClient(srv.URL, "").Complete(context.Background(), newReq("m")); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteToolUseResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, readFixture(t, "message_tool.json"))
	}))
	defer srv.Close()

	req := newReq("claude-fixture-1")
	before, _ := json.Marshal(req)
	got, err := newClient(srv.URL, "k").Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if after, _ := json.Marshal(req); string(before) != string(after) {
		t.Errorf("caller request mutated: %s -> %s", before, after)
	}
	c := got.Choices[0]
	if c.FinishReason != "tool_calls" || len(c.Message.ToolCalls) != 2 || c.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("unexpected choice: %+v", c)
	}
	if got.Usage == nil || got.Usage.PromptTokens != 125 || got.Usage.CompletionTokens != 40 || got.Usage.TotalTokens != 165 {
		t.Errorf("usage = %+v", got.Usage)
	}
}

func TestTranslationErrorMakesNoHTTPCall(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
	}))
	defer srv.Close()

	req := newReq("m")
	req.Stop = json.RawMessage(`42`)

	check := func(err error) {
		t.Helper()
		var apiErr *openai.APIError
		if !errors.As(err, &apiErr) || apiErr.Status != 400 || apiErr.Param != "stop" {
			t.Errorf("error = %v (%T), want 400 *openai.APIError param stop", err, err)
		}
	}
	c := newClient(srv.URL, "k")
	_, err := c.Complete(context.Background(), req)
	check(err)
	st, err := c.Stream(context.Background(), req)
	if st != nil {
		t.Error("expected nil stream on error")
	}
	check(err)
	if n := calls.Load(); n != 0 {
		t.Errorf("server received %d calls, want 0", n)
	}
}

func TestCompleteHTTPErrorClassified(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		retryAfter string
		retryable  bool
		after      time.Duration
	}{
		{"rate limited", 429, "2", true, 2 * time.Second},
		{"overloaded", 529, "", true, 0},
		{"bad request", 400, "", false, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"type":"error","error":{"type":"some_error","message":"upstream says no"}}`)
			}))
			defer srv.Close()
			_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))
			pe := mustProviderErr(t, err)
			if pe.Kind != provider.KindHTTP || pe.Status != tc.status || pe.Retryable != tc.retryable ||
				pe.RetryAfter != tc.after || pe.Message != "upstream says no" {
				t.Errorf("unexpected error: %+v", pe)
			}
		})
	}
}

func TestStreamPreStreamOverloaded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(529)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if st != nil {
		t.Error("expected nil stream on error")
	}
	if pe := mustProviderErr(t, err); pe.Status != 529 || !pe.Retryable || pe.Kind != provider.KindHTTP {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestCompleteBadSuccessBodies(t *testing.T) {
	cases := map[string]func(w http.ResponseWriter){
		"invalid json": func(w http.ResponseWriter) { _, _ = io.WriteString(w, "not json") },
		"oversized": func(w http.ResponseWriter) {
			_, _ = io.WriteString(w, `{"id":"`)
			chunk := []byte(strings.Repeat("a", 1<<20))
			for i := 0; i < 17; i++ {
				if _, err := w.Write(chunk); err != nil {
					return
				}
			}
			_, _ = io.WriteString(w, `"}`)
		},
	}
	for name, write := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { write(w) }))
			defer srv.Close()
			_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))
			if pe := mustProviderErr(t, err); pe.Kind != provider.KindProtocol || pe.Retryable {
				t.Errorf("unexpected error: %+v", pe)
			}
		})
	}
}

func TestCompleteConnectErrorAndCancel(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	_, err := newClient(url, "k").Complete(context.Background(), newReq("m"))
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindConnect || !pe.Retryable {
		t.Errorf("unexpected connect error: %+v", pe)
	}

	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer slow.Close()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err = newClient(slow.URL, "k").Complete(ctx, newReq("m"))
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindCanceled || pe.Retryable {
		t.Errorf("unexpected cancel error: %+v", pe)
	}
}

func TestStreamTextFixture(t *testing.T) {
	var sawStream bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		sawStream = body.Stream
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, readFixture(t, "stream_text.sse"))
	}))
	defer srv.Close()

	req := newReq("claude-fixture-1")
	st, err := newClient(srv.URL, "k").Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chunks := drain(t, st)

	if !sawStream || req.Stream {
		t.Errorf("server stream=%v caller stream=%v, want true/false", sawStream, req.Stream)
	}
	// role, 3 content, finish, usage
	if len(chunks) != 6 {
		t.Fatalf("got %d chunks, want 6", len(chunks))
	}
	for _, ch := range chunks {
		if ch.ID != "msg_01TextFixture" || ch.Model != "claude-fixture-1" || ch.Object != openai.ObjectChatCompletionChunk || ch.Created == 0 {
			t.Errorf("bad chunk header: %+v", ch)
		}
	}
	if d := chunks[0].Choices[0].Delta; d.Role != "assistant" {
		t.Errorf("first delta = %+v, want role assistant", d)
	}
	var text strings.Builder
	for _, ch := range chunks[1:4] {
		if c := ch.Choices[0].Delta.Content; c != nil {
			text.WriteString(*c)
		}
	}
	if text.String() != "Hello, world!" {
		t.Errorf("text = %q", text.String())
	}
	fin := chunks[4].Choices[0]
	if fin.FinishReason == nil || *fin.FinishReason != "stop" {
		t.Errorf("finish chunk = %+v", chunks[4])
	}
	last := chunks[5]
	if last.Choices == nil || len(last.Choices) != 0 {
		t.Errorf("usage chunk choices = %#v, want empty non-nil", last.Choices)
	}
	if b, _ := json.Marshal(last); !strings.Contains(string(b), `"choices":[]`) {
		t.Errorf("usage chunk JSON = %s", b)
	}
	if u := last.Usage; u == nil || u.PromptTokens != 25 || u.CompletionTokens != 15 || u.TotalTokens != 40 {
		t.Errorf("usage = %+v", last.Usage)
	}
	// io.EOF is sticky after a clean end.
	if _, err := st.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("Recv after EOF = %v, want io.EOF", err)
	}
	if err := st.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestStreamToolFixture(t *testing.T) {
	srv := sseServer(t, readFixture(t, "stream_tool.sse"))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("claude-fixture-1"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chunks := drain(t, st)

	type call struct {
		id, name string
		args     strings.Builder
	}
	calls := map[int]*call{}
	var finish string
	var usage *openai.Usage
	for _, ch := range chunks {
		if ch.Usage != nil {
			usage = ch.Usage
		}
		for _, c := range ch.Choices {
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
			for _, tc := range c.Delta.ToolCalls {
				if tc.Index == nil {
					t.Fatalf("tool call delta without index: %+v", tc)
				}
				cl := calls[*tc.Index]
				if cl == nil {
					cl = &call{}
					calls[*tc.Index] = cl
				}
				if tc.ID != "" {
					if tc.Type != "function" {
						t.Errorf("tool call type = %q", tc.Type)
					}
					cl.id, cl.name = tc.ID, tc.Function.Name
					if tc.Function.Arguments != "" {
						t.Errorf("start chunk arguments = %q, want empty", tc.Function.Arguments)
					}
				}
				cl.args.WriteString(tc.Function.Arguments)
			}
		}
	}
	// Tool-call indices count tool_use blocks only: the text block does not take index 0.
	if len(calls) != 2 || calls[0] == nil || calls[1] == nil {
		t.Fatalf("calls = %v, want indices 0 and 1", calls)
	}
	if calls[0].id != "toolu_01FixtureA" || calls[0].name != "get_weather" || calls[1].id != "toolu_01FixtureB" || calls[1].name != "get_time" {
		t.Errorf("calls = %+v %+v", calls[0], calls[1])
	}
	for i, want := range map[int]string{0: `{"city": "Paris"}`, 1: `{"tz":"UTC"}`} {
		got := calls[i].args.String()
		if got != want || !json.Valid([]byte(got)) {
			t.Errorf("call %d arguments = %q, want %q", i, got, want)
		}
	}
	if finish != "tool_calls" {
		t.Errorf("finish = %q", finish)
	}
	if usage == nil || usage.PromptTokens != 480 || usage.CompletionTokens != 89 || usage.TotalTokens != 569 {
		t.Errorf("usage = %+v", usage)
	}
}

func TestStreamZeroInputToolGetsEmptyObjectArguments(t *testing.T) {
	body := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"ping","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}` + "\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"pong","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}` + "\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}` + "\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	srv := sseServer(t, body)
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	args := map[int]*strings.Builder{}
	for _, ch := range drain(t, st) {
		for _, c := range ch.Choices {
			for _, tc := range c.Delta.ToolCalls {
				if args[*tc.Index] == nil {
					args[*tc.Index] = &strings.Builder{}
				}
				args[*tc.Index].WriteString(tc.Function.Arguments)
			}
		}
	}
	if len(args) != 2 || args[0].String() != "{}" || args[1].String() != `{"a":1}` {
		t.Errorf("arguments = %q / %q, want {} and {\"a\":1}", args[0], args[1])
	}
}

func TestStreamMessageDeltaWithoutUsageKeepsOutputTokens(t *testing.T) {
	body := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":2}}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{},"usage":{"output_tokens":7}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}` + "\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	srv := sseServer(t, body)
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chunks := drain(t, st)
	u := chunks[len(chunks)-1].Usage
	if u == nil || u.CompletionTokens != 7 || u.TotalTokens != 9 {
		t.Errorf("usage = %+v, want completion 7 total 9", u)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	body := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1}}}` + "\n\n" +
		"event: error\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n"
	srv := sseServer(t, body)
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	_, err = st.Recv()
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindProtocol || pe.Retryable || !strings.Contains(pe.Message, "overloaded_error") {
		t.Errorf("unexpected error: %+v", pe)
	}
	// The error is sticky.
	if _, err2 := st.Recv(); err2 != err {
		t.Errorf("second Recv = %v, want the same sticky error", err2)
	}
}

func TestStreamProtocolFailures(t *testing.T) {
	start := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1}}}` + "\n\n"
	cases := map[string]string{
		"eof before message_stop": start,
		"invalid event json":      start + "event: content_block_delta\ndata: {not json\n\n",
		"delta for unknown block": start + "event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{}"}}` + "\n\n",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			srv := sseServer(t, body)
			defer srv.Close()
			st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
			if err != nil {
				t.Fatal(err)
			}
			defer st.Close()
			if _, err := st.Recv(); err != nil {
				t.Fatalf("first Recv: %v", err)
			}
			_, err = st.Recv()
			if pe := mustProviderErr(t, err); pe.Kind != provider.KindProtocol || pe.Retryable {
				t.Errorf("unexpected error: %+v", pe)
			}
		})
	}
}

func TestStreamIgnoresUnknownEvents(t *testing.T) {
	body := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":2}}}` + "\n\n" +
		"event: future_event\ndata: {\"anything\":true}\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":3}}` + "\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	srv := sseServer(t, body)
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chunks := drain(t, st)
	// role, finish (pass-through reason), usage
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	if f := chunks[1].Choices[0].FinishReason; f == nil || *f != "refusal" {
		t.Errorf("finish = %v, want refusal passthrough", f)
	}
	if u := chunks[2].Usage; u == nil || u.TotalTokens != 5 {
		t.Errorf("usage = %+v", u)
	}
}

func TestStreamCloseUnblocksRecvAsCanceled(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() { _, err := st.Recv(); errc <- err }()
	time.Sleep(50 * time.Millisecond)
	_ = st.Close()
	select {
	case err := <-errc:
		if pe := mustProviderErr(t, err); pe.Kind != provider.KindCanceled {
			t.Errorf("Kind = %s, want canceled", pe.Kind)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Recv did not unblock after Close")
	}
	if _, err := st.Recv(); mustProviderErr(t, err).Kind != provider.KindCanceled {
		t.Errorf("Recv after Close = %v, want canceled", err)
	}
}
