package openaicompat_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

const completionJSON = `{"id":"c1","object":"chat.completion","created":1,"model":"upstream-m",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`

func newReq(model string) *openai.ChatCompletionRequest {
	text := "hello"
	return &openai.ChatCompletionRequest{
		Model:    model,
		Messages: []openai.Message{{Role: "user", Content: openai.Content{Text: &text}}},
	}
}

func newClient(baseURL, key string) *openaicompat.Client {
	return openaicompat.New(config.ProviderConfig{Name: "p", Type: config.ProviderOpenAICompat, BaseURL: baseURL}, key,
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

func TestCompleteSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key-a" {
			t.Errorf("Authorization = %q", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "upstream-m" {
			t.Errorf("model = %v", body["model"])
		}
		if _, ok := body["stream"]; ok {
			t.Errorf("stream should not be set for Complete: %v", body["stream"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completionJSON)
	}))
	defer srv.Close()

	// A trailing slash in base_url must not produce a double slash.
	c := newClient(srv.URL+"/v1/", "test-key-a")
	got, err := c.Complete(context.Background(), newReq("upstream-m"))
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "c1" || got.Usage == nil || got.Usage.TotalTokens != 5 || got.Choices[0].Message.Content.PlainText() != "hi" {
		t.Errorf("unexpected completion: %+v", got)
	}
}

func TestCompleteDoesNotMutateRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, completionJSON)
	}))
	defer srv.Close()
	req := newReq("upstream-m")
	if _, err := newClient(srv.URL, "k").Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if req.Stream || req.StreamOptions != nil {
		t.Errorf("request mutated: %+v", req)
	}
}

func TestCompleteNoKeyNoAuthHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := r.Header["Authorization"]; ok {
			t.Errorf("Authorization header sent without a key: %q", r.Header.Get("Authorization"))
		}
		_, _ = io.WriteString(w, completionJSON)
	}))
	defer srv.Close()
	if _, err := newClient(srv.URL, "").Complete(context.Background(), newReq("m")); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteForwardsUnknownFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&body)
		if got := string(body["logit_bias"]); got != `{"50256":-100}` {
			t.Errorf("logit_bias = %q", got)
		}
		_, _ = io.WriteString(w, completionJSON)
	}))
	defer srv.Close()
	req := newReq("m")
	req.Extra = map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"50256":-100}`)}
	if _, err := newClient(srv.URL, "k").Complete(context.Background(), req); err != nil {
		t.Fatal(err)
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
		{"bad request", 400, "", false, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"error":{"message":"upstream says no","type":"x"}}`)
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

func TestCompleteNonJSONErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(502)
		_, _ = io.WriteString(w, "<html>bad gateway</html>")
	}))
	defer srv.Close()
	_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))
	pe := mustProviderErr(t, err)
	if pe.Message != "Bad Gateway" || strings.Contains(err.Error(), "<html>") {
		t.Errorf("Message = %q, err = %q", pe.Message, err.Error())
	}
}

func TestCompleteInvalidSuccessBodyIsProtocolError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "not json")
	}))
	defer srv.Close()
	_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindProtocol || pe.Retryable {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestCompleteOversizedSuccessBodyIsProtocolError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"id":"`)
		chunk := []byte(strings.Repeat("a", 1<<20))
		for i := 0; i < 17; i++ {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
		_, _ = io.WriteString(w, `"}`)
	}))
	defer srv.Close()
	_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindProtocol {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestCompleteConnectError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	_, err := newClient(url, "k").Complete(context.Background(), newReq("m"))
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindConnect || !pe.Retryable {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestCompleteContextCanceled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body) // lets the server notice the client going away
		<-r.Context().Done()
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err := newClient(srv.URL, "k").Complete(ctx, newReq("m"))
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindCanceled || pe.Retryable {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func writeSSE(w http.ResponseWriter, data string) {
	_, _ = io.WriteString(w, "data: "+data+"\n\n")
	w.(http.Flusher).Flush()
}

func chunkJSON(content string) string {
	b, _ := json.Marshal(map[string]any{
		"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "upstream-m",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": content}, "finish_reason": nil}},
	})
	return string(b)
}

func TestStreamChunksAndUsage(t *testing.T) {
	var sawUsage bool
	var sawStream bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream        bool `json:"stream"`
			StreamOptions struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		sawUsage, sawStream = body.StreamOptions.IncludeUsage, body.Stream
		w.Header().Set("Content-Type", "text/event-stream")
		for _, s := range []string{"a", "b", "c"} {
			writeSSE(w, chunkJSON(s))
		}
		writeSSE(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"upstream-m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}`)
		writeSSE(w, "[DONE]")
	}))
	defer srv.Close()

	req := newReq("upstream-m")
	st, err := newClient(srv.URL, "k").Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	var chunks []*openai.ChatCompletionChunk
	for {
		ch, err := st.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, ch)
	}
	if len(chunks) != 4 {
		t.Fatalf("got %d chunks, want 4", len(chunks))
	}
	if last := chunks[3]; last.Usage == nil || last.Usage.TotalTokens != 6 {
		t.Errorf("usage chunk = %+v", last)
	}
	if !sawStream || !sawUsage {
		t.Errorf("server saw stream=%v include_usage=%v, want both true", sawStream, sawUsage)
	}
	if req.Stream || req.StreamOptions != nil {
		t.Errorf("caller request mutated: %+v", req)
	}
	// io.EOF is sticky after [DONE].
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

func TestStreamPreStreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
		_, _ = io.WriteString(w, `{"error":{"message":"overloaded"}}`)
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if st != nil {
		t.Error("expected nil stream on error")
	}
	pe := mustProviderErr(t, err)
	if pe.Status != 503 || pe.Kind != provider.KindHTTP || !pe.Retryable || pe.Message != "overloaded" {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestStreamMidStreamDisconnect(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(w, chunkJSON("a"))
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		_ = conn.(*net.TCPConn).SetLinger(0) // RST instead of a clean FIN
		_ = conn.Close()
	}))
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
	if errors.Is(err, io.EOF) {
		t.Fatal("mid-stream disconnect surfaced as io.EOF")
	}
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindProtocol && pe.Kind != provider.KindConnect {
		t.Errorf("Kind = %s, want protocol or connect", pe.Kind)
	}
}

func TestStreamEndsWithoutDoneIsProtocolError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, chunkJSON("a"))
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatal(err)
	}
	_, err = st.Recv()
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindProtocol || pe.Retryable {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestStreamInvalidChunkIsProtocolError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, "{not json")
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	_, err = st.Recv()
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindProtocol || pe.Retryable {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestStreamOversizedEventIsProtocolError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: ")
		chunk := []byte(strings.Repeat("a", 1<<16))
		for i := 0; i < 20; i++ { // > MaxEventBytes (1 MiB)
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
		_, _ = io.WriteString(w, "\n\n")
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	_, err = st.Recv()
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindProtocol {
		t.Errorf("unexpected error: %+v", pe)
	}
}

func TestStreamRecvAfterClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		writeSSE(w, chunkJSON("a"))
		<-r.Context().Done()
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Recv(); err != nil {
		t.Fatal(err)
	}
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Recv(); err == nil || errors.Is(err, io.EOF) {
		t.Errorf("Recv after Close = %v, want a non-EOF error", err)
	}
}

func TestStreamCloseDuringRecvIsCanceled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		writeSSE(w, chunkJSON("a"))
		<-r.Context().Done()
	}))
	defer srv.Close()
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Recv(); err != nil {
		t.Fatal(err)
	}

	got := make(chan error, 1)
	go func() {
		_, err := st.Recv() // blocks: the server sends nothing more
		got <- err
	}()
	time.Sleep(100 * time.Millisecond) // let Recv block in the read
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case err := <-got:
		pe := mustProviderErr(t, err)
		if pe.Kind != provider.KindCanceled || pe.Retryable {
			t.Errorf("blocked Recv after Close = %+v, want non-retryable KindCanceled", pe)
		}
		// The classification is sticky.
		_, err2 := st.Recv()
		if pe2 := mustProviderErr(t, err2); pe2.Kind != provider.KindCanceled || pe2.Retryable {
			t.Errorf("later Recv = %+v, want non-retryable KindCanceled", pe2)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Recv did not return after Close")
	}
}

func TestStreamContextCancelPropagates(t *testing.T) {
	serverSawCancel := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body) // lets the server notice the client going away
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(w, chunkJSON("a"))
		select {
		case <-r.Context().Done():
			close(serverSawCancel)
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st, err := newClient(srv.URL, "k").Stream(ctx, newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-serverSawCancel:
	case <-time.After(time.Second):
		t.Fatal("server did not observe cancellation within 1s")
	}
	_, err = st.Recv()
	if pe := mustProviderErr(t, err); pe.Kind != provider.KindCanceled {
		t.Errorf("Recv after cancel: %+v", pe)
	}
}

func TestTransportReusesConnections(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, completionJSON)
	}))
	defer srv.Close()
	c := newClient(srv.URL, "k")

	var mu sync.Mutex
	var reused []bool
	trace := &httptrace.ClientTrace{GotConn: func(i httptrace.GotConnInfo) {
		mu.Lock()
		defer mu.Unlock()
		reused = append(reused, i.Reused)
	}}
	for i := 0; i < 2; i++ {
		ctx := httptrace.WithClientTrace(context.Background(), trace)
		if _, err := c.Complete(ctx, newReq("m")); err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reused) != 2 || reused[0] || !reused[1] {
		t.Errorf("GotConn reused = %v, want [false true]", reused)
	}
}

func TestTransportReusesConnectionsAfterErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = io.WriteString(w, `{"error":{"message":"x"}}`)
	}))
	defer srv.Close()
	c := newClient(srv.URL, "k")
	var reused []bool
	trace := &httptrace.ClientTrace{GotConn: func(i httptrace.GotConnInfo) { reused = append(reused, i.Reused) }}
	for i := 0; i < 2; i++ {
		_, _ = c.Complete(httptrace.WithClientTrace(context.Background(), trace), newReq("m"))
	}
	if len(reused) != 2 || !reused[1] {
		t.Errorf("GotConn reused = %v, want second reused", reused)
	}
}

var _ provider.Provider = (*openaicompat.Client)(nil)
