package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/provider/factory"
)

// sseUpstream serves the given SSE data payloads in order, flushing after
// each, as an OpenAI-compatible streaming upstream.
func sseUpstream(payloads ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, p := range payloads {
			_, _ = io.WriteString(w, "data: "+p+"\n\n")
			w.(http.Flusher).Flush()
		}
	})
}

func contentChunk(text string) string {
	return `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"content":"` + text + `"},"finish_reason":null}]}`
}

const inBandError = `{"error":{"message":"engine died","type":"server_error"}}`

func TestStreamInBandErrorAfterContentIsOneErrorEvent(t *testing.T) {
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(contentChunk("partial"), inBandError, "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", true))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200 (a chunk was committed)", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 2 {
		t.Fatalf("got %d events, want content chunk + one error event: %q", len(events), events)
	}
	if c := decodeChunk(t, events[0]); len(c.Choices) != 1 || c.Choices[0].Delta.Content == nil || *c.Choices[0].Delta.Content != "partial" {
		t.Errorf("first event is not the content chunk: %s", events[0])
	}
	if !strings.Contains(events[1], `"type":"upstream_error"`) || !strings.Contains(events[1], `"code":"upstream_error"`) {
		t.Errorf("second event is not an upstream_error: %s", events[1])
	}
	for _, ev := range events {
		if ev == "[DONE]" {
			t.Error("stream ended with [DONE] after an upstream error")
		}
	}
	if strings.Contains(events[1], "engine died") {
		t.Errorf("error event leaks the upstream message: %s", events[1])
	}
}

func TestStreamInBandErrorFirstEventIs502JSON(t *testing.T) {
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(inBandError, "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", true))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"code":"upstream_error"`) {
		t.Errorf("body is not an upstream_error envelope: %s", body)
	}
}

func TestCompleteInBandErrorBodyIs502(t *testing.T) {
	for name, body := range map[string]string{
		"error":      inBandError,
		"no choices": `{"id":"up","object":"chat.completion","created":1,"model":"up-a"}`,
	} {
		t.Run(name, func(t *testing.T) {
			e := newUpstreamEnv(t, cfgOpts{}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			}))
			rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
			}
			if got := decodeErr(t, rec); got.Error.Code != "upstream_error" {
				t.Errorf("code %q, want upstream_error", got.Error.Code)
			}
		})
	}
}

// anthropicStreamServer serves POST /v1/messages as an Anthropic SSE stream.
// Each step is "event name\x00data" or a time.Duration to sleep.
func anthropicStreamServer(t *testing.T, steps ...any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		for _, st := range steps {
			switch v := st.(type) {
			case time.Duration:
				select {
				case <-time.After(v):
				case <-r.Context().Done():
					return
				}
			case [2]string:
				fmt.Fprintf(w, "event: %s\ndata: %s\n\n", v[0], v[1])
				w.(http.Flusher).Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

const (
	anthMessageStart = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-x","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}`
	anthBlockStart   = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	anthOverloaded   = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`
)

func anthropicGateway(t *testing.T, claudeURL string, ttftMs int) string {
	t.Helper()
	sum := sha256.Sum256([]byte(keyA))
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: claude
    type: anthropic
    base_url: %q
    ttft_timeout_ms: %d
models:
  - id: claude-model
    deployments:
      - provider: claude
        upstream_model: claude-x
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
`, claudeURL, ttftMs, hex.EncodeToString(sum[:]))))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	provs, err := factory.Build(cfg.Providers, func(string) string { return "" }, nil)
	if err != nil {
		t.Fatalf("factory.Build: %v", err)
	}
	gw, err := New(cfg, provs, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	front := httptest.NewServer(gw.Handler())
	t.Cleanup(front.Close)
	return front.URL
}

func TestStreamAnthropicNoTokenBeforeTTFTIs504JSON(t *testing.T) {
	up := anthropicStreamServer(t,
		[2]string{"message_start", anthMessageStart},
		600*time.Millisecond,
		[2]string{"error", anthOverloaded},
	)
	url := anthropicGateway(t, up.URL, 100)

	start := time.Now()
	resp := postStream(t, newStreamClient(), url, streamReqBody("claude-model", false))
	elapsed := time.Since(start)
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status %d after %v, want 504 JSON", resp.StatusCode, elapsed)
	}
	if elapsed > time.Second {
		t.Errorf("504 took %v, want under 1s", elapsed)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"code":"upstream_timeout"`) {
		t.Errorf("body is not an upstream_timeout envelope: %s", body)
	}
}

func TestStreamAnthropicRoleChunkIsNotLost(t *testing.T) {
	up := anthropicStreamServer(t,
		[2]string{"message_start", anthMessageStart},
		[2]string{"content_block_start", anthBlockStart},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	url := anthropicGateway(t, up.URL, 5000)

	resp := postStream(t, newStreamClient(), url, streamReqBody("claude-model", false))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 4 {
		t.Fatalf("got %d events, want role, content, finish, [DONE]: %q", len(events), events)
	}
	role := decodeChunk(t, events[0])
	if role.Choices[0].Delta.Role != "assistant" {
		t.Errorf("first event is not the role chunk: %s", events[0])
	}
	content := decodeChunk(t, events[1])
	if c := content.Choices[0].Delta.Content; c == nil || *c != "Hello" {
		t.Errorf("second event is not the content chunk: %s", events[1])
	}
	if role.ID != content.ID || role.Model != "claude-model" || role.Object != "chat.completion.chunk" {
		t.Errorf("held role chunk was not rewritten like the rest: %s", events[0])
	}
	if events[3] != "[DONE]" {
		t.Errorf("last event %q, want [DONE]", events[3])
	}
}

func TestStreamHeldChunkCapIs502BeforeFirstByte(t *testing.T) {
	roleOnly := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`
	payloads := make([]string, 0, 70)
	for i := 0; i < 70; i++ {
		payloads = append(payloads, roleOnly)
	}
	payloads = append(payloads, contentChunk("late"), "[DONE]")
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(payloads...))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502 JSON", resp.StatusCode)
	}
}

func TestStreamEndsBeforeAnyContentIs502JSON(t *testing.T) {
	roleOnly := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(roleOnly, "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502 JSON", resp.StatusCode)
	}
}

func TestStreamCommittingChunks(t *testing.T) {
	roleOnly := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`
	toolCall := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`
	finishOnly := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	for name, commit := range map[string]string{"tool call": toolCall, "finish reason": finishOnly} {
		t.Run(name, func(t *testing.T) {
			e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(roleOnly, commit, "[DONE]"))
			resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status %d, want 200", resp.StatusCode)
			}
			events := readAll(t, resp)
			if len(events) != 3 || events[2] != "[DONE]" {
				t.Fatalf("events = %q, want role, %s, [DONE]", events, name)
			}
		})
	}
}

// The mock-like upstream below returns fields outside the typed subset.
const completionWithExtras = `{"id":"up","object":"chat.completion","created":1,"model":"up-a",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"I cannot help with that"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":1}},"service_tier":"default"}`

func TestProxyKeepsRefusalAndUsageDetails(t *testing.T) {
	e := newUpstreamEnv(t, cfgOpts{}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completionWithExtras)
	}))
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var got struct {
		Choices []struct {
			Message map[string]any `json:"message"`
		} `json:"choices"`
		Usage       map[string]any `json:"usage"`
		ServiceTier string         `json:"service_tier"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Choices[0].Message["refusal"] != "I cannot help with that" {
		t.Errorf("refusal lost: %s", rec.Body.String())
	}
	if d, _ := got.Usage["completion_tokens_details"].(map[string]any); d["reasoning_tokens"] != float64(1) {
		t.Errorf("usage details lost: %s", rec.Body.String())
	}
	if got.ServiceTier != "default" {
		t.Errorf("service_tier lost: %s", rec.Body.String())
	}
}

func TestStreamKeepsReasoningContent(t *testing.T) {
	reasoning := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"thinking"},"finish_reason":null}]}`
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(reasoning, contentChunk("answer"), "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 3 {
		t.Fatalf("events = %q, want reasoning chunk, content chunk, [DONE]", events)
	}
	if !strings.Contains(events[0], `"reasoning_content":"thinking"`) {
		t.Errorf("reasoning_content dropped: %s", events[0])
	}
}

func TestStreamReasoningOnlyPrefixIsNotCappedAsEmpty(t *testing.T) {
	// A reasoning model streams many reasoning deltas before any answer. They
	// carry content, so they commit the response instead of being held.
	var payloads []string
	for i := 0; i < maxHeldChunks+10; i++ {
		payloads = append(payloads, `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"reasoning_content":"r"},"finish_reason":null}]}`)
	}
	payloads = append(payloads, contentChunk("answer"), "[DONE]")
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(payloads...))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
}

func TestStreamRefusalNullOnFirstChunkStillHeld(t *testing.T) {
	// OpenAI's own opener carries "refusal":null; that must not commit.
	opener := `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"role":"assistant","content":"","refusal":null},"finish_reason":null}]}`
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(opener, "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502 (opener alone is not content)", resp.StatusCode)
	}
}

func TestProxyForwardsInputAudioPartIntact(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=2"})
	body := `{"model":"model-a","messages":[{"role":"user","content":[` +
		`{"type":"text","text":""},` +
		`{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}},` +
		`{"type":"file","file":{"file_id":"file-1"}}]}]}`
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var up struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(e.upstream.last(t), &up); err != nil {
		t.Fatal(err)
	}
	parts := up.Messages[0].Content
	if len(parts) != 3 {
		t.Fatalf("upstream saw %d parts: %s", len(parts), e.upstream.last(t))
	}
	if v, ok := parts[0]["text"]; !ok || v != "" {
		t.Errorf("empty text part lost its text: %v", parts[0])
	}
	audio, _ := parts[1]["input_audio"].(map[string]any)
	if audio["data"] != "AAAA" || audio["format"] != "wav" {
		t.Errorf("input_audio stripped: %v", parts[1])
	}
	file, _ := parts[2]["file"].(map[string]any)
	if file["file_id"] != "file-1" {
		t.Errorf("file part stripped: %v", parts[2])
	}
}

func TestObjectIsNormalized(t *testing.T) {
	t.Run("completion", func(t *testing.T) {
		e := newUpstreamEnv(t, cfgOpts{}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"up","object":"","created":1,"model":"up-a","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
		}))
		rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
		if got := decodeMap(t, rec.Body.Bytes())["object"]; got != "chat.completion" {
			t.Errorf("object = %v, want chat.completion: %s", got, rec.Body.String())
		}
	})
	t.Run("chunks", func(t *testing.T) {
		noObject := `{"id":"up","created":1,"model":"up-a","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}`
		e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(noObject, noObject, "[DONE]"))
		resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
		events := readAll(t, resp)
		if len(events) != 3 {
			t.Fatalf("events = %q", events)
		}
		for _, ev := range events[:2] {
			if got := decodeMap(t, []byte(ev))["object"]; got != "chat.completion.chunk" {
				t.Errorf("object = %v, want chat.completion.chunk: %s", got, ev)
			}
		}
	})
}

func decodeMap(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return m
}

func TestLogsCatalogModelNotRawInput(t *testing.T) {
	t.Run("known model logs the catalog id", func(t *testing.T) {
		e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=2"})
		rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d", rec.Code)
		}
		if !strings.Contains(e.logs.String(), "model=model-a ") {
			t.Errorf("log does not carry the catalog model id: %s", e.logs.String())
		}
	})
	t.Run("unknown model is capped at 256 bytes", func(t *testing.T) {
		e := newMockEnv(t, cfgOpts{})
		long := strings.Repeat("x", 100000)
		rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody(long, "hi"))
		if rec.Code != http.StatusNotFound {
			t.Fatalf("status %d, want 404: %.200s", rec.Code, rec.Body.String())
		}
		logs := e.logs.String()
		if !strings.Contains(logs, "model="+strings.Repeat("x", 256)+" ") {
			t.Errorf("log should carry the first 256 bytes of the model: %.400s", logs)
		}
		if len(logs) > 2048 {
			t.Errorf("log line is %d bytes, the raw model leaked into it", len(logs))
		}
	})
}

func TestRequestExtrasReachUpstreamAndRoutepilotIsStripped(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=2"})
	body := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"logit_bias":{"50256":-100},"reasoning_effort":"low","routepilot":{"timeout_ms":5000}}`
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	up := decodeMap(t, e.upstream.last(t))
	if bias, _ := up["logit_bias"].(map[string]any); bias["50256"] != float64(-100) {
		t.Errorf("logit_bias did not reach the upstream: %v", up)
	}
	if up["reasoning_effort"] != "low" {
		t.Errorf("reasoning_effort did not reach the upstream: %v", up)
	}
	if _, ok := up["routepilot"]; ok {
		t.Errorf("routepilot reached the upstream: %v", up)
	}
}

// bigEvent is an SSE data payload just over the 1 MiB event cap.
func bigEvent() string {
	return `{"id":"up","object":"chat.completion.chunk","created":1,"model":"up-a","choices":[{"index":0,"delta":{"content":"` +
		strings.Repeat("a", 1<<20) + `"},"finish_reason":null}]}`
}

func TestStreamOversizedFirstEventIs502JSON(t *testing.T) {
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(bigEvent(), "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want 502", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
}

func TestStreamOversizedEventAfterCommitIsOneErrorEvent(t *testing.T) {
	e := newUpstreamEnv(t, cfgOpts{}, sseUpstream(contentChunk("first"), bigEvent(), "[DONE]"))
	resp := postStream(t, newStreamClient(), startGateway(t, e), streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	events := readAll(t, resp)
	if len(events) != 2 {
		t.Fatalf("got %d events, want the content chunk and one error event", len(events))
	}
	if !strings.Contains(events[1], `"code":"upstream_error"`) {
		t.Errorf("second event is not an upstream_error: %.200s", events[1])
	}
	for _, ev := range events {
		if ev == "[DONE]" {
			t.Error("stream ended with [DONE] after an oversized event")
		}
	}
}
