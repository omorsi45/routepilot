package gateway

import (
	"io"
	"net/http"
	"strings"
	"testing"
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
