package gateway_test

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/gateway"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/provider/factory"
)

const e2eKey = "test-key-e2e"

// anthropicFixture serves POST /v1/messages in the Anthropic wire format:
// a JSON message, or an SSE stream when the request sets "stream": true.
func anthropicFixture(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_e2e","type":"message","role":"assistant","model":"claude-fixture-1",`+
				`"content":[{"type":"text","text":"Hello, world!"}],"stop_reason":"end_turn","stop_sequence":null,`+
				`"usage":{"input_tokens":25,"output_tokens":15}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"message_start","message":{"id":"msg_e2e","type":"message","role":"assistant","content":[],"model":"claude-fixture-1","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world!"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`,
			`{"type":"message_stop"}`,
		}
		for _, ev := range events {
			var head struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal([]byte(ev), &head)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", head.Type, ev)
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// startGateway wires a real gateway over the factory-built providers: the
// openaicompat adapter against the mock and the anthropic adapter against the
// fixture server. It returns the gateway's base URL.
func startGateway(t *testing.T) string {
	t.Helper()
	mockSrv := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(mockSrv.Close)
	claudeSrv := anthropicFixture(t)

	sum := sha256.Sum256([]byte(e2eKey))
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
  - name: claude
    type: anthropic
    base_url: %q
models:
  - id: mock-model
    deployments:
      - provider: mockp
        upstream_model: "m;tokens=3"
  - id: mock-conc
    deployments:
      - provider: mockp
        upstream_model: "m;tokens=8;ttft_ms=0;tpot_ms=0"
  - id: claude-model
    deployments:
      - provider: claude
        upstream_model: claude-fixture-1
tenants:
  - id: e2e
    api_key_sha256: %s
    allowed_models: ["*"]
`, mockSrv.URL+"/v1", claudeSrv.URL, hex.EncodeToString(sum[:]))))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}
	providers, err := factory.Build(cfg.Providers, func(string) string { return "" }, nil)
	if err != nil {
		t.Fatalf("factory.Build() error = %v", err)
	}
	gw, err := gateway.New(cfg, providers, gateway.Options{})
	if err != nil {
		t.Fatalf("gateway.New() error = %v", err)
	}
	front := httptest.NewServer(gw.Handler())
	t.Cleanup(front.Close)
	return front.URL
}

func chatRequest(t testing.TB, baseURL, model string, stream bool) *http.Request {
	t.Helper()
	body := fmt.Sprintf(`{"model":%q,"stream":%t,"messages":[{"role":"user","content":"hi"}]}`, model, stream)
	req, err := http.NewRequest(http.MethodPost, baseURL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+e2eKey)
	req.Header.Set("Content-Type", "application/json")
	return req
}

// field returns m[key] and records an error when it is absent. Decoding into
// map[string]any means a missing field is seen as missing, never zero-filled.
func field(m map[string]any, key string) (any, error) {
	v, ok := m[key]
	if !ok {
		return nil, fmt.Errorf("missing field %q", key)
	}
	return v, nil
}

// checkCompletion returns the problems found with an OpenAI chat.completion.
func checkCompletion(m map[string]any) []string {
	var problems []string
	bad := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }

	if id, _ := m["id"].(string); id == "" {
		bad("id = %v, want a non-empty string", m["id"])
	}
	if m["object"] != "chat.completion" {
		bad("object = %v, want chat.completion", m["object"])
	}
	if created, _ := m["created"].(float64); created <= 0 {
		bad("created = %v, want > 0", m["created"])
	}
	if model, _ := m["model"].(string); model == "" {
		bad("model = %v, want a non-empty string", m["model"])
	}
	choices, _ := m["choices"].([]any)
	if len(choices) == 0 {
		bad("choices = %v, want a non-empty array", m["choices"])
	} else {
		c, _ := choices[0].(map[string]any)
		if idx, ok := c["index"].(float64); !ok || idx != 0 {
			bad("choices[0].index = %v, want 0", c["index"])
		}
		msg, _ := c["message"].(map[string]any)
		if msg["role"] != "assistant" {
			bad("choices[0].message.role = %v, want assistant", msg["role"])
		}
		if _, ok := msg["content"].(string); !ok {
			bad("choices[0].message.content = %v, want a string", msg["content"])
		}
		if fr, _ := c["finish_reason"].(string); fr == "" {
			bad("choices[0].finish_reason = %v, want a non-empty string", c["finish_reason"])
		}
	}
	usage, _ := m["usage"].(map[string]any)
	for _, k := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if _, ok := usage[k].(float64); !ok {
			bad("usage.%s = %v, want a number", k, usage[k])
		}
	}
	return problems
}

// checkChunk returns the problems found with an OpenAI chat.completion.chunk.
func checkChunk(m map[string]any) []string {
	var problems []string
	bad := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }

	if id, _ := m["id"].(string); id == "" {
		bad("id = %v, want a non-empty string", m["id"])
	}
	if m["object"] != "chat.completion.chunk" {
		bad("object = %v, want chat.completion.chunk", m["object"])
	}
	if created, _ := m["created"].(float64); created <= 0 {
		bad("created = %v, want > 0", m["created"])
	}
	if model, _ := m["model"].(string); model == "" {
		bad("model = %v, want a non-empty string", m["model"])
	}
	if _, ok := m["choices"].([]any); !ok {
		bad("choices = %v, want an array", m["choices"])
	}
	return problems
}

// readStream reads an SSE body and returns the JSON events plus whether the
// final data event was [DONE].
func readStream(body io.Reader) (events []map[string]any, doneLast bool, err error) {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			doneLast = true
			continue
		}
		if doneLast {
			return events, false, fmt.Errorf("data event after [DONE]: %s", data)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			return events, false, fmt.Errorf("event is not JSON: %w: %s", err, data)
		}
		events = append(events, m)
	}
	return events, doneLast, sc.Err()
}

func doCompletion(client *http.Client, req *http.Request) (map[string]any, error) {
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("status = %d, want 200: %s", resp.StatusCode, b)
	}
	var m map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if problems := checkCompletion(m); len(problems) > 0 {
		return m, fmt.Errorf("malformed completion: %s", strings.Join(problems, "; "))
	}
	return m, nil
}

func doStream(client *http.Client, req *http.Request) ([]map[string]any, error) {
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("status = %d, want 200: %s", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		return nil, fmt.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	events, doneLast, err := readStream(resp.Body)
	if err != nil {
		return events, err
	}
	if len(events) == 0 {
		return events, fmt.Errorf("stream carried no chunks")
	}
	if !doneLast {
		return events, fmt.Errorf("stream did not end with [DONE]")
	}
	for i, ev := range events {
		if problems := checkChunk(ev); len(problems) > 0 {
			return events, fmt.Errorf("chunk %d malformed: %s", i, strings.Join(problems, "; "))
		}
	}
	return events, nil
}

func TestContractNonStreaming(t *testing.T) {
	base := startGateway(t)
	for _, model := range []string{"mock-model", "claude-model"} {
		t.Run(model, func(t *testing.T) {
			m, err := doCompletion(http.DefaultClient, chatRequest(t, base, model, false))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := field(m, "usage"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestContractStreaming(t *testing.T) {
	base := startGateway(t)
	for _, model := range []string{"mock-model", "claude-model"} {
		t.Run(model, func(t *testing.T) {
			events, err := doStream(http.DefaultClient, chatRequest(t, base, model, true))
			if err != nil {
				t.Fatal(err)
			}
			var text strings.Builder
			for _, ev := range events {
				choices, _ := ev["choices"].([]any)
				for _, c := range choices {
					delta, _ := c.(map[string]any)["delta"].(map[string]any)
					if s, ok := delta["content"].(string); ok {
						text.WriteString(s)
					}
				}
			}
			if text.Len() == 0 {
				t.Error("stream carried no content deltas")
			}
		})
	}
}

func TestConcurrentMixedRequests(t *testing.T) {
	const (
		workers  = 64
		requests = 20
	)
	base := startGateway(t)
	transport := &http.Transport{MaxIdleConnsPerHost: workers}
	t.Cleanup(transport.CloseIdleConnections)
	client := &http.Client{Transport: transport, Timeout: 30 * time.Second}

	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < requests; i++ {
				stream := (w+i)%2 == 0
				req := chatRequest(t, base, "mock-conc", stream)
				var err error
				if stream {
					_, err = doStream(client, req)
				} else {
					_, err = doCompletion(client, req)
				}
				if err != nil {
					t.Errorf("worker %d request %d (stream=%t): %v", w, i, stream, err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	t.Logf("%d requests in %s", workers*requests, time.Since(start).Round(time.Millisecond))
}
