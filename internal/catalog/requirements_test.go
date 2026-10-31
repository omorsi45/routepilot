package catalog

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
)

func capSetOf(caps ...Capability) CapSet {
	var s CapSet
	for _, c := range caps {
		s = s.with(c)
	}
	return s
}

func strContent(s string) openai.Content { return openai.Content{Text: &s} }

func userMsg(s string) openai.Message {
	return openai.Message{Role: "user", Content: strContent(s)}
}

func parseReq(t *testing.T, body string) *openai.ChatCompletionRequest {
	t.Helper()
	req, apiErr := openai.ParseChatCompletionRequest([]byte(body))
	if apiErr != nil {
		t.Fatalf("parse request: %v", apiErr)
	}
	return req
}

func TestDeriveCapabilities(t *testing.T) {
	tool := `"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]`
	user := `{"role":"user","content":"hi"}`
	cases := []struct {
		name string
		body string
		want CapSet
	}{
		{"plain text", `{"model":"m","messages":[` + user + `]}`, 0},
		{"tools", `{"model":"m","messages":[` + user + `],` + tool + `}`, capSetOf(CapTools)},
		{"tools with tool_choice none still required", `{"model":"m","messages":[` + user + `],` + tool + `,"tool_choice":"none"}`, capSetOf(CapTools)},
		{"tool_choice without tools is not a requirement", `{"model":"m","messages":[` + user + `],"tool_choice":"auto"}`, 0},
		{"assistant tool_calls", `{"model":"m","messages":[` + user +
			`,{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`, capSetOf(CapTools)},
		{"tool role message", `{"model":"m","messages":[` + user +
			`,{"role":"tool","tool_call_id":"c1","content":"ok"}]}`, capSetOf(CapTools)},
		{"json_object", `{"model":"m","messages":[` + user + `],"response_format":{"type":"json_object"}}`, capSetOf(CapJSONObject)},
		{"json_schema", `{"model":"m","messages":[` + user + `],"response_format":{"type":"json_schema","json_schema":{"name":"x"}}}`, capSetOf(CapJSONSchema)},
		{"text response_format", `{"model":"m","messages":[` + user + `],"response_format":{"type":"text"}}`, 0},
		{"image part", `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"http://x/y.png"}}]}]}`, capSetOf(CapVision)},
		{"text parts only", `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"}]}]}`, 0},
		{"stream", `{"model":"m","messages":[` + user + `],"stream":true}`, capSetOf(CapStreaming)},
		{"everything", `{"model":"m","stream":true,"response_format":{"type":"json_object"},` + tool + `,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"u"}}]}]}`,
			capSetOf(CapTools, CapJSONObject, CapVision, CapStreaming)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Derive(parseReq(t, tc.body)).Caps; got != tc.want {
				t.Errorf("Caps = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestDeriveMaxTokensAndPromptBytes(t *testing.T) {
	r := Derive(parseReq(t, `{"model":"m","max_tokens":50,"messages":[{"role":"user","content":"abc"}]}`))
	if !r.HasMaxTokens || r.MaxTokens != 50 {
		t.Errorf("max tokens = %d/%v, want 50/true", r.MaxTokens, r.HasMaxTokens)
	}
	r = Derive(parseReq(t, `{"model":"m","max_tokens":50,"max_completion_tokens":70,"messages":[{"role":"user","content":"abc"}]}`))
	if !r.HasMaxTokens || r.MaxTokens != 70 {
		t.Errorf("max_completion_tokens should win: %d/%v", r.MaxTokens, r.HasMaxTokens)
	}
	r = Derive(parseReq(t, `{"model":"m","messages":[{"role":"user","content":"abc"}]}`))
	if r.HasMaxTokens || r.MaxTokens != 0 {
		t.Errorf("unset max tokens = %d/%v", r.MaxTokens, r.HasMaxTokens)
	}

	// String content plus text parts only, no separators; image parts and null content add nothing.
	r = Derive(parseReq(t, `{"model":"m","messages":[`+
		`{"role":"system","content":"héllo"},`+ // 6 bytes
		`{"role":"user","content":[{"type":"text","text":"ab"},{"type":"image_url","image_url":{"url":"http://long/url"}},{"type":"text","text":"cd"}]},`+ // 4
		`{"role":"assistant","content":null}]}`))
	if r.PromptBytes != 10 {
		t.Errorf("PromptBytes = %d, want 10", r.PromptBytes)
	}
	if want := "héllo\nab\ncd\n"; r.Text != want {
		t.Errorf("Text = %q, want %q", r.Text, want)
	}
}

func TestDeriveInUpper(t *testing.T) {
	args := `{"city":"Zürich"}`
	req := &openai.ChatCompletionRequest{
		Model: "m",
		Messages: []openai.Message{
			{Role: "user", Name: "alice", Content: strContent("héllo wörld")}, // text 13 bytes (é, ö are 2 each), name 5
			{Role: "assistant", Content: openai.Content{}, ToolCalls: []openai.ToolCall{
				{ID: "c1", Type: "function", Function: openai.FunctionCall{Name: "get_weather", Arguments: args}},
			}},
		},
		Tools: []openai.Tool{{Type: "function", Function: openai.FunctionDef{
			Name: "get_weather", Description: "weather ☀", Parameters: json.RawMessage(`{"type":"object"}`),
		}}},
	}
	toolsJSON, err := json.Marshal(req.Tools)
	if err != nil {
		t.Fatal(err)
	}
	want := 13 + // PromptBytes
		len("alice") +
		len("c1") + len("function") + len("get_weather") + len(args) + // tool call id, type, name, arguments
		len(toolsJSON) +
		8*2 + 16
	r := Derive(req)
	if r.PromptBytes != 13 {
		t.Errorf("PromptBytes = %d, want 13 (bytes, not runes)", r.PromptBytes)
	}
	if r.InUpper != want {
		t.Errorf("InUpper = %d, want %d", r.InUpper, want)
	}

	// No tools, no names: PromptBytes + 8 per message + 16.
	plain := Derive(&openai.ChatCompletionRequest{Messages: []openai.Message{userMsg("abc"), userMsg("de")}})
	if plain.InUpper != 5+16+16 {
		t.Errorf("plain InUpper = %d, want %d", plain.InUpper, 5+16+16)
	}
}

// TestDeriveInUpperCountsForwardedContent pins that everything the
// upstream receives is counted, not only message text.
func TestDeriveInUpperCountsForwardedContent(t *testing.T) {
	const base = 1 + 8 + 16 // one message "x", per-message and fixed overhead
	user := `{"role":"user","content":"x"}`
	cases := []struct {
		name string
		body string
		want int
	}{
		{"baseline", `{"model":"m","messages":[` + user + `]}`, base},
		{"image part uses the per-image allowance, not the URL",
			`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"ab"},` +
				`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("A", 5000) + `"}}]}]}`,
			2 + ImageTokenAllowance + 8 + 16},
		{"two images",
			`{"model":"m","messages":[{"role":"user","content":[` +
				`{"type":"image_url","image_url":{"url":"http://a/1.png"}},{"type":"image_url","image_url":{"url":"http://a/2.png"}}]}]}`,
			2*ImageTokenAllowance + 8 + 16},
		{"json_schema response_format",
			`{"model":"m","messages":[` + user + `],"response_format":{"type":"json_schema","json_schema":{"name":"out","strict":true}}}`,
			base + len(`{"name":"out","strict":true}`)},
		{"named tool_choice",
			`{"model":"m","messages":[` + user + `],"tool_choice":{"type":"function","function":{"name":"lookup"}}}`,
			base + len(`{"type":"function","function":{"name":"lookup"}}`)},
		{"unknown top-level field counts key and raw value",
			`{"model":"m","messages":[` + user + `],"logit_bias":{"50256":-100}}`,
			base + len("logit_bias") + len(`{"50256":-100}`)},
		{"unknown message field counts key and raw value",
			`{"model":"m","messages":[{"role":"user","content":"x","cache_control":{"type":"ephemeral"}}]}`,
			base + len("cache_control") + len(`{"type":"ephemeral"}`)},
		{"input_audio part counts its whole JSON",
			`{"model":"m","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]}`,
			len(`{"input_audio":{"data":"AAAA","format":"wav"},"type":"input_audio"}`) + 8 + 16},
		{"unknown fields on text and image_url parts count key and raw value",
			`{"model":"m","messages":[{"role":"user","content":[` +
				`{"type":"text","text":"ab","cache_control":{"type":"ephemeral"}},` +
				`{"type":"image_url","image_url":{"url":"u"},"detail_hint":"low"}]}]}`,
			2 + len("cache_control") + len(`{"type":"ephemeral"}`) +
				ImageTokenAllowance + len("detail_hint") + len(`"low"`) + 8 + 16},
		{"tool_call_id counts",
			`{"model":"m","messages":[{"role":"tool","tool_call_id":"call_abc123","content":"x"}]}`,
			base + len("call_abc123")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Derive(parseReq(t, tc.body)).InUpper; got != tc.want {
				t.Errorf("InUpper = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestDeriveInUpperToolsMarshalFallback covers a tool whose parameters are not
// valid JSON, where json.Marshal fails: the count must stay conservative.
func TestDeriveInUpperToolsMarshalFallback(t *testing.T) {
	bad := json.RawMessage(`{"type":`)
	req := &openai.ChatCompletionRequest{
		Messages: []openai.Message{userMsg("x")},
		Tools: []openai.Tool{
			{Type: "function", Function: openai.FunctionDef{Name: "alpha", Description: "does a thing", Parameters: bad}},
			{Type: "function", Function: openai.FunctionDef{Name: "beta"}},
		},
	}
	if _, err := json.Marshal(req.Tools); err == nil {
		t.Fatal("precondition: marshaling the tools should fail")
	}
	want := 1 + 8 + 16 +
		(len("alpha") + len("does a thing") + len(bad) + 64) +
		(len("beta") + 64)
	if got := Derive(req).InUpper; got != want {
		t.Errorf("InUpper = %d, want %d", got, want)
	}
}

func TestDeriveTextTruncatesOnRuneBoundary(t *testing.T) {
	// 3-byte runes never align to 32768 (32768 % 3 == 2), so the cut must back up.
	long := strings.Repeat("€", 20000)
	r := Derive(&openai.ChatCompletionRequest{Messages: []openai.Message{userMsg(long)}})
	if len(r.Text) > textLimit {
		t.Fatalf("Text is %d bytes, limit %d", len(r.Text), textLimit)
	}
	if !utf8.ValidString(r.Text) {
		t.Error("Text is not valid UTF-8 after truncation")
	}
	if want := (textLimit / 3) * 3; len(r.Text) != want {
		t.Errorf("Text len = %d, want %d", len(r.Text), want)
	}
	if r.PromptBytes != len(long) {
		t.Errorf("PromptBytes = %d, want untruncated %d", r.PromptBytes, len(long))
	}

	// Short text is untouched; ASCII at the limit is kept whole.
	ascii := strings.Repeat("a", textLimit+100)
	r = Derive(&openai.ChatCompletionRequest{Messages: []openai.Message{userMsg(ascii)}})
	if len(r.Text) != textLimit {
		t.Errorf("ASCII Text len = %d, want %d", len(r.Text), textLimit)
	}
	r = Derive(&openai.ChatCompletionRequest{Messages: []openai.Message{userMsg("hi")}})
	if r.Text != "hi" {
		t.Errorf("short Text = %q", r.Text)
	}
}

// stringBytes sums the byte length of every string value in a decoded JSON
// tree. Object keys are not counted, and neither is a "role" value: roles are
// framing that the per-message overhead covers.
func stringBytes(v any) int {
	switch x := v.(type) {
	case string:
		return len(x)
	case []any:
		n := 0
		for _, e := range x {
			n += stringBytes(e)
		}
		return n
	case map[string]any:
		n := 0
		for k, e := range x {
			if k == "role" {
				continue
			}
			n += stringBytes(e)
		}
		return n
	}
	return 0
}

// TestInUpperBoundsMockPromptTokens checks the upper bound on generated
// requests against two oracles: the mock provider's own prompt token count,
// and the total bytes of every string value that is forwarded upstream
// (messages, tools, tool_choice, response_format) plus request Extra.
func TestInUpperBoundsMockPromptTokens(t *testing.T) {
	h := mock.NewHandler(mock.Config{Seed: 1})
	rng := rand.New(rand.NewPCG(7, 11))
	snippets := []string{"", "a", "hello world", "héllo wörld", "日本語のテキスト", "emoji 😀😀", strings.Repeat("x", 301)}
	snippet := func() string { return snippets[rng.IntN(len(snippets))] }
	roles := []string{"system", "developer", "user", "assistant"}

	for i := 0; i < 40; i++ {
		req := &openai.ChatCompletionRequest{Model: "m;tokens=3;ttft_ms=0;tpot_ms=0"}
		for j, n := 0, 1+rng.IntN(6); j < n; j++ {
			m := openai.Message{Role: roles[rng.IntN(len(roles))]}
			switch rng.IntN(3) {
			case 0:
				m.Content = strContent(snippet())
			case 1:
				m.Content = openai.Content{Parts: []openai.ContentPart{
					{Type: "text", Text: snippet()},
					{Type: "image_url", ImageURL: &openai.ImageURL{URL: "http://example.invalid/i.png"}},
					{Type: "text", Text: snippet()},
				}}
				if rng.IntN(3) == 0 {
					m.Content.Parts = append(m.Content.Parts, openai.ContentPart{
						Type:  "input_audio",
						Extra: map[string]json.RawMessage{"input_audio": json.RawMessage(`{"data":"` + snippet() + `","format":"wav"}`)},
					})
				}
			}
			if rng.IntN(2) == 0 {
				m.Name = fmt.Sprintf("n%d", j)
			}
			switch {
			case m.Role == "assistant" && rng.IntN(2) == 0:
				m.ToolCalls = []openai.ToolCall{{
					ID: fmt.Sprintf("call_%d", j), Type: "function",
					Function: openai.FunctionCall{Name: "lookup", Arguments: `{"q":"` + snippet() + `"}`},
				}}
			case m.Role == "user" && rng.IntN(4) == 0:
				m.Role = "tool"
				m.ToolCallID = fmt.Sprintf("call_%d", j)
			}
			if rng.IntN(3) == 0 {
				m.Extra = map[string]json.RawMessage{"annotations": json.RawMessage(`["` + snippet() + `"]`)}
			}
			req.Messages = append(req.Messages, m)
		}
		if rng.IntN(2) == 0 {
			req.Tools = []openai.Tool{{Type: "function", Function: openai.FunctionDef{
				Name: "f", Description: snippet(), Parameters: json.RawMessage(`{"type":"object"}`),
			}}}
		}
		if rng.IntN(2) == 0 {
			req.ToolChoice = json.RawMessage(`{"type":"function","function":{"name":"lookup"}}`)
		}
		if rng.IntN(2) == 0 {
			req.ResponseFormat = &openai.ResponseFormat{
				Type: "json_schema", JSONSchema: json.RawMessage(`{"name":"` + snippet() + `","schema":{"type":"object"}}`),
			}
		}
		if rng.IntN(2) == 0 {
			req.Extra = map[string]json.RawMessage{"metadata": json.RawMessage(`{"trace":"` + snippet() + `"}`)}
		}

		body, err := json.Marshal(req)
		if err != nil {
			t.Fatal(err)
		}
		inUpper := Derive(req).InUpper

		// Oracle 1: the mock provider's prompt token count.
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body))))
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d: mock status %d: %s", i, rec.Code, rec.Body.String())
		}
		var resp openai.ChatCompletion
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("request %d: decode: %v", i, err)
		}
		if resp.Usage == nil {
			t.Fatalf("request %d: no usage", i)
		}
		if resp.Usage.PromptTokens > inUpper {
			t.Errorf("request %d: mock prompt_tokens %d > InUpper %d", i, resp.Usage.PromptTokens, inUpper)
		}

		// Oracle 2: every forwarded string byte, plus request Extra.
		var tree map[string]any
		if err := json.Unmarshal(body, &tree); err != nil {
			t.Fatal(err)
		}
		forwarded := 0
		for _, key := range []string{"messages", "tools", "tool_choice", "response_format"} {
			forwarded += stringBytes(tree[key])
		}
		for k, v := range req.Extra {
			forwarded += len(k) + len(v)
		}
		if forwarded > inUpper {
			t.Errorf("request %d: forwarded string bytes %d > InUpper %d", i, forwarded, inUpper)
		}
	}
}
