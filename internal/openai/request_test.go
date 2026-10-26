package openai

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func decodeMap(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return m
}

func TestRequestPreservesUnknownFields(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"logit_bias":{"1":2},"reasoning_effort":"low"}`
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Extra) != 2 {
		t.Fatalf("Extra should hold only the 2 unknown keys, got %v", req.Extra)
	}
	if _, ok := req.Extra["model"]; ok {
		t.Fatal("known key model must not be in Extra")
	}
	out, err := json.Marshal(&req)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := decodeMap(t, out), decodeMap(t, []byte(in)); !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip mismatch\n got: %v\nwant: %v", got, want)
	}
}

func TestRequestKnownFieldsRoundTrip(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"max_completion_tokens":7,` +
		`"temperature":0.5,"top_p":0.9,"n":1,"stream":true,"stream_options":{"include_usage":true},"stop":["a"],` +
		`"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto",` +
		`"response_format":{"type":"json_object"},"seed":3,"user":"u","routepilot":{"min_quality":0.5,"timeout_ms":100}}`
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Extra) != 0 {
		t.Fatalf("Extra should be empty, got %v", req.Extra)
	}
	out, err := json.Marshal(&req)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := decodeMap(t, out), decodeMap(t, []byte(in)); !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip mismatch\n got: %v\nwant: %v", got, want)
	}
}

func TestRoutePilotFieldOmittedWhenNil(t *testing.T) {
	req := &ChatCompletionRequest{Model: "m", Messages: []Message{{Role: "user"}}}
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	m := decodeMap(t, out)
	if _, ok := m["routepilot"]; ok {
		t.Fatalf("routepilot must be omitted: %s", out)
	}
	if _, ok := m["stream"]; ok {
		t.Fatalf("stream must be omitted when false: %s", out)
	}
	if _, ok := m["model"]; !ok {
		t.Fatalf("model must always be present: %s", out)
	}
	if _, ok := m["messages"]; !ok {
		t.Fatalf("messages must always be present: %s", out)
	}
}

func TestContentStringOrParts(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		var c Content
		if err := json.Unmarshal([]byte(`"hi"`), &c); err != nil {
			t.Fatal(err)
		}
		if c.Text == nil || *c.Text != "hi" || c.Parts != nil {
			t.Fatalf("got %+v", c)
		}
		if c.PlainText() != "hi" || c.HasImage() {
			t.Fatal("PlainText/HasImage wrong for string")
		}
		out, _ := json.Marshal(c)
		if string(out) != `"hi"` {
			t.Fatalf("re-marshal = %s", out)
		}
	})
	t.Run("parts", func(t *testing.T) {
		in := `[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}}]`
		var c Content
		if err := json.Unmarshal([]byte(in), &c); err != nil {
			t.Fatal(err)
		}
		if c.Text != nil || len(c.Parts) != 2 {
			t.Fatalf("got %+v", c)
		}
		if c.PlainText() != "a" {
			t.Fatalf("PlainText = %q", c.PlainText())
		}
		if !c.HasImage() {
			t.Fatal("HasImage should be true")
		}
		out, _ := json.Marshal(c)
		var want, got any
		_ = json.Unmarshal([]byte(in), &want)
		_ = json.Unmarshal(out, &got)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("re-marshal = %s", out)
		}
	})
	t.Run("plain text joins text parts", func(t *testing.T) {
		c := Content{Parts: []ContentPart{{Type: "text", Text: "a"}, {Type: "image_url", ImageURL: &ImageURL{URL: "u"}}, {Type: "text", Text: "b"}}}
		if c.PlainText() != "a\nb" {
			t.Fatalf("PlainText = %q", c.PlainText())
		}
	})
	t.Run("null", func(t *testing.T) {
		var c Content
		if err := json.Unmarshal([]byte(`null`), &c); err != nil {
			t.Fatal(err)
		}
		if c.Text != nil || c.Parts != nil || c.PlainText() != "" || c.HasImage() {
			t.Fatalf("got %+v", c)
		}
		out, _ := json.Marshal(c)
		if string(out) != `null` {
			t.Fatalf("re-marshal = %s", out)
		}
	})
	t.Run("invalid", func(t *testing.T) {
		var c Content
		if err := json.Unmarshal([]byte(`42`), &c); err == nil {
			t.Fatal("expected error for number content")
		}
	})
	t.Run("message always emits content", func(t *testing.T) {
		out, _ := json.Marshal(Message{Role: "assistant"})
		if !strings.Contains(string(out), `"content":null`) {
			t.Fatalf("got %s", out)
		}
	})
}

func validReq() *ChatCompletionRequest {
	return &ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: textContent("hi")}},
	}
}

func textContent(s string) Content { return Content{Text: &s} }

func ptr[T any](v T) *T { return &v }

func TestValidateAcceptsMinimal(t *testing.T) {
	if err := validReq().Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateAcceptsBoundaries(t *testing.T) {
	r := validReq()
	r.N = ptr(1)
	r.Temperature = ptr(0.0)
	r.TopP = ptr(1.0)
	r.MaxTokens = ptr(1)
	r.MaxCompletionTokens = ptr(1)
	r.ResponseFormat = &ResponseFormat{Type: "json_schema"}
	r.RoutePilot = &RoutingOptions{MinQuality: ptr(1.0), SLOTTFTMs: ptr(1), SLOE2EMs: ptr(1), TimeoutMs: ptr(1)}
	r.Messages = append(r.Messages,
		Message{Role: "assistant"},
		Message{Role: "tool", ToolCallID: "call_1", Content: textContent("x")},
		Message{Role: "system"}, Message{Role: "developer"})
	if err := r.Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	r.Temperature = ptr(2.0)
	r.RoutePilot.MinQuality = ptr(0.0)
	if err := r.Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(r *ChatCompletionRequest)
		param  string
	}{
		{"empty model", func(r *ChatCompletionRequest) { r.Model = "" }, "model"},
		{"no messages", func(r *ChatCompletionRequest) { r.Messages = nil }, "messages"},
		{"bad role", func(r *ChatCompletionRequest) { r.Messages[0].Role = "robot" }, "messages[0].role"},
		{"tool without id", func(r *ChatCompletionRequest) {
			r.Messages = append(r.Messages, Message{Role: "tool"})
		}, "messages[1].tool_call_id"},
		{"n=2", func(r *ChatCompletionRequest) { r.N = ptr(2) }, "n"},
		{"n=0", func(r *ChatCompletionRequest) { r.N = ptr(0) }, "n"},
		{"temperature high", func(r *ChatCompletionRequest) { r.Temperature = ptr(2.5) }, "temperature"},
		{"temperature negative", func(r *ChatCompletionRequest) { r.Temperature = ptr(-0.1) }, "temperature"},
		{"top_p zero", func(r *ChatCompletionRequest) { r.TopP = ptr(0.0) }, "top_p"},
		{"top_p high", func(r *ChatCompletionRequest) { r.TopP = ptr(1.1) }, "top_p"},
		{"max_tokens zero", func(r *ChatCompletionRequest) { r.MaxTokens = ptr(0) }, "max_tokens"},
		{"max_completion_tokens zero", func(r *ChatCompletionRequest) { r.MaxCompletionTokens = ptr(0) }, "max_completion_tokens"},
		{"min_quality high", func(r *ChatCompletionRequest) { r.RoutePilot = &RoutingOptions{MinQuality: ptr(1.5)} }, "routepilot.min_quality"},
		{"min_quality negative", func(r *ChatCompletionRequest) { r.RoutePilot = &RoutingOptions{MinQuality: ptr(-0.5)} }, "routepilot.min_quality"},
		{"slo_ttft zero", func(r *ChatCompletionRequest) { r.RoutePilot = &RoutingOptions{SLOTTFTMs: ptr(0)} }, "routepilot.slo_ttft_ms"},
		{"slo_e2e zero", func(r *ChatCompletionRequest) { r.RoutePilot = &RoutingOptions{SLOE2EMs: ptr(0)} }, "routepilot.slo_e2e_ms"},
		{"timeout zero", func(r *ChatCompletionRequest) { r.RoutePilot = &RoutingOptions{TimeoutMs: ptr(0)} }, "routepilot.timeout_ms"},
		{"response_format xml", func(r *ChatCompletionRequest) { r.ResponseFormat = &ResponseFormat{Type: "xml"} }, "response_format.type"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := validReq()
			tc.mutate(r)
			err := r.Validate()
			assertInvalid(t, err, tc.param)
		})
	}
}

func assertInvalid(t *testing.T, err *APIError, param string) {
	t.Helper()
	if err == nil {
		t.Fatal("expected validation error")
	}
	if err.Status != 400 || err.Type != "invalid_request_error" {
		t.Fatalf("got status=%d type=%q", err.Status, err.Type)
	}
	if err.Param != param {
		t.Fatalf("param = %q, want %q (message %q)", err.Param, param, err.Message)
	}
}

func TestParseChatCompletionRequest(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		r, err := ParseChatCompletionRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if r.Model != "m" || len(r.Messages) != 1 {
			t.Fatalf("got %+v", r)
		}
	})
	t.Run("malformed json", func(t *testing.T) {
		_, err := ParseChatCompletionRequest([]byte(`{"model":`))
		assertInvalid(t, err, "")
		if !strings.HasPrefix(err.Message, "invalid JSON body: ") {
			t.Fatalf("message = %q", err.Message)
		}
	})
	t.Run("unknown routepilot field", func(t *testing.T) {
		_, err := ParseChatCompletionRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"foo":1}}`))
		assertInvalid(t, err, "routepilot")
	})
	t.Run("known routepilot fields", func(t *testing.T) {
		r, err := ParseChatCompletionRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"min_quality":0.7}}`))
		if err != nil {
			t.Fatal(err)
		}
		if r.RoutePilot == nil || r.RoutePilot.MinQuality == nil || *r.RoutePilot.MinQuality != 0.7 {
			t.Fatalf("got %+v", r.RoutePilot)
		}
	})
	t.Run("validation runs", func(t *testing.T) {
		_, err := ParseChatCompletionRequest([]byte(`{"model":"","messages":[{"role":"user","content":"hi"}]}`))
		assertInvalid(t, err, "model")
	})
	t.Run("invalid content type", func(t *testing.T) {
		_, err := ParseChatCompletionRequest([]byte(`{"model":"m","messages":[{"role":"user","content":42}]}`))
		assertInvalid(t, err, "")
	})
}

func TestEffectiveMaxTokensPrecedence(t *testing.T) {
	r := validReq()
	if v, ok := r.EffectiveMaxTokens(); ok || v != 0 {
		t.Fatalf("none set: got (%d,%v)", v, ok)
	}
	r.MaxTokens = ptr(10)
	if v, ok := r.EffectiveMaxTokens(); !ok || v != 10 {
		t.Fatalf("max_tokens only: got (%d,%v)", v, ok)
	}
	r.MaxCompletionTokens = ptr(20)
	if v, ok := r.EffectiveMaxTokens(); !ok || v != 20 {
		t.Fatalf("both: got (%d,%v)", v, ok)
	}
}

func TestCloneIsDeep(t *testing.T) {
	orig := validReq()
	orig.StreamOptions = &StreamOptions{IncludeUsage: false}
	orig.RoutePilot = &RoutingOptions{MinQuality: ptr(0.5)}
	orig.Extra = map[string]json.RawMessage{"k": json.RawMessage(`1`)}
	orig.MaxTokens = ptr(5)
	orig.Stop = json.RawMessage(`["a"]`)

	c := orig.Clone()
	c.Model = "other"
	c.StreamOptions.IncludeUsage = true
	c.RoutePilot.MinQuality = ptr(0.9)
	c.Extra["k"] = json.RawMessage(`2`)
	c.Extra["new"] = json.RawMessage(`3`)
	*c.MaxTokens = 99
	c.Messages[0].Role = "system"
	c.Stop[2] = 'z'

	if orig.Model != "m" {
		t.Fatal("Model leaked")
	}
	if orig.StreamOptions.IncludeUsage {
		t.Fatal("StreamOptions leaked")
	}
	if *orig.RoutePilot.MinQuality != 0.5 {
		t.Fatal("RoutePilot leaked")
	}
	if string(orig.Extra["k"]) != "1" || len(orig.Extra) != 1 {
		t.Fatalf("Extra leaked: %v", orig.Extra)
	}
	if *orig.MaxTokens != 5 {
		t.Fatal("MaxTokens leaked")
	}
	if orig.Messages[0].Role != "user" {
		t.Fatal("Messages leaked")
	}
	if string(orig.Stop) != `["a"]` {
		t.Fatalf("Stop leaked: %s", orig.Stop)
	}
}

func TestCloneNil(t *testing.T) {
	var r *ChatCompletionRequest
	if r.Clone() != nil {
		t.Fatal("Clone of nil should be nil")
	}
}
