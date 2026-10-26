package anthropic

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/omorsi45/routepilot/internal/openai"
)

func strp(s string) *string { return &s }

func textMsg(role, text string) openai.Message {
	return openai.Message{Role: role, Content: openai.Content{Text: strp(text)}}
}

// toMap marshals the translated request and decodes it for structural comparison.
func toMap(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func mustTranslate(t *testing.T, req *openai.ChatCompletionRequest) map[string]any {
	t.Helper()
	out, apiErr := translateRequest(req)
	if apiErr != nil {
		t.Fatalf("translateRequest: %v", apiErr)
	}
	return toMap(t, out)
}

func mustInvalid(t *testing.T, req *openai.ChatCompletionRequest, param string) {
	t.Helper()
	out, apiErr := translateRequest(req)
	if apiErr == nil {
		t.Fatalf("expected error, got %+v", out)
	}
	if apiErr.Status != 400 || apiErr.Type != "invalid_request_error" || apiErr.Param != param {
		t.Errorf("unexpected error: %+v", apiErr)
	}
}

func TestTranslateRequestSystemToolsAndToolResults(t *testing.T) {
	temp, topP, maxTok := 0.5, 0.9, 100
	req := &openai.ChatCompletionRequest{
		Model: "claude-fixture-1",
		Messages: []openai.Message{
			textMsg("system", "You are helpful."),
			textMsg("developer", "Answer briefly."),
			textMsg("user", "Weather in Paris and Rome?"),
			{Role: "assistant", Content: openai.Content{Text: strp("Checking.")}, ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: "function", Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city": "Paris"}`}},
				{ID: "call_2", Type: "function", Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city":"Rome"}`}},
			}},
			{Role: "tool", ToolCallID: "call_1", Content: openai.Content{Text: strp("18C")}},
			{Role: "tool", ToolCallID: "call_2", Content: openai.Content{Text: strp("24C")}},
			{Role: "user", Content: openai.Content{Parts: []openai.ContentPart{
				{Type: "text", Text: "Compare them."},
				{Type: "image_url", ImageURL: &openai.ImageURL{URL: "data:image/png;base64,QUJD"}},
				{Type: "image_url", ImageURL: &openai.ImageURL{URL: "https://example.com/a.png"}},
			}}},
		},
		MaxTokens:   &maxTok,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        json.RawMessage(`["END","STOP"]`),
		Tools: []openai.Tool{{Type: "function", Function: openai.FunctionDef{
			Name: "get_weather", Description: "Get weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}}},
		ToolChoice:     json.RawMessage(`"required"`),
		ResponseFormat: &openai.ResponseFormat{Type: "json_object"},
		User:           "u1",
		Extra:          map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"1":2}`)},
	}

	got := mustTranslate(t, req)
	want := map[string]any{
		"model":          "claude-fixture-1",
		"max_tokens":     float64(100),
		"system":         "You are helpful.\n\nAnswer briefly.",
		"temperature":    0.5,
		"top_p":          0.9,
		"stop_sequences": []any{"END", "STOP"},
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "Weather in Paris and Rome?"},
			}},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "Checking."},
				map[string]any{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": map[string]any{"city": "Paris"}},
				map[string]any{"type": "tool_use", "id": "call_2", "name": "get_weather", "input": map[string]any{"city": "Rome"}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": "18C"},
				map[string]any{"type": "tool_result", "tool_use_id": "call_2", "content": "24C"},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "Compare them."},
				map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "QUJD"}},
				map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": "https://example.com/a.png"}},
			}},
		},
		"tools": []any{map[string]any{
			"name": "get_weather", "description": "Get weather",
			"input_schema": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}},
		"tool_choice": map[string]any{"type": "any"},
	}
	if !reflect.DeepEqual(got, want) {
		gb, _ := json.MarshalIndent(got, "", "  ")
		wb, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("translated request mismatch\n got: %s\nwant: %s", gb, wb)
	}
}

func TestTranslateDoesNotMutateRequest(t *testing.T) {
	req := &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{textMsg("system", "s"), textMsg("user", "u")}}
	before, _ := json.Marshal(req)
	mustTranslate(t, req)
	after, _ := json.Marshal(req)
	if string(before) != string(after) {
		t.Errorf("request mutated:\nbefore %s\nafter  %s", before, after)
	}
}

func TestTranslateToolChoice(t *testing.T) {
	tools := []openai.Tool{{Type: "function", Function: openai.FunctionDef{Name: "f"}}}
	cases := []struct {
		name   string
		choice string
		want   map[string]any
	}{
		{"auto", `"auto"`, map[string]any{"type": "auto"}},
		{"required", `"required"`, map[string]any{"type": "any"}},
		{"none", `"none"`, map[string]any{"type": "none"}},
		{"named", `{"type":"function","function":{"name":"f"}}`, map[string]any{"type": "tool", "name": "f"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := mustTranslate(t, &openai.ChatCompletionRequest{
				Model: "m", Messages: []openai.Message{textMsg("user", "hi")},
				Tools: tools, ToolChoice: json.RawMessage(tc.choice),
			})
			if !reflect.DeepEqual(got["tool_choice"], tc.want) {
				t.Errorf("tool_choice = %v, want %v", got["tool_choice"], tc.want)
			}
		})
	}
}

func TestTranslateInvalidToolChoice(t *testing.T) {
	for _, choice := range []string{`"bogus"`, `42`, `{"type":"function"}`, `{"type":"other","function":{"name":"f"}}`} {
		mustInvalid(t, &openai.ChatCompletionRequest{
			Model: "m", Messages: []openai.Message{textMsg("user", "hi")},
			ToolChoice: json.RawMessage(choice),
		}, "tool_choice")
	}
}

func TestTranslateInvalidToolArgs(t *testing.T) {
	for _, args := range []string{`{not json`, `[1,2]`, `"str"`} {
		mustInvalid(t, &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{
			textMsg("user", "hi"),
			{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "c", Type: "function", Function: openai.FunctionCall{Name: "f", Arguments: args}}}},
		}}, "messages")
	}
}

func TestTranslateEmptyToolArgsBecomeEmptyObject(t *testing.T) {
	got := mustTranslate(t, &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{
		textMsg("user", "hi"),
		{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "c", Type: "function", Function: openai.FunctionCall{Name: "f", Arguments: ""}}}},
	}})
	blocks := got["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if in := blocks[0].(map[string]any)["input"]; !reflect.DeepEqual(in, map[string]any{}) {
		t.Errorf("input = %#v, want {}", in)
	}
}

func TestTranslateDefaultMaxTokens(t *testing.T) {
	base := func() *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{textMsg("user", "hi")}}
	}
	if got := mustTranslate(t, base())["max_tokens"]; got != float64(4096) {
		t.Errorf("default max_tokens = %v, want 4096", got)
	}
	mt, mct := 10, 20
	r := base()
	r.MaxTokens, r.MaxCompletionTokens = &mt, &mct
	if got := mustTranslate(t, r)["max_tokens"]; got != float64(20) {
		t.Errorf("max_tokens = %v, want 20 (max_completion_tokens wins)", got)
	}
}

func TestTranslateStop(t *testing.T) {
	mk := func(stop string) *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{textMsg("user", "hi")}, Stop: json.RawMessage(stop)}
	}
	if got := mustTranslate(t, mk(`"END"`))["stop_sequences"]; !reflect.DeepEqual(got, []any{"END"}) {
		t.Errorf("string stop -> %v", got)
	}
	if got := mustTranslate(t, mk(`["a","b"]`))["stop_sequences"]; !reflect.DeepEqual(got, []any{"a", "b"}) {
		t.Errorf("list stop -> %v", got)
	}
	if _, ok := mustTranslate(t, mk(`null`))["stop_sequences"]; ok {
		t.Error("null stop should be omitted")
	}
	for _, bad := range []string{`42`, `[1,2]`, `{"a":1}`, `["a",2]`} {
		mustInvalid(t, mk(bad), "stop")
	}
}

func TestTranslateInvalidMessages(t *testing.T) {
	cases := map[string][]openai.Message{
		"malformed data url": {{Role: "user", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "image_url", ImageURL: &openai.ImageURL{URL: "data:image/png,QUJD"}}}}}},
		"empty data url payload": {{Role: "user", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "image_url", ImageURL: &openai.ImageURL{URL: "data:image/png;base64,"}}}}}},
		"unsupported part": {{Role: "user", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "input_audio"}}}}},
		"image without url": {{Role: "user", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "image_url"}}}}},
		"image from assistant": {textMsg("user", "hi"), {Role: "assistant", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "image_url", ImageURL: &openai.ImageURL{URL: "https://example.com/a.png"}}}}}},
		"only system": {textMsg("system", "s")},
		// Empty turns are skipped, so a request with nothing left is still a 400.
		"only empty user":      {textMsg("user", "")},
		"only refusal turn":    {{Role: "assistant", Content: openai.Content{Parts: []openai.ContentPart{{Type: "refusal"}}}}},
		"empty then null user": {textMsg("user", ""), {Role: "assistant"}},
	}
	for name, msgs := range cases {
		t.Run(name, func(t *testing.T) {
			mustInvalid(t, &openai.ChatCompletionRequest{Model: "m", Messages: msgs}, "messages")
		})
	}
}

func roles(t *testing.T, got map[string]any) []string {
	t.Helper()
	var out []string
	for _, m := range got["messages"].([]any) {
		out = append(out, m.(map[string]any)["role"].(string))
	}
	return out
}

func TestTranslateSkipsEmptyTurns(t *testing.T) {
	cases := map[string][]openai.Message{
		"empty assistant string": {
			textMsg("user", "one"), textMsg("assistant", ""), textMsg("user", "two")},
		"null assistant content": {
			textMsg("user", "one"), {Role: "assistant"}, textMsg("user", "two")},
		"refusal-only assistant": {
			textMsg("user", "one"),
			{Role: "assistant", Content: openai.Content{Parts: []openai.ContentPart{{Type: "refusal"}}}},
			textMsg("user", "two")},
	}
	for name, msgs := range cases {
		t.Run(name, func(t *testing.T) {
			got := mustTranslate(t, &openai.ChatCompletionRequest{Model: "m", Messages: msgs})
			if r := roles(t, got); !reflect.DeepEqual(r, []string{"user", "user"}) {
				t.Errorf("roles = %v, want [user user]", r)
			}
		})
	}
}

func TestTranslateDropsRefusalPartKeepsText(t *testing.T) {
	got := mustTranslate(t, &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{
		textMsg("user", "hi"),
		{Role: "assistant", Content: openai.Content{Parts: []openai.ContentPart{
			{Type: "refusal"}, {Type: "text", Text: "partial"}}}},
	}})
	msgs := got["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2: %v", len(msgs), msgs)
	}
	blocks := msgs[1].(map[string]any)["content"].([]any)
	if len(blocks) != 1 || blocks[0].(map[string]any)["text"] != "partial" {
		t.Errorf("blocks = %v, want one text block", blocks)
	}
}

func TestTranslateNullParametersGetDefaultSchema(t *testing.T) {
	got := mustTranslate(t, &openai.ChatCompletionRequest{
		Model: "m", Messages: []openai.Message{textMsg("user", "hi")},
		Tools: []openai.Tool{{Type: "function", Function: openai.FunctionDef{Name: "ping", Parameters: json.RawMessage(` null `)}}},
	})
	schema := got["tools"].([]any)[0].(map[string]any)["input_schema"]
	if !reflect.DeepEqual(schema, map[string]any{"type": "object", "properties": map[string]any{}}) {
		t.Errorf("input_schema = %#v", schema)
	}
}

func TestTranslateInvalidTool(t *testing.T) {
	mustInvalid(t, &openai.ChatCompletionRequest{
		Model: "m", Messages: []openai.Message{textMsg("user", "hi")},
		Tools: []openai.Tool{{Type: "retrieval"}},
	}, "tools")
}

func TestTranslateToolWithoutSchemaGetsDefault(t *testing.T) {
	got := mustTranslate(t, &openai.ChatCompletionRequest{
		Model: "m", Messages: []openai.Message{textMsg("user", "hi")},
		Tools: []openai.Tool{{Type: "function", Function: openai.FunctionDef{Name: "ping"}}},
	})
	schema := got["tools"].([]any)[0].(map[string]any)["input_schema"]
	if !reflect.DeepEqual(schema, map[string]any{"type": "object", "properties": map[string]any{}}) {
		t.Errorf("input_schema = %#v", schema)
	}
}

func TestToolResultsMergeOnlyWhenConsecutive(t *testing.T) {
	// Tool results separated by an assistant turn stay in separate user messages.
	got := mustTranslate(t, &openai.ChatCompletionRequest{Model: "m", Messages: []openai.Message{
		textMsg("user", "hi"),
		{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "c1", Type: "function", Function: openai.FunctionCall{Name: "f", Arguments: "{}"}}}},
		{Role: "tool", ToolCallID: "c1", Content: openai.Content{Text: strp("r1")}},
		textMsg("assistant", "done"),
		{Role: "tool", ToolCallID: "c2", Content: openai.Content{Text: strp("r2")}},
	}})
	msgs := got["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("got %d messages, want 5: %v", len(msgs), msgs)
	}
}

func TestResponseFixtureMapping(t *testing.T) {
	raw, err := os.ReadFile("testdata/message_tool.json")
	if err != nil {
		t.Fatal(err)
	}
	got, perr := translateResponse(raw)
	if perr != nil {
		t.Fatal(perr)
	}
	if got.ID != "msg_01MessageFixture" || got.Model != "claude-fixture-1" || got.Object != openai.ObjectChatCompletion {
		t.Errorf("unexpected header: %+v", got)
	}
	c := got.Choices[0]
	if c.FinishReason != "tool_calls" || c.Message.Role != "assistant" || c.Message.Content.PlainText() != "I will look that up." {
		t.Errorf("unexpected choice: %+v", c)
	}
	if len(c.Message.ToolCalls) != 2 {
		t.Fatalf("tool calls = %+v", c.Message.ToolCalls)
	}
	tc := c.Message.ToolCalls[0]
	if tc.ID != "toolu_01FixtureA" || tc.Type != "function" || tc.Function.Name != "get_weather" ||
		tc.Function.Arguments != `{"city":"Paris","units":"metric"}` {
		t.Errorf("tool call 0 = %+v", tc)
	}
	if got.Choices[0].Message.ToolCalls[1].Function.Arguments != `{}` {
		t.Errorf("tool call 1 args = %q", got.Choices[0].Message.ToolCalls[1].Function.Arguments)
	}
	if u := got.Usage; u == nil || u.PromptTokens != 125 || u.CompletionTokens != 40 || u.TotalTokens != 165 {
		t.Errorf("usage = %+v", got.Usage)
	}
}

func TestFinishReasonMapping(t *testing.T) {
	cases := map[string]string{
		"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
		"tool_use": "tool_calls", "refusal": "refusal", "pause_turn": "pause_turn",
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}
