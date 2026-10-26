package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// defaultMaxTokens is used when the request sets no output cap, because the
// Messages API requires max_tokens.
const defaultMaxTokens = 4096

// messagesRequest is the Anthropic Messages API request body.
type messagesRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	System        string          `json:"system,omitempty"`
	Messages      []message       `json:"messages"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Tools         []tool          `json:"tools,omitempty"`
	ToolChoice    json.RawMessage `json:"tool_choice,omitempty"`
}

type message struct {
	Role    string  `json:"role"`
	Content []block `json:"content"`
}

// block is one content block. Which fields are set depends on Type.
type block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Source    *imageSource    `json:"source,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// translateRequest converts an OpenAI chat request into a Messages request. It
// reads req without modifying it. Every error is invalid client input.
func translateRequest(req *openai.ChatCompletionRequest) (*messagesRequest, *openai.APIError) {
	out := &messagesRequest{
		Model:       req.Model,
		MaxTokens:   defaultMaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}
	if n, ok := req.EffectiveMaxTokens(); ok {
		out.MaxTokens = n
	}

	var system []string
	// lastIsToolResults is true while the final output message is a user turn
	// made only of tool results, so a following tool message joins it.
	lastIsToolResults := false
	for i, m := range req.Messages {
		param := "messages"
		switch m.Role {
		case "system", "developer":
			if text := m.Content.PlainText(); text != "" {
				system = append(system, text)
			}
			lastIsToolResults = false
		case "user", "assistant":
			blocks, apiErr := contentBlocks(m)
			if apiErr != nil {
				return nil, apiErr
			}
			if m.Role == "assistant" {
				calls, apiErr := toolUseBlocks(m.ToolCalls)
				if apiErr != nil {
					return nil, apiErr
				}
				blocks = append(blocks, calls...)
			}
			if len(blocks) == 0 {
				return nil, openai.ErrInvalidRequest(fmt.Sprintf("messages[%d] has no content", i), param)
			}
			out.Messages = append(out.Messages, message{Role: m.Role, Content: blocks})
			lastIsToolResults = false
		case "tool":
			res := block{Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content.PlainText()}
			if lastIsToolResults {
				last := &out.Messages[len(out.Messages)-1]
				last.Content = append(last.Content, res)
			} else {
				out.Messages = append(out.Messages, message{Role: "user", Content: []block{res}})
			}
			lastIsToolResults = true
		default:
			return nil, openai.ErrInvalidRequest(fmt.Sprintf("invalid role %q", m.Role), param)
		}
	}
	if len(out.Messages) == 0 {
		return nil, openai.ErrInvalidRequest("messages must contain at least one user or assistant message", "messages")
	}
	out.System = strings.Join(system, "\n\n")

	var apiErr *openai.APIError
	if out.StopSequences, apiErr = stopSequences(req.Stop); apiErr != nil {
		return nil, apiErr
	}
	if out.Tools, apiErr = translateTools(req.Tools); apiErr != nil {
		return nil, apiErr
	}
	if out.ToolChoice, apiErr = translateToolChoice(req.ToolChoice); apiErr != nil {
		return nil, apiErr
	}
	return out, nil
}

// contentBlocks converts the content of a user or assistant message.
func contentBlocks(m openai.Message) ([]block, *openai.APIError) {
	if m.Content.Text != nil {
		if *m.Content.Text == "" {
			return nil, nil
		}
		return []block{{Type: "text", Text: *m.Content.Text}}, nil
	}
	var blocks []block
	for _, p := range m.Content.Parts {
		switch {
		case p.Type == "text":
			if p.Text != "" {
				blocks = append(blocks, block{Type: "text", Text: p.Text})
			}
		case p.Type == "image_url":
			if m.Role != "user" {
				return nil, openai.ErrInvalidRequest("images are only supported in user messages", "messages")
			}
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return nil, openai.ErrInvalidRequest("image_url part requires a url", "messages")
			}
			src, apiErr := imageSourceFor(p.ImageURL.URL)
			if apiErr != nil {
				return nil, apiErr
			}
			blocks = append(blocks, block{Type: "image", Source: src})
		default:
			return nil, openai.ErrInvalidRequest(fmt.Sprintf("unsupported content part type %q", p.Type), "messages")
		}
	}
	return blocks, nil
}

// imageSourceFor maps an image URL to an Anthropic image source. data: URLs
// must be of the form data:<media_type>;base64,<data>.
func imageSourceFor(u string) (*imageSource, *openai.APIError) {
	rest, ok := strings.CutPrefix(u, "data:")
	if !ok {
		return &imageSource{Type: "url", URL: u}, nil
	}
	header, data, found := strings.Cut(rest, ",")
	mediaType, hasB64 := strings.CutSuffix(header, ";base64")
	if !found || !hasB64 || mediaType == "" || data == "" {
		return nil, openai.ErrInvalidRequest("image data URL must look like data:<media_type>;base64,<data>", "messages")
	}
	return &imageSource{Type: "base64", MediaType: mediaType, Data: data}, nil
}

func toolUseBlocks(calls []openai.ToolCall) ([]block, *openai.APIError) {
	var blocks []block
	for _, tc := range calls {
		input, apiErr := toolInput(tc)
		if apiErr != nil {
			return nil, apiErr
		}
		blocks = append(blocks, block{Type: "tool_use", ID: tc.ID, Name: tc.Function.Name, Input: input})
	}
	return blocks, nil
}

// toolInput parses tool call arguments, which Anthropic requires to be a JSON
// object. Empty arguments become {}.
func toolInput(tc openai.ToolCall) (json.RawMessage, *openai.APIError) {
	args := strings.TrimSpace(tc.Function.Arguments)
	if args == "" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid([]byte(args)) || args[0] != '{' {
		return nil, openai.ErrInvalidRequest(
			fmt.Sprintf("arguments of tool call %q must be a JSON object", tc.ID), "messages")
	}
	return json.RawMessage(args), nil
}

func stopSequences(raw json.RawMessage) ([]string, *openai.APIError) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		return many, nil
	}
	return nil, openai.ErrInvalidRequest("stop must be a string or an array of strings", "stop")
}

func translateTools(in []openai.Tool) ([]tool, *openai.APIError) {
	var out []tool
	for _, t := range in {
		if t.Type != "function" {
			return nil, openai.ErrInvalidRequest(fmt.Sprintf("unsupported tool type %q", t.Type), "tools")
		}
		schema := t.Function.Parameters
		if len(bytes.TrimSpace(schema)) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, tool{Name: t.Function.Name, Description: t.Function.Description, InputSchema: schema})
	}
	return out, nil
}

func translateToolChoice(raw json.RawMessage) (json.RawMessage, *openai.APIError) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	invalid := openai.ErrInvalidRequest("tool_choice must be auto, required, none or a named function", "tool_choice")

	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		switch mode {
		case "auto":
			return json.RawMessage(`{"type":"auto"}`), nil
		case "required":
			return json.RawMessage(`{"type":"any"}`), nil
		case "none":
			return json.RawMessage(`{"type":"none"}`), nil
		}
		return nil, invalid
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &named) != nil || named.Type != "function" || named.Function.Name == "" {
		return nil, invalid
	}
	b, err := json.Marshal(map[string]string{"type": "tool", "name": named.Function.Name})
	if err != nil {
		return nil, invalid
	}
	return b, nil
}

// messagesResponse is the Anthropic Messages API response body.
type messagesResponse struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Content    []responseBlock `json:"content"`
	StopReason string          `json:"stop_reason"`
	Usage      usage           `json:"usage"`
}

type responseBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type usage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

// promptTokens counts cached input as prompt tokens.
func (u usage) promptTokens() int {
	return u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
}

func openaiUsage(prompt, completion int) *openai.Usage {
	return &openai.Usage{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: prompt + completion}
}

// translateResponse converts a Messages response body into a chat completion.
func translateResponse(body []byte) (*openai.ChatCompletion, *provider.Error) {
	var resp messagesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "invalid completion JSON", Err: err}
	}
	var text strings.Builder
	var calls []openai.ToolCall
	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			text.WriteString(b.Text)
		case "tool_use":
			args := "{}"
			if len(bytes.TrimSpace(b.Input)) > 0 {
				var buf bytes.Buffer
				if err := json.Compact(&buf, b.Input); err != nil {
					return nil, &provider.Error{Kind: provider.KindProtocol, Message: "invalid tool_use input JSON", Err: err}
				}
				args = buf.String()
			}
			calls = append(calls, openai.ToolCall{
				ID: b.ID, Type: "function",
				Function: openai.FunctionCall{Name: b.Name, Arguments: args},
			})
		}
	}
	msg := openai.Message{Role: "assistant", ToolCalls: calls}
	if text.Len() > 0 || len(calls) == 0 {
		s := text.String()
		msg.Content = openai.Content{Text: &s}
	}
	return &openai.ChatCompletion{
		ID:      resp.ID,
		Object:  openai.ObjectChatCompletion,
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []openai.Choice{{Index: 0, Message: msg, FinishReason: mapStopReason(resp.StopReason)}},
		Usage:   openaiUsage(resp.Usage.promptTokens(), resp.Usage.OutputTokens),
	}, nil
}

// mapStopReason converts an Anthropic stop_reason to an OpenAI finish_reason.
// Reasons without an OpenAI equivalent pass through unchanged.
func mapStopReason(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	}
	return r
}

// streamTranslator converts Messages stream events into chat completion
// chunks. The SSE event name selects the handler; unknown names are ignored so
// new event types do not break the adapter.
type streamTranslator struct {
	id           string
	model        string
	created      int64
	promptTokens int
	outputTokens int
	toolCount    int
	toolIndex    map[int]int // content block index -> tool call index
}

func newStreamTranslator() *streamTranslator {
	return &streamTranslator{toolIndex: map[int]int{}}
}

// handle processes one event. It returns the chunk to emit (nil for none) and
// whether the stream ended cleanly with this event.
func (s *streamTranslator) handle(name string, data []byte) (chunk *openai.ChatCompletionChunk, done bool, err *provider.Error) {
	switch name {
	case "message_start":
		var ev struct {
			Message struct {
				ID    string `json:"id"`
				Model string `json:"model"`
				Usage usage  `json:"usage"`
			} `json:"message"`
		}
		if perr := decodeEvent(name, data, &ev); perr != nil {
			return nil, false, perr
		}
		s.id, s.model, s.created = ev.Message.ID, ev.Message.Model, time.Now().Unix()
		s.promptTokens = ev.Message.Usage.promptTokens()
		empty := ""
		return s.chunk(openai.Delta{Role: "assistant", Content: &empty}, nil), false, nil

	case "content_block_start":
		var ev struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
		}
		if perr := decodeEvent(name, data, &ev); perr != nil {
			return nil, false, perr
		}
		if ev.ContentBlock.Type != "tool_use" {
			return nil, false, nil
		}
		idx := s.toolCount
		s.toolCount++
		s.toolIndex[ev.Index] = idx
		call := openai.ToolCall{
			Index: &idx, ID: ev.ContentBlock.ID, Type: "function",
			Function: openai.FunctionCall{Name: ev.ContentBlock.Name, Arguments: ""},
		}
		return s.chunk(openai.Delta{ToolCalls: []openai.ToolCall{call}}, nil), false, nil

	case "content_block_delta":
		var ev struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if perr := decodeEvent(name, data, &ev); perr != nil {
			return nil, false, perr
		}
		switch ev.Delta.Type {
		case "text_delta":
			if ev.Delta.Text == "" {
				return nil, false, nil
			}
			text := ev.Delta.Text
			return s.chunk(openai.Delta{Content: &text}, nil), false, nil
		case "input_json_delta":
			idx, ok := s.toolIndex[ev.Index]
			if !ok {
				return nil, false, &provider.Error{Kind: provider.KindProtocol, Message: "input_json_delta for a block that is not a tool_use"}
			}
			if ev.Delta.PartialJSON == "" {
				return nil, false, nil
			}
			call := openai.ToolCall{Index: &idx, Function: openai.FunctionCall{Arguments: ev.Delta.PartialJSON}}
			return s.chunk(openai.Delta{ToolCalls: []openai.ToolCall{call}}, nil), false, nil
		}
		return nil, false, nil

	case "message_delta":
		var ev struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage usage `json:"usage"`
		}
		if perr := decodeEvent(name, data, &ev); perr != nil {
			return nil, false, perr
		}
		s.outputTokens = ev.Usage.OutputTokens
		if ev.Delta.StopReason == "" {
			return nil, false, nil
		}
		reason := mapStopReason(ev.Delta.StopReason)
		return s.chunk(openai.Delta{}, &reason), false, nil

	case "message_stop":
		ch := s.chunk(openai.Delta{}, nil)
		ch.Choices = []openai.ChunkChoice{}
		ch.Usage = openaiUsage(s.promptTokens, s.outputTokens)
		return ch, true, nil

	case "error":
		var ev struct {
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		if perr := decodeEvent(name, data, &ev); perr != nil {
			return nil, false, perr
		}
		return nil, false, &provider.Error{
			Kind:    provider.KindProtocol,
			Message: "upstream stream error (" + safeErrorType(ev.Error.Type) + ")",
		}
	}
	return nil, false, nil
}

func (s *streamTranslator) chunk(d openai.Delta, finish *string) *openai.ChatCompletionChunk {
	return &openai.ChatCompletionChunk{
		ID: s.id, Object: openai.ObjectChatCompletionChunk, Created: s.created, Model: s.model,
		Choices: []openai.ChunkChoice{{Index: 0, Delta: d, FinishReason: finish}},
	}
}

func decodeEvent(name string, data []byte, v any) *provider.Error {
	if err := json.Unmarshal(data, v); err != nil {
		return &provider.Error{Kind: provider.KindProtocol, Message: "invalid " + name + " event JSON", Err: err}
	}
	return nil
}

// safeErrorType returns an upstream error type for use in a message, or
// "unknown" when it is not a short snake_case identifier.
func safeErrorType(t string) string {
	if t == "" || len(t) > 64 {
		return "unknown"
	}
	for i := 0; i < len(t); i++ {
		if c := t[i]; c != '_' && (c < 'a' || c > 'z') {
			return "unknown"
		}
	}
	return t
}
