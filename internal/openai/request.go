package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ChatCompletionRequest is an OpenAI chat completion request plus the
// RoutePilot routing extension. Unknown top-level keys are kept in Extra and
// written back on marshal so the gateway can pass provider-specific options
// through untouched.
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	N                   *int            `json:"n,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	User                string          `json:"user,omitempty"`
	RoutePilot          *RoutingOptions `json:"routepilot,omitempty"`

	// Extra holds unknown top-level fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// StreamOptions controls streaming behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// RoutingOptions are per-request routing hints under the "routepilot" key.
type RoutingOptions struct {
	MinQuality *float64 `json:"min_quality,omitempty"`
	SLOTTFTMs  *int     `json:"slo_ttft_ms,omitempty"`
	SLOE2EMs   *int     `json:"slo_e2e_ms,omitempty"`
	TimeoutMs  *int     `json:"timeout_ms,omitempty"`
}

// ResponseFormat selects the output format.
type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// Tool is a tool the model may call.
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef describes a callable function.
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ToolCall is a model-issued tool call. Index is set on streamed deltas.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall is the function name and JSON-encoded arguments of a tool call.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// Message is one chat message. Content is always emitted (null when empty)
// because assistant messages carrying tool_calls may have null content.
type Message struct {
	Role       string     `json:"role"`
	Content    Content    `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Content is message content: a JSON string, an array of parts, or null.
type Content struct {
	Text  *string
	Parts []ContentPart
}

// ContentPart is one element of array-form content.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL references an image in a content part.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// UnmarshalJSON accepts a string, an array of parts, or null.
func (c *Content) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	*c = Content{}
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return fmt.Errorf("decode content string: %w", err)
		}
		c.Text = &s
	case '[':
		var parts []ContentPart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return fmt.Errorf("decode content parts: %w", err)
		}
		if parts == nil {
			parts = []ContentPart{}
		}
		c.Parts = parts
	default:
		return errors.New("content must be a string, an array of parts, or null")
	}
	return nil
}

// MarshalJSON writes the string form, the parts form, or null.
func (c Content) MarshalJSON() ([]byte, error) {
	switch {
	case c.Text != nil:
		return json.Marshal(*c.Text)
	case c.Parts != nil:
		return json.Marshal(c.Parts)
	default:
		return []byte("null"), nil
	}
}

// PlainText returns the string content, or the text parts joined by "\n".
func (c Content) PlainText() string {
	if c.Text != nil {
		return *c.Text
	}
	var texts []string
	for _, p := range c.Parts {
		if p.Type == "text" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// HasImage reports whether any part is an image.
func (c Content) HasImage() bool {
	for _, p := range c.Parts {
		if p.Type == "image_url" || p.ImageURL != nil {
			return true
		}
	}
	return false
}

// requestAlias has the same fields and tags as ChatCompletionRequest but none
// of its methods, so the custom (un)marshalers can reuse default behavior.
type requestAlias ChatCompletionRequest

var knownRequestKeys = map[string]struct{}{
	"model": {}, "messages": {}, "max_tokens": {}, "max_completion_tokens": {},
	"temperature": {}, "top_p": {}, "n": {}, "stream": {}, "stream_options": {},
	"stop": {}, "tools": {}, "tool_choice": {}, "response_format": {},
	"seed": {}, "user": {}, "routepilot": {},
}

// UnmarshalJSON decodes known fields, rejects unknown fields inside
// "routepilot", and stores unknown top-level fields in Extra.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	aux := struct {
		*requestAlias
		RoutePilot json.RawMessage `json:"routepilot"`
	}{requestAlias: (*requestAlias)(r)}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	r.RoutePilot = nil
	if raw := bytes.TrimSpace(aux.RoutePilot); len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		var opts RoutingOptions
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&opts); err != nil {
			return ErrInvalidRequest("invalid routepilot options: "+err.Error(), "routepilot")
		}
		r.RoutePilot = &opts
	}

	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return err
	}
	r.Extra = nil
	for k, v := range all {
		if _, known := knownRequestKeys[k]; known {
			continue
		}
		if r.Extra == nil {
			r.Extra = make(map[string]json.RawMessage)
		}
		r.Extra[k] = append(json.RawMessage(nil), v...)
	}
	return nil
}

// MarshalJSON writes known fields followed by Extra. Extra never overrides a
// known field.
func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	known, err := json.Marshal(requestAlias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return known, nil
	}
	merged := make(map[string]json.RawMessage, len(r.Extra)+8)
	if err := json.Unmarshal(known, &merged); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		if _, isKnown := knownRequestKeys[k]; isKnown {
			continue
		}
		merged[k] = v
	}
	return json.Marshal(merged)
}

// ParseChatCompletionRequest decodes body and validates it.
func ParseChatCompletionRequest(body []byte) (*ChatCompletionRequest, *APIError) {
	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		var apiErr *APIError
		if errors.As(err, &apiErr) {
			return nil, apiErr
		}
		return nil, ErrInvalidRequest("invalid JSON body: "+err.Error(), "")
	}
	if apiErr := req.Validate(); apiErr != nil {
		return nil, apiErr
	}
	return &req, nil
}

// Validate checks the request against the gateway's rules.
func (r *ChatCompletionRequest) Validate() *APIError {
	if r.Model == "" {
		return ErrInvalidRequest("model is required", "model")
	}
	if len(r.Messages) == 0 {
		return ErrInvalidRequest("messages must contain at least one message", "messages")
	}
	for i, m := range r.Messages {
		switch m.Role {
		case "system", "developer", "user", "assistant":
		case "tool":
			if m.ToolCallID == "" {
				return ErrInvalidRequest("tool messages require tool_call_id", fmt.Sprintf("messages[%d].tool_call_id", i))
			}
		default:
			return ErrInvalidRequest(fmt.Sprintf("invalid role %q", m.Role), fmt.Sprintf("messages[%d].role", i))
		}
	}
	if r.N != nil && *r.N != 1 {
		return ErrInvalidRequest("only n=1 is supported", "n")
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 2) {
		return ErrInvalidRequest("temperature must be between 0 and 2", "temperature")
	}
	if r.TopP != nil && (*r.TopP <= 0 || *r.TopP > 1) {
		return ErrInvalidRequest("top_p must be greater than 0 and at most 1", "top_p")
	}
	if r.MaxTokens != nil && *r.MaxTokens <= 0 {
		return ErrInvalidRequest("max_tokens must be greater than 0", "max_tokens")
	}
	if r.MaxCompletionTokens != nil && *r.MaxCompletionTokens <= 0 {
		return ErrInvalidRequest("max_completion_tokens must be greater than 0", "max_completion_tokens")
	}
	if rp := r.RoutePilot; rp != nil {
		if rp.MinQuality != nil && (*rp.MinQuality < 0 || *rp.MinQuality > 1) {
			return ErrInvalidRequest("min_quality must be between 0 and 1", "routepilot.min_quality")
		}
		if rp.SLOTTFTMs != nil && *rp.SLOTTFTMs <= 0 {
			return ErrInvalidRequest("slo_ttft_ms must be greater than 0", "routepilot.slo_ttft_ms")
		}
		if rp.SLOE2EMs != nil && *rp.SLOE2EMs <= 0 {
			return ErrInvalidRequest("slo_e2e_ms must be greater than 0", "routepilot.slo_e2e_ms")
		}
		if rp.TimeoutMs != nil && *rp.TimeoutMs <= 0 {
			return ErrInvalidRequest("timeout_ms must be greater than 0", "routepilot.timeout_ms")
		}
	}
	if rf := r.ResponseFormat; rf != nil {
		switch rf.Type {
		case "text", "json_object", "json_schema":
		default:
			return ErrInvalidRequest(fmt.Sprintf("invalid response_format type %q", rf.Type), "response_format.type")
		}
	}
	return nil
}

// EffectiveMaxTokens returns the output token cap; max_completion_tokens wins
// over max_tokens. The bool is false when neither is set.
func (r *ChatCompletionRequest) EffectiveMaxTokens() (int, bool) {
	if r.MaxCompletionTokens != nil {
		return *r.MaxCompletionTokens, true
	}
	if r.MaxTokens != nil {
		return *r.MaxTokens, true
	}
	return 0, false
}

// Clone returns a deep copy that is safe to mutate independently.
func (r *ChatCompletionRequest) Clone() *ChatCompletionRequest {
	if r == nil {
		return nil
	}
	c := *r
	c.Messages = cloneMessages(r.Messages)
	c.MaxTokens = clonePtr(r.MaxTokens)
	c.MaxCompletionTokens = clonePtr(r.MaxCompletionTokens)
	c.Temperature = clonePtr(r.Temperature)
	c.TopP = clonePtr(r.TopP)
	c.N = clonePtr(r.N)
	c.StreamOptions = clonePtr(r.StreamOptions)
	c.Stop = cloneRaw(r.Stop)
	c.Tools = cloneTools(r.Tools)
	c.ToolChoice = cloneRaw(r.ToolChoice)
	if r.ResponseFormat != nil {
		rf := *r.ResponseFormat
		rf.JSONSchema = cloneRaw(rf.JSONSchema)
		c.ResponseFormat = &rf
	}
	c.Seed = clonePtr(r.Seed)
	c.RoutePilot = clonePtr(r.RoutePilot)
	if r.RoutePilot != nil {
		c.RoutePilot.MinQuality = clonePtr(r.RoutePilot.MinQuality)
		c.RoutePilot.SLOTTFTMs = clonePtr(r.RoutePilot.SLOTTFTMs)
		c.RoutePilot.SLOE2EMs = clonePtr(r.RoutePilot.SLOE2EMs)
		c.RoutePilot.TimeoutMs = clonePtr(r.RoutePilot.TimeoutMs)
	}
	if r.Extra != nil {
		c.Extra = make(map[string]json.RawMessage, len(r.Extra))
		for k, v := range r.Extra {
			c.Extra[k] = cloneRaw(v)
		}
	}
	return &c
}

func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func cloneRaw(b json.RawMessage) json.RawMessage {
	if b == nil {
		return nil
	}
	return append(json.RawMessage{}, b...)
}

func cloneMessages(in []Message) []Message {
	if in == nil {
		return nil
	}
	out := make([]Message, len(in))
	for i, m := range in {
		m.Content.Text = clonePtr(m.Content.Text)
		if m.Content.Parts != nil {
			parts := make([]ContentPart, len(m.Content.Parts))
			for j, p := range m.Content.Parts {
				p.ImageURL = clonePtr(p.ImageURL)
				parts[j] = p
			}
			m.Content.Parts = parts
		}
		if m.ToolCalls != nil {
			calls := make([]ToolCall, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				tc.Index = clonePtr(tc.Index)
				calls[j] = tc
			}
			m.ToolCalls = calls
		}
		out[i] = m
	}
	return out
}

func cloneTools(in []Tool) []Tool {
	if in == nil {
		return nil
	}
	out := make([]Tool, len(in))
	for i, t := range in {
		t.Function.Parameters = cloneRaw(t.Function.Parameters)
		t.Function.Strict = clonePtr(t.Function.Strict)
		out[i] = t
	}
	return out
}
