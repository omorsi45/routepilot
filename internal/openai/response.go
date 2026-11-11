package openai

import "encoding/json"

// Object type constants for completion responses.
const (
	ObjectChatCompletion      = "chat.completion"
	ObjectChatCompletionChunk = "chat.completion.chunk"
)

// ChatCompletion is a non-streaming chat completion response.
type ChatCompletion struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// Choice is one completion alternative.
type Choice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// Usage reports token counts.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatCompletionChunk is one streamed chat completion event.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChunkChoice is one streamed alternative. FinishReason is null until the end.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}

// Delta is the incremental content of a streamed choice.
type Delta struct {
	Role      string     `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// Extra holds unknown fields, preserved on marshal.
	Extra map[string]json.RawMessage `json:"-"`
}
