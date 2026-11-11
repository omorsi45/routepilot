package openai

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Fields outside the typed subset must survive a pass through the gateway, in
// both directions. Types that can carry such fields hold them in an Extra map
// (unknown JSON key to raw value) that is filled on decode and written back on
// encode. The helpers below are shared by all of them and by
// ChatCompletionRequest.

// keySet is a set of known JSON keys. Lookup is case-insensitive because
// encoding/json matches field names that way, so a case variant of a known key
// is decoded into the typed field and must never also land in Extra.
type keySet []string

func (k keySet) has(key string) bool {
	for _, known := range k {
		if strings.EqualFold(key, known) {
			return true
		}
	}
	return false
}

// collectExtra returns the members of the JSON object in data whose keys are
// not in known, or nil when there are none. Values are copied.
func collectExtra(data []byte, known keySet) (map[string]json.RawMessage, error) {
	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return nil, err
	}
	var extra map[string]json.RawMessage
	for k, v := range all {
		if known.has(k) {
			continue
		}
		if extra == nil {
			extra = make(map[string]json.RawMessage)
		}
		extra[k] = append(json.RawMessage(nil), v...)
	}
	return extra, nil
}

// mergeExtra adds extra to the JSON object knownJSON. An extra key that names a
// known field never overrides it.
func mergeExtra(knownJSON []byte, extra map[string]json.RawMessage, known keySet) ([]byte, error) {
	if len(extra) == 0 {
		return knownJSON, nil
	}
	merged := make(map[string]json.RawMessage, len(extra)+8)
	if err := json.Unmarshal(knownJSON, &merged); err != nil {
		return nil, err
	}
	for k, v := range extra {
		if known.has(k) {
			continue
		}
		merged[k] = v
	}
	return json.Marshal(merged)
}

// unmarshalWithExtra decodes data into v (a method-free alias of the target
// type) and stores the unknown members in *extra.
func unmarshalWithExtra(data []byte, v any, known keySet, extra *map[string]json.RawMessage) error {
	if err := json.Unmarshal(data, v); err != nil {
		return err
	}
	e, err := collectExtra(data, known)
	if err != nil {
		return err
	}
	*extra = e
	return nil
}

// marshalWithExtra encodes v (a method-free alias of the source type) and
// merges extra into the result.
func marshalWithExtra(v any, extra map[string]json.RawMessage, known keySet) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return mergeExtra(b, extra, known)
}

func cloneExtra(m map[string]json.RawMessage) map[string]json.RawMessage {
	if m == nil {
		return nil
	}
	out := make(map[string]json.RawMessage, len(m))
	for k, v := range m {
		out[k] = cloneRaw(v)
	}
	return out
}

// hasContent reports whether any value is something other than null or an
// empty string. Servers send openers such as "refusal":null, which carry no
// content.
func hasContent(extra map[string]json.RawMessage) bool {
	for _, v := range extra {
		v = bytes.TrimSpace(v)
		if len(v) == 0 || bytes.Equal(v, []byte("null")) || bytes.Equal(v, []byte(`""`)) {
			continue
		}
		return true
	}
	return false
}

// extraContentBytes is the size of the values hasContent counts: a JSON string counts as the
// length of its text, any other value as its raw JSON, and a null or an empty string as
// nothing. It sizes the output a model produced in fields outside the typed subset, such as
// reasoning_content.
func extraContentBytes(extra map[string]json.RawMessage) int {
	n := 0
	for _, v := range extra {
		v = bytes.TrimSpace(v)
		if len(v) == 0 || bytes.Equal(v, []byte("null")) || bytes.Equal(v, []byte(`""`)) {
			continue
		}
		var s string
		if v[0] == '"' && json.Unmarshal(v, &s) == nil {
			n += len(s)
			continue
		}
		n += len(v)
	}
	return n
}

var (
	contentPartKeys = keySet{"type", "text", "image_url"}
	messageKeys     = keySet{"role", "content", "name", "tool_calls", "tool_call_id"}
	choiceKeys      = keySet{"index", "message", "finish_reason", "logprobs"}
	chunkChoiceKeys = keySet{"index", "delta", "finish_reason"}
	deltaKeys       = keySet{"role", "content", "tool_calls"}
	usageKeys       = keySet{"prompt_tokens", "completion_tokens", "total_tokens"}
	completionKeys  = keySet{"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
	chunkKeys       = keySet{"id", "object", "created", "model", "choices", "usage"}
)

type (
	contentPartAlias ContentPart
	messageAlias     Message
	choiceAlias      Choice
	chunkChoiceAlias ChunkChoice
	deltaAlias       Delta
	usageAlias       Usage
	completionAlias  ChatCompletion
	chunkAlias       ChatCompletionChunk
)

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (p *ContentPart) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*contentPartAlias)(p), contentPartKeys, &p.Extra)
}

// MarshalJSON writes known fields, then Extra. A text part always carries its
// text, even when empty.
func (p ContentPart) MarshalJSON() ([]byte, error) {
	if p.Type == "text" {
		return marshalWithExtra(struct {
			Type     string    `json:"type"`
			Text     string    `json:"text"`
			ImageURL *ImageURL `json:"image_url,omitempty"`
		}{p.Type, p.Text, p.ImageURL}, p.Extra, contentPartKeys)
	}
	return marshalWithExtra(contentPartAlias(p), p.Extra, contentPartKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (m *Message) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*messageAlias)(m), messageKeys, &m.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (m Message) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(messageAlias(m), m.Extra, messageKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (c *Choice) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*choiceAlias)(c), choiceKeys, &c.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (c Choice) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(choiceAlias(c), c.Extra, choiceKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (c *ChunkChoice) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*chunkChoiceAlias)(c), chunkChoiceKeys, &c.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (c ChunkChoice) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(chunkChoiceAlias(c), c.Extra, chunkChoiceKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (d *Delta) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*deltaAlias)(d), deltaKeys, &d.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (d Delta) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(deltaAlias(d), d.Extra, deltaKeys)
}

// HasExtraContent reports whether the delta carries an unknown field with a
// real value, such as reasoning_content or refusal text. A null or empty
// string does not count.
func (d Delta) HasExtraContent() bool { return hasContent(d.Extra) }

// ExtraContentBytes is the size of the unknown fields' real values (see HasExtraContent):
// the text of reasoning_content or refusal, which the model produced like any other output.
func (d Delta) ExtraContentBytes() int { return extraContentBytes(d.Extra) }

// ExtraContentBytes is the size of the message's unknown fields' real values, such as the
// reasoning_content of a complete answer.
func (m Message) ExtraContentBytes() int { return extraContentBytes(m.Extra) }

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (u *Usage) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*usageAlias)(u), usageKeys, &u.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (u Usage) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(usageAlias(u), u.Extra, usageKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (c *ChatCompletion) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*completionAlias)(c), completionKeys, &c.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (c ChatCompletion) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(completionAlias(c), c.Extra, completionKeys)
}

// UnmarshalJSON decodes known fields and keeps unknown ones in Extra.
func (c *ChatCompletionChunk) UnmarshalJSON(data []byte) error {
	return unmarshalWithExtra(data, (*chunkAlias)(c), chunkKeys, &c.Extra)
}

// MarshalJSON writes known fields, then Extra.
func (c ChatCompletionChunk) MarshalJSON() ([]byte, error) {
	return marshalWithExtra(chunkAlias(c), c.Extra, chunkKeys)
}
