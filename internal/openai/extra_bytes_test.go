package openai

import (
	"encoding/json"
	"testing"
)

// TestExtraContentBytes: the real values of the unknown fields of a delta or a message count
// as output the model produced. A string counts as the length of its text, any other value
// as its JSON, and a null or an empty string as nothing, as in HasExtraContent.
func TestExtraContentBytes(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int
	}{
		{"reasoning text is its length, not its quoted form", `{"reasoning_content":"step"}`, 4},
		{"an escaped newline is one byte of text", `{"reasoning_content":"a\nb"}`, 3},
		{"several fields add up", `{"reasoning_content":"abc","refusal":"no"}`, 5},
		{"null counts for nothing", `{"refusal":null}`, 0},
		{"an empty string counts for nothing", `{"reasoning_content":""}`, 0},
		{"a structured value counts as its JSON", `{"audio":{"id":"x"}}`, 10},
		{"no extra fields", `{"content":"hello"}`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var d Delta
			if err := json.Unmarshal([]byte(tc.in), &d); err != nil {
				t.Fatal(err)
			}
			if got := d.ExtraContentBytes(); got != tc.want {
				t.Errorf("Delta.ExtraContentBytes(%s) = %d, want %d", tc.in, got, tc.want)
			}
			var m Message
			if err := json.Unmarshal([]byte(tc.in), &m); err != nil {
				t.Fatal(err)
			}
			if got := m.ExtraContentBytes(); got != tc.want {
				t.Errorf("Message.ExtraContentBytes(%s) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}
