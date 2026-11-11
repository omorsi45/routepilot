package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

// roundTrip decodes in into v, encodes it again, and fails unless the two
// JSON documents are equal. It returns the decoded value for further checks.
func roundTrip(t *testing.T, in string, v any) {
	t.Helper()
	if err := json.Unmarshal([]byte(in), v); err != nil {
		t.Fatalf("unmarshal %s: %v", in, err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got, want := decodeMap(t, out), decodeMap(t, []byte(in)); !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip mismatch\n got: %s\nwant: %s", out, in)
	}
}

func TestUnknownFieldsRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   string
		v    any
	}{
		{"content part input_audio", `{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}`, &ContentPart{}},
		{"content part file", `{"type":"file","file":{"file_id":"file-1"}}`, &ContentPart{}},
		{"content part unknown key on text", `{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}`, &ContentPart{}},
		{"message refusal", `{"role":"assistant","content":null,"refusal":"I cannot help with that"}`, &Message{}},
		{"message reasoning_content", `{"role":"assistant","content":"hi","reasoning_content":"thinking"}`, &Message{}},
		{"choice", `{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop","stop_reason":"eos"}`, &Choice{}},
		{"chunk choice logprobs", `{"index":0,"delta":{"content":"a"},"finish_reason":null,"logprobs":{"content":[]}}`, &ChunkChoice{}},
		{"delta reasoning_content", `{"role":"assistant","content":"","reasoning_content":"step 1"}`, &Delta{}},
		{"usage details", `{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":1},"prompt_tokens_details":{"cached_tokens":0}}`, &Usage{}},
		{"completion", `{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[],"service_tier":"default","system_fingerprint":"fp"}`, &ChatCompletion{}},
		{"chunk", `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"service_tier":"default"}`, &ChatCompletionChunk{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) { roundTrip(t, tc.in, tc.v) })
	}
}

func TestKnownKeysAreNotExtraOnResponseTypes(t *testing.T) {
	var m Message
	roundTrip(t, `{"role":"assistant","content":"hi","refusal":"no"}`, &m)
	if len(m.Extra) != 1 || string(m.Extra["refusal"]) != `"no"` {
		t.Errorf("Message.Extra = %v, want only refusal", m.Extra)
	}
	var d Delta
	// A case variant of a known key fills the typed field (and is written back
	// under its canonical name), never Extra.
	if err := json.Unmarshal([]byte(`{"CONTENT":"x","reasoning_content":"r"}`), &d); err != nil {
		t.Fatal(err)
	}
	if d.Content == nil || *d.Content != "x" || len(d.Extra) != 1 {
		t.Errorf("case-variant known key leaked into Extra: %+v", d)
	}
	var u Usage
	roundTrip(t, `{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}`, &u)
	if u.Extra != nil {
		t.Errorf("Usage.Extra = %v, want nil", u.Extra)
	}
}

func TestTextPartEmitsEmptyText(t *testing.T) {
	out, err := json.Marshal(ContentPart{Type: "text"})
	if err != nil {
		t.Fatal(err)
	}
	if got := decodeMap(t, out); !reflect.DeepEqual(got, map[string]any{"type": "text", "text": ""}) {
		t.Errorf("got %s, want type text with text empty string", out)
	}
	out, _ = json.Marshal(ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: "u"}})
	if got := decodeMap(t, out); got["text"] != nil {
		t.Errorf("image part carries text: %s", out)
	}
}

func TestCloneDeepCopiesMessageAndPartExtra(t *testing.T) {
	var req ChatCompletionRequest
	in := `{"model":"m","messages":[{"role":"user","refusal":"r","content":[{"type":"input_audio","input_audio":{"data":"AA"}}]}]}`
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatal(err)
	}
	c := req.Clone()
	c.Messages[0].Extra["refusal"][1] = 'X'
	c.Messages[0].Extra["added"] = json.RawMessage(`1`)
	c.Messages[0].Content.Parts[0].Extra["input_audio"][2] = 'X'
	c.Messages[0].Content.Parts[0].Extra["added"] = json.RawMessage(`1`)

	if string(req.Messages[0].Extra["refusal"]) != `"r"` || len(req.Messages[0].Extra) != 1 {
		t.Errorf("message Extra leaked through Clone: %v", req.Messages[0].Extra)
	}
	if p := req.Messages[0].Content.Parts[0]; string(p.Extra["input_audio"]) != `{"data":"AA"}` || len(p.Extra) != 1 {
		t.Errorf("part Extra leaked through Clone: %v", p.Extra)
	}
}

func TestIsCommittingDeltaExtra(t *testing.T) {
	for in, want := range map[string]bool{
		`{"reasoning_content":"step"}`: true,
		`{"refusal":"no"}`:             true,
		`{"refusal":null}`:             false,
		`{"reasoning_content":""}`:     false,
		`{}`:                           false,
	} {
		var d Delta
		if err := json.Unmarshal([]byte(in), &d); err != nil {
			t.Fatal(err)
		}
		if got := d.HasExtraContent(); got != want {
			t.Errorf("HasExtraContent(%s) = %v, want %v", in, got, want)
		}
	}
}
