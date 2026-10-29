package mock

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/openai"
)

// TestMockCapsTheCompletionAtMaxTokens (Ruling V1-R39): like a real provider, the mock never
// answers with more tokens than the request allows, and the usage it reports is what it sent.
// The length is the smaller of the configured or directive length and the request's cap, where
// max_completion_tokens wins over max_tokens as it does in the gateway.
func TestMockCapsTheCompletionAtMaxTokens(t *testing.T) {
	cases := []struct {
		name  string
		model string // carries the tokens directive, if any
		extra string // request fields
		want  int
	}{
		{"cap below the default length", "m", `,"max_tokens":5`, 5},
		{"cap above the default length", "m", `,"max_tokens":100`, 16},
		{"no cap keeps the default length", "m", ``, 16},
		{"cap below a directive length", "m;tokens=50", `,"max_tokens":7`, 7},
		{"cap above a directive length", "m;tokens=4", `,"max_tokens":100`, 4},
		{"max_completion_tokens caps too", "m;tokens=50", `,"max_completion_tokens":9`, 9},
		{"max_completion_tokens wins over max_tokens", "m;tokens=50", `,"max_tokens":3,"max_completion_tokens":11`, 11},
		{"a cap of one", "m;tokens=50", `,"max_tokens":1`, 1},
	}
	for _, tc := range cases {
		model := tc.model + ";ttft_ms=0;tpot_ms=0"
		t.Run(tc.name+"/unary", func(t *testing.T) {
			_, srv := newServer(t, Config{Seed: 1})
			resp := mustPost(t, srv.URL, chatBody(model, tc.extra), nil)
			var got openai.ChatCompletion
			if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
				t.Fatal(err)
			}
			text := got.Choices[0].Message.Content.PlainText()
			if n := len(strings.Fields(text)); n != tc.want {
				t.Errorf("body holds %d tokens (%q), want %d", n, text, tc.want)
			}
			if got.Usage == nil || got.Usage.CompletionTokens != tc.want || got.Usage.TotalTokens != got.Usage.PromptTokens+tc.want {
				t.Errorf("usage = %+v, want %d completion tokens", got.Usage, tc.want)
			}
		})
		t.Run(tc.name+"/stream", func(t *testing.T) {
			_, srv := newServer(t, Config{Seed: 1})
			resp := mustPost(t, srv.URL, chatBody(model, tc.extra+`,"stream":true,"stream_options":{"include_usage":true}`), nil)
			events, err := readEvents(t, resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			content, completion := 0, -1
			for _, ev := range events {
				if ev.IsDone() {
					continue
				}
				var c openai.ChatCompletionChunk
				if err := json.Unmarshal(ev.Data, &c); err != nil {
					t.Fatal(err)
				}
				if c.Usage != nil {
					completion = c.Usage.CompletionTokens
				}
				if len(c.Choices) == 1 && c.Choices[0].Delta.Content != nil && *c.Choices[0].Delta.Content != "" {
					content++
				}
			}
			if content != tc.want {
				t.Errorf("stream holds %d content chunks, want %d", content, tc.want)
			}
			if completion != tc.want {
				t.Errorf("stream usage reports %d completion tokens, want %d", completion, tc.want)
			}
		})
	}
}

// TestMockDisconnectCutsAStreamAfterTheSecondChunk (Ruling V1-R40): fail=disconnect drops the
// connection once two chunks are out, the role chunk and the first content chunk. That is the
// first point at which a gateway has committed the response to its client, so the cut is a
// failure after the commit. fail_after still moves the cut.
func TestMockDisconnectCutsAStreamAfterTheSecondChunk(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model string
		want  int
	}{
		{"default cut", "m;fail=disconnect;tokens=5;ttft_ms=0;tpot_ms=0", 2},
		{"fail_after moves it", "m;fail=disconnect;fail_after=3;tokens=5;ttft_ms=0;tpot_ms=0", 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, srv := newServer(t, Config{Seed: 1})
			resp := mustPost(t, srv.URL, chatBody(tc.model, `,"stream":true`), nil)
			events, err := readEvents(t, resp.Body)
			if err == nil {
				t.Fatal("expected the connection to break mid-stream")
			}
			if len(events) != tc.want {
				t.Fatalf("got %d events, want exactly %d", len(events), tc.want)
			}
			for _, ev := range events {
				if ev.IsDone() {
					t.Fatal("must not send [DONE]")
				}
			}
			if tc.want >= 2 {
				var second openai.ChatCompletionChunk
				if err := json.Unmarshal(events[1].Data, &second); err != nil {
					t.Fatal(err)
				}
				if c := second.Choices[0].Delta.Content; c == nil || *c != "tok0" {
					t.Errorf("second chunk content = %v, want the first content chunk", c)
				}
			}
			if s := h.Stats(); s.Completed != 0 || s.EventsWritten != int64(tc.want) {
				t.Fatalf("stats = %+v, want nothing completed and %d events written", s, tc.want)
			}
		})
	}
}

// TestMockDisconnectIsAResetForANonStreamingRequest: there is no second chunk to cut after, so
// the request ends the way fail=reset does, before any header.
func TestMockDisconnectIsAResetForANonStreamingRequest(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})
	resp, err := post(t, context.Background(), srv.URL, chatBody("m;fail=disconnect;tokens=5;ttft_ms=0;tpot_ms=0", ""), nil)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatalf("expected a transport error, got status %d", resp.StatusCode)
	}
	if s := h.Stats(); s.Requests != 1 || s.Completed != 0 || s.EventsWritten != 0 {
		t.Fatalf("stats = %+v, want one request received and nothing written", s)
	}
}

// TestMockDisconnectRate: the FailRates key disconnect injects the same failure into requests
// that carry no fail directive, at the configured rate, from the seeded generator.
func TestMockDisconnectRate(t *testing.T) {
	count := func(seed int64) int {
		_, srv := newServer(t, Config{Seed: seed, FailRates: map[string]float64{"disconnect": 0.2}})
		cut := 0
		for i := 0; i < 500; i++ {
			resp := mustPost(t, srv.URL, chatBody("m;tokens=4;ttft_ms=0;tpot_ms=0", `,"stream":true`), nil)
			events, err := readEvents(t, resp.Body)
			switch {
			case err != nil && len(events) == 2:
				cut++
			case err != nil:
				t.Fatalf("stream broke after %d events, want a cut after exactly 2", len(events))
			case len(events) != 7: // role, 4 content, finish, [DONE]
				t.Fatalf("a stream that was not cut holds %d events, want 7", len(events))
			}
		}
		return cut
	}
	a, b := count(7), count(7)
	if a != b {
		t.Fatalf("same seed cut %d and %d streams", a, b)
	}
	if a < 70 || a > 130 {
		t.Fatalf("cut %d of 500 streams at rate 0.2, want within [70,130]", a)
	}
}

// TestMockDisconnectIsAKnownFailureMode: the directive and the FailRates key are accepted, and
// a rate counts toward the sum limit like the others.
func TestMockDisconnectIsAKnownFailureMode(t *testing.T) {
	NewHandler(Config{FailRates: map[string]float64{"disconnect": 0.2, "http500": 0.3}}) // must not panic
	defer func() {
		if recover() == nil {
			t.Fatal("rates summing above 1 with disconnect did not panic")
		}
	}()
	NewHandler(Config{FailRates: map[string]float64{"disconnect": 0.6, "http500": 0.6}})
}
