package gateway

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// usageModel prices tokens at 1,000 picodollars in and 3,000 out, with a hidden prompt
// overhead of 4 tokens, so every term of an attempt's cost is visible in the totals.
func usageModel() catalog.Model {
	return catalog.Model{ID: "m", PriceIn: 1000, PriceOut: 3000, PromptOverhead: 4}
}

func newTestAttempt(inUpper, k int) *attemptUsage {
	m := usageModel()
	return newAttemptUsage(m, catalog.Deployment{Name: "m-a"}, catalog.Requirements{InUpper: inUpper}, k)
}

func TestAttemptUsageReportedUsageIsExact(t *testing.T) {
	a := newTestAttempt(100, 50)
	a.reportUsage(&openai.Usage{PromptTokens: 7, CompletionTokens: 3})
	a.resolve()

	if a.Model != "m" || a.Deployment != "m-a" {
		t.Errorf("identity = %q / %q, want m / m-a", a.Model, a.Deployment)
	}
	if a.Prompt != 7 || a.Completion != 3 || a.Estimated || a.Cost != 7*1000+3*3000 {
		t.Errorf("usage = %+v, want 7 prompt, 3 completion, exact, cost 16,000", a)
	}
	// A reported usage is final: a later failure or more streamed bytes change nothing.
	a.failed(&provider.Error{Kind: provider.KindTimeout})
	a.addStreamed(1000)
	a.resolve()
	if a.Cost != 16_000 || a.Estimated {
		t.Errorf("after a late failure: %+v, want the reported usage kept", a)
	}
}

func TestAttemptUsageUnreportedAttemptIsEstimated(t *testing.T) {
	// The input bound is the request's 100 plus the model's 4 hidden tokens.
	tests := []struct {
		name       string
		k          int
		streamed   int
		wantPrompt int
		wantComp   int
	}{
		{"nothing streamed", 50, 0, 104, 0},
		{"streamed bytes count as tokens", 50, 20, 104, 20},
		{"never more than the output budget", 50, 500, 104, 50},
		{"exactly the output budget", 50, 50, 104, 50},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newTestAttempt(100, tc.k)
			a.addStreamed(tc.streamed)
			a.resolve()
			want := money.PicoUSD(tc.wantPrompt*1000 + tc.wantComp*3000)
			if a.Prompt != tc.wantPrompt || a.Completion != tc.wantComp || !a.Estimated || a.Cost != want {
				t.Errorf("usage = %+v, want %d prompt, %d completion, estimated, cost %d", a, tc.wantPrompt, tc.wantComp, want)
			}
			if got := a.charge(); got != want {
				t.Errorf("charge before resolve = %d, want %d", got, want)
			}
		})
	}
}

func TestAttemptUsageFailuresBeforeAnyOutput(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		streamed int
		free     bool
	}{
		{"an HTTP status is not billed", &provider.Error{Kind: provider.KindHTTP, Status: 500}, 0, true},
		{"a 429 is not billed", &provider.Error{Kind: provider.KindHTTP, Status: 429}, 0, true},
		{"a client error is not billed", &provider.Error{Kind: provider.KindHTTP, Status: 400}, 0, true},
		{"a connect error is not billed", &provider.Error{Kind: provider.KindConnect}, 0, true},
		{"a wrapped HTTP status is not billed", errors.Join(errors.New("ctx"), &provider.Error{Kind: provider.KindHTTP, Status: 503}), 0, true},
		{"a timeout is estimated", &provider.Error{Kind: provider.KindTimeout}, 0, false},
		{"a cancel is estimated", &provider.Error{Kind: provider.KindCanceled}, 0, false},
		{"a protocol failure is estimated", &provider.Error{Kind: provider.KindProtocol}, 0, false},
		{"an adapter error is estimated", openai.ErrInternal(), 0, false},
		{"a plain error is estimated", errors.New("boom"), 0, false},
		{"a context error is estimated", context.DeadlineExceeded, 0, false},
		{"a connect error after output is estimated", &provider.Error{Kind: provider.KindConnect}, 12, false},
		{"an HTTP status after output is estimated", &provider.Error{Kind: provider.KindHTTP, Status: 500}, 12, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newTestAttempt(100, 50)
			a.addStreamed(tc.streamed)
			a.failed(tc.err)
			a.resolve()
			if tc.free {
				if a.Cost != 0 || a.Prompt != 0 || a.Completion != 0 || a.Estimated {
					t.Errorf("usage = %+v, want a free attempt: no tokens, no cost, not estimated", a)
				}
				return
			}
			want := money.PicoUSD(104*1000 + tc.streamed*3000)
			if !a.Estimated || a.Cost != want {
				t.Errorf("usage = %+v, want an estimate costing %d", a, want)
			}
		})
	}
}

func TestAttemptUsageSaturatesAndIgnoresNegatives(t *testing.T) {
	a := newTestAttempt(100, 50)
	a.reportUsage(&openai.Usage{PromptTokens: -5, CompletionTokens: -9})
	a.resolve()
	if a.Prompt != 0 || a.Completion != 0 || a.Cost != 0 {
		t.Errorf("negative usage = %+v, want zeros", a)
	}

	big := catalog.Model{ID: "big", PriceIn: money.PicoUSD(math.MaxInt64 / 2), PriceOut: money.PicoUSD(math.MaxInt64 / 2)}
	b := newAttemptUsage(big, catalog.Deployment{Name: "big-a"}, catalog.Requirements{InUpper: 10}, 10)
	b.reportUsage(&openai.Usage{PromptTokens: math.MaxInt32, CompletionTokens: math.MaxInt32})
	b.resolve()
	if b.Cost != money.PicoUSD(math.MaxInt64) {
		t.Errorf("saturated cost = %d, want MaxInt64", b.Cost)
	}
}

func TestRequestAttemptsCostIsTheSum(t *testing.T) {
	rs := &reqState{reqs: catalog.Requirements{InUpper: 100}}
	m := usageModel()
	dep := catalog.Deployment{Name: "m-a"}

	if got := rs.chargedSoFar(); got != 0 {
		t.Errorf("no attempts: chargedSoFar = %d, want 0", got)
	}
	first := rs.beginAttempt(m, dep, 50)
	first.failed(&provider.Error{Kind: provider.KindHTTP, Status: 503})
	second := rs.beginAttempt(m, dep, 50)
	second.addStreamed(10) // cut off mid-stream, no usage
	third := rs.beginAttempt(m, dep, 50)
	third.reportUsage(&openai.Usage{PromptTokens: 5, CompletionTokens: 5})

	want := money.PicoUSD(0 + (104*1000 + 10*3000) + (5*1000 + 5*3000))
	if got := rs.chargedSoFar(); got != want {
		t.Errorf("chargedSoFar = %d, want %d", got, want)
	}
	if got := rs.settleAttempts(); got != want {
		t.Errorf("settleAttempts = %d, want %d", got, want)
	}
	if len(rs.attemptUsages) != 3 || rs.attemptUsages[0].Cost != 0 || rs.attemptUsages[0].Estimated ||
		!rs.attemptUsages[1].Estimated || rs.attemptUsages[2].Estimated {
		t.Errorf("records = %+v %+v %+v", rs.attemptUsages[0], rs.attemptUsages[1], rs.attemptUsages[2])
	}
}

func TestStreamedBytesCountsContentAndToolCalls(t *testing.T) {
	text := func(s string) *string { return &s }
	chunk := &openai.ChatCompletionChunk{Choices: []openai.ChunkChoice{
		{Delta: openai.Delta{Content: text("hello")}},
		{Delta: openai.Delta{ToolCalls: []openai.ToolCall{
			{ID: "call_1", Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"q":"x"}`}},
			{Function: openai.FunctionCall{Arguments: `abc`}},
		}}},
		{Delta: openai.Delta{Role: "assistant"}},
	}}
	// 5 content bytes, then name 11 and arguments 9, then arguments 3. The tool call ID is
	// not output the model produced token by token and is not counted.
	if got := chunkBytes(chunk); got != 5+11+9+3 {
		t.Errorf("chunkBytes = %d, want 28", got)
	}
	if got := chunkBytes(&openai.ChatCompletionChunk{}); got != 0 {
		t.Errorf("chunkBytes of an empty chunk = %d, want 0", got)
	}
	if got := chunkBytes(&openai.ChatCompletionChunk{Usage: &openai.Usage{CompletionTokens: 9}}); got != 0 {
		t.Errorf("chunkBytes of a usage chunk = %d, want 0", got)
	}
}

func TestCompletionBytesCountsContentPartsAndToolCalls(t *testing.T) {
	text := func(s string) *string { return &s }
	resp := &openai.ChatCompletion{Choices: []openai.Choice{
		{Message: openai.Message{Content: openai.Content{Text: text("abcde")}}},
		{Message: openai.Message{
			Content: openai.Content{Parts: []openai.ContentPart{{Type: "text", Text: "xy"}, {Type: "image_url"}}},
			ToolCalls: []openai.ToolCall{
				{Function: openai.FunctionCall{Name: "f", Arguments: "{}"}},
			},
		}},
	}}
	if got := completionBytes(resp); got != 5+2+1+2 {
		t.Errorf("completionBytes = %d, want 10", got)
	}
	if got := completionBytes(nil); got != 0 {
		t.Errorf("completionBytes(nil) = %d, want 0", got)
	}
}
