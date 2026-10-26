package catalog

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
)

const (
	// textLimit caps Requirements.Text, which feeds the quality classifier.
	textLimit = 32768
	// perMessageOverhead and fixedOverhead pad InUpper above any provider's
	// per-message and per-request framing tokens.
	perMessageOverhead = 8
	fixedOverhead      = 16
)

// Requirements is what a request demands of a model.
type Requirements struct {
	// Caps are the capabilities a model must have.
	Caps CapSet
	// InUpper is an upper bound on prompt tokens, derived from byte counts
	// (a token is at least one byte of text).
	InUpper int
	// MaxTokens is the explicit output cap; valid only when HasMaxTokens.
	MaxTokens    int
	HasMaxTokens bool
	// PromptBytes is the total bytes of message text contents.
	PromptBytes int
	// Text is the message text joined by newlines, cut to textLimit bytes on
	// a rune boundary.
	Text string
}

// Derive computes the requirements of req.
func Derive(req *openai.ChatCompletionRequest) Requirements {
	var r Requirements

	if requiresTools(req) {
		r.Caps = r.Caps.with(CapTools)
	}
	if rf := req.ResponseFormat; rf != nil {
		switch rf.Type {
		case "json_object":
			r.Caps = r.Caps.with(CapJSONObject)
		case "json_schema":
			r.Caps = r.Caps.with(CapJSONSchema)
		}
	}
	if req.Stream {
		r.Caps = r.Caps.with(CapStreaming)
	}
	r.MaxTokens, r.HasMaxTokens = req.EffectiveMaxTokens()

	var text strings.Builder
	extra := 0
	for i, m := range req.Messages {
		if m.Content.HasImage() {
			r.Caps = r.Caps.with(CapVision)
		}
		r.PromptBytes += textBytes(m.Content)
		extra += len(m.Name)
		for _, tc := range m.ToolCalls {
			extra += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
		if text.Len() < textLimit {
			if i > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(m.Content.PlainText())
		}
	}
	r.Text = truncateUTF8(text.String(), textLimit)

	if len(req.Tools) > 0 {
		// Tools came from a parsed request, so marshaling cannot fail.
		if b, err := json.Marshal(req.Tools); err == nil {
			extra += len(b)
		}
	}
	r.InUpper = r.PromptBytes + extra + perMessageOverhead*len(req.Messages) + fixedOverhead
	return r
}

// requiresTools reports whether the request needs tool support: tools are
// declared, an assistant message carries tool calls, or a tool message is
// present. tool_choice does not matter.
func requiresTools(req *openai.ChatCompletionRequest) bool {
	if len(req.Tools) > 0 {
		return true
	}
	for _, m := range req.Messages {
		if m.Role == "tool" || (m.Role == "assistant" && len(m.ToolCalls) > 0) {
			return true
		}
	}
	return false
}

// textBytes counts the string content plus every text part, with no separators.
func textBytes(c openai.Content) int {
	n := 0
	if c.Text != nil {
		n += len(*c.Text)
	}
	for _, p := range c.Parts {
		if p.Type == "text" {
			n += len(p.Text)
		}
	}
	return n
}

// truncateUTF8 cuts s to at most limit bytes without splitting a rune.
func truncateUTF8(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

// EffectiveK is the output token budget used for admission and cost: the
// explicit max tokens when set, else the smaller of the model limit and the
// tenant default.
func (m *Model) EffectiveK(r Requirements, tenantDefault int) int {
	if r.HasMaxTokens {
		return r.MaxTokens
	}
	return min(m.MaxOutputTokens, tenantDefault)
}

// Reason explains why a model cannot serve a request.
type Reason string

// Check returns why m cannot serve r, or an empty slice when it can. Reasons
// come in a stable order: missing capabilities, context window, max tokens.
func (m *Model) Check(r Requirements, tenantDefault int) []Reason {
	var reasons []Reason
	for _, c := range m.Caps.Missing(r.Caps) {
		reasons = append(reasons, Reason("missing capability "+c.String()))
	}
	k := m.EffectiveK(r, tenantDefault)
	if need := r.InUpper + k; need > m.ContextWindow {
		reasons = append(reasons, Reason(fmt.Sprintf("context window %d below required %d", m.ContextWindow, need)))
	}
	if r.HasMaxTokens && r.MaxTokens > m.MaxOutputTokens {
		reasons = append(reasons, Reason(fmt.Sprintf("max_tokens %d above model limit %d", r.MaxTokens, m.MaxOutputTokens)))
	}
	return reasons
}

// CostMax is the worst-case cost of a request: InUpper input tokens plus k
// output tokens, saturating at the maximum PicoUSD.
func (m *Model) CostMax(r Requirements, k int) money.PicoUSD {
	return money.AddSat(
		money.MulSat(m.PriceIn, int64(r.InUpper)),
		money.MulSat(m.PriceOut, int64(k)),
	)
}
