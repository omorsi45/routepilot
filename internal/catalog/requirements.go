package catalog

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
)

// ImageTokenAllowance is the input token allowance charged per image part.
// It is a per-image allowance, not a proof of an upper bound.
const ImageTokenAllowance = 2048

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
	// InUpper is an upper bound on prompt tokens, derived from the bytes of
	// everything forwarded upstream (a token is at least one byte of text),
	// plus ImageTokenAllowance per image part.
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
		extra += messageExtraBytes(m)
		if text.Len() < textLimit {
			if i > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(m.Content.PlainText())
		}
	}
	r.Text = truncateUTF8(text.String(), textLimit)

	extra += len(req.ToolChoice)
	if rf := req.ResponseFormat; rf != nil {
		extra += len(rf.JSONSchema)
	}
	extra += rawMapBytes(req.Extra)
	extra += toolsBytes(req.Tools)
	r.InUpper = r.PromptBytes + extra + perMessageOverhead*len(req.Messages) + fixedOverhead
	return r
}

// messageExtraBytes counts the forwarded bytes of m that are not message text:
// name, tool_call_id, tool calls, unknown fields, content parts that are
// neither text nor image, and the allowance for each image part.
func messageExtraBytes(m openai.Message) int {
	n := len(m.Name) + len(m.ToolCallID) + rawMapBytes(m.Extra)
	for _, tc := range m.ToolCalls {
		n += len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)
	}
	for _, p := range m.Content.Parts {
		if p.Type == "image_url" || p.ImageURL != nil {
			n += ImageTokenAllowance
		}
		if p.Type == "text" || p.Type == "image_url" {
			// Text bytes and the image allowance are counted elsewhere, but
			// unknown keys on these parts are forwarded upstream too.
			n += rawMapBytes(p.Extra)
		} else if b, err := json.Marshal(p); err == nil {
			n += len(b)
		} else {
			// Fall back to the parts we can see.
			n += len(p.Type) + len(p.Text) + rawMapBytes(p.Extra)
		}
	}
	return n
}

// rawMapBytes counts every key plus raw value of an Extra map.
func rawMapBytes(m map[string]json.RawMessage) int {
	n := 0
	for k, v := range m {
		n += len(k) + len(v)
	}
	return n
}

// toolsBytes is the size of the tool definitions as sent upstream. If they
// cannot be marshaled it falls back to a conservative per-tool count.
func toolsBytes(tools []openai.Tool) int {
	if len(tools) == 0 {
		return 0
	}
	if b, err := json.Marshal(tools); err == nil {
		return len(b)
	}
	n := 0
	for _, t := range tools {
		n += len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters) + 64
	}
	return n
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

// InUpper is the upper bound on prompt tokens for m: the request-level bound
// plus the hidden prompt tokens m's provider adds. It
// saturates at the maximum int. Everything that judges a request against one
// specific model (context window, worst-case cost, admission) uses it;
// Requirements.InUpper stays the request-level bound.
func (m *Model) InUpper(r Requirements) int {
	if m.PromptOverhead > math.MaxInt-r.InUpper {
		return math.MaxInt
	}
	return r.InUpper + m.PromptOverhead
}

// AffordableK is the output token budget m can be admitted with when the request may spend at
// most budget, and whether it can be admitted at all. A nil budget is unlimited, and then k
// is EffectiveK and ok is true.
//
// With an explicit max_tokens the cap is kept as the client set it: k is r.MaxTokens and ok
// reports whether its worst case fits the budget. Without one, k is EffectiveK reduced to
// what the budget pays for after the input bound,
//
//	floor((budget - InUpper*PriceIn) / PriceOut)
//
// computed in integer picodollars (never below zero), and ok means that at least one output
// token fits and the worst case at k is within the budget. A model with no output price is
// not reduced. Whenever ok is true and budget is set, CostMax(r, k) <= *budget.
func (m *Model) AffordableK(r Requirements, tenantDefault int, budget *money.PicoUSD) (k int, ok bool) {
	if r.HasMaxTokens {
		k = r.MaxTokens
		return k, budget == nil || m.CostMax(r, k) <= *budget
	}
	k = m.EffectiveK(r, tenantDefault)
	if budget == nil {
		return k, true
	}
	if m.PriceOut > 0 {
		afford := int64(0)
		if inCost := money.MulSat(m.PriceIn, int64(m.InUpper(r))); inCost < *budget {
			afford = int64(*budget-inCost) / int64(m.PriceOut)
		}
		if afford < int64(k) {
			k = int(afford)
		}
	}
	return k, k >= 1 && m.CostMax(r, k) <= *budget
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
	in := m.InUpper(r)
	// Compare by subtraction so a huge explicit k cannot overflow the sum.
	if k > m.ContextWindow-in {
		need := math.MaxInt
		if k <= math.MaxInt-in {
			need = in + k
		}
		reasons = append(reasons, Reason(fmt.Sprintf("context window %d below required %d", m.ContextWindow, need)))
	}
	if r.HasMaxTokens && r.MaxTokens > m.MaxOutputTokens {
		reasons = append(reasons, Reason(fmt.Sprintf("max_tokens %d above model limit %d", r.MaxTokens, m.MaxOutputTokens)))
	}
	return reasons
}

// CostMax is the worst-case cost of a request: the model's InUpper input
// tokens plus k output tokens, saturating at the maximum PicoUSD.
func (m *Model) CostMax(r Requirements, k int) money.PicoUSD {
	return money.AddSat(
		money.MulSat(m.PriceIn, int64(m.InUpper(r))),
		money.MulSat(m.PriceOut, int64(k)),
	)
}
