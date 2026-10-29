// Package mock is a deterministic OpenAI-compatible provider used by gateway
// tests and load tests. Latency comes from seeded log-normal distributions and
// failures are injected per request, either explicitly through directives or
// at configured rates.
package mock

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
)

const (
	defaultOutputTokens = 16
	maxBodyBytes        = 8 << 20
	html502Body         = "<html><body>502 Bad Gateway</body></html>"
)

// Dist is a log-normal latency distribution. Sigma == 0 means a constant
// MedianMs.
type Dist struct {
	MedianMs float64
	Sigma    float64
}

// Sample draws one latency: MedianMs * exp(Sigma * N(0,1)). MedianMs <= 0
// yields 0 and consumes no randomness.
func (d Dist) Sample(r *rand.Rand) time.Duration {
	if d.MedianMs <= 0 {
		return 0
	}
	ms := d.MedianMs
	if d.Sigma != 0 {
		ms *= math.Exp(d.Sigma * r.NormFloat64())
	}
	return msToDuration(ms)
}

func msToDuration(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond))
}

// Config configures a Handler.
type Config struct {
	// Seed seeds the RNG behind latency samples and FailRates decisions.
	Seed int64
	// TTFT is the time to first token distribution.
	TTFT Dist
	// TPOT is the time per output token distribution, sampled once per
	// request; inter-token gaps within one stream are constant.
	TPOT Dist
	// OutputTokens is the default completion length; 0 means 16.
	OutputTokens int
	// FailRates maps a fail directive value to the probability of injecting it
	// into a request that carries no fail directive.
	FailRates map[string]float64
}

// Stats are cumulative counters for a Handler.
type Stats struct {
	// Requests counts chat completion requests received.
	Requests int64
	// Completed counts only successful responses fully written: a unary body
	// written and flushed, or a stream whose [DONE] is written. Injected
	// failures, broken responses, and canceled requests are not counted.
	Completed int64
	// Canceled counts requests interrupted by client cancellation or a write error.
	Canceled int64
	// EventsWritten counts SSE chunk events written, excluding [DONE].
	EventsWritten int64
}

// Handler serves POST /v1/chat/completions and GET /v1/models, and GET /mock/stats for its
// own counters.
type Handler struct {
	cfg       Config
	failKeys  []string // FailRates keys, sorted for a stable cumulative draw
	rngMu     sync.Mutex
	rng       *rand.Rand
	nextID    atomic.Int64
	requests  atomic.Int64
	completed atomic.Int64
	canceled  atomic.Int64
	events    atomic.Int64
}

// NewHandler returns a Handler for cfg. It copies cfg.FailRates and panics
// with a descriptive message when the rates are invalid (an unknown fail key,
// a rate outside [0,1], or rates summing above 1), since that is a
// programming error in the caller's config.
func NewHandler(cfg Config) *Handler {
	if cfg.OutputTokens <= 0 {
		cfg.OutputTokens = defaultOutputTokens
	}
	keys := make([]string, 0, len(cfg.FailRates))
	rates := make(map[string]float64, len(cfg.FailRates))
	for k, v := range cfg.FailRates {
		keys = append(keys, k)
		rates[k] = v
	}
	sort.Strings(keys)
	sum := 0.0
	for _, k := range keys {
		if !validFail(k) {
			panic(fmt.Sprintf("mock: FailRates key %q is not a known failure mode", k))
		}
		if v := rates[k]; !(v >= 0 && v <= 1) {
			panic(fmt.Sprintf("mock: FailRates[%q] = %v is outside [0,1]", k, v))
		}
		sum += rates[k]
	}
	if sum > 1+1e-9 {
		panic(fmt.Sprintf("mock: FailRates sum %v exceeds 1", sum))
	}
	cfg.FailRates = rates
	return &Handler{
		cfg:      cfg,
		failKeys: keys,
		rng:      rand.New(rand.NewPCG(uint64(cfg.Seed), 0x9e3779b97f4a7c15)),
	}
}

// Stats returns a snapshot of the counters.
func (h *Handler) Stats() Stats {
	return Stats{
		Requests:      h.requests.Load(),
		Completed:     h.completed.Load(),
		Canceled:      h.canceled.Load(),
		EventsWritten: h.events.Load(),
	}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/chat/completions":
		if r.Method != http.MethodPost {
			methodNotAllowed(w, http.MethodPost)
			return
		}
		h.chat(w, r)
	case "/v1/models":
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"mock","object":"model","created":0,"owned_by":"rp-mock"}]}`+"\n")
	case "/mock/stats":
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		h.writeStats(w)
	default:
		openai.WriteError(w, &openai.APIError{
			Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found",
			Message: fmt.Sprintf("unknown path %q", r.URL.Path),
		})
	}
}

// writeStats serves the counters of Stats as JSON at GET /mock/stats. The path is outside
// /v1, so it cannot be mistaken for part of the OpenAI API a gateway forwards, and reading it
// is not a chat request.
func (h *Handler) writeStats(w http.ResponseWriter) {
	s := h.Stats()
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(w, `{"requests":%d,"completed":%d,"canceled":%d,"events_written":%d}`+"\n",
		s.Requests, s.Completed, s.Canceled, s.EventsWritten)
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	openai.WriteError(w, &openai.APIError{
		Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed",
		Message: "method not allowed, use " + allow,
	})
}

// plan is the per-request outcome of the seeded RNG: the injected failure (if
// any) and the sampled latencies. All three draws always happen, in this
// order. Randomness consumption depends only on the handler config (Dist
// parameters), never on a request's directives, so a directive does not shift
// the draws seen by later requests.
type plan struct {
	fail string
	ttft time.Duration
	tpot time.Duration
}

func (h *Handler) plan(d directives) plan {
	h.rngMu.Lock()
	defer h.rngMu.Unlock()
	u := h.rng.Float64()
	p := plan{ttft: h.cfg.TTFT.Sample(h.rng), tpot: h.cfg.TPOT.Sample(h.rng)}

	p.fail = d.fail
	if p.fail == "" {
		cum := 0.0
		for _, k := range h.failKeys {
			cum += h.cfg.FailRates[k]
			if u < cum {
				p.fail = k
				break
			}
		}
	}
	if d.ttftMs != nil {
		p.ttft = msToDuration(*d.ttftMs)
	}
	if d.tpotMs != nil {
		p.tpot = msToDuration(*d.tpotMs)
	}
	return p
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	h.requests.Add(1)

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		openai.WriteError(w, openai.ErrInvalidRequest("cannot read request body: "+err.Error(), ""))
		return
	}
	req, apiErr := openai.ParseChatCompletionRequest(body)
	if apiErr != nil {
		openai.WriteError(w, apiErr)
		return
	}
	model, d, apiErr := parseDirectives(req.Model, r.Header)
	if apiErr != nil {
		openai.WriteError(w, apiErr)
		return
	}

	p := h.plan(d)
	prompt := promptTokens(req)
	// The completion is as long as configured or directed, but never longer than the request
	// allows, as with a real provider (max_completion_tokens wins over max_tokens).
	tokens := h.cfg.OutputTokens
	if d.tokens != nil {
		tokens = *d.tokens
	}
	if limit, ok := req.EffectiveMaxTokens(); ok && limit < tokens {
		tokens = limit
	}

	switch p.fail {
	case failHTTP500:
		openai.WriteError(w, openai.ErrInternal())
		return
	case failHTTP429:
		w.Header().Set("Retry-After", strconv.Itoa(d.retryAfter))
		openai.WriteError(w, openai.ErrRateLimited("mock rate limit"))
		return
	case failHTTP400:
		openai.WriteError(w, openai.ErrInvalidRequest("mock injected invalid request", ""))
		return
	case failHTML502:
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, html502Body)
		return
	case failReset:
		h.reset(w)
		return
	case failDisconnect:
		if !req.Stream { // there is no second chunk to cut after
			h.reset(w)
			return
		}
	}

	rs := &respState{
		h: h, w: w, ctx: r.Context(), rc: http.NewResponseController(w),
		id:    fmt.Sprintf("chatcmpl-mock-%d", h.nextID.Add(1)),
		model: model, created: time.Now().Unix(), tokens: tokens, p: p, d: d,
		usage: openai.Usage{
			PromptTokens:     prompt,
			CompletionTokens: tokens,
			TotalTokens:      prompt + tokens,
		},
	}
	if req.Stream {
		rs.stream(req.StreamOptions != nil && req.StreamOptions.IncludeUsage)
	} else {
		rs.unary()
	}
}

// promptTokens is ceil(total text bytes / 4) plus 4 per message. Text bytes
// are the string content plus every text part, with no separators.
func promptTokens(req *openai.ChatCompletionRequest) int {
	bytes := 0
	for _, m := range req.Messages {
		if m.Content.Text != nil {
			bytes += len(*m.Content.Text)
		}
		for _, p := range m.Content.Parts {
			if p.Type == "text" {
				bytes += len(p.Text)
			}
		}
	}
	return (bytes+3)/4 + 4*len(req.Messages)
}

// reset hijacks the connection and closes it with a TCP RST before any
// response bytes are written.
func (h *Handler) reset(w http.ResponseWriter) {
	conn, err := hijack(w)
	if err != nil {
		openai.WriteError(w, openai.ErrInternal())
		return
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		_ = tc.SetLinger(0)
	}
	_ = conn.Close()
}

func hijack(w http.ResponseWriter) (net.Conn, error) {
	conn, _, err := http.NewResponseController(w).Hijack()
	return conn, err
}

// respState carries one request's response so unary and stream share helpers.
type respState struct {
	h       *Handler
	w       http.ResponseWriter
	rc      *http.ResponseController
	ctx     context.Context
	id      string
	model   string
	created int64
	tokens  int
	p       plan
	d       directives
	usage   openai.Usage
}

func tokenText(i int) string {
	if i == 0 {
		return "tok0"
	}
	return " tok" + strconv.Itoa(i)
}

// sleep waits d or until the request is canceled. It reports whether the
// full wait elapsed, counting a cancellation.
func (s *respState) sleep(d time.Duration) bool {
	if d <= 0 {
		return s.ctx.Err() == nil || s.cancel()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-s.ctx.Done():
		return s.cancel()
	}
}

// cancel records an interrupted request; it always returns false.
func (s *respState) cancel() bool {
	s.h.canceled.Add(1)
	return false
}

func (s *respState) stall() {
	<-s.ctx.Done()
	s.cancel()
}

func (s *respState) unary() {
	if s.p.fail == failStall {
		s.stall()
		return
	}
	if !s.sleep(s.p.ttft + time.Duration(s.tokens)*s.p.tpot) {
		return
	}
	var sb strings.Builder
	for i := 0; i < s.tokens; i++ {
		sb.WriteString(tokenText(i))
	}
	text := sb.String()
	resp := openai.ChatCompletion{
		ID: s.id, Object: openai.ObjectChatCompletion, Created: s.created, Model: s.model,
		Choices: []openai.Choice{{
			Message:      openai.Message{Role: "assistant", Content: openai.Content{Text: &text}},
			FinishReason: "stop",
		}},
		Usage: &s.usage,
	}
	b, err := json.Marshal(resp)
	if err != nil {
		openai.WriteError(s.w, openai.ErrInternal())
		return
	}
	s.w.Header().Set("Content-Type", "application/json")
	if s.p.fail == failMidstream {
		s.breakUnary(b)
		return
	}
	if _, err := s.w.Write(b); err != nil {
		s.cancel()
		return
	}
	if err := s.rc.Flush(); err != nil {
		s.cancel()
		return
	}
	s.h.completed.Add(1)
}

func (s *respState) chunk(delta openai.Delta, finish *string) openai.ChatCompletionChunk {
	return openai.ChatCompletionChunk{
		ID: s.id, Object: openai.ObjectChatCompletionChunk, Created: s.created, Model: s.model,
		Choices: []openai.ChunkChoice{{Delta: delta, FinishReason: finish}},
	}
}

func (s *respState) stream(includeUsage bool) {
	hdr := s.w.Header()
	hdr.Set("Content-Type", "text/event-stream")
	hdr.Set("Cache-Control", "no-cache")
	s.w.WriteHeader(http.StatusOK)
	if err := s.rc.Flush(); err != nil {
		s.cancel()
		return
	}
	if s.p.fail == failStall {
		s.stall()
		return
	}
	if !s.sleep(s.p.ttft) {
		return
	}

	sse := openai.NewSSEWriter(s.w, s.rc.Flush)
	written := 0
	// midstream streams are cut after cutAfter chunk events, a disconnect stream by default after
	// the second.
	midstream := s.p.fail == failMidstream || s.p.fail == failDisconnect
	cutAfter := s.d.cutAfter(s.p.fail)
	// emit writes one chunk event, or breaks the connection first when the
	// midstream budget is spent. It reports whether the stream may continue.
	emit := func(c openai.ChatCompletionChunk) bool {
		if midstream && written >= cutAfter {
			s.disconnect()
			return false
		}
		b, err := json.Marshal(c)
		if err != nil {
			return s.cancel()
		}
		if err := sse.WriteData(b); err != nil {
			return s.cancel()
		}
		written++
		s.h.events.Add(1)
		return true
	}

	empty := ""
	if !emit(s.chunk(openai.Delta{Role: "assistant", Content: &empty}, nil)) {
		return
	}
	for i := 0; i < s.tokens; i++ {
		if i > 0 && !s.sleep(s.p.tpot) {
			return
		}
		text := tokenText(i)
		if !emit(s.chunk(openai.Delta{Content: &text}, nil)) {
			return
		}
	}
	stop := "stop"
	if !emit(s.chunk(openai.Delta{}, &stop)) {
		return
	}
	if includeUsage {
		u := s.chunk(openai.Delta{}, nil)
		u.Choices = []openai.ChunkChoice{}
		u.Usage = &s.usage
		if !emit(u) {
			return
		}
	}
	if midstream {
		// The budget was never spent, so break before [DONE].
		s.disconnect()
		return
	}
	if err := sse.WriteDone(); err != nil {
		s.cancel()
		return
	}
	s.h.completed.Add(1)
}

// breakUnary announces the full body length, sends 200 and about half the
// body, then drops the connection so the client sees an unexpected EOF.
func (s *respState) breakUnary(body []byte) {
	s.w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	s.w.WriteHeader(http.StatusOK)
	if _, err := s.w.Write(body[:len(body)/2]); err != nil {
		s.cancel()
		return
	}
	if err := s.rc.Flush(); err != nil {
		s.cancel()
		return
	}
	s.disconnect()
}

// disconnect drops the connection without terminating the chunked body.
func (s *respState) disconnect() {
	conn, err := hijack(s.w)
	if err != nil {
		s.cancel()
		return
	}
	_ = conn.Close()
}
