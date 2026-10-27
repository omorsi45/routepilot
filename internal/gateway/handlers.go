package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
)

// handleText serves the unauthenticated health routes.
func (s *Server) handleText(w http.ResponseWriter, rs *reqState, r *http.Request, body string) {
	if r.Method != http.MethodGet {
		s.methodNotAllowed(w, rs, http.MethodGet)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = io.WriteString(w, body)
}

// handleModels lists the catalog models the authenticated tenant may use.
func (s *Server) handleModels(w http.ResponseWriter, rs *reqState, r *http.Request) {
	tenant, ok := s.authenticate(r)
	if !ok {
		s.fail(w, rs, openai.ErrInvalidAPIKey())
		return
	}
	rs.tenant = tenant.ID

	type modelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	data := make([]modelEntry, 0, len(s.cfg.Models)+1)
	if s.router.HasAuto() {
		data = append(data, modelEntry{ID: config.ReservedModelAuto, Object: "model", OwnedBy: "routepilot"})
	}
	for _, m := range s.cfg.Models {
		if tenant.Allows(m.ID) {
			data = append(data, modelEntry{ID: m.ID, Object: "model", OwnedBy: "routepilot"})
		}
	}
	writeJSON(w, http.StatusOK, struct {
		Object string       `json:"object"`
		Data   []modelEntry `json:"data"`
	}{Object: "list", Data: data})
}

// handleChat authenticates, validates and authorizes a chat completion, then
// dispatches to the streaming or non-streaming path.
func (s *Server) handleChat(w http.ResponseWriter, rs *reqState, r *http.Request) {
	tenant, ok := s.authenticate(r)
	if !ok {
		s.fail(w, rs, openai.ErrInvalidAPIKey())
		return
	}
	rs.tenant = tenant.ID

	// The global in-flight slot is taken before the body is read and held for
	// the whole request, including a stream.
	if !s.acquireSlot(w, rs, r) {
		return
	}
	defer s.releaseSlot()

	body, apiErr := s.readBody(w, r)
	if apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	req, apiErr := openai.ParseChatCompletionRequest(body)
	if apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	if apiErr := mergeRoutingHeaders(r.Header, req); apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	// Log a bounded prefix until the model resolves to a catalog entry: the
	// request model is client text of up to the whole body size.
	rs.model = truncateBytes(req.Model, maxLoggedModelBytes)
	rs.promptBytes = promptBytes(req)

	if m, ok := s.cat.Model(req.Model); ok {
		rs.model = m.ID
	}

	reqs := catalog.Derive(req)
	dec, apiErr := s.router.Decide(router.Input{
		Tenant:    tenant,
		Requested: req.Model,
		Req:       reqs,
		Available: s.health.Available,
	})
	rs.policy = dec.Policy
	if apiErr != nil {
		if apiErr.Code == "model_not_found" {
			// The router echoes the whole client model; cap what we send back.
			apiErr = openai.ErrModelNotFound(truncateBytes(req.Model, maxLoggedModelBytes))
		}
		s.fail(w, rs, apiErr)
		return
	}
	rs.model = dec.Candidates[0].Model.ID

	timeout := s.deadline(tenant, req)
	finish, ok := s.admit(w, rs, r, tenant, int64(reqs.InUpper)+int64(dec.Candidates[0].K), timeout+leaseGrace)
	if !ok {
		return
	}
	defer finish()

	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	if req.Stream {
		s.proxyStream(ctx, w, rs, r, req, dec)
		return
	}
	s.proxyComplete(ctx, w, rs, r, req, dec)
}

// defaultReadBodyTimeout bounds how long a client may take to send its body
// when Options.ReadBodyTimeout is zero.
const defaultReadBodyTimeout = 30 * time.Second

// readBody reads the request body under the size limit and a read deadline,
// so a client that stalls mid-body cannot hold the connection and its slot.
// A server that cannot set deadlines (such as a test recorder) just reads.
func (s *Server) readBody(w http.ResponseWriter, r *http.Request) ([]byte, *openai.APIError) {
	rc := http.NewResponseController(w)
	timeout := s.readBodyTimeout
	if timeout <= 0 {
		timeout = defaultReadBodyTimeout
	}
	if err := rc.SetReadDeadline(time.Now().Add(timeout)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return nil, openai.ErrInvalidRequest("cannot read request body", "")
	}
	limit := s.cfg.Server.MaxBodyBytes
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if errors.Is(err, os.ErrDeadlineExceeded) {
		// Leave the expired deadline in place. Before it writes a response
		// the server tries to discard the unread body, and with the deadline
		// cleared that would wait on the stalled client indefinitely.
		return nil, openai.ErrRequestTimeout()
	}
	// Best effort: the deadline only guards the read above.
	_ = rc.SetReadDeadline(time.Time{})
	if err == nil {
		return body, nil
	}
	var tooBig *http.MaxBytesError
	if errors.As(err, &tooBig) {
		return nil, openai.ErrRequestTooLarge(limit)
	}
	return nil, openai.ErrInvalidRequest("cannot read request body", "")
}

// Header names for the per-request routing controls (spec 7.1).
const (
	headerMinQuality = "x-routepilot-min-quality"
	headerSLOTTFTMs  = "x-routepilot-slo-ttft-ms"
	headerSLOE2EMs   = "x-routepilot-slo-e2e-ms"
	headerTimeoutMs  = "x-routepilot-timeout-ms"
)

// mergeRoutingHeaders fills the routing options the body left unset from the
// x-routepilot-* headers, so the body wins field by field. Every header that
// is present must parse; a header that fills a field must also pass the body
// validation. Both failures are a 400 whose param is the header name.
func mergeRoutingHeaders(h http.Header, req *openai.ChatCompletionRequest) *openai.APIError {
	var opts openai.RoutingOptions
	var found bool
	// sourced maps the body-style param of a header-filled field to its header.
	sourced := map[string]string{}

	raw := func(name string) (string, bool) {
		vals, ok := h[http.CanonicalHeaderKey(name)]
		if !ok || len(vals) == 0 {
			return "", false
		}
		found = true
		return vals[0], true
	}
	parseMs := func(name string) (*int, *openai.APIError) {
		v, ok := raw(name)
		if !ok {
			return nil, nil
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, openai.ErrInvalidRequest(name+" must be an integer", name)
		}
		return &n, nil
	}

	var apiErr *openai.APIError
	if v, ok := raw(headerMinQuality); ok {
		q, err := strconv.ParseFloat(v, 64)
		if err != nil || math.IsNaN(q) {
			return openai.ErrInvalidRequest(headerMinQuality+" must be a number between 0 and 1", headerMinQuality)
		}
		opts.MinQuality = &q
	}
	if opts.SLOTTFTMs, apiErr = parseMs(headerSLOTTFTMs); apiErr != nil {
		return apiErr
	}
	if opts.SLOE2EMs, apiErr = parseMs(headerSLOE2EMs); apiErr != nil {
		return apiErr
	}
	if opts.TimeoutMs, apiErr = parseMs(headerTimeoutMs); apiErr != nil {
		return apiErr
	}
	if !found {
		return nil
	}

	if req.RoutePilot == nil {
		req.RoutePilot = &openai.RoutingOptions{}
	}
	rp := req.RoutePilot
	if rp.MinQuality == nil && opts.MinQuality != nil {
		rp.MinQuality, sourced["routepilot.min_quality"] = opts.MinQuality, headerMinQuality
	}
	if rp.SLOTTFTMs == nil && opts.SLOTTFTMs != nil {
		rp.SLOTTFTMs, sourced["routepilot.slo_ttft_ms"] = opts.SLOTTFTMs, headerSLOTTFTMs
	}
	if rp.SLOE2EMs == nil && opts.SLOE2EMs != nil {
		rp.SLOE2EMs, sourced["routepilot.slo_e2e_ms"] = opts.SLOE2EMs, headerSLOE2EMs
	}
	if rp.TimeoutMs == nil && opts.TimeoutMs != nil {
		rp.TimeoutMs, sourced["routepilot.timeout_ms"] = opts.TimeoutMs, headerTimeoutMs
	}

	// The body already passed Validate, so any failure now came from a header.
	if apiErr := req.Validate(); apiErr != nil {
		if name, ok := sourced[apiErr.Param]; ok {
			apiErr.Param = name
			apiErr.Message = fmt.Sprintf("%s: %s", name, apiErr.Message)
		}
		return apiErr
	}
	return nil
}

// maxLoggedModelBytes caps how much of a client-supplied model is logged or
// echoed back in an error.
const maxLoggedModelBytes = 256

// truncateBytes cuts s to at most n bytes without splitting a UTF-8 rune.
func truncateBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

// deadline is the request timeout: routepilot.timeout_ms if set, else the
// server default, clamped to the tenant maximum when set, else the server's.
func (s *Server) deadline(tenant *config.TenantConfig, req *openai.ChatCompletionRequest) time.Duration {
	ms := s.cfg.Server.DefaultTimeoutMs
	if rp := req.RoutePilot; rp != nil && rp.TimeoutMs != nil {
		ms = *rp.TimeoutMs
	}
	limit := s.cfg.Server.MaxTimeoutMs
	if tenant.MaxTimeoutMs > 0 {
		limit = tenant.MaxTimeoutMs
	}
	if ms > limit {
		ms = limit
	}
	return time.Duration(ms) * time.Millisecond
}

// promptBytes sums the text bytes of every message. It is logged in place of
// the prompt itself.
func promptBytes(req *openai.ChatCompletionRequest) int {
	n := 0
	for _, m := range req.Messages {
		n += len(m.Content.PlainText())
	}
	return n
}

// writeJSON writes a complete JSON response under a write deadline, so a
// client that stops reading cannot hold the handler. A server that cannot set
// deadlines (such as a test recorder) just writes.
func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		openai.WriteError(w, openai.ErrInternal())
		return
	}
	_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(clientWriteTimeout))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}
