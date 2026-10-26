package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
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
	data := make([]modelEntry, 0, len(s.cfg.Models))
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

	limit := s.cfg.Server.MaxBodyBytes
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			s.fail(w, rs, openai.ErrRequestTooLarge(limit))
			return
		}
		s.fail(w, rs, openai.ErrInvalidRequest("cannot read request body", ""))
		return
	}
	req, apiErr := openai.ParseChatCompletionRequest(body)
	if apiErr != nil {
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

	dec, apiErr := s.router.Decide(router.Input{
		Tenant:    tenant,
		Requested: req.Model,
		Req:       catalog.Derive(req),
		Available: s.health.Available,
	})
	rs.policy = dec.Policy
	if apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	rs.model = dec.Candidates[0].Model.ID

	ctx, cancel := context.WithTimeout(r.Context(), s.deadline(tenant, req))
	defer cancel()
	if req.Stream {
		s.proxyStream(ctx, w, rs, r, req, dec)
		return
	}
	s.proxyComplete(ctx, w, rs, r, req, dec)
}

// maxLoggedModelBytes caps how much of an unresolved client model is logged.
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		openai.WriteError(w, openai.ErrInternal())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}
