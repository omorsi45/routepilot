// Package gateway implements the RoutePilot HTTP server: tenant
// authentication, request validation, deadlines, and the proxy to providers.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// Options configures a Server.
type Options struct {
	// Logger receives one line per request. Nil discards logs.
	Logger *slog.Logger
	// Now is the clock. Nil uses time.Now.
	Now func() time.Time
}

// Server is the gateway HTTP server.
type Server struct {
	cfg       *config.Config
	providers map[string]provider.Provider
	// providerCfgs holds the configured timeouts, keyed by provider name.
	providerCfgs map[string]config.ProviderConfig
	logger       *slog.Logger
	now          func() time.Time
}

// New builds a Server. Every deployment's provider must exist in providers.
func New(cfg *config.Config, providers map[string]provider.Provider, opts Options) (*Server, error) {
	if cfg == nil {
		return nil, errors.New("gateway: config is nil")
	}
	for _, m := range cfg.Models {
		for _, d := range m.Deployments {
			if providers[d.Provider] == nil {
				return nil, fmt.Errorf("gateway: model %q deployment %q uses provider %q, which was not supplied", m.ID, d.Name, d.Provider)
			}
			if !hasProviderConfig(cfg, d.Provider) {
				return nil, fmt.Errorf("gateway: model %q deployment %q uses provider %q, which has no provider config entry", m.ID, d.Name, d.Provider)
			}
		}
	}
	s := &Server{cfg: cfg, providers: providers, logger: opts.Logger, now: opts.Now}
	s.providerCfgs = make(map[string]config.ProviderConfig, len(cfg.Providers))
	for _, p := range cfg.Providers {
		s.providerCfgs[p.Name] = p
	}
	if s.logger == nil {
		s.logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if s.now == nil {
		s.now = time.Now
	}
	return s, nil
}

// hasProviderConfig reports whether cfg declares a provider with this name.
func hasProviderConfig(cfg *config.Config, name string) bool {
	for _, p := range cfg.Providers {
		if p.Name == name {
			return true
		}
	}
	return false
}

// Handler returns the http.Handler serving all gateway routes.
func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(s.serve)
}

// reqState is the per-request data collected for the request log.
type reqState struct {
	id          string
	tenant      string
	model       string
	deployment  string
	promptBytes int
	errCode     string
	canceled    bool
}

// statusWriter records the response status. It forwards Flush and exposes the
// wrapped writer through Unwrap so http.ResponseController keeps working.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusWriter) Flush() {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// FlushError flushes and reports the failure, which http.ResponseController
// prefers over Flush. The stream path relies on it to notice a dead client.
func (w *statusWriter) FlushError() error {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// clientClosedStatus is logged when the client went away and nothing was
// written. It follows the common 499 convention and is never sent.
const clientClosedStatus = 499

// serve routes one request, then writes its log line.
func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	start := s.now()
	rs := &reqState{id: newRequestID()}
	w.Header().Set(requestIDHeader, rs.id)
	sw := &statusWriter{ResponseWriter: w}

	switch r.URL.Path {
	case "/healthz":
		s.handleText(sw, rs, r, "ok")
	case "/readyz":
		s.handleText(sw, rs, r, "ready")
	case "/v1/chat/completions":
		if r.Method != http.MethodPost {
			s.methodNotAllowed(sw, rs, http.MethodPost)
			break
		}
		s.handleChat(sw, rs, r)
	case "/v1/models":
		if r.Method != http.MethodGet {
			s.methodNotAllowed(sw, rs, http.MethodGet)
			break
		}
		s.handleModels(sw, rs, r)
	default:
		s.fail(sw, rs, &openai.APIError{
			Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found",
			Message: fmt.Sprintf("no route for %s", r.URL.Path),
		})
	}

	s.logRequest(sw, rs, start)
}

func (s *Server) logRequest(sw *statusWriter, rs *reqState, start time.Time) {
	status := sw.status
	if status == 0 && rs.canceled {
		status = clientClosedStatus
	}
	attrs := []slog.Attr{
		slog.String("request_id", rs.id),
		slog.String("tenant", rs.tenant),
		slog.String("model", rs.model),
		slog.String("deployment", rs.deployment),
		slog.Int("status", status),
		slog.Int64("duration_ms", s.now().Sub(start).Milliseconds()),
		slog.Int("prompt_bytes", rs.promptBytes),
	}
	if rs.errCode != "" {
		attrs = append(attrs, slog.String("error_code", rs.errCode))
	}
	s.logger.LogAttrs(context.Background(), slog.LevelInfo, "request", attrs...)
}

// fail writes an OpenAI-shaped error and records its code for the log.
func (s *Server) fail(w http.ResponseWriter, rs *reqState, e *openai.APIError) {
	rs.errCode = e.Code
	if e.Status == http.StatusUnauthorized {
		w.Header().Set("WWW-Authenticate", "Bearer")
	}
	openai.WriteError(w, e)
}

func (s *Server) methodNotAllowed(w http.ResponseWriter, rs *reqState, allow string) {
	w.Header().Set("Allow", allow)
	s.fail(w, rs, &openai.APIError{
		Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed",
		Message: "method not allowed, use " + allow,
	})
}
