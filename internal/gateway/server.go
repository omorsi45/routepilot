// Package gateway implements the RoutePilot HTTP server: tenant
// authentication, request validation, deadlines, and the proxy to providers.
package gateway

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel/trace"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/decisionlog"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/reliability"
	"github.com/omorsi45/routepilot/internal/router"
	"github.com/omorsi45/routepilot/internal/telemetry"
)

// Options configures a Server.
type Options struct {
	// Logger receives one line per request. Nil discards logs.
	Logger *slog.Logger
	// Now is the clock. Nil uses time.Now.
	Now func() time.Time
	// Rand returns a value in [0,1) for retry jitter and for the learned policy's
	// exploration. Nil uses a mutex-guarded math/rand/v2 source.
	Rand func() float64
	// Sleep waits d between attempts and returns ctx.Err() if ctx ends first.
	// Nil sleeps on a timer.
	Sleep func(ctx context.Context, d time.Duration) error
	// Limiters back the per-tenant request, token and concurrency limits. Any
	// nil field uses the in-process memory implementation.
	Limiters quota.Limiters
	// ReadBodyTimeout bounds how long the gateway waits for a request body.
	// Zero means 30 seconds.
	ReadBodyTimeout time.Duration
	// Policies holds the learned policies (stable, canary, shadow) that decide the
	// "auto" model. Required, and used only, when routing.policy is learned: New returns
	// an error without it.
	Policies *policy.Registry
	// Latency estimates time to first token and time per token per deployment for the
	// learned policy's latency SLOs, and is fed from the streaming path. Nil builds an
	// estimator with no prior profiles, so it knows only what this server observes.
	Latency *latency.Estimator
	// Accounting is the usage ledger: one row per attempt sent to a provider. The rows are
	// queued when the request is over and written in batches in the background, so a request
	// never waits for the store. Nil keeps no ledger. A lost write never fails the request; it
	// is counted (Server.AccountingErrors) and logged. The caller runs Server.CloseUsage after
	// the server has drained, and only then closes the store.
	Accounting accounting.Store
	// DecisionLog receives one record per routing decision (shadow decisions included) and
	// every feedback line. Nil keeps no log, and then POST /v1/feedback is not served. The
	// caller closes it after the server has drained.
	DecisionLog *decisionlog.Writer
	// Registry receives the Prometheus series and backs GET /metrics. Nil uses a registry of
	// the server's own; serve passes one that also holds the Go and process collectors. It
	// must not already hold RoutePilot series: New returns an error then.
	Registry *prometheus.Registry
	// TracerProvider records the request spans (tracing.go). Nil, like the noop provider,
	// records nothing. The caller shuts a provider down after the server has drained.
	TracerProvider trace.TracerProvider
}

// Server is the gateway HTTP server.
type Server struct {
	cfg       *config.Config
	providers map[string]provider.Provider
	// providerCfgs holds the configured timeouts, keyed by provider name.
	providerCfgs map[string]config.ProviderConfig
	logger       *slog.Logger
	now          func() time.Time

	cat    *catalog.Catalog
	router *router.Router
	health *reliability.Health
	// bulkhead caps the in-flight non-streaming attempts of the deployments that set
	// max_inflight; nil when none does (bulkhead.go).
	bulkhead *bulkhead
	rand     func() float64
	sleep    func(ctx context.Context, d time.Duration) error

	// policies is non-nil exactly when routing.policy is learned.
	policies *policy.Registry
	latency  *latency.Estimator

	// guard watches the canary against the stable policy and index maps decision IDs to the
	// slot that made them, for the feedback that reaches the guard (guardrails.go). Both are
	// non-nil exactly when policies is. guardMu serializes EvaluateGuardrails and guardKey is
	// the canary pair the guard is currently comparing, empty while the canary has no share.
	guard    *policy.Guard
	index    *decisionIndex
	guardMu  sync.Mutex
	guardKey string

	// adminKey is the SHA-256 of the admin key and adminOn says whether the admin routes exist
	// (admin.go).
	adminKey [sha256.Size]byte
	adminOn  bool

	limits quota.Limiters
	// budgets holds the budget limit of each tenant that has budget_usd, by tenant ID.
	budgets map[string]quota.BudgetLimit
	// budgetOvershoot is the total committed above reservations (see BudgetOvershoot).
	budgetOvershoot atomic.Int64
	// usageLedger is Options.Accounting, nil for none: it takes the rollout events directly.
	// usage writes the attempt rows to it off the request path (see attachLedger), and
	// accountingErrors counts the writes either lost (see AccountingErrors).
	usageLedger      accounting.Store
	usage            *accounting.Batcher
	accountingErrors atomic.Int64
	// decisions is Options.DecisionLog, nil for none.
	decisions *decisionlog.Writer
	// metrics records the Prometheus series and metricsHandler serves them (metrics.go).
	metrics        *telemetry.Metrics
	metricsHandler http.Handler
	// tracer makes the spans (tracing.go); tracing is false for the noop provider, which
	// lets the request path skip building spans at all.
	tracer  trace.Tracer
	tracing bool
	// readBodyTimeout is Options.ReadBodyTimeout; zero means the default.
	readBodyTimeout time.Duration
	// sem is the global in-flight limit: one token per request in progress.
	// Nil when server.max_inflight is not positive (no limit).
	sem chan struct{}
	// inflight counts the requests being handled, for WaitIdle.
	inflight inflight
}

// New builds a Server. Every deployment's provider must exist in providers.
func New(cfg *config.Config, providers map[string]provider.Provider, opts Options) (*Server, error) {
	if cfg == nil {
		return nil, errors.New("gateway: config is nil")
	}
	learned := cfg.Routing.Policy == config.PolicyLearned
	if learned && opts.Policies == nil {
		return nil, errors.New("gateway: routing.policy is learned, which needs Options.Policies")
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
	for _, p := range cfg.Providers {
		if p.TTFTTimeoutMs <= 0 {
			return nil, fmt.Errorf("gateway: provider %q ttft_timeout_ms must be positive, got %d", p.Name, p.TTFTTimeoutMs)
		}
		if p.IdleTimeoutMs <= 0 {
			return nil, fmt.Errorf("gateway: provider %q idle_timeout_ms must be positive, got %d", p.Name, p.IdleTimeoutMs)
		}
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("gateway: build catalog: %w", err)
	}
	static, err := router.NewPolicy(cfg.Routing, cat)
	if err != nil {
		return nil, fmt.Errorf("gateway: build routing policy: %w", err)
	}
	rt := router.New(cat, static)
	if learned {
		rt = router.NewLearned(cat)
	}
	s := &Server{
		cfg: cfg, providers: providers, logger: opts.Logger, now: opts.Now,
		cat: cat, router: rt, rand: opts.Rand, sleep: opts.Sleep,
	}
	if learned {
		s.policies = opts.Policies
	}
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
	s.latency = opts.Latency
	if s.latency == nil {
		s.latency = latency.NewEstimator(nil, s.now)
	}
	s.health = reliability.NewHealth(s.now)
	// The metrics are built last, so the hook looks s.metrics up when a rejection happens.
	s.bulkhead = newBulkhead(cat, func(deployment string) { s.metrics.BulkheadRejection(deployment) })
	if s.rand == nil {
		s.rand = newLockedRand()
	}
	if s.sleep == nil {
		s.sleep = sleepTimer
	}
	s.limits = fillLimiters(opts.Limiters)
	if s.budgets, err = tenantBudgets(cfg); err != nil {
		return nil, fmt.Errorf("gateway: %w", err)
	}
	s.readBodyTimeout = opts.ReadBodyTimeout
	s.decisions = opts.DecisionLog
	if learned {
		s.initGuard()
	}
	var adminErr error
	if s.adminKey, s.adminOn, adminErr = parseAdminKey(cfg.Server.AdminKeySHA256); adminErr != nil {
		return nil, fmt.Errorf("gateway: %w", adminErr)
	}
	if n := cfg.Server.MaxInflight; n > 0 {
		s.sem = make(chan struct{}, n)
	}
	if err := s.initMetrics(opts.Registry); err != nil {
		return nil, err
	}
	s.setTracerProvider(opts.TracerProvider)
	// Last, so that a New that fails above leaves no writer goroutine behind.
	if opts.Accounting != nil {
		s.attachLedger(opts.Accounting, accounting.BatcherOptions{})
	}
	return s, nil
}

// newLockedRand returns a jitter source that is safe for concurrent use.
func newLockedRand() func() float64 {
	var mu sync.Mutex
	r := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	return func() float64 {
		mu.Lock()
		defer mu.Unlock()
		return r.Float64()
	}
}

// sleepTimer waits d, or returns ctx.Err() if ctx ends first.
func sleepTimer(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
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
	policy      string
	attempts    int // upstream attempts sent
	// blameBudget is the deadline the request would get without a client
	// timeout_ms. A deadline expiry blames the deployment only after 90% of it.
	blameBudget time.Duration
	// usage is the token usage reported by the attempt that produced the
	// response, nil when none was reported. It drives the TPM refund.
	usage    *openai.Usage
	errCode  string
	canceled bool
	// panicked is set when the handler panicked, so the deferred log shows a server error
	// instead of no status.
	panicked bool
	// upstream429 counts the sent attempts that ended in an upstream HTTP 429 and
	// retryAfter is the smallest positive Retry-After among them. When every attempt was
	// a 429 the client gets a 429 too (see failAttempts).
	upstream429 int
	retryAfter  time.Duration

	// The fields below are set when the request reaches the learned policy and are what
	// the decision log (Task 22) reads.

	// policyRole is the role of the slot that decided the request, "stable" or
	// "canary"; empty when the learned policy did not decide it.
	policyRole string
	// policyVersion is the version reported in x-routepilot-policy-version: the slot's
	// version for the learned policy, "static-<policy>" or "pinned" otherwise.
	policyVersion string
	// shadow holds the decision of each shadow slot in snapshot order, and shadowErr the
	// error each one hit (nil when it decided). A shadow decision is never acted on and
	// its error is never answered. Each Decision's PolicyVersion names its slot.
	shadow    []router.Decision
	shadowErr []*openai.APIError
	// estCost is the expected cost of each candidate model under the learned policy, for
	// x-routepilot-estimated-cost-usd.
	estCost map[string]money.PicoUSD

	// The fields below are the budget accounting of the request (Task 20).

	// reqs are the request's derived requirements: the input bound and the cap an attempt is
	// priced with.
	reqs catalog.Requirements
	// budgetLeft is what the tenant may still spend when the request is routed, nil for a
	// tenant without a budget. The router drops the models that do not fit it.
	budgetLeft *money.PicoUSD
	// budget is the request's reservation, nil for a tenant without a budget.
	budget *budgetHold
	// attemptUsages has one record per attempt sent, in order: the model and deployment, the
	// tokens charged, whether they were estimated, and the cost. The budget commit is their
	// sum (budgetHold.finish) and the usage ledger writes one row per record.
	attemptUsages []*attemptUsage

	// The fields below are what the decision log writes once the request is over (decisions.go).
	// They are set only when a decision log is configured.

	// routed is true once the router has answered, success or error, and decision is its
	// answer. decidedAt is when, and required the capabilities the request needs.
	routed    bool
	decision  router.Decision
	decidedAt time.Time
	required  []string
	// promptHash16 is the first 16 hex digits of the SHA-256 of the message texts joined by
	// newlines, the only trace of the prompt that is kept.
	promptHash16 string
	// commitAt is when a stream committed its response, zero for any other request.
	commitAt time.Time

	// The fields below feed the metrics (metrics.go).

	// chat is true for a chat completion request: only those are counted in the request
	// series. role is the policy_role label, set once the router has answered.
	chat bool
	role string
	// limitKind is the kind of rate limit that turned the request away, empty when none did.
	limitKind telemetry.RateLimitKind

	// The fields below feed the guardrails (guardrails.go), for a request the learned policy
	// decided. routeFailed is true when the router refused it and relaxed when the policy gave
	// up a constraint to answer.
	routeFailed bool
	relaxed     bool

	// The fields below are the request's trace (tracing.go). root is never nil on a chat
	// request, and traceCtx carries it for the child spans.
	root     trace.Span
	traceCtx context.Context
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
	// Counted from the first line to the last defer, so WaitIdle returns only after the usage
	// rows are queued.
	s.inflight.enter()
	defer s.inflight.leave()
	start := s.now()
	rs := &reqState{id: newRequestID()}
	w.Header().Set(requestIDHeader, rs.id)
	sw := &statusWriter{ResponseWriter: w}
	// The usage ledger is written once the handler is done, the budget commit included, and
	// also when the handler panics, because the attempts were sent and billed all the same.
	defer s.recordUsage(rs)

	// The trace, the decision record, the request log line and the metrics are written from
	// a defer too, so that a handler that panics still leaves them: it is the request an
	// operator most needs to find. They run before the usage ledger write above. The panic
	// itself goes on to net/http, which closes the connection.
	finished := false
	defer func() {
		rs.panicked = !finished
		if rs.chat {
			s.endTrace(sw, rs)
		}
		s.logDecisions(sw, rs, start)
		s.logRequest(sw, rs, start)
		if rs.chat {
			s.observeRequest(sw, rs, start)
			s.observeGuard(sw, rs, start)
		}
	}()

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
		rs.chat = true
		s.startTrace(r, rs)
		s.handleChat(sw, rs, r)
	case "/v1/models":
		if r.Method != http.MethodGet {
			s.methodNotAllowed(sw, rs, http.MethodGet)
			break
		}
		s.handleModels(sw, rs, r)
	case "/v1/feedback":
		s.handleFeedback(sw, rs, r)
	case adminPoliciesPath, adminRollbackPath:
		s.handleAdmin(sw, rs, r)
	case "/metrics":
		// Unauthenticated, like the health routes (Ruling V1-R18): the series carry
		// identifiers, numbers and outcome names, never a prompt or a key. The endpoint is
		// for the internal network, and is not itself a request in the request series.
		if r.Method != http.MethodGet {
			s.methodNotAllowed(sw, rs, http.MethodGet)
			break
		}
		s.metricsHandler.ServeHTTP(sw, r)
	default:
		s.fail(sw, rs, errRouteNotFound(r.URL.Path))
	}
	finished = true
}

// finalStatus is the status the client got: what was written, 499 when the client left
// before anything was, or 500 when the handler panicked before writing anything (net/http
// then closes the connection, so the client sees no status at all).
func finalStatus(sw *statusWriter, rs *reqState) int {
	if sw.status == 0 && rs.canceled {
		return clientClosedStatus
	}
	if sw.status == 0 && rs.panicked {
		return http.StatusInternalServerError
	}
	return sw.status
}

func (s *Server) logRequest(sw *statusWriter, rs *reqState, start time.Time) {
	status := finalStatus(sw, rs)
	attrs := []slog.Attr{
		slog.String("request_id", rs.id),
		slog.String("tenant", rs.tenant),
		slog.String("model", rs.model),
		slog.String("deployment", rs.deployment),
		slog.Int("status", status),
		slog.Int64("duration_ms", s.now().Sub(start).Milliseconds()),
		slog.Int("prompt_bytes", rs.promptBytes),
		slog.Int("attempts", rs.attempts),
		slog.String("policy", rs.policy),
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
