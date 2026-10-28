package gateway

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/reliability"
	"github.com/omorsi45/routepilot/internal/router"
	"github.com/omorsi45/routepilot/internal/telemetry"
)

// Placeholder label values. Every label of a series that a client can influence takes its
// value from the config, the catalog or this list, never from request text.
const (
	labelNone    = "none"    // nothing to report: the request ended before the field was known
	labelUnknown = "unknown" // a model name that is not in the catalog
)

// initMetrics registers the metrics on reg (a fresh registry when nil) and builds the
// /metrics handler. The series that live in other components are read from them at scrape
// time through Sources. A registry that already holds these series is an error.
func (s *Server) initMetrics(reg *prometheus.Registry) (err error) {
	if reg == nil {
		reg = prometheus.NewRegistry()
	}
	// NewMetrics panics on a duplicate registration, which is a wiring mistake; report it
	// from New like every other.
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("gateway: metrics: %v", r)
		}
	}()

	tenants := make([]string, 0, len(s.cfg.Tenants))
	for _, t := range s.cfg.Tenants {
		tenants = append(tenants, t.ID)
	}
	s.metrics = telemetry.NewMetrics(tenants, reg)

	var deployments []string
	for _, m := range s.cfg.Models {
		for _, d := range m.Deployments {
			deployments = append(deployments, d.Name)
		}
	}
	src := telemetry.Sources{
		BudgetOvershootPico:   func() int64 { return s.budgetOvershoot.Load() },
		AccountingWriteErrors: s.AccountingErrors,
		DecisionLogDropped:    s.DecisionLogDropped,
		BreakerStates: func() map[string]int {
			states := make(map[string]int, len(deployments))
			for _, name := range deployments {
				states[name] = breakerGauge(s.health.State(name))
			}
			return states
		},
	}
	if s.policies != nil {
		src.PolicyReloadErrors = s.policies.ReloadErrors
		src.PolicyRollbacks = s.policies.Rollbacks
		src.PolicyInfo = s.policyInfo
	}
	s.metrics.SetSources(src)

	// A collector that fails must not take the whole scrape down with it.
	s.metricsHandler = promhttp.HandlerFor(reg, promhttp.HandlerOpts{
		ErrorHandling: promhttp.ContinueOnError,
		ErrorLog:      scrapeErrorLog{s.logger},
	})
	return nil
}

// scrapeErrorLog sends the errors of a metrics scrape to the server's logger.
type scrapeErrorLog struct{ l *slog.Logger }

func (e scrapeErrorLog) Println(v ...any) {
	e.l.LogAttrs(context.Background(), slog.LevelWarn, "metrics scrape error", slog.String("error", fmt.Sprint(v...)))
}

// breakerGauge is the value of rp_breaker_state: 0 closed, 1 half-open, 2 open. It is not
// the numeric value of reliability.State, which orders open before half-open.
func breakerGauge(st reliability.State) int {
	switch st {
	case reliability.HalfOpen:
		return telemetry.BreakerHalfOpen
	case reliability.Open:
		return telemetry.BreakerOpen
	}
	return telemetry.BreakerClosed
}

// policyInfo lists the loaded policy slots: stable, the canary if any, and each shadow.
func (s *Server) policyInfo() []telemetry.PolicyLabels {
	snap := s.policies.Snapshot()
	out := []telemetry.PolicyLabels{{Version: snap.Stable.Version, Role: snap.Stable.Role}}
	if snap.Canary != nil {
		out = append(out, telemetry.PolicyLabels{Version: snap.Canary.Version, Role: snap.Canary.Role})
	}
	for _, sh := range snap.Shadow {
		out = append(out, telemetry.PolicyLabels{Version: sh.Version, Role: sh.Role})
	}
	return out
}

// modelLabel is the model label for a request: the catalog model, "auto", "none" before any
// model was named, and "unknown" for anything else, so a client cannot mint label values.
func (s *Server) modelLabel(name string) string {
	switch name {
	case "":
		return labelNone
	case config.ReservedModelAuto:
		return name
	}
	if _, ok := s.cat.Model(name); ok {
		return name
	}
	return labelUnknown
}

// metricRole is the policy_role label: who decided the request. "none" when it ended before
// the router answered.
func metricRole(rs *reqState, d router.Decision) string {
	if d.Policy == "" && d.PolicyVersion == "" {
		return labelNone
	}
	return policyRole(rs, d)
}

// observeDecision records the router's answer: how long it took and, when it chose a model,
// which one. It runs once rs.policyVersion is known.
func (s *Server) observeDecision(rs *reqState, d router.Decision, took time.Duration) {
	rs.role = metricRole(rs, d)
	s.metrics.DecisionDuration(took)
	if len(d.Candidates) > 0 {
		s.metrics.Decision(d.Candidates[0].Model.ID, rs.policyVersion, d.Relaxed, d.OOD)
	}
}

// outcome is how the attempt ended, as the usage ledger and the decision log name it.
func (a *attemptUsage) outcome() string {
	if a.status == "" {
		return statusAborted
	}
	return a.status
}

// observeRequest records a finished chat completion request. It runs after the handler (the
// budget commit included) and before the usage ledger write, so the duration is the same
// as the request log's.
//
// The gateway's own time is the request duration less the time inside the providers' calls,
// attempt by attempt: Complete for a whole answer, the read loop for a stream. It is
// recorded for a request that sent at least one attempt.
func (s *Server) observeRequest(sw *statusWriter, rs *reqState, start time.Time) {
	m := s.metrics
	total := s.now().Sub(start)
	status := finalStatus(sw, rs)
	role := rs.role
	if role == "" {
		role = labelNone
	}
	m.Request(rs.tenant, s.modelLabel(rs.model), status, role, total)
	if !rs.commitAt.IsZero() {
		m.TTFT(rs.commitAt.Sub(start))
	}

	if len(rs.attemptUsages) > 0 {
		actual := rs.settleAttempts()
		var estimated money.PicoUSD
		var inProvider time.Duration
		for _, a := range rs.attemptUsages {
			m.Attempt(a.Deployment, a.outcome())
			estimated = money.AddSat(estimated, a.costEstimate)
			if !a.started.IsZero() && a.ended.After(a.started) {
				inProvider += a.ended.Sub(a.started)
			}
		}
		m.Cost(int64(estimated), int64(actual))
		m.Overhead(total - inProvider)
	}

	// Both refusals are a 429. A request whose client left before the answer was written
	// carries no status and is not a rejection anyone saw.
	if status == http.StatusTooManyRequests {
		if rs.limitKind != "" {
			m.RateLimitRejection(rs.limitKind)
		}
		if rs.errCode == "insufficient_quota" {
			m.BudgetRejection()
		}
	}
}
