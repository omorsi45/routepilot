// Package telemetry holds RoutePilot's Prometheus metrics and OpenTelemetry tracing setup.
//
// Nothing in it sees a prompt, a message or a key: the metrics take identifiers (tenant,
// model, deployment, policy version), numbers and outcome names, and the tenant label is
// bounded to the configured tenants. The components that already keep their own counters
// (the budget ledger, the usage ledger, the decision log, the policy registry, the
// breakers) are read at scrape time through Sources, so none of them imports Prometheus.
package telemetry

import (
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Label values that stand for "none of the configured ones" or "nothing to report".
const (
	// UnknownTenant is the tenant label of a request that carries no configured tenant: a
	// missing or wrong key, or a tenant that is not in the config.
	UnknownTenant = "unknown"
	// NoRelaxation is the relaxed label of a decision that relaxed nothing.
	NoRelaxation = "none"
)

// RateLimitKind is the kind label of rp_ratelimit_rejections_total.
type RateLimitKind string

// The kinds of rate-limit rejection.
const (
	RateLimitRPM         RateLimitKind = "rpm"
	RateLimitTPM         RateLimitKind = "tpm"
	RateLimitConcurrency RateLimitKind = "concurrency"
	// RateLimitUpstream is a request whose every attempt was rate limited by the provider.
	RateLimitUpstream RateLimitKind = "upstream"
)

// The values of rp_breaker_state.
const (
	BreakerClosed   = 0
	BreakerHalfOpen = 1
	BreakerOpen     = 2
)

// PolicyLabels names one loaded policy slot for rp_policy_info.
type PolicyLabels struct {
	Version string
	Role    string
}

// Sources are the readers of the series that live in other components. Each is called at
// every scrape; a nil reader reads as zero (no samples for the two that return a list).
type Sources struct {
	// BudgetOvershootPico is the total committed above reservations, in picodollars.
	BudgetOvershootPico func() int64
	// AccountingWriteErrors counts the usage ledger rows that could not be written.
	AccountingWriteErrors func() int64
	// DecisionLogDropped counts the decision log records dropped on a full queue.
	DecisionLogDropped func() int64
	// PolicyReloadErrors counts the policy reloads that failed and left the old policies.
	PolicyReloadErrors func() uint64
	// PolicyRollbacks counts the rollbacks of the stable policy.
	PolicyRollbacks func() uint64
	// BreakerStates maps each configured deployment to BreakerClosed, BreakerHalfOpen or
	// BreakerOpen.
	BreakerStates func() map[string]int
	// DeploymentErrorRates maps each configured deployment to its error rate, the
	// exponentially weighted average of its recent attempt outcomes, from 0 to 1.
	DeploymentErrorRates func() map[string]float64
	// PolicyInfo lists the loaded policy slots: stable, canary and each shadow.
	PolicyInfo func() []PolicyLabels
}

// Metrics is the set of RoutePilot series on one registry. It is safe for concurrent use.
// A Metrics method never panics: a label value the Prometheus client rejects drops that one
// observation.
type Metrics struct {
	tenants map[string]struct{}
	sources atomic.Pointer[Sources]

	requests     *prometheus.CounterVec
	decisions    *prometheus.CounterVec
	attempts     *prometheus.CounterVec
	rateLimited  *prometheus.CounterVec
	bulkheadFull *prometheus.CounterVec
	requestDur   prometheus.Histogram
	ttft         prometheus.Histogram
	overhead     prometheus.Histogram
	decisionDur  prometheus.Histogram
	costEst      prometheus.Counter
	costActual   prometheus.Counter
	budgetReject prometheus.Counter
}

// requestBuckets are the bounds of the request duration and TTFT histograms: 5 ms to 120 s,
// exponential (a factor of about 1.96, so 16 buckets).
func requestBuckets() []float64 { return exponentialRange(0.005, 120, 16) }

// fastBuckets are the bounds of the decision and gateway overhead histograms: 10 us to 1 s,
// exponential (a factor of about 2.15, so about three buckets a decade).
func fastBuckets() []float64 { return exponentialRange(0.00001, 1, 16) }

// exponentialRange is n bucket bounds growing by a constant factor from lo to hi, rounded to
// four significant digits so the le labels read 0.005 and 120 and not 120.00000000000011.
// The rounding moves a bound by at most 0.05 percent.
func exponentialRange(lo, hi float64, n int) []float64 {
	b := prometheus.ExponentialBucketsRange(lo, hi, n)
	for i, v := range b {
		r, err := strconv.ParseFloat(strconv.FormatFloat(v, 'g', 4, 64), 64)
		if err == nil {
			b[i] = r
		}
	}
	return b
}

// NewMetrics registers every RoutePilot series on reg and returns the handle that records
// them. tenants are the configured tenant IDs: any other tenant becomes UnknownTenant, so a
// client cannot grow the label set. It panics if reg already holds a series of the same
// name, which is a programming error (use one registry per server).
func NewMetrics(tenants []string, reg prometheus.Registerer) *Metrics {
	m := &Metrics{tenants: make(map[string]struct{}, len(tenants))}
	for _, t := range tenants {
		m.tenants[t] = struct{}{}
	}

	m.requests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rp_requests_total",
		Help: "Chat completion requests by tenant, model, HTTP status and policy role.",
	}, []string{"tenant", "model", "status", "policy_role"})
	m.decisions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rp_routing_decisions_total",
		Help: "Routing decisions that chose a model, by model, policy version, relaxed constraints and OOD flag.",
	}, []string{"model", "policy_version", "relaxed", "ood"})
	m.requestDur = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "rp_request_duration_seconds",
		Help:    "Time from the start of a chat completion request to the end of its handling.",
		Buckets: requestBuckets(),
	})
	m.ttft = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "rp_ttft_seconds",
		Help:    "Time from the start of a streaming request to its first committed chunk.",
		Buckets: requestBuckets(),
	})
	m.overhead = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "rp_gateway_overhead_seconds",
		Help:    "Request duration minus the time spent inside the providers' calls, for requests that sent an attempt.",
		Buckets: fastBuckets(),
	})
	m.decisionDur = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "rp_decision_duration_seconds",
		Help:    "Time the router took to decide, shadow policies included.",
		Buckets: fastBuckets(),
	})
	m.attempts = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rp_upstream_attempts_total",
		Help: "Attempts sent to a provider deployment, by outcome.",
	}, []string{"deployment", "outcome"})
	m.costEst = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "rp_cost_estimated_picousd_total",
		Help: "Expected cost in picodollars of the attempts sent, as the routing decision estimated it.",
	})
	m.costActual = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "rp_cost_actual_picousd_total",
		Help: "Cost in picodollars of the attempts sent, from the usage the providers reported.",
	})
	m.budgetReject = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "rp_budget_rejections_total",
		Help: "Requests refused because the tenant budget could not cover them.",
	})
	m.rateLimited = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rp_ratelimit_rejections_total",
		Help: "Requests refused by a rate limit, by kind.",
	}, []string{"kind"})
	// Every kind is present from the start, at zero, so a rate() over it never misses.
	for _, k := range []RateLimitKind{RateLimitRPM, RateLimitTPM, RateLimitConcurrency, RateLimitUpstream} {
		m.rateLimited.WithLabelValues(string(k))
	}
	m.bulkheadFull = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rp_bulkhead_rejections_total",
		Help: "Attempts skipped because the deployment was at its max_inflight limit, by deployment.",
	}, []string{"deployment"})

	reg.MustRegister(m.requests, m.decisions, m.requestDur, m.ttft, m.overhead, m.decisionDur,
		m.attempts, m.costEst, m.costActual, m.budgetReject, m.rateLimited, m.bulkheadFull)

	reg.MustRegister(
		prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "rp_budget_overshoot_picousd_total",
			Help: "Picodollars committed above the reservations of the requests that spent them.",
		}, m.read(func(s *Sources) float64 { return call64(s.BudgetOvershootPico) })),
		prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "rp_policy_rollbacks_total",
			Help: "Rollbacks of the stable policy.",
		}, m.read(func(s *Sources) float64 { return callU64(s.PolicyRollbacks) })),
		prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "rp_policy_reload_errors_total",
			Help: "Policy reloads that failed and left the previous policies serving.",
		}, m.read(func(s *Sources) float64 { return callU64(s.PolicyReloadErrors) })),
		prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "rp_decision_log_dropped_total",
			Help: "Decision log records dropped because the writer's queue was full.",
		}, m.read(func(s *Sources) float64 { return call64(s.DecisionLogDropped) })),
		prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "rp_accounting_write_errors_total",
			Help: "Usage ledger rows that could not be written.",
		}, m.read(func(s *Sources) float64 { return call64(s.AccountingWriteErrors) })),
		&samples{
			desc: prometheus.NewDesc("rp_breaker_state",
				"Circuit breaker state of each configured deployment: 0 closed, 1 half-open, 2 open.",
				[]string{"deployment"}, nil),
			read: m.breakerSamples,
		},
		&samples{
			desc: prometheus.NewDesc("rp_deployment_error_rate",
				"Error rate of each configured deployment: the exponentially weighted average of its recent attempt outcomes, from 0 to 1.",
				[]string{"deployment"}, nil),
			read: m.errorRateSamples,
		},
		&samples{
			desc: prometheus.NewDesc("rp_policy_info",
				"The loaded learned policies, one sample of value 1 per slot (stable, canary, shadow).",
				[]string{"version", "role"}, nil),
			read: m.policySamples,
		},
	)
	return m
}

// SetSources sets where the func-backed series read their values. Until it is called they
// read zero. It may be called again to replace the readers.
func (m *Metrics) SetSources(s Sources) { m.sources.Store(&s) }

// read adapts a reader of Sources to the func a CounterFunc calls.
func (m *Metrics) read(get func(*Sources) float64) func() float64 {
	return func() float64 {
		if s := m.sources.Load(); s != nil {
			return get(s)
		}
		return 0
	}
}

func call64(f func() int64) float64 {
	if f == nil {
		return 0
	}
	return float64(max(f(), 0))
}

func callU64(f func() uint64) float64 {
	if f == nil {
		return 0
	}
	return float64(f())
}

// Tenant returns the tenant label for id: id when it is a configured tenant, else
// UnknownTenant.
func (m *Metrics) Tenant(id string) string {
	if _, ok := m.tenants[id]; ok {
		return id
	}
	return UnknownTenant
}

// Request records one finished chat completion request: the counter by tenant, model, HTTP
// status and policy role, and its duration. The model must be a bounded value (a catalog
// model ID or a fixed placeholder); the caller guarantees that, since this package does not
// know the catalog.
func (m *Metrics) Request(tenant, model string, status int, policyRole string, d time.Duration) {
	if c, err := m.requests.GetMetricWithLabelValues(m.Tenant(tenant), model, strconv.Itoa(status), policyRole); err == nil {
		c.Inc()
	}
	m.requestDur.Observe(seconds(d))
}

// TTFT records the time to first committed chunk of a streaming request.
func (m *Metrics) TTFT(d time.Duration) { m.ttft.Observe(seconds(d)) }

// Overhead records the gateway's own time for a request: its duration less the time spent in
// the providers' calls. A negative value (clock skew) is recorded as zero.
func (m *Metrics) Overhead(d time.Duration) { m.overhead.Observe(seconds(d)) }

// DecisionDuration records how long the router took to decide.
func (m *Metrics) DecisionDuration(d time.Duration) { m.decisionDur.Observe(seconds(d)) }

// Decision counts a routing decision that chose model under policyVersion. relaxed lists the
// constraints the policy relaxed (none, "latency", "quality" or both) and ood says whether
// the request was out of distribution.
func (m *Metrics) Decision(model, policyVersion string, relaxed []string, ood bool) {
	r := NoRelaxation
	if len(relaxed) > 0 {
		r = strings.Join(relaxed, ",")
	}
	if c, err := m.decisions.GetMetricWithLabelValues(model, policyVersion, r, strconv.FormatBool(ood)); err == nil {
		c.Inc()
	}
}

// Attempt counts one attempt sent to a provider deployment and how it ended.
func (m *Metrics) Attempt(deployment, outcome string) {
	if c, err := m.attempts.GetMetricWithLabelValues(deployment, outcome); err == nil {
		c.Inc()
	}
}

// Cost adds the estimated and the actual cost, in picodollars, of the attempts of one
// request. Prometheus counters are floats, so the totals are exact up to 2^53 picodollars
// (about 9,007 USD) and rounded to 53 bits of precision beyond; the usage ledger holds the
// exact amounts.
func (m *Metrics) Cost(estimatedPico, actualPico int64) {
	m.costEst.Add(float64(max(estimatedPico, 0)))
	m.costActual.Add(float64(max(actualPico, 0)))
}

// BudgetRejection counts a request refused for lack of budget.
func (m *Metrics) BudgetRejection() { m.budgetReject.Inc() }

// RateLimitRejection counts a request refused by a rate limit of the given kind.
func (m *Metrics) RateLimitRejection(kind RateLimitKind) {
	if c, err := m.rateLimited.GetMetricWithLabelValues(string(kind)); err == nil {
		c.Inc()
	}
}

// BulkheadRejection counts an attempt skipped because the deployment was at its max_inflight
// limit. The deployment must be a configured one (the caller guarantees it, as for Attempt).
// The series appears with the first rejection of a deployment.
func (m *Metrics) BulkheadRejection(deployment string) {
	if c, err := m.bulkheadFull.GetMetricWithLabelValues(deployment); err == nil {
		c.Inc()
	}
}

func seconds(d time.Duration) float64 { return max(d, 0).Seconds() }

// sample is one labeled value of a func-backed gauge.
type sample struct {
	labels []string
	value  float64
}

// samples is a Collector for a gauge family whose samples are read at scrape time. A sample
// whose labels repeat an earlier one is skipped: Prometheus fails the whole scrape on a
// duplicate, and two shadow slots with the same version would produce one.
type samples struct {
	desc *prometheus.Desc
	read func() []sample
}

func (c *samples) Describe(ch chan<- *prometheus.Desc) { ch <- c.desc }

func (c *samples) Collect(ch chan<- prometheus.Metric) {
	seen := map[string]struct{}{}
	for _, s := range c.read() {
		key := strings.Join(s.labels, "\xff")
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		if mt, err := prometheus.NewConstMetric(c.desc, prometheus.GaugeValue, s.value, s.labels...); err == nil {
			ch <- mt
		}
	}
}

func (m *Metrics) breakerSamples() []sample {
	s := m.sources.Load()
	if s == nil || s.BreakerStates == nil {
		return nil
	}
	states := s.BreakerStates()
	out := make([]sample, 0, len(states))
	for dep, st := range states {
		out = append(out, sample{labels: []string{dep}, value: float64(st)})
	}
	return out
}

func (m *Metrics) errorRateSamples() []sample {
	s := m.sources.Load()
	if s == nil || s.DeploymentErrorRates == nil {
		return nil
	}
	rates := s.DeploymentErrorRates()
	out := make([]sample, 0, len(rates))
	for dep, rate := range rates {
		out = append(out, sample{labels: []string{dep}, value: rate})
	}
	return out
}

func (m *Metrics) policySamples() []sample {
	s := m.sources.Load()
	if s == nil || s.PolicyInfo == nil {
		return nil
	}
	info := s.PolicyInfo()
	out := make([]sample, 0, len(info))
	for _, p := range info {
		out = append(out, sample{labels: []string{p.Version, p.Role}, value: 1})
	}
	return out
}
