package telemetry

import (
	"math"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// series is one sample of a gathered metric family, with only what the tests read.
type series struct {
	labels  map[string]string
	value   float64 // counters and gauges
	count   uint64  // histograms
	sum     float64
	buckets []float64 // upper bounds of a histogram's finite buckets
}

// gather returns every sample of reg by family name, and each family's type name.
func gather(t *testing.T, reg *prometheus.Registry) (map[string][]series, map[string]string) {
	t.Helper()
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	out := map[string][]series{}
	types := map[string]string{}
	for _, mf := range mfs {
		types[mf.GetName()] = mf.GetType().String()
		for _, m := range mf.GetMetric() {
			s := series{labels: map[string]string{}}
			for _, l := range m.GetLabel() {
				s.labels[l.GetName()] = l.GetValue()
			}
			switch {
			case m.GetHistogram() != nil:
				h := m.GetHistogram()
				s.count, s.sum = h.GetSampleCount(), h.GetSampleSum()
				for _, b := range h.GetBucket() {
					if !math.IsInf(b.GetUpperBound(), 1) {
						s.buckets = append(s.buckets, b.GetUpperBound())
					}
				}
			case m.GetCounter() != nil:
				s.value = m.GetCounter().GetValue()
			case m.GetGauge() != nil:
				s.value = m.GetGauge().GetValue()
			}
			out[mf.GetName()] = append(out[mf.GetName()], s)
		}
	}
	return out, types
}

// find returns the one sample of family name whose labels include want, failing the test
// when there is not exactly one.
func find(t *testing.T, got map[string][]series, name string, want map[string]string) series {
	t.Helper()
	var hits []series
	for _, s := range got[name] {
		ok := true
		for k, v := range want {
			if s.labels[k] != v {
				ok = false
			}
		}
		if ok {
			hits = append(hits, s)
		}
	}
	if len(hits) != 1 {
		t.Fatalf("%s%v: %d samples, want 1; family: %+v", name, want, len(hits), got[name])
	}
	return hits[0]
}

func newTestMetrics(tenants ...string) (*Metrics, *prometheus.Registry) {
	reg := prometheus.NewRegistry()
	return NewMetrics(tenants, reg), reg
}

// specSeries are the series of spec 7.7, then the three the plan adds.
var specSeries = []string{
	"rp_requests_total", "rp_routing_decisions_total", "rp_request_duration_seconds", "rp_ttft_seconds",
	"rp_gateway_overhead_seconds", "rp_decision_duration_seconds", "rp_upstream_attempts_total",
	"rp_cost_estimated_picousd_total", "rp_cost_actual_picousd_total", "rp_budget_rejections_total",
	"rp_ratelimit_rejections_total", "rp_budget_overshoot_picousd_total", "rp_breaker_state",
	"rp_deployment_error_rate", "rp_policy_info", "rp_policy_rollbacks_total",
	"rp_policy_reload_errors_total", "rp_decision_log_dropped_total", "rp_accounting_write_errors_total",
}

func TestExactlyTheSpecSeries(t *testing.T) {
	m, reg := newTestMetrics("tenant-a")
	m.SetSources(Sources{
		BreakerStates:        func() map[string]int { return map[string]int{"dep-a": BreakerClosed} },
		DeploymentErrorRates: func() map[string]float64 { return map[string]float64{"dep-a": 0} },
		PolicyInfo:           func() []PolicyLabels { return []PolicyLabels{{Version: "v1", Role: "stable"}} },
	})
	m.Request("tenant-a", "model-a", 200, "static", time.Millisecond)
	m.Decision("model-a", "static-cheapest", nil, false)
	m.Attempt("dep-a", "ok")

	got, types := gather(t, reg)
	names := make([]string, 0, len(got))
	for n := range got {
		names = append(names, n)
	}
	sort.Strings(names)
	want := slices.Clone(specSeries)
	sort.Strings(want)
	if !slices.Equal(names, want) {
		t.Errorf("series = %v\nwant     %v", names, want)
	}

	wantType := map[string]string{
		"rp_requests_total": "COUNTER", "rp_routing_decisions_total": "COUNTER",
		"rp_request_duration_seconds": "HISTOGRAM", "rp_ttft_seconds": "HISTOGRAM",
		"rp_gateway_overhead_seconds": "HISTOGRAM", "rp_decision_duration_seconds": "HISTOGRAM",
		"rp_upstream_attempts_total": "COUNTER", "rp_cost_estimated_picousd_total": "COUNTER",
		"rp_cost_actual_picousd_total": "COUNTER", "rp_budget_rejections_total": "COUNTER",
		"rp_ratelimit_rejections_total": "COUNTER", "rp_budget_overshoot_picousd_total": "COUNTER",
		"rp_breaker_state": "GAUGE", "rp_deployment_error_rate": "GAUGE", "rp_policy_info": "GAUGE", "rp_policy_rollbacks_total": "COUNTER",
		"rp_policy_reload_errors_total": "COUNTER", "rp_decision_log_dropped_total": "COUNTER",
		"rp_accounting_write_errors_total": "COUNTER",
	}
	for n, w := range wantType {
		if types[n] != w {
			t.Errorf("%s has type %s, want %s", n, types[n], w)
		}
	}
}

func TestTenantLabelBounded(t *testing.T) {
	m, reg := newTestMetrics("tenant-a", "tenant-b")
	m.Request("tenant-a", "model-a", 200, "static", time.Millisecond)
	m.Request("tenant-b", "model-a", 200, "static", time.Millisecond)
	for i := 0; i < 500; i++ {
		m.Request("attacker-"+string(rune('a'+i%26))+"-"+time.Duration(i).String(), "model-a", 401, "none", time.Millisecond)
	}
	m.Request("", "model-a", 401, "none", time.Millisecond)

	got, _ := gather(t, reg)
	tenants := map[string]float64{}
	for _, s := range got["rp_requests_total"] {
		tenants[s.labels["tenant"]] += s.value
	}
	if len(tenants) != 3 {
		t.Errorf("tenant labels = %v, want tenant-a, tenant-b and %q only", tenants, UnknownTenant)
	}
	if tenants["tenant-a"] != 1 || tenants["tenant-b"] != 1 || tenants[UnknownTenant] != 501 {
		t.Errorf("tenant counts = %v, want 1, 1 and 501 for %q", tenants, UnknownTenant)
	}
	if m.Tenant("tenant-a") != "tenant-a" || m.Tenant("nobody") != UnknownTenant || m.Tenant("") != UnknownTenant {
		t.Errorf("Tenant() = %q, %q, %q", m.Tenant("tenant-a"), m.Tenant("nobody"), m.Tenant(""))
	}
}

func TestBucketBounds(t *testing.T) {
	m, reg := newTestMetrics()
	m.TTFT(time.Millisecond)
	m.Overhead(time.Microsecond)
	m.DecisionDuration(time.Microsecond)
	m.Request("t", "m", 200, "static", time.Millisecond)
	got, _ := gather(t, reg)

	check := func(name string, lo, hi float64) {
		t.Helper()
		b := got[name][0].buckets
		if len(b) < 10 {
			t.Fatalf("%s has %d buckets, want an exponential ladder", name, len(b))
		}
		if b[0] != lo || b[len(b)-1] != hi {
			t.Errorf("%s buckets span %v..%v, want exactly %v..%v", name, b[0], b[len(b)-1], lo, hi)
		}
		factor := b[1] / b[0]
		for i := 2; i < len(b); i++ {
			if f := b[i] / b[i-1]; math.Abs(f-factor)/factor > 2e-3 {
				t.Errorf("%s bucket %d factor %v, want the constant %v (exponential)", name, i, f, factor)
			}
		}
		if factor < 1.5 || factor > 2.5 {
			t.Errorf("%s factor %v is not a sensible ladder", name, factor)
		}
	}
	check("rp_request_duration_seconds", 0.005, 120)
	check("rp_ttft_seconds", 0.005, 120)
	check("rp_decision_duration_seconds", 0.00001, 1)
	check("rp_gateway_overhead_seconds", 0.00001, 1)
}

func TestRequestAndDecisionLabels(t *testing.T) {
	m, reg := newTestMetrics("tenant-a")
	m.Request("tenant-a", "model-a", 429, "canary", 20*time.Millisecond)
	m.Decision("model-a", "v1+abcd1234", nil, false)
	m.Decision("model-a", "v1+abcd1234", []string{"latency", "quality"}, true)
	m.Decision("model-a", "v1+abcd1234", []string{"latency", "quality"}, true)
	m.Attempt("dep-a", "http_503")
	m.Attempt("dep-a", "ok")

	got, _ := gather(t, reg)
	r := find(t, got, "rp_requests_total", map[string]string{"tenant": "tenant-a", "model": "model-a", "status": "429", "policy_role": "canary"})
	if r.value != 1 {
		t.Errorf("requests = %v, want 1", r.value)
	}
	if d := find(t, got, "rp_request_duration_seconds", nil); d.count != 1 || math.Abs(d.sum-0.02) > 1e-9 {
		t.Errorf("request duration count %d sum %v, want 1 and 0.02", d.count, d.sum)
	}
	none := find(t, got, "rp_routing_decisions_total", map[string]string{"relaxed": "none", "ood": "false"})
	both := find(t, got, "rp_routing_decisions_total", map[string]string{"relaxed": "latency,quality", "ood": "true"})
	if none.value != 1 || both.value != 2 {
		t.Errorf("decisions none=%v both=%v, want 1 and 2", none.value, both.value)
	}
	if a := find(t, got, "rp_upstream_attempts_total", map[string]string{"deployment": "dep-a", "outcome": "http_503"}); a.value != 1 {
		t.Errorf("http_503 attempts = %v, want 1", a.value)
	}
}

func TestCostAndRejectionCounters(t *testing.T) {
	m, reg := newTestMetrics()
	m.Cost(1000, 700)
	m.Cost(500, 0)
	m.Cost(-5, -5) // never subtracts
	m.BudgetRejection()
	m.BudgetRejection()
	m.RateLimitRejection(RateLimitRPM)
	m.RateLimitRejection(RateLimitUpstream)
	m.RateLimitRejection(RateLimitUpstream)

	got, _ := gather(t, reg)
	if v := got["rp_cost_estimated_picousd_total"][0].value; v != 1500 {
		t.Errorf("estimated cost = %v, want 1500", v)
	}
	if v := got["rp_cost_actual_picousd_total"][0].value; v != 700 {
		t.Errorf("actual cost = %v, want 700", v)
	}
	if v := got["rp_budget_rejections_total"][0].value; v != 2 {
		t.Errorf("budget rejections = %v, want 2", v)
	}
	want := map[string]float64{"rpm": 1, "tpm": 0, "concurrency": 0, "upstream": 2}
	if len(got["rp_ratelimit_rejections_total"]) != len(want) {
		t.Errorf("rate limit kinds = %+v, want exactly %v", got["rp_ratelimit_rejections_total"], want)
	}
	for kind, n := range want {
		if s := find(t, got, "rp_ratelimit_rejections_total", map[string]string{"kind": kind}); s.value != n {
			t.Errorf("rate limit %s = %v, want %v", kind, s.value, n)
		}
	}
}

func TestNegativeDurationsAreZero(t *testing.T) {
	m, reg := newTestMetrics()
	m.Overhead(-time.Second)
	m.TTFT(-time.Second)
	m.DecisionDuration(-time.Second)
	got, _ := gather(t, reg)
	for _, n := range []string{"rp_gateway_overhead_seconds", "rp_ttft_seconds", "rp_decision_duration_seconds"} {
		if s := got[n][0]; s.count != 1 || s.sum != 0 {
			t.Errorf("%s count %d sum %v, want 1 and 0", n, s.count, s.sum)
		}
	}
}

// TestSourcesAreReadAtScrapeTime: the series that live in other components follow their
// readers from one scrape to the next, and read zero before any source is set.
func TestSourcesAreReadAtScrapeTime(t *testing.T) {
	m, reg := newTestMetrics()
	zero, _ := gather(t, reg)
	for _, n := range []string{
		"rp_budget_overshoot_picousd_total", "rp_policy_rollbacks_total", "rp_policy_reload_errors_total",
		"rp_decision_log_dropped_total", "rp_accounting_write_errors_total",
	} {
		if len(zero[n]) != 1 || zero[n][0].value != 0 {
			t.Errorf("%s before SetSources = %+v, want one zero sample", n, zero[n])
		}
	}
	if len(zero["rp_breaker_state"]) != 0 || len(zero["rp_policy_info"]) != 0 {
		t.Errorf("breaker and policy samples without sources: %+v %+v", zero["rp_breaker_state"], zero["rp_policy_info"])
	}

	var overshoot, accounting, dropped int64 = 5, 6, 7
	var reloads, rollbacks uint64 = 8, 9
	m.SetSources(Sources{
		BudgetOvershootPico:   func() int64 { return overshoot },
		AccountingWriteErrors: func() int64 { return accounting },
		DecisionLogDropped:    func() int64 { return dropped },
		PolicyReloadErrors:    func() uint64 { return reloads },
		PolicyRollbacks:       func() uint64 { return rollbacks },
		BreakerStates: func() map[string]int {
			return map[string]int{"a": BreakerClosed, "b": BreakerHalfOpen, "c": BreakerOpen}
		},
		PolicyInfo: func() []PolicyLabels {
			return []PolicyLabels{{"v1", "stable"}, {"v2", "canary"}, {"v3", "shadow"}}
		},
	})
	check := func(round string, want map[string]float64) {
		t.Helper()
		got, _ := gather(t, reg)
		for n, w := range want {
			if len(got[n]) != 1 || got[n][0].value != w {
				t.Errorf("%s: %s = %+v, want %v", round, n, got[n], w)
			}
		}
	}
	check("first", map[string]float64{
		"rp_budget_overshoot_picousd_total": 5, "rp_accounting_write_errors_total": 6,
		"rp_decision_log_dropped_total": 7, "rp_policy_reload_errors_total": 8, "rp_policy_rollbacks_total": 9,
	})
	overshoot, accounting, dropped, reloads, rollbacks = 50, 60, 70, 80, 90
	check("second", map[string]float64{
		"rp_budget_overshoot_picousd_total": 50, "rp_accounting_write_errors_total": 60,
		"rp_decision_log_dropped_total": 70, "rp_policy_reload_errors_total": 80, "rp_policy_rollbacks_total": 90,
	})

	got, _ := gather(t, reg)
	for dep, want := range map[string]float64{"a": 0, "b": 1, "c": 2} {
		if s := find(t, got, "rp_breaker_state", map[string]string{"deployment": dep}); s.value != want {
			t.Errorf("breaker %s = %v, want %v (0 closed, 1 half-open, 2 open)", dep, s.value, want)
		}
	}
	for _, p := range []PolicyLabels{{"v1", "stable"}, {"v2", "canary"}, {"v3", "shadow"}} {
		if s := find(t, got, "rp_policy_info", map[string]string{"version": p.Version, "role": p.Role}); s.value != 1 {
			t.Errorf("policy info %v = %v, want 1", p, s.value)
		}
	}
}

// TestDuplicateSamplesDoNotFailTheScrape: two shadow slots can carry the same version, and a
// duplicate would make Prometheus fail the whole scrape.
func TestDuplicateSamplesDoNotFailTheScrape(t *testing.T) {
	m, reg := newTestMetrics()
	m.SetSources(Sources{PolicyInfo: func() []PolicyLabels {
		return []PolicyLabels{{"v1", "stable"}, {"v9", "shadow"}, {"v9", "shadow"}}
	}})
	got, _ := gather(t, reg) // fails the test on a Gather error
	if n := len(got["rp_policy_info"]); n != 2 {
		t.Errorf("policy info has %d samples, want 2 (the duplicate dropped): %+v", n, got["rp_policy_info"])
	}
}

// TestBadLabelNeverPanics: a label value the client library rejects drops that observation
// and nothing else.
func TestBadLabelNeverPanics(t *testing.T) {
	m, reg := newTestMetrics("tenant-a")
	bad := string([]byte{0xff, 0xfe})
	m.Request("tenant-a", bad, 200, "static", time.Millisecond)
	m.Decision(bad, "v", nil, false)
	m.Attempt(bad, "ok")
	m.RateLimitRejection(RateLimitKind(bad))

	got, _ := gather(t, reg)
	if len(got["rp_requests_total"]) != 0 || len(got["rp_routing_decisions_total"]) != 0 || len(got["rp_upstream_attempts_total"]) != 0 {
		t.Errorf("a series with an invalid label value was recorded: %+v", got)
	}
	if s := got["rp_request_duration_seconds"][0]; s.count != 1 {
		t.Errorf("the duration of the request was lost with its label: count %d", s.count)
	}
}

func TestNewMetricsPanicsOnADuplicateRegistration(t *testing.T) {
	reg := prometheus.NewRegistry()
	NewMetrics(nil, reg)
	defer func() {
		if recover() == nil {
			t.Error("a second NewMetrics on the same registry did not panic")
		}
	}()
	NewMetrics(nil, reg)
}
