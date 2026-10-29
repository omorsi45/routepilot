package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/decisionlog"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/reliability"
)

// promSample is one line of a Prometheus text exposition.
type promSample struct {
	name   string
	labels map[string]string
	value  float64
}

// scrape is a parsed GET /metrics.
type scrape struct {
	text    string
	types   map[string]string // metric family name to its # TYPE
	samples []promSample
}

// scrapeOf fetches /metrics from gw the way a Prometheus server would: no key, no body.
func scrapeOf(t *testing.T, gw *Server) *scrape {
	t.Helper()
	rec := do(gw.Handler(), http.MethodGet, "/metrics", "", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics: status %d: %s", rec.Code, rec.Body.String())
	}
	return parseScrape(t, rec.Body.String())
}

func parseScrape(t *testing.T, text string) *scrape {
	t.Helper()
	sc := &scrape{text: text, types: map[string]string{}}
	for _, line := range strings.Split(text, "\n") {
		switch {
		case line == "":
		case strings.HasPrefix(line, "# TYPE "):
			f := strings.Fields(line)
			sc.types[f[2]] = f[3]
		case strings.HasPrefix(line, "#"):
		default:
			sc.samples = append(sc.samples, parseSampleLine(t, line))
		}
	}
	return sc
}

func parseSampleLine(t *testing.T, line string) promSample {
	t.Helper()
	i := strings.IndexAny(line, "{ ")
	if i < 0 {
		t.Fatalf("bad sample line %q", line)
	}
	s := promSample{name: line[:i], labels: map[string]string{}}
	rest := line[i:]
	if rest[0] == '{' {
		j := 1
		for rest[j] != '}' {
			eq := j + strings.IndexByte(rest[j:], '=')
			key := rest[j:eq]
			var val strings.Builder
			k := eq + 2 // past ="
			for ; rest[k] != '"'; k++ {
				if rest[k] == '\\' {
					k++
				}
				val.WriteByte(rest[k])
			}
			s.labels[key] = val.String()
			j = k + 1
			if rest[j] == ',' {
				j++
			}
		}
		rest = rest[j+1:]
	}
	v, err := strconv.ParseFloat(strings.Fields(rest)[0], 64)
	if err != nil {
		t.Fatalf("bad value in %q: %v", line, err)
	}
	s.value = v
	return s
}

// match returns the samples of name whose labels include want.
func (sc *scrape) match(name string, want map[string]string) []promSample {
	var out []promSample
	for _, s := range sc.samples {
		if s.name != name {
			continue
		}
		ok := true
		for k, v := range want {
			if s.labels[k] != v {
				ok = false
			}
		}
		if ok {
			out = append(out, s)
		}
	}
	return out
}

// value is the one sample of name with these labels; it fails the test otherwise.
func (sc *scrape) value(t *testing.T, name string, want map[string]string) float64 {
	t.Helper()
	got := sc.match(name, want)
	if len(got) != 1 {
		t.Fatalf("%s%v: %d samples, want 1\n%s", name, want, len(got), sc.dump(name))
	}
	return got[0].value
}

// count is the value of name, or 0 when it has no such sample.
func (sc *scrape) count(name string, want map[string]string) float64 {
	var sum float64
	for _, s := range sc.match(name, want) {
		sum += s.value
	}
	return sum
}

func (sc *scrape) dump(prefix string) string {
	var lines []string
	for _, l := range strings.Split(sc.text, "\n") {
		if strings.HasPrefix(l, prefix) {
			lines = append(lines, l)
		}
	}
	return strings.Join(lines, "\n")
}

// labelValues is the set of values the label takes across the samples of name.
func (sc *scrape) labelValues(name, label string) []string {
	set := map[string]bool{}
	for _, s := range sc.samples {
		if s.name == name {
			set[s.labels[label]] = true
		}
	}
	out := make([]string, 0, len(set))
	for v := range set {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// specSeries are the series of spec 7.7 and the three the plan adds.
var specSeries = []string{
	"rp_accounting_write_errors_total", "rp_breaker_state", "rp_budget_overshoot_picousd_total",
	"rp_budget_rejections_total", "rp_cost_actual_picousd_total", "rp_cost_estimated_picousd_total",
	"rp_decision_duration_seconds", "rp_decision_log_dropped_total", "rp_gateway_overhead_seconds",
	"rp_policy_info", "rp_policy_reload_errors_total", "rp_policy_rollbacks_total",
	"rp_ratelimit_rejections_total", "rp_request_duration_seconds", "rp_requests_total",
	"rp_routing_decisions_total", "rp_ttft_seconds", "rp_upstream_attempts_total",
}

// TestMetricsSeriesNames: after one routed request on the learned policy, the scrape holds
// exactly the spec 7.7 series, each with at least one sample.
func TestMetricsSeriesNames(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	if rec := e.post(chatBody("auto", learnedPrompt), nil); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	sc := scrapeOf(t, e.gw)

	var names []string
	for n := range sc.types {
		if strings.HasPrefix(n, "rp_") {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	if fmt.Sprint(names) != fmt.Sprint(specSeries) {
		t.Errorf("series\n got %v\nwant %v", names, specSeries)
	}
	for _, n := range specSeries {
		sample := n
		if sc.types[n] == "histogram" {
			sample = n + "_count"
		}
		if len(sc.match(sample, nil)) == 0 {
			t.Errorf("%s has no sample after a routed request", n)
		}
	}
	if got := sc.value(t, "rp_requests_total", map[string]string{"tenant": "tenant-a", "status": "200", "policy_role": "stable"}); got != 1 {
		t.Errorf("rp_requests_total = %v, want 1", got)
	}
	if got := sc.value(t, "rp_routing_decisions_total", map[string]string{"relaxed": "none", "ood": "false"}); got != 1 {
		t.Errorf("rp_routing_decisions_total = %v, want 1", got)
	}
	if got := sc.value(t, "rp_policy_info", map[string]string{"role": "stable"}); got != 1 {
		t.Errorf("rp_policy_info = %v, want 1", got)
	}
}

// TestTenantLabelBounded: whatever tenants and models clients invent, the label sets stay
// the configured ones plus one placeholder each.
func TestTenantLabelBounded(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	for i := 0; i < 25; i++ {
		do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", fmt.Sprintf("test-key-nobody-%d", i), chatBody("m", "hi"))
		e.post(chatBody(fmt.Sprintf("no-such-model-%d", i), "hi"))
	}
	if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	sc := scrapeOf(t, e.gw)

	if got, want := sc.labelValues("rp_requests_total", "tenant"), []string{"tenant-a", "unknown"}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("tenant labels = %v, want %v", got, want)
	}
	if got, want := sc.labelValues("rp_requests_total", "model"), []string{"m", "none", "unknown"}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("model labels = %v, want %v (a client's model name must not become a label)", got, want)
	}
	if got := sc.count("rp_requests_total", map[string]string{"tenant": "unknown", "status": "401", "model": "none"}); got != 25 {
		t.Errorf("401s by unknown keys = %v, want 25", got)
	}
	if got := sc.count("rp_requests_total", map[string]string{"tenant": "tenant-a", "status": "404", "model": "unknown"}); got != 25 {
		t.Errorf("404s for invented models = %v, want 25", got)
	}
}

// TestRequestLabels: tenant, model, status and policy role of each way a request can go.
func TestRequestLabels(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		policy: config.PolicyCheapest,
		models: []modelSpec{{id: "m", in: "1", out: "1", deps: []depSpec{{"dep-a", "ua"}}}},
	})
	e.post(chatBody("m", "hi"))    // pinned by the client
	e.post(chatBody("auto", "hi")) // decided by the static policy
	e.post(chatBody("auto", "hi"))
	do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "", chatBody("m", "hi")) // no key
	e.post(`{"model":`)                                                                  // malformed, after authentication

	sc := scrapeOf(t, e.gw)
	for _, c := range []struct {
		labels map[string]string
		want   float64
	}{
		{map[string]string{"tenant": "tenant-a", "model": "m", "status": "200", "policy_role": "pinned"}, 1},
		{map[string]string{"tenant": "tenant-a", "model": "m", "status": "200", "policy_role": "static"}, 2},
		{map[string]string{"tenant": "unknown", "model": "none", "status": "401", "policy_role": "none"}, 1},
		{map[string]string{"tenant": "tenant-a", "model": "none", "status": "400", "policy_role": "none"}, 1},
	} {
		if got := sc.value(t, "rp_requests_total", c.labels); got != c.want {
			t.Errorf("rp_requests_total%v = %v, want %v", c.labels, got, c.want)
		}
	}
	if got := sc.value(t, "rp_request_duration_seconds_count", nil); got != 5 {
		t.Errorf("request duration count = %v, want one per chat request (5)", got)
	}
	// Decisions are counted for the requests that chose a model, with the chosen model.
	if got := sc.value(t, "rp_routing_decisions_total", map[string]string{"model": "m", "policy_version": "pinned"}); got != 1 {
		t.Errorf("pinned decisions = %v, want 1", got)
	}
	if got := sc.value(t, "rp_routing_decisions_total", map[string]string{"model": "m", "policy_version": "static-cheapest"}); got != 2 {
		t.Errorf("static decisions = %v, want 2", got)
	}
	if got := sc.value(t, "rp_decision_duration_seconds_count", nil); got != 3 {
		t.Errorf("decision duration count = %v, want one per request that reached the router (3)", got)
	}
}

// TestMetricsEndpoint: GET /metrics needs no key (Ruling V1-R18), is plain Prometheus text,
// refuses other methods, and is not itself a request in the request series.
func TestMetricsEndpoint(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})

	rec := do(e.gw.Handler(), http.MethodGet, "/metrics", "", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 without a key: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}
	do(e.gw.Handler(), http.MethodGet, "/metrics", "", "")
	post := do(e.gw.Handler(), http.MethodPost, "/metrics", "", "{}")
	if post.Code != http.StatusMethodNotAllowed || post.Header().Get("Allow") != http.MethodGet {
		t.Errorf("POST /metrics: status %d Allow %q, want 405 and GET", post.Code, post.Header().Get("Allow"))
	}

	sc := scrapeOf(t, e.gw)
	if n := len(sc.match("rp_requests_total", nil)); n != 0 {
		t.Errorf("scrapes were counted as requests: %s", sc.dump("rp_requests_total"))
	}
	if got := sc.value(t, "rp_request_duration_seconds_count", nil); got != 0 {
		t.Errorf("request duration count = %v after only scrapes, want 0", got)
	}
}

// TestOverheadExcludesUpstreamTime: the upstream takes 200 ms to answer, and the overhead
// sample, the gateway's own time, is a small fraction of that. The duration sample holds
// the whole 200 ms.
func TestOverheadExcludesUpstreamTime(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "slow;ttft_ms=200"}}}}})
			body := chatBody("m", "hi")
			if stream {
				body = streamReqBody("m", false)
			}
			rec := e.post(body)
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			sc := scrapeOf(t, e.gw)

			if n := sc.value(t, "rp_gateway_overhead_seconds_count", nil); n != 1 {
				t.Fatalf("overhead samples = %v, want 1", n)
			}
			if sum := sc.value(t, "rp_gateway_overhead_seconds_sum", nil); sum < 0 || sum >= 0.05 {
				t.Errorf("overhead = %v s for a request whose upstream took 200 ms, want under 0.05", sum)
			}
			if sum := sc.value(t, "rp_request_duration_seconds_sum", nil); sum < 0.2 {
				t.Errorf("request duration = %v s, want at least the 0.2 s the upstream took", sum)
			}
		})
	}
}

// TestOverheadNeedsAnAttempt: a request that never reached a provider has no overhead sample.
func TestOverheadNeedsAnAttempt(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	e.post(chatBody("no-such-model", "hi"))
	sc := scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_gateway_overhead_seconds_count", nil); got != 0 {
		t.Errorf("overhead samples = %v for a request with no attempt, want 0", got)
	}
}

// TestStreamRecordsTTFT: a stream that committed has a TTFT sample of about the upstream's
// time to first token; a request that is not a stream has none.
func TestStreamRecordsTTFT(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "slow;ttft_ms=60"}}}}})
	e.post(chatBody("m", "hi"))
	if got := scrapeOf(t, e.gw).value(t, "rp_ttft_seconds_count", nil); got != 0 {
		t.Errorf("TTFT samples after a non-streaming request = %v, want 0", got)
	}
	if rec := e.post(streamReqBody("m", false)); rec.Code != http.StatusOK {
		t.Fatalf("stream status %d: %s", rec.Code, rec.Body.String())
	}
	sc := scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_ttft_seconds_count", nil); got != 1 {
		t.Fatalf("TTFT samples = %v, want 1", got)
	}
	if sum := sc.value(t, "rp_ttft_seconds_sum", nil); sum < 0.05 || sum > 5 {
		t.Errorf("TTFT = %v s, want about 0.06", sum)
	}
}

// TestAttemptsAndCostsMatchTheLedger: a failover is two attempts, and the cost counters are
// the sums of the usage ledger's cost_estimate and cost_actual columns.
func TestAttemptsAndCostsMatchTheLedger(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc"),
	})
	store := &recordingStore{}
	useLedger(t, e.gw, store)
	for i := 0; i < 3; i++ {
		if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
	}
	flushUsage(t, e.gw)
	rows := store.got()
	if len(rows) == 0 {
		t.Fatal("no ledger rows")
	}
	var est, act int64
	for _, r := range rows {
		est += int64(r.CostEstimate)
		act += int64(r.CostActual)
	}
	sc := scrapeOf(t, e.gw)

	if got := sc.value(t, "rp_cost_estimated_picousd_total", nil); got != float64(est) {
		t.Errorf("estimated cost = %v, want the ledger's sum %d", got, est)
	}
	if got := sc.value(t, "rp_cost_actual_picousd_total", nil); got != float64(act) || act == 0 {
		t.Errorf("actual cost = %v, want the ledger's sum %d (non-zero)", got, act)
	}
	if got := sc.value(t, "rp_upstream_attempts_total", map[string]string{"deployment": "cheap-a", "outcome": "http_503"}); got != 3 {
		t.Errorf("cheap-a http_503 attempts = %v, want 3", got)
	}
	if got := sc.value(t, "rp_upstream_attempts_total", map[string]string{"deployment": "pricey-a", "outcome": "ok"}); got != 3 {
		t.Errorf("pricey-a ok attempts = %v, want 3", got)
	}
}

// TestRateLimitRejectionKinds: each way a request is turned away for rate has its own kind.
func TestRateLimitRejectionKinds(t *testing.T) {
	kinds := func(gw *Server) map[string]float64 {
		sc := scrapeOf(t, gw)
		out := map[string]float64{}
		for _, k := range []string{"rpm", "tpm", "concurrency", "upstream"} {
			out[k] = sc.value(t, "rp_ratelimit_rejections_total", map[string]string{"kind": k})
		}
		return out
	}
	want := func(t *testing.T, got map[string]float64, kind string) {
		t.Helper()
		for k, v := range got {
			exp := 0.0
			if k == kind {
				exp = 1
			}
			if v != exp {
				t.Errorf("kind %s = %v, want %v (only %s should have moved): %v", k, v, exp, kind, got)
			}
		}
	}

	t.Run("rpm", func(t *testing.T) {
		gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).RPM, tenantA(cfg).RPMBurst = 60, 1 }, okProvider(), quota.Limiters{})
		chat(gw, keyA, chatBody("model-a", "hi"))
		wantRateLimited(t, chat(gw, keyA, chatBody("model-a", "hi")), "tenant request rate exceeded", "1")
		want(t, kinds(gw), "rpm")
	})
	t.Run("tpm", func(t *testing.T) {
		gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).TPM, tenantA(cfg).TPMBurst = 1000, 500 }, okProvider(), quota.Limiters{})
		wantRateLimited(t, chat(gw, keyA, bodyWithMaxTokens("model-a", maxTokensFor(t, 600), false)), "request exceeds tenant token burst; lower max_tokens", "")
		want(t, kinds(gw), "tpm")
	})
	t.Run("concurrency", func(t *testing.T) {
		release := make(chan struct{})
		started := make(chan struct{}, 4)
		gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).MaxConcurrency = 1 },
			stalledProvider(release, started, textChunk("hello")), quota.Limiters{})
		srv := httptest.NewServer(gw.Handler())
		t.Cleanup(srv.Close)
		resp := startStream(t, context.Background(), srv.URL, streamReqBody("model-a", false))
		<-started
		wantRateLimited(t, chat(gw, keyA, chatBody("model-a", "hi")), "tenant concurrency limit reached", "1")
		close(release)
		_, _ = resp.Body.Read(make([]byte, 1))
		want(t, kinds(gw), "concurrency")
	})
	t.Run("upstream", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{
			models:   []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
			upstream: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusTooManyRequests) }),
		})
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusTooManyRequests {
			t.Fatalf("status %d, want the upstream's 429 passed on: %s", rec.Code, rec.Body.String())
		}
		want(t, kinds(e.gw), "upstream")
	})
}

// TestBudgetRejections: both ways a budget turns a request away are counted.
func TestBudgetRejections(t *testing.T) {
	oneToken := worstCase(t, stdIn, stdOut, 1)
	e := newBudgetEnv(t, budgetOpts{budget: oneToken - 1, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	wantInsufficientQuota(t, e.post(chatBody("auto", "hi")), "budget remaining below the cheapest worst-case cost")
	if got := scrapeOf(t, e.gw).value(t, "rp_budget_rejections_total", nil); got != 1 {
		t.Errorf("budget rejections after an unaffordable request = %v, want 1", got)
	}

	// Enough budget when the request is routed, taken by someone else before it reserves.
	race := newBudgetEnv(t, budgetOpts{budget: worstCase(t, stdIn, stdOut, 100) * 2, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	race.squat(worstCase(t, stdIn, stdOut, 100) * 2)
	race.post(chatBody("auto", "hi"))
	if got := scrapeOf(t, race.gw).value(t, "rp_budget_rejections_total", nil); got != 1 {
		t.Errorf("budget rejections after an exhausted budget = %v, want 1", got)
	}

	ok := newBudgetEnv(t, budgetOpts{budget: worstCase(t, stdIn, stdOut, 100) * 10, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	ok.post(chatBody("auto", "hi"))
	if got := scrapeOf(t, ok.gw).value(t, "rp_budget_rejections_total", nil); got != 0 {
		t.Errorf("budget rejections after a served request = %v, want 0", got)
	}
}

// TestBreakerStateSeries: every configured deployment has a sample, 0 closed, 1 half-open,
// 2 open, and reading it claims nothing.
func TestBreakerStateSeries(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest})
	sc := scrapeOf(t, e.gw)
	for _, dep := range []string{"cheap-a", "pricey-a"} {
		if got := sc.value(t, "rp_breaker_state", map[string]string{"deployment": dep}); got != 0 {
			t.Errorf("breaker %s = %v before any request, want 0 (closed)", dep, got)
		}
	}
	e.gw.health.Trip("cheap-a", time.Minute)
	sc = scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_breaker_state", map[string]string{"deployment": "cheap-a"}); got != 2 {
		t.Errorf("tripped breaker = %v, want 2 (open)", got)
	}
	if got := sc.value(t, "rp_breaker_state", map[string]string{"deployment": "pricey-a"}); got != 0 {
		t.Errorf("untouched breaker = %v, want 0", got)
	}
}

func TestBreakerGauge(t *testing.T) {
	for st, want := range map[reliability.State]int{reliability.Closed: 0, reliability.HalfOpen: 1, reliability.Open: 2} {
		if got := breakerGauge(st); got != want {
			t.Errorf("breakerGauge(%v) = %d, want %d", st, got, want)
		}
	}
}

// TestComponentCountersAreExported: the counters the other components keep appear in the
// scrape, read at the moment of the scrape.
func TestComponentCountersAreExported(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	useLedger(t, e.gw, &recordingStore{fail: errors.New("disk full")})
	e.gw.addOvershoot(123)
	log, err := decisionlog.Open(filepath.Join(t.TempDir(), "decisions.jsonl"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := log.Close(); err != nil { // a closed writer drops what it is given
		t.Fatal(err)
	}
	e.gw.decisions = log

	sc := scrapeOf(t, e.gw)
	for _, n := range []string{"rp_budget_overshoot_picousd_total", "rp_accounting_write_errors_total", "rp_decision_log_dropped_total"} {
		if got := sc.value(t, n, nil); n == "rp_budget_overshoot_picousd_total" && got != 123 || n != "rp_budget_overshoot_picousd_total" && got != 0 {
			t.Errorf("%s = %v before any request", n, got)
		}
	}
	if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("a failing ledger and a dead decision log failed the request: %d %s", rec.Code, rec.Body.String())
	}
	flushUsage(t, e.gw)
	sc = scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_accounting_write_errors_total", nil); got != 1 {
		t.Errorf("rp_accounting_write_errors_total = %v, want 1 (the one row that failed)", got)
	}
	if got := sc.value(t, "rp_decision_log_dropped_total", nil); got < 1 {
		t.Errorf("rp_decision_log_dropped_total = %v, want at least 1", got)
	}
}

// TestPolicySeries: one policy_info sample per loaded slot, and the registry's reload error
// and rollback counters.
func TestPolicySeries(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{
		canary: policyYAML(0.1, 0, "best_effort"), canaryPct: 10,
		shadow: []string{policyYAML(0.2, 0, "best_effort")},
	})
	sc := scrapeOf(t, e.gw)
	for role, yaml := range map[string]string{
		"stable": policyYAML(0, 0, "best_effort"), "canary": policyYAML(0.1, 0, "best_effort"), "shadow": policyYAML(0.2, 0, "best_effort"),
	} {
		if got := sc.value(t, "rp_policy_info", map[string]string{"role": role, "version": e.slotVersion(t, yaml)}); got != 1 {
			t.Errorf("rp_policy_info{role=%s} = %v, want 1", role, got)
		}
	}
	if n := len(sc.match("rp_policy_info", nil)); n != 3 {
		t.Errorf("rp_policy_info has %d samples, want 3: %s", n, sc.dump("rp_policy_info"))
	}

	// A reload that fails keeps the old policies and is counted.
	good := e.cfg.Policies
	bad := good
	bad.Stable.Version = "does-not-exist"
	e.cfg.Policies = bad
	if err := e.reg.Reload(); err == nil {
		t.Fatal("reload of a missing artifact succeeded")
	}
	e.cfg.Policies = good
	sc = scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_policy_reload_errors_total", nil); got != 1 {
		t.Errorf("rp_policy_reload_errors_total = %v, want 1", got)
	}
	if got := sc.value(t, "rp_policy_rollbacks_total", nil); got != 0 {
		t.Errorf("rp_policy_rollbacks_total = %v, want 0", got)
	}

	// Change the stable policy, then roll it back.
	changed := good
	changed.Stable.Config = writePolicyFile(t, e.dir, "stable2.yaml", policyYAML(0.3, 0, "best_effort"))
	e.cfg.Policies = changed
	if err := e.reg.Reload(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := e.reg.RollbackStable(); err != nil {
		t.Fatal(err)
	}
	sc = scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_policy_rollbacks_total", nil); got != 1 {
		t.Errorf("rp_policy_rollbacks_total = %v, want 1", got)
	}
}

func writePolicyFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return filepath.ToSlash(p)
}

// TestOptionsRegistry: a registry the caller supplies receives the series (serve adds the Go
// and process collectors to it), and reusing one registry for two servers is an error, not
// a panic.
func TestOptionsRegistry(t *testing.T) {
	reg := prometheus.NewRegistry()
	cfg := buildConfig(t, cfgOpts{})
	provs := map[string]provider.Provider{"mockp": okProvider()}
	gw, err := New(cfg, provs, Options{Registry: reg})
	if err != nil {
		t.Fatal(err)
	}
	do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, mf := range mfs {
		if mf.GetName() == "rp_requests_total" {
			found = true
		}
	}
	if !found {
		t.Error("the supplied registry has no rp_requests_total")
	}

	if _, err := New(cfg, provs, Options{Registry: reg}); err == nil {
		t.Error("a second server on the same registry was accepted, want an error")
	} else if !strings.Contains(err.Error(), "metrics") {
		t.Errorf("error %q does not say it is about the metrics", err)
	}
}

// brokenCollector fails every scrape it takes part in.
type brokenCollector struct{ desc *prometheus.Desc }

func (b brokenCollector) Describe(ch chan<- *prometheus.Desc) { ch <- b.desc }
func (b brokenCollector) Collect(ch chan<- prometheus.Metric) {
	ch <- prometheus.NewInvalidMetric(b.desc, errors.New("collector is broken"))
}

// TestScrapeSurvivesABrokenCollector: one collector on the shared registry (serve adds the Go
// and process collectors) that fails must not take the RoutePilot series down with it, and
// the failure is logged.
func TestScrapeSurvivesABrokenCollector(t *testing.T) {
	reg := prometheus.NewRegistry()
	reg.MustRegister(brokenCollector{desc: prometheus.NewDesc("broken_thing", "always fails", nil, nil)})
	logs := &syncLog{}
	gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": okProvider()},
		Options{Registry: reg, Logger: slog.New(slog.NewTextHandler(logs, nil))})
	if err != nil {
		t.Fatal(err)
	}
	do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))

	sc := scrapeOf(t, gw) // fails the test unless the status is 200
	if got := sc.value(t, "rp_requests_total", map[string]string{"status": "200"}); got != 1 {
		t.Errorf("rp_requests_total = %v, want 1 next to a broken collector", got)
	}
	if !strings.Contains(logs.String(), "metrics scrape error") || !strings.Contains(logs.String(), "collector is broken") {
		t.Errorf("the broken collector was not logged: %s", logs.String())
	}
}

// TestMetricsNeverBreakARequest: label values come from the config and the catalog only, so
// a request whose model is not valid UTF-8 still gets its answer and a clean scrape.
func TestMetricsNeverBreakARequest(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	rec := e.post(`{"model":"` + string([]byte{0xff, 0xfe}) + `","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code < 400 {
		t.Fatalf("status %d, want an error for an unknown model", rec.Code)
	}
	sc := scrapeOf(t, e.gw)
	if got := sc.labelValues("rp_requests_total", "model"); len(got) != 1 || got[0] != "unknown" {
		t.Errorf("model labels = %v, want [unknown]", got)
	}
}
