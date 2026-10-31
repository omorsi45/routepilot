package telemetry

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// dashboardFile is the Grafana dashboard the compose stack provisions.
var dashboardFile = filepath.Join("..", "..", "deploy", "grafana", "dashboards", "routepilot.json")

type dashboardPanel struct {
	Title      string `json:"title"`
	Type       string `json:"type"`
	Datasource struct {
		UID string `json:"uid"`
	} `json:"datasource"`
	Targets []struct {
		Expr string `json:"expr"`
	} `json:"targets"`
	// Panels holds the children of a row panel.
	Panels []dashboardPanel `json:"panels"`
}

type dashboardDoc struct {
	UID    string           `json:"uid"`
	Title  string           `json:"title"`
	Panels []dashboardPanel `json:"panels"`
}

// exposedFamily is what a scrape of /metrics exposes under one name: its Prometheus type and the
// label names its samples carry.
type exposedFamily struct {
	typ    string
	labels map[string]bool
}

// exposedSeries registers every RoutePilot series on a fresh registry, records one
// observation on each so that labeled families have samples, and returns what a scrape
// exposes: the family names, and for a histogram its _bucket, _sum and _count series.
func exposedSeries(t *testing.T) map[string]exposedFamily {
	t.Helper()
	reg := prometheus.NewRegistry()
	m := NewMetrics([]string{"t"}, reg)
	m.SetSources(Sources{
		BudgetOvershootPico:   func() int64 { return 1 },
		AccountingWriteErrors: func() int64 { return 1 },
		DecisionLogDropped:    func() int64 { return 1 },
		PolicyReloadErrors:    func() uint64 { return 1 },
		PolicyRollbacks:       func() uint64 { return 1 },
		BreakerStates:         func() map[string]int { return map[string]int{"d": BreakerClosed} },
		PolicyInfo:            func() []PolicyLabels { return []PolicyLabels{{Version: "v", Role: "stable"}} },
	})
	m.Request("t", "m", 200, "stable", time.Second)
	m.TTFT(time.Second)
	m.Overhead(time.Millisecond)
	m.DecisionDuration(time.Millisecond)
	m.Decision("m", "v", []string{"quality"}, true)
	m.Attempt("d", "ok")
	m.Cost(1, 1)
	m.BudgetRejection()
	m.RateLimitRejection(RateLimitRPM)
	m.BulkheadRejection("d")

	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	out := map[string]exposedFamily{}
	for _, mf := range families {
		labels := map[string]bool{}
		for _, metric := range mf.GetMetric() {
			for _, l := range metric.GetLabel() {
				labels[l.GetName()] = true
			}
		}
		typ := strings.ToLower(mf.GetType().String())
		if typ != "histogram" {
			out[mf.GetName()] = exposedFamily{typ: typ, labels: labels}
			continue
		}
		withLE := map[string]bool{"le": true}
		for l := range labels {
			withLE[l] = true
		}
		out[mf.GetName()+"_bucket"] = exposedFamily{typ: typ, labels: withLE}
		out[mf.GetName()+"_sum"] = exposedFamily{typ: typ, labels: labels}
		out[mf.GetName()+"_count"] = exposedFamily{typ: typ, labels: labels}
	}
	return out
}

var (
	promString   = regexp.MustCompile(`"(?:[^"\\]|\\.)*"`)
	promMatcher  = regexp.MustCompile(`\{([^}]*)\}`)
	promLabel    = regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=~|!~|!=|=)`)
	promGrouping = regexp.MustCompile(`\b(?:by|without)\s*\(([^)]*)\)`)
	promRange    = regexp.MustCompile(`\[[^\]]*\]`)
	promIdent    = regexp.MustCompile(`[a-zA-Z_:][a-zA-Z0-9_:]*`)
)

// promWords are the PromQL functions, aggregations and keywords the dashboard may use. An
// identifier outside this list must be an exposed series.
var promWords = map[string]bool{
	"rate": true, "irate": true, "increase": true, "sum": true, "avg": true, "min": true, "max": true,
	"count": true, "histogram_quantile": true, "clamp_min": true, "clamp_max": true, "label_replace": true,
	"by": true, "without": true, "on": true, "ignoring": true, "and": true, "or": true, "unless": true,
	"offset": true, "bool": true, "vector": true, "time": true, "round": true, "abs": true,
}

// checkExpr reports every way expr names something a scrape does not expose: an unknown
// series, or a label that none of the series it uses carries.
func checkExpr(expr string, known map[string]exposedFamily) []string {
	var problems []string
	rest := promString.ReplaceAllString(expr, `""`)

	var labelsUsed []string
	for _, m := range promMatcher.FindAllStringSubmatch(rest, -1) {
		for _, l := range promLabel.FindAllStringSubmatch(promString.ReplaceAllString(m[1], `""`), -1) {
			labelsUsed = append(labelsUsed, l[1])
		}
	}
	rest = promMatcher.ReplaceAllString(rest, "")
	for _, m := range promGrouping.FindAllStringSubmatch(rest, -1) {
		for _, l := range strings.FieldsFunc(m[1], func(r rune) bool { return r == ',' || r == ' ' }) {
			labelsUsed = append(labelsUsed, l)
		}
	}
	rest = promGrouping.ReplaceAllString(rest, "")
	rest = promRange.ReplaceAllString(rest, "")

	carried := map[string]bool{}
	named := 0
	for _, id := range promIdent.FindAllString(rest, -1) {
		if promWords[id] {
			continue
		}
		s, ok := known[id]
		if !ok {
			problems = append(problems, fmt.Sprintf("unknown series %q", id))
			continue
		}
		named++
		for l := range s.labels {
			carried[l] = true
		}
	}
	if named == 0 && len(problems) == 0 {
		problems = append(problems, "names no series")
	}
	for _, l := range labelsUsed {
		if len(problems) == 0 && !carried[l] {
			problems = append(problems, fmt.Sprintf("label %q is carried by none of the series used", l))
		}
	}
	return problems
}

// TestCheckExprFlagsDrift: the dashboard check is not vacuous. It accepts a query over
// exposed series and labels, and rejects a misspelt series, a series that never existed and a
// label the series does not carry.
func TestCheckExprFlagsDrift(t *testing.T) {
	known := exposedSeries(t)
	cases := []struct {
		name string
		expr string
		want string // empty: the expression is fine
	}{
		{"counter rate by label", `sum by (status) (rate(rp_requests_total{tenant="t"}[1m]))`, ""},
		{"histogram quantile", `histogram_quantile(0.95, sum by (le) (rate(rp_request_duration_seconds_bucket[5m])))`, ""},
		{"gauge with matcher", `max by (deployment) (rp_breaker_state{deployment=~".+"})`, ""},
		{"misspelt series", `rate(rp_request_total[1m])`, `unknown series "rp_request_total"`},
		{"series that does not exist", `rp_cache_hits_total`, `unknown series "rp_cache_hits_total"`},
		{"histogram base name has no samples", `rate(rp_ttft_seconds[1m])`, `unknown series "rp_ttft_seconds"`},
		{"label the series lacks", `sum by (modle) (rate(rp_routing_decisions_total[1m]))`, `label "modle"`},
		{"matcher label the series lacks", `rp_requests_total{route="x"}`, `label "route"`},
		{"no series at all", `vector(1)`, "names no series"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			problems := checkExpr(tc.expr, known)
			switch {
			case tc.want == "" && len(problems) > 0:
				t.Errorf("%s: unexpected problems %v", tc.expr, problems)
			case tc.want != "" && !strings.Contains(strings.Join(problems, "; "), tc.want):
				t.Errorf("%s: problems %v, want one containing %q", tc.expr, problems, tc.want)
			}
		})
	}
}

func walkPanels(panels []dashboardPanel, visit func(dashboardPanel)) {
	for _, p := range panels {
		visit(p)
		walkPanels(p.Panels, visit)
	}
}

// TestDashboardQueriesUseExposedSeries: every query of the provisioned Grafana dashboard
// names only series, and labels, that GET /metrics exposes, so a renamed series cannot leave
// a panel empty without a test failing. When a series is added to metrics.go and the
// dashboard uses it, add a sample for it in exposedSeries.
func TestDashboardQueriesUseExposedSeries(t *testing.T) {
	raw, err := os.ReadFile(dashboardFile)
	if err != nil {
		t.Fatalf("read dashboard: %v", err)
	}
	var d dashboardDoc
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("dashboard is not valid JSON: %v", err)
	}
	if d.UID == "" || d.Title == "" {
		t.Errorf("dashboard uid %q title %q: both must be set", d.UID, d.Title)
	}
	known := exposedSeries(t)

	var titles []string
	queries := 0
	walkPanels(d.Panels, func(p dashboardPanel) {
		if p.Type == "row" {
			return
		}
		titles = append(titles, p.Title)
		if len(p.Targets) == 0 {
			t.Errorf("panel %q has no query", p.Title)
		}
		if p.Datasource.UID != "prometheus" {
			t.Errorf("panel %q datasource uid %q, want the provisioned \"prometheus\"", p.Title, p.Datasource.UID)
		}
		for _, tg := range p.Targets {
			queries++
			if strings.TrimSpace(tg.Expr) == "" {
				t.Errorf("panel %q has an empty query", p.Title)
				continue
			}
			for _, problem := range checkExpr(tg.Expr, known) {
				t.Errorf("panel %q: %s in %s", p.Title, problem, tg.Expr)
			}
		}
	})
	if queries == 0 {
		t.Fatal("the dashboard has no queries")
	}

	// The panels the compose stack promises, each by a word of its title.
	sort.Strings(titles)
	for _, want := range []string{
		"request rate", "latency", "ttft", "overhead", "routing decisions", "relaxed", "ood",
		"breaker", "budget rejections", "overshoot", "policy info", "rollbacks",
	} {
		found := false
		for _, title := range titles {
			if strings.Contains(strings.ToLower(title), want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("no panel title mentions %q; titles: %v", want, titles)
		}
	}
}
