package gateway

import (
	"fmt"
	"net/http"
	"slices"
	"sort"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/quota"
)

// traceTo gives gw a tracer that records every span in memory and returns the exporter.
func traceTo(t *testing.T, gw *Server) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(t.Context()) })
	gw.setTracerProvider(tp)
	return exp
}

// spanTree is the spans of one trace, found by name.
type spanTree struct {
	t     *testing.T
	spans tracetest.SpanStubs
	root  tracetest.SpanStub
}

func treeOf(t *testing.T, exp *tracetest.InMemoryExporter) *spanTree {
	t.Helper()
	spans := exp.GetSpans()
	tr := &spanTree{t: t, spans: spans}
	roots := 0
	for _, s := range spans {
		if !s.Parent.IsValid() {
			tr.root = s
			roots++
		}
	}
	if roots != 1 {
		t.Fatalf("%d root spans, want 1: %v", roots, tr.names())
	}
	return tr
}

func (tr *spanTree) names() []string {
	var out []string
	for _, s := range tr.spans {
		out = append(out, s.Name)
	}
	return out
}

// children are the spans named name directly under the root, in the order they started.
func (tr *spanTree) children(name string) []tracetest.SpanStub {
	var out []tracetest.SpanStub
	for _, s := range tr.spans {
		if s.Name == name && s.Parent.SpanID() == tr.root.SpanContext.SpanID() {
			out = append(out, s)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].StartTime.Before(out[j].StartTime) })
	return out
}

// render is the tree as indented text, children in start order, for the test log.
func (tr *spanTree) render() string {
	var sb strings.Builder
	line := func(indent string, s tracetest.SpanStub) {
		fmt.Fprintf(&sb, "%s%s", indent, s.Name)
		for _, kv := range s.Attributes {
			fmt.Fprintf(&sb, " %s=%s", kv.Key, kv.Value.Emit())
		}
		if s.Status.Code == codes.Error {
			fmt.Fprintf(&sb, " [error: %s]", s.Status.Description)
		}
		sb.WriteString("\n")
	}
	line("", tr.root)
	kids := slices.Clone(tr.spans)
	sort.SliceStable(kids, func(i, j int) bool { return kids[i].StartTime.Before(kids[j].StartTime) })
	for _, s := range kids {
		if s.Parent.SpanID() == tr.root.SpanContext.SpanID() {
			line("  ", s)
		}
	}
	return sb.String()
}

func (tr *spanTree) only(name string) tracetest.SpanStub {
	tr.t.Helper()
	got := tr.children(name)
	if len(got) != 1 {
		tr.t.Fatalf("%d %q spans under the root, want 1; all spans: %v", len(got), name, tr.names())
	}
	return got[0]
}

// attr is the value of the span's attribute key, and whether it is set.
func attr(s tracetest.SpanStub, key string) (attribute.Value, bool) {
	for _, kv := range s.Attributes {
		if string(kv.Key) == key {
			return kv.Value, true
		}
	}
	return attribute.Value{}, false
}

func wantStr(t *testing.T, s tracetest.SpanStub, key, want string) {
	t.Helper()
	v, ok := attr(s, key)
	if !ok || v.Type() != attribute.STRING || v.AsString() != want {
		t.Errorf("%s: %s = %v (set %v), want %q", s.Name, key, v.Emit(), ok, want)
	}
}

func wantInt(t *testing.T, s tracetest.SpanStub, key string, want int64) {
	t.Helper()
	v, ok := attr(s, key)
	if !ok || v.Type() != attribute.INT64 || v.AsInt64() != want {
		t.Errorf("%s: %s = %v (set %v), want %d", s.Name, key, v.Emit(), ok, want)
	}
}

func wantAbsent(t *testing.T, s tracetest.SpanStub, key string) {
	t.Helper()
	if v, ok := attr(s, key); ok {
		t.Errorf("%s: %s is set to %v, want it absent", s.Name, key, v.Emit())
	}
}

// allowedAttributes are the only attribute keys a span may carry: identifiers, numbers,
// booleans and outcome names. Anything else, message text above all, is a leak.
var allowedAttributes = []string{
	"rp.tenant", "rp.request_id", "rp.model", "rp.policy_version", "rp.p", "rp.l", "rp.relaxed", "rp.ood",
	"rp.deployment", "rp.attempt", "rp.outcome", "http.response.status_code",
}

func wantOnlyAllowedAttributes(t *testing.T, exp *tracetest.InMemoryExporter) {
	t.Helper()
	for _, s := range exp.GetSpans() {
		for _, kv := range s.Attributes {
			if !slices.Contains(allowedAttributes, string(kv.Key)) {
				t.Errorf("span %s carries the attribute %s=%v, which is not on the allow list", s.Name, kv.Key, kv.Value.Emit())
			}
		}
		if len(s.Events) != 0 {
			t.Errorf("span %s has %d events, want none", s.Name, len(s.Events))
		}
		if len(s.Links) != 0 {
			t.Errorf("span %s has %d links, want none", s.Name, len(s.Links))
		}
	}
}

// TestSpansAndAttributes: one request on the learned policy for a tenant with a budget is
// the root chat.completions span with the stages under it, and each carries the attributes
// of its stage.
func TestSpansAndAttributes(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	e.cfg.Tenants[0].BudgetUSD, e.cfg.Tenants[0].BudgetPeriod = "10", "monthly"
	gw := rebuild(t, e, Options{})
	exp := traceTo(t, gw)

	rec := doHeaders(gw.Handler(), keyA, chatBody("auto", learnedPrompt), nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	tr := treeOf(t, exp)
	t.Logf("span tree:\n%s", tr.render())

	if tr.root.Name != "chat.completions" {
		t.Errorf("root span is %q, want chat.completions", tr.root.Name)
	}
	wantStr(t, tr.root, "rp.tenant", "tenant-a")
	wantStr(t, tr.root, "rp.request_id", rec.Header().Get(requestIDHeader))
	wantInt(t, tr.root, "http.response.status_code", 200)
	if tr.root.Status.Code == codes.Error {
		t.Errorf("root span status = %v on a success", tr.root.Status)
	}

	// Admission and the budget each take two steps: the slot and lease before the body is
	// read and the limits after the decision; the read of what is left and the reservation.
	if got := len(tr.children("admission")); got != 2 {
		t.Errorf("%d admission spans, want 2", got)
	}
	if got := len(tr.children("budget")); got != 2 {
		t.Errorf("%d budget spans, want 2", got)
	}

	d := tr.only("decide")
	wantStr(t, d, "rp.model", rec.Header().Get("x-routepilot-model"))
	wantStr(t, d, "rp.policy_version", rec.Header().Get("x-routepilot-policy-version"))
	wantStr(t, d, "rp.relaxed", "none")
	for _, k := range []string{"rp.p", "rp.l"} {
		v, ok := attr(d, k)
		if !ok || v.Type() != attribute.FLOAT64 {
			t.Errorf("decide: %s = %v (set %v), want a float", k, v.Emit(), ok)
		}
	}
	if v, ok := attr(d, "rp.p"); ok && (v.AsFloat64() < 0 || v.AsFloat64() > 1) {
		t.Errorf("decide: rp.p = %v, want a probability", v.AsFloat64())
	}
	if v, ok := attr(d, "rp.ood"); !ok || v.Type() != attribute.BOOL || v.AsBool() {
		t.Errorf("decide: rp.ood = %v (set %v), want false", v.Emit(), ok)
	}

	a := tr.only("upstream.attempt")
	wantStr(t, a, "rp.deployment", rec.Header().Get("x-routepilot-deployment"))
	wantInt(t, a, "rp.attempt", 1)
	wantStr(t, a, "rp.outcome", "ok")
	if a.Status.Code == codes.Error {
		t.Errorf("a successful attempt has status %v", a.Status)
	}
	wantOnlyAllowedAttributes(t, exp)
}

// rebuild makes a new server over the config, registry and upstream of a learned env, with
// extra options on top of the ones the env uses, so a test can change what New sees.
func rebuild(t *testing.T, e *learnedEnv, extra Options) *Server {
	t.Helper()
	extra.Policies = e.reg
	if extra.Rand == nil {
		extra.Rand = func() float64 { return 0.5 }
	}
	if extra.Sleep == nil {
		extra.Sleep = e.gw.sleep
	}
	prov := e.gw.providers["mockp"]
	gw, err := New(e.cfg, map[string]provider.Provider{"mockp": prov}, extra)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return gw
}

// TestStaticDecisionHasNoProbabilities: a static policy has no p, L or OOD flag, and the
// span says so by leaving them out.
func TestStaticDecisionHasNoProbabilities(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		policy: config.PolicyCheapest,
		models: []modelSpec{{id: "m", in: "1", out: "1", deps: []depSpec{{"dep-a", "ua"}}}},
	})
	exp := traceTo(t, e.gw)
	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	d := treeOf(t, exp).only("decide")
	wantStr(t, d, "rp.model", "m")
	wantStr(t, d, "rp.policy_version", "static-cheapest")
	wantStr(t, d, "rp.relaxed", "none")
	for _, k := range []string{"rp.p", "rp.l", "rp.ood"} {
		wantAbsent(t, d, k)
	}
}

// TestFailoverSpans: every attempt is a span of its own, numbered from 1 with how it ended,
// and one that failed is an error span whose description is the outcome, not a message.
func TestFailoverSpans(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc"),
	})
	exp := traceTo(t, e.gw)
	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	tr := treeOf(t, exp)
	attempts := tr.children("upstream.attempt")
	if len(attempts) != 2 {
		t.Fatalf("%d attempt spans, want 2: %v", len(attempts), tr.names())
	}
	wantStr(t, attempts[0], "rp.deployment", "cheap-a")
	wantInt(t, attempts[0], "rp.attempt", 1)
	wantStr(t, attempts[0], "rp.outcome", "http_503")
	if attempts[0].Status.Code != codes.Error || attempts[0].Status.Description != "http_503" {
		t.Errorf("failed attempt status = %+v, want Error with the outcome", attempts[0].Status)
	}
	wantStr(t, attempts[1], "rp.deployment", "pricey-a")
	wantInt(t, attempts[1], "rp.attempt", 2)
	wantStr(t, attempts[1], "rp.outcome", "ok")
	wantOnlyAllowedAttributes(t, exp)
}

// TestStreamSpans: a stream has the same tree, its attempt ends with the stream.
func TestStreamSpans(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	exp := traceTo(t, e.gw)
	if rec := e.post(streamReqBody("m", false)); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	tr := treeOf(t, exp)
	a := tr.only("upstream.attempt")
	wantStr(t, a, "rp.outcome", "ok")
	wantInt(t, a, "rp.attempt", 1)
	// The attempt must lie inside the root span. A clock as coarse as Windows' 0.5 ms tick can
	// give every one of these timestamps the same value, so equal times are allowed; what
	// must hold is the order.
	if a.EndTime.Before(a.StartTime) || a.StartTime.Before(tr.root.StartTime) || tr.root.EndTime.Before(a.EndTime) {
		t.Errorf("attempt %v..%v is not inside the root %v..%v", a.StartTime, a.EndTime, tr.root.StartTime, tr.root.EndTime)
	}
	wantOnlyAllowedAttributes(t, exp)
}

// TestFailureSpans: a failed request is an error span with the OpenAI error code as its
// description, on the root and on the stage that failed.
func TestFailureSpans(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	exp := traceTo(t, e.gw)

	do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("m", "hi"))
	tr := treeOf(t, exp)
	if tr.root.Status.Code != codes.Error || tr.root.Status.Description != "invalid_api_key" {
		t.Errorf("401 root status = %+v, want Error invalid_api_key", tr.root.Status)
	}
	wantInt(t, tr.root, "http.response.status_code", 401)
	wantAbsent(t, tr.root, "rp.tenant") // nobody was authenticated
	if len(tr.spans) != 1 {
		t.Errorf("a request refused at authentication has spans %v, want only the root", tr.names())
	}

	exp.Reset()
	e.post(chatBody("no-such-model", "hi"))
	tr = treeOf(t, exp)
	if tr.root.Status.Code != codes.Error || tr.root.Status.Description != "model_not_found" {
		t.Errorf("404 root status = %+v, want Error model_not_found", tr.root.Status)
	}
	d := tr.only("decide")
	if d.Status.Code != codes.Error || d.Status.Description != "model_not_found" {
		t.Errorf("decide status = %+v, want Error model_not_found", d.Status)
	}
	if got := tr.children("upstream.attempt"); len(got) != 0 {
		t.Errorf("%d attempt spans for a request that never reached a provider", len(got))
	}
	for _, s := range exp.GetSpans() {
		if strings.Contains(s.Status.Description, "no-such-model") || strings.Contains(s.Status.Description, " ") {
			t.Errorf("span %s status description %q looks like a message, want a bare code", s.Name, s.Status.Description)
		}
	}
	wantOnlyAllowedAttributes(t, exp)
}

// TestRateLimitedSpan: a refusal in admission ends that span with the error code.
func TestRateLimitedSpan(t *testing.T) {
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) { tenantA(cfg).RPM, tenantA(cfg).RPMBurst = 60, 1 }, okProvider(), quota.Limiters{})
	exp := traceTo(t, gw)
	chat(gw, keyA, chatBody("model-a", "hi"))
	exp.Reset()
	if rec := chat(gw, keyA, chatBody("model-a", "hi")); rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429", rec.Code)
	}
	tr := treeOf(t, exp)
	adm := tr.children("admission")
	if len(adm) != 2 || adm[1].Status.Code != codes.Error || adm[1].Status.Description != "rate_limit_exceeded" {
		t.Errorf("admission spans = %+v, want the second to be Error rate_limit_exceeded", adm)
	}
	if adm[0].Status.Code == codes.Error {
		t.Errorf("the first admission step (slot and lease) failed: %+v", adm[0].Status)
	}
}

// TestOptionsTracerProvider: the provider in Options is what the server records with, and
// without one nothing is recorded and requests are served all the same.
func TestOptionsTracerProvider(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	provs := map[string]provider.Provider{"mockp": okProvider()}

	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(t.Context()) })
	gw, err := New(cfg, provs, Options{TracerProvider: tp})
	if err != nil {
		t.Fatal(err)
	}
	do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if got := fmt.Sprint(treeOf(t, exp).names()); !strings.Contains(got, "chat.completions") || !strings.Contains(got, "upstream.attempt") {
		t.Errorf("spans = %s", got)
	}

	quiet, err := New(cfg, provs, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rec := do(quiet.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
		t.Errorf("status %d with no tracer provider", rec.Code)
	}
}

// TestNonChatRoutesAreNotTraced: only chat completions have a trace.
func TestNonChatRoutesAreNotTraced(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	exp := traceTo(t, e.gw)
	do(e.gw.Handler(), http.MethodGet, "/healthz", "", "")
	do(e.gw.Handler(), http.MethodGet, "/metrics", "", "")
	do(e.gw.Handler(), http.MethodGet, "/v1/models", keyA, "")
	if n := len(exp.GetSpans()); n != 0 {
		t.Errorf("%d spans for routes that are not chat completions", n)
	}
}
