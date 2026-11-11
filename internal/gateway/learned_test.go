package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
	"github.com/omorsi45/routepilot/internal/router"
)

// learnedPrompt is an invented prompt the committed tiny artifact scores in distribution:
// tiny-strong has the higher lower bound (about 0.66 against 0.41 for tiny-weak), and
// tiny-weak is the cheaper model.
const learnedPrompt = `Solve \frac{1}{2} + \frac{1}{3}`

// policyYAML renders a policy configuration. Different arguments give different Hash8
// values, so two configurations of the one artifact have different slot versions.
func policyYAML(floor, epsilon float64, onInfeasible string) string {
	return fmt.Sprintf("mode: min_cost\nlambda: 0\ndefault_floor: %v\nepsilon: %v\nood_action: conservative\non_infeasible: %s\n",
		floor, epsilon, onInfeasible)
}

// tinyArtifact returns the committed tiny artifact's directory and version, and the
// directory that holds it as <root>/tiny/<version>.
func tinyArtifact(t *testing.T) (dir, version, root string) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "testdata", "policy"))
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Join(root, "tiny"))
	if err != nil || len(entries) != 1 {
		t.Fatalf("want one artifact under %s/tiny, got %v (err %v)", root, entries, err)
	}
	version = entries[0].Name()
	return filepath.Join(root, "tiny", version), version, root
}

type learnedOpts struct {
	policy     string   // stable policy configuration; default: min_cost, floor 0, no exploration
	canary     string   // canary policy configuration; empty means no canary slot
	canaryPct  float64  // percent of requests the canary takes
	shadow     []string // one shadow slot per configuration
	weakUp     string   // upstream_model of tiny-weak, which can carry mock directives
	strongUp   string   // upstream_model of tiny-strong
	rand       func() float64
	latency    *latency.Estimator
	now        func() time.Time
	provider   provider.Provider // replaces the HTTP provider when set
	extraModel string            // YAML of one more catalog model, appended as is
}

// learnedEnv is a gateway routing "auto" with the learned policy over the tiny artifact.
type learnedEnv struct {
	gw       *Server
	reg      *policy.Registry
	cfg      *config.Config
	art      *policy.Artifact
	dir      string // holds the policy configuration files
	upstream *httptest.Server
	logs     *syncLog

	mu   sync.Mutex
	hits map[string]int // upstream requests by upstream model string
}

// syncLog is a log sink safe for the gateway's goroutines.
type syncLog struct {
	mu sync.Mutex
	sb strings.Builder
}

func (l *syncLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.sb.Write(p)
}

func (l *syncLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.sb.String()
}

func (e *learnedEnv) count(upstreamModel string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.hits[upstreamModel]
}

func (e *learnedEnv) total() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := 0
	for _, c := range e.hits {
		n += c
	}
	return n
}

func newLearnedEnv(t *testing.T, o learnedOpts) *learnedEnv {
	t.Helper()
	artDir, version, root := tinyArtifact(t)
	art, err := policy.Load(artDir)
	if err != nil {
		t.Fatal(err)
	}
	e := &learnedEnv{art: art, dir: t.TempDir(), logs: &syncLog{}, hits: map[string]int{}}

	write := func(name, content string) string {
		p := filepath.Join(e.dir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return filepath.ToSlash(p)
	}
	if o.policy == "" {
		o.policy = policyYAML(0, 0, "best_effort")
	}
	if o.weakUp == "" {
		o.weakUp = "weak-up"
	}
	if o.strongUp == "" {
		o.strongUp = "strong-up"
	}
	ref := func(file string) string {
		return fmt.Sprintf("name: tiny\n      version: %s\n      config: %q", version, file)
	}
	var pol strings.Builder
	fmt.Fprintf(&pol, "policies:\n  root: %q\n  stable:\n    name: tiny\n    version: %s\n    config: %q\n",
		filepath.ToSlash(root), version, write("stable.yaml", o.policy))
	if o.canary != "" {
		fmt.Fprintf(&pol, "  canary:\n    name: tiny\n    version: %s\n    config: %q\n    percent: %v\n",
			version, write("canary.yaml", o.canary), o.canaryPct)
	}
	if len(o.shadow) > 0 {
		pol.WriteString("  shadow:\n")
		for i, s := range o.shadow {
			fmt.Fprintf(&pol, "    - %s\n", ref(write(fmt.Sprintf("shadow-%d.yaml", i), s)))
		}
	}

	mockH := mock.NewHandler(mock.Config{})
	e.upstream = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		var body struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(b, &body)
		e.mu.Lock()
		e.hits[body.Model]++
		e.mu.Unlock()
		mockH.ServeHTTP(w, r)
	}))
	t.Cleanup(e.upstream.Close)

	y := fmt.Sprintf(`providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: tiny-weak
    rank: 1
    input_price_per_mtok: "0.100000"
    output_price_per_mtok: "0.400000"
    deployments:
      - name: weak-dep
        provider: mockp
        upstream_model: %q
  - id: tiny-strong
    rank: 2
    input_price_per_mtok: "1.000000"
    output_price_per_mtok: "4.000000"
    deployments:
      - name: strong-dep
        provider: mockp
        upstream_model: %q
%stenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["*"]
routing:
  policy: learned
%s`, e.upstream.URL+"/v1", o.weakUp, o.strongUp, o.extraModel, keyHash(keyA), pol.String())
	e.cfg, err = config.Parse([]byte(y))
	if err != nil {
		t.Fatalf("parse config: %v\n%s", err, y)
	}
	e.reg, err = policy.NewRegistry(func() (config.PoliciesConfig, error) { return e.cfg.Policies, nil }, e.cfg.BaseDir())
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	var prov provider.Provider = openaicompat.New(e.cfg.Providers[0], "", e.upstream.Client())
	if o.provider != nil {
		prov = o.provider
	}
	rnd := o.rand
	if rnd == nil {
		rnd = func() float64 { return 0.5 }
	}
	e.gw, err = New(e.cfg, map[string]provider.Provider{"mockp": prov}, Options{
		Logger: slog.New(slog.NewTextHandler(e.logs, nil)), Policies: e.reg, Rand: rnd,
		Latency: o.latency, Now: o.now,
		Sleep: func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func (e *learnedEnv) post(body string, headers map[string]string) *httptest.ResponseRecorder {
	return doHeaders(e.gw.Handler(), keyA, body, headers)
}

// slotVersion is the version a slot with this policy configuration must report.
func (e *learnedEnv) slotVersion(t *testing.T, yaml string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "p.yaml")
	if err := os.WriteFile(p, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := policy.LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	return e.art.Version() + "+" + c.Hash8()
}

// expected routes a request through the router directly, with the optimizer the gateway
// must have built, and returns the model it chooses and the expected cost it reports.
func (e *learnedEnv) expected(t *testing.T, yaml, prompt string, floor *float64) (string, money.PicoUSD, []string) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "p.yaml")
	if err := os.WriteFile(p, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	pc, err := policy.LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	cat, err := catalog.New(e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	req, apiErr := openai.ParseChatCompletionRequest([]byte(chatBody("auto", prompt)))
	if apiErr != nil {
		t.Fatal(apiErr)
	}
	d, apiErr := router.NewLearned(cat).Decide(router.Input{
		Tenant: &e.cfg.Tenants[0], Requested: "auto", Req: catalog.Derive(req),
		Learned: router.NewOptimizer(e.art, pc, nil), Messages: features.Messages(req), Floor: floor,
	})
	if apiErr != nil {
		t.Fatalf("expected decision failed: %v", apiErr)
	}
	chosen := d.Candidates[0].Model.ID
	for _, s := range d.Scores {
		if s.Model == chosen {
			return chosen, s.CostExp, d.Relaxed
		}
	}
	t.Fatalf("no score for %s", chosen)
	return "", 0, nil
}

func f64(v float64) *float64 { return &v }

func TestLearnedAutoRoutingHeaders(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	wantVersion := e.slotVersion(t, policyYAML(0, 0, "best_effort"))

	cases := []struct {
		name        string
		floor       string
		wantModel   string
		wantRelaxed string
	}{
		{"no floor takes the cheaper model", "0", "tiny-weak", ""},
		{"floor above the weak model", "0.5", "tiny-strong", ""},
		{"floor nobody reaches relaxes quality", "0.95", "tiny-strong", "quality"},
	}
	var weakCost, strongCost float64
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fl, _ := strconv.ParseFloat(tc.floor, 64)
			rec := e.post(chatBody("auto", learnedPrompt), map[string]string{headerMinQuality: tc.floor})
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			h := rec.Header()
			if got := h.Get("x-routepilot-model"); got != tc.wantModel {
				t.Errorf("model header %q, want %q", got, tc.wantModel)
			}
			if body := modelOf(t, rec); body != h.Get("x-routepilot-model") {
				t.Errorf("body model %q differs from the model header %q", body, h.Get("x-routepilot-model"))
			}
			if got, want := h.Get("x-routepilot-deployment"), strings.TrimPrefix(tc.wantModel, "tiny-")+"-dep"; got != want {
				t.Errorf("deployment header %q, want %q", got, want)
			}
			if got := h.Get("x-routepilot-policy-version"); got != wantVersion {
				t.Errorf("policy version %q, want %q", got, wantVersion)
			}
			if got, want := h.Get("x-routepilot-decision-id"), "dec_"+h.Get(requestIDHeader); got != want || len(got) != len("dec_")+24 {
				t.Errorf("decision id %q, want %q", got, want)
			}
			if got := h.Get("x-routepilot-relaxed"); got != tc.wantRelaxed {
				t.Errorf("relaxed header %q, want %q", got, tc.wantRelaxed)
			}
			if _, present := h["X-Routepilot-Relaxed"]; present != (tc.wantRelaxed != "") {
				t.Errorf("relaxed header present = %v, want %v", present, tc.wantRelaxed != "")
			}

			model, cost, relaxed := e.expected(t, policyYAML(0, 0, "best_effort"), learnedPrompt, f64(fl))
			if model != tc.wantModel || strings.Join(relaxed, ",") != tc.wantRelaxed {
				t.Fatalf("test expectation is wrong: router chose %s relaxed %v", model, relaxed)
			}
			if cost <= 0 {
				t.Fatalf("expected cost %d is not positive", cost)
			}
			got := h.Get("x-routepilot-estimated-cost-usd")
			if got != cost.USD() {
				t.Errorf("estimated cost header %q, want %q", got, cost.USD())
			}
			v, err := strconv.ParseFloat(got, 64)
			if err != nil {
				t.Fatalf("estimated cost %q is not a decimal: %v", got, err)
			}
			if tc.wantModel == "tiny-weak" {
				weakCost = v
			} else {
				strongCost = v
			}
		})
	}
	if weakCost >= strongCost {
		t.Errorf("estimated cost of tiny-weak %v should be below tiny-strong %v", weakCost, strongCost)
	}
}

func TestLearnedAutoRoutingHeadersOnStreams(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	body := fmt.Sprintf(`{"model":"auto","stream":true,"messages":[{"role":"user","content":%q}]}`, learnedPrompt)
	rec := e.post(body, map[string]string{headerMinQuality: "0.5"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	h := rec.Header()
	if got := h.Get("x-routepilot-model"); got != "tiny-strong" {
		t.Errorf("model header %q, want tiny-strong", got)
	}
	if got, want := h.Get("x-routepilot-policy-version"), e.slotVersion(t, policyYAML(0, 0, "best_effort")); got != want {
		t.Errorf("policy version %q, want %q", got, want)
	}
	if got, want := h.Get("x-routepilot-decision-id"), "dec_"+h.Get(requestIDHeader); got != want {
		t.Errorf("decision id %q, want %q", got, want)
	}
	if h.Get("x-routepilot-estimated-cost-usd") == "" {
		t.Error("stream response has no estimated cost header")
	}
	if !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Errorf("stream did not complete: %s", rec.Body.String())
	}
}

func TestLearnedBodyAndHeaderControlsReachTheOptimizer(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	// The request body's floor wins over the header, as for every routing option.
	body := fmt.Sprintf(`{"model":"auto","routepilot":{"min_quality":0.5},"messages":[{"role":"user","content":%q}]}`, learnedPrompt)
	rec := e.post(body, map[string]string{headerMinQuality: "0"})
	if got := rec.Header().Get("x-routepilot-model"); got != "tiny-strong" {
		t.Errorf("model %q, want tiny-strong from the body floor 0.5 (status %d)", got, rec.Code)
	}
}

func TestLearnedPolicyDefaultFloor(t *testing.T) {
	// The policy's default floor applies when neither the request nor the tenant sets one.
	e := newLearnedEnv(t, learnedOpts{policy: policyYAML(0.5, 0, "best_effort")})
	rec := e.post(chatBody("auto", learnedPrompt), nil)
	if got := rec.Header().Get("x-routepilot-model"); got != "tiny-strong" {
		t.Errorf("model %q with a policy default floor of 0.5, want tiny-strong (status %d)", got, rec.Code)
	}
	// A request floor below it wins.
	rec = e.post(chatBody("auto", learnedPrompt), map[string]string{headerMinQuality: "0"})
	if got := rec.Header().Get("x-routepilot-model"); got != "tiny-weak" {
		t.Errorf("model %q with a request floor of 0, want tiny-weak", got)
	}
}

func TestLearnedRejectIs422(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{policy: policyYAML(0, 0, "reject")})
	rec := e.post(chatBody("auto", learnedPrompt), map[string]string{headerMinQuality: "0.99"})
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status %d, want 422: %s", rec.Code, rec.Body.String())
	}
	body := decodeErr(t, rec)
	if body.Error.Code != "no_feasible_model" || body.Error.Type != "invalid_request_error" {
		t.Errorf("error %+v, want invalid_request_error no_feasible_model", body.Error)
	}
	if !strings.Contains(body.Error.Message, "quality floor") {
		t.Errorf("message %q does not name the binding constraint", body.Error.Message)
	}
	h := rec.Header()
	if got, want := h.Get("x-routepilot-policy-version"), e.slotVersion(t, policyYAML(0, 0, "reject")); got != want {
		t.Errorf("policy version %q on the 422, want %q", got, want)
	}
	if got, want := h.Get("x-routepilot-decision-id"), "dec_"+h.Get(requestIDHeader); got != want {
		t.Errorf("decision id %q on the 422, want %q", got, want)
	}
	if h.Get("x-routepilot-model") != "" || h.Get("x-routepilot-deployment") != "" {
		t.Errorf("a rejected request carries model headers: %v", h)
	}
	if n := e.total(); n != 0 {
		t.Errorf("upstream saw %d requests for a rejected request, want 0", n)
	}
	if !strings.Contains(e.logs.String(), "no_feasible_model") {
		t.Errorf("log does not carry the error code: %s", e.logs.String())
	}
}

func TestLearnedHeadersOnUpstreamFailure(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{weakUp: "weak-up;fail=http500", strongUp: "strong-up;fail=http500"})
	rec := e.post(chatBody("auto", learnedPrompt), map[string]string{headerMinQuality: "0"})
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	h := rec.Header()
	if got, want := h.Get("x-routepilot-policy-version"), e.slotVersion(t, policyYAML(0, 0, "best_effort")); got != want {
		t.Errorf("policy version %q on a 502, want %q", got, want)
	}
	if got, want := h.Get("x-routepilot-decision-id"), "dec_"+h.Get(requestIDHeader); got != want {
		t.Errorf("decision id %q on a 502, want %q", got, want)
	}
}

func TestPinnedAndStaticPolicyVersionHeaders(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	rec := e.post(chatBody("tiny-weak", "hi"), nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	h := rec.Header()
	if got := h.Get("x-routepilot-policy-version"); got != "pinned" {
		t.Errorf("pinned policy version %q, want pinned", got)
	}
	if got, want := h.Get("x-routepilot-decision-id"), "dec_"+h.Get(requestIDHeader); got != want {
		t.Errorf("decision id %q, want %q", got, want)
	}
	for _, name := range []string{"x-routepilot-estimated-cost-usd", "x-routepilot-relaxed"} {
		if _, present := h[http.CanonicalHeaderKey(name)]; present {
			t.Errorf("pinned response carries %s: %q", name, h.Get(name))
		}
	}

	// A pinned model the tenant may not use still reached the router: it carries the
	// decision headers on the 4xx.
	nf := e.post(chatBody("no-such-model", "hi"), nil)
	if nf.Code != http.StatusNotFound || nf.Header().Get("x-routepilot-policy-version") != "pinned" ||
		nf.Header().Get("x-routepilot-decision-id") == "" {
		t.Errorf("unknown model: status %d, headers %v", nf.Code, nf.Header())
	}

	// A static policy reports static-<name>.
	se := newAttemptEnv(t, attemptOpts{policy: "cheapest", models: []modelSpec{
		{id: "cheap", in: "0.1", out: "0.4", deps: []depSpec{{"d1", "u1"}}},
		{id: "dear", in: "5", out: "9", rank: 2, deps: []depSpec{{"d2", "u2"}}},
	}})
	rec = se.post(chatBody("auto", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("static auto: status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-policy-version"); got != "static-cheapest" {
		t.Errorf("static policy version %q, want static-cheapest", got)
	}
	if got, want := rec.Header().Get("x-routepilot-decision-id"), "dec_"+rec.Header().Get(requestIDHeader); got != want {
		t.Errorf("static decision id %q, want %q", got, want)
	}
	if _, present := rec.Header()["X-Routepilot-Estimated-Cost-Usd"]; present {
		t.Error("a static policy response carries an estimated cost")
	}
}

func TestLearnedModelsListsAuto(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	rec := do(e.gw.Handler(), http.MethodGet, "/v1/models", keyA, "")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"id":"auto"`) {
		t.Errorf("status %d, body %s, want a list that includes auto", rec.Code, rec.Body.String())
	}
}

func TestNewRequiresARegistryForLearned(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	prov := openaicompat.New(e.cfg.Providers[0], "", e.upstream.Client())
	_, err := New(e.cfg, map[string]provider.Provider{"mockp": prov}, Options{})
	if err == nil || !strings.Contains(err.Error(), "Policies") {
		t.Fatalf("New without a registry: err = %v, want one naming Options.Policies", err)
	}
}

func TestLearnedCanaryRoutesByPercent(t *testing.T) {
	// The canary takes the same floor-0 decision but is a different configuration, so only
	// its version tells the slots apart.
	canaryYAML := policyYAML(0.1, 0, "best_effort")
	e := newLearnedEnv(t, learnedOpts{canary: canaryYAML, canaryPct: 100})
	stableV := e.slotVersion(t, policyYAML(0, 0, "best_effort"))
	canaryV := e.slotVersion(t, canaryYAML)

	rec := e.post(chatBody("auto", learnedPrompt), nil)
	if got := rec.Header().Get("x-routepilot-policy-version"); got != canaryV {
		t.Errorf("at 100%% the version is %q, want the canary's %q", got, canaryV)
	}
	e.reg.SetCanaryPercent(0)
	rec = e.post(chatBody("auto", learnedPrompt), nil)
	if got := rec.Header().Get("x-routepilot-policy-version"); got != stableV {
		t.Errorf("at 0%% the version is %q, want the stable %q", got, stableV)
	}

	// The role reaches the request state.
	e.reg.SetCanaryPercent(100)
	rs := &reqState{id: "abc"}
	req, _ := openai.ParseChatCompletionRequest([]byte(chatBody("auto", learnedPrompt)))
	if _, apiErr := e.gw.decide(rs, &e.cfg.Tenants[0], req, catalog.Derive(req)); apiErr != nil {
		t.Fatal(apiErr)
	}
	if rs.policyRole != "canary" {
		t.Errorf("request state role %q, want canary", rs.policyRole)
	}
	rs2 := &reqState{id: "abc"}
	dec, _ := e.gw.decide(rs2, &e.cfg.Tenants[0], req, catalog.Derive(req))
	if dec.PolicyVersion != canaryV {
		t.Errorf("decision version %q, want the canary's %q", dec.PolicyVersion, canaryV)
	}
}

func TestShadowDecidesWithoutActing(t *testing.T) {
	shadowPicksStrong := policyYAML(0.5, 0, "best_effort") // the primary (floor 0) picks tiny-weak
	shadowRejects := policyYAML(0.99, 0, "reject")         // 422 for this request
	shadowExplores := policyYAML(0, 0.9, "best_effort")    // would explore, if it were given a Rand

	plain := newLearnedEnv(t, learnedOpts{})
	shadowed := newLearnedEnv(t, learnedOpts{
		shadow: []string{shadowPicksStrong, shadowRejects, shadowExplores},
		rand:   func() float64 { return 0 },
	})

	want := plain.post(chatBody("auto", learnedPrompt), nil)
	got := shadowed.post(chatBody("auto", learnedPrompt), nil)
	if want.Code != http.StatusOK || got.Code != http.StatusOK {
		t.Fatalf("status %d and %d, want 200 both", want.Code, got.Code)
	}
	if g, w := got.Header().Get("x-routepilot-model"), want.Header().Get("x-routepilot-model"); g != w || g != "tiny-weak" {
		t.Errorf("the shadow changed the served model: %q, without a shadow %q", g, w)
	}
	if g, w := got.Header().Get("x-routepilot-policy-version"), want.Header().Get("x-routepilot-policy-version"); g != w {
		t.Errorf("the shadow changed the reported policy version: %q, without a shadow %q", g, w)
	}
	// Exactly one upstream call, to the primary's model.
	if n, weak := shadowed.total(), shadowed.count("weak-up"); n != 1 || weak != 1 {
		t.Errorf("upstream saw %d requests (%d to tiny-weak), want exactly the one", n, weak)
	}
	if n := shadowed.count("strong-up"); n != 0 {
		t.Errorf("the shadow's choice was called upstream %d times", n)
	}

	// The shadow decisions are kept on the request state for the decision log.
	rs := &reqState{id: "abc"}
	req, _ := openai.ParseChatCompletionRequest([]byte(chatBody("auto", learnedPrompt)))
	dec, apiErr := shadowed.gw.decide(rs, &shadowed.cfg.Tenants[0], req, catalog.Derive(req))
	if apiErr != nil {
		t.Fatal(apiErr)
	}
	if dec.Candidates[0].Model.ID != "tiny-weak" {
		t.Errorf("primary chose %s, want tiny-weak", dec.Candidates[0].Model.ID)
	}
	if len(rs.shadow) != 3 || len(rs.shadowErr) != 3 {
		t.Fatalf("kept %d decisions and %d errors, want 3 and 3", len(rs.shadow), len(rs.shadowErr))
	}
	if rs.shadowErr[0] != nil || rs.shadow[0].Candidates[0].Model.ID != "tiny-strong" {
		t.Errorf("shadow 0: err %v, decision %+v, want tiny-strong", rs.shadowErr[0], rs.shadow[0])
	}
	if got, want := rs.shadow[0].PolicyVersion, shadowed.slotVersion(t, shadowPicksStrong); got != want {
		t.Errorf("shadow 0 version %q, want %q", got, want)
	}
	// An error the shadow hit is kept, never answered; its decision still names the policy.
	if rs.shadowErr[1] == nil || rs.shadowErr[1].Code != "no_feasible_model" {
		t.Errorf("shadow 1 error = %v, want no_feasible_model", rs.shadowErr[1])
	}
	if got, want := rs.shadow[1].PolicyVersion, shadowed.slotVersion(t, shadowRejects); got != want {
		t.Errorf("shadow 1 version %q, want %q", got, want)
	}
	// A shadow gets no Rand, so it never explores.
	if rs.shadow[2].Explored || rs.shadow[2].Epsilon != 0 || rs.shadow[2].Propensity != 1 {
		t.Errorf("shadow 2 explored: %+v", rs.shadow[2])
	}
	if rs.policyRole != "stable" {
		t.Errorf("policy role %q, want stable", rs.policyRole)
	}
}

func TestShadowRunsWhenThePrimaryRejects(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{policy: policyYAML(0, 0, "reject"), shadow: []string{policyYAML(0, 0, "best_effort")}})
	req, _ := openai.ParseChatCompletionRequest([]byte(chatBody("auto", learnedPrompt)))
	q := 0.99
	req.RoutePilot = &openai.RoutingOptions{MinQuality: &q}
	rs := &reqState{id: "abc"}
	_, apiErr := e.gw.decide(rs, &e.cfg.Tenants[0], req, catalog.Derive(req))
	if apiErr == nil || apiErr.Code != "no_feasible_model" {
		t.Fatalf("primary error = %v, want no_feasible_model", apiErr)
	}
	if len(rs.shadow) != 1 || rs.shadowErr[0] != nil || len(rs.shadow[0].Candidates) == 0 {
		t.Errorf("shadow after a primary rejection: %+v err %v", rs.shadow, rs.shadowErr)
	}
}

func TestLearnedExplorationDrawsFromTheGatewayRand(t *testing.T) {
	// With epsilon 0.5 the optimizer draws once to decide whether to explore, and a second
	// time to pick the random model when it does. The draws come from Options.Rand.
	for _, tc := range []struct {
		draw      float64
		wantDraws int64
	}{{0.9, 1}, {0.1, 2}} {
		var draws atomic.Int64
		e := newLearnedEnv(t, learnedOpts{
			policy: policyYAML(0, 0.5, "best_effort"),
			rand:   func() float64 { draws.Add(1); return tc.draw },
		})
		rec := e.post(chatBody("auto", learnedPrompt), nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		if got := draws.Load(); got != tc.wantDraws {
			t.Errorf("draw %v: exploration drew %d values, want %d", tc.draw, got, tc.wantDraws)
		}
	}
}

func TestAllAttemptsUpstream429SurfacesRetryAfter(t *testing.T) {
	cases := []struct {
		name       string
		deps       []depSpec
		stream     bool
		wantStatus int
		wantCode   string
		wantRA     string // "" means the header must be absent
	}{
		{"smallest Retry-After wins", []depSpec{{"dep-a", "ua;fail=http429;retry_after=7"}, {"dep-b", "ub;fail=http429;retry_after=3"}}, false, 429, "rate_limit_exceeded", "3"},
		{"smallest positive wins when one has none", []depSpec{{"dep-a", "ua;fail=http429;retry_after=5"}, {"dep-b", "ub;fail=http429;retry_after=0"}}, false, 429, "rate_limit_exceeded", "5"},
		{"no Retry-After anywhere omits the header", []depSpec{{"dep-a", "ua;fail=http429;retry_after=0"}, {"dep-b", "ub;fail=http429;retry_after=0"}}, false, 429, "rate_limit_exceeded", ""},
		{"streaming path before the commit", []depSpec{{"dep-a", "ua;fail=http429;retry_after=7"}, {"dep-b", "ub;fail=http429;retry_after=4"}}, true, 429, "rate_limit_exceeded", "4"},
		{"a 429 and a 500 stay a 502", []depSpec{{"dep-a", "ua;fail=http429;retry_after=7"}, {"dep-b", "ub;fail=http500"}}, false, 502, "upstream_error", ""},
		{"a 500 and a 429 stay a 502", []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub;fail=http429;retry_after=7"}}, false, 502, "upstream_error", ""},
		{"all 500 stay a 502", []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub;fail=http500"}}, false, 502, "upstream_error", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: tc.deps}}})
			body := chatBody("m", "hi")
			if tc.stream {
				body = streamBody("m")
			}
			rec := e.post(body)
			if rec.Code != tc.wantStatus {
				t.Fatalf("status %d, want %d: %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			eb := decodeErr(t, rec)
			if eb.Error.Code != tc.wantCode {
				t.Errorf("code %q, want %q", eb.Error.Code, tc.wantCode)
			}
			ra, present := rec.Header()["Retry-After"]
			if tc.wantRA == "" && present {
				t.Errorf("Retry-After = %v, want absent", ra)
			}
			if tc.wantRA != "" && (len(ra) != 1 || ra[0] != tc.wantRA) {
				t.Errorf("Retry-After = %v, want [%s]", ra, tc.wantRA)
			}
			if tc.wantStatus == 429 {
				if eb.Error.Message != "upstream rate limited" || eb.Error.Type != "rate_limit_error" {
					t.Errorf("error %+v, want rate_limit_error %q", eb.Error, "upstream rate limited")
				}
				if n := e.total(); n != 2 {
					t.Errorf("upstream saw %d attempts, want both deployments tried", n)
				}
			}
		})
	}
}

func TestUpstream429RetryAfterIsCeilAtLeastOne(t *testing.T) {
	cases := []struct {
		after time.Duration
		want  string // "" means absent
	}{
		{1500 * time.Millisecond, "2"},
		{2 * time.Second, "2"},
		{100 * time.Millisecond, "1"},
		{90 * time.Second, "90"},
		{0, ""},
	}
	for _, tc := range cases {
		t.Run(tc.after.String(), func(t *testing.T) {
			p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
				return nil, &provider.Error{Kind: provider.KindHTTP, Status: 429, Retryable: true, RetryAfter: tc.after, Message: "slow down"}
			}}
			gw, _ := newFakeGateway(t, cfgOpts{}, p)
			rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
			if rec.Code != http.StatusTooManyRequests {
				t.Fatalf("status %d, want 429: %s", rec.Code, rec.Body.String())
			}
			if got := rec.Header().Get("Retry-After"); got != tc.want {
				t.Errorf("Retry-After %q, want %q", got, tc.want)
			}
			if got := decodeErr(t, rec).Error.Message; got != "upstream rate limited" {
				t.Errorf("message %q", got)
			}
		})
	}
}

// fakeClock is a manually advanced clock.
type fakeClock struct{ ns atomic.Int64 }

func (c *fakeClock) now() time.Time {
	return time.Unix(1_700_000_000, 0).Add(time.Duration(c.ns.Load()))
}
func (c *fakeClock) advance(d time.Duration) { c.ns.Add(int64(d)) }

// clockedStream advances a clock before it returns each chunk, so a test controls the time
// to first token and the time between the first token and the end of the stream.
type clockedStream struct {
	clock  *fakeClock
	steps  []time.Duration
	chunks []*openai.ChatCompletionChunk
}

func (s *clockedStream) Recv() (*openai.ChatCompletionChunk, error) {
	if len(s.chunks) == 0 {
		s.clock.advance(s.steps[0])
		return nil, io.EOF
	}
	s.clock.advance(s.steps[0])
	s.steps = s.steps[1:]
	c := s.chunks[0]
	s.chunks = s.chunks[1:]
	return c, nil
}

func (s *clockedStream) Close() error { return nil }

// streamProvider serves a stream whose first token takes ttft and whose remaining time to
// the end is rest, reporting completion tokens in a usage chunk.
func streamProvider(clock *fakeClock, ttft, rest time.Duration, completionTokens int) *fakeProvider {
	return &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
		usage := &openai.ChatCompletionChunk{Usage: &openai.Usage{PromptTokens: 5, CompletionTokens: completionTokens, TotalTokens: 5 + completionTokens}}
		return &clockedStream{
			clock:  clock,
			steps:  []time.Duration{ttft, 0, rest},
			chunks: []*openai.ChatCompletionChunk{textChunk("hello"), usage},
		}, nil
	}}
}

// constantClock never moves, so the estimator's decay does not blur the counts.
func constantClock() func() time.Time {
	t := time.Unix(1_700_000_000, 0)
	return func() time.Time { return t }
}

func newClockedGateway(t *testing.T, p provider.Provider, clock *fakeClock, est *latency.Estimator) *Server {
	t.Helper()
	gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": p}, Options{
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)), Now: clock.now, Latency: est,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return gw
}

func TestStreamFeedsTheLatencyEstimator(t *testing.T) {
	clock := &fakeClock{}
	est := latency.NewEstimator(nil, constantClock())
	gw := newClockedGateway(t, streamProvider(clock, 300*time.Millisecond, time.Second, 100), clock, est)
	deps := []string{"dep-a"}

	post := func() {
		rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamReqBody("model-a", false))
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatalf("status %d, body %s", rec.Code, rec.Body.String())
		}
	}
	for range latency.MinObservations - 1 {
		post()
	}
	if _, ok := latency.TTFTQuantile(est, "model-a", deps, 0.5); ok {
		t.Fatalf("an estimate exists after %d streams, before the %d observations it needs", latency.MinObservations-1, latency.MinObservations)
	}
	post()

	ttft, tpot, ok := est.Quantiles("model-a", deps, 0.5)
	if !ok {
		t.Fatal("no estimate after enough streams")
	}
	// Histogram buckets are 20 per decade: a value reads back as its bucket's upper edge,
	// at most 12.2 percent above it.
	if ttft < 299 || ttft > 340 {
		t.Errorf("TTFT median %v ms, want the bucket of 300 ms", ttft)
	}
	// 1 s between the commit and the end, over 100 completion tokens, is 10 ms per token.
	if tpot < 9.9 || tpot > 11.3 {
		t.Errorf("TPOT median %v ms, want the bucket of 10 ms", tpot)
	}
}

func TestStreamTPOTNeedsMoreThanOneCompletionToken(t *testing.T) {
	for _, tokens := range []int{0, 1} {
		clock := &fakeClock{}
		est := latency.NewEstimator(nil, constantClock())
		gw := newClockedGateway(t, streamProvider(clock, 300*time.Millisecond, time.Second, tokens), clock, est)
		for range latency.MinObservations {
			rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamReqBody("model-a", false))
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d", rec.Code)
			}
		}
		if _, ok := latency.TTFTQuantile(est, "model-a", []string{"dep-a"}, 0.5); !ok {
			t.Errorf("tokens=%d: the TTFT was not observed", tokens)
		}
		if _, _, ok := est.Quantiles("model-a", []string{"dep-a"}, 0.5); ok {
			t.Errorf("tokens=%d: a TPOT was observed from %d completion tokens", tokens, tokens)
		}
	}
}

func TestNonStreamingFeedsNothingToTheEstimator(t *testing.T) {
	est := latency.NewEstimator(nil, constantClock())
	p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		return &openai.ChatCompletion{}, nil
	}}
	gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": p}, Options{Latency: est})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 * latency.MinObservations {
		if rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
	}
	if _, ok := latency.TTFTQuantile(est, "model-a", []string{"dep-a"}, 0.5); ok {
		t.Error("a non-streaming request produced a TTFT observation")
	}
}
