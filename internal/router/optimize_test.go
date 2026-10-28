package router

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
)

// The optimizer tests use a fake Scorer, so no artifact is needed. Prices are round
// numbers in USD per million tokens, which are picodollars per token times 1e-6:
//
//	cheap 1 / 2   rank 1   cost_exp 140,000,000   cost_max 200,000,000
//	mid   2 / 4   rank 2   cost_exp 280,000,000   cost_max 400,000,000
//	top  10 / 20  rank 3   cost_exp 1,400,000,000 cost_max 2,000,000,000
//
// for a request with an input bound of 100 tokens, an expected output of 20 tokens and an
// output budget k of 50 (cost_exp = 100*in + 20*out, cost_max = 100*in + 50*out).
type optModel struct {
	id      string
	in, out string
	rank    int
	caps    []string
}

var optModels = []optModel{
	{"cheap", "1", "2", 1, []string{"streaming"}},
	{"mid", "2", "4", 2, []string{"streaming"}},
	{"top", "10", "20", 3, []string{"streaming", "tools"}},
}

func optCatalogOf(t *testing.T, models []optModel) *catalog.Catalog {
	t.Helper()
	cfg := &config.Config{}
	for _, m := range models {
		cfg.Models = append(cfg.Models, config.ModelConfig{
			ID: m.id,
			Deployments: []config.DeploymentConfig{
				{Name: m.id + "-a", Provider: "p", UpstreamModel: m.id},
				{Name: m.id + "-b", Provider: "p", UpstreamModel: m.id},
			},
			ContextWindow: 1_000_000, MaxOutputTokens: 100_000, Capabilities: m.caps,
			InputPricePerMTok: m.in, OutputPricePerMTok: m.out, Rank: m.rank,
		})
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog.New: %v", err)
	}
	return cat
}

func ptr[T any](v T) *T { return &v }

// fakeScorer answers with fixed scores and counts the calls.
type fakeScorer struct {
	version string
	scores  policy.Scores
	calls   int
	msgs    []features.Message
}

func (f *fakeScorer) Version() string { return f.version }
func (f *fakeScorer) HasModel(id string) bool {
	_, ok := f.scores.Models[id]
	return ok
}
func (f *fakeScorer) Score(msgs []features.Message) policy.Scores {
	f.calls++
	f.msgs = msgs
	return f.scores
}

// modelScore is a prediction with the given calibrated probability and lower bound, an
// expected output of 20 tokens, a 0.99 output quantile of 40 and an input ratio of 1.
func modelScore(p, l float64) policy.ModelScore {
	return policy.ModelScore{
		P: p, L: l, PooledRate: 0.3, PooledLCB: 0.2, EOut: 20,
		OutQ: [4]float64{10, 15, 25, 40}, InRatio: 1,
	}
}

func scoresOf(m map[string]policy.ModelScore) policy.Scores {
	return policy.Scores{Models: m}
}

// optRig bundles a catalog, a router, a fake scorer and the default request.
type optRig struct {
	t      *testing.T
	cat    *catalog.Catalog
	router *Router
	scorer *fakeScorer
	cfg    policy.Config
	est    *latency.Estimator
	tenant *config.TenantConfig
}

func defaultPolicyConfig() policy.Config {
	return policy.Config{
		Mode: policy.ModeMinCost, OODAction: policy.OODConservative,
		OnInfeasible: config.OnInfeasibleBestEffort,
	}
}

func newOptRig(t *testing.T, scores map[string]policy.ModelScore, cfg policy.Config) *optRig {
	t.Helper()
	cat := optCatalogOf(t, optModels)
	return &optRig{
		t: t, cat: cat, router: New(cat, nil),
		scorer: &fakeScorer{version: "artv1", scores: scoresOf(scores)},
		cfg:    cfg,
		tenant: &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 50},
	}
}

// input builds the routing question. mutate may adjust it.
func (r *optRig) input(mutate ...func(*Input)) Input {
	in := Input{
		Tenant: r.tenant, Requested: config.ReservedModelAuto,
		Req:     catalog.Requirements{InUpper: 100},
		Learned: NewOptimizer(r.scorer, r.cfg, r.est),
	}
	for _, m := range mutate {
		m(&in)
	}
	return in
}

func (r *optRig) decide(mutate ...func(*Input)) (Decision, *openai.APIError) {
	r.t.Helper()
	return r.router.Decide(r.input(mutate...))
}

func withFloor(f float64) func(*Input)       { return func(in *Input) { in.Floor = ptr(f) } }
func withBudget(b int64) func(*Input)        { return func(in *Input) { in.Budget = ptr(money.PicoUSD(b)) } }
func withE2ESLO(ms int) func(*Input)         { return func(in *Input) { in.SLOE2EMs = ptr(ms) } }
func withTTFTSLO(ms int) func(*Input)        { return func(in *Input) { in.SLOTTFTMs = ptr(ms) } }
func withRand(f func() float64) func(*Input) { return func(in *Input) { in.Rand = f } }

func mustDecide(t *testing.T, r *optRig, mutate ...func(*Input)) Decision {
	t.Helper()
	d, err := r.decide(mutate...)
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	return d
}

func scoreOf(t *testing.T, d Decision, model string) ModelScore {
	t.Helper()
	for _, s := range d.Scores {
		if s.Model == model {
			return s
		}
	}
	t.Fatalf("no score for %s in %+v", model, d.Scores)
	return ModelScore{}
}

// tieredScores is the usual three-model situation: stronger models cost more and have a
// higher lower bound.
func tieredScores(lCheap, lMid, lTop float64) map[string]policy.ModelScore {
	return map[string]policy.ModelScore{
		"cheap": modelScore(0.5, lCheap),
		"mid":   modelScore(0.7, lMid),
		"top":   modelScore(0.9, lTop),
	}
}

// priorEstimator gives every model the same constant (sigma 0) latency, so a request's
// end-to-end p95 is ttft + 40*tpot with the fake scores' 0.99 output quantile of 40.
func priorEstimator(profiles map[string][2]float64) *latency.Estimator {
	priors := map[string]latency.Profile{}
	for model, v := range profiles {
		priors[model] = latency.Profile{
			Model: model, TTFTMedianMs: v[0], TPOTMedianMs: v[1],
			Source: "test", Retrieved: "2026-10-28",
		}
	}
	return latency.NewEstimator(priors, func() time.Time { return time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC) })
}

func TestMinCostPicksCheapestAboveFloor(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.DefaultFloor = 0.5
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), cfg)

	d := mustDecide(t, r)
	if got := candidateIDs(d); !slices.Equal(got, []string{"mid", "top", "cheap"}) {
		t.Errorf("candidates = %v, want [mid top cheap]: cheapest above the floor, then the rest of the safe set, then the others by lower bound", got)
	}
	if d.Policy != "learned" || d.PolicyVersion != "artv1+"+cfg.Hash8() {
		t.Errorf("policy %q version %q, want learned and artv1+%s", d.Policy, d.PolicyVersion, cfg.Hash8())
	}
	if d.Floor != 0.5 || len(d.Relaxed) != 0 || d.Explored || d.Propensity != 1 {
		t.Errorf("decision = floor %v relaxed %v explored %v propensity %v", d.Floor, d.Relaxed, d.Explored, d.Propensity)
	}
	mid := scoreOf(t, d, "mid")
	if !mid.Feasible || mid.P != 0.7 || mid.L != 0.6 || mid.CostExp != 280_000_000 || mid.CostMax != 400_000_000 {
		t.Errorf("mid score = %+v", mid)
	}
	if cheap := scoreOf(t, d, "cheap"); cheap.Feasible || len(cheap.Reasons) == 0 || !strings.Contains(cheap.Reasons[0], "quality") {
		t.Errorf("cheap score = %+v, want infeasible with a quality reason", cheap)
	}

	// A request floor overrides the policy default, a tenant floor sits between them.
	if got := candidateIDs(mustDecide(t, r, withFloor(0.3))); got[0] != "cheap" {
		t.Errorf("request floor 0.3: first = %s, want cheap", got[0])
	}
	if got := candidateIDs(mustDecide(t, r, withFloor(0.8))); got[0] != "top" {
		t.Errorf("request floor 0.8: first = %s, want top", got[0])
	}
	r.tenant = &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 50, MinQuality: ptr(0.8)}
	if d := mustDecide(t, r); candidateIDs(d)[0] != "top" || d.Floor != 0.8 {
		t.Errorf("tenant min_quality 0.8: first %s floor %v, want top and 0.8", candidateIDs(d)[0], d.Floor)
	}
	if d := mustDecide(t, r, withFloor(0.3)); candidateIDs(d)[0] != "cheap" || d.Floor != 0.3 {
		t.Errorf("request floor beats the tenant floor: first %s floor %v, want cheap and 0.3", candidateIDs(d)[0], d.Floor)
	}

	// With on_infeasible reject the unsafe models are no failover candidates.
	r.tenant = &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 50}
	r.cfg.OnInfeasible = config.OnInfeasibleReject
	if got := candidateIDs(mustDecide(t, r)); !slices.Equal(got, []string{"mid", "top"}) {
		t.Errorf("reject mode candidates = %v, want [mid top]", got)
	}
}

func TestMinCostTiesBreakOnLThenQThenID(t *testing.T) {
	equal := []optModel{
		{"a", "1", "2", 1, nil}, {"b", "1", "2", 1, nil}, {"c", "1", "2", 1, nil}, {"d", "1", "2", 1, nil},
	}
	build := func(l map[string]float64) *optRig {
		scores := map[string]policy.ModelScore{}
		for id, v := range l {
			scores[id] = modelScore(0.5, v)
		}
		cat := optCatalogOf(t, equal)
		return &optRig{
			t: t, cat: cat, router: New(cat, nil), scorer: &fakeScorer{version: "v", scores: scoresOf(scores)},
			cfg:    defaultPolicyConfig(),
			tenant: &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 50},
		}
	}

	// Same cost: the higher lower bound wins.
	r := build(map[string]float64{"a": 0.6, "b": 0.8, "c": 0.7, "d": 0.65})
	if got := candidateIDs(mustDecide(t, r))[0]; got != "b" {
		t.Errorf("higher L: first = %s, want b", got)
	}
	// Same cost and L: the lower p95 (the quantile of the SLO) wins.
	r = build(map[string]float64{"a": 0.5, "b": 0.8, "c": 0.8, "d": 0.8})
	r.est = priorEstimator(map[string][2]float64{"a": {100, 1}, "b": {500, 1}, "c": {300, 1}, "d": {400, 1}})
	d := mustDecide(t, r, withTTFTSLO(10_000))
	if got := candidateIDs(d)[0]; got != "c" {
		t.Errorf("same L, lower QMs: first = %s, want c (ttft p95 300)", got)
	}
	if q := scoreOf(t, d, "c").QMs; q == nil || *q != 300 {
		t.Errorf("c QMs = %v, want 300", q)
	}
	// Same cost, L and QMs: the smaller ID wins.
	r = build(map[string]float64{"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.8})
	if got := candidateIDs(mustDecide(t, r)); !slices.Equal(got, []string{"a", "b", "c", "d"}) {
		t.Errorf("full tie: %v, want ID order", got)
	}
	r.est = priorEstimator(map[string][2]float64{"a": {100, 1}, "b": {100, 1}, "c": {100, 1}, "d": {100, 1}})
	if got := candidateIDs(mustDecide(t, r, withTTFTSLO(10_000)))[0]; got != "a" {
		t.Errorf("tie on QMs too: first = %s, want a", got)
	}
}

func TestDualMaximizesPMinusLambdaCost(t *testing.T) {
	scores := map[string]policy.ModelScore{
		"cheap": modelScore(0.5, 0.6), "mid": modelScore(0.7, 0.6), "top": modelScore(0.9, 0.6),
	}
	cases := []struct {
		lambda float64
		want   string
		why    string
	}{
		{0, "top", "no cost penalty: the highest P"},
		{100, "top", "0.9-0.14=0.76 beats 0.7-0.028=0.672 and 0.5-0.014=0.486"},
		{1000, "mid", "0.7-0.28=0.42 beats 0.5-0.14=0.36 and 0.9-1.4=-0.5"},
		{10_000, "cheap", "0.5-1.4=-0.9 beats 0.7-2.8 and 0.9-14"},
	}
	for _, tc := range cases {
		cfg := defaultPolicyConfig()
		cfg.Mode, cfg.Lambda = policy.ModeDual, tc.lambda
		r := newOptRig(t, scores, cfg)
		d := mustDecide(t, r)
		if got := candidateIDs(d)[0]; got != tc.want {
			t.Errorf("lambda %v: first = %s, want %s (%s)", tc.lambda, got, tc.want, tc.why)
		}
		// The failover order follows the same objective.
		if len(d.Candidates) != 3 {
			t.Errorf("lambda %v: %d candidates, want 3", tc.lambda, len(d.Candidates))
		}
	}

	// The floor still filters in dual mode.
	cfg := defaultPolicyConfig()
	cfg.Mode, cfg.Lambda, cfg.DefaultFloor = policy.ModeDual, 10_000, 0.7
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), cfg)
	if got := candidateIDs(mustDecide(t, r))[0]; got != "top" {
		t.Errorf("dual with floor 0.7: first = %s, want top (the only model above the floor)", got)
	}
}

func TestBudgetIsHard(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.DefaultFloor = 0.5
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), cfg)

	// Nothing fits: 429, never relaxed, whatever on_infeasible says. With no explicit
	// max_tokens a model is admitted at the output budget it can afford (Task 20), so the
	// budget must be below the input cost plus one output token of the cheapest model
	// (100,000,000 + 2,000,000); 199,999,999 now buys cheap 49 tokens.
	for _, onInf := range []string{config.OnInfeasibleBestEffort, config.OnInfeasibleReject} {
		r.cfg.OnInfeasible = onInf
		d, err := r.decide(withBudget(101_999_999))
		if err == nil || err.Status != 429 || err.Code != "insufficient_quota" || err.Type != "insufficient_quota" {
			t.Fatalf("%s: err = %+v, want 429 insufficient_quota", onInf, err)
		}
		if err.Message != "budget remaining below the cheapest worst-case cost" {
			t.Errorf("message = %q", err.Message)
		}
		if len(d.Candidates) != 0 {
			t.Errorf("%s: candidates %v with an unaffordable budget", onInf, candidateIDs(d))
		}
	}

	// Exactly the cheapest worst case fits (<=), the pricier models never do.
	r.cfg.OnInfeasible = config.OnInfeasibleBestEffort
	d := mustDecide(t, r, withBudget(200_000_000))
	if got := candidateIDs(d); !slices.Equal(got, []string{"cheap"}) {
		t.Errorf("budget = cheap's cost_max: candidates %v, want [cheap]", got)
	}
	if !slices.Equal(d.Relaxed, []string{"quality"}) {
		t.Errorf("relaxed = %v, want [quality]: the floor yields before the budget does", d.Relaxed)
	}
	if s := scoreOf(t, d, "top"); s.Feasible || !strings.Contains(strings.Join(s.Reasons, " "), "budget") {
		t.Errorf("top score = %+v, want infeasible with a budget reason", s)
	}

	// With reject the floor is enforced and the unmet floor is a 422, not a 429.
	r.cfg.OnInfeasible = config.OnInfeasibleReject
	if _, err := r.decide(withBudget(200_000_000)); err == nil || err.Status != 422 {
		t.Errorf("reject within a budget that only fits an unsafe model: err = %+v, want 422", err)
	}

	// Budget for two models: the failover list never leaves it.
	r.cfg.OnInfeasible = config.OnInfeasibleBestEffort
	d = mustDecide(t, r, withBudget(400_000_000))
	if got := candidateIDs(d); !slices.Equal(got, []string{"mid", "cheap"}) {
		t.Errorf("budget fits cheap and mid: candidates %v, want [mid cheap]", got)
	}
	for _, c := range d.Candidates {
		if c.CostMax > 400_000_000 {
			t.Errorf("candidate %s has cost_max %d above the budget", c.Model.ID, c.CostMax)
		}
	}
	// A nil budget is unlimited.
	if got := candidateIDs(mustDecide(t, r)); len(got) != 3 {
		t.Errorf("no budget: candidates %v, want all three", got)
	}
}

func TestRejectListsBindingConstraints(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.OnInfeasible = config.OnInfeasibleReject
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.81), cfg)

	d, err := r.decide(withFloor(0.9))
	if err == nil || err.Status != 422 || err.Code != "no_feasible_model" || err.Type != "invalid_request_error" {
		t.Fatalf("err = %+v, want 422 no_feasible_model", err)
	}
	if want := "no feasible model: quality floor 0.90 above best lower bound 0.81"; err.Message != want {
		t.Errorf("message = %q, want %q", err.Message, want)
	}
	if len(d.Candidates) != 0 {
		t.Errorf("candidates %v on a rejection", candidateIDs(d))
	}

	// A latency SLO that every model misses, estimated from the simulated prior:
	// p95 = ttft 400 + 40 output tokens * 50 ms = 2400 ms.
	r.est = priorEstimator(map[string][2]float64{"cheap": {400, 50}, "mid": {400, 50}, "top": {400, 50}})
	_, err = r.decide(withFloor(0.3), withE2ESLO(2000))
	want := "no feasible model: latency slo 2000 ms below best p95 2400 ms (simulated profile)"
	if err == nil || err.Message != want {
		t.Errorf("slo message = %v, want %q", err, want)
	}

	// Both constraints bind.
	_, err = r.decide(withFloor(0.9), withE2ESLO(2000))
	want = "no feasible model: quality floor 0.90 above best lower bound 0.81; latency slo 2000 ms below best p95 2400 ms (simulated profile)"
	if err == nil || err.Message != want {
		t.Errorf("both message = %v, want %q", err, want)
	}

	// Estimates from observations carry no "simulated" suffix.
	observed := latency.NewEstimator(nil, func() time.Time { return time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC) })
	for _, m := range optModels {
		for _, dep := range []string{m.id + "-a", m.id + "-b"} {
			for i := 0; i < 60; i++ {
				observed.ObserveTTFT(dep, 3*time.Second)
				observed.ObserveTPOT(dep, 100*time.Millisecond)
			}
		}
	}
	r.est = observed
	_, err = r.decide(withFloor(0.3), withE2ESLO(2000))
	if err == nil || !strings.Contains(err.Message, "latency slo 2000 ms below best p95") || strings.Contains(err.Message, "simulated") {
		t.Errorf("observed estimate message = %v, want the slo sentence without a simulated suffix", err)
	}

	// No estimate at all.
	r.est = nil
	_, err = r.decide(withFloor(0.3), withE2ESLO(2000))
	if err == nil || !strings.Contains(err.Message, "latency slo 2000 ms") || !strings.Contains(err.Message, "no latency estimate") {
		t.Errorf("no-estimate message = %v, want one saying there is no latency estimate", err)
	}

	// A time-to-first-token SLO is named as such.
	r.est = priorEstimator(map[string][2]float64{"cheap": {400, 50}, "mid": {400, 50}, "top": {400, 50}})
	_, err = r.decide(withFloor(0.3), withTTFTSLO(300))
	if err == nil || err.Message != "no feasible model: ttft slo 300 ms below best p95 400 ms (simulated profile)" {
		t.Errorf("ttft slo message = %v", err)
	}

	// Each constraint is met by a different model, but no model meets both.
	r.est = priorEstimator(map[string][2]float64{"cheap": {200, 20}, "mid": {400, 50}, "top": {400, 50}})
	r.scorer.scores = scoresOf(tieredScores(0.4, 0.6, 0.3))
	_, err = r.decide(withFloor(0.5), withE2ESLO(2000))
	if err == nil || err.Message != "no feasible model: no model meets quality floor 0.50 and latency slo 2000 ms together" {
		t.Errorf("joint message = %v", err)
	}

	// The tenant setting beats the policy's, in both directions.
	r = newOptRig(t, tieredScores(0.4, 0.6, 0.81), defaultPolicyConfig()) // policy: best_effort
	r.tenant.OnInfeasible = config.OnInfeasibleReject
	if _, err := r.decide(withFloor(0.9)); err == nil || err.Status != 422 {
		t.Errorf("tenant reject over policy best_effort: err = %+v, want 422", err)
	}
	r = newOptRig(t, tieredScores(0.4, 0.6, 0.81), cfg) // policy: reject
	r.tenant.OnInfeasible = config.OnInfeasibleBestEffort
	if d := mustDecide(t, r, withFloor(0.9)); candidateIDs(d)[0] != "top" {
		t.Errorf("tenant best_effort over policy reject: first = %s, want top", candidateIDs(d)[0])
	}
}

func TestBestEffortRelaxesLatencyThenQuality(t *testing.T) {
	cfg := defaultPolicyConfig()
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), cfg)
	r.est = priorEstimator(map[string][2]float64{"cheap": {400, 50}, "mid": {400, 50}, "top": {400, 50}}) // p95 2400

	// The SLO is out of reach, the floor is not: latency yields, the objective still picks.
	d := mustDecide(t, r, withFloor(0.5), withE2ESLO(2000))
	if got := candidateIDs(d); !slices.Equal(got, []string{"mid", "top", "cheap"}) {
		t.Errorf("latency relaxed: candidates %v, want [mid top cheap]", got)
	}
	if !slices.Equal(d.Relaxed, []string{"latency"}) {
		t.Errorf("relaxed = %v, want [latency]", d.Relaxed)
	}
	if d.Explored || d.Propensity != 1 {
		t.Errorf("a relaxed decision explored %v propensity %v, want a greedy one", d.Explored, d.Propensity)
	}

	// Dual mode picks by its objective among the models that meet the floor.
	r.cfg.Mode, r.cfg.Lambda = policy.ModeDual, 0
	if d := mustDecide(t, r, withFloor(0.5), withE2ESLO(2000)); candidateIDs(d)[0] != "top" || !slices.Equal(d.Relaxed, []string{"latency"}) {
		t.Errorf("dual latency relaxed: first %s relaxed %v, want top and [latency]", candidateIDs(d)[0], d.Relaxed)
	}
	r.cfg.Mode = policy.ModeMinCost

	// Neither is reachable: both yield and the highest lower bound wins.
	d = mustDecide(t, r, withFloor(0.95), withE2ESLO(2000))
	if got := candidateIDs(d); !slices.Equal(got, []string{"top", "mid", "cheap"}) {
		t.Errorf("both relaxed: candidates %v, want [top mid cheap] (by lower bound)", got)
	}
	if !slices.Equal(d.Relaxed, []string{"latency", "quality"}) {
		t.Errorf("relaxed = %v, want [latency quality]", d.Relaxed)
	}

	// Without an SLO latency is never reported.
	d = mustDecide(t, r, withFloor(0.95))
	if !slices.Equal(d.Relaxed, []string{"quality"}) || candidateIDs(d)[0] != "top" {
		t.Errorf("no slo, floor out of reach: relaxed %v first %s, want [quality] and top", d.Relaxed, candidateIDs(d)[0])
	}
	// Nothing relaxed when everything is reachable.
	d = mustDecide(t, r, withFloor(0.5), withE2ESLO(3000))
	if len(d.Relaxed) != 0 || candidateIDs(d)[0] != "mid" {
		t.Errorf("reachable constraints: relaxed %v first %s, want none and mid", d.Relaxed, candidateIDs(d)[0])
	}
	// A model with no latency estimate fails an SLO, so latency relaxes.
	r.est = nil
	d = mustDecide(t, r, withFloor(0.5), withE2ESLO(3000))
	if !slices.Equal(d.Relaxed, []string{"latency"}) {
		t.Errorf("no estimate: relaxed %v, want [latency]", d.Relaxed)
	}
	if s := scoreOf(t, d, "mid"); s.QMs != nil || s.Feasible {
		t.Errorf("mid score = %+v, want no QMs and infeasible", s)
	}
}

func TestSLOChecksBothMetricsWhenBothAreSet(t *testing.T) {
	cfg := defaultPolicyConfig()
	r := newOptRig(t, tieredScores(0.9, 0.9, 0.9), cfg)
	// cheap: ttft 100, e2e 100+40*10=500. mid: ttft 900, e2e 900+40*10=1300. top: ttft 100, e2e 100+40*100=4100.
	r.est = priorEstimator(map[string][2]float64{"cheap": {100, 10}, "mid": {900, 10}, "top": {100, 100}})

	d := mustDecide(t, r, withTTFTSLO(500), withE2ESLO(2000))
	if got := candidateIDs(d)[0]; got != "cheap" {
		t.Fatalf("first = %s, want cheap (the only model inside both SLOs)", got)
	}
	for _, id := range []string{"mid", "top"} {
		if s := scoreOf(t, d, id); s.Feasible {
			t.Errorf("%s is feasible with both SLOs set: %+v", id, s)
		}
	}
	// QMs reports the end-to-end quantile when that SLO is set.
	if q := scoreOf(t, d, "cheap").QMs; q == nil || *q != 500 {
		t.Errorf("cheap QMs = %v, want the e2e 500", q)
	}
}

func TestOODConservativeUsesPooled(t *testing.T) {
	scores := map[string]policy.ModelScore{
		"cheap": {P: 0.9, L: 0.8, PooledRate: 0.3, PooledLCB: 0.2, EOut: 20, InRatio: 1},
		"mid":   {P: 0.9, L: 0.8, PooledRate: 0.5, PooledLCB: 0.45, EOut: 20, InRatio: 1},
		"top":   {P: 0.95, L: 0.9, PooledRate: 0.6, PooledLCB: 0.5, EOut: 20, InRatio: 1},
	}
	cfg := defaultPolicyConfig()
	cfg.DefaultFloor = 0.4
	r := newOptRig(t, scores, cfg)

	// In distribution the cheap model's own bound clears the floor.
	if d := mustDecide(t, r); candidateIDs(d)[0] != "cheap" || d.OOD {
		t.Fatalf("in distribution: first %s ood %v, want cheap and false", candidateIDs(d)[0], d.OOD)
	}

	// Out of distribution the pooled bounds replace P and L: cheap drops under the floor.
	r.scorer.scores.OOD, r.scorer.scores.OODScore = true, 0.97
	d := mustDecide(t, r)
	if got := candidateIDs(d); !slices.Equal(got, []string{"mid", "top", "cheap"}) {
		t.Errorf("ood conservative: candidates %v, want [mid top cheap]", got)
	}
	if !d.OOD || d.OODScore != 0.97 {
		t.Errorf("ood %v score %v, want true and 0.97", d.OOD, d.OODScore)
	}
	if s := scoreOf(t, d, "mid"); s.P != 0.5 || s.L != 0.45 {
		t.Errorf("mid score P %v L %v, want the pooled 0.5 and 0.45", s.P, s.L)
	}
	// Exploration is allowed under the conservative action: it stays inside the safe set.
	r.cfg.Epsilon = 0.5
	d = mustDecide(t, r, withRand(func() float64 { return 0 }))
	if !d.Explored {
		t.Error("conservative OOD request with a draw below epsilon: want an exploration")
	}
}

func TestOODStrongestPicksHighestRank(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.OODAction = policy.OODStrongest
	cfg.DefaultFloor = 0.4
	cfg.Epsilon = 0.5
	r := newOptRig(t, tieredScores(0.8, 0.7, 0.5), cfg)

	// In distribution the cheapest safe model wins.
	if got := candidateIDs(mustDecide(t, r))[0]; got != "cheap" {
		t.Fatalf("in distribution: first = %s, want cheap", got)
	}

	r.scorer.scores.OOD, r.scorer.scores.OODScore = true, 0.99
	calls := 0
	draw := func() float64 { calls++; return 0 }
	d := mustDecide(t, r, withRand(draw))
	if got := candidateIDs(d); got[0] != "top" {
		t.Errorf("ood strongest: first = %s, want top (highest rank in the safe set)", got[0])
	}
	if !d.OOD || d.Explored || d.Propensity != 1 || d.Epsilon != 0 || calls != 0 {
		t.Errorf("ood strongest: ood %v explored %v propensity %v epsilon %v draws %d; want no exploration at all",
			d.OOD, d.Explored, d.Propensity, d.Epsilon, calls)
	}

	// Floor out of reach: the highest rank in the budget set, with the floor relaxed.
	d = mustDecide(t, r, withFloor(0.95))
	if candidateIDs(d)[0] != "top" || !slices.Equal(d.Relaxed, []string{"quality"}) {
		t.Errorf("floor out of reach: first %s relaxed %v, want top and [quality]", candidateIDs(d)[0], d.Relaxed)
	}
	// The budget is still hard: the strongest model that fits.
	d = mustDecide(t, r, withBudget(400_000_000))
	if got := candidateIDs(d); got[0] != "mid" {
		t.Errorf("budget fits cheap and mid: first = %s, want mid", got[0])
	}
}

func TestExplorationStaysInSafeSetWithPropensity(t *testing.T) {
	const eps, draws = 0.2, 10_000
	cfg := defaultPolicyConfig()
	cfg.DefaultFloor, cfg.Epsilon = 0.5, eps
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), cfg) // safe set {mid, top}, greedy mid
	rng := rand.New(rand.NewPCG(1, 2))

	picks := map[string]int{}
	explored := 0
	for i := 0; i < draws; i++ {
		d := mustDecide(t, r, withRand(rng.Float64))
		first := candidateIDs(d)[0]
		picks[first]++
		if d.Explored {
			explored++
		}
		var want float64
		switch first {
		case "mid": // the greedy choice: (1-eps) + eps/|S|
			want = (1 - eps) + eps/2
		case "top":
			want = eps / 2
		default:
			t.Fatalf("draw %d picked %s, outside the safe set {mid, top}", i, first)
		}
		if math.Abs(d.Propensity-want) > 1e-12 || d.Epsilon != eps {
			t.Fatalf("draw %d: pick %s propensity %v epsilon %v, want %v and %v", i, first, d.Propensity, d.Epsilon, want, eps)
		}
		if len(d.Relaxed) != 0 {
			t.Fatalf("draw %d relaxed %v", i, d.Relaxed)
		}
		// Whatever was picked, the failover list still starts with it and holds the rest of the safe set.
		if len(d.Candidates) != 3 || !slices.Contains(candidateIDs(d), "mid") || !slices.Contains(candidateIDs(d), "top") {
			t.Fatalf("draw %d candidates %v", i, candidateIDs(d))
		}
	}
	if picks["cheap"] != 0 {
		t.Errorf("the unsafe model was picked %d times", picks["cheap"])
	}
	if frac := float64(explored) / draws; math.Abs(frac-eps) > 0.02 {
		t.Errorf("explored fraction %.4f, want about %v", frac, eps)
	}
	// mid is chosen greedily 80% of the time plus half the explorations.
	if frac := float64(picks["mid"]) / draws; math.Abs(frac-0.9) > 0.02 {
		t.Errorf("mid picked %.4f of the time, want about 0.9", frac)
	}
	if picks["top"] == 0 {
		t.Error("the other safe model was never explored")
	}

	// One safe model: every draw gives it with propensity 1.
	r.scorer.scores.Models = tieredScores(0.4, 0.6, 0.9)
	d := mustDecide(t, r, withFloor(0.8), withRand(func() float64 { return 0 }))
	if candidateIDs(d)[0] != "top" || math.Abs(d.Propensity-1) > 1e-12 || !d.Explored {
		t.Errorf("single safe model: first %s propensity %v explored %v, want top, 1 and true", candidateIDs(d)[0], d.Propensity, d.Explored)
	}
}

func TestExplorationIsOffWhenItShouldBe(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.DefaultFloor, cfg.Epsilon = 0.5, 0.2
	scores := tieredScores(0.4, 0.6, 0.9)
	always := func() float64 { return 0 }

	t.Run("nil Rand", func(t *testing.T) {
		d := mustDecide(t, newOptRig(t, scores, cfg))
		if d.Explored || d.Propensity != 1 || d.Epsilon != 0 || candidateIDs(d)[0] != "mid" {
			t.Errorf("explored %v propensity %v epsilon %v first %s, want a plain greedy decision", d.Explored, d.Propensity, d.Epsilon, candidateIDs(d)[0])
		}
	})
	t.Run("epsilon zero", func(t *testing.T) {
		c := cfg
		c.Epsilon = 0
		calls := 0
		d := mustDecide(t, newOptRig(t, scores, c), withRand(func() float64 { calls++; return 0 }))
		if d.Explored || d.Propensity != 1 || calls != 0 {
			t.Errorf("explored %v propensity %v draws %d, want greedy with no draw", d.Explored, d.Propensity, calls)
		}
	})
	t.Run("draw above epsilon is greedy with the greedy propensity", func(t *testing.T) {
		d := mustDecide(t, newOptRig(t, scores, cfg), withRand(func() float64 { return 0.5 }))
		if d.Explored || candidateIDs(d)[0] != "mid" || math.Abs(d.Propensity-0.9) > 1e-12 {
			t.Errorf("explored %v first %s propensity %v, want greedy mid with 0.9", d.Explored, candidateIDs(d)[0], d.Propensity)
		}
	})
	t.Run("relaxed", func(t *testing.T) {
		d := mustDecide(t, newOptRig(t, scores, cfg), withFloor(0.95), withRand(always))
		if d.Explored || d.Propensity != 1 || d.Epsilon != 0 || len(d.Relaxed) == 0 {
			t.Errorf("explored %v propensity %v epsilon %v relaxed %v, want no exploration on a relaxed decision", d.Explored, d.Propensity, d.Epsilon, d.Relaxed)
		}
	})
}

func TestUnavailableModelsExcludedWithReason(t *testing.T) {
	cfg := defaultPolicyConfig()
	r := newOptRig(t, tieredScores(0.9, 0.9, 0.9), cfg)

	// Both deployments of mid are down (a breaker is open).
	down := func(in *Input) {
		in.Available = func(dep string) bool { return !strings.HasPrefix(dep, "mid-") }
	}
	d := mustDecide(t, r, down)
	if got := candidateIDs(d); !slices.Equal(got, []string{"cheap", "top"}) {
		t.Errorf("candidates %v, want [cheap top]", got)
	}
	mid := scoreOf(t, d, "mid")
	if mid.Feasible || len(mid.Reasons) != 1 || !strings.Contains(mid.Reasons[0], "no available deployment") {
		t.Errorf("mid score = %+v, want infeasible because it has no available deployment", mid)
	}
	if got := []string{d.Scores[0].Model, d.Scores[1].Model, d.Scores[2].Model}; !slices.Equal(got, []string{"cheap", "mid", "top"}) {
		t.Errorf("scores are in catalog order, got %v", got)
	}

	// One deployment down is not an exclusion.
	one := func(in *Input) { in.Available = func(dep string) bool { return dep != "mid-a" } }
	if s := scoreOf(t, mustDecide(t, r, one), "mid"); !s.Feasible {
		t.Errorf("mid with one live deployment: %+v, want feasible", s)
	}

	// A model that fails the capability check appears with its reasons.
	d = mustDecide(t, r, func(in *Input) { in.Req.Caps = capSetOf(catalog.CapTools) })
	if got := candidateIDs(d); !slices.Equal(got, []string{"top"}) {
		t.Errorf("tools request: candidates %v, want [top]", got)
	}
	cheap := scoreOf(t, d, "cheap")
	if cheap.Feasible || len(cheap.Reasons) == 0 || !strings.Contains(cheap.Reasons[0], "missing capability tools") {
		t.Errorf("cheap score = %+v, want infeasible with the missing capability", cheap)
	}
	if len(d.Rejected) != 2 {
		t.Errorf("Rejected = %v, want the two models without tools", d.Rejected)
	}

	// A model the artifact does not cover appears with that reason.
	delete(r.scorer.scores.Models, "mid")
	d = mustDecide(t, r)
	if got := candidateIDs(d); !slices.Equal(got, []string{"cheap", "top"}) {
		t.Errorf("candidates %v, want [cheap top]", got)
	}
	if s := scoreOf(t, d, "mid"); s.Feasible || len(s.Reasons) != 1 || !strings.Contains(s.Reasons[0], "not in the policy artifact") {
		t.Errorf("mid score = %+v, want infeasible because it is not in the artifact", s)
	}

	// A model the tenant may not use is not scored at all.
	r.tenant = &config.TenantConfig{ID: "t", AllowedModels: []string{"cheap", "top"}, DefaultMaxTokens: 50}
	for _, s := range mustDecide(t, r).Scores {
		if s.Model == "mid" {
			t.Errorf("mid is outside the tenant boundary but appears in the scores")
		}
	}

	// Everything down is the usual 503.
	r = newOptRig(t, tieredScores(0.9, 0.9, 0.9), cfg)
	d, err := r.decide(func(in *Input) { in.Available = func(string) bool { return false } })
	if err == nil || err.Code != "no_healthy_deployment" {
		t.Errorf("all deployments down: err = %+v, want no_healthy_deployment", err)
	}
	if len(d.Scores) != 3 {
		t.Errorf("all down: %d scores, want 3 with their reasons", len(d.Scores))
	}
	// No model covered by the artifact is a capability error.
	r.scorer.scores.Models = map[string]policy.ModelScore{"stranger": modelScore(0.5, 0.5)}
	if _, err := r.decide(); err == nil || err.Code != "model_capability_unavailable" {
		t.Errorf("no covered model: err = %+v, want model_capability_unavailable", err)
	}
	// Every model failing the check keeps the existing capability error.
	r = newOptRig(t, tieredScores(0.9, 0.9, 0.9), cfg)
	_, err = r.decide(func(in *Input) { in.Req.Caps = capSetOf(catalog.CapVision) })
	if err == nil || err.Code != "model_capability_unavailable" || !strings.Contains(err.Message, "missing capability vision") {
		t.Errorf("no model has vision: err = %+v", err)
	}
}

func TestCostExpArithmetic(t *testing.T) {
	cfg := defaultPolicyConfig()
	scores := map[string]policy.ModelScore{
		"cheap": {P: 0.5, L: 0.5, EOut: 0.2, InRatio: 0.5, OutQ: [4]float64{1, 1, 1, 1}}, // in 50 out 1
		"mid":   {P: 0.5, L: 0.5, EOut: 7.3, InRatio: 1, OutQ: [4]float64{1, 1, 1, 1}},   // in 100 out ceil(7.3)=8
		"top":   {P: 0.5, L: 0.5, EOut: 5000, InRatio: 0, OutQ: [4]float64{1, 1, 1, 1}},  // out clamps to k=50, bad ratio counts as 1
	}
	r := newOptRig(t, scores, cfg)
	d := mustDecide(t, r)
	want := map[string]money.PicoUSD{
		"cheap": 50*1_000_000 + 1*2_000_000,
		"mid":   100*2_000_000 + 8*4_000_000,
		"top":   100*10_000_000 + 50*20_000_000,
	}
	for id, w := range want {
		if got := scoreOf(t, d, id).CostExp; got != w {
			t.Errorf("%s cost_exp = %d, want %d", id, got, w)
		}
	}

	// The model's own overhead enters the input bound.
	cat := optCatalogOf(t, optModels)
	m, _ := cat.Model("mid")
	m.PromptOverhead = 20
	r.cat, r.router = cat, New(cat, nil)
	r.scorer.scores.Models["mid"] = modelScore(0.5, 0.5)
	if got := scoreOf(t, mustDecide(t, r), "mid").CostExp; got != (100+20)*2_000_000+20*4_000_000 {
		t.Errorf("mid with overhead 20: cost_exp = %d, want %d", got, (100+20)*2_000_000+20*4_000_000)
	}
}

func TestCostExpSaturates(t *testing.T) {
	cat := optCatalogOf(t, []optModel{{"huge", "9000000", "9000000", 1, nil}})
	scores := map[string]policy.ModelScore{"huge": modelScore(0.5, 0.5)}
	r := &optRig{
		t: t, cat: cat, router: New(cat, nil), scorer: &fakeScorer{version: "v", scores: scoresOf(scores)},
		cfg:    defaultPolicyConfig(),
		tenant: &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 50},
	}
	m, _ := cat.Model("huge")
	m.PriceIn, m.PriceOut = money.PicoUSD(math.MaxInt64/2), money.PicoUSD(math.MaxInt64/2)
	d := mustDecide(t, r)
	if s := scoreOf(t, d, "huge"); s.CostExp != money.PicoUSD(math.MaxInt64) || s.CostMax != money.PicoUSD(math.MaxInt64) {
		t.Errorf("score = %+v, want saturated costs", s)
	}
}

func TestScorerIsCalledOncePerRequestWithTheMessages(t *testing.T) {
	r := newOptRig(t, tieredScores(0.9, 0.9, 0.9), defaultPolicyConfig())
	msgs := []features.Message{{Role: "user", Text: "hello"}}
	mustDecide(t, r, func(in *Input) { in.Messages = msgs })
	if r.scorer.calls != 1 {
		t.Errorf("Score called %d times for three models, want once", r.scorer.calls)
	}
	if !slices.Equal(r.scorer.msgs, msgs) {
		t.Errorf("Score got %v, want the input messages", r.scorer.msgs)
	}
}

func TestOptimizerVersionAndName(t *testing.T) {
	cfg := defaultPolicyConfig()
	o := NewOptimizer(&fakeScorer{version: "abc123"}, cfg, nil)
	if o.Name() != "learned" || o.Version() != "abc123+"+cfg.Hash8() {
		t.Errorf("name %q version %q", o.Name(), o.Version())
	}
	cfg.Lambda = 3
	if NewOptimizer(&fakeScorer{version: "abc123"}, cfg, nil).Version() == o.Version() {
		t.Error("a different policy config gives the same version")
	}
}

// TestLearnedDoesNotNeedAStaticPolicy: the learned path must work on a router built
// without a static policy, and must not change what the static policies do.
func TestLearnedDoesNotNeedAStaticPolicy(t *testing.T) {
	r := newOptRig(t, tieredScores(0.9, 0.9, 0.9), defaultPolicyConfig())
	if r.router.HasAuto() {
		t.Fatal("test setup: the router should have no static policy")
	}
	if _, err := r.decide(); err != nil {
		t.Fatalf("learned decision on a router with no static policy: %v", err)
	}
	in := r.input(func(in *Input) { in.Learned = nil })
	if _, err := r.router.Decide(in); err == nil {
		t.Error("auto without a learned policy and without a static one should still fail")
	}
}

// floorOnScores gives each model a point estimate well above its lower bound, so the two
// ways of reading the quality floor disagree:
//
//	cheap P 0.55 L 0.30   mid P 0.70 L 0.45   top P 0.90 L 0.80
func floorOnScores() map[string]policy.ModelScore {
	return map[string]policy.ModelScore{
		"cheap": modelScore(0.55, 0.30),
		"mid":   modelScore(0.70, 0.45),
		"top":   modelScore(0.90, 0.80),
	}
}

func withOptimizer(o *Optimizer) func(*Input) { return func(in *Input) { in.Learned = o } }

// TestFloorOnPointUsesP: with the point arm the floor is compared with P, the calibrated
// success probability, instead of the lower confidence bound L. Nothing else changes: the
// default arm still reads L, the expected costs and the failover order are the same.
func TestFloorOnPointUsesP(t *testing.T) {
	r := newOptRig(t, floorOnScores(), defaultPolicyConfig())
	lcb := NewOptimizer(r.scorer, r.cfg, nil)
	point := lcb.WithFloorOnPoint()

	// Floor 0.5: only top has L >= 0.5, but every model has P >= 0.5, so min_cost takes cheap.
	d := mustDecide(t, r, withFloor(0.5), withOptimizer(lcb))
	if got := candidateIDs(d); got[0] != "top" || len(d.Relaxed) != 0 {
		t.Errorf("lcb arm, floor 0.5: first = %s relaxed %v, want top and nothing relaxed", got[0], d.Relaxed)
	}
	d = mustDecide(t, r, withFloor(0.5), withOptimizer(point))
	if got := candidateIDs(d); !slices.Equal(got, []string{"cheap", "mid", "top"}) || len(d.Relaxed) != 0 {
		t.Errorf("point arm, floor 0.5: candidates %v relaxed %v, want [cheap mid top] and nothing relaxed", got, d.Relaxed)
	}
	for _, id := range []string{"cheap", "mid", "top"} {
		if s := scoreOf(t, d, id); !s.Feasible || len(s.Reasons) != 0 {
			t.Errorf("point arm, floor 0.5: %s = %+v, want feasible with no reasons", id, s)
		}
	}
	// The scores themselves are the same in both arms.
	cheap := scoreOf(t, d, "cheap")
	if cheap.P != 0.55 || cheap.L != 0.30 || cheap.CostExp != 140_000_000 {
		t.Errorf("point arm cheap score = %+v, want P 0.55 L 0.30 cost_exp 140000000", cheap)
	}

	// Floor 0.6: P rules cheap out (0.55) and keeps mid (0.70); the reason names P.
	d = mustDecide(t, r, withFloor(0.6), withOptimizer(point))
	if got := candidateIDs(d); !slices.Equal(got, []string{"mid", "top", "cheap"}) {
		t.Errorf("point arm, floor 0.6: candidates %v, want [mid top cheap]", got)
	}
	if s := scoreOf(t, d, "cheap"); s.Feasible || len(s.Reasons) != 1 || !strings.Contains(s.Reasons[0], "point estimate 0.5500 below the floor 0.6000") {
		t.Errorf("point arm, floor 0.6: cheap = %+v, want a reason naming the point estimate", s)
	}
	if s := scoreOf(t, mustDecide(t, r, withFloor(0.6), withOptimizer(lcb)), "cheap"); len(s.Reasons) != 1 || !strings.Contains(s.Reasons[0], "lower bound 0.3000") {
		t.Errorf("lcb arm, floor 0.6: cheap reasons = %v, want a reason naming the lower bound", s.Reasons)
	}

	// Floor 0.95 is above every P too: the point arm relaxes quality like the lcb arm does.
	d = mustDecide(t, r, withFloor(0.95), withOptimizer(point))
	if !slices.Equal(d.Relaxed, []string{"quality"}) {
		t.Errorf("point arm, floor 0.95: relaxed = %v, want [quality]", d.Relaxed)
	}

	// Dual mode reads the floor the same way.
	dual := r.cfg
	dual.Mode = policy.ModeDual
	dual.Lambda = 1e4 // a dollar-scale price of cost: the 140 and 280 million picodollar models win
	pointDual := NewOptimizer(r.scorer, dual, nil).WithFloorOnPoint()
	if got := candidateIDs(mustDecide(t, r, withFloor(0.5), withOptimizer(pointDual)))[0]; got != "cheap" {
		t.Errorf("point arm dual, floor 0.5: first = %s, want cheap (P 0.55 clears the floor)", got)
	}
	if got := candidateIDs(mustDecide(t, r, withFloor(0.5), withOptimizer(NewOptimizer(r.scorer, dual, nil))))[0]; got != "top" {
		t.Errorf("lcb arm dual, floor 0.5: first = %s, want top (only L 0.80 clears the floor)", got)
	}
}

// TestFloorOnPointUsesThePooledRateOutOfDistribution: an out-of-distribution request under
// the conservative action is scored with the pooled rate and pooled bound, and the point arm
// compares the floor with the pooled rate, the P the policy actually used.
func TestFloorOnPointUsesThePooledRateOutOfDistribution(t *testing.T) {
	scores := floorOnScores() // pooled rate 0.3 and pooled bound 0.2 for every model
	r := newOptRig(t, scores, defaultPolicyConfig())
	r.scorer.scores.OOD = true
	lcb := NewOptimizer(r.scorer, r.cfg, nil)
	point := lcb.WithFloorOnPoint()

	d := mustDecide(t, r, withFloor(0.25), withOptimizer(point))
	if len(d.Relaxed) != 0 || candidateIDs(d)[0] != "cheap" {
		t.Errorf("point arm: first %s relaxed %v, want cheap and nothing relaxed (pooled rate 0.3 clears 0.25)", candidateIDs(d)[0], d.Relaxed)
	}
	if d := mustDecide(t, r, withFloor(0.25), withOptimizer(lcb)); !slices.Equal(d.Relaxed, []string{"quality"}) {
		t.Errorf("lcb arm: relaxed = %v, want [quality] (pooled bound 0.2 is below 0.25)", d.Relaxed)
	}
}

func TestFloorOnPointShowsInTheVersion(t *testing.T) {
	cfg := defaultPolicyConfig()
	lcb := NewOptimizer(&fakeScorer{version: "abc123"}, cfg, nil)
	point := lcb.WithFloorOnPoint()
	if lcb.Version() != "abc123+"+cfg.Hash8() {
		t.Errorf("default version = %q, want abc123+%s", lcb.Version(), cfg.Hash8())
	}
	if want := "abc123+" + cfg.Hash8() + "+floor-point"; point.Version() != want {
		t.Errorf("point version = %q, want %q", point.Version(), want)
	}
	if point.Name() != "learned" {
		t.Errorf("point name = %q, want learned", point.Name())
	}
	// WithFloorOnPoint returns a copy: the optimizer it was called on keeps reading L.
	if lcb.Version() == point.Version() {
		t.Error("WithFloorOnPoint changed the receiver")
	}
}

// TestFloorOnPointRejectMessageNamesPointEstimate: when nothing clears the floor and the
// tenant asks for a rejection, the message names what the floor was compared with.
func TestFloorOnPointRejectMessageNamesPointEstimate(t *testing.T) {
	cfg := defaultPolicyConfig()
	cfg.OnInfeasible = config.OnInfeasibleReject
	r := newOptRig(t, floorOnScores(), cfg)
	lcb := NewOptimizer(r.scorer, r.cfg, nil)

	_, err := r.decide(withFloor(0.95), withOptimizer(lcb))
	if err == nil || !strings.Contains(err.Message, "quality floor 0.95 above best lower bound 0.80") {
		t.Fatalf("lcb arm error = %v, want the best lower bound", err)
	}
	_, err = r.decide(withFloor(0.95), withOptimizer(lcb.WithFloorOnPoint()))
	if err == nil || !strings.Contains(err.Message, "quality floor 0.95 above best point estimate 0.90") {
		t.Fatalf("point arm error = %v, want the best point estimate", err)
	}
}
