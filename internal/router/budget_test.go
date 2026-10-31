package router

import (
	"reflect"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
)

// The fixture models (see router_test.go) cost, for a request with an input bound of 100
// tokens, in picodollars:
//
//	model    in/token  out/token  input cost  max output
//	small    100,000     400,000   10,000,000     1,000
//	bargain  200,000     500,000   20,000,000     4,000
//	mid      500,000   1,500,000   50,000,000     4,000   (mid2 and alt cost the same)
//	big    3,000,000  15,000,000  300,000,000     8,000
//
// and the tenant default of 256 output tokens.

func budgetInput(budget int64, req catalog.Requirements) Input {
	b := money.PicoUSD(budget)
	return Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto, Req: req, Budget: &b}
}

// TestStaticPolicyShrinksKToTheBudget: with a budget, every candidate is costed with the
// output budget it can afford, the order follows those costs, and a model that cannot pay
// for one output token is dropped.
func TestStaticPolicyShrinksKToTheBudget(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	const budget = 100_000_000

	d, apiErr := rt.Decide(budgetInput(budget, catalog.Requirements{InUpper: 100}))
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	// alt, mid, mid2 afford 33 tokens (99,500,000); bargain 160 and small 225 (both
	// 100,000,000). Equal costs break on rank, then ID. big cannot pay for its input.
	wantIDs := []string{"alt", "mid", "mid2", "bargain", "small"}
	if got := candidateIDs(d); !reflect.DeepEqual(got, wantIDs) {
		t.Fatalf("order = %v, want %v", got, wantIDs)
	}
	wantK := map[string]int{"alt": 33, "mid": 33, "mid2": 33, "bargain": 160, "small": 225}
	for _, c := range d.Candidates {
		if c.K != wantK[c.Model.ID] {
			t.Errorf("%s: K = %d, want %d", c.Model.ID, c.K, wantK[c.Model.ID])
		}
		if c.CostMax > budget {
			t.Errorf("%s: CostMax %d is above the budget %d", c.Model.ID, c.CostMax, budget)
		}
	}
	if len(d.Rejected) != 1 || d.Rejected[0].Model != "big" || !strings.HasPrefix(string(d.Rejected[0].Reasons[0]), "budget") {
		t.Errorf("Rejected = %+v, want big with a budget reason", d.Rejected)
	}
}

// TestNoBudgetKeepsTheEffectiveK: a nil budget changes nothing.
func TestNoBudgetKeepsTheEffectiveK(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	in := budgetInput(0, catalog.Requirements{InUpper: 100})
	in.Budget = nil
	d, apiErr := rt.Decide(in)
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if len(d.Candidates) != 6 || len(d.Rejected) != 0 {
		t.Fatalf("candidates %v rejected %v, want all six and none", candidateIDs(d), d.Rejected)
	}
	for _, c := range d.Candidates {
		if c.K != 256 {
			t.Errorf("%s: K = %d, want the tenant default 256", c.Model.ID, c.K)
		}
	}
}

// TestBudgetRejectsWhenNothingIsAffordable: the answer is 429 insufficient_quota for every
// static policy and for a named model, never a 400 or a 503.
func TestBudgetRejectsWhenNothingIsAffordable(t *testing.T) {
	rulesCfg := config.RoutingConfig{
		Policy: config.PolicyRules, DefaultModel: "small",
		Rules: []config.RuleConfig{{Name: "long", Model: "big", When: config.RuleMatch{MinPromptBytes: 1}}},
	}
	policies := map[string]config.RoutingConfig{
		"cheapest":  {Policy: config.PolicyCheapest},
		"strongest": {Policy: config.PolicyStrongest},
		"rules":     rulesCfg,
	}
	req := catalog.Requirements{InUpper: 100, PromptBytes: 10}
	// small needs its 10,000,000 input plus 400,000 for one output token.
	const cheapestOneToken = 10_400_000

	for name, cfg := range policies {
		t.Run(name, func(t *testing.T) {
			rt, _ := newRouter(t, cfg)
			d, apiErr := rt.Decide(budgetInput(cheapestOneToken-1, req))
			if apiErr == nil || apiErr.Status != 429 || apiErr.Code != "insufficient_quota" || apiErr.Type != "insufficient_quota" {
				t.Fatalf("err = %+v, want 429 insufficient_quota", apiErr)
			}
			if len(d.Candidates) != 0 {
				t.Errorf("candidates %v with an unaffordable budget", candidateIDs(d))
			}

			d, apiErr = rt.Decide(budgetInput(cheapestOneToken, req))
			if apiErr != nil {
				t.Fatalf("budget for one token of the cheapest model: %v", apiErr)
			}
			if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"small"}) {
				t.Fatalf("candidates = %v, want [small]", got)
			}
			if c := d.Candidates[0]; c.K != 1 || c.CostMax != cheapestOneToken {
				t.Errorf("small: K = %d CostMax = %d, want 1 and %d", c.K, c.CostMax, cheapestOneToken)
			}
		})
	}

	t.Run("pinned", func(t *testing.T) {
		rt, _ := newRouter(t, config.RoutingConfig{})
		in := budgetInput(cheapestOneToken-1, req)
		in.Requested = "small"
		_, apiErr := rt.Decide(in)
		if apiErr == nil || apiErr.Status != 429 || apiErr.Code != "insufficient_quota" {
			t.Fatalf("err = %+v, want 429 insufficient_quota", apiErr)
		}
		in.Budget = ptr(money.PicoUSD(cheapestOneToken))
		d, apiErr := rt.Decide(in)
		if apiErr != nil || d.Candidates[0].K != 1 {
			t.Fatalf("pinned small with one token of budget: %+v %v, want K 1", d.Candidates, apiErr)
		}
	})
}

// TestExplicitCapIsNeverShrunk: a request that sets max_tokens is admitted at that cap or
// not at all. When no capable model can afford it the answer is 429 insufficient_quota,
// not a 400.
func TestExplicitCapIsNeverShrunk(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	req := catalog.Requirements{InUpper: 100, MaxTokens: 1000, HasMaxTokens: true}

	// small at 1,000 tokens: 10,000,000 + 400,000,000. bargain would need 520,000,000.
	d, apiErr := rt.Decide(budgetInput(410_000_000, req))
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"small"}) || d.Candidates[0].K != 1000 {
		t.Fatalf("candidates %v K %d, want [small] at the explicit 1000", got, d.Candidates[0].K)
	}

	_, apiErr = rt.Decide(budgetInput(409_999_999, req))
	if apiErr == nil || apiErr.Status != 429 || apiErr.Code != "insufficient_quota" {
		t.Fatalf("err = %+v, want 429 insufficient_quota", apiErr)
	}

	// Pinned, with a cap the model allows but the budget cannot pay.
	in := budgetInput(409_999_999, req)
	in.Requested = "small"
	if _, apiErr := rt.Decide(in); apiErr == nil || apiErr.Status != 429 {
		t.Fatalf("pinned: err = %+v, want 429", apiErr)
	}
}

// TestBudgetDoesNotHideCapabilityErrors: a request no model can serve stays a 400 whatever
// the budget is.
func TestBudgetDoesNotHideCapabilityErrors(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	req := reqWithCaps(catalog.CapVision, catalog.CapJSONSchema, catalog.CapTools)
	req.MaxTokens, req.HasMaxTokens = 100_000, true // above every model's output limit
	_, apiErr := rt.Decide(budgetInput(1, req))
	if apiErr == nil || apiErr.Status != 400 {
		t.Fatalf("err = %+v, want the 400 capability error", apiErr)
	}
}

// TestOptimizerShrinksKToTheBudget: the learned policy costs and admits each model with
// the output budget its share of the budget can pay for.
func TestOptimizerShrinksKToTheBudget(t *testing.T) {
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), defaultPolicyConfig())

	// cheap: 100 input tokens cost 100,000,000; 49 output tokens cost 98,000,000.
	d := mustDecide(t, r, withBudget(199_999_999))
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"cheap"}) {
		t.Fatalf("candidates = %v, want [cheap]", got)
	}
	if c := d.Candidates[0]; c.K != 49 || c.CostMax != 198_000_000 {
		t.Errorf("cheap: K = %d CostMax = %d, want 49 and 198,000,000", c.K, c.CostMax)
	}
	if s := scoreOf(t, d, "cheap"); s.CostMax != 198_000_000 || !s.Feasible {
		t.Errorf("cheap score = %+v, want CostMax 198,000,000 and feasible", s)
	}
	if s := scoreOf(t, d, "mid"); s.Feasible || !strings.Contains(strings.Join(s.Reasons, " "), "budget") {
		t.Errorf("mid score = %+v, want infeasible with a budget reason", s)
	}

	// One token of output is the least a model can be admitted with.
	d = mustDecide(t, r, withBudget(102_000_000))
	if c := d.Candidates[0]; c.K != 1 || c.CostMax != 102_000_000 {
		t.Errorf("cheap at the one-token budget: K = %d CostMax = %d, want 1 and 102,000,000", c.K, c.CostMax)
	}
	_, apiErr := r.decide(withBudget(101_999_999))
	if apiErr == nil || apiErr.Status != 429 || apiErr.Code != "insufficient_quota" {
		t.Fatalf("below one token: err = %+v, want 429 insufficient_quota", apiErr)
	}
}

// TestOptimizerExplicitCapIsNeverShrunk: with max_tokens set the learned policy keeps the
// cap and drops the models that cannot afford it.
func TestOptimizerExplicitCapIsNeverShrunk(t *testing.T) {
	r := newOptRig(t, tieredScores(0.4, 0.6, 0.9), defaultPolicyConfig())
	explicit := func(in *Input) {
		in.Req = catalog.Requirements{InUpper: 100, MaxTokens: 50, HasMaxTokens: true}
	}
	_, apiErr := r.decide(explicit, withBudget(199_999_999))
	if apiErr == nil || apiErr.Status != 429 || apiErr.Code != "insufficient_quota" {
		t.Fatalf("err = %+v, want 429 insufficient_quota: 50 tokens of cheap cost 200,000,000", apiErr)
	}
	d := mustDecide(t, r, explicit, withBudget(200_000_000))
	if c := d.Candidates[0]; c.K != 50 || c.CostMax != 200_000_000 {
		t.Errorf("cheap: K = %d CostMax = %d, want 50 and 200,000,000", c.K, c.CostMax)
	}
}

// TestStaticOrderUsesTheAffordableK: two models whose unshrunk worst cases rank one way
// and whose affordable worst cases rank the other.
func TestStaticOrderUsesTheAffordableK(t *testing.T) {
	cfg := &config.Config{Models: []config.ModelConfig{
		{ID: "long", Deployments: []config.DeploymentConfig{{Name: "long-a", Provider: "p", UpstreamModel: "long"}},
			ContextWindow: 1_000_000, MaxOutputTokens: 100_000, InputPricePerMTok: "0", OutputPricePerMTok: "1", Rank: 3},
		{ID: "short", Deployments: []config.DeploymentConfig{{Name: "short-a", Provider: "p", UpstreamModel: "short"}},
			ContextWindow: 1_000_000, MaxOutputTokens: 100, InputPricePerMTok: "0", OutputPricePerMTok: "10", Rank: 2},
	}}
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog.New: %v", err)
	}
	pol, _ := NewPolicy(config.RoutingConfig{Policy: config.PolicyCheapest}, cat)
	rt := New(cat, pol)
	tenant := &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 100_000}
	req := catalog.Requirements{InUpper: 10}

	// Unshrunk, long costs 100,000 * 1,000,000 = 1e11 and short 100 * 10,000,000 = 1e9, so
	// short is cheaper. Under a budget of 5e8 long affords 500 tokens and short 50, both at
	// exactly 5e8, so rank decides and long goes first.
	b := money.PicoUSD(500_000_000)
	d, apiErr := rt.Decide(Input{Tenant: tenant, Requested: config.ReservedModelAuto, Req: req, Budget: &b})
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"long", "short"}) {
		t.Fatalf("candidates = %v, want [long short]", got)
	}
	if d.Candidates[0].K != 500 || d.Candidates[1].K != 50 {
		t.Errorf("K = %d and %d, want 500 and 50", d.Candidates[0].K, d.Candidates[1].K)
	}
	// With enough for both, the shrunk costs still order them: short 1e9 vs long at its
	// own affordable K.
	b = money.PicoUSD(1_500_000_000)
	d, _ = rt.Decide(Input{Tenant: tenant, Requested: config.ReservedModelAuto, Req: req, Budget: &b})
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"short", "long"}) {
		t.Fatalf("candidates = %v, want [short long]", got)
	}
	if d.Candidates[1].K != 1500 {
		t.Errorf("long: K = %d, want 1500", d.Candidates[1].K)
	}
}
