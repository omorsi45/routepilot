package router

import (
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
)

// fixtureModel describes one model of the test catalog. Every model gets two
// deployments named "<id>-a" and "<id>-b".
type fixtureModel struct {
	id       string
	in, out  string
	rank     int
	caps     []string
	ctx, max int
}

// fixtureModels is in catalog (config) order. Prices and ranks are chosen so
// every tie-break of the static policies is exercised:
//
//	cheapest: small, bargain, alt (rank 7), mid, mid2 (ID), big
//	strongest: big, alt, bargain (cheaper), mid, mid2 (ID), small
var fixtureModels = []fixtureModel{
	{"big", "3", "15", 9, []string{"streaming", "tools", "json_object", "json_schema", "vision"}, 200000, 8000},
	{"mid2", "0.5", "1.5", 5, []string{"streaming", "tools"}, 100000, 4000},
	{"mid", "0.5", "1.5", 5, []string{"streaming", "tools"}, 100000, 4000},
	{"alt", "0.5", "1.5", 7, []string{"streaming", "tools"}, 100000, 4000},
	{"bargain", "0.2", "0.5", 5, []string{"streaming", "tools"}, 100000, 4000},
	{"small", "0.1", "0.4", 1, []string{"streaming"}, 8000, 1000},
}

func fixtureCatalog(t *testing.T) *catalog.Catalog {
	t.Helper()
	cfg := &config.Config{}
	for _, fm := range fixtureModels {
		cfg.Models = append(cfg.Models, config.ModelConfig{
			ID: fm.id,
			Deployments: []config.DeploymentConfig{
				{Name: fm.id + "-a", Provider: "p", UpstreamModel: fm.id},
				{Name: fm.id + "-b", Provider: "p", UpstreamModel: fm.id},
			},
			ContextWindow:      fm.ctx,
			MaxOutputTokens:    fm.max,
			Capabilities:       fm.caps,
			InputPricePerMTok:  fm.in,
			OutputPricePerMTok: fm.out,
			Rank:               fm.rank,
		})
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog.New: %v", err)
	}
	return cat
}

func tenantOf(allowed ...string) *config.TenantConfig {
	return &config.TenantConfig{ID: "t", AllowedModels: allowed, DefaultMaxTokens: 256}
}

func newRouter(t *testing.T, cfg config.RoutingConfig) (*Router, *catalog.Catalog) {
	t.Helper()
	cat := fixtureCatalog(t)
	pol, err := NewPolicy(cfg, cat)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}
	return New(cat, pol), cat
}

func ids(models []*catalog.Model) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.ID
	}
	return out
}

func candidateIDs(d Decision) []string {
	out := make([]string, len(d.Candidates))
	for i, c := range d.Candidates {
		out[i] = c.Model.ID
	}
	return out
}

// capSetOf builds a CapSet from capabilities. CapSet is a bitmask indexed by
// Capability, so this needs no catalog helper.
func capSetOf(caps ...catalog.Capability) catalog.CapSet {
	var s catalog.CapSet
	for _, c := range caps {
		s |= catalog.CapSet(1) << c
	}
	return s
}

func reqWithCaps(caps ...catalog.Capability) catalog.Requirements {
	return catalog.Requirements{Caps: capSetOf(caps...), InUpper: 100}
}

func TestDecidePinned(t *testing.T) {
	rt, cat := newRouter(t, config.RoutingConfig{})
	mid, _ := cat.Model("mid")
	req := catalog.Requirements{InUpper: 100}

	t.Run("feasible", func(t *testing.T) {
		d, apiErr := rt.Decide(Input{Tenant: tenantOf("mid"), Requested: "mid", Req: req})
		if apiErr != nil {
			t.Fatalf("unexpected error: %v", apiErr)
		}
		if d.Policy != "pinned" {
			t.Errorf("Policy = %q, want pinned", d.Policy)
		}
		if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid"}) {
			t.Fatalf("candidates = %v, want [mid]", got)
		}
		c := d.Candidates[0]
		wantK := mid.EffectiveK(req, 256)
		if c.K != wantK || c.K != 256 {
			t.Errorf("K = %d, want %d (tenant default 256)", c.K, wantK)
		}
		if c.CostMax != mid.CostMax(req, wantK) || c.CostMax <= 0 {
			t.Errorf("CostMax = %d, want %d", c.CostMax, mid.CostMax(req, wantK))
		}
		if got := []string{c.Deployments[0].Name, c.Deployments[1].Name}; !reflect.DeepEqual(got, []string{"mid-a", "mid-b"}) || len(c.Deployments) != 2 {
			t.Errorf("deployments = %v", c.Deployments)
		}
		if len(d.Rejected) != 0 {
			t.Errorf("Rejected = %v, want none", d.Rejected)
		}
	})

	t.Run("not allowed is 403", func(t *testing.T) {
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("mid"), Requested: "big", Req: req})
		if apiErr == nil || apiErr.Status != 403 || apiErr.Code != "model_not_allowed" {
			t.Fatalf("err = %v, want 403 model_not_allowed", apiErr)
		}
	})

	t.Run("wildcard tenant is allowed", func(t *testing.T) {
		if _, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "big", Req: req}); apiErr != nil {
			t.Fatalf("unexpected error: %v", apiErr)
		}
	})

	t.Run("infeasible is 400 with reasons", func(t *testing.T) {
		r := reqWithCaps(catalog.CapTools, catalog.CapVision)
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "mid", Req: r})
		if apiErr == nil || apiErr.Status != 400 || apiErr.Code != "model_capability_unavailable" || apiErr.Param != "model" {
			t.Fatalf("err = %v, want 400 model_capability_unavailable param model", apiErr)
		}
		const want = "model mid cannot serve this request: missing capability vision"
		if apiErr.Message != want {
			t.Errorf("message = %q, want %q", apiErr.Message, want)
		}
	})

	t.Run("several reasons are joined", func(t *testing.T) {
		r := reqWithCaps(catalog.CapTools, catalog.CapVision)
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "small", Req: r})
		if apiErr == nil {
			t.Fatal("want error")
		}
		const want = "model small cannot serve this request: missing capability tools, missing capability vision"
		if apiErr.Message != want {
			t.Errorf("message = %q, want %q", apiErr.Message, want)
		}
	})

	t.Run("unknown is 404", func(t *testing.T) {
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "nope", Req: req})
		if apiErr == nil || apiErr.Status != 404 || apiErr.Code != "model_not_found" {
			t.Fatalf("err = %v, want 404 model_not_found", apiErr)
		}
	})
}

func TestDecideAutoWithoutPolicy(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{})
	if rt.HasAuto() {
		t.Fatal("HasAuto = true without a policy")
	}
	_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto})
	if apiErr == nil || apiErr.Status != 400 || apiErr.Code != "invalid_request" || apiErr.Param != "model" {
		t.Fatalf("err = %v, want 400 invalid_request param model", apiErr)
	}
	if apiErr.Message != "routing policy not configured" {
		t.Errorf("message = %q", apiErr.Message)
	}

	withPolicy, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	if !withPolicy.HasAuto() {
		t.Error("HasAuto = false with a policy")
	}
}

func TestNewPolicy(t *testing.T) {
	cat := fixtureCatalog(t)

	if p, err := NewPolicy(config.RoutingConfig{}, cat); p != nil || err != nil {
		t.Errorf("empty policy = (%v, %v), want (nil, nil)", p, err)
	}
	for policy, name := range map[string]string{config.PolicyCheapest: "cheapest", config.PolicyStrongest: "strongest", config.PolicyRules: "rules"} {
		p, err := NewPolicy(config.RoutingConfig{Policy: policy, DefaultModel: "mid"}, cat)
		if err != nil || p == nil || p.Name() != name {
			t.Errorf("policy %q = (%v, %v), want name %q", policy, p, err, name)
		}
	}

	errCases := []struct {
		name string
		cfg  config.RoutingConfig
		want string
	}{
		{"unknown policy", config.RoutingConfig{Policy: "random"}, "random"},
		{"default model missing", config.RoutingConfig{Policy: config.PolicyRules, DefaultModel: "ghost"}, "ghost"},
		{"rule model missing", config.RoutingConfig{Policy: config.PolicyRules, DefaultModel: "mid",
			Rules: []config.RuleConfig{{Name: "r1", Model: "ghost", When: config.RuleMatch{MinPromptBytes: 1}}}}, "ghost"},
		{"unknown requires name", config.RoutingConfig{Policy: config.PolicyRules, DefaultModel: "mid",
			Rules: []config.RuleConfig{{Name: "r1", Model: "big", When: config.RuleMatch{Requires: []string{"telepathy"}}}}}, "telepathy"},
	}
	for _, tc := range errCases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewPolicy(tc.cfg, cat)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to mention %q", err, tc.want)
			}
			if p != nil {
				t.Errorf("policy = %v, want nil on error", p)
			}
		})
	}
}

func TestCheapestOrder(t *testing.T) {
	rt, cat := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	d, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto, Req: catalog.Requirements{InUpper: 100}})
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if d.Policy != "cheapest" {
		t.Errorf("Policy = %q, want cheapest", d.Policy)
	}
	// small is cheapest, then bargain; alt/mid/mid2 tie on cost and break by
	// rank (alt, rank 7) then ID; big is last.
	want := []string{"small", "bargain", "alt", "mid", "mid2", "big"}
	if got := candidateIDs(d); !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
	for i := 1; i < len(d.Candidates); i++ {
		if d.Candidates[i-1].CostMax > d.Candidates[i].CostMax {
			t.Errorf("CostMax not ascending at %d", i)
		}
	}

	// The order must not depend on the order the models are handed in.
	pol, _ := NewPolicy(config.RoutingConfig{Policy: config.PolicyCheapest}, cat)
	models := cat.Models()
	slices.Reverse(models)
	got := ids(pol.Order(models, Input{Tenant: tenantOf("*"), Req: catalog.Requirements{InUpper: 100}}))
	if !reflect.DeepEqual(got, want) {
		t.Errorf("reversed input order = %v, want %v", got, want)
	}
}

func TestStrongestOrder(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyStrongest})
	d, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto, Req: catalog.Requirements{InUpper: 100}})
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if d.Policy != "strongest" {
		t.Errorf("Policy = %q, want strongest", d.Policy)
	}
	// rank 9, 7, then the rank 5 group by cost (bargain) then ID, then rank 1.
	want := []string{"big", "alt", "bargain", "mid", "mid2", "small"}
	if got := candidateIDs(d); !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
}

func TestRulesOrder(t *testing.T) {
	cfg := config.RoutingConfig{
		Policy:       config.PolicyRules,
		DefaultModel: "mid",
		Rules: []config.RuleConfig{
			{Name: "long", Model: "big", When: config.RuleMatch{MinPromptBytes: 1000}},
			{Name: "code", Model: "alt", When: config.RuleMatch{ContainsAny: []string{"Refactor", "stack trace"}}},
			{Name: "tooling", Model: "mid2", When: config.RuleMatch{Requires: []string{config.CapTools}}},
			{Name: "short-and-tagged", Model: "bargain", When: config.RuleMatch{MinPromptBytes: 10, MaxPromptBytes: 50, ContainsAny: []string{"tagged"}}},
		},
	}
	rt, _ := newRouter(t, cfg)
	tools := capSetOf(catalog.CapTools)

	cases := []struct {
		name string
		req  catalog.Requirements
		want []string
	}{
		{"long prompt rule", catalog.Requirements{PromptBytes: 2000, InUpper: 100},
			[]string{"big", "mid", "alt", "bargain", "mid2", "small"}},
		{"contains_any is case insensitive", catalog.Requirements{PromptBytes: 20, InUpper: 100, Text: "please REFACTOR this"},
			[]string{"alt", "mid", "big", "bargain", "mid2", "small"}},
		{"contains_any second entry", catalog.Requirements{PromptBytes: 20, InUpper: 100, Text: "got a Stack Trace here"},
			[]string{"alt", "mid", "big", "bargain", "mid2", "small"}},
		{"requires rule", catalog.Requirements{PromptBytes: 20, InUpper: 100, Caps: tools},
			[]string{"mid2", "mid", "big", "alt", "bargain"}}, // small lacks tools
		{"first matching rule wins", catalog.Requirements{PromptBytes: 2000, InUpper: 100, Caps: tools, Text: "refactor"},
			[]string{"big", "mid", "alt", "bargain", "mid2"}}, // small lacks tools
		{"every set field must match", catalog.Requirements{PromptBytes: 5, InUpper: 100, Text: "tagged"},
			[]string{"mid", "big", "alt", "bargain", "mid2", "small"}},
		{"max_prompt_bytes bounds the match", catalog.Requirements{PromptBytes: 51, InUpper: 100, Text: "tagged"},
			[]string{"mid", "big", "alt", "bargain", "mid2", "small"}},
		{"all fields match", catalog.Requirements{PromptBytes: 20, InUpper: 100, Text: "a TAGGED one"},
			[]string{"bargain", "mid", "big", "alt", "mid2", "small"}},
		{"no match puts default first", catalog.Requirements{PromptBytes: 20, InUpper: 100, Text: "hello"},
			[]string{"mid", "big", "alt", "bargain", "mid2", "small"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: config.ReservedModelAuto, Req: tc.req})
			if apiErr != nil {
				t.Fatalf("unexpected error: %v", apiErr)
			}
			if d.Policy != "rules" {
				t.Errorf("Policy = %q, want rules", d.Policy)
			}
			if got := candidateIDs(d); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("order = %v, want %v", got, tc.want)
			}
		})
	}

	t.Run("rule model the tenant lacks is skipped", func(t *testing.T) {
		d, apiErr := rt.Decide(Input{Tenant: tenantOf("mid", "small"), Requested: config.ReservedModelAuto,
			Req: catalog.Requirements{PromptBytes: 2000, InUpper: 100}})
		if apiErr != nil {
			t.Fatalf("unexpected error: %v", apiErr)
		}
		if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid", "small"}) {
			t.Errorf("order = %v, want [mid small]", got)
		}
	})
}

func TestAutoFiltersByTenantAndCapability(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	req := reqWithCaps(catalog.CapTools)

	d, apiErr := rt.Decide(Input{Tenant: tenantOf("small", "mid", "big"), Requested: config.ReservedModelAuto, Req: req})
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	// small lacks tools, so it is rejected; alt, bargain and mid2 are not
	// allowed and must not appear anywhere.
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid", "big"}) {
		t.Fatalf("candidates = %v, want [mid big]", got)
	}
	if len(d.Rejected) != 1 || d.Rejected[0].Model != "small" ||
		!reflect.DeepEqual(d.Rejected[0].Reasons, []catalog.Reason{"missing capability tools"}) {
		t.Errorf("Rejected = %+v, want small: missing capability tools", d.Rejected)
	}
}

func TestAutoRespectsContextWindow(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	// small has an 8000 token window; the prompt bound rules it out.
	d, apiErr := rt.Decide(Input{Tenant: tenantOf("small", "mid"), Requested: config.ReservedModelAuto, Req: catalog.Requirements{InUpper: 20000}})
	if apiErr != nil {
		t.Fatalf("unexpected error: %v", apiErr)
	}
	if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid"}) {
		t.Fatalf("candidates = %v, want [mid]", got)
	}
	if len(d.Rejected) != 1 || d.Rejected[0].Model != "small" || len(d.Rejected[0].Reasons) != 1 ||
		!strings.HasPrefix(string(d.Rejected[0].Reasons[0]), "context window 8000") {
		t.Errorf("Rejected = %+v", d.Rejected)
	}
}

func TestNoFeasibleModelListsReasons(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	req := reqWithCaps(catalog.CapTools, catalog.CapVision)

	_, apiErr := rt.Decide(Input{Tenant: tenantOf("small", "mid", "ghost"), Requested: config.ReservedModelAuto, Req: req})
	if apiErr == nil || apiErr.Status != 400 || apiErr.Code != "model_capability_unavailable" || apiErr.Param != "model" {
		t.Fatalf("err = %v, want 400 model_capability_unavailable param model", apiErr)
	}
	// Catalog order (mid before small), reasons joined with ", ", models with "; ".
	const want = "no model satisfies the request: mid: missing capability vision; small: missing capability tools, missing capability vision"
	if apiErr.Message != want {
		t.Errorf("message = %q\nwant      %q", apiErr.Message, want)
	}

	t.Run("tenant allows nothing in the catalog", func(t *testing.T) {
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("ghost"), Requested: config.ReservedModelAuto, Req: catalog.Requirements{InUpper: 100}})
		if apiErr == nil || apiErr.Status != 400 || apiErr.Code != "model_capability_unavailable" {
			t.Fatalf("err = %v, want 400 model_capability_unavailable", apiErr)
		}
		if apiErr.Message != "no model available to this tenant" {
			t.Errorf("message = %q", apiErr.Message)
		}
	})
}

func TestUnavailableDeploymentsDropped(t *testing.T) {
	rt, _ := newRouter(t, config.RoutingConfig{Policy: config.PolicyCheapest})
	down := func(names ...string) func(string) bool {
		return func(dep string) bool { return !slices.Contains(names, dep) }
	}
	req := catalog.Requirements{InUpper: 100}
	auto := func(avail func(string) bool, allowed ...string) (Decision, int, string) {
		d, apiErr := rt.Decide(Input{Tenant: tenantOf(allowed...), Requested: config.ReservedModelAuto, Req: req, Available: avail})
		if apiErr != nil {
			return d, apiErr.Status, apiErr.Code
		}
		return d, 0, ""
	}

	t.Run("nil Available means all available", func(t *testing.T) {
		d, status, _ := auto(nil, "small", "mid")
		if status != 0 || len(d.Candidates) != 2 || len(d.Candidates[0].Deployments) != 2 {
			t.Fatalf("status=%d candidates=%+v", status, d.Candidates)
		}
	})

	t.Run("one deployment down keeps the model with the other", func(t *testing.T) {
		d, status, _ := auto(down("small-a"), "small", "mid")
		if status != 0 {
			t.Fatalf("status = %d", status)
		}
		if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"small", "mid"}) {
			t.Fatalf("candidates = %v", got)
		}
		deps := d.Candidates[0].Deployments
		if len(deps) != 1 || deps[0].Name != "small-b" {
			t.Errorf("small deployments = %+v, want [small-b]", deps)
		}
		if len(d.Candidates[1].Deployments) != 2 {
			t.Errorf("mid deployments = %+v, want both", d.Candidates[1].Deployments)
		}
	})

	t.Run("model with no deployment left is dropped", func(t *testing.T) {
		d, status, _ := auto(down("small-a", "small-b"), "small", "mid")
		if status != 0 {
			t.Fatalf("status = %d", status)
		}
		if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid"}) {
			t.Fatalf("candidates = %v, want [mid]", got)
		}
	})

	t.Run("all gone is 503", func(t *testing.T) {
		_, status, code := auto(down("small-a", "small-b", "mid-a", "mid-b"), "small", "mid")
		if status != 503 || code != "no_healthy_deployment" {
			t.Fatalf("status=%d code=%q, want 503 no_healthy_deployment", status, code)
		}
	})

	t.Run("pinned with every deployment down is 503", func(t *testing.T) {
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("*"), Requested: "mid", Req: req, Available: down("mid-a", "mid-b")})
		if apiErr == nil || apiErr.Status != 503 || apiErr.Code != "no_healthy_deployment" {
			t.Fatalf("err = %v, want 503 no_healthy_deployment", apiErr)
		}
	})

	t.Run("infeasible beats unavailable", func(t *testing.T) {
		r := catalog.Requirements{InUpper: 100, Caps: capSetOf(catalog.CapVision)}
		_, apiErr := rt.Decide(Input{Tenant: tenantOf("small", "mid"), Requested: config.ReservedModelAuto, Req: r,
			Available: down("small-a", "small-b", "mid-a", "mid-b")})
		if apiErr == nil || apiErr.Code != "model_capability_unavailable" {
			t.Fatalf("err = %v, want model_capability_unavailable", apiErr)
		}
	})
}

// extraPolicy is a policy stub that returns every model it is given, plus one
// model it was not given.
type extraPolicy struct{ extra *catalog.Model }

func (extraPolicy) Name() string { return "extra" }

func (p extraPolicy) Order(models []*catalog.Model, _ Input) []*catalog.Model {
	return append(slices.Clone(models), p.extra)
}

func TestAutoPolicyCannotEscapeFeasibleSet(t *testing.T) {
	cat := fixtureCatalog(t)
	big, _ := cat.Model("big")     // not allowed for the tenant below
	small, _ := cat.Model("small") // allowed but cannot serve the tools request
	for name, extra := range map[string]*catalog.Model{"tenant-disallowed": big, "infeasible": small} {
		t.Run(name, func(t *testing.T) {
			rt := New(cat, extraPolicy{extra: extra})
			d, apiErr := rt.Decide(Input{
				Tenant:    tenantOf("small", "mid"),
				Requested: config.ReservedModelAuto,
				Req:       reqWithCaps(catalog.CapTools),
			})
			if apiErr != nil {
				t.Fatalf("Decide: %v", apiErr)
			}
			if got := candidateIDs(d); !reflect.DeepEqual(got, []string{"mid"}) {
				t.Fatalf("candidates = %v, want [mid]", got)
			}
		})
	}
}
