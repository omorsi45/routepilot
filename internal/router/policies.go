package router

import (
	"cmp"
	"fmt"
	"slices"
	"strings"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
)

// Policy orders models for the "auto" model. Order never filters by request
// feasibility: it only ranks the models it is given, and the router drops the
// ones that cannot serve the request afterwards.
type Policy interface {
	// Name is the policy name reported in Decision.Policy.
	Name() string
	// Order returns models best first. It does not modify its argument, and
	// the result is deterministic whatever the input order.
	Order(models []*catalog.Model, in Input) []*catalog.Model
}

// NewPolicy builds the configured policy. An empty policy name returns
// (nil, nil): automatic routing is not configured. The learned policy also returns
// (nil, nil): it is not a static policy, and arrives with each request as Input.Learned
// (see NewLearned). It fails on an unknown policy name, or on a rule or default model
// missing from the catalog.
func NewPolicy(cfg config.RoutingConfig, cat *catalog.Catalog) (Policy, error) {
	switch cfg.Policy {
	case "", config.PolicyLearned:
		return nil, nil
	case config.PolicyCheapest:
		return cheapest{}, nil
	case config.PolicyStrongest:
		return strongest{}, nil
	case config.PolicyRules:
		p, err := newRules(cfg, cat)
		if err != nil {
			// Return an untyped nil: a nil *rules in a Policy is not == nil.
			return nil, err
		}
		return p, nil
	default:
		return nil, fmt.Errorf("unknown routing policy %q", cfg.Policy)
	}
}

// outputBudget is the output token budget a model is costed at for this
// request, the same K the router reports in its Candidate: the effective K
// reduced to what the request's remaining budget pays for.
func outputBudget(m *catalog.Model, in Input) int {
	k, _ := m.AffordableK(in.Req, in.Tenant.DefaultMaxTokens, in.Budget)
	return k
}

func costMax(m *catalog.Model, in Input) money.PicoUSD {
	return m.CostMax(in.Req, outputBudget(m, in))
}

// sorted returns a sorted copy of models. ties break on ID ascending last, so
// the result does not depend on the input order.
func sorted(models []*catalog.Model, in Input, cmpFn func(a, b *catalog.Model, in Input) int) []*catalog.Model {
	out := slices.Clone(models)
	slices.SortStableFunc(out, func(a, b *catalog.Model) int {
		if c := cmpFn(a, b, in); c != 0 {
			return c
		}
		return strings.Compare(a.ID, b.ID)
	})
	return out
}

// cheapest orders by worst-case cost ascending, then rank descending.
type cheapest struct{}

func (cheapest) Name() string { return config.PolicyCheapest }

func (cheapest) Order(models []*catalog.Model, in Input) []*catalog.Model {
	return sorted(models, in, func(a, b *catalog.Model, in Input) int {
		if c := cmp.Compare(costMax(a, in), costMax(b, in)); c != 0 {
			return c
		}
		return cmp.Compare(b.Rank, a.Rank)
	})
}

// strongest orders by rank descending, then worst-case cost ascending.
type strongest struct{}

func (strongest) Name() string { return config.PolicyStrongest }

func (strongest) Order(models []*catalog.Model, in Input) []*catalog.Model {
	return sorted(models, in, func(a, b *catalog.Model, in Input) int {
		if c := cmp.Compare(b.Rank, a.Rank); c != 0 {
			return c
		}
		return cmp.Compare(costMax(a, in), costMax(b, in))
	})
}

// rules puts the model of the first matching rule first, then the default
// model, then every other model in strongest order.
type rules struct {
	rules        []rule
	defaultModel string
}

// rule is a config rule with its conditions resolved once at startup.
type rule struct {
	model          string
	minPromptBytes int
	maxPromptBytes int
	containsAny    []string // lowercased
	requires       []catalog.Capability
}

func newRules(cfg config.RoutingConfig, cat *catalog.Catalog) (*rules, error) {
	p := &rules{defaultModel: cfg.DefaultModel}
	if cfg.DefaultModel != "" {
		if _, ok := cat.Model(cfg.DefaultModel); !ok {
			return nil, fmt.Errorf("routing default_model %q is not a configured model", cfg.DefaultModel)
		}
	}
	for i, rc := range cfg.Rules {
		if _, ok := cat.Model(rc.Model); !ok {
			return nil, fmt.Errorf("routing rule %d (%q): model %q is not a configured model", i, rc.Name, rc.Model)
		}
		r := rule{
			model:          rc.Model,
			minPromptBytes: rc.When.MinPromptBytes,
			maxPromptBytes: rc.When.MaxPromptBytes,
		}
		for _, s := range rc.When.ContainsAny {
			r.containsAny = append(r.containsAny, strings.ToLower(s))
		}
		for _, name := range rc.When.Requires {
			c, ok := catalog.ParseCapability(name)
			if !ok {
				return nil, fmt.Errorf("routing rule %d (%q): unknown capability %q", i, rc.Name, name)
			}
			r.requires = append(r.requires, c)
		}
		p.rules = append(p.rules, r)
	}
	return p, nil
}

func (*rules) Name() string { return config.PolicyRules }

func (p *rules) Order(models []*catalog.Model, in Input) []*catalog.Model {
	byID := make(map[string]*catalog.Model, len(models))
	for _, m := range models {
		byID[m.ID] = m
	}

	out := make([]*catalog.Model, 0, len(models))
	seen := make(map[string]bool, len(models))
	add := func(id string) {
		if m, ok := byID[id]; ok && !seen[id] {
			seen[id] = true
			out = append(out, m)
		}
	}

	lowerText := strings.ToLower(in.Req.Text)
	for _, r := range p.rules {
		if r.matches(in.Req, lowerText) {
			add(r.model)
			break
		}
	}
	add(p.defaultModel)
	for _, m := range (strongest{}).Order(models, in) {
		add(m.ID)
	}
	return out
}

// matches reports whether every condition the rule sets holds. lowerText is
// req.Text lowercased by the caller so it is computed once per request.
func (r rule) matches(req catalog.Requirements, lowerText string) bool {
	if req.PromptBytes < r.minPromptBytes {
		return false
	}
	if r.maxPromptBytes > 0 && req.PromptBytes > r.maxPromptBytes {
		return false
	}
	if len(r.containsAny) > 0 && !slices.ContainsFunc(r.containsAny, func(s string) bool {
		return strings.Contains(lowerText, s)
	}) {
		return false
	}
	for _, c := range r.requires {
		if !req.Caps.Has(c) {
			return false
		}
	}
	return true
}
