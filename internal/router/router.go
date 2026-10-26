// Package router decides which models and deployments may serve a request. It
// filters by tenant permission, model capabilities and limits, and deployment
// availability, and orders the survivors with a static policy. It does not
// call providers; the gateway walks Decision.Candidates in order.
package router

import (
	"fmt"
	"strings"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
)

// PolicyPinned is the Decision.Policy value when the client named a model.
const PolicyPinned = "pinned"

// Input is one routing question.
type Input struct {
	Tenant *config.TenantConfig
	// Requested is a catalog model ID or config.ReservedModelAuto.
	Requested string
	Req       catalog.Requirements
	// Available reports whether a deployment may be tried now. Nil means all
	// deployments are available.
	Available func(deployment string) bool
}

// Candidate is one model the gateway may try, with the deployments to try.
type Candidate struct {
	Model *catalog.Model
	// K is the output token budget the model is admitted and costed with.
	K int
	// CostMax is the worst-case cost of the request on this model.
	CostMax money.PicoUSD
	// Deployments are the available deployments, in config order.
	Deployments []catalog.Deployment
}

// Rejection records why an allowed model cannot serve the request.
type Rejection struct {
	Model   string
	Reasons []catalog.Reason
}

// Decision is the router's answer: candidates in the order to try them.
type Decision struct {
	// Policy is the policy name for auto, or PolicyPinned for a named model.
	Policy     string
	Candidates []Candidate
	// Rejected lists allowed models that failed the capability and limit
	// checks, in catalog order. Models the tenant may not use never appear.
	Rejected []Rejection
}

// Router applies feasibility filtering and a policy. It is immutable and safe
// for concurrent use.
type Router struct {
	cat    *catalog.Catalog
	policy Policy
}

// New returns a Router. A nil policy means "auto" is not configured.
func New(cat *catalog.Catalog, policy Policy) *Router {
	return &Router{cat: cat, policy: policy}
}

// HasAuto reports whether the "auto" model is configured.
func (r *Router) HasAuto() bool { return r.policy != nil }

// Decide picks the candidates for in, or returns the OpenAI error to send.
// On a capability error the Decision still carries Policy and Rejected.
func (r *Router) Decide(in Input) (Decision, *openai.APIError) {
	if in.Tenant == nil {
		return Decision{}, openai.ErrInternal()
	}
	if in.Requested == config.ReservedModelAuto {
		return r.decideAuto(in)
	}
	return r.decidePinned(in)
}

func (r *Router) decidePinned(in Input) (Decision, *openai.APIError) {
	d := Decision{Policy: PolicyPinned}
	m, ok := r.cat.Model(in.Requested)
	if !ok {
		return d, openai.ErrModelNotFound(in.Requested)
	}
	if !in.Tenant.Allows(m.ID) {
		return d, openai.ErrModelNotAllowed(m.ID)
	}
	if reasons := m.Check(in.Req, in.Tenant.DefaultMaxTokens); len(reasons) > 0 {
		d.Rejected = []Rejection{{Model: m.ID, Reasons: reasons}}
		return d, openai.ErrCapabilityUnavailable(
			fmt.Sprintf("model %s cannot serve this request: %s", m.ID, joinReasons(reasons)))
	}
	return finish(d, []*catalog.Model{m}, in)
}

func (r *Router) decideAuto(in Input) (Decision, *openai.APIError) {
	if r.policy == nil {
		return Decision{}, openai.ErrInvalidRequest("routing policy not configured", "model")
	}
	d := Decision{Policy: r.policy.Name()}

	var allowed []*catalog.Model
	for _, m := range r.cat.Models() {
		if in.Tenant.Allows(m.ID) {
			allowed = append(allowed, m)
		}
	}
	if len(allowed) == 0 {
		return d, openai.ErrCapabilityUnavailable("no model available to this tenant")
	}

	// Check each allowed model once, in catalog order, so Rejected and the
	// error message are stable whatever the policy order is.
	infeasible := make(map[string]bool)
	feasibleIDs := make(map[string]bool, len(allowed))
	var rejectedText []string
	for _, m := range allowed {
		reasons := m.Check(in.Req, in.Tenant.DefaultMaxTokens)
		if len(reasons) == 0 {
			feasibleIDs[m.ID] = true
			continue
		}
		infeasible[m.ID] = true
		d.Rejected = append(d.Rejected, Rejection{Model: m.ID, Reasons: reasons})
		rejectedText = append(rejectedText, m.ID+": "+joinReasons(reasons))
	}
	if len(infeasible) == len(allowed) {
		return d, openai.ErrCapabilityUnavailable("no model satisfies the request: " + strings.Join(rejectedText, "; "))
	}

	// Keep only models in the positive feasible set, each once, so a policy
	// cannot return a model outside the tenant boundary or the check results.
	var feasible []*catalog.Model
	for _, m := range r.policy.Order(allowed, in) {
		if feasibleIDs[m.ID] {
			delete(feasibleIDs, m.ID)
			feasible = append(feasible, m)
		}
	}
	return finish(d, feasible, in)
}

// finish turns ordered feasible models into candidates, dropping models with
// no available deployment. It fails with no_healthy_deployment when none is left.
func finish(d Decision, models []*catalog.Model, in Input) (Decision, *openai.APIError) {
	for _, m := range models {
		deps := availableDeployments(m, in.Available)
		if len(deps) == 0 {
			continue
		}
		k := m.EffectiveK(in.Req, in.Tenant.DefaultMaxTokens)
		d.Candidates = append(d.Candidates, Candidate{
			Model:       m,
			K:           k,
			CostMax:     m.CostMax(in.Req, k),
			Deployments: deps,
		})
	}
	if len(d.Candidates) == 0 {
		return d, openai.ErrNoHealthyDeployment()
	}
	return d, nil
}

// availableDeployments copies the available deployments of m in config order.
func availableDeployments(m *catalog.Model, available func(string) bool) []catalog.Deployment {
	deps := make([]catalog.Deployment, 0, len(m.Deployments))
	for _, dep := range m.Deployments {
		if available == nil || available(dep.Name) {
			deps = append(deps, dep)
		}
	}
	return deps
}

func joinReasons(reasons []catalog.Reason) string {
	parts := make([]string, len(reasons))
	for i, r := range reasons {
		parts[i] = string(r)
	}
	return strings.Join(parts, ", ")
}
