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
	"github.com/omorsi45/routepilot/internal/features"
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

	// The fields below are read only by a Learned policy.

	// Learned, when set, decides "auto" requests instead of the static policy.
	Learned Learned
	// Messages are the request messages with their roles; Req.Text is cut and joined, so
	// the scorer needs these.
	Messages []features.Message
	// Floor is the request's own quality floor (min_quality), nil when it sets none.
	Floor *float64
	// SLOTTFTMs and SLOE2EMs are the request's latency SLOs in milliseconds, nil when it
	// sets none.
	SLOTTFTMs, SLOE2EMs *int
	// Budget is the cost the request may still spend, nil for unlimited.
	Budget *money.PicoUSD
	// Rand draws uniform values in [0, 1) for exploration. Nil turns exploration off.
	Rand func() float64
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
	// checks, in catalog order, then the models that passed them but cannot be
	// admitted within the request's budget (reason "budget: ..."), in the
	// policy's order. Models the tenant may not use never appear.
	Rejected []Rejection

	// The fields below are set by a Learned policy. They stay set on a routing error
	// that follows the scoring (no_feasible_model, insufficient_quota).

	// PolicyVersion is Learned.Version().
	PolicyVersion string
	// Relaxed, OOD, OODScore, Explored, Propensity, Epsilon and Floor are the Choice's.
	Relaxed    []string
	OOD        bool
	OODScore   float64
	Explored   bool
	Propensity float64
	Epsilon    float64
	Floor      float64
	// Scores has one entry for every model the tenant may use, in catalog order. A model
	// the policy could not consider has Feasible false and the reasons.
	Scores []ModelScore
}

// Router applies feasibility filtering and a policy. It is immutable and safe
// for concurrent use.
type Router struct {
	cat    *catalog.Catalog
	policy Policy
	// learned is true for a router whose "auto" model is decided by a Learned policy that
	// arrives with each request in Input.Learned.
	learned bool
}

// New returns a Router. A nil policy means "auto" is not configured.
func New(cat *catalog.Catalog, policy Policy) *Router {
	return &Router{cat: cat, policy: policy}
}

// NewLearned returns a Router whose "auto" model is available and is decided by the
// Learned policy each request brings in Input.Learned. The router holds no policy of its
// own because the one to use depends on the request (stable, canary).
func NewLearned(cat *catalog.Catalog) *Router {
	return &Router{cat: cat, learned: true}
}

// HasAuto reports whether the "auto" model is configured.
func (r *Router) HasAuto() bool { return r.policy != nil || r.learned }

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
	if in.Learned != nil {
		return r.decideLearned(in)
	}
	if r.policy == nil {
		return Decision{}, openai.ErrInvalidRequest("routing policy not configured", "model")
	}
	d := Decision{Policy: r.policy.Name()}

	allowed, feasible, apiErr := r.screen(in, &d)
	if apiErr != nil {
		return d, apiErr
	}
	feasibleIDs := make(map[string]bool, len(feasible))
	for _, m := range feasible {
		feasibleIDs[m.ID] = true
	}

	// Keep only models in the positive feasible set, each once, so a policy
	// cannot return a model outside the tenant boundary or the check results.
	var ordered []*catalog.Model
	for _, m := range r.policy.Order(allowed, in) {
		if feasibleIDs[m.ID] {
			delete(feasibleIDs, m.ID)
			ordered = append(ordered, m)
		}
	}
	return finish(d, ordered, in)
}

// screen applies the tenant boundary and then the capability and limit checks to every
// model, once, in catalog order, so Rejected and the error message are stable whatever
// the policy order is. It returns the models the tenant may use and those among them that
// pass the checks, and fails when there are none of either.
func (r *Router) screen(in Input, d *Decision) (allowed, feasible []*catalog.Model, apiErr *openai.APIError) {
	for _, m := range r.cat.Models() {
		if in.Tenant.Allows(m.ID) {
			allowed = append(allowed, m)
		}
	}
	if len(allowed) == 0 {
		return nil, nil, openai.ErrCapabilityUnavailable("no model available to this tenant")
	}

	var rejectedText []string
	for _, m := range allowed {
		reasons := m.Check(in.Req, in.Tenant.DefaultMaxTokens)
		if len(reasons) == 0 {
			feasible = append(feasible, m)
			continue
		}
		d.Rejected = append(d.Rejected, Rejection{Model: m.ID, Reasons: reasons})
		rejectedText = append(rejectedText, m.ID+": "+joinReasons(reasons))
	}
	if len(feasible) == 0 {
		return allowed, nil, openai.ErrCapabilityUnavailable("no model satisfies the request: " + strings.Join(rejectedText, "; "))
	}
	return allowed, feasible, nil
}

// decideLearned routes with in.Learned. F(x), the models it may choose among, are those
// the tenant may use, that pass the capability and limit checks and that have an available
// deployment (the policy itself drops those its artifact does not cover). Every other
// allowed model still appears in Decision.Scores, infeasible, with its reasons.
func (r *Router) decideLearned(in Input) (Decision, *openai.APIError) {
	d := Decision{Policy: in.Learned.Name(), PolicyVersion: in.Learned.Version()}
	allowed, feasible, apiErr := r.screen(in, &d)

	excluded := make(map[string][]string, len(d.Rejected))
	for _, rej := range d.Rejected {
		reasons := make([]string, len(rej.Reasons))
		for i, reason := range rej.Reasons {
			reasons[i] = string(reason)
		}
		excluded[rej.Model] = reasons
	}
	if apiErr != nil {
		d.Scores = mergeScores(allowed, excluded, nil)
		return d, apiErr
	}

	var usable []*catalog.Model
	for _, m := range feasible {
		if len(availableDeployments(m, in.Available)) == 0 {
			excluded[m.ID] = []string{reasonNoDeployment}
			continue
		}
		usable = append(usable, m)
	}
	if len(usable) == 0 {
		d.Scores = mergeScores(allowed, excluded, nil)
		return d, openai.ErrNoHealthyDeployment()
	}

	choice, apiErr := in.Learned.Choose(usable, in)
	d.Relaxed, d.OOD, d.OODScore = choice.Relaxed, choice.OOD, choice.OODScore
	d.Explored, d.Propensity, d.Epsilon, d.Floor = choice.Explored, choice.Propensity, choice.Epsilon, choice.Floor
	d.Scores = mergeScores(allowed, excluded, choice.Scores)
	if apiErr != nil {
		return d, apiErr
	}
	return finish(d, choice.Order, in)
}

// mergeScores returns one ModelScore per allowed model, in catalog order: the policy's
// score when it made one, else an infeasible entry with the exclusion reasons.
func mergeScores(allowed []*catalog.Model, excluded map[string][]string, scored []ModelScore) []ModelScore {
	byModel := make(map[string]ModelScore, len(scored))
	for _, s := range scored {
		byModel[s.Model] = s
	}
	out := make([]ModelScore, 0, len(allowed))
	for _, m := range allowed {
		if s, ok := byModel[m.ID]; ok {
			out = append(out, s)
			continue
		}
		out = append(out, ModelScore{Model: m.ID, Reasons: excluded[m.ID]})
	}
	return out
}

// finish turns ordered feasible models into candidates, dropping models with no available
// deployment and models that cannot be admitted within the budget (each is costed with the
// output budget it can afford, see catalog.Model.AffordableK). It fails with
// insufficient_quota when a model was dropped for its cost and none is left, and with
// no_healthy_deployment when none is left otherwise.
func finish(d Decision, models []*catalog.Model, in Input) (Decision, *openai.APIError) {
	unaffordable := false
	for _, m := range models {
		deps := availableDeployments(m, in.Available)
		if len(deps) == 0 {
			continue
		}
		k, ok := m.AffordableK(in.Req, in.Tenant.DefaultMaxTokens, in.Budget)
		if !ok {
			unaffordable = true
			d.Rejected = append(d.Rejected, Rejection{Model: m.ID, Reasons: []catalog.Reason{catalog.Reason(budgetReason(m, in, k))}})
			continue
		}
		d.Candidates = append(d.Candidates, Candidate{
			Model:       m,
			K:           k,
			CostMax:     m.CostMax(in.Req, k),
			Deployments: deps,
		})
	}
	if len(d.Candidates) == 0 {
		if unaffordable {
			return d, openai.ErrInsufficientQuota(budgetMessage)
		}
		return d, openai.ErrNoHealthyDeployment()
	}
	return d, nil
}

// budgetMessage is the message of the 429 insufficient_quota a request gets when no model
// that could serve it fits its tenant's remaining budget.
const budgetMessage = "budget remaining below the cheapest worst-case cost"

// budgetReason says why m cannot be admitted within in.Budget, which must be set, at the
// output budget k that AffordableK returned.
func budgetReason(m *catalog.Model, in Input, k int) string {
	if cost := m.CostMax(in.Req, k); cost > *in.Budget {
		return fmt.Sprintf("budget: worst-case cost %d picousd above the remaining %d", cost, *in.Budget)
	}
	return fmt.Sprintf("budget: remaining %d picousd cannot pay for the input and one output token", *in.Budget)
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
