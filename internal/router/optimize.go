package router

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
)

// PolicyLearned is the Decision.Policy value and the Optimizer's name.
const PolicyLearned = "learned"

// sloDelta is the miss probability a latency SLO is checked at: the p95.
const sloDelta = 0.05

const (
	reasonNoDeployment = "no available deployment"
	reasonNotInPolicy  = "not in the policy artifact"
)

// Learned is a routing policy learned from data. It replaces the static policy for the
// "auto" model: the router hands it the models that may serve the request and it returns
// the models to try, in order, with the evidence behind the choice.
type Learned interface {
	// Name is the policy name reported in Decision.Policy.
	Name() string
	// Version identifies the exact policy, the artifact and its configuration.
	Version() string
	// Choose picks among models, which the router has already checked for tenant
	// permission, capabilities, limits and an available deployment. It returns a 422
	// no_feasible_model, a 429 insufficient_quota or a 400 model_capability_unavailable
	// when it cannot pick; the Choice still carries the scores it computed.
	Choose(models []*catalog.Model, in Input) (Choice, *openai.APIError)
}

// Choice is a Learned policy's answer.
type Choice struct {
	// Order is the models to try: the choice first, then the failover candidates.
	Order []*catalog.Model
	// Relaxed names the constraints that were given up to answer anyway: "latency"
	// and/or "quality". It is empty for a decision that met every constraint.
	Relaxed []string
	// OOD and OODScore say whether the request looked unlike the training data.
	OOD      bool
	OODScore float64
	// Explored is true when the choice came from the random branch of epsilon-greedy
	// exploration. Propensity is the probability the policy had of making this choice
	// (1 when it did not explore) and Epsilon the exploration rate in effect (0 when the
	// decision was not eligible for exploration).
	Explored   bool
	Propensity float64
	Epsilon    float64
	// Floor is the quality floor the request was held to.
	Floor float64
	// Scores has one entry per model passed to Choose, in that order.
	Scores []ModelScore
}

// ModelScore is the evidence about one model for one request.
type ModelScore struct {
	Model string
	// P is the calibrated success probability and L its lower confidence bound, as the
	// policy used them (the pooled values for an out-of-distribution request under the
	// conservative action).
	P, L float64
	// CostExp is the expected cost and CostMax the worst-case cost of the request on
	// this model.
	CostExp, CostMax money.PicoUSD
	// QMs is the p95 latency in milliseconds checked against the SLO, nil when no SLO
	// was set or no estimate exists.
	QMs *float64
	// Feasible is true when the model is in the safe set: within the budget, at or
	// above the quality floor, inside the SLO. Reasons says what excluded it otherwise.
	Feasible bool
	Reasons  []string
}

// Scorer predicts how each model would do on a request. *policy.Artifact is one.
type Scorer interface {
	Version() string
	HasModel(id string) bool
	Score(msgs []features.Message) policy.Scores
}

// Optimizer is the constrained optimizer over a Scorer: the cheapest model (or the best
// trade of success against cost) whose lower confidence bound clears the quality floor,
// inside the budget and the latency SLO. It is safe for concurrent use.
type Optimizer struct {
	scorer Scorer
	cfg    policy.Config
	est    *latency.Estimator
}

// NewOptimizer returns an Optimizer. A nil est means there are no latency estimates, so a
// request with an SLO fails it on every model ("no latency estimate").
func NewOptimizer(s Scorer, cfg policy.Config, est *latency.Estimator) *Optimizer {
	return &Optimizer{scorer: s, cfg: cfg, est: est}
}

// Name implements Learned.
func (o *Optimizer) Name() string { return PolicyLearned }

// Version is the scorer's version and the first 8 hex digits of the configuration hash.
func (o *Optimizer) Version() string { return o.scorer.Version() + "+" + o.cfg.Hash8() }

// optCand is one model under evaluation.
type optCand struct {
	m    *catalog.Model
	s    ModelScore
	rank int
	obj  float64 // dual objective

	budgetOK, floorOK, sloOK bool
	// End-to-end and time-to-first-token p95, nil without an SLO or an estimate, and
	// whether the estimate takes a value from the simulated prior profile.
	qE2E, qTTFT         *float64
	priorE2E, priorTTFT bool
}

// Choose implements Learned. The selection follows Ruling V1-R10:
//
//  1. The budget set keeps the models that can be admitted within the remaining budget: with
//     no explicit max_tokens each model gets the output budget the budget pays for (see
//     catalog.Model.AffordableK), and is out if not even one output token fits; with an
//     explicit cap its worst case must fit. If the set is empty the request fails with 429
//     insufficient_quota and nothing is relaxed.
//  2. The safe set S keeps those whose lower bound clears the floor and whose p95 meets
//     the SLO (a missing estimate fails it).
//  3. If S is not empty, min_cost takes its cheapest expected cost and dual its highest
//     P - lambda * cost; an out-of-distribution request under the strongest action takes
//     the highest rank. Ties break on a higher L, a lower p95 and the smaller ID.
//  4. If S is empty, reject answers 422 no_feasible_model naming the binding constraints.
//  5. best_effort relaxes latency first (keeping the floor), then quality (the highest L).
//  6. Only a decision that relaxed nothing explores: with probability epsilon it takes a
//     uniformly random model of S, and Propensity records the probability of the choice.
//  7. The failover order is the choice, the rest of S, then (best_effort) the rest of the
//     budget set by lower bound. A model outside the budget set is never a candidate.
func (o *Optimizer) Choose(models []*catalog.Model, in Input) (Choice, *openai.APIError) {
	floor := o.floor(in)
	ch := Choice{Floor: floor, Propensity: 1}
	scores := o.scorer.Score(in.Messages)
	ch.OOD, ch.OODScore = scores.OOD, scores.OODScore

	cands := make([]*optCand, 0, len(models))
	var covered []*optCand
	for _, m := range models {
		sc, ok := scores.Models[m.ID]
		if !ok || !o.scorer.HasModel(m.ID) {
			cands = append(cands, &optCand{m: m, s: ModelScore{Model: m.ID, Reasons: []string{reasonNotInPolicy}}})
			continue
		}
		c := o.evaluate(m, sc, scores.OOD, floor, in)
		cands = append(cands, c)
		covered = append(covered, c)
	}
	done := func(err *openai.APIError) (Choice, *openai.APIError) {
		ch.Scores = make([]ModelScore, len(cands))
		for i, c := range cands {
			ch.Scores[i] = c.s
		}
		return ch, err
	}

	if len(covered) == 0 {
		return done(openai.ErrCapabilityUnavailable("no model in the routing policy can serve this request"))
	}
	var bset, safe []*optCand
	for _, c := range covered {
		if c.budgetOK {
			bset = append(bset, c)
		}
	}
	if len(bset) == 0 {
		return done(openai.ErrInsufficientQuota(budgetMessage))
	}
	for _, c := range bset {
		if c.floorOK && c.sloOK {
			c.s.Feasible = true
			safe = append(safe, c)
		}
	}

	useRank := scores.OOD && o.cfg.OODAction == policy.OODStrongest
	onInfeasible := o.onInfeasible(in)
	sloSet := in.SLOTTFTMs != nil || in.SLOE2EMs != nil

	var primary []*optCand
	switch {
	case len(safe) > 0:
		primary = slices.Clone(safe)
		slices.SortFunc(primary, o.byObjective(useRank))
	case onInfeasible == config.OnInfeasibleReject:
		return done(openai.ErrNoFeasibleModel(o.infeasibleMessage(bset, floor, in)))
	default:
		if sloSet { // keep the floor, give up the latency SLO
			var s1 []*optCand
			for _, c := range bset {
				if c.floorOK {
					s1 = append(s1, c)
				}
			}
			if len(s1) > 0 {
				primary = s1
				slices.SortFunc(primary, o.byObjective(useRank))
				ch.Relaxed = []string{"latency"}
			}
		}
		if primary == nil { // give up the floor too: the most reliable model in the budget
			primary = slices.Clone(bset)
			slices.SortFunc(primary, byLowerBound(useRank))
			ch.Relaxed = []string{"quality"}
			if sloSet {
				ch.Relaxed = []string{"latency", "quality"}
			}
		}
	}

	pick := 0
	if len(safe) > 0 && len(ch.Relaxed) == 0 && !useRank && in.Rand != nil {
		eps := o.cfg.Epsilon
		ch.Epsilon = eps
		n := len(primary)
		if eps > 0 && in.Rand() < eps {
			ch.Explored = true
			pick = min(max(int(in.Rand()*float64(n)), 0), n-1)
		}
		ch.Propensity = eps / float64(n)
		if pick == 0 {
			ch.Propensity += 1 - eps
		}
	}

	order := make([]*optCand, 0, len(bset))
	order = append(order, primary[pick])
	for i, c := range primary {
		if i != pick {
			order = append(order, c)
		}
	}
	if onInfeasible != config.OnInfeasibleReject {
		var rest []*optCand
		for _, c := range bset {
			if !slices.Contains(primary, c) {
				rest = append(rest, c)
			}
		}
		slices.SortFunc(rest, byLowerBound(useRank))
		order = append(order, rest...)
	}
	for _, c := range order {
		ch.Order = append(ch.Order, c.m)
	}
	return done(nil)
}

// floor is the quality floor of the request: its own, else the tenant's, else the policy's.
func (o *Optimizer) floor(in Input) float64 {
	switch {
	case in.Floor != nil:
		return *in.Floor
	case in.Tenant.MinQuality != nil:
		return *in.Tenant.MinQuality
	default:
		return o.cfg.DefaultFloor
	}
}

// onInfeasible is the tenant's setting, else the policy's.
func (o *Optimizer) onInfeasible(in Input) string {
	if in.Tenant.OnInfeasible != "" {
		return in.Tenant.OnInfeasible
	}
	return o.cfg.OnInfeasible
}

// evaluate scores one model that the scorer covers.
func (o *Optimizer) evaluate(m *catalog.Model, sc policy.ModelScore, ood bool, floor float64, in Input) *optCand {
	p, l := sc.P, sc.L
	if ood && o.cfg.OODAction == policy.OODConservative {
		p, l = sc.PooledRate, sc.PooledLCB
	}
	if math.IsNaN(p) {
		p = 0
	}
	if math.IsNaN(l) {
		l = 0
	}

	k, affordable := m.AffordableK(in.Req, in.Tenant.DefaultMaxTokens, in.Budget)
	ratio := sc.InRatio
	if !(ratio > 0) { // also catches NaN: no usable ratio means no reduction
		ratio = 1
	}
	inEst := ceilTokens(float64(m.InUpper(in.Req)) * ratio)
	outEst := ceilTokens(clampOut(sc.EOut, k))
	c := &optCand{
		m: m, rank: m.Rank,
		s: ModelScore{
			Model: m.ID, P: p, L: l,
			CostExp: money.AddSat(money.MulSat(m.PriceIn, inEst), money.MulSat(m.PriceOut, outEst)),
			CostMax: m.CostMax(in.Req, k),
		},
	}
	c.obj = p - o.cfg.Lambda*float64(c.s.CostExp)/1e12

	c.budgetOK = affordable
	if !c.budgetOK {
		c.s.Reasons = append(c.s.Reasons, budgetReason(m, in, k))
	}
	c.floorOK = l >= floor
	if !c.floorOK {
		c.s.Reasons = append(c.s.Reasons, fmt.Sprintf("quality: lower bound %.4f below the floor %.4f", l, floor))
	}

	c.sloOK = true
	if in.SLOE2EMs != nil || in.SLOTTFTMs != nil {
		deps := deploymentNames(availableDeployments(m, in.Available))
		if in.SLOE2EMs != nil {
			if q, ok := latency.E2EQuantile(o.est, m.ID, deps, clampOut(sc.OutQ[3], k), sloDelta); ok {
				c.qE2E, c.priorE2E = &q, o.est.UsesPrior(m.ID, deps, true)
			}
			c.sloOK = c.checkSLO(c.qE2E, *in.SLOE2EMs, "latency") && c.sloOK
		}
		if in.SLOTTFTMs != nil {
			if q, ok := latency.TTFTQuantile(o.est, m.ID, deps, sloDelta); ok {
				c.qTTFT, c.priorTTFT = &q, o.est.UsesPrior(m.ID, deps, false)
			}
			c.sloOK = c.checkSLO(c.qTTFT, *in.SLOTTFTMs, "ttft") && c.sloOK
		}
		if in.SLOE2EMs != nil {
			c.s.QMs = c.qE2E
		} else {
			c.s.QMs = c.qTTFT
		}
	}
	return c
}

// checkSLO reports whether the p95 q meets the SLO and records the reason when not.
func (c *optCand) checkSLO(q *float64, sloMs int, name string) bool {
	switch {
	case q == nil:
		c.s.Reasons = append(c.s.Reasons, fmt.Sprintf("%s: no latency estimate for the slo of %d ms", name, sloMs))
		return false
	case *q > float64(sloMs):
		c.s.Reasons = append(c.s.Reasons, fmt.Sprintf("%s: p95 %.0f ms above the slo of %d ms", name, math.Ceil(*q), sloMs))
		return false
	}
	return true
}

func deploymentNames(deps []catalog.Deployment) []string {
	names := make([]string, len(deps))
	for i, d := range deps {
		names[i] = d.Name
	}
	return names
}

// clampOut limits an expected or quantile output length to [1, k] tokens. A NaN counts as
// the whole budget.
func clampOut(tokens float64, k int) float64 {
	hi := float64(max(k, 1))
	if math.IsNaN(tokens) {
		return hi
	}
	return math.Min(math.Max(tokens, 1), hi)
}

// ceilTokens rounds a token estimate up to a whole number, saturating.
func ceilTokens(x float64) int64 {
	switch {
	case !(x > 0):
		return 0
	case x >= 9.2e18:
		return math.MaxInt64
	}
	return int64(math.Ceil(x))
}

// compareQ orders latency estimates, lower first, with a missing estimate last.
func compareQ(a, b *float64) int {
	switch {
	case a == nil && b == nil:
		return 0
	case a == nil:
		return 1
	case b == nil:
		return -1
	}
	return cmp.Compare(*a, *b)
}

// byObjective orders the candidates best first for the mode: lowest expected cost for
// min_cost, highest P - lambda * cost for dual, and the highest rank for an
// out-of-distribution request under the strongest action. Ties break on a higher lower
// bound, a lower p95 and the smaller ID.
func (o *Optimizer) byObjective(useRank bool) func(a, b *optCand) int {
	return func(a, b *optCand) int {
		switch {
		case useRank:
			if c := cmp.Compare(b.rank, a.rank); c != 0 {
				return c
			}
			if c := cmp.Compare(a.s.CostExp, b.s.CostExp); c != 0 {
				return c
			}
			return strings.Compare(a.m.ID, b.m.ID)
		case o.cfg.Mode == policy.ModeDual:
			if c := cmp.Compare(b.obj, a.obj); c != 0 {
				return c
			}
		default:
			if c := cmp.Compare(a.s.CostExp, b.s.CostExp); c != 0 {
				return c
			}
		}
		if c := cmp.Compare(b.s.L, a.s.L); c != 0 {
			return c
		}
		if c := compareQ(a.s.QMs, b.s.QMs); c != 0 {
			return c
		}
		return strings.Compare(a.m.ID, b.m.ID)
	}
}

// byLowerBound orders the candidates by the highest lower bound, ties on a lower p95 and
// the smaller ID; with useRank by the highest rank, ties on a lower expected cost.
func byLowerBound(useRank bool) func(a, b *optCand) int {
	return func(a, b *optCand) int {
		if useRank {
			if c := cmp.Compare(b.rank, a.rank); c != 0 {
				return c
			}
			if c := cmp.Compare(a.s.CostExp, b.s.CostExp); c != 0 {
				return c
			}
			return strings.Compare(a.m.ID, b.m.ID)
		}
		if c := cmp.Compare(b.s.L, a.s.L); c != 0 {
			return c
		}
		if c := compareQ(a.s.QMs, b.s.QMs); c != 0 {
			return c
		}
		return strings.Compare(a.m.ID, b.m.ID)
	}
}

// infeasibleMessage names the constraints that leave nothing to choose from the budget
// set: a floor above every lower bound and an SLO below every p95, or, when each is met
// by some model but not by the same one, both together.
func (o *Optimizer) infeasibleMessage(bset []*optCand, floor float64, in Input) string {
	var parts, slos []string
	bestL := math.Inf(-1)
	for _, c := range bset {
		bestL = math.Max(bestL, c.s.L)
	}
	if bestL < floor {
		parts = append(parts, fmt.Sprintf("quality floor %.2f above best lower bound %.2f", floor, bestL))
	}

	type sloSpec struct {
		name  string
		ms    *int
		q     func(*optCand) *float64
		prior func(*optCand) bool
	}
	for _, s := range []sloSpec{
		{"latency", in.SLOE2EMs, func(c *optCand) *float64 { return c.qE2E }, func(c *optCand) bool { return c.priorE2E }},
		{"ttft", in.SLOTTFTMs, func(c *optCand) *float64 { return c.qTTFT }, func(c *optCand) bool { return c.priorTTFT }},
	} {
		if s.ms == nil {
			continue
		}
		slos = append(slos, fmt.Sprintf("%s slo %d ms", s.name, *s.ms))
		var best *optCand
		for _, c := range bset {
			if q := s.q(c); q != nil && (best == nil || *q < *s.q(best)) {
				best = c
			}
		}
		switch {
		case best == nil:
			parts = append(parts, fmt.Sprintf("%s slo %d ms with no latency estimate", s.name, *s.ms))
		case *s.q(best) > float64(*s.ms):
			part := fmt.Sprintf("%s slo %d ms below best p95 %.0f ms", s.name, *s.ms, math.Ceil(*s.q(best)))
			if s.prior(best) {
				part += " (simulated profile)"
			}
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		parts = []string{fmt.Sprintf("no model meets quality floor %.2f and %s together", floor, strings.Join(slos, " and "))}
	}
	return "no feasible model: " + strings.Join(parts, "; ")
}
