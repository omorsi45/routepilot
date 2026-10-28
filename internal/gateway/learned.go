package gateway

import (
	"errors"
	"net/http"
	"strings"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/router"
	"github.com/omorsi45/routepilot/internal/telemetry"
)

// Response headers that tell the client how a request was routed. The model and
// deployment headers are set where the response starts (setRouteHeaders); the others are
// set as soon as the router has answered, so an error after a decision carries them too.
const (
	headerPolicyVersion  = "x-routepilot-policy-version"
	headerDecisionID     = "x-routepilot-decision-id"
	headerRelaxed        = "x-routepilot-relaxed"
	headerEstimatedCost  = "x-routepilot-estimated-cost-usd"
	policyVersionPinned  = "pinned"
	policyVersionStatic  = "static-"
	decisionIDPrefix     = "dec_"
	upstreamRateLimitMsg = "upstream rate limited"
)

// decide asks the router where the request goes. For "auto" under the learned policy it
// takes one registry snapshot, picks the stable or canary slot for this request and routes
// with that slot's optimizer; the shadow slots of the same snapshot then decide on the
// same input, to be logged and never acted on. It records the slot role, the shadow
// decisions and the expected costs on rs.
func (s *Server) decide(rs *reqState, tenant *config.TenantConfig, req *openai.ChatCompletionRequest, reqs catalog.Requirements) (router.Decision, *openai.APIError) {
	in := router.Input{
		Tenant:    tenant,
		Requested: req.Model,
		Req:       reqs,
		Available: s.health.Available,
		Budget:    rs.budgetLeft,
	}
	if s.policies == nil || req.Model != config.ReservedModelAuto {
		return s.router.Decide(in)
	}

	snap := s.policies.Snapshot()
	slot := snap.Pick(tenant.ID, rs.id)
	rs.policyRole = slot.Role
	in.Learned = router.NewOptimizer(slot.Artifact, slot.Config, s.latency)
	in.Messages = features.Messages(req)
	if rp := req.RoutePilot; rp != nil {
		in.Floor, in.SLOTTFTMs, in.SLOE2EMs = rp.MinQuality, rp.SLOTTFTMs, rp.SLOE2EMs
	}
	in.Rand = s.rand

	dec, apiErr := s.router.Decide(in)
	s.decideShadows(rs, snap, in)
	s.noteLearnedDecision(rs, tenant.ID, slot.Role, dec, apiErr != nil)
	if len(dec.Candidates) > 0 {
		rs.estCost = make(map[string]money.PicoUSD, len(dec.Candidates))
		for _, c := range dec.Candidates {
			for _, sc := range dec.Scores {
				if sc.Model == c.Model.ID {
					rs.estCost[c.Model.ID] = sc.CostExp
				}
			}
		}
	}
	return dec, apiErr
}

// decideShadows runs every shadow slot of snap on in, after the primary decision. A shadow
// gets no Rand, so it never explores, and nothing it returns, an error included, reaches
// the client or changes the upstream call.
func (s *Server) decideShadows(rs *reqState, snap *policy.Snapshot, in router.Input) {
	for _, sh := range snap.Shadow {
		si := in
		si.Learned = router.NewOptimizer(sh.Artifact, sh.Config, s.latency)
		si.Rand = nil
		d, apiErr := s.router.Decide(si)
		rs.shadow = append(rs.shadow, d)
		rs.shadowErr = append(rs.shadowErr, apiErr)
	}
}

// setDecisionHeaders sets the headers a routed response carries, success or error, from the
// router's decision: the policy version, the decision ID, the constraints the policy
// relaxed (only when it relaxed any) and, for the learned policy, the expected cost of the
// chosen model.
func (s *Server) setDecisionHeaders(w http.ResponseWriter, rs *reqState, dec router.Decision) {
	h := w.Header()
	switch {
	case dec.PolicyVersion != "":
		rs.policyVersion = dec.PolicyVersion
	case dec.Policy == router.PolicyPinned:
		rs.policyVersion = policyVersionPinned
	case dec.Policy != "":
		rs.policyVersion = policyVersionStatic + dec.Policy
	}
	if rs.policyVersion != "" {
		h.Set(headerPolicyVersion, rs.policyVersion)
	}
	h.Set(headerDecisionID, decisionIDPrefix+rs.id)
	if len(dec.Relaxed) > 0 {
		h.Set(headerRelaxed, strings.Join(dec.Relaxed, ","))
	}
	if len(dec.Candidates) > 0 {
		if cost, ok := rs.estCost[dec.Candidates[0].Model.ID]; ok {
			h.Set(headerEstimatedCost, cost.USD())
		}
	}
}

// setRouteHeaders tells the client which model and deployment served it. When a failover
// moved the request off the model the policy chose, the estimated cost header is changed
// to that of the model that served.
func (rs *reqState) setRouteHeaders(w http.ResponseWriter, modelID, deployment string) {
	h := w.Header()
	h.Set("x-routepilot-model", modelID)
	h.Set("x-routepilot-deployment", deployment)
	if cost, ok := rs.estCost[modelID]; ok {
		h.Set(headerEstimatedCost, cost.USD())
	}
}

// noteUpstreamError records an attempt's failure for the upstream-429 passthrough: it
// counts the attempts that ended in an HTTP 429 and keeps the smallest positive
// Retry-After among them.
func (rs *reqState) noteUpstreamError(err error) {
	var perr *provider.Error
	if !errors.As(err, &perr) || perr.Kind != provider.KindHTTP || perr.Status != http.StatusTooManyRequests {
		return
	}
	rs.upstream429++
	if perr.RetryAfter > 0 && (rs.retryAfter == 0 || perr.RetryAfter < rs.retryAfter) {
		rs.retryAfter = perr.RetryAfter
	}
}

// allUpstream429 reports whether at least one attempt was sent and every one of them ended
// in an upstream 429.
func (rs *reqState) allUpstream429() bool {
	return rs.attempts > 0 && rs.upstream429 == rs.attempts
}

// failUpstreamRateLimited answers 429 rate_limit_exceeded for a request whose every
// attempt was rate limited upstream. Retry-After is the smallest one an upstream sent, in
// whole seconds rounded up and at least 1, and is left out when none carried one.
func (s *Server) failUpstreamRateLimited(w http.ResponseWriter, rs *reqState) {
	rs.limitKind = telemetry.RateLimitUpstream
	if rs.retryAfter > 0 {
		w.Header().Set("Retry-After", retryAfterSeconds(rs.retryAfter))
	}
	s.fail(w, rs, openai.ErrRateLimited(upstreamRateLimitMsg))
}
