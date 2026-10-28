package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/decisionlog"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
)

// Values of the policy_role field of a decision record.
const (
	roleShadow = "shadow"
	rolePinned = "pinned"
	roleStatic = "static"
)

// DecisionLogDropped is the number of decision log records dropped because the writer's queue
// was full. The metrics export it as rp_decisionlog_dropped_total. It is 0 without a log.
func (s *Server) DecisionLogDropped() int64 {
	if s.decisions == nil {
		return 0
	}
	return s.decisions.Dropped()
}

// DecisionLogErrors is the number of decision log records lost to an encoding or file error.
// The metrics export it as rp_decisionlog_errors_total. It is 0 without a log.
func (s *Server) DecisionLogErrors() int64 {
	if s.decisions == nil {
		return 0
	}
	return s.decisions.Errors()
}

// noteDecision keeps what the decision log needs from a request that has reached the router:
// the decision, when it was made, the capabilities the request needs, and the prompt hash.
// The prompt itself is not kept, only its hash prefix, and nothing is kept without a log.
func (s *Server) noteDecision(rs *reqState, req *openai.ChatCompletionRequest, reqs catalog.Requirements, dec router.Decision) {
	if s.decisions == nil {
		return
	}
	rs.routed = true
	rs.decision = dec
	rs.decidedAt = s.now()
	rs.required = reqs.Caps.Names()
	rs.promptHash16 = promptHash16(req)
}

// promptHash16 is the first 16 hex digits of the SHA-256 of the message texts joined by
// newlines, the same bytes the feature extractor reads as the full text.
func promptHash16(req *openai.ChatCompletionRequest) string {
	h := sha256.New()
	for i, m := range req.Messages {
		if i > 0 {
			_, _ = h.Write([]byte{'\n'})
		}
		_, _ = io.WriteString(h, m.Content.PlainText())
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// logDecisions writes the decision records of a request that reached the router: the primary
// decision with its outcome, then one record per shadow decision without one. It runs after
// the handler, so the status, the attempts and their costs are final.
func (s *Server) logDecisions(sw *statusWriter, rs *reqState, start time.Time) {
	if s.decisions == nil || !rs.routed {
		return
	}
	primary := s.decisionRecord(rs, rs.decision, rs.policyVersion)
	primary.Outcome = s.decisionOutcome(sw, rs, start)
	s.decisions.Write(primary)
	for _, d := range rs.shadow {
		shadow := s.decisionRecord(rs, d, d.PolicyVersion)
		shadow.PolicyRole = roleShadow
		s.decisions.Write(shadow)
	}
}

// decisionRecord is the decision half of a record for d, without an outcome. A static or
// pinned decision has no probabilities, so the fields only a learned policy fills are null.
func (s *Server) decisionRecord(rs *reqState, d router.Decision, version string) decisionlog.Record {
	rec := decisionlog.Record{
		Type: "decision", DecisionID: decisionIDPrefix + rs.id, Time: rs.decidedAt.UTC(), Tenant: rs.tenant,
		PolicyVersion: version, PolicyRole: policyRole(rs, d),
		PromptSHA16: rs.promptHash16, PromptBytes: rs.promptBytes, Required: rs.required,
		Candidates: decisionCandidates(d), Relaxed: append([]string{}, d.Relaxed...),
	}
	if len(d.Candidates) > 0 {
		rec.Chosen = d.Candidates[0].Model.ID
	}
	if d.PolicyVersion == "" { // not the learned policy
		return rec
	}
	rec.OODScore, rec.OOD = finite(d.OODScore), &d.OOD
	if rec.Chosen != "" { // no action was taken otherwise, so there is no propensity of it
		rec.Propensity, rec.Epsilon, rec.Explored = finite(d.Propensity), finite(d.Epsilon), &d.Explored
	}
	return rec
}

// policyRole names who decided: the learned slot (stable or canary), a static policy, or the
// client, who named the model.
func policyRole(rs *reqState, d router.Decision) string {
	switch {
	case d.PolicyVersion != "":
		return rs.policyRole
	case d.Policy == router.PolicyPinned:
		return rolePinned
	}
	return roleStatic
}

// decisionCandidates lists the models the policy considered. The learned policy scored every
// model the tenant may use; a static or pinned decision has the candidates it will try, with
// their worst-case cost, and the models it ruled out, with the reasons.
func decisionCandidates(d router.Decision) []decisionlog.Candidate {
	out := make([]decisionlog.Candidate, 0, len(d.Scores)+len(d.Candidates)+len(d.Rejected))
	if d.PolicyVersion != "" {
		for _, sc := range d.Scores {
			c := decisionlog.Candidate{Model: sc.Model, Feasible: sc.Feasible, Reasons: append([]string{}, sc.Reasons...)}
			// A model ruled out before it was scored has nothing but its reasons.
			if sc.Feasible || sc.P != 0 || sc.L != 0 || sc.CostExp != 0 || sc.CostMax != 0 || sc.QMs != nil {
				c.P, c.L = finite(sc.P), finite(sc.L)
				c.CostExpPicoUSD, c.CostMaxPicoUSD = pico(sc.CostExp), pico(sc.CostMax)
				if sc.QMs != nil {
					c.QMs = finite(*sc.QMs)
				}
			}
			out = append(out, c)
		}
		return out
	}
	for _, c := range d.Candidates {
		out = append(out, decisionlog.Candidate{Model: c.Model.ID, CostMaxPicoUSD: pico(c.CostMax), Feasible: true, Reasons: []string{}})
	}
	for _, r := range d.Rejected {
		reasons := make([]string, len(r.Reasons))
		for i, reason := range r.Reasons {
			reasons[i] = string(reason)
		}
		out = append(out, decisionlog.Candidate{Model: r.Model, Feasible: false, Reasons: reasons})
	}
	return out
}

// decisionOutcome is what came of the request: the status the client got, the time it took,
// every attempt and what they cost together, and the tokens of the attempt that answered.
func (s *Server) decisionOutcome(sw *statusWriter, rs *reqState, start time.Time) *decisionlog.Outcome {
	o := &decisionlog.Outcome{
		Status:            finalStatus(sw, rs),
		LatencyMs:         s.now().Sub(start).Milliseconds(),
		CostActualPicoUSD: int64(rs.settleAttempts()),
		Attempts:          []decisionlog.Attempt{},
	}
	if !rs.commitAt.IsZero() {
		ttft := rs.commitAt.Sub(start).Milliseconds()
		o.TTFTMs = &ttft
	}
	for _, a := range rs.attemptUsages {
		status := a.status
		if status == "" {
			status = statusAborted
		}
		o.Attempts = append(o.Attempts, decisionlog.Attempt{Deployment: a.Deployment, Outcome: status})
		if status == statusOK {
			o.PromptTokens, o.CompletionTokens, o.UsageEstimated = a.Prompt, a.Completion, a.Estimated
		}
	}
	return o
}

// finite returns &f, or nil for NaN and the infinities, which JSON cannot hold.
func finite(f float64) *float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return &f
}

func pico(m money.PicoUSD) *int64 {
	v := int64(m)
	return &v
}
