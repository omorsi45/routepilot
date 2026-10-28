package gateway

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/router"
)

// Actors named in the rollout events the gateway writes.
const (
	actorGuardrail = "guardrail"
	actorAdmin     = "admin"
)

// Limits of the decision index (Ruling V1-R31): the guard's window is ten minutes, and a
// verdict on an older decision says nothing about the canary as it is now.
const (
	decisionIndexMax = 100_000
	decisionIndexTTL = 10 * time.Minute
)

// initGuard builds the guardrail state of a gateway that routes with the learned policy: the
// guard (on the server's clock), the index from decision IDs to the slot that decided them,
// and the identity of the canary the guard is watching right now.
func (s *Server) initGuard() {
	s.guard = policy.NewGuard(s.now)
	s.index = newDecisionIndex(s.now)
	if snap := s.policies.Snapshot(); canaryActive(snap) {
		s.guardKey = guardKey(snap)
	}
}

func canaryActive(snap *policy.Snapshot) bool { return snap.Canary != nil && snap.CanaryPercent > 0 }

// guardKey identifies the pair the guard compares: a different stable or canary policy is a
// different comparison, and starts from an empty window.
func guardKey(snap *policy.Snapshot) string { return snap.Stable.Version + "|" + snap.Canary.Version }

// noteLearnedDecision records what the guardrails need of a learned decision: whether the
// router refused it and whether it relaxed a constraint (both read when the request ends), and
// the decision's slot in the index feedback is matched against. Only a decision that picked a
// model is indexed; a refusal has no answer a client could give feedback on. Without a
// decision log there is no feedback route, so nothing is indexed.
func (s *Server) noteLearnedDecision(rs *reqState, tenantID, role string, dec router.Decision, refused bool) {
	rs.routeFailed = refused
	rs.relaxed = len(dec.Relaxed) > 0
	if !refused && s.decisions != nil {
		s.index.add(decisionIDPrefix+rs.id, tenantID, role)
	}
}

// observeGuard tells the guard how a learned request ended. It runs once the handler is done
// and the request has its final status. Pinned and static requests, and requests refused
// before the router (a bad key, a malformed body), have no role and are not observed.
//
// A request failed on the policy's side when it got a server error, or when the router
// refused it with a 4xx other than 400: the 400s a router answers are the client's mistakes
// (a capability no model has, for one), and not something to blame on a policy.
func (s *Server) observeGuard(sw *statusWriter, rs *reqState, start time.Time) {
	if s.guard == nil || (rs.policyRole != policy.RoleStable && rs.policyRole != policy.RoleCanary) {
		return
	}
	status := finalStatus(sw, rs)
	s.guard.Observe(rs.policyRole, policy.Observation{
		Error:   status >= http.StatusInternalServerError || (rs.routeFailed && status >= http.StatusBadRequest && status != http.StatusBadRequest),
		Relaxed: rs.relaxed,
		Latency: s.now().Sub(start),
	})
}

// feedbackToGuard passes a client's verdict to the guard when it is about a learned decision
// of the same tenant that is still inside the window. The verdict of a decision counts once.
// Anything else, an unknown or expired decision or one of another tenant, is only in the
// decision log, where the feedback handler already wrote it.
func (s *Server) feedbackToGuard(tenantID string, in feedbackInput) {
	if s.index == nil {
		return
	}
	role, ok := s.index.take(in.decisionID, tenantID)
	if !ok {
		return
	}
	success := in.success
	s.guard.Observe(role, policy.Observation{Feedback: &success})
}

// EvaluateGuardrails compares the canary with the stable policy and, when the guard trips,
// pulls the canary: its share goes to 0 percent, the event is written to the usage ledger
// (when there is one), the rollback is counted and the reason is logged. serve calls it every
// ten seconds, and tests call it directly.
//
// It does nothing while the canary has no share of the traffic. When the canary or the stable
// policy changes, or the canary's share is raised again after a rollback, the guard starts
// from an empty window, so a new canary is not judged on an old one's requests. A canary
// that has been rolled back is reported once.
func (s *Server) EvaluateGuardrails() {
	if s.policies == nil || s.guard == nil {
		return
	}
	s.guardMu.Lock()
	defer s.guardMu.Unlock()
	snap := s.policies.Snapshot()
	if !canaryActive(snap) {
		s.guardKey = ""
		return
	}
	if key := guardKey(snap); key != s.guardKey {
		s.guard.Reset()
		s.guardKey = key
	}
	trip, reason := s.guard.Evaluate()
	if !trip {
		return
	}
	// The guard is tripped and stays quiet until it is reset. Forget the pair it was
	// comparing, so the next evaluation that finds the canary with a share starts over.
	s.guardKey = ""
	old := s.policies.RollbackCanary()
	if old <= 0 { // an operator stopped the canary first
		return
	}
	s.recordRollout(accounting.RolloutEvent{
		At: s.now(), Action: accounting.ActionAutoRollback,
		From: snap.Canary.Version, To: snap.Stable.Version,
		Percent: 0, Reason: reason, Actor: actorGuardrail,
	})
	s.logger.LogAttrs(context.Background(), slog.LevelWarn, "guardrails rolled back the canary",
		slog.String("reason", reason),
		slog.String("canary", snap.Canary.Version),
		slog.String("stable", snap.Stable.Version),
		slog.Float64("previous_percent", old),
	)
}

// disarmGuard forgets the canary pair the guard was comparing. The next evaluation that finds
// the canary with a share starts from an empty window, whatever happened to the share in
// between. A manual rollback calls it, because the canary can be raised again before the next
// evaluation has seen it at 0.
func (s *Server) disarmGuard() {
	s.guardMu.Lock()
	s.guardKey = ""
	s.guardMu.Unlock()
}

// recordRollout writes a rollout event to the usage ledger, if there is one. A failure never
// stops the rollback it describes; it is counted with the other lost ledger writes and logged.
func (s *Server) recordRollout(ev accounting.RolloutEvent) {
	if s.usageLedger == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), usageWriteTimeout)
	defer cancel()
	if err := s.usageLedger.RecordRollout(ctx, ev); err != nil {
		s.accountingErrors.Add(1)
		s.logger.LogAttrs(context.Background(), slog.LevelWarn, "rollout event write failed",
			slog.String("action", ev.Action), slog.String("error", err.Error()))
	}
}

// decisionIndex maps the decision IDs of learned decisions to the tenant they were made for
// and the role of the slot that made them, so a feedback line can be credited to the right
// side of the comparison. It holds at most decisionIndexMax entries and drops those older than
// decisionIndexTTL, whichever limit is reached first. It is safe for concurrent use.
type decisionIndex struct {
	now func() time.Time

	mu      sync.Mutex
	entries map[string]indexEntry
	// queue lists the IDs in insertion order, from head. It may name entries that were
	// taken, which are skipped when they reach the front.
	queue []indexKey
	head  int
}

type indexEntry struct {
	tenant, role string
	at           time.Time
}

type indexKey struct {
	id string
	at time.Time
}

func newDecisionIndex(now func() time.Time) *decisionIndex {
	return &decisionIndex{now: now, entries: make(map[string]indexEntry)}
}

// add records a decision.
func (x *decisionIndex) add(id, tenant, role string) {
	now := x.now()
	x.mu.Lock()
	defer x.mu.Unlock()
	x.expire(now)
	x.entries[id] = indexEntry{tenant: tenant, role: role, at: now}
	x.queue = append(x.queue, indexKey{id: id, at: now})
	for len(x.queue)-x.head > decisionIndexMax {
		x.pop()
	}
	x.compact()
}

// take returns the role of decision id when it is in the index, is not older than
// decisionIndexTTL and was made for tenant, and removes it, so a decision is credited once.
// Another tenant's attempt leaves the entry alone.
func (x *decisionIndex) take(id, tenant string) (role string, ok bool) {
	now := x.now()
	x.mu.Lock()
	defer x.mu.Unlock()
	e, found := x.entries[id]
	if !found {
		return "", false
	}
	if now.Sub(e.at) > decisionIndexTTL {
		delete(x.entries, id)
		return "", false
	}
	if e.tenant != tenant {
		return "", false
	}
	delete(x.entries, id)
	return e.role, true
}

// size is the number of decisions that can still be taken.
func (x *decisionIndex) size() int {
	x.mu.Lock()
	defer x.mu.Unlock()
	return len(x.entries)
}

// expire drops the decisions older than decisionIndexTTL from the front of the queue.
func (x *decisionIndex) expire(now time.Time) {
	for x.head < len(x.queue) && now.Sub(x.queue[x.head].at) > decisionIndexTTL {
		x.pop()
	}
}

// pop removes the oldest queue entry and its decision, unless the ID has been reused since.
func (x *decisionIndex) pop() {
	k := x.queue[x.head]
	x.queue[x.head] = indexKey{}
	x.head++
	if e, ok := x.entries[k.id]; ok && e.at.Equal(k.at) {
		delete(x.entries, k.id)
	}
}

// compact moves the live part of the queue to its start once at least half of it is spent, so
// the backing array does not grow without bound.
func (x *decisionIndex) compact() {
	const minSpent = 4096
	if x.head < minSpent || x.head*2 < len(x.queue) {
		return
	}
	n := copy(x.queue, x.queue[x.head:])
	clear(x.queue[n:])
	x.queue, x.head = x.queue[:n], 0
}
