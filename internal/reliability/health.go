package reliability

import (
	"math"
	"sync"
	"time"
)

// DeploymentHealth is a point-in-time view of one deployment.
type DeploymentHealth struct {
	State     State
	ErrorRate float64
}

type healthEntry struct {
	breaker *Breaker

	mu   sync.Mutex // guards the EWMA fields
	rate float64
	last time.Time
	seen bool
}

// Health tracks a circuit breaker and an EWMA error rate per deployment.
// Entries are created lazily. It is safe for concurrent use.
type Health struct {
	now func() time.Time

	mu      sync.RWMutex
	entries map[string]*healthEntry
}

// NewHealth returns an empty registry that reads time from now.
func NewHealth(now func() time.Time) *Health {
	return &Health{now: now, entries: make(map[string]*healthEntry)}
}

func (h *Health) entry(deployment string) *healthEntry {
	h.mu.RLock()
	e, ok := h.entries[deployment]
	h.mu.RUnlock()
	if ok {
		return e
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if e, ok = h.entries[deployment]; ok {
		return e
	}
	e = &healthEntry{breaker: NewBreaker(h.now)}
	h.entries[deployment] = e
	return e
}

// lookup returns the entry of a deployment seen before, without creating one.
func (h *Health) lookup(deployment string) (*healthEntry, bool) {
	h.mu.RLock()
	e, ok := h.entries[deployment]
	h.mu.RUnlock()
	return e, ok
}

// Available reports whether the deployment would admit an attempt, without
// claiming anything. It never creates an entry.
func (h *Health) Available(deployment string) bool {
	h.mu.RLock()
	e, ok := h.entries[deployment]
	h.mu.RUnlock()
	if !ok {
		return true
	}
	return e.breaker.Available()
}

// Allow claims an attempt on the deployment. The Ticket it returns is what the caller hands
// back to Record or Abort, exactly once; a refusal returns the zero Ticket and false.
func (h *Health) Allow(deployment string) (Ticket, bool) {
	t, ok := h.entry(deployment).breaker.Allow()
	if !ok {
		return Ticket{}, false
	}
	t.deployment = deployment
	return t, true
}

// Abort releases the half-open probe t holds without recording an outcome. It does nothing
// for a closed-state ticket, a ticket superseded by a later state change, or the zero Ticket,
// and it never creates an entry.
func (h *Health) Abort(t Ticket) {
	if e, ok := h.lookup(t.deployment); ok {
		e.breaker.Abort(t)
	}
}

// Record reports the outcome of the attempt t was claimed for to the breaker and updates the
// EWMA. A ticket of an older breaker generation changes nothing in the breaker (see
// Breaker.Record) but is still an observation of the deployment, so it feeds the EWMA. The
// zero Ticket is ignored.
func (h *Health) Record(t Ticket, success bool) {
	e, ok := h.lookup(t.deployment)
	if !ok {
		return
	}
	e.breaker.Record(t, success)

	x := 1.0
	if success {
		x = 0
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	now := h.now() // read under e.mu so concurrent Records never see dt < 0
	if !e.seen {
		e.rate, e.last, e.seen = x, now, true
		return
	}
	dt := now.Sub(e.last)
	if dt < 0 {
		dt = 0 // clock moved backwards: treat as no time elapsed
	} else {
		e.last = now
	}
	decay := math.Pow(0.5, float64(dt)/float64(HalfLife))
	e.rate = e.rate*decay + (1-decay)*x
}

// Trip forces the deployment's breaker open for at least d.
func (h *Health) Trip(deployment string, d time.Duration) {
	h.entry(deployment).breaker.Trip(d)
}

// State is the breaker state of the deployment: Closed for one never seen, HalfOpen while a
// probe is in flight, Open otherwise while the breaker is open. It claims nothing and never
// creates an entry, so reading it (a metrics scrape does) cannot change the outcome.
func (h *Health) State(deployment string) State {
	h.mu.RLock()
	e, ok := h.entries[deployment]
	h.mu.RUnlock()
	if !ok {
		return Closed
	}
	return e.breaker.State()
}

// Snapshot returns a copy of every known deployment's state and error rate.
func (h *Health) Snapshot() map[string]DeploymentHealth {
	h.mu.RLock()
	entries := make(map[string]*healthEntry, len(h.entries))
	for name, e := range h.entries {
		entries[name] = e
	}
	h.mu.RUnlock()

	out := make(map[string]DeploymentHealth, len(entries))
	for name, e := range entries {
		e.mu.Lock()
		rate := e.rate
		e.mu.Unlock()
		out[name] = DeploymentHealth{State: e.breaker.State(), ErrorRate: rate}
	}
	return out
}
