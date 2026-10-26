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

// Allow claims an attempt on the deployment.
func (h *Health) Allow(deployment string) bool {
	return h.entry(deployment).breaker.Allow()
}

// Abort releases a claimed half-open probe on the deployment without recording
// an outcome. It never creates an entry.
func (h *Health) Abort(deployment string) {
	h.mu.RLock()
	e, ok := h.entries[deployment]
	h.mu.RUnlock()
	if ok {
		e.breaker.Abort()
	}
}

// Record reports an attempt outcome to the breaker and updates the EWMA.
func (h *Health) Record(deployment string, success bool) {
	e := h.entry(deployment)
	e.breaker.Record(success)

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
