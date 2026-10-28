// Package reliability provides per-deployment circuit breakers, an error-rate
// health registry, and jittered exponential backoff.
package reliability

import (
	"sync"
	"time"
)

// Breaker and health tuning constants.
const (
	// Window is the rolling window over which request outcomes are counted.
	Window = 30 * time.Second
	// MinRequests is the minimum window volume before the failure ratio applies.
	MinRequests = 20
	// FailureRatio is the failure ratio at or above which the breaker opens.
	FailureRatio = 0.5
	// ConsecutiveFailures is the failure streak that opens the breaker.
	ConsecutiveFailures = 5
	// OpenFor is the first open duration after a trip from Closed.
	OpenFor = 10 * time.Second
	// MaxOpenFor caps the doubled open duration.
	MaxOpenFor = 120 * time.Second
	// HalfLife is the EWMA half-life of the health error rate.
	HalfLife = 30 * time.Second
)

// State is the externally visible state of a Breaker.
type State int

// Breaker states.
const (
	Closed State = iota
	Open
	HalfOpen
)

// String returns the lowercase state name.
func (s State) String() string {
	switch s {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

const bucketCount = int(Window / time.Second)

type bucket struct {
	sec      int64 // unix second this bucket currently holds
	requests int
	failures int
}

// Ticket is the claim an admitted attempt holds, and what it hands back to report
// the outcome. It records the breaker generation at claim time and whether it holds the
// half-open probe. A ticket from an older generation resolves nothing, so a late Record or
// Abort from a superseded attempt can never settle another caller's probe.
//
// The zero Ticket is what a refused Allow returns; resolving it is a no-op.
type Ticket struct {
	deployment string
	gen        uint64
	probe      bool
}

// Deployment is the deployment the ticket was claimed on, empty for the zero Ticket and for
// a ticket claimed from a Breaker directly.
func (t Ticket) Deployment() string { return t.deployment }

// Probe reports whether the ticket holds the half-open probe, as opposed to a closed-state
// attempt.
func (t Ticket) Probe() bool { return t.probe }

// Breaker is a circuit breaker with a rolling failure window and a
// single-probe half-open phase. It is safe for concurrent use.
//
// Every state transition increments gen: closed to open, open to half-open (the probe
// claim), half-open to closed or open (the probe's outcome or its Abort), and every Trip.
// A Ticket carries the generation it was claimed in.
type Breaker struct {
	mu  sync.Mutex
	now func() time.Time

	open      bool      // false: closed; true: open (or half-open when probing)
	probing   bool      // a half-open probe has been claimed
	openUntil time.Time // earliest time a probe may be claimed
	openFor   time.Duration
	gen       uint64 // starts at 1, so the zero Ticket never matches

	consecutive int
	buckets     [bucketCount]bucket
	latest      time.Time // newest clock reading seen; guards against a clock moving backwards
}

// NewBreaker returns a closed breaker that reads time from now.
func NewBreaker(now func() time.Time) *Breaker {
	return &Breaker{now: now, openFor: OpenFor, gen: 1}
}

// tick returns the current time, never earlier than a previous reading.
// The caller must hold b.mu.
func (b *Breaker) tick() time.Time {
	n := b.now()
	if n.Before(b.latest) {
		return b.latest
	}
	b.latest = n
	return n
}

// Available reports what Allow would return, without claiming anything.
func (b *Breaker) Available() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()
	if !b.open {
		return true
	}
	return !b.probing && !now.Before(b.openUntil)
}

// Allow claims an attempt. When closed it admits every attempt with a closed-state ticket.
// When open and the cooldown has elapsed it admits exactly one probe, with a probe ticket;
// further calls return the zero Ticket and false until the probe resolves.
func (b *Breaker) Allow() (Ticket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()
	if !b.open {
		return Ticket{gen: b.gen}, true
	}
	if b.probing || now.Before(b.openUntil) {
		return Ticket{}, false
	}
	b.probing = true
	b.gen++ // open to half-open
	return Ticket{gen: b.gen, probe: true}, true
}

// Abort releases the probe that t holds without recording an outcome. It is a no-op for a
// closed-state ticket and for a ticket of an older generation.
func (b *Breaker) Abort(t Ticket) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if t.gen != b.gen || !t.probe || !b.open || !b.probing {
		return
	}
	b.probing = false
	b.gen++ // half-open to open
}

// Record reports the outcome of the attempt t was claimed for. A ticket of an older
// generation is ignored: it neither resolves a probe nor counts in the window. A probe ticket
// resolves the probe: a success closes the breaker and a failure re-opens it for twice as
// long. A closed-state ticket of the current generation counts in the window.
func (b *Breaker) Record(t Ticket, success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()

	if t.gen != b.gen {
		return // zero ticket, or an attempt from before the last state transition
	}

	if t.probe {
		if !b.open || !b.probing {
			return // cannot happen for a current ticket; never settle without a probe
		}
		b.probing = false
		if success {
			b.close()
			return
		}
		b.gen++ // half-open to open
		b.openFor = min(b.openFor*2, MaxOpenFor)
		if until := now.Add(b.openFor); until.After(b.openUntil) {
			b.openUntil = until // never shorten a longer quarantine
		}
		return
	}

	if b.open {
		return // cannot happen for a current closed-state ticket
	}

	bk := b.bucketFor(now)
	bk.requests++
	if success {
		b.consecutive = 0
	} else {
		bk.failures++
		b.consecutive++
	}

	requests, failures := b.windowSums(now)
	if b.consecutive >= ConsecutiveFailures ||
		(requests >= MinRequests && float64(failures)/float64(requests) >= FailureRatio) {
		b.open = true
		b.probing = false
		b.openUntil = now.Add(b.openFor)
		b.gen++ // closed to open
	}
}

// Trip forces the breaker open until at least d from now. It never shortens an
// existing open period and does not change the doubling sequence. A probe in
// flight is superseded and its outcome is ignored. Every Trip starts a new generation, so
// the tickets claimed before it resolve nothing.
func (b *Breaker) Trip(d time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()
	until := now.Add(d)
	b.gen++
	if !b.open {
		b.open = true
		b.probing = false
		b.openUntil = until
		return
	}
	if until.After(b.openUntil) {
		b.openUntil = until
	}
	// The trip supersedes any in-flight probe: its later Record is ignored
	// (an older generation), so it neither closes nor shortens the period.
	b.probing = false
}

// State returns Closed, Open, or HalfOpen. HalfOpen means a probe is in flight;
// an open breaker whose cooldown elapsed but has no probe claimed reports Open.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.tick()
	switch {
	case !b.open:
		return Closed
	case b.probing:
		return HalfOpen
	default:
		return Open
	}
}

// close transitions to Closed and clears all failure history.
// The caller must hold b.mu.
func (b *Breaker) close() {
	b.open = false
	b.probing = false
	b.openFor = OpenFor
	b.consecutive = 0
	b.buckets = [bucketCount]bucket{}
	b.gen++ // half-open to closed
}

// bucketFor returns the bucket for now's second, resetting it if it holds an
// older second. The caller must hold b.mu.
func (b *Breaker) bucketFor(now time.Time) *bucket {
	sec := now.Unix()
	bk := &b.buckets[int(((sec%int64(bucketCount))+int64(bucketCount))%int64(bucketCount))]
	if bk.sec != sec {
		*bk = bucket{sec: sec}
	}
	return bk
}

// windowSums totals buckets whose second lies in (now-Window, now].
// The caller must hold b.mu.
func (b *Breaker) windowSums(now time.Time) (requests, failures int) {
	sec := now.Unix()
	oldest := sec - int64(bucketCount)
	for i := range b.buckets {
		bk := &b.buckets[i]
		if bk.sec > oldest && bk.sec <= sec {
			requests += bk.requests
			failures += bk.failures
		}
	}
	return requests, failures
}
