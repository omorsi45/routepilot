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

// Breaker is a circuit breaker with a rolling failure window and a
// single-probe half-open phase. It is safe for concurrent use.
type Breaker struct {
	mu  sync.Mutex
	now func() time.Time

	open      bool      // false: closed; true: open (or half-open when probing)
	probing   bool      // a half-open probe has been claimed
	openUntil time.Time // earliest time a probe may be claimed
	openFor   time.Duration

	consecutive int
	buckets     [bucketCount]bucket
	latest      time.Time // newest clock reading seen; guards against a clock moving backwards
}

// NewBreaker returns a closed breaker that reads time from now.
func NewBreaker(now func() time.Time) *Breaker {
	return &Breaker{now: now, openFor: OpenFor}
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

// Allow claims an attempt. When open and the cooldown has elapsed it admits
// exactly one probe; further calls return false until the probe resolves.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()
	if !b.open {
		return true
	}
	if b.probing || now.Before(b.openUntil) {
		return false
	}
	b.probing = true
	return true
}

// Abort releases a claimed probe without recording an outcome. It is a no-op
// in any other state.
func (b *Breaker) Abort() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.open && b.probing {
		b.probing = false
	}
}

// Record reports the outcome of an attempt admitted by Allow.
func (b *Breaker) Record(success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()

	if b.open {
		if !b.probing {
			return // straggler from before the trip, or no probe claimed
		}
		b.probing = false
		if success {
			b.close()
			return
		}
		b.openFor = min(b.openFor*2, MaxOpenFor)
		if until := now.Add(b.openFor); until.After(b.openUntil) {
			b.openUntil = until // never shorten a longer quarantine
		}
		return
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
	}
}

// Trip forces the breaker open until at least d from now. It never shortens an
// existing open period and does not change the doubling sequence. A probe in
// flight is superseded and its outcome is ignored.
func (b *Breaker) Trip(d time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.tick()
	until := now.Add(d)
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
	// (open, no probe claimed), so it neither closes nor shortens the period.
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
