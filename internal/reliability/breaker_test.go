package reliability

import (
	"sync"
	"testing"
	"time"
)

// fakeClock is a manually advanced clock safe for concurrent use.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func (c *fakeClock) Set(t time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = t
}

// alternate records n results starting with a failure: F, S, F, S, ...
func alternate(b *Breaker, n int) {
	for i := 0; i < n; i++ {
		b.Record(i%2 != 0)
	}
}

func failN(b *Breaker, n int) {
	for i := 0; i < n; i++ {
		b.Record(false)
	}
}

// tripOpen forces the breaker open through consecutive failures.
func tripOpen(t *testing.T, b *Breaker) {
	t.Helper()
	failN(b, ConsecutiveFailures)
	if got := b.State(); got != Open {
		t.Fatalf("state after %d failures = %v, want Open", ConsecutiveFailures, got)
	}
}

func TestStateString(t *testing.T) {
	cases := map[State]string{Closed: "closed", Open: "open", HalfOpen: "half-open", State(99): "unknown"}
	for s, want := range cases {
		if got := s.String(); got != want {
			t.Errorf("State(%d).String() = %q, want %q", int(s), got, want)
		}
	}
}

func TestBreakerOpensOnRatio(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, MinRequests-1) // 19 requests, 10 failures, never consecutive
	if got := b.State(); got != Closed {
		t.Fatalf("state with 19 requests = %v, want Closed", got)
	}
	b.Record(true) // 20th request: 10/20 = 0.5
	if got := b.State(); got != Open {
		t.Fatalf("state with 20 requests and ratio 0.5 = %v, want Open", got)
	}
}

func TestBreakerStaysClosedBelowRatio(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	// 20 requests, 9 failures (never more than one in a row): ratio 0.45.
	for i := 0; i < 20; i++ {
		b.Record(!(i%2 == 0 && i < 18))
	}
	if got := b.State(); got != Closed {
		t.Fatalf("state with ratio 0.45 = %v, want Closed", got)
	}
}

func TestBreakerOpensOnConsecutive(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	failN(b, ConsecutiveFailures-1)
	if got := b.State(); got != Closed {
		t.Fatalf("state after 4 failures = %v, want Closed", got)
	}
	b.Record(false)
	if got := b.State(); got != Open {
		t.Fatalf("state after 5 consecutive failures = %v, want Open", got)
	}
}

func TestBreakerSuccessResetsConsecutive(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	failN(b, ConsecutiveFailures-1)
	b.Record(true)
	failN(b, ConsecutiveFailures-1)
	if got := b.State(); got != Closed {
		t.Fatalf("state = %v, want Closed (success resets the streak)", got)
	}
}

func TestBreakerWindowExpires(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, MinRequests-1) // 10 failures in 19 requests
	clk.Advance(Window + time.Second)
	b.Record(true) // would be the 20th request, but the old ones expired
	if got := b.State(); got != Closed {
		t.Fatalf("state after window expiry = %v, want Closed", got)
	}
}

func TestBreakerWindowEdge(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, MinRequests-1)
	clk.Advance(Window - 2*time.Second) // still inside the 30 bucket window
	b.Record(true)
	if got := b.State(); got != Open {
		t.Fatalf("state with old records still in window = %v, want Open", got)
	}
}

func TestBreakerBucketReuseDoesNotLeakOldCounts(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, MinRequests-1)
	// Exactly one ring revolution later the same bucket index is reused.
	clk.Advance(Window)
	b.Record(true)
	if got := b.State(); got != Closed {
		t.Fatalf("state after ring wrap = %v, want Closed", got)
	}
}

func TestBreakerClockMovingBackwardsIsSafe(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, 6)
	clk.Advance(-time.Hour)
	b.Record(false)
	b.Record(true)
	_ = b.Allow()
	_ = b.Available()
	_ = b.State()
	b.Trip(time.Second)
	if got := b.State(); got != Open {
		t.Fatalf("state after Trip = %v, want Open", got)
	}
}

func TestBreakerHalfOpenSingleProbe(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)

	if b.Available() || b.Allow() {
		t.Fatal("open breaker within cooldown must be unavailable")
	}
	clk.Advance(OpenFor - time.Nanosecond)
	if b.Available() || b.Allow() {
		t.Fatal("breaker must stay unavailable until the cooldown elapses")
	}
	clk.Advance(time.Nanosecond)
	if got := b.State(); got != Open {
		t.Fatalf("state with elapsed cooldown and no probe = %v, want Open", got)
	}
	if !b.Available() {
		t.Fatal("Available must be true once the cooldown elapsed")
	}
	if !b.Available() || b.State() != Open {
		t.Fatal("Available must not claim the probe")
	}
	if !b.Allow() {
		t.Fatal("first Allow after cooldown must claim the probe")
	}
	if b.Allow() {
		t.Fatal("second Allow while the probe is in flight must be false")
	}
	if b.Available() {
		t.Fatal("Available must be false while the probe is in flight")
	}
	if s := b.State(); s != HalfOpen {
		t.Fatalf("state with probe in flight = %v, want HalfOpen", s)
	}
}

func TestBreakerAllowWhenClosed(t *testing.T) {
	b := NewBreaker(newFakeClock().Now)
	for i := 0; i < 3; i++ {
		if !b.Allow() || !b.Available() {
			t.Fatal("closed breaker must allow every attempt")
		}
	}
}

func TestBreakerProbeSuccessCloses(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)
	clk.Advance(OpenFor)
	if !b.Allow() {
		t.Fatal("probe not admitted")
	}
	b.Record(true)
	if s := b.State(); s != Closed {
		t.Fatalf("state after probe success = %v, want Closed", s)
	}
	if !b.Allow() {
		t.Fatal("closed breaker must allow")
	}

	// The next trip uses the base duration again.
	tripOpen(t, b)
	clk.Advance(OpenFor - time.Nanosecond)
	if b.Available() {
		t.Fatal("must still be open just before 10 s")
	}
	clk.Advance(time.Nanosecond)
	if !b.Available() {
		t.Fatal("next trip after a successful close must reopen for 10 s")
	}
}

func TestBreakerProbeSuccessAfterDoublingResetsDuration(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)
	clk.Advance(OpenFor)
	b.Allow()
	b.Record(false) // now open for 20 s
	clk.Advance(2 * OpenFor)
	b.Allow()
	b.Record(true)
	tripOpen(t, b)
	clk.Advance(OpenFor)
	if !b.Available() {
		t.Fatal("duration must reset to 10 s after a successful close")
	}
}

func TestBreakerReTripDoubles(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)
	want := []time.Duration{10, 20, 40, 80, 120, 120}
	for i, secs := range want {
		d := secs * time.Second
		if i > 0 {
			// probe failure re-opens with the doubled duration
			if !b.Allow() {
				t.Fatalf("step %d: probe not admitted", i)
			}
			b.Record(false)
		}
		clk.Advance(d - time.Nanosecond)
		if b.Available() || b.Allow() {
			t.Fatalf("step %d: available %v before the open duration elapsed", i, d)
		}
		clk.Advance(time.Nanosecond)
		if !b.Available() {
			t.Fatalf("step %d: not available after %v", i, d)
		}
	}
}

func TestBreakerRecordIgnoredWhileOpen(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)
	b.Record(true) // straggler from before the trip
	b.Record(false)
	if s := b.State(); s != Open {
		t.Fatalf("state = %v, want Open", s)
	}
	clk.Advance(OpenFor - time.Nanosecond)
	if b.Available() {
		t.Fatal("straggler must not alter the open deadline")
	}
	clk.Advance(OpenFor)
	// Elapsed cooldown, no probe claimed: records are still ignored.
	b.Record(true)
	if s := b.State(); s != Open {
		t.Fatalf("record without a claimed probe closed the breaker: %v", s)
	}
	if !b.Allow() {
		t.Fatal("probe must still be claimable")
	}
}

func TestBreakerTripForcesOpen(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	b.Trip(30 * time.Second)
	if b.Available() || b.Allow() {
		t.Fatal("tripped breaker must be unavailable")
	}
	if s := b.State(); s != Open {
		t.Fatalf("state = %v, want Open", s)
	}
	clk.Advance(30*time.Second - time.Nanosecond)
	if b.Available() {
		t.Fatal("still unavailable until 30 s elapse")
	}
	clk.Advance(time.Nanosecond)
	if !b.Available() || !b.Allow() {
		t.Fatal("probe must be admitted once the forced period elapsed")
	}
	b.Record(true)
	if s := b.State(); s != Closed {
		t.Fatalf("state after probe success = %v, want Closed", s)
	}
}

func TestBreakerTripNeverShortens(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	b.Trip(30 * time.Second)
	b.Trip(time.Second)
	clk.Advance(29 * time.Second)
	if b.Available() {
		t.Fatal("a shorter Trip must not shorten the open period")
	}
	b.Trip(60 * time.Second)
	clk.Advance(59 * time.Second)
	if b.Available() {
		t.Fatal("a longer Trip must extend the open period")
	}
}

func TestBreakerTripDoesNotChangeDoublingSequence(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	b.Trip(30 * time.Second) // forced open, sequence still at 10 s
	clk.Advance(30 * time.Second)
	if !b.Allow() {
		t.Fatal("probe not admitted")
	}
	b.Record(false) // first probe failure doubles the base: 20 s
	clk.Advance(20*time.Second - time.Nanosecond)
	if b.Available() {
		t.Fatal("expected 20 s after the first probe failure")
	}
	clk.Advance(time.Nanosecond)
	if !b.Available() {
		t.Fatal("expected availability after 20 s")
	}
}

func TestBreakerAbortReleasesProbe(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b)
	clk.Advance(OpenFor)
	if !b.Allow() {
		t.Fatal("probe not admitted")
	}
	b.Abort()
	if s := b.State(); s != Open {
		t.Fatalf("state after Abort = %v, want Open", s)
	}
	if !b.Available() {
		t.Fatal("cooldown is still elapsed so the probe must be claimable")
	}
	if !b.Allow() {
		t.Fatal("next Allow must claim the probe again")
	}
	b.Record(false) // the aborted attempt recorded nothing: duration is 20 s, not 40 s
	clk.Advance(20 * time.Second)
	if !b.Available() {
		t.Fatal("Abort must not have advanced the doubling sequence")
	}
}

func TestBreakerAbortNoOps(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	b.Abort() // closed
	if s := b.State(); s != Closed {
		t.Fatalf("Abort on closed changed state to %v", s)
	}
	tripOpen(t, b)
	b.Abort() // open, no probe
	if s := b.State(); s != Open {
		t.Fatalf("Abort on open changed state to %v", s)
	}
	if b.Available() {
		t.Fatal("Abort without a probe must not shorten the cooldown")
	}
	clk.Advance(OpenFor)
	b.Abort() // open, cooldown elapsed, no probe claimed
	if !b.Allow() {
		t.Fatal("probe must remain claimable after a no-op Abort")
	}
}

func TestBreakerProbeSuccessAfterRatioTripDoesNotReopenOnOneFailure(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	alternate(b, MinRequests) // ratio trip
	if s := b.State(); s != Open {
		t.Fatalf("state = %v, want Open", s)
	}
	clk.Advance(OpenFor)
	if !b.Allow() {
		t.Fatal("probe not admitted")
	}
	b.Record(true)
	if s := b.State(); s != Closed {
		t.Fatalf("state = %v, want Closed", s)
	}
	b.Record(false)
	if s := b.State(); s != Closed {
		t.Fatalf("single failure after close re-opened the breaker: %v", s)
	}
}

func TestBreakerCloseClearsConsecutiveCount(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	tripOpen(t, b) // 5 consecutive failures
	clk.Advance(OpenFor)
	b.Allow()
	b.Record(true)
	failN(b, ConsecutiveFailures-1)
	if s := b.State(); s != Closed {
		t.Fatalf("stale consecutive count survived the close: %v", s)
	}
}
