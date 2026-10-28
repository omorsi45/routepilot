package reliability

import (
	"testing"
	"time"
)

// supersededProbe sets up the race the tickets exist for: caller A claims the half-open probe,
// a Trip supersedes it, and after the quarantine caller B claims the next probe. A's late
// outcome must not touch B's probe.
func supersededProbe(t *testing.T) (b *Breaker, clk *fakeClock, probeA, probeB Ticket) {
	t.Helper()
	clk = newFakeClock()
	b = NewBreaker(clk.Now)
	probeA = claimProbe(t, clk, b)
	b.Trip(30 * time.Second)
	clk.Advance(30 * time.Second)
	probeB, ok := b.Allow()
	if !ok {
		t.Fatal("probe B not admitted after the quarantine")
	}
	if got := b.State(); got != HalfOpen {
		t.Fatalf("state with probe B in flight = %v, want HalfOpen", got)
	}
	return b, clk, probeA, probeB
}

// TestStaleTicketCannotResolveProbe: a ticket from an older generation resolves nothing, so
// a late Record or Abort from a superseded attempt can never settle another caller's probe.
func TestStaleTicketCannotResolveProbe(t *testing.T) {
	t.Run("late success of the superseded probe", func(t *testing.T) {
		b, _, probeA, probeB := supersededProbe(t)
		b.Record(probeA, true)
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's stale success = %v, want HalfOpen (B's probe still in flight)", got)
		}
		if allowed(b) {
			t.Fatal("a second probe was admitted while B's is in flight")
		}
		b.Record(probeB, true)
		if got := b.State(); got != Closed {
			t.Fatalf("state after B's success = %v, want Closed", got)
		}
	})
	t.Run("late failure of the superseded probe", func(t *testing.T) {
		b, clk, probeA, probeB := supersededProbe(t)
		b.Record(probeA, false)
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's stale failure = %v, want HalfOpen (B's probe still in flight)", got)
		}
		// A's failure did not advance the doubling sequence: B's failure gives 20 s, not 40 s.
		b.Record(probeB, false)
		assertQuarantined(t, clk, b, 2*OpenFor)
	})
	t.Run("late abort of the superseded probe", func(t *testing.T) {
		b, _, probeA, probeB := supersededProbe(t)
		b.Abort(probeA)
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's stale abort = %v, want HalfOpen (B's probe still in flight)", got)
		}
		if allowed(b) {
			t.Fatal("A's stale abort released B's probe: a second probe was admitted")
		}
		b.Record(probeB, true)
		if got := b.State(); got != Closed {
			t.Fatalf("state after B's success = %v, want Closed", got)
		}
	})
	t.Run("a probe that already failed cannot be settled again", func(t *testing.T) {
		clk := newFakeClock()
		b := NewBreaker(clk.Now)
		probeA := claimProbe(t, clk, b)
		b.Record(probeA, false) // open for 20 s
		clk.Advance(2 * OpenFor)
		probeB, ok := b.Allow()
		if !ok {
			t.Fatal("probe B not admitted")
		}
		b.Record(probeA, true) // a second, stray resolution of A's spent ticket
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's second resolution = %v, want HalfOpen", got)
		}
		b.Abort(probeA)
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's stray abort = %v, want HalfOpen", got)
		}
		b.Record(probeB, true)
		if got := b.State(); got != Closed {
			t.Fatalf("state after B's success = %v, want Closed", got)
		}
	})
	t.Run("an aborted probe cannot be recorded afterwards", func(t *testing.T) {
		clk := newFakeClock()
		b := NewBreaker(clk.Now)
		probeA := claimProbe(t, clk, b)
		b.Abort(probeA)
		probeB, ok := b.Allow()
		if !ok {
			t.Fatal("probe B not admitted after A's abort")
		}
		b.Record(probeA, true) // A resolves a ticket it already gave back
		if got := b.State(); got != HalfOpen {
			t.Fatalf("state after A's record of an aborted ticket = %v, want HalfOpen", got)
		}
		b.Record(probeB, true)
		if got := b.State(); got != Closed {
			t.Fatalf("state after B's success = %v, want Closed", got)
		}
	})
}

// TestStaleClosedTicketDoesNotCountInWindow: a closed-state attempt claimed before the breaker
// tripped and finished after it closed again belongs to an earlier generation, so it neither
// counts in the failure window nor feeds the failure streak.
func TestStaleClosedTicketDoesNotCountInWindow(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	straggler, ok := b.Allow() // claimed while closed
	if !ok {
		t.Fatal("closed breaker refused an attempt")
	}
	tripOpen(t, b)
	clk.Advance(OpenFor)
	probe, ok := b.Allow()
	if !ok {
		t.Fatal("probe not admitted")
	}
	b.Record(probe, true)
	if got := b.State(); got != Closed {
		t.Fatalf("state after the probe = %v, want Closed", got)
	}
	for i := 0; i < ConsecutiveFailures; i++ {
		b.Record(straggler, false) // the old attempt reports in, again and again
	}
	if got := b.State(); got != Closed {
		t.Fatalf("stale closed-state failures re-opened the breaker: state %v", got)
	}
}

// TestCurrentClosedTicketCountsInWindow: a closed-state ticket of the current generation counts,
// also when other attempts were claimed and recorded in between.
func TestCurrentClosedTicketCountsInWindow(t *testing.T) {
	b := NewBreaker(newFakeClock().Now)
	var tickets []Ticket
	for i := 0; i < ConsecutiveFailures; i++ {
		tk, ok := b.Allow()
		if !ok {
			t.Fatalf("closed breaker refused attempt %d", i)
		}
		tickets = append(tickets, tk)
	}
	for i, tk := range tickets {
		if got := b.State(); got != Closed {
			t.Fatalf("state before failure %d = %v, want Closed", i, got)
		}
		b.Record(tk, false)
	}
	if got := b.State(); got != Open {
		t.Fatalf("state after %d failures from concurrent tickets = %v, want Open", ConsecutiveFailures, got)
	}
}

// TestZeroTicketResolvesNothing: the zero Ticket is what a refused Allow returns, and
// resolving it is a no-op.
func TestZeroTicketResolvesNothing(t *testing.T) {
	clk := newFakeClock()
	b := NewBreaker(clk.Now)
	for i := 0; i < 2*ConsecutiveFailures; i++ {
		b.Record(Ticket{}, false)
	}
	if got := b.State(); got != Closed {
		t.Fatalf("zero-ticket failures opened the breaker: state %v", got)
	}
	probe := claimProbe(t, clk, b)
	b.Record(Ticket{}, true)
	b.Abort(Ticket{})
	if got := b.State(); got != HalfOpen {
		t.Fatalf("zero ticket settled the probe: state %v, want HalfOpen", got)
	}
	b.Record(probe, true)
	if got := b.State(); got != Closed {
		t.Fatalf("state after the real probe = %v, want Closed", got)
	}

	h := NewHealth(clk.Now)
	h.Record(Ticket{}, false)
	h.Abort(Ticket{})
	if n := len(h.Snapshot()); n != 0 {
		t.Fatalf("resolving a zero ticket created %d health entries, want none", n)
	}
}

// TestTicketAccessors: a ticket knows its deployment and whether it holds the probe.
func TestTicketAccessors(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	closed, ok := h.Allow("dep-a")
	if !ok || closed.Deployment() != "dep-a" || closed.Probe() {
		t.Fatalf("closed-state ticket = {deployment %q, probe %v, ok %v}, want {dep-a, false, true}",
			closed.Deployment(), closed.Probe(), ok)
	}
	h.Trip("dep-b", time.Second)
	clk.Advance(time.Second)
	probe, ok := h.Allow("dep-b")
	if !ok || probe.Deployment() != "dep-b" || !probe.Probe() {
		t.Fatalf("probe ticket = {deployment %q, probe %v, ok %v}, want {dep-b, true, true}",
			probe.Deployment(), probe.Probe(), ok)
	}
	if refused, ok := h.Allow("dep-b"); ok || refused != (Ticket{}) {
		t.Fatalf("second Allow with the probe in flight = %+v, %v, want the zero ticket and false", refused, ok)
	}
}

// TestHealthStaleTicketCannotResolveProbe: the same guarantee through the registry.
func TestHealthStaleTicketCannotResolveProbe(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	h.Trip("a", OpenFor)
	clk.Advance(OpenFor)
	probeA, ok := h.Allow("a")
	if !ok {
		t.Fatal("probe A not admitted")
	}
	h.Trip("a", 30*time.Second) // supersedes A
	clk.Advance(30 * time.Second)
	probeB, ok := h.Allow("a")
	if !ok {
		t.Fatal("probe B not admitted")
	}

	h.Record(probeA, true)
	if got := h.State("a"); got != HalfOpen {
		t.Fatalf("state after A's stale success = %v, want HalfOpen", got)
	}
	h.Abort(probeA)
	if got := h.State("a"); got != HalfOpen {
		t.Fatalf("state after A's stale abort = %v, want HalfOpen", got)
	}
	if _, ok := h.Allow("a"); ok {
		t.Fatal("a second probe was admitted while B's is in flight")
	}
	h.Record(probeB, true)
	if got := h.State("a"); got != Closed {
		t.Fatalf("state after B's success = %v, want Closed", got)
	}
}
