package reliability

import (
	"testing"
	"time"
)

// TestHealthState: State reads a deployment's breaker state without claiming anything and
// without creating an entry, so a metrics scrape cannot change what it observes.
func TestHealthState(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	const dep = "dep-a"

	if got := h.State("never-seen"); got != Closed {
		t.Errorf("State of an unknown deployment = %v, want closed", got)
	}
	if n := len(h.Snapshot()); n != 0 {
		t.Errorf("State created %d entries, want none", n)
	}

	h.Record(dep, true)
	if got := h.State(dep); got != Closed {
		t.Errorf("State after a success = %v, want closed", got)
	}

	h.Trip(dep, 30*time.Second)
	if got := h.State(dep); got != Open {
		t.Errorf("State right after a trip = %v, want open", got)
	}

	clk.Advance(31 * time.Second) // the cooldown is over: a probe may be claimed
	if got := h.State(dep); got != Open {
		t.Errorf("State after the cooldown but before a probe = %v, want open (State claims nothing)", got)
	}
	if !h.Allow(dep) {
		t.Fatal("the probe was refused after the cooldown")
	}
	if got := h.State(dep); got != HalfOpen {
		t.Errorf("State with a probe in flight = %v, want half-open", got)
	}

	h.Record(dep, true) // the probe succeeded
	if got := h.State(dep); got != Closed {
		t.Errorf("State after a successful probe = %v, want closed", got)
	}
}
