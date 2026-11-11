package reliability

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// healthAllowed reports whether the registry admits an attempt on the deployment, dropping
// the ticket.
func healthAllowed(h *Health, deployment string) bool {
	_, ok := h.Allow(deployment)
	return ok
}

// healthRecord claims an attempt on the deployment and reports its outcome at once.
func healthRecord(h *Health, deployment string, success bool) {
	tk, _ := h.Allow(deployment)
	h.Record(tk, success)
}

func TestHealthEWMA(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	const dep = "openai/gpt-4o"

	healthRecord(h, dep, false) // first observation sets the rate exactly
	if got := h.Snapshot()[dep].ErrorRate; got != 1 {
		t.Fatalf("rate after first failure = %v, want exactly 1", got)
	}
	clk.Advance(HalfLife)
	healthRecord(h, dep, true)
	if got := h.Snapshot()[dep].ErrorRate; math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("rate after 30 s and a success = %v, want 0.5", got)
	}
	clk.Advance(HalfLife)
	healthRecord(h, dep, true)
	if got := h.Snapshot()[dep].ErrorRate; math.Abs(got-0.25) > 1e-9 {
		t.Fatalf("rate after 60 s and a second success = %v, want 0.25", got)
	}
}

func TestHealthEWMAFirstSuccessIsZero(t *testing.T) {
	h := NewHealth(newFakeClock().Now)
	healthRecord(h, "d", true)
	if got := h.Snapshot()["d"].ErrorRate; got != 0 {
		t.Fatalf("rate after first success = %v, want 0", got)
	}
}

func TestHealthEWMAZeroElapsedKeepsRate(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	healthRecord(h, "d", false)
	healthRecord(h, "d", true) // dt = 0: decay 1, the observation has no weight
	if got := h.Snapshot()["d"].ErrorRate; math.Abs(got-1) > 1e-9 {
		t.Fatalf("rate with dt=0 = %v, want 1", got)
	}
}

func TestHealthEWMAClockBackwards(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	healthRecord(h, "d", false)
	clk.Advance(-time.Minute)
	healthRecord(h, "d", true)
	got := h.Snapshot()["d"].ErrorRate
	if math.IsNaN(got) || got < 0 || got > 1 {
		t.Fatalf("rate after backwards clock = %v, want within [0,1]", got)
	}
}

func TestHealthLazyAndIndependent(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	if len(h.Snapshot()) != 0 {
		t.Fatal("new Health must have no deployments")
	}
	if !h.Available("a") {
		t.Fatal("unknown deployment must be available")
	}
	if len(h.Snapshot()) != 0 {
		t.Fatal("Available must not create an entry")
	}
	if !healthAllowed(h, "a") {
		t.Fatal("unknown deployment must admit an attempt")
	}
	if _, ok := h.Snapshot()["a"]; !ok {
		t.Fatal("Allow must create the entry")
	}
	for i := 0; i < ConsecutiveFailures; i++ {
		healthRecord(h, "a", false)
	}
	if h.Available("a") || healthAllowed(h, "a") {
		t.Fatal("deployment a must be open")
	}
	if !h.Available("b") {
		t.Fatal("deployment b must be unaffected")
	}
	healthRecord(h, "b", true) // b only appears in the snapshot once something is recorded
	snap := h.Snapshot()
	if snap["a"].State != Open || snap["a"].ErrorRate != 1 {
		t.Fatalf("snapshot a = %+v", snap["a"])
	}
	if s, ok := snap["b"]; !ok || s.State != Closed {
		t.Fatalf("snapshot b = %+v, ok=%v", s, ok)
	}
}

func TestHealthSnapshotReturnsCopy(t *testing.T) {
	h := NewHealth(newFakeClock().Now)
	healthRecord(h, "a", false)
	snap := h.Snapshot()
	delete(snap, "a")
	snap["x"] = DeploymentHealth{State: Open}
	again := h.Snapshot()
	if _, ok := again["a"]; !ok {
		t.Fatal("mutating a snapshot removed a deployment")
	}
	if _, ok := again["x"]; ok {
		t.Fatal("mutating a snapshot added a deployment")
	}
}

func TestHealthTripAndAbort(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	h.Trip("a", 30*time.Second)
	if h.Available("a") {
		t.Fatal("tripped deployment must be unavailable")
	}
	clk.Advance(30 * time.Second)
	probe, ok := h.Allow("a")
	if !ok {
		t.Fatal("probe must be admitted after the trip elapsed")
	}
	if h.Available("a") {
		t.Fatal("probe in flight")
	}
	h.Abort(probe)
	if !h.Available("a") || !healthAllowed(h, "a") {
		t.Fatal("Abort must release the probe")
	}
	h.Abort(Ticket{deployment: "never-seen", gen: 1}) // must not panic or create state
	if _, ok := h.Snapshot()["never-seen"]; ok {
		t.Fatal("Abort must not create a breaker")
	}
}

func TestHealthConcurrentUse(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	var wg sync.WaitGroup
	for g := 0; g < 64; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			dep := fmt.Sprintf("dep-%d", g%4)
			for i := 0; i < 200; i++ {
				if tk, ok := h.Allow(dep); ok {
					if i%7 == 0 {
						h.Abort(tk) // admitted attempt abandoned before being sent
					} else {
						h.Record(tk, (g+i)%3 != 0)
					}
				}
				_ = h.Available(dep)
				if i%50 == 0 {
					h.Trip(dep, time.Second)
					_ = h.Snapshot()
				}
				clk.Advance(10 * time.Millisecond)
			}
		}(g)
	}
	wg.Wait()
	if len(h.Snapshot()) != 4 {
		t.Fatalf("expected 4 deployments, got %d", len(h.Snapshot()))
	}

	// Invariant: once the cooldown elapses, exactly one concurrent caller is
	// admitted as the half-open probe.
	h.Trip("probe", OpenFor)
	clk.Advance(OpenFor)
	var admitted atomic.Int32
	start := make(chan struct{})
	for g := 0; g < 64; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if healthAllowed(h, "probe") {
				admitted.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	if got := admitted.Load(); got != 1 {
		t.Fatalf("%d concurrent callers admitted as probe, want exactly 1", got)
	}
	if got := h.Snapshot()["probe"].State; got != HalfOpen {
		t.Fatalf("state with probe in flight = %v, want HalfOpen", got)
	}
}
