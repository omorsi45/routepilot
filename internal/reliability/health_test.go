package reliability

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

func TestHealthEWMA(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	const dep = "openai/gpt-4o"

	h.Record(dep, false) // first observation sets the rate exactly
	if got := h.Snapshot()[dep].ErrorRate; got != 1 {
		t.Fatalf("rate after first failure = %v, want exactly 1", got)
	}
	clk.Advance(HalfLife)
	h.Record(dep, true)
	if got := h.Snapshot()[dep].ErrorRate; math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("rate after 30 s and a success = %v, want 0.5", got)
	}
	clk.Advance(HalfLife)
	h.Record(dep, true)
	if got := h.Snapshot()[dep].ErrorRate; math.Abs(got-0.25) > 1e-9 {
		t.Fatalf("rate after 60 s and a second success = %v, want 0.25", got)
	}
}

func TestHealthEWMAFirstSuccessIsZero(t *testing.T) {
	h := NewHealth(newFakeClock().Now)
	h.Record("d", true)
	if got := h.Snapshot()["d"].ErrorRate; got != 0 {
		t.Fatalf("rate after first success = %v, want 0", got)
	}
}

func TestHealthEWMAZeroElapsedKeepsRate(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	h.Record("d", false)
	h.Record("d", true) // dt = 0: decay 1, the observation has no weight
	if got := h.Snapshot()["d"].ErrorRate; math.Abs(got-1) > 1e-9 {
		t.Fatalf("rate with dt=0 = %v, want 1", got)
	}
}

func TestHealthEWMAClockBackwards(t *testing.T) {
	clk := newFakeClock()
	h := NewHealth(clk.Now)
	h.Record("d", false)
	clk.Advance(-time.Minute)
	h.Record("d", true)
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
	if !h.Available("a") || !h.Allow("a") {
		t.Fatal("unknown deployment must be available")
	}
	for i := 0; i < ConsecutiveFailures; i++ {
		h.Record("a", false)
	}
	if h.Available("a") || h.Allow("a") {
		t.Fatal("deployment a must be open")
	}
	if !h.Available("b") {
		t.Fatal("deployment b must be unaffected")
	}
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
	h.Record("a", false)
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
	if !h.Allow("a") {
		t.Fatal("probe must be admitted after the trip elapsed")
	}
	if h.Available("a") {
		t.Fatal("probe in flight")
	}
	h.Abort("a")
	if !h.Available("a") || !h.Allow("a") {
		t.Fatal("Abort must release the probe")
	}
	h.Abort("never-seen") // must not panic or create state
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
				if h.Allow(dep) {
					h.Record(dep, (g+i)%3 != 0)
				} else {
					h.Abort(dep)
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
}
