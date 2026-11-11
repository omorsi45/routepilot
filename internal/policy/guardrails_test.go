package policy

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// guardClock is a clock the test moves by hand. The start sits in the middle of a minute so
// that advancing by less than a minute stays in the same bucket.
type guardClock struct {
	mu sync.Mutex
	t  time.Time
}

func newGuardClock() *guardClock {
	return &guardClock{t: time.Date(2026, 10, 28, 12, 0, 30, 0, time.UTC)}
}

func (c *guardClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *guardClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// observeN records n requests for the role: bad of them errors, relaxed of them relaxed, all
// with the given latency.
func observeN(g *Guard, role string, n, bad, relaxed int, latency time.Duration) {
	for i := 0; i < n; i++ {
		g.Observe(role, Observation{Error: i < bad, Relaxed: i < relaxed, Latency: latency})
	}
}

// labelN records n feedback labels for the role, ok of them successes.
func labelN(g *Guard, role string, n, ok int) {
	for i := 0; i < n; i++ {
		success := i < ok
		g.Observe(role, Observation{Feedback: &success})
	}
}

func mustNotTrip(t *testing.T, g *Guard, what string) {
	t.Helper()
	if trip, reason := g.Evaluate(); trip {
		t.Fatalf("%s: the guard tripped (%q), want quiet", what, reason)
	}
}

func mustTrip(t *testing.T, g *Guard, what, wantInReason string) {
	t.Helper()
	trip, reason := g.Evaluate()
	if !trip {
		t.Fatalf("%s: the guard did not trip, want a trip naming %q", what, wantInReason)
	}
	if !strings.Contains(reason, wantInReason) {
		t.Errorf("%s: reason = %q, want it to name %q", what, reason, wantInReason)
	}
}

// TestGuardTripsOnErrorRate: with 200 canary requests the guard trips when the canary's error
// rate is more than 0.02 above the stable one, and not at exactly 0.02.
func TestGuardTripsOnErrorRate(t *testing.T) {
	const ms = time.Millisecond
	t.Run("more than 0.02 above trips", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 10, 0, 100*ms) // 1%
		observeN(g, "canary", 200, 8, 0, 100*ms)   // 4%, 0.03 above
		mustTrip(t, g, "0.03 above", "error rate")
	})
	t.Run("exactly 0.02 above does not trip", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 200, 4, 0, 100*ms) // 2% against 0%
		mustNotTrip(t, g, "exactly 0.02 above")

		g = NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 10, 0, 100*ms) // 1%
		observeN(g, "canary", 200, 6, 0, 100*ms)   // 3%: the float difference is 0.019999999999999997
		mustNotTrip(t, g, "exactly 0.02 above, away from zero")
		g = NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 50, 0, 100*ms) // 5%
		observeN(g, "canary", 200, 14, 0, 100*ms)  // 7%: the float difference is 0.020000000000000004
		mustNotTrip(t, g, "exactly 0.02 above, rounding up")
	})
	t.Run("a canary that errors less than stable never trips", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 300, 0, 100*ms)
		observeN(g, "canary", 200, 0, 0, 100*ms)
		mustNotTrip(t, g, "better canary")
	})
	t.Run("the reason carries numbers and no request data", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 200, 20, 0, 100*ms)
		_, reason := g.Evaluate()
		for _, want := range []string{"0.1000", "0.0000", "200"} {
			if !strings.Contains(reason, want) {
				t.Errorf("reason = %q, want it to contain %q (canary rate, stable rate, sample)", reason, want)
			}
		}
	})
}

func TestGuardTripsOnRelaxationRate(t *testing.T) {
	const ms = time.Millisecond
	g := NewGuard(newGuardClock().now)
	observeN(g, "stable", 1000, 0, 100, 100*ms) // 10% relaxed
	observeN(g, "canary", 200, 0, 30, 100*ms)   // 15%: exactly 0.05 above
	mustNotTrip(t, g, "exactly 0.05 above")

	g = NewGuard(newGuardClock().now)
	observeN(g, "stable", 1000, 0, 100, 100*ms)
	observeN(g, "canary", 200, 0, 31, 100*ms) // 15.5%
	mustTrip(t, g, "0.055 above", "relax")
}

func TestGuardTripsOnLatency(t *testing.T) {
	const ms = time.Millisecond
	t.Run("p95 above 1.2 times stable trips", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 200, 0, 0, 130*ms) // 1.3 times
		mustTrip(t, g, "1.3 times", "latency")
	})
	t.Run("p95 within 1.2 times stable is quiet", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 200, 0, 0, 110*ms)
		mustNotTrip(t, g, "1.1 times")
	})
	t.Run("a slow tail below the 95th percentile does not move p95", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 190, 0, 0, 100*ms)
		observeN(g, "canary", 10, 0, 0, 5000*ms) // 5 percent: the 95th percentile is still 100 ms
		mustNotTrip(t, g, "5 percent slow tail")
	})
	t.Run("a slow tail above the 95th percentile does", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 180, 0, 0, 100*ms)
		observeN(g, "canary", 20, 0, 0, 5000*ms) // 10 percent
		mustTrip(t, g, "10 percent slow tail", "latency")
	})
	t.Run("the estimate is a percentile of the whole window, not of one sample", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 1000*ms)
		observeN(g, "canary", 200, 0, 0, 1150*ms) // 1.15 times, inside the limit
		mustNotTrip(t, g, "1.15 times")
		g = NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 1000*ms)
		observeN(g, "canary", 200, 0, 0, 1250*ms) // 1.25 times, outside it
		mustTrip(t, g, "1.25 times", "latency")
	})
}

// TestGuardNeedsMinimumSample: nothing trips before the canary has 200 requests, however bad
// it looks, and the guard cannot compare against a stable policy it has not seen.
func TestGuardNeedsMinimumSample(t *testing.T) {
	const ms = time.Millisecond
	g := NewGuard(newGuardClock().now)
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 199, 199, 199, 10000*ms) // every request bad, one short
	mustNotTrip(t, g, "199 canary requests")
	observeN(g, "canary", 1, 1, 1, 10000*ms)
	mustTrip(t, g, "200 canary requests", "error rate")

	t.Run("no stable requests to compare with", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "canary", 500, 500, 500, 10000*ms)
		mustNotTrip(t, g, "a canary and no stable traffic")
	})
	t.Run("feedback needs its own 200 labels", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 500, 0, 0, 100*ms)
		labelN(g, "stable", 2000, 1900)
		labelN(g, "canary", 199, 100) // far worse, one label short
		mustNotTrip(t, g, "199 canary labels")
		labelN(g, "canary", 1, 0)
		mustTrip(t, g, "200 canary labels", "feedback")
	})
	t.Run("labels do not count as requests", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 100, 100, 100, 10000*ms) // 100 requests, all bad
		labelN(g, "canary", 500, 0)                    // 500 labels, none of them a request
		mustNotTrip(t, g, "100 requests and 500 labels")
	})
	t.Run("an unknown role is ignored", func(t *testing.T) {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "shadow", 500, 500, 500, 10000*ms)
		observeN(g, "", 500, 500, 500, 10000*ms)
		observeN(g, "canary", 100, 0, 0, 100*ms)
		mustNotTrip(t, g, "shadow and empty roles")
	})
}

// TestGuardFeedbackZTest: with 200 canary labels, success lower than stable by more than 0.02
// trips only when a one-sided two-proportion z-test gives z above 1.645.
func TestGuardFeedbackZTest(t *testing.T) {
	const ms = time.Millisecond
	setup := func(stableN, stableOK, canaryN, canaryOK int) *Guard {
		g := NewGuard(newGuardClock().now)
		observeN(g, "stable", 1000, 0, 0, 100*ms)
		observeN(g, "canary", 200, 0, 0, 100*ms)
		labelN(g, "stable", stableN, stableOK)
		labelN(g, "canary", canaryN, canaryOK)
		return g
	}
	t.Run("a clear difference trips", func(t *testing.T) {
		// 90% against 85%: the difference is 0.05, z is about 2.1.
		mustTrip(t, setup(1000, 900, 200, 170), "0.05 lower, z 2.1", "feedback")
	})
	t.Run("more than 0.02 lower but z at most 1.645 is noise", func(t *testing.T) {
		// 90% against 87%: the difference is 0.03, z is about 1.26.
		mustNotTrip(t, setup(1000, 900, 200, 174), "0.03 lower, z 1.26")
	})
	t.Run("z above 1.645 but a difference of at most 0.02 is too small to matter", func(t *testing.T) {
		// 90% against 88.5% on large samples: the difference is 0.015, z is about 5.
		mustNotTrip(t, setup(100000, 90000, 20000, 17700), "0.015 lower, z 5")
	})
	t.Run("a better canary never trips", func(t *testing.T) {
		mustNotTrip(t, setup(1000, 800, 200, 190), "better canary")
	})
	t.Run("identical rates have no spread to test", func(t *testing.T) {
		mustNotTrip(t, setup(1000, 1000, 200, 200), "all successes on both sides")
		mustNotTrip(t, setup(1000, 0, 200, 0), "all failures on both sides")
	})
	t.Run("no stable labels to compare with", func(t *testing.T) {
		mustNotTrip(t, setup(0, 0, 200, 0), "canary labels only")
	})
}

// TestGuardWindowIsTenMinutes: an observation counts for ten minutes, in one-minute buckets.
func TestGuardWindowIsTenMinutes(t *testing.T) {
	const ms = time.Millisecond
	clock := newGuardClock() // 12:00:30
	g := NewGuard(clock.now)
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 200, 100, 0, 100*ms)

	clock.advance(9*time.Minute + 29*time.Second) // 12:09:59: still in the window
	mustTrip(t, g, "after 9m29s", "error rate")

	g.Reset()
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 200, 100, 0, 100*ms) // recorded at 12:09:59
	clock.advance(time.Minute)                 // 12:10:59: that minute is 1 minute old, the window is 10
	mustTrip(t, g, "bucket still inside", "error rate")
	clock.advance(9 * time.Minute) // 12:19:59: the 12:09 bucket is now 10 minutes old
	mustNotTrip(t, g, "after the bucket left the window")
}

func TestGuardOldObservationsAreForgotten(t *testing.T) {
	const ms = time.Millisecond
	clock := newGuardClock()
	g := NewGuard(clock.now)
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 150, 150, 0, 100*ms)
	clock.advance(8 * time.Minute)
	observeN(g, "canary", 100, 100, 0, 100*ms) // 250 bad in the window
	mustTrip(t, g, "250 bad requests over eight minutes", "error rate")

	g.Reset()
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 150, 150, 0, 100*ms)
	clock.advance(11 * time.Minute) // the first 150 are gone
	observeN(g, "canary", 100, 100, 0, 100*ms)
	mustNotTrip(t, g, "only 100 canary requests left in the window")
}

// TestGuardTripsOnceThenStaysQuiet: a tripped guard stays quiet until it is reset, which is
// what the gateway does when the canary percent is raised again.
func TestGuardTripsOnceThenStaysQuiet(t *testing.T) {
	const ms = time.Millisecond
	g := NewGuard(newGuardClock().now)
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 200, 200, 0, 100*ms)
	mustTrip(t, g, "first evaluation", "error rate")
	for i := 0; i < 3; i++ {
		mustNotTrip(t, g, "evaluation after the trip")
	}

	g.Reset()
	mustNotTrip(t, g, "after Reset the window is empty")
	observeN(g, "stable", 1000, 0, 0, 100*ms)
	observeN(g, "canary", 200, 200, 0, 100*ms)
	mustTrip(t, g, "a new bad canary after Reset", "error rate")
}

// TestGuardStats: the counts the guard holds for a role, as the gateway tests read them.
func TestGuardStats(t *testing.T) {
	const ms = time.Millisecond
	clock := newGuardClock()
	g := NewGuard(clock.now)
	observeN(g, "canary", 10, 3, 2, 100*ms)
	observeN(g, "canary", 10, 0, 0, 300*ms)
	labelN(g, "canary", 5, 4)
	observeN(g, "stable", 7, 0, 0, 50*ms)

	got := g.Stats("canary")
	if got.Requests != 20 || got.Errors != 3 || got.Relaxed != 2 || got.Labels != 5 || got.Successes != 4 {
		t.Errorf("canary stats = %+v, want 20 requests, 3 errors, 2 relaxed, 5 labels, 4 successes", got)
	}
	if lo, hi := 300*ms, 303*ms; got.P95 < lo || got.P95 > hi {
		t.Errorf("canary p95 = %v, want about 300ms (at most 1 percent above)", got.P95)
	}
	if s := g.Stats("stable"); s.Requests != 7 || s.Errors != 0 || s.Labels != 0 {
		t.Errorf("stable stats = %+v, want 7 requests", s)
	}
	if s := g.Stats("shadow"); s != (GuardStats{}) {
		t.Errorf("shadow stats = %+v, want zero: it is not recorded", s)
	}

	clock.advance(10 * time.Minute)
	if s := g.Stats("canary"); s != (GuardStats{}) {
		t.Errorf("canary stats after the window = %+v, want zero", s)
	}
}

func TestGuardConcurrentUse(t *testing.T) {
	g := NewGuard(time.Now)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			role := "stable"
			if w%2 == 0 {
				role = "canary"
			}
			ok := w%3 == 0
			for i := 0; i < 500; i++ {
				g.Observe(role, Observation{Error: i%50 == 0, Latency: time.Duration(i) * time.Millisecond})
				g.Observe(role, Observation{Feedback: &ok})
				if i%100 == 0 {
					g.Evaluate()
				}
			}
		}(w)
	}
	wg.Wait()
	g.Reset()
}
