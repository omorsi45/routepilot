package policy

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// The guardrail thresholds. The canary is judged against the stable policy over a
// window of the last ten minutes, kept in one-minute buckets.
const (
	guardWindowMinutes = 10
	// guardMinSample is the number of canary requests (or feedback labels) the guard needs
	// before it judges anything.
	guardMinSample = 200
	// The canary trips when its error rate is more than guardErrorMargin above stable's, its
	// relaxation rate more than guardRelaxMargin above, or its p95 latency more than
	// guardLatencyRatio times stable's.
	guardErrorMargin  = 0.02
	guardRelaxMargin  = 0.05
	guardLatencyRatio = 1.2
	// Feedback: the canary's success rate is lower than stable's by more than
	// guardSuccessMargin, and a one-sided two-proportion z-test gives z above guardZ.
	guardSuccessMargin = 0.02
	guardZ             = 1.645
	// guardEpsilon keeps a rate difference that is exactly a margin in decimal from tripping
	// because of binary rounding (0.07 - 0.05 is 0.020000000000000004).
	guardEpsilon = 1e-9
)

// The latency histogram has a bucket for every 1 percent of growth from 1 microsecond, so a
// percentile it reports is at most 1 percent above the true value. It needs no allocation per
// observation and its size does not depend on the request rate.
const (
	latencyGrowth  = 1.01
	latencyBuckets = 2112 // the last bucket reaches about 22 minutes
)

var latencyLogGrowth = math.Log(latencyGrowth)

// latencyIndex is the histogram bucket of d: the smallest i with latencyGrowth^i
// microseconds at or above d.
func latencyIndex(d time.Duration) int {
	us := float64(d) / float64(time.Microsecond)
	if !(us > 1) { // also zero, negative and NaN
		return 0
	}
	return min(int(math.Ceil(math.Log(us)/latencyLogGrowth)), latencyBuckets-1)
}

// latencyUpper is the upper bound of histogram bucket i.
func latencyUpper(i int) time.Duration {
	return time.Duration(math.Pow(latencyGrowth, float64(i)) * float64(time.Microsecond))
}

// Observation is what the guard learns about one request, or about one verdict on a request.
type Observation struct {
	// Error is true when the request failed on the policy's side: a server error, or a
	// routing error other than a client validation error.
	Error bool
	// Relaxed is true when the policy gave up a constraint (latency or quality) to answer.
	Relaxed bool
	// Latency is the total time the request took.
	Latency time.Duration
	// Feedback, when set, makes the observation a client's verdict on a request the guard has
	// already seen. It is counted as a feedback label and nothing else: the other fields are
	// ignored, so a label is never a second request.
	Feedback *bool
}

// Guard compares the canary policy with the stable one over the last ten minutes and says
// when the canary should be pulled. It is safe for concurrent use.
//
// Only the roles "stable" and "canary" are recorded. Once Evaluate has tripped, it stays
// quiet until Reset, so one bad canary is reported once.
type Guard struct {
	now func() time.Time

	mu      sync.Mutex
	stable  window
	canary  window
	tripped bool
}

// NewGuard returns a Guard that reads time from now.
func NewGuard(now func() time.Time) *Guard {
	if now == nil {
		now = time.Now
	}
	return &Guard{now: now}
}

// window is one role's last ten minutes in one-minute buckets, a ring indexed by the minute.
type window struct {
	buckets [guardWindowMinutes]bucket
}

type bucket struct {
	used   bool
	minute int64 // Unix minute
	// requests, requests that failed, requests that relaxed, and the feedback labels.
	n, errs, relaxed int
	labels, labelsOK int
	latency          [latencyBuckets]uint32
}

// totals are a window's counts.
type totals struct {
	n, errs, relaxed int
	labels, labelsOK int
}

func (g *Guard) windowOf(role string) *window {
	switch role {
	case RoleStable:
		return &g.stable
	case RoleCanary:
		return &g.canary
	}
	return nil
}

func unixMinute(t time.Time) int64 { return t.Unix() / 60 }

// Observe records o for role, which is RoleStable or RoleCanary; any other role is ignored.
func (g *Guard) Observe(role string, o Observation) {
	w := g.windowOf(role)
	if w == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	m := unixMinute(g.now())
	b := &w.buckets[((m%guardWindowMinutes)+guardWindowMinutes)%guardWindowMinutes]
	if !b.used || b.minute != m {
		*b = bucket{used: true, minute: m}
	}
	if o.Feedback != nil {
		b.labels++
		if *o.Feedback {
			b.labelsOK++
		}
		return
	}
	b.n++
	if o.Error {
		b.errs++
	}
	if o.Relaxed {
		b.relaxed++
	}
	b.latency[latencyIndex(o.Latency)]++
}

// live reports whether b is inside the window that ends in minute now: the current minute
// and the nine before it.
func (b *bucket) live(now int64) bool {
	return b.used && b.minute > now-guardWindowMinutes && b.minute <= now
}

func (w *window) totals(now int64) totals {
	var t totals
	for i := range w.buckets {
		if b := &w.buckets[i]; b.live(now) {
			t.n += b.n
			t.errs += b.errs
			t.relaxed += b.relaxed
			t.labels += b.labels
			t.labelsOK += b.labelsOK
		}
	}
	return t
}

// p95 is the 95th percentile latency of the window's requests (the nearest rank), at most 1
// percent above the true value.
func (w *window) p95(now int64, n int) time.Duration {
	var counts [latencyBuckets]uint32
	for i := range w.buckets {
		if b := &w.buckets[i]; b.live(now) {
			for j, c := range b.latency {
				counts[j] += c
			}
		}
	}
	rank := (95*n + 99) / 100 // ceil(0.95 n)
	cum := 0
	for i, c := range counts {
		cum += int(c)
		if cum >= rank {
			return latencyUpper(i)
		}
	}
	return latencyUpper(latencyBuckets - 1)
}

// exceeds reports whether diff is more than limit, ignoring binary rounding.
func exceeds(diff, limit float64) bool { return diff > limit+guardEpsilon }

// Evaluate judges the canary against stable over the window. With at least 200 canary
// requests (and some stable ones to compare with) it trips when the canary's error rate is
// more than 0.02 above stable's, its relaxation rate more than 0.05 above, or its p95
// latency more than 1.2 times stable's. With at least 200 canary feedback labels (and some
// stable ones) it also trips when the canary's success rate is lower by more than 0.02 and a
// one-sided two-proportion z-test gives z above 1.645.
//
// It returns true and a reason when a rule trips. The reason holds counts, rates and
// latencies only. After a trip it returns false until Reset.
func (g *Guard) Evaluate() (trip bool, reason string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.tripped {
		return false, ""
	}
	now := unixMinute(g.now())
	st, ca := g.stable.totals(now), g.canary.totals(now)

	if st.n > 0 && ca.n >= guardMinSample {
		stRate, caRate := rate(st.errs, st.n), rate(ca.errs, ca.n)
		if exceeds(caRate-stRate, guardErrorMargin) {
			return g.trip(fmt.Sprintf("canary error rate %.4f is more than %.2f above stable %.4f (canary %d requests, stable %d)",
				caRate, guardErrorMargin, stRate, ca.n, st.n))
		}
		stRate, caRate = rate(st.relaxed, st.n), rate(ca.relaxed, ca.n)
		if exceeds(caRate-stRate, guardRelaxMargin) {
			return g.trip(fmt.Sprintf("canary relaxation rate %.4f is more than %.2f above stable %.4f (canary %d requests, stable %d)",
				caRate, guardRelaxMargin, stRate, ca.n, st.n))
		}
		stP95, caP95 := g.stable.p95(now, st.n), g.canary.p95(now, ca.n)
		if float64(caP95) > guardLatencyRatio*float64(stP95) {
			return g.trip(fmt.Sprintf("canary p95 latency %.1f ms is more than %.1f times stable %.1f ms (canary %d requests, stable %d)",
				ms(caP95), guardLatencyRatio, ms(stP95), ca.n, st.n))
		}
	}

	if st.labels > 0 && ca.labels >= guardMinSample {
		stOK, caOK := rate(st.labelsOK, st.labels), rate(ca.labelsOK, ca.labels)
		if exceeds(stOK-caOK, guardSuccessMargin) {
			if z := twoProportionZ(stOK, st.labels, caOK, ca.labels, st.labelsOK, ca.labelsOK); z > guardZ {
				return g.trip(fmt.Sprintf("canary feedback success %.4f is lower than stable %.4f by more than %.2f, z = %.2f (canary %d labels, stable %d)",
					caOK, stOK, guardSuccessMargin, z, ca.labels, st.labels))
			}
		}
	}
	return false, ""
}

func (g *Guard) trip(reason string) (bool, string) {
	g.tripped = true
	return true, reason
}

// GuardStats is what the guard holds for one role over the current window.
type GuardStats struct {
	Requests, Errors, Relaxed int
	// Labels is the number of feedback labels and Successes how many of them said success.
	Labels, Successes int
	// P95 is the 95th percentile latency of the requests, 0 without any.
	P95 time.Duration
}

// Stats returns the counts the guard holds for role over the window that ends now. A role
// the guard does not record (anything but stable and canary) has zero stats.
func (g *Guard) Stats(role string) GuardStats {
	w := g.windowOf(role)
	if w == nil {
		return GuardStats{}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	now := unixMinute(g.now())
	t := w.totals(now)
	s := GuardStats{Requests: t.n, Errors: t.errs, Relaxed: t.relaxed, Labels: t.labels, Successes: t.labelsOK}
	if t.n > 0 {
		s.P95 = w.p95(now, t.n)
	}
	return s
}

// Reset forgets every observation and re-arms a tripped guard. The gateway calls it when the
// canary changes or its share is raised again, so a new canary is not judged on the last
// one's requests.
func (g *Guard) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.stable, g.canary, g.tripped = window{}, window{}, false
}

func rate(k, n int) float64 {
	if n == 0 {
		return 0
	}
	return float64(k) / float64(n)
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// twoProportionZ is the z statistic of the one-sided test that a (the stable success rate,
// over na labels with ka successes) is higher than b (the canary's, over nb with kb): the
// difference over the standard error under the pooled rate. It is 0 when the pooled rate
// leaves no spread to test (everything succeeded, or everything failed).
func twoProportionZ(a float64, na int, b float64, nb int, ka, kb int) float64 {
	pooled := float64(ka+kb) / float64(na+nb)
	se := math.Sqrt(pooled * (1 - pooled) * (1/float64(na) + 1/float64(nb)))
	if !(se > 0) {
		return 0
	}
	return (a - b) / se
}
