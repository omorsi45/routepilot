package latency

import (
	"math"
	"sync"
	"time"
)

// Histogram geometry: log-spaced buckets, 20 per decade, from 0.1 ms to 1,000,000 ms.
// A value at or below the lowest edge lands in the first bucket and one above the
// highest edge in the last, so an extreme sample is counted but its quantile is clipped.
const (
	bucketsPerDecade = 20
	minBucketMs      = 0.1
	decades          = 7
	numBuckets       = bucketsPerDecade * decades
)

const (
	// HalfLife is the half-life of an observation's weight.
	HalfLife = 5 * time.Minute
	// MinObservations is the effective count a deployment's histogram needs before its
	// quantile replaces the model's prior.
	MinObservations = 50
	// minDelta is the smallest delta E2EQuantile accepts: the output-length quantile it
	// is given is the stored 0.99 one, which bounds the 1-delta/3 quantile only when
	// 1-delta/3 <= 0.99.
	minDelta = 0.03
)

// bucketIndex is the bucket of a value in milliseconds.
func bucketIndex(ms float64) int {
	if !(ms > minBucketMs) {
		return 0
	}
	i := int(math.Floor(math.Log10(ms/minBucketMs) * bucketsPerDecade))
	return min(max(i, 0), numBuckets-1)
}

// upperEdge is the upper edge in milliseconds of bucket i.
func upperEdge(i int) float64 {
	return minBucketMs * math.Pow(10, float64(i+1)/bucketsPerDecade)
}

// histogram holds exponentially decayed counts. Decay is applied lazily, whenever the
// histogram is touched, from the time of the previous touch.
type histogram struct {
	w     [numBuckets]float64
	total float64
	at    time.Time
}

func (h *histogram) decayTo(now time.Time) {
	if h.at.IsZero() {
		h.at = now
		return
	}
	dt := now.Sub(h.at)
	if dt <= 0 {
		return
	}
	f := math.Exp2(-dt.Seconds() / HalfLife.Seconds())
	for i := range h.w {
		h.w[i] *= f
	}
	h.total *= f
	h.at = now
}

func (h *histogram) add(ms float64) {
	h.w[bucketIndex(ms)]++
	h.total++
}

// quantile is the upper edge of the first bucket whose cumulative weight reaches u of
// the total. The histogram must hold weight.
func (h *histogram) quantile(u float64) float64 {
	target := u * h.total
	var cum float64
	last := 0
	for i, w := range h.w {
		if w == 0 {
			continue
		}
		last = i
		cum += w
		if cum >= target {
			return upperEdge(i)
		}
	}
	return upperEdge(last) // rounding left the cumulative sum just below the target
}

// depState keeps the two histograms of one deployment.
type depState struct {
	ttft, tpot histogram
}

// Estimator estimates latency quantiles per model from the observed time to first token
// and time per token of each deployment, with the model's prior profile standing in
// until a deployment has enough data. It is safe for concurrent use.
type Estimator struct {
	priors map[string]Profile
	now    func() time.Time

	mu          sync.Mutex
	deployments map[string]*depState
}

// NewEstimator returns an Estimator with the given priors, keyed by model ID. A nil now
// means time.Now. The priors map is copied.
func NewEstimator(priors map[string]Profile, now func() time.Time) *Estimator {
	if now == nil {
		now = time.Now
	}
	cp := make(map[string]Profile, len(priors))
	for k, v := range priors {
		cp[k] = v
	}
	return &Estimator{priors: cp, now: now, deployments: make(map[string]*depState)}
}

// ObserveTTFT records the time to first token of a request served by deployment. A
// non-positive duration is ignored.
func (e *Estimator) ObserveTTFT(deployment string, d time.Duration) {
	e.observe(deployment, d, func(dep *depState) *histogram { return &dep.ttft })
}

// ObserveTPOT records the time per output token of a request served by deployment. A
// non-positive duration is ignored.
func (e *Estimator) ObserveTPOT(deployment string, d time.Duration) {
	e.observe(deployment, d, func(dep *depState) *histogram { return &dep.tpot })
}

func (e *Estimator) observe(name string, d time.Duration, pick func(*depState) *histogram) {
	if d <= 0 {
		return
	}
	now := e.now()
	e.mu.Lock()
	defer e.mu.Unlock()
	dep := e.deployments[name]
	if dep == nil {
		dep = &depState{}
		e.deployments[name] = dep
	}
	h := pick(dep)
	h.decayTo(now)
	h.add(float64(d) / float64(time.Millisecond))
}

// field picks one of the two quantities and its prior parameters.
type field struct {
	hist  func(*depState) *histogram
	prior func(Profile) (median, sigma float64)
}

var (
	ttftField = field{
		hist:  func(d *depState) *histogram { return &d.ttft },
		prior: func(p Profile) (float64, float64) { return p.TTFTMedianMs, p.TTFTSigma },
	}
	tpotField = field{
		hist:  func(d *depState) *histogram { return &d.tpot },
		prior: func(p Profile) (float64, float64) { return p.TPOTMedianMs, p.TPOTSigma },
	}
)

// quantile is the maximum over deployments of the u-quantile of f: the deployment's own
// histogram when its effective count has reached MinObservations, else the model's
// prior. A deployment with neither contributes nothing. fromPrior reports whether any
// contribution came from the prior. ok is false when nothing contributed.
func (e *Estimator) quantile(model string, deployments []string, u float64, f field) (value float64, fromPrior, ok bool) {
	if !(u > 0 && u < 1) {
		return 0, false, false
	}
	now := e.now()
	prior, hasPrior := e.priors[model]

	e.mu.Lock()
	defer e.mu.Unlock()
	for _, name := range deployments {
		var q float64
		fromData := false
		if dep := e.deployments[name]; dep != nil {
			h := f.hist(dep)
			h.decayTo(now)
			if h.total >= MinObservations {
				q, fromData = h.quantile(u), true
			}
		}
		if !fromData {
			if !hasPrior {
				continue
			}
			median, sigma := f.prior(prior)
			q = LogNormalQuantile(median, sigma, u)
			fromPrior = true
		}
		if !ok || q > value {
			value = q
		}
		ok = true
	}
	return value, fromPrior, ok
}

// Quantiles returns the u-quantiles in milliseconds of the time to first token and the
// time per output token of model. Failover may land on any deployment, so each is the
// maximum over the given deployments (Ruling V1-R9a); a deployment with fewer than
// MinObservations effective observations contributes the model's prior profile quantile,
// and one with neither data nor prior contributes nothing. ok is false when no
// deployment contributes to either quantity, or u is outside (0, 1).
func (e *Estimator) Quantiles(model string, deployments []string, u float64) (ttftMs, tpotMs float64, ok bool) {
	ttft, _, okTTFT := e.quantile(model, deployments, u, ttftField)
	tpot, _, okTPOT := e.quantile(model, deployments, u, tpotField)
	if !okTTFT || !okTPOT {
		return 0, 0, false
	}
	return ttft, tpot, true
}

// UsesPrior reports whether the estimate for model over deployments takes any value from
// the model's prior profile, which is a simulated figure and not a measurement: some
// deployment has fewer than MinObservations effective TTFT observations (or TPOT ones,
// when tpot is true) and the model has a prior.
func (e *Estimator) UsesPrior(model string, deployments []string, tpot bool) bool {
	// The quantile level does not matter for the question; any level inside (0, 1) works.
	_, ttftPrior, _ := e.quantile(model, deployments, 0.5, ttftField)
	if ttftPrior || !tpot {
		return ttftPrior
	}
	_, tpotPrior, _ := e.quantile(model, deployments, 0.5, tpotField)
	return tpotPrior
}

// E2EQuantile bounds the 1-delta quantile of the end-to-end time of model by the union
// bound: the time to first token and the time per token each at the 1-delta/3 quantile
// (maximum over deployments), and outQ99, an upper bound on the number of output tokens,
// standing in for its 1-delta/3 quantile. The stored 0.99 quantile of the output length
// is such a bound when 1-delta/3 <= 0.99, which is why delta below 0.03 returns false
// (Ruling V1-R9). Pass delta 0.05 for the p95 the router checks. ok is false without an
// estimate, with a nil estimator or for a delta outside [0.03, 1).
func E2EQuantile(e *Estimator, model string, deps []string, outQ99 float64, delta float64) (float64, bool) {
	if e == nil || !(delta >= minDelta && delta < 1) {
		return 0, false
	}
	u := 1 - delta/3
	ttft, _, okTTFT := e.quantile(model, deps, u, ttftField)
	tpot, _, okTPOT := e.quantile(model, deps, u, tpotField)
	if !okTTFT || !okTPOT {
		return 0, false
	}
	return UnionBound(ttft, outQ99, tpot), true
}

// TTFTQuantile is the 1-delta quantile of the time to first token of model in
// milliseconds, the maximum over deps. ok is false without an estimate, with a nil
// estimator or for a delta outside (0, 1).
func TTFTQuantile(e *Estimator, model string, deps []string, delta float64) (float64, bool) {
	if e == nil || !(delta > 0 && delta < 1) {
		return 0, false
	}
	ttft, _, ok := e.quantile(model, deps, 1-delta, ttftField)
	return ttft, ok
}
