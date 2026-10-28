package latency

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func relErr(got, want float64) float64 {
	return math.Abs(got-want) / math.Max(math.Abs(want), 1e-300)
}

// fakeClock is a settable clock for the estimator.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// testPrior is a test-only profile. The numbers are not a measurement of any model.
func testPrior(model string, ttftMedian, ttftSigma, tpotMedian, tpotSigma float64) Profile {
	return Profile{
		Model: model, TTFTMedianMs: ttftMedian, TTFTSigma: ttftSigma,
		TPOTMedianMs: tpotMedian, TPOTSigma: tpotSigma,
		Source: "test", Retrieved: "2026-10-28",
	}
}

func observeN(e *Estimator, dep string, n int, ttft, tpot time.Duration) {
	for i := 0; i < n; i++ {
		e.ObserveTTFT(dep, ttft)
		e.ObserveTPOT(dep, tpot)
	}
}

func TestLogNormalQuantileKnown(t *testing.T) {
	const median, sigma = 800.0, 0.5
	if got := LogNormalQuantile(median, sigma, 0.5); got != median {
		t.Errorf("u=0.5: got %v, want exactly the median %v", got, median)
	}
	want := median * math.Exp(1.959964*sigma)
	if got := LogNormalQuantile(median, sigma, 0.975); relErr(got, want) > 1e-6 {
		t.Errorf("u=0.975: got %v, want %v (median*exp(1.959964*sigma))", got, want)
	}
	// The distribution of ln(x) is symmetric around ln(median).
	lo, hi := LogNormalQuantile(median, sigma, 0.1), LogNormalQuantile(median, sigma, 0.9)
	if relErr(lo*hi, median*median) > 1e-12 {
		t.Errorf("q(0.1)*q(0.9) = %v, want median^2 = %v", lo*hi, median*median)
	}
	if got := LogNormalQuantile(median, 0, 0.99); got != median {
		t.Errorf("sigma 0: got %v, want the median", got)
	}
	prev := 0.0
	for _, u := range []float64{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999} {
		got := LogNormalQuantile(median, sigma, u)
		if got <= prev {
			t.Errorf("quantile at u=%v is %v, not above the previous %v", u, got, prev)
		}
		prev = got
	}
	if got := LogNormalQuantile(median, sigma, 0); got != 0 {
		t.Errorf("u=0: got %v, want 0", got)
	}
	if got := LogNormalQuantile(median, sigma, 1); !math.IsInf(got, 1) {
		t.Errorf("u=1: got %v, want +Inf", got)
	}
	if got := LogNormalQuantile(median, sigma, math.NaN()); !math.IsNaN(got) {
		t.Errorf("u=NaN: got %v, want NaN", got)
	}
}

// TestNormalQuantileAccuracy checks the Acklam approximation with its Halley step three
// ways: against well-known constants, against the standard library's inverse error
// function away from the extreme tails (where math.Erfcinv, computed as Erfinv(1-x), loses
// digits itself), and by feeding the result back through the CDF, which is accurate in
// the tails too.
func TestNormalQuantileAccuracy(t *testing.T) {
	for _, k := range []struct{ p, z float64 }{
		{0.5, 0}, {0.975, 1.959963984540054}, {0.995, 2.5758293035489004}, {0.999, 3.090232306167813},
		{0.025, -1.959963984540054}, {0.001, -3.090232306167813},
	} {
		if got := normalQuantile(k.p); math.Abs(got-k.z) > 1e-9 {
			t.Errorf("normalQuantile(%v) = %.15g, want %.15g", k.p, got, k.z)
		}
	}

	for _, p := range []float64{1e-5, 1e-3, 0.01, 0.02424, 0.02425, 0.02426, 0.025, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.97575, 0.99, 0.999, 1 - 1e-5} {
		var want float64
		if p <= 0.5 {
			want = -math.Sqrt2 * math.Erfcinv(2*p)
		} else {
			want = math.Sqrt2 * math.Erfcinv(2*(1-p))
		}
		if got := normalQuantile(p); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("normalQuantile(%v) = %.15g, stdlib says %.15g", p, got, want)
		}
	}

	for _, p := range []float64{1e-15, 1e-12, 1e-9, 1e-6, 0.01, 0.02424, 0.02426, 0.3, 0.5} {
		x := normalQuantile(p)
		if cdf := 0.5 * math.Erfc(-x/math.Sqrt2); relErr(cdf, p) > 1e-9 {
			t.Errorf("Phi(normalQuantile(%v)) = %.15g, relative error %.2g", p, cdf, relErr(cdf, p))
		}

	}
}
func TestUnionBound(t *testing.T) {
	if got := UnionBound(100, 10, 5); got != 150 {
		t.Errorf("UnionBound(100, 10, 5) = %v, want 150", got)
	}
}

// TestUnionBoundIsUpperBound draws independent TTFT, output length and TPOT, and checks
// that the union bound at 1-delta/3 each sits above the empirical 1-delta quantile of the
// end-to-end time. The draws use the standard library's normal generator, not the
// quantile function under test.
func TestUnionBoundIsUpperBound(t *testing.T) {
	const (
		draws = 100_000
		delta = 0.05
	)
	rng := rand.New(rand.NewPCG(11, 13))
	ttft := func() float64 { return 300 * math.Exp(0.5*rng.NormFloat64()) }
	tokens := func() float64 { return 200 * math.Exp(0.6*rng.NormFloat64()) }
	tpot := func() float64 { return 20 * math.Exp(0.3*rng.NormFloat64()) }

	u := 1 - delta/3
	bound := UnionBound(LogNormalQuantile(300, 0.5, u), LogNormalQuantile(200, 0.6, u), LogNormalQuantile(20, 0.3, u))

	e2e := make([]float64, draws)
	over := 0
	for i := range e2e {
		e2e[i] = ttft() + tokens()*tpot()
		if e2e[i] > bound {
			over++
		}
	}
	slices.Sort(e2e)
	q95 := e2e[int(0.95*draws)]
	if q95 > bound {
		t.Errorf("empirical 0.95 quantile %.1f ms above the union bound %.1f ms", q95, bound)
	}
	if frac := float64(over) / draws; frac > delta {
		t.Errorf("%.4f of the draws exceed the bound, want at most %v", frac, delta)
	}
	if bound > 4*q95 {
		t.Errorf("bound %.1f ms is more than 4x the empirical quantile %.1f ms: the test would pass for any large number", bound, q95)
	}
}

func TestProfileDraw(t *testing.T) {
	p := testPrior("m", 1000, 0.5, 20, 0.3)
	ttft, tpot := p.Draw(0.5, 0.5)
	if ttft != 1000 || tpot != 20 {
		t.Errorf("Draw(0.5, 0.5) = %v, %v; want the medians 1000, 20", ttft, tpot)
	}
	ttft, tpot = p.Draw(0.975, 0.025)
	if relErr(ttft, 1000*math.Exp(1.959964*0.5)) > 1e-6 || relErr(tpot, 20*math.Exp(-1.959964*0.3)) > 1e-6 {
		t.Errorf("Draw(0.975, 0.025) = %v, %v", ttft, tpot)
	}
}

func TestEstimatorFallsBackToPrior(t *testing.T) {
	clk := newFakeClock()
	prior := testPrior("m", 1000, 0.5, 20, 0.3)
	e := NewEstimator(map[string]Profile{"m": prior}, clk.now)
	wantTTFT, wantTPOT := LogNormalQuantile(1000, 0.5, 0.95), LogNormalQuantile(20, 0.3, 0.95)

	check := func(label string, wantTTFT, wantTPOT float64, tol float64) {
		t.Helper()
		ttft, tpot, ok := e.Quantiles("m", []string{"d1"}, 0.95)
		if !ok {
			t.Fatalf("%s: ok = false, want true", label)
		}
		if relErr(ttft, wantTTFT) > tol || relErr(tpot, wantTPOT) > tol {
			t.Errorf("%s: got ttft %v tpot %v, want %v %v", label, ttft, tpot, wantTTFT, wantTPOT)
		}
	}

	check("no observations", wantTTFT, wantTPOT, 1e-12)
	observeN(e, "d1", 49, 105*time.Millisecond, 7*time.Millisecond)
	check("49 observations", wantTTFT, wantTPOT, 1e-12)

	// The 50th observation switches to the data: 105 ms falls in the bucket whose upper
	// edge is 0.1*10^(61/20) = 112.2018 ms, and 7 ms in the one that ends at 7.0795 ms.
	e.ObserveTTFT("d1", 105*time.Millisecond)
	e.ObserveTPOT("d1", 7*time.Millisecond)
	check("50 observations", 112.2018454, 7.0794578, 1e-6)

	// The two histograms count separately.
	e2 := NewEstimator(map[string]Profile{"m": prior}, clk.now)
	for i := 0; i < 50; i++ {
		e2.ObserveTTFT("d1", 105*time.Millisecond)
	}
	ttft, tpot, ok := e2.Quantiles("m", []string{"d1"}, 0.95)
	if !ok || relErr(ttft, 112.2018454) > 1e-6 || relErr(tpot, wantTPOT) > 1e-12 {
		t.Errorf("TTFT-only data: got %v %v %v, want ttft from data and tpot from the prior", ttft, tpot, ok)
	}
}

func TestEstimatorHistogramRange(t *testing.T) {
	e := NewEstimator(nil, newFakeClock().now)
	observeN(e, "tiny", 50, time.Nanosecond, time.Nanosecond)
	observeN(e, "huge", 50, 2000*time.Second, 2000*time.Second)
	if ttft, _, ok := e.Quantiles("x", []string{"tiny"}, 0.5); !ok || relErr(ttft, 0.1122018) > 1e-5 {
		t.Errorf("below 0.1 ms: got %v %v, want the first bucket's upper edge 0.1122018", ttft, ok)
	}
	if ttft, _, ok := e.Quantiles("x", []string{"huge"}, 0.5); !ok || relErr(ttft, 1e6) > 1e-9 {
		t.Errorf("above 1e6 ms: got %v %v, want the last bucket's upper edge 1e6", ttft, ok)
	}
}

func TestEstimatorQuantileFollowsTheData(t *testing.T) {
	e := NewEstimator(nil, newFakeClock().now)
	// 90 fast and 10 slow observations: p50 is in the fast bucket, p95 in the slow one.
	observeN(e, "d1", 90, 105*time.Millisecond, 10*time.Millisecond)
	observeN(e, "d1", 10, 2100*time.Millisecond, 10*time.Millisecond)
	p50, _, _ := e.Quantiles("m", []string{"d1"}, 0.5)
	p95, _, _ := e.Quantiles("m", []string{"d1"}, 0.95)
	if relErr(p50, 112.2018454) > 1e-6 {
		t.Errorf("p50 = %v, want the 105 ms bucket's upper edge 112.2018", p50)
	}
	// 2100 ms: index floor(20*log10(21000)) = 86, upper edge 0.1*10^(87/20) = 2238.7211.
	if relErr(p95, 2238.7211386) > 1e-6 {
		t.Errorf("p95 = %v, want the 2100 ms bucket's upper edge 2238.7211", p95)
	}
}

func TestEstimatorDecay(t *testing.T) {
	clk := newFakeClock()
	prior := testPrior("m", 1000, 0.5, 20, 0.3)
	e := NewEstimator(map[string]Profile{"m": prior}, clk.now)
	priorTTFT := LogNormalQuantile(1000, 0.5, 0.5)

	observeN(e, "d1", 100, 105*time.Millisecond, 7*time.Millisecond)
	if ttft, _, _ := e.Quantiles("m", []string{"d1"}, 0.5); relErr(ttft, 112.2018454) > 1e-6 {
		t.Fatalf("fresh data: ttft = %v, want 112.2018", ttft)
	}
	// One half-life later the effective count is 50: still enough.
	clk.advance(5 * time.Minute)
	if ttft, _, _ := e.Quantiles("m", []string{"d1"}, 0.5); relErr(ttft, 112.2018454) > 1e-6 {
		t.Errorf("after one half-life (count 50): ttft = %v, want the data", ttft)
	}
	// A little more and the count drops below 50: the prior takes over again.
	clk.advance(time.Second)
	if ttft, _, _ := e.Quantiles("m", []string{"d1"}, 0.5); relErr(ttft, priorTTFT) > 1e-12 {
		t.Errorf("after one half-life and a second (count < 50): ttft = %v, want the prior %v", ttft, priorTTFT)
	}

	// Old observations lose weight against new ones: 60 old fast samples decayed over two
	// half-lives weigh 15, the 50 new slow samples weigh 50, so the median is the slow one.
	e = NewEstimator(nil, clk.now)
	observeN(e, "d1", 60, 105*time.Millisecond, 7*time.Millisecond)
	clk.advance(10 * time.Minute)
	observeN(e, "d1", 50, 2100*time.Millisecond, 7*time.Millisecond)
	if ttft, _, _ := e.Quantiles("m", []string{"d1"}, 0.5); relErr(ttft, 2238.7211386) > 1e-6 {
		t.Errorf("median after decay = %v, want the new 2100 ms bucket 2238.7211 (without decay it would be 112.2)", ttft)
	}
}

func TestEstimatorMaxAcrossDeployments(t *testing.T) {
	clk := newFakeClock()
	prior := testPrior("m", 1000, 0.5, 20, 0.3)
	e := NewEstimator(map[string]Profile{"m": prior}, clk.now)
	observeN(e, "fast", 60, 105*time.Millisecond, 7*time.Millisecond)
	observeN(e, "slow", 60, 2100*time.Millisecond, 70*time.Millisecond)
	priorTTFT, priorTPOT := LogNormalQuantile(1000, 0.5, 0.5), LogNormalQuantile(20, 0.3, 0.5)

	ttft, tpot, ok := e.Quantiles("m", []string{"fast", "slow"}, 0.5)
	if !ok || relErr(ttft, 2238.7211386) > 1e-6 || relErr(tpot, 70.7945784) > 1e-6 {
		t.Errorf("fast+slow: got %v %v %v, want the slow deployment's 2238.72 and 70.79", ttft, tpot, ok)
	}
	// The result does not depend on the order of the deployments.
	ttft2, tpot2, _ := e.Quantiles("m", []string{"slow", "fast"}, 0.5)
	if ttft2 != ttft || tpot2 != tpot {
		t.Errorf("reordered deployments changed the result: %v %v vs %v %v", ttft2, tpot2, ttft, tpot)
	}
	ttft, tpot, ok = e.Quantiles("m", []string{"fast"}, 0.5)
	if !ok || relErr(ttft, 112.2018454) > 1e-6 || relErr(tpot, 7.0794578) > 1e-6 {
		t.Errorf("fast only: got %v %v %v, want 112.2018 and 7.0795", ttft, tpot, ok)
	}
	// A deployment with no data uses the prior, and the maximum picks it when it is larger.
	ttft, tpot, ok = e.Quantiles("m", []string{"fast", "unseen"}, 0.5)
	if !ok || relErr(ttft, priorTTFT) > 1e-12 || relErr(tpot, priorTPOT) > 1e-12 {
		t.Errorf("fast+unseen: got %v %v %v, want the prior %v %v", ttft, tpot, ok, priorTTFT, priorTPOT)
	}
}

func TestEstimatorNoDataNoPrior(t *testing.T) {
	e := NewEstimator(map[string]Profile{"m": testPrior("m", 1000, 0.5, 20, 0.3)}, newFakeClock().now)
	if _, _, ok := e.Quantiles("ghost", []string{"d9"}, 0.95); ok {
		t.Error("a model with neither data nor prior: ok = true, want false")
	}
	if _, _, ok := e.Quantiles("m", nil, 0.95); ok {
		t.Error("no deployments: ok = true, want false")
	}
	for _, u := range []float64{0, 1, -0.1, 1.5, math.NaN()} {
		if _, _, ok := e.Quantiles("m", []string{"d1"}, u); ok {
			t.Errorf("u=%v: ok = true, want false", u)
		}
	}
	// With data but no prior the estimate exists only once the count is reached.
	observeN(e, "d9", 49, 105*time.Millisecond, 7*time.Millisecond)
	if _, _, ok := e.Quantiles("ghost", []string{"d9"}, 0.95); ok {
		t.Error("49 observations and no prior: ok = true, want false")
	}
	observeN(e, "d9", 1, 105*time.Millisecond, 7*time.Millisecond)
	if _, _, ok := e.Quantiles("ghost", []string{"d9"}, 0.95); !ok {
		t.Error("50 observations and no prior: ok = false, want true")
	}
	// A TTFT-only question needs no TPOT estimate.
	e = NewEstimator(nil, newFakeClock().now)
	for i := 0; i < 50; i++ {
		e.ObserveTTFT("d1", 105*time.Millisecond)
	}
	if v, ok := TTFTQuantile(e, "m", []string{"d1"}, 0.05); !ok || v <= 0 {
		t.Errorf("TTFTQuantile with TTFT data only: %v %v, want ok", v, ok)
	}
	if _, ok := E2EQuantile(e, "m", []string{"d1"}, 100, 0.05); ok {
		t.Error("E2EQuantile without TPOT data or prior: ok = true, want false")
	}
}

func TestEstimatorIgnoresNonPositiveDurations(t *testing.T) {
	e := NewEstimator(nil, newFakeClock().now)
	observeN(e, "d1", 100, 0, -time.Second)
	if _, _, ok := e.Quantiles("m", []string{"d1"}, 0.5); ok {
		t.Error("zero and negative durations counted as observations")
	}
}

func TestE2EAndTTFTQuantile(t *testing.T) {
	prior := testPrior("m", 400, 0.5, 50, 0.3)
	e := NewEstimator(map[string]Profile{"m": prior}, newFakeClock().now)
	deps := []string{"d1"}

	const outQ99 = 40.0
	u := 1 - 0.05/3
	want := UnionBound(LogNormalQuantile(400, 0.5, u), outQ99, LogNormalQuantile(50, 0.3, u))
	got, ok := E2EQuantile(e, "m", deps, outQ99, 0.05)
	if !ok || relErr(got, want) > 1e-12 {
		t.Errorf("E2EQuantile = %v %v, want %v", got, ok, want)
	}
	if got, ok := TTFTQuantile(e, "m", deps, 0.05); !ok || relErr(got, LogNormalQuantile(400, 0.5, 0.95)) > 1e-12 {
		t.Errorf("TTFTQuantile = %v %v, want the 0.95 quantile of the prior", got, ok)
	}
	// 1-delta/3 above 0.99 would make the stored 0.99 output quantile no upper bound.
	if _, ok := E2EQuantile(e, "m", deps, outQ99, 0.02); ok {
		t.Error("delta 0.02: ok = true, want false (0.99 is below 1-delta/3)")
	}
	for _, d := range []float64{0, 1, -1, math.NaN()} {
		if _, ok := E2EQuantile(e, "m", deps, outQ99, d); ok {
			t.Errorf("E2EQuantile with delta %v: ok = true, want false", d)
		}
		if _, ok := TTFTQuantile(e, "m", deps, d); ok {
			t.Errorf("TTFTQuantile with delta %v: ok = true, want false", d)
		}
	}
	if _, ok := E2EQuantile(nil, "m", deps, outQ99, 0.05); ok {
		t.Error("nil estimator: ok = true, want false")
	}
	if _, ok := TTFTQuantile(nil, "m", deps, 0.05); ok {
		t.Error("nil estimator: ok = true, want false")
	}
}

func TestUsesPrior(t *testing.T) {
	e := NewEstimator(map[string]Profile{"m": testPrior("m", 1000, 0.5, 20, 0.3)}, newFakeClock().now)
	if !e.UsesPrior("m", []string{"d1"}, false) {
		t.Error("no data: UsesPrior = false, want true")
	}
	if e.UsesPrior("ghost", []string{"d1"}, false) {
		t.Error("no prior for the model: UsesPrior = true, want false")
	}
	for i := 0; i < 50; i++ {
		e.ObserveTTFT("d1", 105*time.Millisecond)
	}
	if e.UsesPrior("m", []string{"d1"}, false) {
		t.Error("50 TTFT observations: UsesPrior(ttft) = true, want false")
	}
	if !e.UsesPrior("m", []string{"d1"}, true) {
		t.Error("no TPOT data: UsesPrior(with tpot) = false, want true")
	}
	if !e.UsesPrior("m", []string{"d1", "d2"}, false) {
		t.Error("d2 has no data: UsesPrior = false, want true")
	}
}

func TestEstimatorConcurrentUse(t *testing.T) {
	clk := newFakeClock()
	e := NewEstimator(map[string]Profile{"m": testPrior("m", 1000, 0.5, 20, 0.3)}, clk.now)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				e.ObserveTTFT("d1", time.Duration(100+i)*time.Millisecond)
				e.ObserveTPOT("d1", 10*time.Millisecond)
				e.Quantiles("m", []string{"d1", "d2"}, 0.95)
				E2EQuantile(e, "m", []string{"d1"}, 50, 0.05)
				clk.advance(time.Millisecond)
			}
		}()
	}
	wg.Wait()
	if _, _, ok := e.Quantiles("m", []string{"d1"}, 0.95); !ok {
		t.Error("Quantiles after concurrent use: ok = false")
	}
}

func writeProfiles(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "profiles.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const goodEntry = `
  - model: m1
    ttft_median_ms: 1000
    ttft_sigma: 0.5
    tpot_median_ms: 20
    tpot_sigma: 0.3
    source: https://example.invalid/m1
    retrieved: "2026-10-28"
    note: sigma assumed
`

func TestLoadProfiles(t *testing.T) {
	got, err := LoadProfiles(writeProfiles(t, "profiles:"+goodEntry))
	if err != nil {
		t.Fatalf("LoadProfiles: %v", err)
	}
	want := Profile{
		Model: "m1", TTFTMedianMs: 1000, TTFTSigma: 0.5, TPOTMedianMs: 20, TPOTSigma: 0.3,
		Source: "https://example.invalid/m1", Retrieved: "2026-10-28", Note: "sigma assumed",
	}
	if len(got) != 1 || got["m1"] != want {
		t.Errorf("got %+v, want {m1: %+v}", got, want)
	}

	// note is optional, sigma may be zero.
	noNote := strings.Replace(strings.Replace(goodEntry, "    note: sigma assumed\n", "", 1), "ttft_sigma: 0.5", "ttft_sigma: 0", 1)
	got, err = LoadProfiles(writeProfiles(t, "profiles:"+noNote))
	if err != nil || got["m1"].Note != "" || got["m1"].TTFTSigma != 0 {
		t.Errorf("optional note and zero sigma: %+v, %v", got, err)
	}
}

func TestLoadProfilesRejects(t *testing.T) {
	const noModel = "profiles:\n  - ttft_median_ms: 1000\n    ttft_sigma: 0.5\n    tpot_median_ms: 20\n    tpot_sigma: 0.3\n    source: x\n    retrieved: \"2026-10-28\"\n"
	without := func(key string) string {
		var keep []string
		for _, l := range strings.Split(goodEntry, "\n") {
			if !strings.Contains(l, key+":") {
				keep = append(keep, l)
			}
		}
		return "profiles:" + strings.Join(keep, "\n")
	}
	replace := func(old, new string) string {
		return "profiles:" + strings.Replace(goodEntry, old, new, 1)
	}
	cases := map[string]struct{ body, want string }{
		"empty file":          {"", "empty"},
		"no profiles":         {"profiles: []\n", "no profiles"},
		"missing model":       {noModel, "model"},
		"missing ttft median": {without("ttft_median_ms"), "ttft_median_ms"},
		"missing ttft sigma":  {without("ttft_sigma"), "ttft_sigma"},
		"missing tpot median": {without("tpot_median_ms"), "tpot_median_ms"},
		"missing tpot sigma":  {without("tpot_sigma"), "tpot_sigma"},
		"missing source":      {without("source"), "source"},
		"missing retrieved":   {without("retrieved"), "retrieved"},
		"zero ttft median":    {replace("ttft_median_ms: 1000", "ttft_median_ms: 0"), "ttft_median_ms"},
		"negative tpot":       {replace("tpot_median_ms: 20", "tpot_median_ms: -1"), "tpot_median_ms"},
		"negative sigma":      {replace("tpot_sigma: 0.3", "tpot_sigma: -0.1"), "tpot_sigma"},
		"nan sigma":           {replace("ttft_sigma: 0.5", "ttft_sigma: .nan"), "ttft_sigma"},
		"infinite median":     {replace("ttft_median_ms: 1000", "ttft_median_ms: .inf"), "ttft_median_ms"},
		"bad date":            {replace(`"2026-10-28"`, `"28/10/2026"`), "retrieved"},
		"unknown key":         {replace("note: sigma assumed", "notes: sigma assumed"), "notes"},
		"duplicate model":     {"profiles:" + goodEntry + goodEntry, "duplicate"},
		"two documents":       {"profiles:" + goodEntry + "---\nprofiles: []\n", "document"},
	}
	for name, tc := range cases {
		_, err := LoadProfiles(writeProfiles(t, tc.body))
		if err == nil {
			t.Errorf("%s: want an error", name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %q does not mention %q", name, err, tc.want)
		}
	}
	if _, err := LoadProfiles(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Error("a missing file: want an error")
	}
}

var flagshipModels = []string{
	"claude-sonnet-4", "deepseek-r1-0528", "deepseek-v3-0324", "gemini-2.5-flash", "gemini-2.5-pro",
	"glm-4.6", "gpt-5", "gpt-5-chat", "intern-s1", "kimi-k2-0905", "qwen3-235b-a22b-2507",
	"qwen3-235b-a22b-thinking-2507",
}

// TestShippedProfiles checks bench/latency/profiles.yaml against the sourcing rules: one
// profile per flagship model, sigmas as assumed, the TTFT of reasoning variants replaced
// by the non-reasoning median, tpot equal to 1000 over the output speed the page showed,
// and the median tpot for the models whose page shows no speed.
func TestShippedProfiles(t *testing.T) {
	profiles, err := LoadProfiles(filepath.Join("..", "..", "bench", "latency", "profiles.yaml"))
	if err != nil {
		t.Fatalf("LoadProfiles: %v", err)
	}
	if len(profiles) != len(flagshipModels) {
		t.Errorf("%d profiles, want %d", len(profiles), len(flagshipModels))
	}
	speedNote := regexp.MustCompile(`output speed ([0-9.]+) tokens/s as displayed`)
	ttftNote := regexp.MustCompile(`TTFT ([0-9.]+) s as displayed`)
	reasoning := map[string]bool{
		"deepseek-r1-0528": true, "gemini-2.5-pro": true, "glm-4.6": true, "gpt-5": true,
		"intern-s1": true, "qwen3-235b-a22b-thinking-2507": true,
	}

	var displayedTTFT []float64 // non-reasoning models whose page shows a TTFT
	var tpots []float64         // models whose page shows a speed
	for _, m := range flagshipModels {
		p, ok := profiles[m]
		if !ok {
			t.Errorf("no profile for %s", m)
			continue
		}
		if p.Source == "" || p.Retrieved != "2026-10-28" {
			t.Errorf("%s: source %q retrieved %q", m, p.Source, p.Retrieved)
		}
		if p.TTFTSigma != 0.5 || p.TPOTSigma != 0.3 || !strings.Contains(p.Note, "sigma assumed") {
			t.Errorf("%s: sigmas %v %v note %q, want 0.5 and 0.3 with a sigma assumed note", m, p.TTFTSigma, p.TPOTSigma, p.Note)
		}
		if sm := speedNote.FindStringSubmatch(p.Note); sm != nil {
			speed, _ := strconv.ParseFloat(sm[1], 64)
			if math.Abs(p.TPOTMedianMs-1000/speed) > 1e-6 {
				t.Errorf("%s: tpot %v, want 1000/%v = %v", m, p.TPOTMedianMs, speed, 1000/speed)
			}
			tpots = append(tpots, p.TPOTMedianMs)
		} else if !strings.Contains(p.Note, "assumed, no public figure") {
			t.Errorf("%s: no displayed speed and no assumed note: %q", m, p.Note)
		}
		if tm := ttftNote.FindStringSubmatch(p.Note); tm != nil && !reasoning[m] {
			secs, _ := strconv.ParseFloat(tm[1], 64)
			if math.Abs(p.TTFTMedianMs-secs*1000) > 1e-6 {
				t.Errorf("%s: ttft %v ms, want the displayed %v s", m, p.TTFTMedianMs, secs)
			}
			displayedTTFT = append(displayedTTFT, p.TTFTMedianMs)
		}
		if reasoning[m] && !strings.Contains(p.Note, "TTFT replaced by the non-reasoning median") {
			t.Errorf("%s: reasoning variant without the TTFT replacement note: %q", m, p.Note)
		}
	}

	slices.Sort(displayedTTFT)
	slices.Sort(tpots)
	if len(displayedTTFT) == 0 || len(tpots) == 0 {
		t.Fatal("no displayed figures found in the notes")
	}
	medianTTFT := median(displayedTTFT)
	medianTPOT := median(tpots)
	for _, m := range flagshipModels {
		p := profiles[m]
		replaced := reasoning[m] || strings.Contains(p.Note, "assumed, no public figure")
		if replaced && math.Abs(p.TTFTMedianMs-medianTTFT) > 1e-6 {
			t.Errorf("%s: ttft %v, want the median %v of the displayed non-reasoning TTFTs", m, p.TTFTMedianMs, medianTTFT)
		}
		if strings.Contains(p.Note, "assumed, no public figure") && math.Abs(p.TPOTMedianMs-medianTPOT) > 1e-6 {
			t.Errorf("%s: tpot %v, want the median %v of the displayed ones", m, p.TPOTMedianMs, medianTPOT)
		}
	}
}

func median(sorted []float64) float64 {
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}
