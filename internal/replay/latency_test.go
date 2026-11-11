package replay

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/latency"
)

// testLatencyProfiles are test-only figures for the three invented fixture models. They
// are not measurements of anything.
func testLatencyProfiles() map[string]latency.Profile {
	mk := func(model string, ttft, tpot float64) latency.Profile {
		return latency.Profile{
			Model: model, TTFTMedianMs: ttft, TTFTSigma: 0.5, TPOTMedianMs: tpot, TPOTSigma: 0.3,
			Source: "test", Retrieved: "2026-10-28",
		}
	}
	return map[string]latency.Profile{
		"cheap-a": mk("cheap-a", 400, 10),
		"mid-b":   mk("mid-b", 800, 20),
		"top-c":   mk("top-c", 1600, 40),
	}
}

func ptrVal[T any](p *T) any {
	if p == nil {
		return nil
	}
	return *p
}

func TestReplaySimulatedLatencyDeterministic(t *testing.T) {
	cfg, input := loadFixtureConfig(t), fixtureInput(t)
	base := Options{Split: "all", Policy: "cheapest", Profiles: testLatencyProfiles(), LatencySeed: 1}

	a, _, rawA := replayInput(t, cfg, input, base)
	_, _, rawB := replayInput(t, cfg, input, base)
	if !bytes.Equal(rawA, rawB) {
		t.Fatal("same latency seed produced different output")
	}
	for _, r := range a {
		if r.TTFTMsSim == nil || r.E2EMsSim == nil {
			t.Fatalf("%s: ttft_ms_sim %v e2e_ms_sim %v, want both set", r.ItemID, r.TTFTMsSim, r.E2EMsSim)
		}
		if *r.TTFTMsSim <= 0 || *r.E2EMsSim < *r.TTFTMsSim {
			t.Errorf("%s: ttft %v e2e %v, want 0 < ttft <= e2e", r.ItemID, *r.TTFTMsSim, *r.E2EMsSim)
		}
		if r.SLOMs != nil || r.SLOViolated != nil {
			t.Errorf("%s: slo fields set without an SLO flag", r.ItemID)
		}
	}

	other := base
	other.LatencySeed = 2
	c, _, _ := replayInput(t, cfg, input, other)
	same := 0
	for i := range a {
		if *a[i].TTFTMsSim == *c[i].TTFTMsSim {
			same++
		}
	}
	if same == len(a) {
		t.Error("seeds 1 and 2 drew identical latencies for every item")
	}

	// A seed, an item and a model fix the draw, whatever else is in the input.
	subset, _, _ := replayInput(t, cfg, input, Options{Split: "test", Policy: "cheapest", Profiles: testLatencyProfiles(), LatencySeed: 1})
	byID := map[string]Record{}
	for _, r := range a {
		byID[r.ItemID] = r
	}
	for _, r := range subset {
		want := byID[r.ItemID]
		if *r.TTFTMsSim != *want.TTFTMsSim || *r.E2EMsSim != *want.E2EMsSim {
			t.Errorf("%s: latency differs between the test split and all items", r.ItemID)
		}
	}

	// The model is part of the seed: the same item on another model gets another draw
	// (compared as multiples of each model's own median).
	strong, _, _ := replayInput(t, cfg, input, Options{Split: "all", Policy: "strongest", Profiles: testLatencyProfiles(), LatencySeed: 1})
	differ := 0
	for i := range a {
		if *a[i].TTFTMsSim/400 != *strong[i].TTFTMsSim/1600 {
			differ++
		}
	}
	if differ != len(a) {
		t.Errorf("%d of %d items drew the same TTFT multiple on two models", len(a)-differ, len(a))
	}
}

// TestReplaySimulatedLatencyFollowsTheProfile checks the draw itself: e2e is the TTFT plus
// the recorded completion tokens times a TPOT, and over many items the draws are
// log-normal around the profile median with the profile sigma.
func TestReplaySimulatedLatencyFollowsTheProfile(t *testing.T) {
	const items, tokens = 4000, 100
	var in strings.Builder
	for i := 0; i < items; i++ {
		fmt.Fprintf(&in, `{"item_id":"syn/%d","dataset":"syn","split":"test","prompt_sha256":"0123456789abcdef0123",`+
			`"messages":[{"role":"user","content":"hello"}],"outcomes":{`+
			`"cheap-a":{"success":true,"cost_picousd":1,"prompt_tokens":1,"completion_tokens":%d},`+
			`"mid-b":{"success":true,"cost_picousd":1,"prompt_tokens":1,"completion_tokens":%d},`+
			`"top-c":{"success":true,"cost_picousd":1,"prompt_tokens":1,"completion_tokens":%d}}}`+"\n",
			i, tokens, tokens, tokens)
	}
	recs, _, _ := replayInput(t, loadFixtureConfig(t), []byte(in.String()),
		Options{Split: "test", Policy: "fixed", Model: "mid-b", Profiles: testLatencyProfiles(), LatencySeed: 7})

	var lnTTFT, lnTPOT []float64
	for _, r := range recs {
		ttft := *r.TTFTMsSim
		tpot := (*r.E2EMsSim - ttft) / tokens
		if ttft <= 0 || tpot <= 0 {
			t.Fatalf("%s: ttft %v tpot %v, want both positive", r.ItemID, ttft, tpot)
		}
		lnTTFT = append(lnTTFT, math.Log(ttft/800))
		lnTPOT = append(lnTPOT, math.Log(tpot/20))
	}
	for _, c := range []struct {
		name   string
		ln     []float64
		sigma  float64
		median float64
	}{{"ttft", lnTTFT, 0.5, 800}, {"tpot", lnTPOT, 0.3, 20}} {
		var sum, sq float64
		for _, v := range c.ln {
			sum += v
			sq += v * v
		}
		n := float64(len(c.ln))
		mean := sum / n
		std := math.Sqrt(sq/n - mean*mean)
		if math.Abs(mean) > 5*c.sigma/math.Sqrt(n) { // 5 standard errors
			t.Errorf("%s: mean of ln(x/median) = %.4f, want about 0", c.name, mean)
		}
		if math.Abs(std-c.sigma) > 0.05*c.sigma {
			t.Errorf("%s: std of ln(x/median) = %.4f, want about %v", c.name, std, c.sigma)
		}
	}
}

func TestReplayLatencyFormulaUsesRecordedTokens(t *testing.T) {
	cfg, input := loadFixtureConfig(t), fixtureInput(t)
	recs, _, _ := replayInput(t, cfg, input, Options{Split: "all", Policy: "strongest", Profiles: testLatencyProfiles(), LatencySeed: 5})
	byID := map[string]Record{}
	for _, r := range recs {
		byID[r.ItemID] = r
	}
	// top-c recorded 40, 5, 15, -2 and 9 completion tokens. A negative count adds no
	// decode time, so e2e equals the TTFT for fx-b/4.
	if r := byID["fx-b/4"]; *r.E2EMsSim != *r.TTFTMsSim {
		t.Errorf("fx-b/4 (-2 recorded tokens): e2e %v, want the TTFT %v", *r.E2EMsSim, *r.TTFTMsSim)
	}
	// Every other item decodes for a positive, finite time.
	for _, id := range []string{"fx-a/1", "fx-a/2", "fx-b/3", "fx-b/5"} {
		r := byID[id]
		if decode := *r.E2EMsSim - *r.TTFTMsSim; !(decode > 0) || math.IsInf(decode, 0) {
			t.Errorf("%s: decode time %v, want positive", id, decode)
		}
	}
}

func TestReplaySLOFields(t *testing.T) {
	cfg, input := loadFixtureConfig(t), fixtureInput(t)

	recs, _, _ := replayInput(t, cfg, input, Options{Split: "all", Policy: "strongest", Profiles: testLatencyProfiles(), LatencySeed: 5, SLOTTFTMs: 1600})
	violated := 0
	for _, r := range recs {
		if r.SLOMs == nil || *r.SLOMs != 1600 || r.SLOViolated == nil {
			t.Fatalf("%s: slo_ms %v slo_violated %v, want 1600 and a verdict", r.ItemID, ptrVal(r.SLOMs), ptrVal(r.SLOViolated))
		}
		if want := *r.TTFTMsSim > 1600; *r.SLOViolated != want {
			t.Errorf("%s: slo_violated %v with ttft %v against 1600", r.ItemID, *r.SLOViolated, *r.TTFTMsSim)
		}
		if *r.SLOViolated {
			violated++
		}
	}
	if violated == 0 || violated == len(recs) {
		t.Errorf("%d of %d items violate a 1600 ms TTFT SLO on top-c (median 1600): want some but not all", violated, len(recs))
	}

	recs, _, _ = replayInput(t, cfg, input, Options{Split: "all", Policy: "strongest", Profiles: testLatencyProfiles(), LatencySeed: 5, SLOE2EMs: 2000})
	for _, r := range recs {
		if r.SLOMs == nil || *r.SLOMs != 2000 || r.SLOViolated == nil {
			t.Fatalf("%s: slo_ms %v slo_violated %v, want 2000 and a verdict", r.ItemID, ptrVal(r.SLOMs), ptrVal(r.SLOViolated))
		}
		if want := *r.E2EMsSim > 2000; *r.SLOViolated != want {
			t.Errorf("%s: slo_violated %v with e2e %v against 2000", r.ItemID, *r.SLOViolated, *r.E2EMsSim)
		}
	}

	// A routing error has no latency.
	recs, _, _ = replayInput(t, cfg, input, Options{Split: "all", Policy: "fixed", Model: "ghost", Profiles: testLatencyProfiles(), SLOTTFTMs: 100})
	for _, r := range recs {
		if r.TTFTMsSim != nil || r.E2EMsSim != nil || r.SLOViolated != nil {
			t.Errorf("%s: latency fields set on a routing error: %+v", r.ItemID, r)
		}
	}
}

func TestReplayLatencyOptionsRejected(t *testing.T) {
	prof := testLatencyProfiles()
	bad := map[string]Options{
		"ttft slo without profiles": {Split: "all", Policy: "cheapest", SLOTTFTMs: 100},
		"e2e slo without profiles":  {Split: "all", Policy: "cheapest", SLOE2EMs: 100},
		"both slos":                 {Split: "all", Policy: "cheapest", Profiles: prof, SLOTTFTMs: 100, SLOE2EMs: 200},
		"negative ttft slo":         {Split: "all", Policy: "cheapest", Profiles: prof, SLOTTFTMs: -1},
		"negative e2e slo":          {Split: "all", Policy: "cheapest", Profiles: prof, SLOE2EMs: -5},
	}
	for name, opts := range bad {
		var out bytes.Buffer
		_, err := Run(context.Background(), loadFixtureConfig(t), bytes.NewReader(fixtureInput(t)), &out, opts)
		if !errors.Is(err, ErrUsage) {
			t.Errorf("%s: err = %v, want ErrUsage", name, err)
		}
		if out.Len() != 0 {
			t.Errorf("%s: wrote output before failing", name)
		}
	}
}

func TestReplayLatencyMissingProfileIsHardError(t *testing.T) {
	prof := testLatencyProfiles()
	delete(prof, "top-c")
	var out bytes.Buffer
	_, err := Run(context.Background(), loadFixtureConfig(t), bytes.NewReader(fixtureInput(t)), &out,
		Options{Split: "all", Policy: "strongest", Profiles: prof})
	if err == nil || !strings.Contains(err.Error(), "top-c") || !strings.Contains(err.Error(), "latency profile") {
		t.Errorf("err = %v, want an error naming the model and the latency profile", err)
	}
	if errors.Is(err, ErrUsage) {
		t.Error("a missing profile is an input error, not a usage error")
	}
}

func TestReplayWithoutProfilesLeavesLatencyNull(t *testing.T) {
	recs, _, raw := replayFixture(t, Options{Split: "all", Policy: "cheapest"})
	for _, r := range recs {
		if r.TTFTMsSim != nil || r.E2EMsSim != nil || r.SLOMs != nil || r.SLOViolated != nil {
			t.Errorf("%s: latency fields set without profiles", r.ItemID)
		}
	}
	if !bytes.Contains(raw, []byte(`"ttft_ms_sim":null`)) {
		t.Error("ttft_ms_sim is not null in the output")
	}
}

func TestStreamIsSeededBySeedItemAndModel(t *testing.T) {
	// Different (seed, item, model) triples give different streams; the same triple gives
	// the same one.
	a := newStream(1, "item", "m")
	b := newStream(1, "item", "m")
	var xs, ys []float64
	for i := 0; i < 4; i++ {
		xs, ys = append(xs, a.float64()), append(ys, b.float64())
	}
	if !slices.Equal(xs, ys) {
		t.Errorf("same triple: %v vs %v", xs, ys)
	}
	for _, other := range []*stream{newStream(2, "item", "m"), newStream(1, "item2", "m"), newStream(1, "item", "m2")} {
		if other.float64() == xs[0] {
			t.Error("a different triple repeated the first draw")
		}
	}
	for _, x := range xs {
		if !(x > 0 && x < 1) {
			t.Errorf("draw %v outside (0, 1)", x)
		}
	}
}
