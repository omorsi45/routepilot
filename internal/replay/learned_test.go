package replay

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/router"
)

var (
	learnedFixtureDir = filepath.Join("..", "..", "testdata", "replay", "learned")
	tinyArtifactDir   = filepath.Join("..", "..", "testdata", "policy", "tiny", "0782d801a5a3")
)

// learnedRig loads the fixture config and items and the committed tiny artifact, whose
// models are tiny-weak and tiny-strong.
type learnedRig struct {
	t     *testing.T
	cfg   *config.Config
	cat   *catalog.Catalog
	input []byte
	items []Item
	art   *policy.Artifact
}

func newLearnedRig(t *testing.T) *learnedRig {
	t.Helper()
	cfg, err := config.Load(filepath.Join(learnedFixtureDir, "config.yaml"))
	if err != nil {
		t.Fatalf("load fixture config: %v", err)
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	art, err := policy.Load(tinyArtifactDir)
	if err != nil {
		t.Fatalf("load tiny artifact: %v", err)
	}
	input, err := os.ReadFile(filepath.Join(learnedFixtureDir, "items.jsonl"))
	if err != nil {
		t.Fatalf("read fixture items: %v", err)
	}
	var items []Item
	for _, line := range bytes.Split(bytes.TrimSpace(input), []byte("\n")) {
		var it Item
		if err := json.Unmarshal(line, &it); err != nil {
			t.Fatalf("decode fixture item: %v", err)
		}
		items = append(items, it)
	}
	return &learnedRig{t: t, cfg: cfg, cat: cat, input: input, items: items, art: art}
}

func policyConfig(mutate ...func(*policy.Config)) policy.Config {
	c := policy.Config{Mode: policy.ModeMinCost, OODAction: policy.OODConservative, OnInfeasible: config.OnInfeasibleBestEffort}
	for _, m := range mutate {
		m(&c)
	}
	return c
}

// run replays every fixture item with the learned policy built from pc and est.
func (r *learnedRig) run(pc policy.Config, est *latency.Estimator, opts Options) ([]Record, []byte) {
	r.t.Helper()
	opts.Split, opts.Policy = SplitAll, PolicyLearned
	opts.Learned = router.NewOptimizer(r.art, pc, est)
	var out bytes.Buffer
	if _, err := Run(context.Background(), r.cfg, bytes.NewReader(r.input), &out, opts); err != nil {
		r.t.Fatalf("Run(%+v): %v", opts, err)
	}
	return decodeRecords(r.t, out.Bytes()), out.Bytes()
}

// expectedCostExp recomputes the expected cost of running modelID on the item from the
// artifact's own numbers: ceil(in_upper * in_ratio) input tokens and the expected output
// clamped to [1, k], at the model's integer prices.
func (r *learnedRig) expectedCostExp(item Item, modelID string, s policy.ModelScore) int64 {
	req := &openai.ChatCompletionRequest{Model: "auto", Messages: item.Messages}
	reqs := catalog.Derive(req)
	m, _ := r.cat.Model(modelID)
	in := int64(math.Ceil(float64(reqs.InUpper+m.PromptOverhead) * s.InRatio))
	out := int64(math.Ceil(math.Min(math.Max(s.EOut, 1), 1024)))
	return int64(m.PriceIn)*in + int64(m.PriceOut)*out
}

func (r *learnedRig) expectedChoice(item Item, sc policy.Scores, floor float64) (model string, relaxed []string) {
	type cand struct {
		id   string
		l    float64
		cost int64
	}
	var all, safe []cand
	for _, id := range []string{"tiny-weak", "tiny-strong"} {
		s := sc.Models[id]
		l := s.L
		if sc.OOD {
			l = s.PooledLCB
		}
		c := cand{id, l, r.expectedCostExp(item, id, s)}
		all = append(all, c)
		if l >= floor {
			safe = append(safe, c)
		}
	}
	if len(safe) > 0 {
		slices.SortFunc(safe, func(a, b cand) int {
			if a.cost != b.cost {
				return cmp.Compare(a.cost, b.cost)
			}
			return cmp.Compare(b.l, a.l)
		})
		return safe[0].id, nil
	}
	slices.SortFunc(all, func(a, b cand) int { return cmp.Compare(b.l, a.l) })
	return all[0].id, []string{"quality"}
}

func (r *learnedRig) scores(item Item) policy.Scores {
	req := &openai.ChatCompletionRequest{Model: "auto", Messages: item.Messages}
	return r.art.Score(features.Messages(req))
}

func TestReplayLearnedFillsDecisionFields(t *testing.T) {
	r := newLearnedRig(t)
	pc := policyConfig(func(c *policy.Config) { c.DefaultFloor = 0.3 })
	opt := router.NewOptimizer(r.art, pc, nil)
	recs, _ := r.run(pc, nil, Options{})
	if len(recs) != len(r.items) {
		t.Fatalf("%d records for %d items", len(recs), len(r.items))
	}

	chosen := map[string]int{}
	for i, rec := range recs {
		item := r.items[i]
		sc := r.scores(item)
		if rec.Policy != "learned" || rec.PolicyVersion == nil || *rec.PolicyVersion != opt.Version() {
			t.Errorf("%s: policy %q version %v, want learned and %s", rec.ItemID, rec.Policy, rec.PolicyVersion, opt.Version())
		}
		if rec.Model == nil || rec.P == nil || rec.L == nil || rec.CostExpPicoUSD == nil || rec.Floor == nil ||
			rec.OODScore == nil || rec.Propensity == nil || rec.K == nil || rec.CostMaxPicoUSD == nil {
			t.Fatalf("%s: missing fields in %+v", rec.ItemID, rec)
		}
		model := *rec.Model
		chosen[model]++
		s := sc.Models[model]
		wantP, wantL := s.P, s.L
		if sc.OOD { // the conservative action replaces the model's numbers by the pooled ones
			wantP, wantL = s.PooledRate, s.PooledLCB
		}
		if *rec.P != wantP || *rec.L != wantL {
			t.Errorf("%s: p %v l %v, want %v and %v", rec.ItemID, *rec.P, *rec.L, wantP, wantL)
		}
		if rec.OOD != sc.OOD || *rec.OODScore != sc.OODScore {
			t.Errorf("%s: ood %v score %v, want %v and %v", rec.ItemID, rec.OOD, *rec.OODScore, sc.OOD, sc.OODScore)
		}
		if *rec.Floor != 0.3 {
			t.Errorf("%s: floor %v, want 0.3", rec.ItemID, *rec.Floor)
		}
		if want := r.expectedCostExp(item, model, s); *rec.CostExpPicoUSD != want {
			t.Errorf("%s: cost_exp_picousd %d, want %d", rec.ItemID, *rec.CostExpPicoUSD, want)
		}
		if *rec.K != 1024 || rec.Infeasible || rec.Error != "" {
			t.Errorf("%s: k %d infeasible %v error %q", rec.ItemID, *rec.K, rec.Infeasible, rec.Error)
		}
		if rec.Explored || *rec.Propensity != 1 {
			t.Errorf("%s: explored %v propensity %v with epsilon 0", rec.ItemID, rec.Explored, *rec.Propensity)
		}
		if rec.Relaxed == nil {
			t.Errorf("%s: relaxed is null, want a list", rec.ItemID)
		}
		// The rule for min_cost: the lowest expected cost among the models whose lower
		// bound clears the floor, else (best effort) the highest lower bound.
		want, wantRelaxed := r.expectedChoice(item, sc, 0.3)
		if model != want || !slices.Equal(append([]string{}, rec.Relaxed...), append([]string{}, wantRelaxed...)) {
			t.Errorf("%s: chose %s relaxed %v, want %s relaxed %v", rec.ItemID, model, rec.Relaxed, want, wantRelaxed)
		} // The recorded outcome of the chosen model is joined in.
		if rec.Success != item.Outcomes[model].Success || rec.CostPicoUSD != item.Outcomes[model].CostPicoUSD {
			t.Errorf("%s: outcome not joined: success %v cost %d", rec.ItemID, rec.Success, rec.CostPicoUSD)
		}
	}
	if len(chosen) < 2 {
		t.Logf("only %v were chosen at floor 0.3", chosen)
	}
}

// TestReplayLearnedFloorSweepUsesBothModels makes the test above non-vacuous: across a
// sweep of floors the policy must send some items to each model, with a relaxed floor at
// the top of the sweep.
func TestReplayLearnedFloorSweepUsesBothModels(t *testing.T) {
	r := newLearnedRig(t)
	seen := map[string]bool{}
	for _, floor := range []float64{0, 0.2, 0.4, 0.6, 0.8, 0.95, 1} {
		recs, _ := r.run(policyConfig(func(c *policy.Config) { c.DefaultFloor = floor }), nil, Options{})
		for _, rec := range recs {
			seen[*rec.Model] = true
			if floor == 0 {
				item := r.items[slices.IndexFunc(r.items, func(it Item) bool { return it.ItemID == rec.ItemID })]
				if want, _ := r.expectedChoice(item, r.scores(item), 0); *rec.Model != want {
					t.Errorf("floor 0 picked %s for %s, want the cheaper %s", *rec.Model, rec.ItemID, want)
				}
			}
			if floor == 1 && !slices.Equal(rec.Relaxed, []string{"quality"}) {
				t.Errorf("floor 1: %s relaxed %v, want [quality]", rec.ItemID, rec.Relaxed)
			}
		}
	}
	if !seen["tiny-weak"] || !seen["tiny-strong"] {
		t.Errorf("models chosen across the floor sweep: %v, want both", seen)
	}
}

func TestReplayLearnedRejectIsInfeasible(t *testing.T) {
	r := newLearnedRig(t)
	pc := policyConfig(func(c *policy.Config) { c.DefaultFloor = 1; c.OnInfeasible = config.OnInfeasibleReject })
	recs, raw := r.run(pc, nil, Options{})
	for _, rec := range recs {
		if rec.Error != "no_feasible_model" || !rec.Infeasible {
			t.Errorf("%s: error %q infeasible %v, want no_feasible_model and true", rec.ItemID, rec.Error, rec.Infeasible)
		}
		if rec.Model != nil || rec.Success || rec.CostPicoUSD != 0 || rec.P != nil || rec.L != nil || rec.CostExpPicoUSD != nil {
			t.Errorf("%s: a rejected item must be a failure at zero cost with no model: %+v", rec.ItemID, rec)
		}
		if rec.Floor == nil || *rec.Floor != 1 || rec.PolicyVersion == nil || rec.OODScore == nil {
			t.Errorf("%s: floor %v version %v ood_score %v, want the decision context kept", rec.ItemID, rec.Floor, rec.PolicyVersion, rec.OODScore)
		}
		if rec.Relaxed == nil || len(rec.Relaxed) != 0 {
			t.Errorf("%s: relaxed %v, want an empty list", rec.ItemID, rec.Relaxed)
		}
	}
	if !bytes.Contains(raw, []byte(`"error":"no_feasible_model"`)) {
		t.Error("the error code is missing from the output")
	}
}

func TestReplayLearnedExplorationIsSeededPerItem(t *testing.T) {
	r := newLearnedRig(t)
	pc := policyConfig(func(c *policy.Config) { c.DefaultFloor = 0; c.Epsilon = 0.9 })
	a, rawA := r.run(pc, nil, Options{Seed: 1})
	_, rawB := r.run(pc, nil, Options{Seed: 1})
	if !bytes.Equal(rawA, rawB) {
		t.Fatal("same seed, different output")
	}
	explored := 0
	for _, rec := range a {
		if rec.Explored {
			explored++
		}
		if rec.Propensity == nil || *rec.Propensity <= 0 || *rec.Propensity > 1 {
			t.Errorf("%s: propensity %v", rec.ItemID, rec.Propensity)
		}
	}
	if explored == 0 {
		t.Error("epsilon 0.9 never explored on six items")
	}
	var differs bool
	for seed := uint64(2); seed < 8 && !differs; seed++ {
		_, other := r.run(pc, nil, Options{Seed: seed})
		differs = !bytes.Equal(rawA, other)
	}
	if !differs {
		t.Error("seeds 2 to 7 never changed an exploration outcome")
	}
	// An item's draw does not depend on which other items are replayed.
	var out bytes.Buffer
	opts := Options{Split: "test", Policy: PolicyLearned, Learned: router.NewOptimizer(r.art, pc, nil), Seed: 1}
	if _, err := Run(context.Background(), r.cfg, bytes.NewReader(r.input), &out, opts); err != nil {
		t.Fatal(err)
	}
	byID := map[string]Record{}
	for _, rec := range a {
		byID[rec.ItemID] = rec
	}
	for _, rec := range decodeRecords(t, out.Bytes()) {
		want := byID[rec.ItemID]
		if modelOf(rec) != modelOf(want) || rec.Explored != want.Explored {
			t.Errorf("%s: %s/%v on the test split but %s/%v on all items", rec.ItemID, modelOf(rec), rec.Explored, modelOf(want), want.Explored)
		}
	}
}

func TestReplayLearnedUsesLatencyProfilesForTheSLO(t *testing.T) {
	r := newLearnedRig(t)
	// Test-only figures: tiny-strong is fast, tiny-weak is slow, by a wide margin.
	prof := map[string]latency.Profile{
		"tiny-strong": {Model: "tiny-strong", TTFTMedianMs: 100, TTFTSigma: 0.5, TPOTMedianMs: 1, TPOTSigma: 0.3, Source: "test", Retrieved: "2026-10-28"},
		"tiny-weak":   {Model: "tiny-weak", TTFTMedianMs: 10_000, TTFTSigma: 0.5, TPOTMedianMs: 100, TPOTSigma: 0.3, Source: "test", Retrieved: "2026-10-28"},
	}
	est := latency.NewEstimator(prof, nil)
	pc := policyConfig()

	// cheapest is the model with the lower expected cost, which is who wins at floor 0
	// when no SLO gets in the way.
	cheapest := func(rec Record) string {
		item := r.items[slices.IndexFunc(r.items, func(it Item) bool { return it.ItemID == rec.ItemID })]
		model, _ := r.expectedChoice(item, r.scores(item), 0)
		return model
	}
	recs, _ := r.run(pc, est, Options{Profiles: prof})
	for _, rec := range recs {
		if modelOf(rec) != cheapest(rec) {
			t.Errorf("%s: no slo picked %s, want the cheaper %s", rec.ItemID, modelOf(rec), cheapest(rec))
		}
	}

	// An end-to-end SLO only the fast model meets moves everything to it, with nothing relaxed.
	recs, _ = r.run(pc, est, Options{Profiles: prof, SLOE2EMs: 5000})
	for _, rec := range recs {
		if modelOf(rec) != "tiny-strong" || len(rec.Relaxed) != 0 {
			t.Errorf("%s: slo 5000 picked %s relaxed %v, want tiny-strong and nothing relaxed", rec.ItemID, modelOf(rec), rec.Relaxed)
		}
		if rec.SLOMs == nil || *rec.SLOMs != 5000 || rec.E2EMsSim == nil {
			t.Errorf("%s: slo fields missing: %+v", rec.ItemID, rec)
		}
	}

	// A time-to-first-token SLO works the same way.
	recs, _ = r.run(pc, est, Options{Profiles: prof, SLOTTFTMs: 2000})
	for _, rec := range recs {
		if modelOf(rec) != "tiny-strong" || len(rec.Relaxed) != 0 {
			t.Errorf("%s: ttft slo picked %s relaxed %v, want tiny-strong", rec.ItemID, modelOf(rec), rec.Relaxed)
		}
	}

	// An SLO nobody meets relaxes latency and falls back to the cheapest above the floor.
	recs, _ = r.run(pc, est, Options{Profiles: prof, SLOE2EMs: 50})
	for _, rec := range recs {
		if modelOf(rec) != cheapest(rec) || !slices.Equal(rec.Relaxed, []string{"latency"}) {
			t.Errorf("%s: slo 50 picked %s relaxed %v, want %s and [latency]", rec.ItemID, modelOf(rec), rec.Relaxed, cheapest(rec))
		}
	}
}

func TestReplayLearnedNeedsALearnedPolicy(t *testing.T) {
	r := newLearnedRig(t)
	var out bytes.Buffer
	_, err := Run(context.Background(), r.cfg, bytes.NewReader(r.input), &out, Options{Split: SplitAll, Policy: PolicyLearned})
	if !errors.Is(err, ErrUsage) {
		t.Errorf("learned without a Learned: err = %v, want ErrUsage", err)
	}
	if out.Len() != 0 {
		t.Error("wrote output before failing")
	}
	// A Learned with another policy is a usage error, like a model with a policy other than fixed.
	_, err = Run(context.Background(), r.cfg, bytes.NewReader(r.input), &out,
		Options{Split: SplitAll, Policy: PolicyCheapest, Learned: router.NewOptimizer(r.art, policyConfig(), nil)})
	if !errors.Is(err, ErrUsage) {
		t.Errorf("Learned with policy cheapest: err = %v, want ErrUsage", err)
	}
}

func TestReplayLearnedOutputHasNoPromptText(t *testing.T) {
	r := newLearnedRig(t)
	_, raw := r.run(policyConfig(), nil, Options{})
	for _, it := range r.items {
		text := it.Messages[0].Content.PlainText()
		if len(text) > 20 && bytes.Contains(raw, []byte(text[:20])) {
			t.Errorf("%s: output carries prompt text", it.ItemID)
		}
	}
	if bytes.Contains(raw, []byte(`"messages"`)) {
		t.Error("output carries the messages key")
	}
}
