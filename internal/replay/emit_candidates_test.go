package replay

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/policy"
)

// TestReplayEmitCandidates: with EmitCandidates the learned policy's record lists every model
// the tenant may use with the numbers the optimizer scored it on, so a record
// is the logging policy's own outcome model for the item.
func TestReplayEmitCandidates(t *testing.T) {
	r := newLearnedRig(t)
	pc := policyConfig(func(c *policy.Config) { c.DefaultFloor = 0.3 })
	recs, raw := r.run(pc, nil, Options{EmitCandidates: true})
	if len(recs) != len(r.items) {
		t.Fatalf("%d records for %d items", len(recs), len(r.items))
	}
	wantKeys := []string{"cost_exp_picousd", "cost_max_picousd", "feasible", "l", "model", "p"}
	for i, rec := range recs {
		item := r.items[i]
		if got := len(rec.Candidates); got != 2 {
			t.Fatalf("%s: %d candidates, want one per model of the tenant (2)", rec.ItemID, got)
		}
		sc := r.scores(item)
		for j, id := range []string{"tiny-weak", "tiny-strong"} { // catalog order
			c := rec.Candidates[j]
			s := sc.Models[id]
			wantP, wantL := s.P, s.L
			if sc.OOD { // the conservative action scores an out-of-distribution item by the pooled numbers
				wantP, wantL = s.PooledRate, s.PooledLCB
			}
			if c.Model != id || c.P == nil || c.L == nil || c.CostExpPicoUSD == nil || c.CostMaxPicoUSD == nil {
				t.Fatalf("%s: candidate %d = %+v, want %s with every number", rec.ItemID, j, c, id)
			}
			if *c.P != wantP || *c.L != wantL {
				t.Errorf("%s/%s: p %v l %v, want %v and %v", rec.ItemID, id, *c.P, *c.L, wantP, wantL)
			}
			if want := r.expectedCostExp(item, id, s); *c.CostExpPicoUSD != want {
				t.Errorf("%s/%s: cost_exp_picousd %d, want %d", rec.ItemID, id, *c.CostExpPicoUSD, want)
			}
			if *c.CostMaxPicoUSD < *c.CostExpPicoUSD {
				t.Errorf("%s/%s: cost_max %d below cost_exp %d", rec.ItemID, id, *c.CostMaxPicoUSD, *c.CostExpPicoUSD)
			}
			if *rec.Model == id {
				// The chosen model's candidate is the record's own p, l and cost_exp.
				if *c.P != *rec.P || *c.L != *rec.L || *c.CostExpPicoUSD != *rec.CostExpPicoUSD || *c.CostMaxPicoUSD != *rec.CostMaxPicoUSD {
					t.Errorf("%s: chosen candidate %+v differs from the record's numbers", rec.ItemID, c)
				}
				if !c.Feasible && len(rec.Relaxed) == 0 {
					t.Errorf("%s: the chosen model is infeasible and nothing was relaxed", rec.ItemID)
				}
			}
		}
	}

	// The key set of a candidate is exactly the documented one.
	var line struct {
		Candidates []map[string]json.RawMessage `json:"candidates"`
	}
	if err := json.Unmarshal(bytes.SplitN(raw, []byte("\n"), 2)[0], &line); err != nil {
		t.Fatal(err)
	}
	var keys []string
	for k := range line.Candidates[0] {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	if !slices.Equal(keys, wantKeys) {
		t.Errorf("candidate keys = %v, want %v", keys, wantKeys)
	}
}

// TestReplayEmitCandidatesHasNoPromptText: the scores are numbers and model names.
func TestReplayEmitCandidatesHasNoPromptText(t *testing.T) {
	r := newLearnedRig(t)
	_, raw := r.run(policyConfig(), nil, Options{EmitCandidates: true})
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

// TestReplayWithoutEmitCandidatesHasNoCandidates: the key is absent unless asked for, so the
// default output is the one every consumer already reads.
func TestReplayWithoutEmitCandidatesHasNoCandidates(t *testing.T) {
	r := newLearnedRig(t)
	_, raw := r.run(policyConfig(), nil, Options{})
	if bytes.Contains(raw, []byte(`"candidates"`)) {
		t.Error("the output has candidates without EmitCandidates")
	}
}

// TestReplayEmitCandidatesOnARefusedItem: an item the policy refused (no model chosen) still
// carries the scores it was refused on.
func TestReplayEmitCandidatesOnARefusedItem(t *testing.T) {
	r := newLearnedRig(t)
	pc := policyConfig(func(c *policy.Config) { c.DefaultFloor = 1; c.OnInfeasible = config.OnInfeasibleReject })
	recs, _ := r.run(pc, nil, Options{EmitCandidates: true})
	for _, rec := range recs {
		if rec.Error != "no_feasible_model" || rec.Model != nil {
			t.Fatalf("%s: error %q model %v, want a refusal", rec.ItemID, rec.Error, rec.Model)
		}
		if len(rec.Candidates) != 2 {
			t.Errorf("%s: %d candidates on a refused item, want 2", rec.ItemID, len(rec.Candidates))
		}
		for _, c := range rec.Candidates {
			if c.Feasible {
				t.Errorf("%s/%s: feasible at floor 1", rec.ItemID, c.Model)
			}
			if c.P == nil {
				t.Errorf("%s/%s: no p: the model was scored", rec.ItemID, c.Model)
			}
		}
	}
}

// TestReplayEmitCandidatesNeedsTheLearnedPolicy: the other policies score nothing.
func TestReplayEmitCandidatesNeedsTheLearnedPolicy(t *testing.T) {
	r := newLearnedRig(t)
	var out bytes.Buffer
	_, err := Run(context.Background(), r.cfg, bytes.NewReader(r.input), &out,
		Options{Split: SplitAll, Policy: PolicyCheapest, EmitCandidates: true})
	if !errors.Is(err, ErrUsage) {
		t.Errorf("EmitCandidates with policy cheapest: err = %v, want ErrUsage", err)
	}
	if out.Len() != 0 {
		t.Error("wrote output before failing")
	}
}
