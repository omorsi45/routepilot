package replay

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"slices"
	"strings"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
)

// Policy names accepted in Options.Policy. The first three are the gateway's
// own policies, built from the config's routing section; the rest exist only
// for offline evaluation.
const (
	PolicyCheapest  = config.PolicyCheapest
	PolicyStrongest = config.PolicyStrongest
	PolicyRules     = config.PolicyRules
	// PolicyRandom orders models by a hash of the seed, the item and the model.
	PolicyRandom = "random"
	// PolicyOracle looks at the recorded outcomes, so it is an upper bound and
	// never a policy the gateway could run.
	PolicyOracle = "oracle"
	// PolicyFixed always asks for Options.Model.
	PolicyFixed = "fixed"
)

const policyNames = "cheapest, strongest, rules, random, oracle or fixed"

// selector routes one item and returns the candidate to replay. A routing
// error comes back as the *openai.APIError; the error result is for problems
// that stop the run.
type selector func(item *Item, in router.Input) (router.Candidate, *openai.APIError, error)

func newSelector(cat *catalog.Catalog, routing config.RoutingConfig, opts Options) (selector, error) {
	switch opts.Policy {
	case PolicyCheapest, PolicyStrongest, PolicyRules:
		routing.Policy = opts.Policy
		p, err := router.NewPolicy(routing, cat)
		if err != nil {
			return nil, fmt.Errorf("build %s policy: %w", opts.Policy, err)
		}
		r := router.New(cat, p)
		return func(_ *Item, in router.Input) (router.Candidate, *openai.APIError, error) {
			return decideFirst(r, in)
		}, nil

	case PolicyFixed:
		// A pinned request: the router checks that the model exists, that the
		// tenant may use it and that it can serve the request. No policy needed.
		r := router.New(cat, nil)
		model := opts.Model
		return func(_ *Item, in router.Input) (router.Candidate, *openai.APIError, error) {
			in.Requested = model
			return decideFirst(r, in)
		}, nil

	case PolicyRandom:
		// A router is cheap to build, so each item gets one whose policy is
		// fixed by the seed and the item ID.
		seed := opts.Seed
		return func(item *Item, in router.Input) (router.Candidate, *openai.APIError, error) {
			return decideFirst(router.New(cat, randomPolicy{seed: seed, itemID: item.ItemID}), in)
		}, nil

	case PolicyOracle:
		// Any ordering yields the same feasible set; the oracle picks from all of it.
		p, err := router.NewPolicy(config.RoutingConfig{Policy: config.PolicyStrongest}, cat)
		if err != nil {
			return nil, fmt.Errorf("build oracle candidate set: %w", err)
		}
		r := router.New(cat, p)
		return func(item *Item, in router.Input) (router.Candidate, *openai.APIError, error) {
			d, apiErr := r.Decide(in)
			if apiErr != nil {
				return router.Candidate{}, apiErr, nil
			}
			c, err := pickOracle(item, d.Candidates)
			return c, nil, err
		}, nil
	}
	return nil, usageError("unknown policy %q (want %s)", opts.Policy, policyNames)
}

// decideFirst takes the first candidate of the router's decision.
func decideFirst(r *router.Router, in router.Input) (router.Candidate, *openai.APIError, error) {
	d, apiErr := r.Decide(in)
	if apiErr != nil {
		return router.Candidate{}, apiErr, nil
	}
	if len(d.Candidates) == 0 {
		return router.Candidate{}, nil, fmt.Errorf("router returned no candidate and no error for %q", in.Requested)
	}
	return d.Candidates[0], nil, nil
}

// pickOracle chooses, among every feasible candidate, a model that succeeded at
// the lowest recorded cost. When none succeeded it chooses the lowest recorded
// cost. Ties go to the lower model ID.
func pickOracle(item *Item, cands []router.Candidate) (router.Candidate, error) {
	var best router.Candidate
	var bestOutcome Outcome
	for i, c := range cands {
		o, ok := item.Outcomes[c.Model.ID]
		if !ok {
			return router.Candidate{}, fmt.Errorf("item %s: no recorded outcome for candidate model %s", item.ItemID, c.Model.ID)
		}
		if i == 0 || oracleBetter(o, c.Model.ID, bestOutcome, best.Model.ID) {
			best, bestOutcome = c, o
		}
	}
	return best, nil
}

func oracleBetter(a Outcome, aID string, b Outcome, bID string) bool {
	if a.Success != b.Success {
		return a.Success
	}
	if a.CostPicoUSD != b.CostPicoUSD {
		return a.CostPicoUSD < b.CostPicoUSD
	}
	return aID < bID
}

// randomPolicy orders models by the FNV-1a 64 hash of the seed (8 bytes, big
// endian), the item ID and the model ID, so a seed and an item fix the order.
type randomPolicy struct {
	seed   uint64
	itemID string
}

func (randomPolicy) Name() string { return PolicyRandom }

func (p randomPolicy) Order(models []*catalog.Model, _ router.Input) []*catalog.Model {
	keys := make(map[string]uint64, len(models))
	for _, m := range models {
		keys[m.ID] = p.key(m.ID)
	}
	out := slices.Clone(models)
	slices.SortStableFunc(out, func(a, b *catalog.Model) int {
		if c := cmp.Compare(keys[a.ID], keys[b.ID]); c != 0 {
			return c
		}
		return strings.Compare(a.ID, b.ID)
	})
	return out
}

func (p randomPolicy) key(modelID string) uint64 {
	var seed [8]byte
	binary.BigEndian.PutUint64(seed[:], p.seed)
	h := fnv.New64a()
	// A hash.Hash64 write never fails.
	_, _ = h.Write(seed[:])
	_, _ = io.WriteString(h, p.itemID)
	_, _ = io.WriteString(h, modelID)
	return h.Sum64()
}
