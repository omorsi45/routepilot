// Package replay runs the router offline over recorded outcomes. Each input
// line is one benchmark item with the recorded result of every model; for each
// item the chosen policy picks a model through the real router and catalog, and
// the record joins that model's recorded outcome. Nothing calls a provider.
//
// The output never carries prompt text: an item is identified by its item_id
// and the first 16 hex digits of its prompt SHA-256.
package replay

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
)

// DefaultTenant is the tenant requests are made as when Options.Tenant is empty.
const DefaultTenant = "replay"

// SplitAll keeps every item whatever its split.
const SplitAll = "all"

// hashPrefixLen is the number of hex digits of the prompt hash kept in a Record.
const hashPrefixLen = 16

// ErrUsage marks an error caused by the options, not by the input data.
var ErrUsage = errors.New("invalid replay options")

// infeasibleCodes are the routing error codes that mean no model could serve
// the request, as opposed to a request that named a model that does not exist.
var infeasibleCodes = map[string]bool{
	openai.ErrCapabilityUnavailable("").Code: true,
	openai.ErrNoHealthyDeployment().Code:     true,
	openai.ErrNoFeasibleModel("").Code:       true,
}

// scoredCodes are the routing error codes that the learned policy returns after it has
// scored the request, so the decision's floor and out-of-distribution verdict are real.
var scoredCodes = map[string]bool{
	openai.ErrNoFeasibleModel("").Code:   true,
	openai.ErrInsufficientQuota("").Code: true,
}

// Outcome is the recorded result of one model on one item. Token counts are
// kept as recorded, so they can be negative.
type Outcome struct {
	Success          bool  `json:"success"`
	CostPicoUSD      int64 `json:"cost_picousd"`
	PromptTokens     int   `json:"prompt_tokens"`
	CompletionTokens int   `json:"completion_tokens"`
}

// Item is one line of the replay input.
type Item struct {
	ItemID       string             `json:"item_id"`
	Dataset      string             `json:"dataset"`
	Split        string             `json:"split"`
	PromptSHA256 string             `json:"prompt_sha256"`
	Messages     []openai.Message   `json:"messages"`
	Outcomes     map[string]Outcome `json:"outcomes"`
}

// Record is one line of the replay output. Fields that do not apply to the
// policy that ran are null. It holds no prompt text. InUpper is the
// request-level bound; PromptOverhead is the chosen model's hidden prompt
// tokens (Ruling V1-R24, null when no model was chosen), so the model's own
// bound is InUpper plus PromptOverhead.
type Record struct {
	ItemID         string   `json:"item_id"`
	Dataset        string   `json:"dataset"`
	Split          string   `json:"split"`
	PromptSHA25616 string   `json:"prompt_sha256_16"`
	Policy         string   `json:"policy"`
	PolicyVersion  *string  `json:"policy_version"`
	Model          *string  `json:"model"`
	Success        bool     `json:"success"`
	CostPicoUSD    int64    `json:"cost_picousd"`
	InUpper        int      `json:"in_upper"`
	PromptOverhead *int     `json:"prompt_overhead"`
	K              *int     `json:"k"`
	P              *float64 `json:"p"`
	L              *float64 `json:"l"`
	CostExpPicoUSD *int64   `json:"cost_exp_picousd"`
	CostMaxPicoUSD *int64   `json:"cost_max_picousd"`
	Floor          *float64 `json:"floor"`
	Relaxed        []string `json:"relaxed"`
	OOD            bool     `json:"ood"`
	OODScore       *float64 `json:"ood_score"`
	Explored       bool     `json:"explored"`
	Propensity     *float64 `json:"propensity"`
	TTFTMsSim      *float64 `json:"ttft_ms_sim"`
	E2EMsSim       *float64 `json:"e2e_ms_sim"`
	SLOMs          *float64 `json:"slo_ms"`
	SLOViolated    *bool    `json:"slo_violated"`
	Infeasible     bool     `json:"infeasible"`
	// Error is the routing error code when the policy returned one. The item
	// then counts as a failure at zero cost.
	Error string `json:"error"`
	// Candidates is the learned policy's score for every model the tenant may use, in
	// catalog order. It is written only with Options.EmitCandidates (Ruling V1-R32).
	Candidates []Candidate `json:"candidates,omitempty"`
}

// Candidate is what the learned policy scored one model on for one item, the same numbers
// the decision log writes for a candidate. P, L and the costs are null for a model the policy
// ruled out before it scored it (for example one missing from the artifact).
type Candidate struct {
	Model          string   `json:"model"`
	P              *float64 `json:"p"`
	L              *float64 `json:"l"`
	CostExpPicoUSD *int64   `json:"cost_exp_picousd"`
	CostMaxPicoUSD *int64   `json:"cost_max_picousd"`
	Feasible       bool     `json:"feasible"`
}

// Options select the items and the policy.
type Options struct {
	// Split keeps the items of one split; SplitAll keeps every item.
	Split string
	// Policy is cheapest, strongest, rules, random, oracle, fixed or learned.
	Policy string
	// Seed fixes the random policy.
	Seed uint64
	// Tenant is the config tenant requests are made as; empty means DefaultTenant.
	Tenant string
	// Model is the model the fixed policy always asks for.
	Model string
	// Learned is the policy the learned policy replays; it is required by, and only valid
	// with, PolicyLearned. Epsilon, the floor, the mode and the rest are part of it.
	Learned router.Learned
	// EmitCandidates adds the learned policy's per-model scores to every record (the
	// Candidates field). It is only valid with PolicyLearned.
	EmitCandidates bool
	// MF and Threshold are the scorer and the cut of PolicyRouteLLMMF: an item goes to the
	// strong model when its score is at least Threshold, a number in [0, 1]. Both are
	// required by, and only valid with, that policy.
	MF        *MF
	Threshold float64
	// Profiles are the simulated latency profiles by model. When set, every item that
	// routes to a model gets a simulated latency drawn from that model's profile, seeded
	// by LatencySeed, and a missing profile for a chosen model is an error.
	Profiles    map[string]latency.Profile
	LatencySeed uint64
	// SLOTTFTMs and SLOE2EMs, in milliseconds, set the latency SLO the simulated
	// latency is judged against (zero means none). At most one may be set, and either
	// needs Profiles.
	SLOTTFTMs, SLOE2EMs int
}

// Summary totals a run. Routing errors count as failures at zero cost.
type Summary struct {
	Items       int
	Errors      int
	Successes   int
	CostPicoUSD int64
}

// SuccessRate is Successes over Items, or 0 for an empty run.
func (s Summary) SuccessRate() float64 {
	if s.Items == 0 {
		return 0
	}
	return float64(s.Successes) / float64(s.Items)
}

// MeanCostPicoUSD is the mean cost per item in whole picodollars, rounded down.
func (s Summary) MeanCostPicoUSD() int64 {
	if s.Items == 0 {
		return 0
	}
	return s.CostPicoUSD / int64(s.Items)
}

// Validate checks the options that do not depend on the config. Its errors
// wrap ErrUsage.
func (o Options) Validate() error {
	if o.Split == "" {
		return usageError("split is required")
	}
	switch o.Policy {
	case "":
		return usageError("policy is required")
	case PolicyFixed:
		if o.Model == "" {
			return usageError("policy %s requires a model", PolicyFixed)
		}
		if o.Learned != nil {
			return usageError("a learned policy is only valid with policy %s", PolicyLearned)
		}
	case PolicyRouteLLMMF:
		if o.Model != "" || o.Learned != nil {
			return usageError("policy %s takes only an MF file and a threshold, no model or learned policy", PolicyRouteLLMMF)
		}
		if math.IsNaN(o.Threshold) || o.Threshold < 0 || o.Threshold > 1 {
			return usageError("the %s threshold must be in [0, 1], got %v", PolicyRouteLLMMF, o.Threshold)
		}
	case PolicyCheapest, PolicyStrongest, PolicyRules, PolicyRandom, PolicyOracle, PolicyLearned:
		if o.Model != "" {
			return usageError("a model is only valid with policy %s", PolicyFixed)
		}
		if o.Learned != nil && o.Policy != PolicyLearned {
			return usageError("a learned policy is only valid with policy %s", PolicyLearned)
		}
	default:
		return usageError("unknown policy %q (want %s)", o.Policy, policyNames)
	}
	if o.Policy != PolicyRouteLLMMF && (o.MF != nil || o.Threshold != 0) {
		return usageError("an MF file and a threshold are only valid with policy %s", PolicyRouteLLMMF)
	}
	if o.EmitCandidates && o.Policy != PolicyLearned {
		return usageError("candidates are only emitted for policy %s", PolicyLearned)
	}
	if o.SLOTTFTMs < 0 || o.SLOE2EMs < 0 {
		return usageError("an SLO must be positive")
	}
	if o.SLOTTFTMs > 0 && o.SLOE2EMs > 0 {
		return usageError("set only one of the TTFT and the end-to-end SLO")
	}
	return nil
}

// hasSLO reports whether a latency SLO is set.
func (o Options) hasSLO() bool { return o.SLOTTFTMs > 0 || o.SLOE2EMs > 0 }

func usageError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrUsage, fmt.Sprintf(format, args...))
}

// Run replays every kept item of in through the policy and writes one JSON
// Record per line to out. A routing error is recorded and the run continues;
// invalid input, or a chosen model with no recorded outcome, stops the run
// with an error that names the item. An input with no kept item is an error.
func Run(ctx context.Context, cfg *config.Config, in io.Reader, out io.Writer, opts Options) (Summary, error) {
	if opts.Tenant == "" {
		opts.Tenant = DefaultTenant
	}
	if err := opts.Validate(); err != nil {
		return Summary{}, err
	}
	if opts.hasSLO() && len(opts.Profiles) == 0 {
		return Summary{}, usageError("an SLO needs latency profiles")
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		return Summary{}, fmt.Errorf("build catalog: %w", err)
	}
	tenant, err := findTenant(cfg, opts.Tenant)
	if err != nil {
		return Summary{}, err
	}
	sel, err := newSelector(cat, cfg.Routing, opts)
	if err != nil {
		return Summary{}, err
	}

	reader := bufio.NewReaderSize(in, 1<<20)
	w := bufio.NewWriter(out)
	enc := json.NewEncoder(w)
	var sum Summary
	for lineNo := 1; ; lineNo++ {
		if err := ctx.Err(); err != nil {
			return sum, err
		}
		// ReadBytes has no line length limit; replay lines reach tens of KiB.
		line, readErr := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			item, err := decodeItem(line, lineNo)
			if err != nil {
				return sum, err
			}
			if opts.Split == SplitAll || item.Split == opts.Split {
				rec, err := replayItem(sel, &tenant, item, opts)
				if err != nil {
					return sum, err
				}
				sum.add(rec)
				if err := enc.Encode(rec); err != nil {
					return sum, fmt.Errorf("write record for %s: %w", item.ItemID, err)
				}
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				return sum, fmt.Errorf("read input line %d: %w", lineNo, readErr)
			}
			break
		}
	}
	if err := w.Flush(); err != nil {
		return sum, fmt.Errorf("write output: %w", err)
	}
	if sum.Items == 0 {
		return sum, fmt.Errorf("no items in split %q", opts.Split)
	}
	return sum, nil
}

func findTenant(cfg *config.Config, id string) (config.TenantConfig, error) {
	for _, t := range cfg.Tenants {
		if t.ID == id {
			return t, nil
		}
	}
	return config.TenantConfig{}, usageError("tenant %q is not in the config", id)
}

// decodeItem parses one input line. Errors name the line, never its content.
func decodeItem(line []byte, lineNo int) (*Item, error) {
	var item Item
	if err := json.Unmarshal(bytes.TrimSpace(line), &item); err != nil {
		return nil, fmt.Errorf("input line %d: %w", lineNo, err)
	}
	if item.ItemID == "" {
		return nil, fmt.Errorf("input line %d: item_id is empty", lineNo)
	}
	for model, o := range item.Outcomes {
		if o.CostPicoUSD < 0 {
			return nil, fmt.Errorf("item %s: model %s has a negative recorded cost", item.ItemID, model)
		}
	}
	return &item, nil
}

// replayItem routes one item and joins the chosen model's recorded outcome.
func replayItem(sel selector, tenant *config.TenantConfig, item *Item, opts Options) (Record, error) {
	req := &openai.ChatCompletionRequest{Model: config.ReservedModelAuto, Messages: item.Messages}
	if apiErr := req.Validate(); apiErr != nil {
		return Record{}, fmt.Errorf("item %s: invalid request: %s", item.ItemID, apiErr.Message)
	}
	reqs := catalog.Derive(req)

	rec := Record{
		ItemID:         item.ItemID,
		Dataset:        item.Dataset,
		Split:          item.Split,
		PromptSHA25616: hashPrefix(item.PromptSHA256),
		Policy:         opts.Policy,
		InUpper:        reqs.InUpper,
		Relaxed:        []string{},
	}
	input := router.Input{Tenant: tenant, Requested: config.ReservedModelAuto, Req: reqs}
	chosen, apiErr, err := sel(item, input)
	if err != nil {
		return Record{}, err
	}
	cand := chosen.cand
	if opts.Policy == PolicyLearned {
		fillLearned(&rec, chosen.dec, apiErr)
		if opts.EmitCandidates {
			rec.Candidates = candidatesOf(chosen.dec)
		}
	}
	if chosen.score != nil {
		score := *chosen.score
		rec.P = &score
	}
	if apiErr != nil {
		rec.Error = apiErr.Code
		rec.Infeasible = infeasibleCodes[apiErr.Code]
		return rec, nil
	}

	outcome, ok := item.Outcomes[cand.Model.ID]
	if !ok {
		return Record{}, fmt.Errorf("item %s: no recorded outcome for chosen model %s", item.ItemID, cand.Model.ID)
	}
	model, k, costMax := cand.Model.ID, cand.K, int64(cand.CostMax)
	overhead := cand.Model.PromptOverhead
	rec.Model = &model
	rec.K = &k
	rec.PromptOverhead = &overhead
	rec.CostMaxPicoUSD = &costMax
	rec.Success = outcome.Success
	rec.CostPicoUSD = outcome.CostPicoUSD
	if opts.Policy == PolicyLearned {
		fillChosenScore(&rec, chosen.dec, model)
	}
	if len(opts.Profiles) > 0 {
		profile, ok := opts.Profiles[model]
		if !ok {
			return Record{}, fmt.Errorf("item %s: no latency profile for chosen model %s", item.ItemID, model)
		}
		ttft, e2e := simulateLatency(profile, opts.LatencySeed, item.ItemID, model, outcome.CompletionTokens)
		rec.TTFTMsSim, rec.E2EMsSim = &ttft, &e2e
		switch {
		case opts.SLOTTFTMs > 0:
			slo, violated := float64(opts.SLOTTFTMs), ttft > float64(opts.SLOTTFTMs)
			rec.SLOMs, rec.SLOViolated = &slo, &violated
		case opts.SLOE2EMs > 0:
			slo, violated := float64(opts.SLOE2EMs), e2e > float64(opts.SLOE2EMs)
			rec.SLOMs, rec.SLOViolated = &slo, &violated
		}
	}
	return rec, nil
}

// fillLearned records the learned policy's decision context: its version always, and
// the floor, the out-of-distribution verdict and the exploration outcome whenever the
// request was scored (a routing error that came before the scoring has none to report).
func fillLearned(rec *Record, d router.Decision, apiErr *openai.APIError) {
	version := d.PolicyVersion
	rec.PolicyVersion = &version
	if apiErr != nil && !scoredCodes[apiErr.Code] {
		return
	}
	floor, oodScore, propensity := d.Floor, d.OODScore, d.Propensity
	rec.Floor, rec.OODScore, rec.Propensity = &floor, &oodScore, &propensity
	rec.OOD, rec.Explored = d.OOD, d.Explored
	rec.Relaxed = append([]string{}, d.Relaxed...)
}

// candidatesOf lists the scores of every model the tenant may use, as the decision log does:
// a model ruled out before it was scored has only its model name and feasibility.
func candidatesOf(d router.Decision) []Candidate {
	out := make([]Candidate, 0, len(d.Scores))
	for _, s := range d.Scores {
		c := Candidate{Model: s.Model, Feasible: s.Feasible}
		if s.Feasible || s.P != 0 || s.L != 0 || s.CostExp != 0 || s.CostMax != 0 || s.QMs != nil {
			c.P, c.L = finiteOrNil(s.P), finiteOrNil(s.L)
			exp, max := int64(s.CostExp), int64(s.CostMax)
			c.CostExpPicoUSD, c.CostMaxPicoUSD = &exp, &max
		}
		out = append(out, c)
	}
	return out
}

// finiteOrNil returns &f, or nil for NaN and the infinities, which JSON cannot hold.
func finiteOrNil(f float64) *float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return &f
}

// fillChosenScore records the numbers the optimizer used for the model it chose.
func fillChosenScore(rec *Record, d router.Decision, model string) {
	for _, s := range d.Scores {
		if s.Model == model {
			p, l, costExp := s.P, s.L, int64(s.CostExp)
			rec.P, rec.L, rec.CostExpPicoUSD = &p, &l, &costExp
			return
		}
	}
}

func (s *Summary) add(rec Record) {
	s.Items++
	if rec.Error != "" {
		s.Errors++
	}
	if rec.Success {
		s.Successes++
	}
	s.CostPicoUSD = int64(money.AddSat(money.PicoUSD(s.CostPicoUSD), money.PicoUSD(rec.CostPicoUSD)))
}

func hashPrefix(sha string) string {
	if len(sha) > hashPrefixLen {
		return sha[:hashPrefixLen]
	}
	return sha
}
