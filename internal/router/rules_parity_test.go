package router

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
)

// rulesParityFile is testdata/rules/cases.json. The Python side
// (routepilot_train.baselines.static_rules) writes the rules and checks the
// same cases with match_rule, so both sides agree on which rule matches.
type rulesParityFile struct {
	Rules []struct {
		Name string `json:"name"`
		When struct {
			ContainsAny    []string `json:"contains_any"`
			MinPromptBytes int      `json:"min_prompt_bytes"`
			MaxPromptBytes int      `json:"max_prompt_bytes"`
		} `json:"when"`
	} `json:"rules"`
	Cases []struct {
		Note        string  `json:"note"`
		Text        string  `json:"text"`
		Pad         int     `json:"pad"`
		Tail        string  `json:"tail"`
		PromptBytes int     `json:"prompt_bytes"`
		Expect      *string `json:"expect"`
	} `json:"cases"`
}

// TestRulesParity builds the rules policy from the rules in the shared JSON,
// giving each rule a model of the same name and a "default" model for the
// no-match case, and checks that the first model of the decision is the one
// the Python match_rule predicts.
func TestRulesParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "rules", "cases.json"))
	if err != nil {
		t.Fatalf("read cases: %v", err)
	}
	var data rulesParityFile
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("decode cases: %v", err)
	}
	if len(data.Rules) == 0 || len(data.Cases) < 15 {
		t.Fatalf("got %d rules and %d cases, want at least 1 and 15", len(data.Rules), len(data.Cases))
	}

	cfg := &config.Config{}
	routing := config.RoutingConfig{Policy: config.PolicyRules, DefaultModel: "default"}
	addModel := func(id string) {
		cfg.Models = append(cfg.Models, config.ModelConfig{
			ID:                 id,
			Deployments:        []config.DeploymentConfig{{Name: id + "-d", Provider: "p", UpstreamModel: id}},
			ContextWindow:      1 << 20,
			MaxOutputTokens:    4096,
			Capabilities:       []string{"streaming"},
			InputPricePerMTok:  "0",
			OutputPricePerMTok: "0",
		})
	}
	addModel("default")
	for _, r := range data.Rules {
		addModel(r.Name)
		routing.Rules = append(routing.Rules, config.RuleConfig{
			Name:  r.Name,
			Model: r.Name,
			When: config.RuleMatch{
				ContainsAny:    r.When.ContainsAny,
				MinPromptBytes: r.When.MinPromptBytes,
				MaxPromptBytes: r.When.MaxPromptBytes,
			},
		})
	}
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	pol, err := NewPolicy(routing, cat)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}
	rt := New(cat, pol)
	tenant := &config.TenantConfig{ID: "t", AllowedModels: []string{"*"}, DefaultMaxTokens: 256}

	for _, c := range data.Cases {
		text := c.Text + strings.Repeat("x", c.Pad) + c.Tail
		// Derive cuts the text the way the gateway does; the case sets the
		// prompt byte count directly because two cases decouple it from the text.
		reqs := catalog.Derive(&openai.ChatCompletionRequest{
			Model:    config.ReservedModelAuto,
			Messages: []openai.Message{{Role: "user", Content: openai.Content{Text: &text}}},
		})
		reqs.PromptBytes = c.PromptBytes

		d, apiErr := rt.Decide(Input{Tenant: tenant, Requested: config.ReservedModelAuto, Req: reqs})
		if apiErr != nil {
			t.Fatalf("%s: Decide: %v", c.Note, apiErr)
		}
		want := "default"
		if c.Expect != nil {
			want = *c.Expect
		}
		if got := d.Candidates[0].Model.ID; got != want {
			t.Errorf("%s: first model = %q, want %q", c.Note, got, want)
		}
	}
}
