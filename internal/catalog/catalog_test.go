package catalog

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
)

func testConfig() *config.Config {
	return &config.Config{Models: []config.ModelConfig{
		{
			ID: "small",
			Deployments: []config.DeploymentConfig{
				{Name: "small-a", Provider: "p1", UpstreamModel: "s-1"},
				{Name: "small-b", Provider: "p2", UpstreamModel: "s-2"},
			},
			ContextWindow:      8192,
			MaxOutputTokens:    1024,
			Capabilities:       []string{config.CapStreaming, config.CapTools},
			InputPricePerMTok:  "0.15",
			OutputPricePerMTok: "0.6",
			Rank:               1,
		},
		{
			ID:                 "big",
			Deployments:        []config.DeploymentConfig{{Name: "big-a", Provider: "p1", UpstreamModel: "b-1"}},
			ContextWindow:      128000,
			MaxOutputTokens:    16384,
			Capabilities:       []string{config.CapVision, config.CapJSONSchema},
			InputPricePerMTok:  "2.50",
			OutputPricePerMTok: "10",
			Rank:               3,
		},
		{
			ID:                 "default-caps",
			Deployments:        []config.DeploymentConfig{{Name: "dc", Provider: "p1", UpstreamModel: "d"}},
			ContextWindow:      100,
			MaxOutputTokens:    10,
			InputPricePerMTok:  "0",
			OutputPricePerMTok: "0",
		},
	}}
}

func TestNewCatalogFromConfig(t *testing.T) {
	c, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models := c.Models()
	if len(models) != 3 || models[0].ID != "small" || models[1].ID != "big" || models[2].ID != "default-caps" {
		t.Fatalf("Models() order wrong: %+v", models)
	}

	small, ok := c.Model("small")
	if !ok || small != models[0] {
		t.Fatalf("Model(small) = %v, %v", small, ok)
	}
	if small.PriceIn != 150000 || small.PriceOut != 600000 {
		t.Errorf("small prices = %d/%d, want 150000/600000", small.PriceIn, small.PriceOut)
	}
	if small.ContextWindow != 8192 || small.MaxOutputTokens != 1024 || small.Rank != 1 {
		t.Errorf("small limits = %+v", small)
	}
	if !small.Caps.Has(CapStreaming) || !small.Caps.Has(CapTools) ||
		small.Caps.Has(CapVision) || small.Caps.Has(CapJSONObject) || small.Caps.Has(CapJSONSchema) {
		t.Errorf("small caps = %s", small.Caps)
	}
	wantDeps := []Deployment{{Name: "small-a", Provider: "p1", UpstreamModel: "s-1"}, {Name: "small-b", Provider: "p2", UpstreamModel: "s-2"}}
	if !reflect.DeepEqual(small.Deployments, wantDeps) {
		t.Errorf("small deployments = %+v", small.Deployments)
	}

	big, _ := c.Model("big")
	if big.PriceIn != 2500000 || big.PriceOut != 10000000 {
		t.Errorf("big prices = %d/%d", big.PriceIn, big.PriceOut)
	}
	if !big.Caps.Has(CapVision) || !big.Caps.Has(CapJSONSchema) || big.Caps.Has(CapStreaming) {
		t.Errorf("big caps = %s", big.Caps)
	}

	// A nil capability list follows config.HasCapability: streaming only.
	dc, _ := c.Model("default-caps")
	if !dc.Caps.Has(CapStreaming) || dc.Caps.Has(CapTools) {
		t.Errorf("default-caps caps = %s, want streaming only", dc.Caps)
	}

	m, d, ok := c.Deployment("small-b")
	if !ok || m != small || d != wantDeps[1] {
		t.Errorf("Deployment(small-b) = %v, %+v, %v", m, d, ok)
	}
	if _, _, ok := c.Deployment("nope"); ok {
		t.Error("Deployment(nope) should not be found")
	}
	if _, ok := c.Model("nope"); ok {
		t.Error("Model(nope) should not be found")
	}
}

func TestNewRejectsBadConfig(t *testing.T) {
	cfg := testConfig()
	cfg.Models[0].InputPricePerMTok = "abc"
	if _, err := New(cfg); err == nil {
		t.Error("bad input price should error")
	}
	cfg = testConfig()
	cfg.Models[0].OutputPricePerMTok = "-1"
	if _, err := New(cfg); err == nil {
		t.Error("bad output price should error")
	}
	cfg = testConfig()
	cfg.Models[0].Capabilities = []string{"telepathy"}
	if _, err := New(cfg); err == nil {
		t.Error("unknown capability should error")
	}
}

func TestNewRejectsDuplicates(t *testing.T) {
	cfg := testConfig()
	cfg.Models[1].ID = "small"
	if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), `duplicate model id "small"`) {
		t.Errorf("duplicate model id: err = %v", err)
	}
	cfg = testConfig()
	cfg.Models[1].Deployments[0].Name = "small-b" // clashes across models
	if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), `duplicate deployment name "small-b"`) {
		t.Errorf("duplicate deployment across models: err = %v", err)
	}
	cfg = testConfig()
	cfg.Models[0].Deployments[1].Name = "small-a" // clashes within a model
	if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), `duplicate deployment name "small-a"`) {
		t.Errorf("duplicate deployment within model: err = %v", err)
	}
}

func TestCapSetMissingAndString(t *testing.T) {
	names := map[Capability]string{
		CapTools: "tools", CapJSONObject: "json_object", CapJSONSchema: "json_schema",
		CapVision: "vision", CapStreaming: "streaming",
	}
	for c, want := range names {
		if got := c.String(); got != want {
			t.Errorf("Capability(%d).String() = %q, want %q", c, got, want)
		}
	}
	have := capSetOf(CapStreaming, CapTools)
	req := capSetOf(CapTools, CapVision, CapJSONObject)
	got := have.Missing(req)
	want := []Capability{CapJSONObject, CapVision}
	// Missing reports in constant order: tools, json_object, json_schema, vision, streaming.
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Missing = %v, want %v", got, want)
	}
	if len(have.Missing(capSetOf(CapStreaming))) != 0 {
		t.Error("nothing should be missing for a subset")
	}
}

func TestEffectiveK(t *testing.T) {
	m := &Model{MaxOutputTokens: 1000}
	cases := []struct {
		name   string
		r      Requirements
		tenant int
		want   int
	}{
		{"explicit wins", Requirements{MaxTokens: 50, HasMaxTokens: true}, 4096, 50},
		{"explicit above model still returned", Requirements{MaxTokens: 5000, HasMaxTokens: true}, 4096, 5000},
		{"tenant below model", Requirements{}, 300, 300},
		{"model below tenant", Requirements{}, 4096, 1000},
		{"equal", Requirements{}, 1000, 1000},
	}
	for _, tc := range cases {
		if got := m.EffectiveK(tc.r, tc.tenant); got != tc.want {
			t.Errorf("%s: EffectiveK = %d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestCheckReasons(t *testing.T) {
	m := &Model{
		ID: "m", ContextWindow: 1000, MaxOutputTokens: 200,
		Caps: capSetOf(CapStreaming),
	}

	// Feasible: empty, non-nil-or-nil both mean zero reasons.
	ok := Requirements{Caps: capSetOf(CapStreaming), InUpper: 100}
	if got := m.Check(ok, 100); len(got) != 0 {
		t.Errorf("feasible Check = %v, want empty", got)
	}

	// Missing tools and vision are both listed, in constant order.
	r := Requirements{Caps: capSetOf(CapTools, CapVision, CapStreaming), InUpper: 10}
	got := m.Check(r, 10)
	want := []Reason{"missing capability tools", "missing capability vision"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("missing caps = %v, want %v", got, want)
	}

	// Context window: in_upper + k exceeds the window.
	r = Requirements{InUpper: 950}
	got = m.Check(r, 100)
	want = []Reason{"context window 1000 below required 1050"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("context = %v, want %v", got, want)
	}

	// Explicit k above model limit.
	r = Requirements{InUpper: 10, MaxTokens: 300, HasMaxTokens: true}
	got = m.Check(r, 100)
	want = []Reason{"max_tokens 300 above model limit 200"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("max_tokens = %v, want %v", got, want)
	}

	// Everything wrong at once: capabilities, then context, then max tokens.
	r = Requirements{Caps: capSetOf(CapTools), InUpper: 900, MaxTokens: 300, HasMaxTokens: true}
	got = m.Check(r, 100)
	want = []Reason{
		"missing capability tools",
		"context window 1000 below required 1200",
		"max_tokens 300 above model limit 200",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("combined = %v, want %v", got, want)
	}

	// A defaulted k (tenant default above model limit) is clamped, never a max_tokens reason.
	r = Requirements{InUpper: 10}
	if got := m.Check(r, 5000); len(got) != 0 {
		t.Errorf("defaulted k Check = %v, want empty", got)
	}
}

func TestCheckHugeMaxTokensDoesNotOverflow(t *testing.T) {
	m := &Model{ID: "m", ContextWindow: 1000, MaxOutputTokens: 200}
	r := Requirements{InUpper: 10, MaxTokens: math.MaxInt, HasMaxTokens: true}
	got := m.Check(r, 100)
	want := []Reason{
		Reason(fmt.Sprintf("context window 1000 below required %d", math.MaxInt)),
		Reason(fmt.Sprintf("max_tokens %d above model limit 200", math.MaxInt)),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Check = %v, want %v", got, want)
	}
}

func TestCostMaxSaturates(t *testing.T) {
	m := &Model{PriceIn: 150000, PriceOut: 600000}
	r := Requirements{InUpper: 1000}
	// 1000*150000 + 200*600000 = 150,000,000 + 120,000,000
	if got := m.CostMax(r, 200); got != 270000000 {
		t.Errorf("CostMax = %d, want 270000000", got)
	}
	if got := m.CostMax(r, 0); got != 150000000 {
		t.Errorf("CostMax k=0 = %d, want 150000000", got)
	}

	huge := &Model{PriceIn: money.PicoUSD(math.MaxInt64 / 2), PriceOut: money.PicoUSD(math.MaxInt64 / 2)}
	if got := huge.CostMax(Requirements{InUpper: math.MaxInt32}, math.MaxInt32); got != money.PicoUSD(math.MaxInt64) {
		t.Errorf("CostMax overflow = %d, want MaxInt64", got)
	}
}

func TestParseCapability(t *testing.T) {
	// Every capability round-trips through its config name.
	for c := Capability(0); c < numCaps; c++ {
		got, ok := ParseCapability(c.String())
		if !ok || got != c {
			t.Errorf("ParseCapability(%q) = %v, %v; want %v, true", c.String(), got, ok, c)
		}
	}
	for _, name := range []string{"", "telepathy", "Tools", "tools "} {
		if got, ok := ParseCapability(name); ok {
			t.Errorf("ParseCapability(%q) = %v, true; want false", name, got)
		}
	}
}
