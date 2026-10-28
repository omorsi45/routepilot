// Package catalog is the immutable view of configured models that routing
// and admission use: capabilities, limits, prices, and deployments, plus the
// per-request requirements derived from a chat completion request.
package catalog

import (
	"fmt"
	"strings"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
)

// Capability is a model feature a request may require.
type Capability uint8

// Capabilities, in the order Missing and Check report them.
const (
	CapTools Capability = iota
	CapJSONObject
	CapJSONSchema
	CapVision
	CapStreaming
	numCaps
)

// CapSet is a uint8 bitmask, so at most 8 capabilities fit. This fails to
// compile when a ninth is added.
const _ = CapSet(1 << (numCaps - 1))

// capNames maps each Capability to its config name. It is the only place the
// catalog spells those names; they come from the config package constants.
var capNames = [numCaps]string{
	CapTools:      config.CapTools,
	CapJSONObject: config.CapJSONObject,
	CapJSONSchema: config.CapJSONSchema,
	CapVision:     config.CapVision,
	CapStreaming:  config.CapStreaming,
}

// String returns the capability's config name.
func (c Capability) String() string {
	if c < numCaps {
		return capNames[c]
	}
	return fmt.Sprintf("capability(%d)", uint8(c))
}

// ParseCapability resolves a config capability name. It is the only lookup
// from names to capabilities; callers must not keep their own table.
func ParseCapability(name string) (Capability, bool) {
	for c, n := range capNames {
		if n == name {
			return Capability(c), true
		}
	}
	return 0, false
}

// CapSet is a set of capabilities.
type CapSet uint8

// with returns the set plus c.
func (s CapSet) with(c Capability) CapSet { return s | 1<<c }

// Has reports whether c is in the set.
func (s CapSet) Has(c Capability) bool { return s&(1<<c) != 0 }

// Missing returns the capabilities in req that s lacks, in constant order.
func (s CapSet) Missing(req CapSet) []Capability {
	var out []Capability
	for c := Capability(0); c < numCaps; c++ {
		if req.Has(c) && !s.Has(c) {
			out = append(out, c)
		}
	}
	return out
}

// Names lists the capability names in the set, in constant order. The result is never nil, so
// it encodes as an empty JSON list.
func (s CapSet) Names() []string {
	names := []string{}
	for c := Capability(0); c < numCaps; c++ {
		if s.Has(c) {
			names = append(names, c.String())
		}
	}
	return names
}

// String lists the capability names in the set, comma separated.
func (s CapSet) String() string {
	names := s.Names()
	if len(names) == 0 {
		return "none"
	}
	return strings.Join(names, ",")
}

// Deployment is one provider-side target of a model. MaxInflight is the cap on its in-flight
// non-streaming attempts, 0 for none.
type Deployment struct {
	Name          string
	Provider      string
	UpstreamModel string
	MaxInflight   int
}

// Model is a public model with its limits, capabilities, prices, and
// deployments. Prices are picodollars per token. PromptOverhead is the hidden
// prompt tokens the provider adds to every request of this model (Ruling
// V1-R24); see InUpper.
type Model struct {
	ID              string
	ContextWindow   int
	MaxOutputTokens int
	Caps            CapSet
	PriceIn         money.PicoUSD
	PriceOut        money.PicoUSD
	Rank            int
	PromptOverhead  int
	Deployments     []Deployment
}

// Catalog indexes the configured models. It is read-only after New.
type Catalog struct {
	models      []*Model
	byID        map[string]*Model
	deployments map[string]deploymentRef
}

type deploymentRef struct {
	model *Model
	dep   Deployment
}

// New builds a Catalog from a validated config. It still rejects an unparsable
// price or unknown capability name, as defense in depth.
func New(cfg *config.Config) (*Catalog, error) {
	c := &Catalog{
		byID:        make(map[string]*Model, len(cfg.Models)),
		deployments: make(map[string]deploymentRef),
	}
	for i := range cfg.Models {
		mc := cfg.Models[i]
		priceIn, err := money.ParseUSDPerMTok(mc.InputPricePerMTok)
		if err != nil {
			return nil, fmt.Errorf("model %q input price: %w", mc.ID, err)
		}
		priceOut, err := money.ParseUSDPerMTok(mc.OutputPricePerMTok)
		if err != nil {
			return nil, fmt.Errorf("model %q output price: %w", mc.ID, err)
		}
		caps, err := capSetFor(mc)
		if err != nil {
			return nil, fmt.Errorf("model %q: %w", mc.ID, err)
		}
		if _, dup := c.byID[mc.ID]; dup {
			return nil, fmt.Errorf("duplicate model id %q", mc.ID)
		}
		m := &Model{
			ID:              mc.ID,
			ContextWindow:   mc.ContextWindow,
			MaxOutputTokens: mc.MaxOutputTokens,
			Caps:            caps,
			PriceIn:         priceIn,
			PriceOut:        priceOut,
			Rank:            mc.Rank,
			PromptOverhead:  mc.PromptOverheadTokens,
			Deployments:     make([]Deployment, 0, len(mc.Deployments)),
		}
		for _, dc := range mc.Deployments {
			if _, dup := c.deployments[dc.Name]; dup {
				return nil, fmt.Errorf("model %q: duplicate deployment name %q", mc.ID, dc.Name)
			}
			d := Deployment{Name: dc.Name, Provider: dc.Provider, UpstreamModel: dc.UpstreamModel, MaxInflight: dc.MaxInflight}
			m.Deployments = append(m.Deployments, d)
			c.deployments[d.Name] = deploymentRef{model: m, dep: d}
		}
		c.models = append(c.models, m)
		c.byID[m.ID] = m
	}
	return c, nil
}

// capSetFor derives the model's CapSet through ModelConfig.HasCapability, so
// the nil-means-streaming default has a single implementation.
func capSetFor(mc config.ModelConfig) (CapSet, error) {
	for _, name := range mc.Capabilities {
		if _, ok := ParseCapability(name); !ok {
			return 0, fmt.Errorf("unknown capability %q", name)
		}
	}
	var set CapSet
	for c := Capability(0); c < numCaps; c++ {
		if mc.HasCapability(capNames[c]) {
			set = set.with(c)
		}
	}
	return set, nil
}

// Model returns the model with the given public ID.
func (c *Catalog) Model(id string) (*Model, bool) {
	m, ok := c.byID[id]
	return m, ok
}

// Models returns every model in config order. The slice is a copy.
func (c *Catalog) Models() []*Model {
	return append([]*Model(nil), c.models...)
}

// Deployment finds a deployment by its globally unique name.
func (c *Catalog) Deployment(name string) (*Model, Deployment, bool) {
	ref, ok := c.deployments[name]
	return ref.model, ref.dep, ok
}
