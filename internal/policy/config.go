package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/omorsi45/routepilot/internal/config"
)

// Modes of the constrained optimizer.
const (
	// ModeMinCost picks the cheapest model whose quality lower bound clears the floor.
	ModeMinCost = "min_cost"
	// ModeDual picks the model with the highest P - Lambda * expected cost among those
	// that clear the floor.
	ModeDual = "dual"
)

// What the optimizer does with a request whose features are far from the training data.
const (
	// OODConservative scores such a request with each model's pooled success rate and
	// pooled lower bound instead of its per-request prediction.
	OODConservative = "conservative"
	// OODStrongest sends such a request to the highest ranked model that is allowed.
	OODStrongest = "strongest"
)

// Config is how the optimizer uses an artifact: the mode and its price of cost, the
// default quality floor, the exploration rate and the two fallbacks. It is part of the
// policy version the router reports, through Hash8.
type Config struct {
	// Mode is ModeMinCost or ModeDual.
	Mode string `yaml:"mode"`
	// Lambda is the price of one US dollar of expected cost, in units of success
	// probability, in dual mode. It is ignored by min_cost.
	Lambda float64 `yaml:"lambda"`
	// DefaultFloor, in [0, 1], is the quality floor of a request that sets none and
	// whose tenant sets none.
	DefaultFloor float64 `yaml:"default_floor"`
	// Epsilon, in [0, 1), is the probability of picking a uniformly random model of the
	// safe set instead of the optimizer's choice.
	Epsilon float64 `yaml:"epsilon"`
	// OODAction is OODConservative or OODStrongest.
	OODAction string `yaml:"ood_action"`
	// OnInfeasible is config.OnInfeasibleBestEffort or config.OnInfeasibleReject: what
	// to do when no model meets the floor and SLO, unless the tenant says otherwise.
	OnInfeasible string `yaml:"on_infeasible"`
}

// LoadConfig reads a policy configuration. The YAML is strict: unknown keys, a second
// document, and any field outside its range are errors. mode, ood_action and
// on_infeasible are required; lambda, default_floor and epsilon default to zero.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("load policy config: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var c Config
	if err := dec.Decode(&c); err != nil {
		if errors.Is(err, io.EOF) {
			return Config{}, fmt.Errorf("load policy config %s: file is empty", path)
		}
		return Config{}, fmt.Errorf("load policy config %s: %w", path, err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err != nil {
			return Config{}, fmt.Errorf("load policy config %s: %w", path, err)
		}
		return Config{}, fmt.Errorf("load policy config %s: multiple YAML documents are not supported", path)
	}
	if err := c.Validate(); err != nil {
		return Config{}, fmt.Errorf("load policy config %s: %w", path, err)
	}
	return c, nil
}

// Validate checks every field and returns all the problems, joined.
func (c Config) Validate() error {
	var errs []error
	add := func(field, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", field, fmt.Sprintf(format, args...)))
	}
	switch c.Mode {
	case ModeMinCost, ModeDual:
	default:
		add("mode", "unknown mode %q (want %q or %q)", c.Mode, ModeMinCost, ModeDual)
	}
	if math.IsNaN(c.Lambda) || math.IsInf(c.Lambda, 0) || c.Lambda < 0 {
		add("lambda", "must be finite and not negative, got %v", c.Lambda)
	}
	if !(c.DefaultFloor >= 0 && c.DefaultFloor <= 1) {
		add("default_floor", "must be between 0 and 1, got %v", c.DefaultFloor)
	}
	if !(c.Epsilon >= 0 && c.Epsilon < 1) {
		add("epsilon", "must be at least 0 and below 1, got %v", c.Epsilon)
	}
	switch c.OODAction {
	case OODConservative, OODStrongest:
	default:
		add("ood_action", "unknown action %q (want %q or %q)", c.OODAction, OODConservative, OODStrongest)
	}
	switch c.OnInfeasible {
	case config.OnInfeasibleBestEffort, config.OnInfeasibleReject:
	default:
		add("on_infeasible", "unknown value %q (want %q or %q)", c.OnInfeasible, config.OnInfeasibleBestEffort, config.OnInfeasibleReject)
	}
	return errors.Join(errs...)
}

// Hash8 is the first 8 hex digits of the SHA-256 of the configuration's JSON encoding.
// It tells two configurations of one artifact apart in the policy version.
func (c Config) Hash8() string {
	// Marshaling a struct of strings and finite numbers cannot fail; an invalid Lambda
	// (NaN) is refused by Validate before a Config is used.
	b, _ := json.Marshal(c)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:8]
}
