// Package config loads and validates the RoutePilot YAML configuration.
//
// Parsing is strict: unknown keys are errors. Validation collects every
// problem it finds and reports them together, each prefixed with the path of
// the offending field (for example "providers[1].type").
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"regexp"

	"gopkg.in/yaml.v3"
)

// Provider types accepted in ProviderConfig.Type.
const (
	ProviderOpenAICompat = "openaicompat"
	ProviderAnthropic    = "anthropic"
)

// ReservedModelAuto is a model ID reserved for automatic routing.
const ReservedModelAuto = "auto"

// Defaults applied when a field is omitted.
const (
	defaultListen           = ":8080"
	defaultMaxBodyBytes     = 4194304
	defaultDefaultTimeoutMs = 120000
	defaultMaxTimeoutMs     = 600000
	defaultShutdownGraceMs  = 30000
	defaultTTFTTimeoutMs    = 10000
	defaultIdleTimeoutMs    = 30000
)

var sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Config is the root of the configuration file.
type Config struct {
	Server    ServerConfig     `yaml:"server"`
	Providers []ProviderConfig `yaml:"providers"`
	Models    []ModelConfig    `yaml:"models"`
	Tenants   []TenantConfig   `yaml:"tenants"`
}

// ServerConfig holds listener and request limits.
type ServerConfig struct {
	Listen           string `yaml:"listen"`
	MaxBodyBytes     int64  `yaml:"max_body_bytes"`
	DefaultTimeoutMs int    `yaml:"default_timeout_ms"`
	MaxTimeoutMs     int    `yaml:"max_timeout_ms"`
	ShutdownGraceMs  int    `yaml:"shutdown_grace_ms"`
}

// ProviderConfig describes one upstream provider.
type ProviderConfig struct {
	Name          string `yaml:"name"`
	Type          string `yaml:"type"`
	BaseURL       string `yaml:"base_url"`
	APIKeyEnv     string `yaml:"api_key_env"`
	TTFTTimeoutMs int    `yaml:"ttft_timeout_ms"`
	IdleTimeoutMs int    `yaml:"idle_timeout_ms"`
}

// ModelConfig is a public model ID backed by one or more deployments.
type ModelConfig struct {
	ID          string             `yaml:"id"`
	Deployments []DeploymentConfig `yaml:"deployments"`
}

// DeploymentConfig maps a public model to a model on one provider.
type DeploymentConfig struct {
	Name          string `yaml:"name"`
	Provider      string `yaml:"provider"`
	UpstreamModel string `yaml:"upstream_model"`
}

// TenantConfig is one API client, identified by the SHA-256 of its key.
type TenantConfig struct {
	ID            string   `yaml:"id"`
	APIKeySHA256  string   `yaml:"api_key_sha256"`
	AllowedModels []string `yaml:"allowed_models"`
	MaxTimeoutMs  int      `yaml:"max_timeout_ms"`
}

// Parse decodes b strictly, applies defaults, and validates the result.
func Parse(b []byte) (*Config, error) {
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)

	var cfg Config
	if err := dec.Decode(&cfg); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("config is empty")
		}
		return nil, fmt.Errorf("parse config: %w", err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, fmt.Errorf("parse config: %w", err)
		}
		return nil, errors.New("parse config: multiple YAML documents are not supported")
	}

	cfg.applyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Load reads and parses the config file at path.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load config %s: %w", path, err)
	}
	cfg, err := Parse(b)
	if err != nil {
		return nil, fmt.Errorf("load config %s: %w", path, err)
	}
	return cfg, nil
}

func (c *Config) applyDefaults() {
	s := &c.Server
	if s.Listen == "" {
		s.Listen = defaultListen
	}
	if s.MaxBodyBytes == 0 {
		s.MaxBodyBytes = defaultMaxBodyBytes
	}
	if s.DefaultTimeoutMs == 0 {
		s.DefaultTimeoutMs = defaultDefaultTimeoutMs
	}
	if s.MaxTimeoutMs == 0 {
		s.MaxTimeoutMs = defaultMaxTimeoutMs
	}
	if s.ShutdownGraceMs == 0 {
		s.ShutdownGraceMs = defaultShutdownGraceMs
	}
	for i := range c.Providers {
		p := &c.Providers[i]
		if p.TTFTTimeoutMs == 0 {
			p.TTFTTimeoutMs = defaultTTFTTimeoutMs
		}
		if p.IdleTimeoutMs == 0 {
			p.IdleTimeoutMs = defaultIdleTimeoutMs
		}
	}
	for i := range c.Models {
		for j := range c.Models[i].Deployments {
			d := &c.Models[i].Deployments[j]
			if d.Name == "" && d.Provider != "" && d.UpstreamModel != "" {
				d.Name = d.Provider + "/" + d.UpstreamModel
			}
		}
	}
}

// Validate checks the whole configuration and returns every problem found,
// joined into one error. It returns nil when the config is valid.
func (c *Config) Validate() error {
	var errs []error
	add := func(path, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", path, fmt.Sprintf(format, args...)))
	}

	c.validateServer(add)
	providers := c.validateProviders(add)
	models := c.validateModels(providers, add)
	c.validateTenants(models, add)

	return errors.Join(errs...)
}

type addFunc func(path, format string, args ...any)

func (c *Config) validateServer(add addFunc) {
	s := c.Server
	if s.MaxBodyBytes <= 0 {
		add("server.max_body_bytes", "must be positive, got %d", s.MaxBodyBytes)
	}
	if s.DefaultTimeoutMs <= 0 {
		add("server.default_timeout_ms", "must be positive, got %d", s.DefaultTimeoutMs)
	}
	if s.MaxTimeoutMs <= 0 {
		add("server.max_timeout_ms", "must be positive, got %d", s.MaxTimeoutMs)
	}
	if s.ShutdownGraceMs <= 0 {
		add("server.shutdown_grace_ms", "must be positive, got %d", s.ShutdownGraceMs)
	}
	if s.DefaultTimeoutMs > 0 && s.MaxTimeoutMs > 0 && s.DefaultTimeoutMs > s.MaxTimeoutMs {
		add("server.default_timeout_ms", "%d exceeds server.max_timeout_ms %d", s.DefaultTimeoutMs, s.MaxTimeoutMs)
	}
}

func (c *Config) validateProviders(add addFunc) map[string]bool {
	names := make(map[string]bool)
	if len(c.Providers) == 0 {
		add("providers", "at least one provider is required")
	}
	for i, p := range c.Providers {
		path := fmt.Sprintf("providers[%d]", i)
		switch {
		case p.Name == "":
			add(path+".name", "must not be empty")
		case names[p.Name]:
			add(path+".name", "duplicate provider name %q", p.Name)
		default:
			names[p.Name] = true
		}
		switch p.Type {
		case ProviderOpenAICompat, ProviderAnthropic:
		default:
			add(path+".type", "unknown provider type %q (want %q or %q)", p.Type, ProviderOpenAICompat, ProviderAnthropic)
		}
		if u, err := url.Parse(p.BaseURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			add(path+".base_url", "must be an http or https URL with a host, got %q", p.BaseURL)
		}
		if p.TTFTTimeoutMs <= 0 {
			add(path+".ttft_timeout_ms", "must be positive, got %d", p.TTFTTimeoutMs)
		}
		if p.IdleTimeoutMs <= 0 {
			add(path+".idle_timeout_ms", "must be positive, got %d", p.IdleTimeoutMs)
		}
	}
	return names
}

func (c *Config) validateModels(providers map[string]bool, add addFunc) map[string]bool {
	ids := make(map[string]bool)
	if len(c.Models) == 0 {
		add("models", "at least one model is required")
	}
	for i, m := range c.Models {
		path := fmt.Sprintf("models[%d]", i)
		switch {
		case m.ID == "":
			add(path+".id", "must not be empty")
		case m.ID == ReservedModelAuto:
			add(path+".id", "%q is reserved", m.ID)
		case ids[m.ID]:
			add(path+".id", "duplicate model id %q", m.ID)
		default:
			ids[m.ID] = true
		}
		if len(m.Deployments) == 0 {
			add(path+".deployments", "at least one deployment is required")
		}
		names := make(map[string]bool)
		for j, d := range m.Deployments {
			dpath := fmt.Sprintf("%s.deployments[%d]", path, j)
			switch {
			case d.Provider == "":
				add(dpath+".provider", "must not be empty")
			case !providers[d.Provider]:
				add(dpath+".provider", "unknown provider %q", d.Provider)
			}
			if d.UpstreamModel == "" {
				add(dpath+".upstream_model", "must not be empty")
			}
			if d.Name != "" {
				if names[d.Name] {
					add(dpath+".name", "duplicate deployment name %q", d.Name)
				}
				names[d.Name] = true
			}
		}
	}
	return ids
}

func (c *Config) validateTenants(models map[string]bool, add addFunc) {
	if len(c.Tenants) == 0 {
		add("tenants", "at least one tenant is required")
	}
	ids := make(map[string]bool)
	hashes := make(map[string]bool)
	for i, t := range c.Tenants {
		path := fmt.Sprintf("tenants[%d]", i)
		switch {
		case t.ID == "":
			add(path+".id", "must not be empty")
		case ids[t.ID]:
			add(path+".id", "duplicate tenant id %q", t.ID)
		default:
			ids[t.ID] = true
		}
		switch {
		case t.APIKeySHA256 == "":
			add(path+".api_key_sha256", "must not be empty")
		case !sha256Hex.MatchString(t.APIKeySHA256):
			add(path+".api_key_sha256", "must be 64 lowercase hex characters")
		case hashes[t.APIKeySHA256]:
			add(path+".api_key_sha256", "duplicate key hash")
		default:
			hashes[t.APIKeySHA256] = true
		}
		for j, m := range t.AllowedModels {
			if m != "*" && !models[m] {
				add(fmt.Sprintf("%s.allowed_models[%d]", path, j), "unknown model %q", m)
			}
		}
		if t.MaxTimeoutMs < 0 {
			add(path+".max_timeout_ms", "must not be negative, got %d", t.MaxTimeoutMs)
		} else if t.MaxTimeoutMs > c.Server.MaxTimeoutMs {
			add(path+".max_timeout_ms", "%d exceeds server.max_timeout_ms %d", t.MaxTimeoutMs, c.Server.MaxTimeoutMs)
		}
	}
}

// Model returns the model with the given ID.
func (c *Config) Model(id string) (ModelConfig, bool) {
	for _, m := range c.Models {
		if m.ID == id {
			return m, true
		}
	}
	return ModelConfig{}, false
}

// Allows reports whether the tenant may use the given model.
func (t TenantConfig) Allows(model string) bool {
	for _, m := range t.AllowedModels {
		if m == "*" || m == model {
			return true
		}
	}
	return false
}
