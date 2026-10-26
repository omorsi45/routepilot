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

	"github.com/omorsi45/routepilot/internal/money"
)

// Provider types accepted in ProviderConfig.Type.
const (
	ProviderOpenAICompat = "openaicompat"
	ProviderAnthropic    = "anthropic"
)

// ReservedModelAuto is a model ID reserved for automatic routing.
const ReservedModelAuto = "auto"

// Model capability names accepted in ModelConfig.Capabilities and
// RuleMatch.Requires.
const (
	CapStreaming  = "streaming"
	CapTools      = "tools"
	CapJSONObject = "json_object"
	CapJSONSchema = "json_schema"
	CapVision     = "vision"
)

// Routing policies accepted in RoutingConfig.Policy. The empty policy means
// automatic routing is not configured.
const (
	PolicyCheapest  = "cheapest"
	PolicyStrongest = "strongest"
	PolicyRules     = "rules"
)

// Quota backends accepted in QuotaConfig.Backend.
const (
	QuotaMemory = "memory"
	QuotaRedis  = "redis"
)

var knownCapabilities = map[string]bool{
	CapStreaming: true, CapTools: true, CapJSONObject: true, CapJSONSchema: true, CapVision: true,
}

// Defaults applied when a field is omitted.
const (
	defaultListen           = ":8080"
	defaultMaxBodyBytes     = 4194304
	defaultDefaultTimeoutMs = 120000
	defaultMaxTimeoutMs     = 600000
	defaultShutdownGraceMs  = 30000
	defaultTTFTTimeoutMs    = 10000
	defaultIdleTimeoutMs    = 30000
	defaultMaxInflight      = 1024
	defaultQueueWaitMs      = 100
	defaultContextWindow    = 8192
	defaultMaxOutputTokens  = 4096
	defaultMaxTokens        = 4096
	defaultMaxAttempts      = 3
	maxMaxAttempts          = 5
	rpmBurstDivisor         = 6
)

var sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Config is the root of the configuration file.
type Config struct {
	Server    ServerConfig     `yaml:"server"`
	Providers []ProviderConfig `yaml:"providers"`
	Models    []ModelConfig    `yaml:"models"`
	Tenants   []TenantConfig   `yaml:"tenants"`

	Routing     RoutingConfig     `yaml:"routing"`
	Quota       QuotaConfig       `yaml:"quota"`
	Reliability ReliabilityConfig `yaml:"reliability"`
}

// ServerConfig holds listener and request limits.
type ServerConfig struct {
	Listen           string `yaml:"listen"`
	MaxBodyBytes     int64  `yaml:"max_body_bytes"`
	DefaultTimeoutMs int    `yaml:"default_timeout_ms"`
	MaxTimeoutMs     int    `yaml:"max_timeout_ms"`
	ShutdownGraceMs  int    `yaml:"shutdown_grace_ms"`
	MaxInflight      int    `yaml:"max_inflight"`
	QueueWaitMs      int    `yaml:"queue_wait_ms"`
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

	ContextWindow   int `yaml:"context_window"`
	MaxOutputTokens int `yaml:"max_output_tokens"`
	// Capabilities lists supported features. A nil list (key absent) defaults
	// to streaming only; an explicit empty list means none.
	Capabilities []string `yaml:"capabilities"`
	// Prices are decimal USD per million tokens, parsed by money.ParseUSDPerMTok.
	InputPricePerMTok  string `yaml:"input_price_per_mtok"`
	OutputPricePerMTok string `yaml:"output_price_per_mtok"`
	Rank               int    `yaml:"rank"`
}

// HasCapability reports whether the model supports capability c. Call it on
// a config returned by Parse so defaults have been applied.
func (m ModelConfig) HasCapability(c string) bool {
	for _, have := range m.Capabilities {
		if have == c {
			return true
		}
	}
	return false
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

	// Rate limits; zero means unlimited.
	RPM              int `yaml:"rpm"`
	RPMBurst         int `yaml:"rpm_burst"`
	TPM              int `yaml:"tpm"`
	TPMBurst         int `yaml:"tpm_burst"`
	MaxConcurrency   int `yaml:"max_concurrency"`
	DefaultMaxTokens int `yaml:"default_max_tokens"`
}

// RoutingConfig configures static routing for the "auto" model.
type RoutingConfig struct {
	Policy       string       `yaml:"policy"`
	DefaultModel string       `yaml:"default_model"`
	Rules        []RuleConfig `yaml:"rules"`
}

// RuleConfig sends matching requests to Model.
type RuleConfig struct {
	Name  string    `yaml:"name"`
	Model string    `yaml:"model"`
	When  RuleMatch `yaml:"when"`
}

// RuleMatch holds the conditions of a rule; all set conditions must hold.
type RuleMatch struct {
	MinPromptBytes int      `yaml:"min_prompt_bytes"`
	MaxPromptBytes int      `yaml:"max_prompt_bytes"`
	ContainsAny    []string `yaml:"contains_any"`
	Requires       []string `yaml:"requires"`
}

func (r RuleMatch) isEmpty() bool {
	return r.MinPromptBytes == 0 && r.MaxPromptBytes == 0 && len(r.ContainsAny) == 0 && len(r.Requires) == 0
}

// QuotaConfig selects the rate-limit counter backend.
type QuotaConfig struct {
	Backend     string `yaml:"backend"`
	RedisURLEnv string `yaml:"redis_url_env"`
}

// ReliabilityConfig bounds retry and failover behavior.
type ReliabilityConfig struct {
	MaxAttempts int `yaml:"max_attempts"`
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
	if s.MaxInflight == 0 {
		s.MaxInflight = defaultMaxInflight
	}
	if s.QueueWaitMs == 0 {
		s.QueueWaitMs = defaultQueueWaitMs
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
		m := &c.Models[i]
		if m.ContextWindow == 0 {
			m.ContextWindow = defaultContextWindow
		}
		if m.MaxOutputTokens == 0 {
			m.MaxOutputTokens = defaultMaxOutputTokens
		}
		if m.Capabilities == nil {
			m.Capabilities = []string{CapStreaming}
		}
		if m.InputPricePerMTok == "" {
			m.InputPricePerMTok = "0"
		}
		if m.OutputPricePerMTok == "" {
			m.OutputPricePerMTok = "0"
		}
		for j := range c.Models[i].Deployments {
			d := &c.Models[i].Deployments[j]
			if d.Name == "" && d.Provider != "" && d.UpstreamModel != "" {
				d.Name = d.Provider + "/" + d.UpstreamModel
			}
		}
	}
	for i := range c.Tenants {
		t := &c.Tenants[i]
		if t.RPM > 0 && t.RPMBurst == 0 {
			t.RPMBurst = max(1, t.RPM/rpmBurstDivisor)
		}
		if t.TPMBurst == 0 {
			t.TPMBurst = t.TPM
		}
		if t.DefaultMaxTokens == 0 {
			t.DefaultMaxTokens = defaultMaxTokens
		}
	}
	if c.Quota.Backend == "" {
		c.Quota.Backend = QuotaMemory
	}
	if c.Reliability.MaxAttempts == 0 {
		c.Reliability.MaxAttempts = defaultMaxAttempts
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
	c.validateRouting(models, add)
	c.validateQuota(add)
	c.validateReliability(add)

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
	if s.MaxInflight < 0 {
		add("server.max_inflight", "must not be negative, got %d", s.MaxInflight)
	}
	if s.QueueWaitMs < 0 {
		add("server.queue_wait_ms", "must not be negative, got %d", s.QueueWaitMs)
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
	deploymentNames := make(map[string]bool)
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
		c.validateModelCatalog(path, m, add)
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
				if deploymentNames[d.Name] {
					add(dpath+".name", "duplicate deployment name %q", d.Name)
				}
				deploymentNames[d.Name] = true
			}
		}
	}
	return ids
}

func (c *Config) validateModelCatalog(path string, m ModelConfig, add addFunc) {
	if m.ContextWindow < 0 {
		add(path+".context_window", "must not be negative, got %d", m.ContextWindow)
	}
	if m.MaxOutputTokens < 0 {
		add(path+".max_output_tokens", "must not be negative, got %d", m.MaxOutputTokens)
	}
	if m.ContextWindow > 0 && m.MaxOutputTokens > m.ContextWindow {
		add(path+".max_output_tokens", "%d exceeds context_window %d", m.MaxOutputTokens, m.ContextWindow)
	}
	if m.Rank < 0 {
		add(path+".rank", "must not be negative, got %d", m.Rank)
	}
	seen := make(map[string]bool)
	for k, cp := range m.Capabilities {
		cpath := fmt.Sprintf("%s.capabilities[%d]", path, k)
		switch {
		case !knownCapabilities[cp]:
			add(cpath, "unknown capability %q", cp)
		case seen[cp]:
			add(cpath, "duplicate capability %q", cp)
		}
		seen[cp] = true
	}
	if _, err := money.ParseUSDPerMTok(m.InputPricePerMTok); err != nil {
		add(path+".input_price_per_mtok", "%v", err)
	}
	if _, err := money.ParseUSDPerMTok(m.OutputPricePerMTok); err != nil {
		add(path+".output_price_per_mtok", "%v", err)
	}
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
		for _, f := range []struct {
			name string
			v    int
		}{
			{"rpm", t.RPM}, {"rpm_burst", t.RPMBurst}, {"tpm", t.TPM}, {"tpm_burst", t.TPMBurst},
			{"max_concurrency", t.MaxConcurrency}, {"default_max_tokens", t.DefaultMaxTokens},
		} {
			if f.v < 0 {
				add(path+"."+f.name, "must not be negative, got %d", f.v)
			}
		}
		if t.RPM == 0 && t.RPMBurst > 0 {
			add(path+".rpm_burst", "rpm_burst set without rpm")
		}
		if t.TPM == 0 && t.TPMBurst > 0 {
			add(path+".tpm_burst", "tpm_burst set without tpm")
		}
	}
}

func (c *Config) validateRouting(models map[string]bool, add addFunc) {
	r := c.Routing
	switch r.Policy {
	case "", PolicyCheapest, PolicyStrongest, PolicyRules:
	default:
		add("routing.policy", "unknown policy %q (want %q, %q, or %q)", r.Policy, PolicyCheapest, PolicyStrongest, PolicyRules)
	}
	if r.DefaultModel != "" && !models[r.DefaultModel] {
		add("routing.default_model", "unknown model %q", r.DefaultModel)
	}
	if r.Policy == PolicyRules {
		if r.DefaultModel == "" {
			add("routing.default_model", "required when policy is %q", PolicyRules)
		}
		if len(r.Rules) == 0 {
			add("routing.rules", "at least one rule is required when policy is %q", PolicyRules)
		}
	}
	names := make(map[string]bool)
	for i, rule := range r.Rules {
		path := fmt.Sprintf("routing.rules[%d]", i)
		switch {
		case rule.Name == "":
			add(path+".name", "must not be empty")
		case names[rule.Name]:
			add(path+".name", "duplicate rule name %q", rule.Name)
		default:
			names[rule.Name] = true
		}
		if !models[rule.Model] {
			add(path+".model", "unknown model %q", rule.Model)
		}
		w := rule.When
		if w.isEmpty() {
			add(path+".when", "must set at least one condition")
		}
		if w.MinPromptBytes < 0 {
			add(path+".when.min_prompt_bytes", "must not be negative, got %d", w.MinPromptBytes)
		}
		if w.MaxPromptBytes < 0 {
			add(path+".when.max_prompt_bytes", "must not be negative, got %d", w.MaxPromptBytes)
		}
		if w.MinPromptBytes > 0 && w.MaxPromptBytes > 0 && w.MinPromptBytes > w.MaxPromptBytes {
			add(path+".when.min_prompt_bytes", "%d exceeds max_prompt_bytes %d", w.MinPromptBytes, w.MaxPromptBytes)
		}
		for k, s := range w.ContainsAny {
			if s == "" {
				add(fmt.Sprintf("%s.when.contains_any[%d]", path, k), "must not be empty")
			}
		}
		for k, req := range w.Requires {
			if !knownCapabilities[req] {
				add(fmt.Sprintf("%s.when.requires[%d]", path, k), "unknown capability %q", req)
			}
		}
	}
}

func (c *Config) validateQuota(add addFunc) {
	switch c.Quota.Backend {
	case QuotaMemory:
	case QuotaRedis:
		if c.Quota.RedisURLEnv == "" {
			add("quota.redis_url_env", "required when backend is %q", QuotaRedis)
		}
	default:
		add("quota.backend", "unknown backend %q (want %q or %q)", c.Quota.Backend, QuotaMemory, QuotaRedis)
	}
}

func (c *Config) validateReliability(add addFunc) {
	if a := c.Reliability.MaxAttempts; a < 1 || a > maxMaxAttempts {
		add("reliability.max_attempts", "must be between 1 and %d, got %d", maxMaxAttempts, a)
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
