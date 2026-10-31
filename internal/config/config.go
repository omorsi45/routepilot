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
	"math"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
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
// automatic routing is not configured. PolicyLearned routes with a learned policy
// artifact from the policies section (see PoliciesConfig).
const (
	PolicyCheapest  = "cheapest"
	PolicyStrongest = "strongest"
	PolicyRules     = "rules"
	PolicyLearned   = "learned"
)

// Quota backends accepted in QuotaConfig.Backend.
const (
	QuotaMemory = "memory"
	QuotaRedis  = "redis"
)

// What to do when no model meets a request's quality floor and latency SLO, accepted in
// TenantConfig.OnInfeasible and in the routing policy's configuration: relax the
// constraints and answer anyway, or reject the request.
const (
	OnInfeasibleBestEffort = "best_effort"
	OnInfeasibleReject     = "reject"
)

var knownCapabilities = map[string]bool{
	CapStreaming: true, CapTools: true, CapJSONObject: true, CapJSONSchema: true, CapVision: true,
}

// Defaults applied when a field is omitted. The listen address is loopback because the
// gateway speaks plaintext HTTP and serves /metrics unauthenticated: binding another
// interface is a choice to make in the config, not a default.
const (
	defaultListen           = "127.0.0.1:8080"
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
	defaultPoliciesRoot     = "policies"
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

	// Policies names the learned policy artifacts: the stable one, an optional canary
	// and shadow ones. It is the only section a running server reloads.
	Policies PoliciesConfig `yaml:"policies"`

	// Accounting selects where the per-attempt usage ledger is kept. Absent means none.
	Accounting AccountingConfig `yaml:"accounting"`

	// DecisionLog enables the append-only JSONL log of routing decisions and feedback. Absent
	// (no path) disables it, and the feedback endpoint with it.
	DecisionLog DecisionLogConfig `yaml:"decision_log"`

	// Tracing exports OpenTelemetry traces to an OTLP/HTTP endpoint. Absent (no endpoint)
	// means the noop tracer: no spans are recorded and nothing is sent.
	Tracing TracingConfig `yaml:"tracing"`
	// path and baseDir are set by Load: the absolute path of the file the config was
	// read from and its directory, which relative paths in the config resolve against.
	// Parse leaves both empty, which means the working directory.
	path, baseDir string
}

// BaseDir is the directory of the file Load read, the directory relative paths in the
// config (policies.root, a policy's config, server.latency_profiles) are relative to. It
// is empty for a config built by Parse, meaning the working directory.
func (c *Config) BaseDir() string { return c.baseDir }

// Path is the absolute path of the file Load read, empty for a config built by Parse.
func (c *Config) Path() string { return c.path }

// ResolvePath resolves p against baseDir: an absolute p, an empty p or an empty baseDir
// leave p as it is.
func ResolvePath(baseDir, p string) string {
	if p == "" || baseDir == "" || filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(baseDir, p)
}

// PoliciesConfig lists the policy artifacts a learned routing policy serves with. An
// artifact lives in the directory <Root>/<name>/<version>. Root and every Config path are
// relative to the config file's directory.
type PoliciesConfig struct {
	// Root holds the artifact directories. It defaults to "policies".
	Root string `yaml:"root"`
	// Stable serves every request the canary does not take. Required by the learned policy.
	Stable PolicyRef `yaml:"stable"`
	// Canary, when set, takes Percent of the requests instead of Stable.
	Canary *CanaryRef `yaml:"canary"`
	// Shadow policies decide on every request but never act: their decisions are only logged.
	Shadow []PolicyRef `yaml:"shadow"`
}

// PolicyRef names one policy: an artifact version and the optimizer configuration to run
// it with.
type PolicyRef struct {
	// Name and Version are the artifact's directory under Root.
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
	// Config is the path of the optimizer configuration file (mode, lambda, floor, ...).
	Config string `yaml:"config"`
}

// CanaryRef is a policy that takes a share of the traffic.
type CanaryRef struct {
	PolicyRef `yaml:",inline"`
	// Percent is the share of requests, from 0 to 100, that the canary serves.
	Percent float64 `yaml:"percent"`
}

// Validate checks the shape of the section: every named policy needs a name, a version
// and a configuration path, the name and version must be single path segments, and the
// canary percent must be in [0, 100]. An absent stable policy is not an error here; the
// learned routing policy requires it.
func (p PoliciesConfig) Validate() error {
	var errs []error
	p.validate(func(path, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", path, fmt.Sprintf(format, args...)))
	})
	return errors.Join(errs...)
}

func (p PoliciesConfig) validate(add addFunc) {
	if p.Stable != (PolicyRef{}) {
		p.Stable.validate("policies.stable", add)
	}
	if c := p.Canary; c != nil {
		c.PolicyRef.validate("policies.canary", add)
		if !(c.Percent >= 0 && c.Percent <= 100) {
			add("policies.canary.percent", "must be between 0 and 100, got %v", c.Percent)
		}
	}
	for i, s := range p.Shadow {
		s.validate(fmt.Sprintf("policies.shadow[%d]", i), add)
	}
}

func (r PolicyRef) validate(path string, add addFunc) {
	for _, f := range []struct{ name, value string }{{"name", r.Name}, {"version", r.Version}} {
		switch {
		case f.value == "":
			add(path+"."+f.name, "must not be empty")
		case f.value == "." || f.value == ".." || strings.ContainsAny(f.value, `/\`):
			add(path+"."+f.name, "must be a single directory name, got %q", f.value)
		}
	}
	if r.Config == "" {
		add(path+".config", "must not be empty")
	}
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
	// LatencyProfiles is the path of a latency profile file (see the latency package)
	// that gives the learned policy its prior TTFT and TPOT per model. Optional: without
	// it the policy only has live observations.
	LatencyProfiles string `yaml:"latency_profiles"`
	// AdminKeySHA256 is the SHA-256 (64 lowercase hex characters) of the key that opens the
	// admin routes under /admin/policies. Empty, the routes answer 404. It must not be the
	// hash of a tenant's key: a tenant key is not an admin key.
	AdminKeySHA256 string `yaml:"admin_key_sha256"`
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
	// PromptOverheadTokens is the number of hidden prompt tokens the provider
	// adds to every request of this model (Ruling V1-R24). Zero means none.
	PromptOverheadTokens int `yaml:"prompt_overhead_tokens"`
}

// HasCapability reports whether the model supports capability c. A nil
// Capabilities list is treated as streaming only, the same default Parse
// applies, so a hand-built config behaves like a parsed one.
func (m ModelConfig) HasCapability(c string) bool {
	if m.Capabilities == nil {
		return c == CapStreaming
	}
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
	// MaxInflight caps the non-streaming attempts in flight on this deployment (a bulkhead).
	// Zero, the default, means unlimited; a deployment at its cap is skipped like an
	// unavailable one.
	MaxInflight int `yaml:"max_inflight"`
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

	// MinQuality is the quality floor the learned routing policy holds this tenant's
	// requests to, a lower bound on the probability that the answer succeeds. Nil (key
	// absent) means unset: the policy's default floor applies. A request may ask for its
	// own floor.
	MinQuality *float64 `yaml:"min_quality"`
	// OnInfeasible is what the learned policy does for this tenant when no model meets
	// the floor and any latency SLO: OnInfeasibleBestEffort or OnInfeasibleReject. Empty
	// means the policy's own setting.
	OnInfeasible string `yaml:"on_infeasible"`

	// BudgetUSD is the tenant's spending limit per budget period as a decimal number of US
	// dollars with at most twelve fractional digits (money.ParseUSD). Empty means no budget:
	// the tenant never touches the budget ledger. BudgetPeriod is quota.PeriodMonthly (the
	// default when BudgetUSD is set) or quota.PeriodTotal.
	BudgetUSD    string `yaml:"budget_usd"`
	BudgetPeriod string `yaml:"budget_period"`
}

// maxRedisBudgetUSD is the largest budget_usd the Redis ledger keeps exactly (Ruling V1-R29):
// its Lua arithmetic is exact for integers below 2^53, and 9,000 USD is 9e15 picodollars.
const (
	maxRedisBudgetUSD  = 9000
	maxRedisBudgetPico = money.PicoUSD(maxRedisBudgetUSD) * 1_000_000_000_000
)

// BudgetLimit returns the tenant's budget limit and whether it has one. The period defaults
// to monthly. A malformed budget_usd is an error, which a validated config never has.
func (t TenantConfig) BudgetLimit() (quota.BudgetLimit, bool, error) {
	if t.BudgetUSD == "" {
		return quota.BudgetLimit{}, false, nil
	}
	limit, err := money.ParseUSD(t.BudgetUSD)
	if err != nil {
		return quota.BudgetLimit{}, false, fmt.Errorf("tenant %q budget_usd: %w", t.ID, err)
	}
	period := t.BudgetPeriod
	if period == "" {
		period = quota.PeriodMonthly
	}
	return quota.BudgetLimit{Limit: limit, Period: period}, true, nil
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

// Accounting drivers accepted in AccountingConfig.Driver. None keeps no ledger, so a default
// run writes no database file (Ruling V1-R12).
const (
	AccountingNone     = "none"
	AccountingSQLite   = "sqlite"
	AccountingPostgres = "postgres"
)

// AccountingConfig selects the store of the usage ledger. The SQLite path is relative to the
// config file's directory. The PostgreSQL connection string is never written in the file: it
// is read from the environment variable PostgresDSNEnv names, like the Redis URL.
type AccountingConfig struct {
	Driver         string `yaml:"driver"`
	SQLitePath     string `yaml:"sqlite_path"`
	PostgresDSNEnv string `yaml:"postgres_dsn_env"`
}

func (a AccountingConfig) validate(add addFunc) {
	switch a.Driver {
	case AccountingNone:
		if a.SQLitePath != "" {
			add("accounting.sqlite_path", "set but accounting.driver is %q", a.Driver)
		}
		if a.PostgresDSNEnv != "" {
			add("accounting.postgres_dsn_env", "set but accounting.driver is %q", a.Driver)
		}
	case AccountingSQLite:
		if a.SQLitePath == "" {
			add("accounting.sqlite_path", "required when driver is %q", a.Driver)
		}
		if a.PostgresDSNEnv != "" {
			add("accounting.postgres_dsn_env", "set but accounting.driver is %q", a.Driver)
		}
	case AccountingPostgres:
		if a.PostgresDSNEnv == "" {
			add("accounting.postgres_dsn_env", "required when driver is %q", a.Driver)
		}
		if a.SQLitePath != "" {
			add("accounting.sqlite_path", "set but accounting.driver is %q", a.Driver)
		}
	default:
		add("accounting.driver", "unknown driver %q (want %q, %q or %q)", a.Driver, AccountingNone, AccountingSQLite, AccountingPostgres)
	}
}

// defaultDecisionLogMaxBytes is the size at which the decision log is rotated when
// decision_log.max_bytes is not set: 100 MiB.
const defaultDecisionLogMaxBytes = 104857600

// minDecisionLogMaxBytes is the smallest decision_log.max_bytes: 1 MiB. A smaller file would
// rotate several times a second at a few thousand requests per second, leaving a stream of
// tiny files for the training pipeline to read.
const minDecisionLogMaxBytes = 1 << 20

// DecisionLogConfig configures the decision log. Path is relative to the config file's
// directory. MaxBytes is the size past which the file is rotated; it defaults to 100 MiB
// once a path is set and must be at least 1 MiB.
type DecisionLogConfig struct {
	Path     string `yaml:"path"`
	MaxBytes int64  `yaml:"max_bytes"`
}

func (d DecisionLogConfig) validate(add addFunc) {
	if d.MaxBytes < 0 {
		add("decision_log.max_bytes", "must not be negative, got %d", d.MaxBytes)
	} else if d.MaxBytes > 0 && d.MaxBytes < minDecisionLogMaxBytes {
		add("decision_log.max_bytes", "must be at least 1 MiB (%d bytes), got %d", minDecisionLogMaxBytes, d.MaxBytes)
	}
	if d.Path == "" && d.MaxBytes != 0 {
		add("decision_log.max_bytes", "set without decision_log.path")
	}
}

// TracingConfig configures OpenTelemetry tracing. OTLPEndpoint is the URL of an OTLP/HTTP
// collector, such as http://localhost:4318 (the path /v1/traces is added when the URL has
// none); empty means the noop tracer. SampleRatio is the share of new traces that are
// recorded, in [0, 1], applied to a request that carries no sampling decision of its own. It
// defaults to 1 once an endpoint is set. It is a pointer so that an explicit 0 differs from
// an absent value.
type TracingConfig struct {
	OTLPEndpoint string   `yaml:"otlp_endpoint"`
	SampleRatio  *float64 `yaml:"sample_ratio"`
}

// Ratio is the sample ratio: SampleRatio, or 1 when it is not set.
func (t TracingConfig) Ratio() float64 {
	if t.SampleRatio == nil {
		return 1
	}
	return *t.SampleRatio
}

func (t TracingConfig) validate(add addFunc) {
	if t.OTLPEndpoint != "" {
		if err := CheckOTLPEndpoint(t.OTLPEndpoint); err != nil {
			add("tracing.otlp_endpoint", "%v", err)
		}
	}
	if t.SampleRatio != nil {
		if r := *t.SampleRatio; math.IsNaN(r) || r < 0 || r > 1 {
			add("tracing.sample_ratio", "must be between 0 and 1, got %v", r)
		}
		if t.OTLPEndpoint == "" {
			add("tracing.sample_ratio", "set without tracing.otlp_endpoint")
		}
	}
}

// CheckOTLPEndpoint reports why endpoint is not a usable OTLP/HTTP collector URL: it must be
// an http or https URL with a host and no credentials, query or fragment. The error never
// repeats the value, because a collector URL may carry a credential.
func CheckOTLPEndpoint(endpoint string) error {
	u, err := url.Parse(endpoint)
	switch {
	case err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "":
		return errors.New("must be an http or https URL with a host, such as http://localhost:4318")
	case u.User != nil:
		return errors.New("must not carry credentials")
	case u.RawQuery != "" || u.Fragment != "":
		return errors.New("must not carry a query or a fragment")
	}
	return nil
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

// Load reads and parses the config file at path, and records the file's absolute path and
// directory (see Config.BaseDir).
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load config %s: %w", path, err)
	}
	cfg, err := Parse(b)
	if err != nil {
		return nil, fmt.Errorf("load config %s: %w", path, err)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	cfg.path, cfg.baseDir = abs, filepath.Dir(abs)
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
			// The default cap, but never above the model's own window: a small local model
			// would otherwise be refused for a value the operator never wrote.
			m.MaxOutputTokens = defaultMaxOutputTokens
			if m.ContextWindow > 0 {
				m.MaxOutputTokens = min(defaultMaxOutputTokens, m.ContextWindow)
			}
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
		if t.BudgetUSD != "" && t.BudgetPeriod == "" {
			t.BudgetPeriod = quota.PeriodMonthly
		}
	}
	if c.Quota.Backend == "" {
		c.Quota.Backend = QuotaMemory
	}
	if c.Reliability.MaxAttempts == 0 {
		c.Reliability.MaxAttempts = defaultMaxAttempts
	}
	if c.Policies.Root == "" {
		c.Policies.Root = defaultPoliciesRoot
	}
	if c.Accounting.Driver == "" {
		c.Accounting.Driver = AccountingNone
	}
	if c.DecisionLog.Path != "" && c.DecisionLog.MaxBytes == 0 {
		c.DecisionLog.MaxBytes = defaultDecisionLogMaxBytes
	}
	if c.Tracing.OTLPEndpoint != "" && c.Tracing.SampleRatio == nil {
		one := 1.0
		c.Tracing.SampleRatio = &one
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
	c.Policies.validate(add)
	c.validateQuota(add)
	c.validateReliability(add)
	c.Accounting.validate(add)
	c.DecisionLog.validate(add)
	c.Tracing.validate(add)

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
	if h := s.AdminKeySHA256; h != "" {
		// The value is not repeated in the errors: it is the hash of a credential.
		if !sha256Hex.MatchString(h) {
			add("server.admin_key_sha256", "must be 64 lowercase hex characters")
		}
		for _, t := range c.Tenants {
			if t.APIKeySHA256 == h {
				add("server.admin_key_sha256", "must differ from every tenant's api_key_sha256: a tenant key must not open the admin routes")
				break
			}
		}
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
			if d.MaxInflight < 0 {
				add(dpath+".max_inflight", "must not be negative, got %d", d.MaxInflight)
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
	if m.PromptOverheadTokens < 0 {
		add(path+".prompt_overhead_tokens", "must not be negative, got %d", m.PromptOverheadTokens)
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
		if q := t.MinQuality; q != nil && !(*q >= 0 && *q <= 1) {
			add(path+".min_quality", "must be between 0 and 1, got %v", *q)
		}
		switch t.OnInfeasible {
		case "", OnInfeasibleBestEffort, OnInfeasibleReject:
		default:
			add(path+".on_infeasible", "unknown value %q (want %q or %q)", t.OnInfeasible, OnInfeasibleBestEffort, OnInfeasibleReject)
		}
		if t.RPM == 0 && t.RPMBurst > 0 {
			add(path+".rpm_burst", "rpm_burst set without rpm")
		}
		if t.TPM == 0 && t.TPMBurst > 0 {
			add(path+".tpm_burst", "tpm_burst set without tpm")
		}
		if burstWindowTooLong(t.RPM, t.RPMBurst) {
			add(path+".rpm_burst", "window exceeds 24h")
		}
		if burstWindowTooLong(t.TPM, t.TPMBurst) {
			add(path+".tpm_burst", "window exceeds 24h")
		}
		// N3 (Ruling V1-R16): a request without max_tokens reserves up to its input bound plus
		// default_max_tokens tokens, and the input bound is at least 24 (fixed and per-message
		// overhead), so a smaller burst could never admit it.
		if t.TPM > 0 && t.TPMBurst < t.DefaultMaxTokens+n3MinInputBound {
			add(path+".tpm_burst", "below default_max_tokens + %d; requests without max_tokens could never be admitted", n3MinInputBound)
		}
		c.validateTenantBudget(path, t, add)
	}
}

// n3MinInputBound is the smallest input token bound a request can have: the fixed overhead
// (16) plus the per-message overhead (8) of one message.
const n3MinInputBound = 24

func (c *Config) validateTenantBudget(path string, t TenantConfig, add addFunc) {
	switch t.BudgetPeriod {
	case "", quota.PeriodMonthly, quota.PeriodTotal:
	default:
		add(path+".budget_period", "unknown value %q (want %q or %q)", t.BudgetPeriod, quota.PeriodMonthly, quota.PeriodTotal)
	}
	if t.BudgetUSD == "" {
		if t.BudgetPeriod != "" {
			add(path+".budget_period", "budget_period set without budget_usd")
		}
		return
	}
	limit, err := money.ParseUSD(t.BudgetUSD)
	if err != nil {
		add(path+".budget_usd", "%v", err)
		return
	}
	if c.Quota.Backend == QuotaRedis && limit > maxRedisBudgetPico {
		add(path+".budget_usd", "above %d USD, which the Redis budget ledger cannot keep exactly", maxRedisBudgetUSD)
	}
}

// maxBurstWindowMicros is the longest GCRA burst window the limiters accept,
// 24 hours, so a stored theoretical arrival time stays within a day of now.
const maxBurstWindowMicros = 86_400_000_000

// burstWindowTooLong reports whether a limit of perMinute with the given burst
// has a GCRA window over 24 hours. The window is the quota package's exact
// burst*60_000_000/perMinute microseconds (quota.WindowMicros), the same value
// the limiters enforce. A zero limit is unlimited and never too long.
func burstWindowTooLong(perMinute, burst int) bool {
	if perMinute <= 0 || burst <= 0 {
		return false
	}
	return quota.WindowMicros(int64(burst), int64(perMinute)) > maxBurstWindowMicros
}

func (c *Config) validateRouting(models map[string]bool, add addFunc) {
	r := c.Routing
	switch r.Policy {
	case "", PolicyCheapest, PolicyStrongest, PolicyRules, PolicyLearned:
	default:
		add("routing.policy", "unknown policy %q (want %q, %q, %q, or %q)", r.Policy, PolicyCheapest, PolicyStrongest, PolicyRules, PolicyLearned)
	}
	if r.Policy == PolicyLearned && c.Policies.Stable == (PolicyRef{}) {
		add("policies.stable", "required when routing.policy is %q", PolicyLearned)
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
