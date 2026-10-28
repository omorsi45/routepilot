package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func keyHash(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// baseYAML returns a valid config with the given server block and extra
// replacements applied, so each rejection case changes exactly one thing.
func baseYAML(t *testing.T, replacements ...string) string {
	t.Helper()
	y := fmt.Sprintf(`server:
  default_timeout_ms: 1000
  max_timeout_ms: 2000
providers:
  - name: mock
    type: openaicompat
    base_url: http://127.0.0.1:9090/v1
models:
  - id: small
    deployments:
      - provider: mock
        upstream_model: m1
tenants:
  - id: team-a
    api_key_sha256: %s
    allowed_models: ["small"]
`, keyHash("test-key-tenant-a"))
	if len(replacements)%2 != 0 {
		t.Fatal("replacements must be old/new pairs")
	}
	for i := 0; i < len(replacements); i += 2 {
		if !strings.Contains(y, replacements[i]) {
			t.Fatalf("base YAML does not contain %q", replacements[i])
		}
		y = strings.Replace(y, replacements[i], replacements[i+1], 1)
	}
	return y
}

func TestParseValidAppliesDefaults(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "valid.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	s := cfg.Server
	if s.Listen != ":8080" {
		t.Errorf("Listen = %q", s.Listen)
	}
	if s.MaxBodyBytes != 4194304 {
		t.Errorf("MaxBodyBytes = %d", s.MaxBodyBytes)
	}
	if s.DefaultTimeoutMs != 120000 || s.MaxTimeoutMs != 600000 || s.ShutdownGraceMs != 30000 {
		t.Errorf("server timeouts = %+v", s)
	}

	if got := cfg.Providers[0].TTFTTimeoutMs; got != 10000 {
		t.Errorf("default TTFTTimeoutMs = %d", got)
	}
	if got := cfg.Providers[0].IdleTimeoutMs; got != 30000 {
		t.Errorf("default IdleTimeoutMs = %d", got)
	}
	if got := cfg.Providers[1].TTFTTimeoutMs; got != 5000 {
		t.Errorf("explicit TTFTTimeoutMs = %d", got)
	}
	if got := cfg.Providers[1].APIKeyEnv; got != "ANTHROPIC_API_KEY" {
		t.Errorf("APIKeyEnv = %q", got)
	}

	m, ok := cfg.Model("small")
	if !ok {
		t.Fatal("Model(small) not found")
	}
	if got := m.Deployments[0].Name; got != "mock/mock-small-v1" {
		t.Errorf("defaulted deployment name = %q", got)
	}
	if got := m.Deployments[1].Name; got != "fallback" {
		t.Errorf("explicit deployment name = %q", got)
	}
	if _, ok := cfg.Model("nope"); ok {
		t.Error("Model(nope) should not be found")
	}
	if got := cfg.Tenants[0].MaxTimeoutMs; got != 60000 {
		t.Errorf("tenant MaxTimeoutMs = %d", got)
	}
}

func TestParseRejects(t *testing.T) {
	good := keyHash("test-key-tenant-a")
	cases := []struct {
		name string
		yaml string
		want []string
	}{
		{"unknown key", baseYAML(t, "providers:", "bogus: 1\nproviders:"), []string{"bogus"}},
		{"empty document", "", []string{"config is empty"}},
		{"multiple documents", baseYAML(t) + "---\nproviders: []\n", []string{"multiple"}},
		{"unknown provider type", baseYAML(t, "type: openaicompat", "type: foo"),
			[]string{`providers[0].type: unknown provider type "foo"`}},
		{"duplicate provider name", baseYAML(t, "models:", "  - name: mock\n    type: openaicompat\n    base_url: http://x.example/v1\nmodels:"),
			[]string{"providers[1].name: duplicate"}},
		{"duplicate model id", baseYAML(t, "tenants:", "  - id: small\n    deployments:\n      - provider: mock\n        upstream_model: m2\ntenants:"),
			[]string{"models[1].id: duplicate"}},
		{"deployment unknown provider", baseYAML(t, "- provider: mock", "- provider: ghost"),
			[]string{`models[0].deployments[0].provider: unknown provider "ghost"`}},
		{"bad sha256 length", baseYAML(t, good, "abc123"), []string{"tenants[0].api_key_sha256"}},
		{"uppercase sha256", baseYAML(t, good, strings.ToUpper(good)), []string{"tenants[0].api_key_sha256"}},
		{"reserved model id", baseYAML(t, "id: small", "id: auto"),
			[]string{`models[0].id: "auto" is reserved`}},
		{"default above max timeout", baseYAML(t, "default_timeout_ms: 1000", "default_timeout_ms: 3000"),
			[]string{"server.default_timeout_ms"}},
		{"negative timeout", baseYAML(t, "default_timeout_ms: 1000", "default_timeout_ms: -5"),
			[]string{"server.default_timeout_ms"}},
		{"tenant timeout above server max", baseYAML(t, `allowed_models: ["small"]`, "allowed_models: [\"small\"]\n    max_timeout_ms: 9000"),
			[]string{"tenants[0].max_timeout_ms"}},
		{"bad base_url scheme", baseYAML(t, "http://127.0.0.1:9090/v1", "ftp://127.0.0.1/v1"),
			[]string{"providers[0].base_url"}},
		{"base_url without host", baseYAML(t, "http://127.0.0.1:9090/v1", "http:///v1"),
			[]string{"providers[0].base_url"}},
		{"allowed model unknown", baseYAML(t, `["small"]`, `["ghost"]`),
			[]string{`tenants[0].allowed_models[0]: unknown model "ghost"`}},
		{"model without deployments", baseYAML(t, "    deployments:\n      - provider: mock\n        upstream_model: m1\n", "    deployments: []\n"),
			[]string{"models[0].deployments"}},
		{"missing upstream_model", baseYAML(t, "upstream_model: m1", "upstream_model: \"\""),
			[]string{"models[0].deployments[0].upstream_model"}},
		{"no providers", "models: []\ntenants: []\n", []string{"providers: at least one provider"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := Parse([]byte(tc.yaml))
			if err == nil {
				t.Fatalf("Parse succeeded, cfg = %+v", cfg)
			}
			for _, w := range tc.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not contain %q", err, w)
				}
			}
		})
	}
}

func TestParseDuplicatesAcrossEntries(t *testing.T) {
	hashA := keyHash("test-key-tenant-a")
	y := baseYAML(t,
		"tenants:", "tenants:\n  - id: team-a\n    api_key_sha256: "+hashA+"\n    allowed_models: [\"*\"]",
		"      - provider: mock\n        upstream_model: m1\n",
		"      - provider: mock\n        upstream_model: m1\n      - provider: mock\n        upstream_model: m1\n",
	)
	_, err := Parse([]byte(y))
	if err == nil {
		t.Fatal("expected error")
	}
	for _, w := range []string{
		"models[0].deployments[1].name: duplicate",
		"tenants[1].id: duplicate",
		"tenants[1].api_key_sha256: duplicate",
	} {
		if !strings.Contains(err.Error(), w) {
			t.Errorf("error %q does not contain %q", err, w)
		}
	}
}

func TestParseReportsAllErrors(t *testing.T) {
	y := baseYAML(t, "type: openaicompat", "type: foo", "id: small", "id: auto")
	_, err := Parse([]byte(y))
	if err == nil {
		t.Fatal("expected error")
	}
	for _, w := range []string{"providers[0].type", "models[0].id"} {
		if !strings.Contains(err.Error(), w) {
			t.Errorf("error %q does not contain %q", err, w)
		}
	}
}

func TestLoadWrapsPath(t *testing.T) {
	if _, err := Load(filepath.Join("testdata", "missing.yaml")); err == nil || !strings.Contains(err.Error(), "missing.yaml") {
		t.Fatalf("missing file error = %v", err)
	}
	p := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(p, []byte("bogus: 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err == nil || !strings.Contains(err.Error(), p) {
		t.Fatalf("invalid file error = %v", err)
	}
}

func TestTenantAllows(t *testing.T) {
	explicit := TenantConfig{AllowedModels: []string{"a", "b"}}
	if !explicit.Allows("a") || !explicit.Allows("b") {
		t.Error("explicit list should allow a and b")
	}
	if explicit.Allows("c") {
		t.Error("explicit list should not allow c")
	}
	wild := TenantConfig{AllowedModels: []string{"*"}}
	if !wild.Allows("anything") {
		t.Error(`"*" should allow any model`)
	}
	if (TenantConfig{}).Allows("a") {
		t.Error("empty list should allow nothing")
	}
}

func TestExampleConfigLoads(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "examples", "config.local.yaml"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := cfg.Server.Listen; got != "127.0.0.1:8080" {
		t.Errorf("example listen = %q, want loopback 127.0.0.1:8080", got)
	}
	for _, id := range []string{"mock-small", "mock-large", "qwen3-1.7b"} {
		if _, ok := cfg.Model(id); !ok {
			t.Errorf("example config missing model %q", id)
		}
	}
	if got := cfg.Tenants[0].APIKeySHA256; got != keyHash("demo-key-local") {
		t.Errorf("demo tenant hash = %q", got)
	}
	if !cfg.Tenants[0].Allows("qwen3-1.7b") {
		t.Error("demo tenant should allow all models")
	}
}

func TestModelDefaultsAndCapabilities(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	m := cfg.Models[0]
	if m.ContextWindow != 8192 || m.MaxOutputTokens != 4096 || m.Rank != 0 {
		t.Errorf("defaults = %d/%d/%d", m.ContextWindow, m.MaxOutputTokens, m.Rank)
	}
	if m.InputPricePerMTok != "0" || m.OutputPricePerMTok != "0" {
		t.Errorf("price defaults = %q/%q", m.InputPricePerMTok, m.OutputPricePerMTok)
	}
	if m.PromptOverheadTokens != 0 {
		t.Errorf("PromptOverheadTokens default = %d, want 0", m.PromptOverheadTokens)
	}
	if len(m.Capabilities) != 1 || m.Capabilities[0] != "streaming" {
		t.Errorf("Capabilities = %v, want [streaming]", m.Capabilities)
	}
	if !m.HasCapability("streaming") || m.HasCapability("tools") {
		t.Error("HasCapability mismatch for defaulted capabilities")
	}
	if cfg.Routing.Policy != "" || cfg.Quota.Backend != "memory" || cfg.Reliability.MaxAttempts != 3 {
		t.Errorf("section defaults = %+v %+v %+v", cfg.Routing, cfg.Quota, cfg.Reliability)
	}
	if cfg.Server.MaxInflight != 1024 || cfg.Server.QueueWaitMs != 100 {
		t.Errorf("server defaults = %+v", cfg.Server)
	}

	cfg, err = Parse([]byte(baseYAML(t, "    deployments:", "    capabilities: []\n    deployments:")))
	if err != nil {
		t.Fatalf("Parse explicit empty: %v", err)
	}
	if got := cfg.Models[0].Capabilities; got == nil || len(got) != 0 {
		t.Errorf("explicit [] Capabilities = %#v, want non-nil empty", got)
	}
	if cfg.Models[0].HasCapability("streaming") {
		t.Error("explicit [] should have no capabilities")
	}
}

func TestParseReadsPromptOverheadTokens(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t, "    deployments:", "    prompt_overhead_tokens: 150\n    deployments:")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got := cfg.Models[0].PromptOverheadTokens; got != 150 {
		t.Errorf("PromptOverheadTokens = %d, want 150", got)
	}
}

func TestParseRejectsM2(t *testing.T) {
	model2 := "  - id: big\n    deployments:\n      - provider: mock\n        upstream_model: m1\n"
	rules := func(extra string) string {
		return "routing:\n  policy: rules\n  default_model: small\n" + extra + "tenants:"
	}
	cases := []struct {
		name string
		yaml string
		want []string
	}{
		{"unknown capability", baseYAML(t, "    deployments:", "    capabilities: [telepathy]\n    deployments:"),
			[]string{`models[0].capabilities[0]: unknown capability "telepathy"`}},
		{"duplicate capability", baseYAML(t, "    deployments:", "    capabilities: [tools, tools]\n    deployments:"),
			[]string{"models[0].capabilities[1]: duplicate"}},
		{"bad input price", baseYAML(t, "    deployments:", "    input_price_per_mtok: \"1e3\"\n    deployments:"),
			[]string{"models[0].input_price_per_mtok"}},
		{"bad output price", baseYAML(t, "    deployments:", "    output_price_per_mtok: \"-1\"\n    deployments:"),
			[]string{"models[0].output_price_per_mtok"}},
		{"negative rank", baseYAML(t, "    deployments:", "    rank: -1\n    deployments:"),
			[]string{"models[0].rank"}},
		{"negative context window", baseYAML(t, "    deployments:", "    context_window: -1\n    deployments:"),
			[]string{"models[0].context_window"}},
		{"negative prompt overhead", baseYAML(t, "    deployments:", "    prompt_overhead_tokens: -1\n    deployments:"),
			[]string{"models[0].prompt_overhead_tokens"}},
		{"max output above context", baseYAML(t, "    deployments:", "    context_window: 1000\n    max_output_tokens: 2000\n    deployments:"),
			[]string{"models[0].max_output_tokens"}},
		{"duplicate deployment name across models", baseYAML(t, "tenants:", model2+"tenants:"),
			[]string{"models[1].deployments[0].name: duplicate deployment name"}},
		{"negative rpm", baseYAML(t, "allowed_models:", "rpm: -1\n    allowed_models:"),
			[]string{"tenants[0].rpm"}},
		{"negative tpm burst", baseYAML(t, "allowed_models:", "tpm: 10\n    tpm_burst: -1\n    allowed_models:"),
			[]string{"tenants[0].tpm_burst"}},
		{"negative max concurrency", baseYAML(t, "allowed_models:", "max_concurrency: -1\n    allowed_models:"),
			[]string{"tenants[0].max_concurrency"}},
		{"negative default max tokens", baseYAML(t, "allowed_models:", "default_max_tokens: -1\n    allowed_models:"),
			[]string{"tenants[0].default_max_tokens"}},
		{"rpm burst without rpm", baseYAML(t, "allowed_models:", "rpm_burst: 5\n    allowed_models:"),
			[]string{"tenants[0].rpm_burst: rpm_burst set without rpm"}},
		{"tpm burst without tpm", baseYAML(t, "allowed_models:", "tpm_burst: 5\n    allowed_models:"),
			[]string{"tenants[0].tpm_burst: tpm_burst set without tpm"}},
		{"rpm burst window over 24h", baseYAML(t, "allowed_models:", "rpm: 1\n    rpm_burst: 1441\n    allowed_models:"),
			[]string{"tenants[0].rpm_burst: window exceeds 24h"}},
		{"tpm burst window over 24h", baseYAML(t, "allowed_models:", "tpm: 1\n    tpm_burst: 1441\n    allowed_models:"),
			[]string{"tenants[0].tpm_burst: window exceeds 24h"}},
		{"negative max inflight", baseYAML(t, "providers:", "  max_inflight: -1\nproviders:"),
			[]string{"server.max_inflight"}},
		{"unknown policy", baseYAML(t, "tenants:", "routing:\n  policy: random\ntenants:"),
			[]string{"routing.policy"}},
		{"rules policy without default model", baseYAML(t, "tenants:", "routing:\n  policy: rules\n  rules:\n    - name: r\n      model: small\n      when: {min_prompt_bytes: 1}\ntenants:"),
			[]string{"routing.default_model"}},
		{"rules policy without rules", baseYAML(t, "tenants:", rules("")),
			[]string{"routing.rules: at least one rule"}},
		{"unknown default model", baseYAML(t, "tenants:", "routing:\n  policy: cheapest\n  default_model: ghost\ntenants:"),
			[]string{`routing.default_model: unknown model "ghost"`}},
		{"rule with empty when", baseYAML(t, "tenants:", rules("  rules:\n    - name: r\n      model: small\n      when: {}\n")),
			[]string{"routing.rules[0].when"}},
		{"rule unknown model", baseYAML(t, "tenants:", rules("  rules:\n    - name: r\n      model: ghost\n      when: {min_prompt_bytes: 1}\n")),
			[]string{`routing.rules[0].model: unknown model "ghost"`}},
		{"rule without name", baseYAML(t, "tenants:", rules("  rules:\n    - model: small\n      when: {min_prompt_bytes: 1}\n")),
			[]string{"routing.rules[0].name"}},
		{"duplicate rule name", baseYAML(t, "tenants:", rules("  rules:\n    - {name: r, model: small, when: {min_prompt_bytes: 1}}\n    - {name: r, model: small, when: {min_prompt_bytes: 2}}\n")),
			[]string{"routing.rules[1].name: duplicate"}},
		{"rule unknown requires", baseYAML(t, "tenants:", rules("  rules:\n    - {name: r, model: small, when: {requires: [telepathy]}}\n")),
			[]string{"routing.rules[0].when.requires[0]"}},
		{"rule empty contains_any entry", baseYAML(t, "tenants:", rules("  rules:\n    - {name: r, model: small, when: {contains_any: [\"\"]}}\n")),
			[]string{"routing.rules[0].when.contains_any[0]"}},
		{"rule min above max", baseYAML(t, "tenants:", rules("  rules:\n    - {name: r, model: small, when: {min_prompt_bytes: 10, max_prompt_bytes: 5}}\n")),
			[]string{"routing.rules[0].when.min_prompt_bytes"}},
		{"rule negative bytes", baseYAML(t, "tenants:", rules("  rules:\n    - {name: r, model: small, when: {min_prompt_bytes: -1}}\n")),
			[]string{"routing.rules[0].when.min_prompt_bytes"}},
		{"unknown quota backend", baseYAML(t, "tenants:", "quota:\n  backend: etcd\ntenants:"),
			[]string{"quota.backend"}},
		{"redis without env", baseYAML(t, "tenants:", "quota:\n  backend: redis\ntenants:"),
			[]string{"quota.redis_url_env"}},
		{"max attempts too high", baseYAML(t, "tenants:", "reliability:\n  max_attempts: 6\ntenants:"),
			[]string{"reliability.max_attempts"}},
		{"max attempts negative", baseYAML(t, "tenants:", "reliability:\n  max_attempts: -1\ntenants:"),
			[]string{"reliability.max_attempts"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := Parse([]byte(tc.yaml))
			if err == nil {
				t.Fatalf("Parse succeeded, cfg = %+v", cfg)
			}
			for _, w := range tc.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not contain %q", err, w)
				}
			}
		})
	}
}

func TestParseAcceptsM2Sections(t *testing.T) {
	y := baseYAML(t, "tenants:", `routing:
  policy: rules
  default_model: small
  rules:
    - name: long
      model: small
      when: {min_prompt_bytes: 100, max_prompt_bytes: 1000, contains_any: [code], requires: [tools]}
quota:
  backend: redis
  redis_url_env: REDIS_URL
reliability:
  max_attempts: 5
tenants:`)
	y = strings.Replace(y, "    deployments:", "    capabilities: [streaming, tools]\n    deployments:", 1)
	cfg, err := Parse([]byte(y))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Routing.Policy != "rules" || len(cfg.Routing.Rules) != 1 || cfg.Quota.Backend != "redis" || cfg.Reliability.MaxAttempts != 5 {
		t.Errorf("sections = %+v %+v %+v", cfg.Routing, cfg.Quota, cfg.Reliability)
	}
}

func TestTenantLimitDefaults(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t, "allowed_models:", "rpm: 600\n    tpm: 1000\n    allowed_models:")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	tn := cfg.Tenants[0]
	if tn.RPMBurst != 100 || tn.TPMBurst != 1000 || tn.DefaultMaxTokens != 4096 {
		t.Errorf("limits = %+v", tn)
	}

	cfg, err = Parse([]byte(baseYAML(t, "allowed_models:", "rpm: 3\n    allowed_models:")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got := cfg.Tenants[0].RPMBurst; got != 1 {
		t.Errorf("rpm 3 burst = %d, want 1", got)
	}

	cfg, err = Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if tn := cfg.Tenants[0]; tn.RPM != 0 || tn.RPMBurst != 0 || tn.TPM != 0 || tn.TPMBurst != 0 || tn.MaxConcurrency != 0 {
		t.Errorf("unlimited tenant got limits: %+v", tn)
	}
}

func TestExampleConfigM2(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "examples", "config.local.yaml"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Routing.Policy != "cheapest" {
		t.Errorf("routing.policy = %q", cfg.Routing.Policy)
	}
	large, _ := cfg.Model("mock-large")
	if len(large.Deployments) != 2 || large.Deployments[1].Name != "mock-large-b" || large.Deployments[1].UpstreamModel != "mock-large-b" {
		t.Errorf("mock-large deployments = %+v", large.Deployments)
	}
	if large.Rank != 3 || large.InputPricePerMTok != "2.50" || large.OutputPricePerMTok != "10.00" ||
		large.ContextWindow != 128000 || large.MaxOutputTokens != 16384 || !large.HasCapability("vision") {
		t.Errorf("mock-large = %+v", large)
	}
	small, _ := cfg.Model("mock-small")
	if small.Rank != 1 || small.InputPricePerMTok != "0.15" || small.OutputPricePerMTok != "0.60" || small.ContextWindow != 16384 {
		t.Errorf("mock-small = %+v", small)
	}
	qwen, _ := cfg.Model("qwen3-1.7b")
	if qwen.Rank != 2 || qwen.InputPricePerMTok != "0" || qwen.ContextWindow != 32768 || qwen.HasCapability("tools") {
		t.Errorf("qwen3-1.7b = %+v", qwen)
	}
	tn := cfg.Tenants[0]
	if tn.RPM != 600 || tn.TPM != 200000 || tn.MaxConcurrency != 32 || tn.RPMBurst != 100 {
		t.Errorf("demo tenant = %+v", tn)
	}
}

func TestHasCapabilityNilDefaultsToStreaming(t *testing.T) {
	var m ModelConfig // Capabilities is nil, as in a hand-built config
	if !m.HasCapability(CapStreaming) {
		t.Error("nil Capabilities should have streaming")
	}
	for _, c := range []string{CapTools, CapJSONObject, CapJSONSchema, CapVision} {
		if m.HasCapability(c) {
			t.Errorf("nil Capabilities should not have %s", c)
		}
	}
	m.Capabilities = []string{}
	if m.HasCapability(CapStreaming) {
		t.Error("explicit empty Capabilities should have none")
	}
	m.Capabilities = []string{CapTools}
	if m.HasCapability(CapStreaming) || !m.HasCapability(CapTools) {
		t.Error("explicit list should be used verbatim")
	}
}

func TestParseAcceptsBurstWindowAt24h(t *testing.T) {
	// rpm 1 gives a 60 s emission interval, so a burst of 1440 is exactly 24 h.
	y := baseYAML(t, "allowed_models:", "rpm: 1\n    rpm_burst: 1440\n    tpm: 1\n    tpm_burst: 1440\n    allowed_models:")
	if _, err := Parse([]byte(y)); err != nil {
		t.Fatalf("burst window of exactly 24h rejected: %v", err)
	}
}

func TestParseBurstWindowIsExactAtHighRates(t *testing.T) {
	// 40M tokens per minute is 1.5 us per token (not a floored 1 us), so a
	// day holds 57.6e9 tokens of burst and one more is over 24 h.
	at := baseYAML(t, "allowed_models:", "tpm: 40000000\n    tpm_burst: 57600000000\n    allowed_models:")
	if _, err := Parse([]byte(at)); err != nil {
		t.Fatalf("burst window of exactly 24h at 40M/min rejected: %v", err)
	}
	over := baseYAML(t, "allowed_models:", "tpm: 40000000\n    tpm_burst: 57600000001\n    allowed_models:")
	_, err := Parse([]byte(over))
	if err == nil || !strings.Contains(err.Error(), "tenants[0].tpm_burst: window exceeds 24h") {
		t.Fatalf("burst window just over 24h at 40M/min: want a window error, got %v", err)
	}
}

func tenantYAML(t *testing.T, extra string) string {
	t.Helper()
	return baseYAML(t, `allowed_models: ["small"]`, "allowed_models: [\"small\"]\n"+extra)
}

func TestTenantMinQualityAndOnInfeasible(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if tn := cfg.Tenants[0]; tn.MinQuality != nil || tn.OnInfeasible != "" {
		t.Errorf("absent keys: min_quality %v on_infeasible %q, want unset", tn.MinQuality, tn.OnInfeasible)
	}

	cfg, err = Parse([]byte(tenantYAML(t, "    min_quality: 0.7\n    on_infeasible: reject")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if tn := cfg.Tenants[0]; tn.MinQuality == nil || *tn.MinQuality != 0.7 || tn.OnInfeasible != OnInfeasibleReject {
		t.Errorf("min_quality %v on_infeasible %q, want 0.7 and reject", tn.MinQuality, tn.OnInfeasible)
	}

	// An explicit zero is a floor of zero, not "unset".
	cfg, err = Parse([]byte(tenantYAML(t, "    min_quality: 0\n    on_infeasible: best_effort")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if tn := cfg.Tenants[0]; tn.MinQuality == nil || *tn.MinQuality != 0 || tn.OnInfeasible != OnInfeasibleBestEffort {
		t.Errorf("min_quality %v on_infeasible %q, want an explicit 0 and best_effort", tn.MinQuality, tn.OnInfeasible)
	}

	for name, tc := range map[string]struct{ extra, want string }{
		"above one":          {"    min_quality: 1.5", "tenants[0].min_quality"},
		"negative":           {"    min_quality: -0.1", "tenants[0].min_quality"},
		"not a number":       {"    min_quality: .nan", "tenants[0].min_quality"},
		"unknown infeasible": {"    on_infeasible: ignore", "tenants[0].on_infeasible"},
	} {
		if _, err := Parse([]byte(tenantYAML(t, tc.extra))); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one naming %s", name, err, tc.want)
		}
	}
}
