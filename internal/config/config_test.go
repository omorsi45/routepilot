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
