package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

var (
	composeConfigFile = filepath.Join("..", "..", "deploy", "config.compose.yaml")
	composeFile       = filepath.Join("..", "..", "deploy", "docker-compose.yml")
	liveArtifactRoot  = filepath.Join("..", "..", "policies", "live-ollama")
)

// liveArtifact returns the version of the one live-ollama policy artifact in the repository
// and the model IDs its manifest lists, which the compose catalog must match.
func liveArtifact(t *testing.T) (version string, models []string) {
	t.Helper()
	entries, err := os.ReadDir(liveArtifactRoot)
	if err != nil || len(entries) != 1 || !entries[0].IsDir() {
		t.Fatalf("want exactly one artifact directory under %s, got %v (%v)", liveArtifactRoot, entries, err)
	}
	version = entries[0].Name()
	raw, err := os.ReadFile(filepath.Join(liveArtifactRoot, version, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var m struct {
		Models []struct {
			ID string `json:"id"`
		} `json:"models"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("manifest is not valid JSON: %v", err)
	}
	for _, md := range m.Models {
		models = append(models, md.ID)
	}
	slices.Sort(models)
	return version, models
}

// TestComposeConfig: the gateway config of the compose stack loads and wires the stack the
// way docker-compose.yml provides it: the live-ollama artifact and its models served by the
// mock, Redis quota and PostgreSQL accounting named by environment variable, the decision log
// on the state volume, traces to the collector, and the demo tenant.
func TestComposeConfig(t *testing.T) {
	cfg, err := Load(composeConfigFile)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	version, artifactModels := liveArtifact(t)

	if got := cfg.Server.Listen; got != "0.0.0.0:8080" {
		t.Errorf("listen = %q, want 0.0.0.0:8080 (inside the container network; the host publish is loopback)", got)
	}
	if cfg.Routing.Policy != PolicyLearned {
		t.Errorf("routing.policy = %q, want learned", cfg.Routing.Policy)
	}
	want := PolicyRef{Name: "live-ollama", Version: version, Config: "/etc/routepilot/live-ollama.yaml"}
	if cfg.Policies.Root != "/policies" || cfg.Policies.Stable != want {
		t.Errorf("policies = root %q stable %+v, want root /policies stable %+v", cfg.Policies.Root, cfg.Policies.Stable, want)
	}
	if got := cfg.Server.LatencyProfiles; got != "/etc/routepilot/live-ollama-latency.yaml" {
		t.Errorf("server.latency_profiles = %q", got)
	}

	var ids []string
	for _, m := range cfg.Models {
		ids = append(ids, m.ID)
		if len(m.Deployments) != 1 || m.Deployments[0].Provider != "mock" {
			t.Errorf("model %q deployments = %+v, want one on the mock provider", m.ID, m.Deployments)
		}
	}
	slices.Sort(ids)
	if !slices.Equal(ids, artifactModels) {
		t.Errorf("catalog models %v differ from the artifact's %v", ids, artifactModels)
	}
	if len(cfg.Providers) != 1 || cfg.Providers[0].Name != "mock" || cfg.Providers[0].BaseURL != "http://rp-mock:9090/v1" {
		t.Errorf("providers = %+v, want the mock at http://rp-mock:9090/v1", cfg.Providers)
	}

	if cfg.Quota.Backend != QuotaRedis || cfg.Quota.RedisURLEnv != "ROUTEPILOT_REDIS_URL" {
		t.Errorf("quota = %+v, want redis via ROUTEPILOT_REDIS_URL", cfg.Quota)
	}
	if cfg.Accounting.Driver != AccountingPostgres || cfg.Accounting.PostgresDSNEnv != "ROUTEPILOT_POSTGRES_DSN" {
		t.Errorf("accounting = %+v, want postgres via ROUTEPILOT_POSTGRES_DSN", cfg.Accounting)
	}
	if got := cfg.DecisionLog.Path; got != "/var/lib/routepilot/decisions.jsonl" {
		t.Errorf("decision_log.path = %q", got)
	}
	if got := cfg.Tracing.OTLPEndpoint; got != "http://otel-collector:4318" {
		t.Errorf("tracing.otlp_endpoint = %q", got)
	}

	if len(cfg.Tenants) != 1 {
		t.Fatalf("tenants = %d, want only the demo tenant", len(cfg.Tenants))
	}
	tn := cfg.Tenants[0]
	if tn.ID != "demo" || tn.APIKeySHA256 != keyHash("demo-key-local") {
		t.Errorf("tenant %q hash %q, want demo with the hash of demo-key-local", tn.ID, tn.APIKeySHA256)
	}
	if tn.RPM != 600 || tn.TPM != 200000 || tn.MaxConcurrency != 32 || tn.BudgetUSD != "1" || tn.BudgetPeriod != "total" {
		t.Errorf("demo tenant = %+v", tn)
	}
	if tn.TPMBurst < tn.DefaultMaxTokens+minInputBound {
		t.Errorf("tpm_burst %d is below default_max_tokens + %d", tn.TPMBurst, minInputBound)
	}
}

type composeService struct {
	Image       string            `yaml:"image"`
	Build       any               `yaml:"build"`
	Entrypoint  []string          `yaml:"entrypoint"`
	Ports       []string          `yaml:"ports"`
	Volumes     []string          `yaml:"volumes"`
	Environment map[string]string `yaml:"environment"`
	DependsOn   map[string]struct {
		Condition string `yaml:"condition"`
	} `yaml:"depends_on"`
}

var imageWithTag = regexp.MustCompile(`^[a-z0-9][a-z0-9./_-]*:[A-Za-z0-9][A-Za-z0-9._-]*$`)

// TestComposeFile: docker-compose.yml pins every image to an exact tag, publishes only on
// loopback, provisions the demo Grafana as a read-only viewer, makes the gateway wait for
// healthy Redis and PostgreSQL, and mounts the policy artifact that config.compose.yaml names.
func TestComposeFile(t *testing.T) {
	raw, err := os.ReadFile(composeFile)
	if err != nil {
		t.Fatalf("read compose file: %v", err)
	}
	var doc struct {
		Services map[string]composeService `yaml:"services"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("compose file: %v", err)
	}

	for _, name := range []string{"routepilot", "rp-mock", "redis", "postgres", "prometheus", "grafana", "otel-collector"} {
		if _, ok := doc.Services[name]; !ok {
			t.Errorf("service %q is missing", name)
		}
	}
	for name, s := range doc.Services {
		if !imageWithTag.MatchString(s.Image) || strings.HasSuffix(s.Image, ":latest") {
			t.Errorf("service %q image %q must carry an exact tag, never latest", name, s.Image)
		}
		for _, p := range s.Ports {
			if !strings.HasPrefix(p, "127.0.0.1:") {
				t.Errorf("service %q publishes %q; every host port must bind 127.0.0.1", name, p)
			}
		}
	}

	gw := doc.Services["routepilot"]
	if gw.Build == nil {
		t.Error("routepilot must build from deploy/Dockerfile")
	}
	if mock := doc.Services["rp-mock"]; len(mock.Entrypoint) != 1 || mock.Entrypoint[0] != "/rp-mock" || mock.Build == nil {
		t.Errorf("rp-mock = %+v, want the same image with entrypoint /rp-mock", mock)
	}
	for _, dep := range []string{"redis", "postgres"} {
		if got := gw.DependsOn[dep].Condition; got != "service_healthy" {
			t.Errorf("routepilot depends_on %s condition %q, want service_healthy", dep, got)
		}
	}

	version, _ := liveArtifact(t)
	artifactMount := "../policies/live-ollama/" + version + ":/policies/live-ollama/" + version + ":ro"
	if !slices.Contains(gw.Volumes, artifactMount) {
		t.Errorf("routepilot volumes %v lack the read-only artifact mount %q", gw.Volumes, artifactMount)
	}
	for _, v := range gw.Volumes {
		if strings.Contains(v, "/policies") && !strings.HasSuffix(v, ":ro") {
			t.Errorf("policy mount %q must be read-only", v)
		}
	}

	graf := doc.Services["grafana"].Environment
	for k, v := range map[string]string{
		"GF_AUTH_ANONYMOUS_ENABLED":  "true",
		"GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
		"GF_AUTH_DISABLE_LOGIN_FORM": "true",
	} {
		if graf[k] != v {
			t.Errorf("grafana %s = %q, want %q", k, graf[k], v)
		}
	}
}
