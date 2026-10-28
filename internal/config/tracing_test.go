package config

import (
	"strings"
	"testing"
)

func TestTracingIsOffByDefault(t *testing.T) {
	cfg, err := Parse([]byte(accountingYAML("")))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if cfg.Tracing.OTLPEndpoint != "" || cfg.Tracing.SampleRatio != nil {
		t.Errorf("Tracing = %+v, want the zero value: an absent section means noop tracing", cfg.Tracing)
	}
	if got := cfg.Tracing.Ratio(); got != 1 {
		t.Errorf("Ratio() of an absent section = %v, want 1", got)
	}
}

func TestTracingParse(t *testing.T) {
	ratio := func(v float64) *float64 { return &v }
	for _, c := range []struct {
		name, yaml string
		endpoint   string
		ratio      float64
		explicit   bool // the parsed config holds the ratio, whether defaulted or set
	}{
		{"endpoint only defaults the ratio to 1", "tracing:\n  otlp_endpoint: http://localhost:4318\n", "http://localhost:4318", 1, true},
		{"explicit ratio", "tracing:\n  otlp_endpoint: https://otel.example.internal:4318/v1/traces\n  sample_ratio: 0.25\n",
			"https://otel.example.internal:4318/v1/traces", 0.25, true},
		{"zero is a real ratio, not an absent one", "tracing:\n  otlp_endpoint: http://localhost:4318\n  sample_ratio: 0\n", "http://localhost:4318", 0, true},
		{"one", "tracing:\n  otlp_endpoint: http://localhost:4318\n  sample_ratio: 1\n", "http://localhost:4318", 1, true},
	} {
		cfg, err := Parse([]byte(accountingYAML(c.yaml)))
		if err != nil {
			t.Errorf("%s: Parse() error = %v", c.name, err)
			continue
		}
		if cfg.Tracing.OTLPEndpoint != c.endpoint {
			t.Errorf("%s: OTLPEndpoint = %q, want %q", c.name, cfg.Tracing.OTLPEndpoint, c.endpoint)
		}
		if c.explicit && (cfg.Tracing.SampleRatio == nil || *cfg.Tracing.SampleRatio != c.ratio) {
			t.Errorf("%s: SampleRatio = %v, want %v", c.name, cfg.Tracing.SampleRatio, ratio(c.ratio))
		}
		if got := cfg.Tracing.Ratio(); got != c.ratio {
			t.Errorf("%s: Ratio() = %v, want %v", c.name, got, c.ratio)
		}
	}
}

func TestTracingRejects(t *testing.T) {
	for _, c := range []struct{ name, yaml, want string }{
		{"ratio above one", "tracing:\n  otlp_endpoint: http://localhost:4318\n  sample_ratio: 1.5\n", "tracing.sample_ratio"},
		{"negative ratio", "tracing:\n  otlp_endpoint: http://localhost:4318\n  sample_ratio: -0.1\n", "tracing.sample_ratio"},
		{"ratio not a number", "tracing:\n  otlp_endpoint: http://localhost:4318\n  sample_ratio: .nan\n", "tracing.sample_ratio"},
		{"ratio without an endpoint", "tracing:\n  sample_ratio: 0.5\n", "tracing.sample_ratio"},
		{"endpoint without a scheme", "tracing:\n  otlp_endpoint: localhost:4318\n", "tracing.otlp_endpoint"},
		{"endpoint with another scheme", "tracing:\n  otlp_endpoint: grpc://localhost:4317\n", "tracing.otlp_endpoint"},
		{"endpoint without a host", "tracing:\n  otlp_endpoint: http:///v1/traces\n", "tracing.otlp_endpoint"},
		{"endpoint with credentials", "tracing:\n  otlp_endpoint: http://user:hunter2@localhost:4318\n", "tracing.otlp_endpoint"},
		{"endpoint with a query", "tracing:\n  otlp_endpoint: http://localhost:4318/v1/traces?x=1\n", "tracing.otlp_endpoint"},
		{"unknown key", "tracing:\n  otlp_endpoint: http://localhost:4318\n  exporter: grpc\n", "exporter"},
	} {
		_, err := Parse([]byte(accountingYAML(c.yaml)))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Parse() error = %v, want it to mention %q", c.name, err, c.want)
			continue
		}
		if strings.Contains(err.Error(), "hunter2") {
			t.Errorf("%s: the error repeats the password of the endpoint: %v", c.name, err)
		}
	}
}
