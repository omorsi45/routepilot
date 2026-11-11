package factory

import (
	"net/http"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/provider/anthropic"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

func noEnv(string) string { return "" }

func TestBuildProvidersMissingEnv(t *testing.T) {
	cfgs := []config.ProviderConfig{
		{Name: "cloud", Type: config.ProviderAnthropic, BaseURL: "https://example.invalid", APIKeyEnv: "RP_TEST_KEY"},
	}

	_, err := Build(cfgs, noEnv, nil)

	if err == nil {
		t.Fatal("Build() error = nil, want an error for the empty environment variable")
	}
	for _, want := range []string{"RP_TEST_KEY", "cloud", "empty"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestBuildProvidersTypes(t *testing.T) {
	cfgs := []config.ProviderConfig{
		{Name: "local", Type: config.ProviderOpenAICompat, BaseURL: "http://127.0.0.1:9090/v1"},
		{Name: "cloud", Type: config.ProviderAnthropic, BaseURL: "https://example.invalid", APIKeyEnv: "RP_TEST_KEY"},
	}
	getenv := func(k string) string {
		if k == "RP_TEST_KEY" {
			return "fake-key-for-test"
		}
		return ""
	}

	got, err := Build(cfgs, getenv, http.DefaultTransport)
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("len(providers) = %d, want 2", len(got))
	}
	if _, ok := got["local"].(*openaicompat.Client); !ok {
		t.Errorf("providers[local] = %T, want *openaicompat.Client", got["local"])
	}
	if _, ok := got["cloud"].(*anthropic.Client); !ok {
		t.Errorf("providers[cloud] = %T, want *anthropic.Client", got["cloud"])
	}
}

func TestBuildProvidersUnknownType(t *testing.T) {
	cfgs := []config.ProviderConfig{{Name: "odd", Type: "gemini", BaseURL: "http://127.0.0.1:1"}}

	_, err := Build(cfgs, noEnv, nil)

	if err == nil || !strings.Contains(err.Error(), "odd") {
		t.Fatalf("Build() error = %v, want an error naming provider odd", err)
	}
}

func TestBuildProvidersNilTransportBuildsOneEach(t *testing.T) {
	cfgs := []config.ProviderConfig{
		{Name: "a", Type: config.ProviderOpenAICompat, BaseURL: "http://127.0.0.1:1/v1"},
		{Name: "b", Type: config.ProviderOpenAICompat, BaseURL: "http://127.0.0.1:2/v1"},
	}

	got, err := Build(cfgs, noEnv, nil)

	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(providers) = %d, want 2", len(got))
	}
}
