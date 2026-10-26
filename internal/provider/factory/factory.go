// Package factory builds provider adapters from configuration. It is separate
// from package provider because the adapters import provider.
package factory

import (
	"fmt"
	"net/http"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/anthropic"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

// Build creates one provider per config entry, keyed by name. Each provider
// gets its own http.Client. When transport is nil, each client gets a fresh
// provider.NewTransport so connection pools are never shared between
// providers.
//
// A provider with an empty api_key_env is built with no key (local
// providers). One whose variable is set in the config but empty in the
// environment is an error that names the variable, never its value.
func Build(cfgs []config.ProviderConfig, getenv func(string) string, transport http.RoundTripper) (map[string]provider.Provider, error) {
	out := make(map[string]provider.Provider, len(cfgs))
	for _, cfg := range cfgs {
		key := ""
		if cfg.APIKeyEnv != "" {
			key = getenv(cfg.APIKeyEnv)
			if key == "" {
				return nil, fmt.Errorf("provider %s: environment variable %s is empty", cfg.Name, cfg.APIKeyEnv)
			}
		}

		rt := transport
		if rt == nil {
			rt = provider.NewTransport()
		}
		client := &http.Client{Transport: rt}

		switch cfg.Type {
		case config.ProviderOpenAICompat:
			out[cfg.Name] = openaicompat.New(cfg, key, client)
		case config.ProviderAnthropic:
			out[cfg.Name] = anthropic.New(cfg, key, client)
		default:
			return nil, fmt.Errorf("provider %s: unknown type %q", cfg.Name, cfg.Type)
		}
	}
	return out, nil
}
