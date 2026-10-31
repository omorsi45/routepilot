package config

import "testing"

// TestMaxOutputTokensDefaultFitsTheContextWindow: a model with a small context window and no
// max_output_tokens gets the default cap, but never one larger than its window. Small local
// models have 2k to 4k windows, and refusing a config for a value the operator never wrote
// is the wrong answer.
func TestMaxOutputTokensDefaultFitsTheContextWindow(t *testing.T) {
	for _, c := range []struct {
		name   string
		model  string // lines added to the model entry
		window int
		want   int
	}{
		{"small window", "    context_window: 2048\n", 2048, 2048},
		{"window at the default cap", "    context_window: 4096\n", 4096, 4096},
		{"window above the default cap", "    context_window: 16384\n", 16384, 4096},
		{"no window", "", 8192, 4096},
		{"explicit cap below a small window", "    context_window: 2048\n    max_output_tokens: 1000\n", 2048, 1000},
		{"explicit cap above the default", "    context_window: 16384\n    max_output_tokens: 8000\n", 16384, 8000},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg, err := Parse([]byte(baseYAML(t, "    deployments:", c.model+"    deployments:")))
			if err != nil {
				t.Fatalf("Parse() error = %v", err)
			}
			if got := cfg.Models[0].ContextWindow; got != c.window {
				t.Errorf("ContextWindow = %d, want %d", got, c.window)
			}
			if got := cfg.Models[0].MaxOutputTokens; got != c.want {
				t.Errorf("MaxOutputTokens = %d, want %d", got, c.want)
			}
		})
	}
}
