package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestParseFailRates: every injectable failure has a flag, and the flags map to the mock's
// failure keys. The 429 flag is the one a failover experiment needs for a rate-limited
// deployment (Retry-After 1 s is the mock's default).
func TestParseFailRates(t *testing.T) {
	var stderr bytes.Buffer
	opts, code := parseFlags([]string{
		"--listen", "127.0.0.1:0",
		"--fail-rate-http500", "0.10",
		"--fail-rate-http429", "0.05",
		"--fail-rate-reset", "0.05",
	}, &stderr)
	if code != -1 {
		t.Fatalf("exit code = %d (stderr %q), want -1 (keep going)", code, stderr.String())
	}
	want := map[string]float64{"http500": 0.10, "http429": 0.05, "reset": 0.05}
	got := opts.config.FailRates
	if len(got) != len(want) {
		t.Fatalf("FailRates = %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("FailRates[%q] = %v, want %v", k, got[k], v)
		}
	}
	if opts.listen != "127.0.0.1:0" {
		t.Errorf("listen = %q", opts.listen)
	}
}

func TestParseNoFailRatesInjectsNothing(t *testing.T) {
	var stderr bytes.Buffer
	opts, code := parseFlags(nil, &stderr)
	if code != -1 {
		t.Fatalf("exit code = %d, want -1", code)
	}
	if len(opts.config.FailRates) != 0 {
		t.Fatalf("FailRates = %v, want none", opts.config.FailRates)
	}
}

func TestParseRejectsBadFailRates(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"negative 429", []string{"--fail-rate-http429", "-0.1"}},
		{"429 above one", []string{"--fail-rate-http429", "1.5"}},
		{"sum above one", []string{"--fail-rate-http500", "0.6", "--fail-rate-http429", "0.3", "--fail-rate-reset", "0.2"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			if _, code := parseFlags(tc.args, &stderr); code != 2 {
				t.Fatalf("exit code = %d, want 2", code)
			}
			if !strings.Contains(stderr.String(), "fail rates") {
				t.Fatalf("stderr = %q, want it to mention the fail rates", stderr.String())
			}
		})
	}
}

func TestParseHelpAndUnknownFlag(t *testing.T) {
	var stderr bytes.Buffer
	if _, code := parseFlags([]string{"--help"}, &stderr); code != 0 {
		t.Fatalf("--help exit code = %d, want 0", code)
	}
	if _, code := parseFlags([]string{"--no-such-flag"}, &bytes.Buffer{}); code != 2 {
		t.Fatalf("unknown flag exit code = %d, want 2", code)
	}
}
