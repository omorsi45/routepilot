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

// TestParseFailRateDisconnect (Ruling V1-R40): --fail-rate-disconnect is the probability that a
// stream is cut after its second chunk (a non-streaming request is reset), and it counts
// toward the sum of the rates.
func TestParseFailRateDisconnect(t *testing.T) {
	var stderr bytes.Buffer
	opts, code := parseFlags([]string{"--fail-rate-disconnect", "0.2"}, &stderr)
	if code != -1 {
		t.Fatalf("exit code = %d (stderr %q), want -1 (keep going)", code, stderr.String())
	}
	if got := opts.config.FailRates; len(got) != 1 || got["disconnect"] != 0.2 {
		t.Fatalf("FailRates = %v, want only disconnect at 0.2", got)
	}

	for name, args := range map[string][]string{
		"negative":      {"--fail-rate-disconnect", "-0.1"},
		"above one":     {"--fail-rate-disconnect", "1.5"},
		"sum above one": {"--fail-rate-disconnect", "0.6", "--fail-rate-http500", "0.5"},
		"sum of all":    {"--fail-rate-http500", "0.3", "--fail-rate-http429", "0.3", "--fail-rate-reset", "0.3", "--fail-rate-disconnect", "0.2"},
	} {
		var errBuf bytes.Buffer
		if _, code := parseFlags(args, &errBuf); code != 2 || !strings.Contains(errBuf.String(), "fail rates") {
			t.Errorf("%s: exit code %d, stderr %q, want 2 and a message about the fail rates", name, code, errBuf.String())
		}
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
