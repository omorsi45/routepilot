package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const (
	testKey = "test-key-rp-load"
	okBody  = `{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`
)

func env(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

// summary is the part of the JSON result these tests read.
type summary struct {
	Stream       bool           `json:"stream"`
	RPSTarget    float64        `json:"rps_target"`
	Scheduled    int            `json:"scheduled"`
	Completed    int            `json:"completed"`
	Errors       int            `json:"errors"`
	Dropped      int            `json:"dropped"`
	ErrorsByKind map[string]int `json:"errors_by_kind"`
	Throughput   float64        `json:"throughput_rps"`
	Latency      *struct {
		Count int64   `json:"count"`
		P50   float64 `json:"p50"`
		P90   float64 `json:"p90"`
		P95   float64 `json:"p95"`
		P99   float64 `json:"p99"`
		P999  float64 `json:"p99.9"`
		Max   float64 `json:"max"`
	} `json:"latency_ms"`
	TTFT *struct {
		P50 float64 `json:"p50"`
	} `json:"ttft_ms"`
	LaunchLag *struct {
		P99 float64 `json:"p99"`
	} `json:"launch_lag_ms"`
}

func runCLI(t *testing.T, getenv func(string) string, args ...string) (code int, out summary, stdout, stderr string) {
	t.Helper()
	var so, se bytes.Buffer
	code = run(args, &so, &se, getenv)
	if code == 0 {
		if err := json.Unmarshal(so.Bytes(), &out); err != nil {
			t.Fatalf("stdout is not one JSON object: %v\n%s", err, so.String())
		}
	}
	return code, out, so.String(), se.String()
}

func TestRunPrintsSummaryAndNeverTheKey(t *testing.T) {
	var gotAuth atomic.Value
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth.Store(r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, okBody)
	}))
	defer srv.Close()

	code, out, stdout, stderr := runCLI(t, env(map[string]string{"MY_KEY": testKey}),
		"--url", srv.URL+"/v1/chat/completions", "--api-key-env", "MY_KEY", "--model", "m",
		"--rps", "100", "--duration", "400ms")

	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if got, _ := gotAuth.Load().(string); got != "Bearer "+testKey {
		t.Errorf("Authorization = %q, want the key read from MY_KEY", got)
	}
	if strings.Contains(stdout, testKey) || strings.Contains(stderr, testKey) {
		t.Error("the API key must never be printed")
	}
	if out.Scheduled == 0 || out.Completed != out.Scheduled || out.Errors != 0 || out.Dropped != 0 {
		t.Fatalf("scheduled=%d completed=%d errors=%d dropped=%d", out.Scheduled, out.Completed, out.Errors, out.Dropped)
	}
	if out.RPSTarget != 100 || out.Stream {
		t.Errorf("rps_target=%v stream=%v, want 100/false", out.RPSTarget, out.Stream)
	}
	l := out.Latency
	if l == nil || l.Count != int64(out.Completed) {
		t.Fatalf("latency_ms = %+v, want %d samples", l, out.Completed)
	}
	if !(l.P50 > 0 && l.P50 <= l.P90 && l.P90 <= l.P95 && l.P95 <= l.P99 && l.P99 <= l.P999 && l.P999 <= l.Max) {
		t.Errorf("latency percentiles are not ordered: %+v", *l)
	}
	if out.TTFT != nil {
		t.Errorf("ttft_ms = %+v, want null for a non-streaming run", out.TTFT)
	}
	if out.LaunchLag == nil {
		t.Error("launch_lag_ms is missing")
	}
	if out.Throughput <= 0 {
		t.Errorf("throughput_rps = %v, want > 0", out.Throughput)
	}
	if !strings.Contains(stderr, "started") {
		t.Errorf("stderr = %q, want a started marker so callers can align a scrape with the window", stderr)
	}
}

func TestRunDefaultKeyVariable(t *testing.T) {
	var gotAuth atomic.Value
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth.Store(r.Header.Get("Authorization"))
		_, _ = io.WriteString(w, okBody)
	}))
	defer srv.Close()

	code, _, _, stderr := runCLI(t, env(map[string]string{"RP_LOAD_API_KEY": testKey}),
		"--url", srv.URL, "--model", "m", "--rps", "50", "--duration", "200ms")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if got, _ := gotAuth.Load().(string); got != "Bearer "+testKey {
		t.Errorf("Authorization = %q, want the key from RP_LOAD_API_KEY", got)
	}
}

func TestRunStreamReportsTTFT(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"tok"}}]}`+"\n\n"+"data: [DONE]\n\n")
	}))
	defer srv.Close()

	code, out, _, stderr := runCLI(t, env(nil),
		"--url", srv.URL, "--model", "m", "--stream", "--rps", "50", "--duration", "300ms")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if !out.Stream || out.TTFT == nil || out.TTFT.P50 <= 0 || out.Completed == 0 {
		t.Errorf("stream=%v ttft_ms=%+v completed=%d, want a TTFT summary", out.Stream, out.TTFT, out.Completed)
	}
}

func TestRunReportsErrorsWithoutLatency(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	code, out, _, stderr := runCLI(t, env(nil),
		"--url", srv.URL, "--model", "m", "--rps", "50", "--duration", "300ms")
	if code != 0 {
		t.Fatalf("exit code = %d: errors are data, not a failure; stderr: %s", code, stderr)
	}
	if out.Errors == 0 || out.ErrorsByKind["http_500"] != out.Errors || out.Completed != 0 {
		t.Errorf("errors=%d kinds=%v completed=%d", out.Errors, out.ErrorsByKind, out.Completed)
	}
	if out.Latency != nil {
		t.Errorf("latency_ms = %+v, want null when nothing completed", out.Latency)
	}
}

func TestRunUsageErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"no url", []string{"--model", "m", "--rps", "10"}},
		{"no model", []string{"--url", "http://127.0.0.1:1", "--rps", "10"}},
		{"no rps", []string{"--url", "http://127.0.0.1:1", "--model", "m"}},
		{"negative rps", []string{"--url", "http://127.0.0.1:1", "--model", "m", "--rps", "-5"}},
		{"unknown flag", []string{"--bogus"}},
		{"stray argument", []string{"--url", "http://127.0.0.1:1", "--model", "m", "--rps", "10", "extra"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var so, se bytes.Buffer
			code := run(tt.args, &so, &se, env(map[string]string{"RP_LOAD_API_KEY": testKey}))
			if code != 2 {
				t.Errorf("exit code = %d, want 2; stderr: %s", code, se.String())
			}
			if so.Len() != 0 {
				t.Errorf("stdout = %q, want nothing on a usage error", so.String())
			}
			if strings.Contains(se.String(), testKey) {
				t.Error("the API key must never be printed")
			}
		})
	}
}

func TestClockProbePrintsTheStepAndTheZeroFraction(t *testing.T) {
	var so, se bytes.Buffer
	code := run([]string{"--clock-probe"}, &so, &se, env(nil))
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr: %s", code, se.String())
	}
	if se.Len() != 0 {
		t.Errorf("stderr = %q, want nothing: the probe sends no request and has no start marker", se.String())
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(so.Bytes(), &fields); err != nil {
		t.Fatalf("stdout is not one JSON object: %v\n%s", err, so.String())
	}
	if len(fields) != 2 {
		t.Errorf("keys = %v, want exactly clock_step_ns and zero_fraction", fields)
	}
	var out struct {
		StepNS       int64   `json:"clock_step_ns"`
		ZeroFraction float64 `json:"zero_fraction"`
	}
	if err := json.Unmarshal(so.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v\n%s", err, so.String())
	}
	if out.StepNS <= 0 {
		t.Errorf("clock_step_ns = %d, want > 0", out.StepNS)
	}
	if out.ZeroFraction < 0 || out.ZeroFraction > 1 {
		t.Errorf("zero_fraction = %v, want a value in [0, 1]", out.ZeroFraction)
	}
}

func TestRunHelpExitsZero(t *testing.T) {
	var so, se bytes.Buffer
	if code := run([]string{"--help"}, &so, &se, env(nil)); code != 0 {
		t.Errorf("exit code = %d, want 0", code)
	}
}
