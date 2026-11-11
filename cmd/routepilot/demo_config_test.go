package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

const (
	demoConfigPath = "../../examples/config.demo.yaml"
	demoKey        = "demo-key-local"
	demoTightKey   = "demo-key-tight"
)

// demoClient opens a connection per request and closes it after. A pooled client can dial a
// spare connection that never carries a request, and the server's shutdown waits five seconds
// for such a connection before it treats it as idle.
var demoClient = &http.Client{Transport: &http.Transport{DisableKeepAlives: true}, Timeout: 10 * time.Second}

// startDemoGateway serves examples/config.demo.yaml, the config scripts/demo.sh runs, with the
// mock upstream in process on a free port and the gateway on another. The policy artifact, its
// optimizer configuration and the latency profiles are the real ones, resolved relative to the
// example file as in the script. It returns the gateway's base URL.
func startDemoGateway(t *testing.T) string {
	t.Helper()
	cfg, err := config.Load(demoConfigPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	if len(cfg.Providers) != 1 {
		t.Fatalf("providers = %+v, want only the mock", cfg.Providers)
	}
	cfg.Providers[0].BaseURL = upstream.URL + "/v1"
	cfg.DecisionLog.Path = filepath.Join(t.TempDir(), "decisions.jsonl")

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil))) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("serve: %v", err)
			}
		case <-time.After(10 * time.Second):
			t.Error("serve did not return after shutdown")
		}
	})

	base := "http://" + ln.Addr().String()
	waitFor(t, "the gateway to answer /readyz", func() bool {
		resp, err := demoClient.Get(base + "/readyz")
		if err != nil {
			return false
		}
		resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	})
	return base
}

// demoReply is one response of the demo gateway.
type demoReply struct {
	status int
	header http.Header
	body   string
}

func demoDo(t *testing.T, method, url, key string, headers map[string]string, body string) demoReply {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := demoClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return demoReply{status: resp.StatusCode, header: resp.Header, body: string(b)}
}

func demoChat(t *testing.T, base, key, model string, headers map[string]string) demoReply {
	t.Helper()
	body := `{"model":"` + model + `","messages":[{"role":"user","content":"What is 17 times 23?"}]}`
	return demoDo(t, http.MethodPost, base+"/v1/chat/completions", key, headers, body)
}

// TestDemoConfigTenantsAuthenticate: both demo tenants are stored as the SHA-256 of their fake
// key, and each key opens the gateway while a wrong one does not.
func TestDemoConfigTenantsAuthenticate(t *testing.T) {
	cfg, err := config.Load(demoConfigPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := map[string]string{"demo": sha256Hex(demoKey), "demo-tight": sha256Hex(demoTightKey)}
	if len(cfg.Tenants) != len(want) {
		t.Fatalf("tenants = %d, want %d (demo and demo-tight)", len(cfg.Tenants), len(want))
	}
	for _, tn := range cfg.Tenants {
		if want[tn.ID] == "" || tn.APIKeySHA256 != want[tn.ID] {
			t.Errorf("tenant %q hash %q, want %q", tn.ID, tn.APIKeySHA256, want[tn.ID])
		}
	}
	if got := cfg.Server.Listen; !strings.HasPrefix(got, "127.0.0.1:") {
		t.Errorf("listen = %q, the demo must bind loopback", got)
	}

	base := startDemoGateway(t)
	for _, key := range []string{demoKey, demoTightKey} {
		if r := demoDo(t, http.MethodGet, base+"/v1/models", key, nil, ""); r.status != http.StatusOK {
			t.Errorf("GET /v1/models with %s: status %d, want 200: %s", key, r.status, r.body)
		}
	}
	if r := demoDo(t, http.MethodGet, base+"/v1/models", "demo-key-wrong", nil, ""); r.status != http.StatusUnauthorized {
		t.Errorf("GET /v1/models with a wrong key: status %d, want 401", r.status)
	}
}

// TestDemoScenarios walks the sequence scripts/demo.sh shows, against the real example config:
// an auto request with its routing headers, a quality floor no model clears (relaxed), a pinned
// model, a failover to the second deployment, a tenant that runs out of budget, and the metrics
// that record them.
func TestDemoScenarios(t *testing.T) {
	base := startDemoGateway(t)

	t.Run("auto request carries the routing headers", func(t *testing.T) {
		r := demoChat(t, base, demoKey, "auto", nil)
		if r.status != http.StatusOK {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
		v := r.header.Get("x-routepilot-policy-version")
		if !strings.HasPrefix(v, "21183f8ab3fe+") {
			t.Errorf("x-routepilot-policy-version = %q, want the live-ollama artifact version plus its config hash", v)
		}
		for _, h := range []string{"x-routepilot-model", "x-routepilot-deployment", "x-routepilot-decision-id", "x-routepilot-estimated-cost-usd"} {
			if r.header.Get(h) == "" {
				t.Errorf("header %s is missing", h)
			}
		}
		if m := r.header.Get("x-routepilot-model"); !strings.HasPrefix(m, "qwen2.5-") {
			t.Errorf("x-routepilot-model = %q, want one of the artifact's models", m)
		}
	})

	t.Run("a quality floor no model clears is relaxed", func(t *testing.T) {
		r := demoChat(t, base, demoKey, "auto", map[string]string{"x-routepilot-min-quality": "0.99"})
		if r.status != http.StatusOK {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
		if got := r.header.Get("x-routepilot-relaxed"); !strings.Contains(got, "quality") {
			t.Errorf("x-routepilot-relaxed = %q, want quality", got)
		}
	})

	t.Run("a pinned model bypasses the policy", func(t *testing.T) {
		r := demoChat(t, base, demoKey, "qwen2.5-3b", nil)
		if r.status != http.StatusOK {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
		if got := r.header.Get("x-routepilot-model"); got != "qwen2.5-3b" {
			t.Errorf("x-routepilot-model = %q, want qwen2.5-3b", got)
		}
		if got := r.header.Get("x-routepilot-policy-version"); got != "pinned" {
			t.Errorf("x-routepilot-policy-version = %q, want pinned", got)
		}
	})

	t.Run("a failing first deployment fails over to the second", func(t *testing.T) {
		r := demoChat(t, base, demoKey, "demo-failover", nil)
		if r.status != http.StatusOK {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
		if got := r.header.Get("x-routepilot-deployment"); got != "demo-failover-b" {
			t.Errorf("x-routepilot-deployment = %q, want demo-failover-b", got)
		}
	})

	t.Run("auto never routes to the failover model", func(t *testing.T) {
		// demo-failover is outside the policy artifact, so the learned policy cannot pick it.
		for range 5 {
			r := demoChat(t, base, demoKey, "auto", nil)
			if got := r.header.Get("x-routepilot-model"); r.status != http.StatusOK || got == "demo-failover" {
				t.Fatalf("auto request: status %d model %q", r.status, got)
			}
		}
	})

	t.Run("a tenant with a tiny budget is refused with 429", func(t *testing.T) {
		served, refused := 0, 0
		for range 100 {
			r := demoChat(t, base, demoTightKey, "auto", nil)
			if r.status == http.StatusOK {
				served++
				continue
			}
			var e struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			_ = json.Unmarshal([]byte(r.body), &e)
			if r.status != http.StatusTooManyRequests || e.Error.Code != "insufficient_quota" {
				t.Fatalf("refusal: status %d body %s, want 429 insufficient_quota", r.status, r.body)
			}
			refused++
			break
		}
		if served == 0 || refused == 0 {
			t.Errorf("served %d then refused %d, want some requests served before the budget ran out", served, refused)
		}
		t.Logf("demo-tight was served %d requests before the budget refused one", served)
	})

	t.Run("metrics record the scenarios", func(t *testing.T) {
		r := demoDo(t, http.MethodGet, base+"/metrics", "", nil, "")
		for _, want := range []string{
			"rp_requests_total{", `relaxed="quality"`, `policy_version="pinned"`, "rp_budget_rejections_total 1",
			`rp_upstream_attempts_total{deployment="demo-failover-a",outcome="`,
		} {
			if !strings.Contains(r.body, want) {
				t.Errorf("/metrics lacks %q", want)
			}
		}
	})
}
