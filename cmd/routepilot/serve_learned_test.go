package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/policy"
)

const (
	serveKey    = "test-key-serve-learned"
	serveFloor0 = "mode: min_cost\nlambda: 0\ndefault_floor: 0\nepsilon: 0\nood_action: conservative\non_infeasible: best_effort\n"
	serveFloor4 = "mode: min_cost\nlambda: 0\ndefault_floor: 0.4\nepsilon: 0\nood_action: conservative\non_infeasible: best_effort\n"
	// tinyPrompt is an invented prompt the committed tiny artifact scores in distribution.
	// It sits inside a JSON string, so the backslashes are doubled.
	tinyPrompt = `Solve \\frac{1}{2} + \\frac{1}{3}`
)

// addrRE finds the listen address in the JSON log line serve writes at startup.
var addrRE = regexp.MustCompile(`"addr":"([^"]+)"`)

// serveBump hands out strictly increasing modification times, so every rewrite is a
// visible change however coarse the file system's timestamps are.
var serveBump atomic.Int64

func writeFileAtomic(t *testing.T, path, content string) {
	t.Helper()
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	mod := time.Now().Add(time.Duration(serveBump.Add(1)) * time.Second)
	if err := os.Chtimes(tmp, mod, mod); err != nil {
		t.Fatal(err)
	}
	var err error
	for range 100 { // a reader may hold the target open for a moment on Windows
		if err = os.Rename(tmp, path); err == nil {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal(err)
}

// serveFixture is a config directory holding a learned-policy gateway config, a copy of
// the tiny artifact and a stable policy configuration.
type serveFixture struct {
	dir      string
	cfgPath  string
	version  string // artifact version
	weights  string
	upstream *httptest.Server
}

// profilesYAML holds invented latency priors for the two tiny models.
const profilesYAML = `profiles:
  - model: tiny-weak
    ttft_median_ms: 100
    ttft_sigma: 0.5
    tpot_median_ms: 10
    tpot_sigma: 0.3
    source: invented test value
    retrieved: 2026-10-28
  - model: tiny-strong
    ttft_median_ms: 200
    ttft_sigma: 0.5
    tpot_median_ms: 20
    tpot_sigma: 0.3
    source: invented test value
    retrieved: 2026-10-28
`

func newServeFixture(t *testing.T, serverExtra string) *serveFixture {
	t.Helper()
	srcRoot := filepath.Join("..", "..", "testdata", "policy", "tiny")
	entries, err := os.ReadDir(srcRoot)
	if err != nil || len(entries) != 1 {
		t.Fatalf("want one artifact under %s, got %v (%v)", srcRoot, entries, err)
	}
	f := &serveFixture{dir: t.TempDir(), version: entries[0].Name()}
	artDir := filepath.Join(f.dir, "policies", "tiny", f.version)
	if err := os.MkdirAll(artDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		b, err := os.ReadFile(filepath.Join(srcRoot, f.version, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(artDir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	f.weights = filepath.Join(artDir, "weights.bin")
	writeFileAtomic(t, filepath.Join(f.dir, "stable.yaml"), serveFloor0)
	writeFileAtomic(t, filepath.Join(f.dir, "profiles.yaml"), profilesYAML)

	f.upstream = httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(f.upstream.Close)
	f.cfgPath = filepath.Join(f.dir, "routepilot.yaml")
	f.writeConfig(t, 1000, serverExtra)
	return f
}

// writeConfig rewrites the config file. defaultTimeoutMs is a non-policy setting a test can
// change to see that a reload ignores it.
func (f *serveFixture) writeConfig(t *testing.T, defaultTimeoutMs int, serverExtra string) {
	t.Helper()
	y := fmt.Sprintf(`server:
  listen: 127.0.0.1:0
  shutdown_grace_ms: 1000
  default_timeout_ms: %d
%sproviders:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: tiny-weak
    rank: 1
    input_price_per_mtok: "0.100000"
    output_price_per_mtok: "0.400000"
    deployments:
      - name: weak-dep
        provider: mockp
        upstream_model: weak-up
  - id: tiny-strong
    rank: 2
    input_price_per_mtok: "1.000000"
    output_price_per_mtok: "4.000000"
    deployments:
      - name: strong-dep
        provider: mockp
        upstream_model: strong-up
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
routing:
  policy: learned
policies:
  root: policies
  stable:
    name: tiny
    version: %s
    config: stable.yaml
`, defaultTimeoutMs, serverExtra, f.upstream.URL+"/v1", sha256Hex(serveKey), f.version)
	writeFileAtomic(t, f.cfgPath, y)
}

// slotVersion is the version a stable slot with this policy configuration reports.
func (f *serveFixture) slotVersion(t *testing.T, policyYAML string) string {
	t.Helper()
	art, err := policy.Load(filepath.Join(f.dir, "policies", "tiny", f.version))
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "p.yaml")
	if err := os.WriteFile(p, []byte(policyYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := policy.LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	return art.Version() + "+" + c.Hash8()
}

// running is a serve process under test: its address, its log and how to stop it.
type running struct {
	addr   string
	stderr *syncBuffer
	exit   chan int
	cancel context.CancelFunc
}

// inject replaces the signal sources: the returned channel delivers reload signals, and the
// policy poll is slowed to an hour, so only a signal can cause a reload during the test.
func inject(t *testing.T) (cancel context.CancelFunc, hup chan os.Signal, registrations *atomic.Int32) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	hup = make(chan os.Signal, 4)
	registrations = new(atomic.Int32)
	origNotify, origReload, origPoll := notifyContext, notifyReload, policyPollEvery
	notifyContext = func(context.Context, ...os.Signal) (context.Context, context.CancelFunc) {
		return ctx, func() {}
	}
	notifyReload = func() (<-chan os.Signal, func()) {
		registrations.Add(1)
		return hup, func() {}
	}
	policyPollEvery = time.Hour
	t.Cleanup(func() {
		cancel()
		notifyContext, notifyReload, policyPollEvery = origNotify, origReload, origPoll
	})
	return cancel, hup, registrations
}

func startServe(t *testing.T, cfgPath string, cancel context.CancelFunc) *running {
	t.Helper()
	r := &running{stderr: &syncBuffer{}, exit: make(chan int, 1), cancel: cancel}
	go func() { r.exit <- runServe([]string{"-config", cfgPath}, io.Discard, r.stderr) }()
	for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if m := addrRE.FindStringSubmatch(r.stderr.String()); m != nil {
			r.addr = m[1]
			return r
		}
		select {
		case code := <-r.exit:
			t.Fatalf("serve exited with %d before listening; stderr: %s", code, r.stderr.String())
		default:
		}
	}
	t.Fatalf("server did not report its address; stderr: %s", r.stderr.String())
	return nil
}

func (r *running) stop(t *testing.T) int {
	t.Helper()
	r.cancel()
	select {
	case code := <-r.exit:
		return code
	case <-time.After(5 * time.Second):
		t.Fatal("serve did not return after shutdown")
		return -1
	}
}

// chat sends an auto request and returns the status and the response headers.
func (r *running) chat(t *testing.T, headers map[string]string) (int, http.Header) {
	t.Helper()
	body := fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":"%s"}]}`, tinyPrompt)
	req, err := http.NewRequest(http.MethodPost, "http://"+r.addr+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+serveKey)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode, resp.Header
}

func (r *running) policyVersion(t *testing.T) string {
	t.Helper()
	status, h := r.chat(t, nil)
	if status != http.StatusOK {
		t.Fatalf("chat status %d; stderr: %s", status, r.stderr.String())
	}
	return h.Get("x-routepilot-policy-version")
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestServeSIGHUPReloads(t *testing.T) {
	f := newServeFixture(t, "")
	cancel, hup, registrations := inject(t)
	r := startServe(t, f.cfgPath, cancel)

	if got := registrations.Load(); got != 1 {
		t.Fatalf("serve registered for the reload signal %d times, want 1", got)
	}
	v1 := f.slotVersion(t, serveFloor0)
	if got := r.policyVersion(t); got != v1 {
		t.Fatalf("policy version %q, want %q", got, v1)
	}

	// A change on disk is not picked up without a signal (the poll is an hour away).
	writeFileAtomic(t, filepath.Join(f.dir, "stable.yaml"), serveFloor4)
	v2 := f.slotVersion(t, serveFloor4)
	time.Sleep(100 * time.Millisecond)
	if got := r.policyVersion(t); got != v1 {
		t.Fatalf("policy version %q changed before any signal, want %q", got, v1)
	}

	hup <- syscall.SIGHUP
	waitFor(t, "SIGHUP to swap the policy", func() bool { return r.policyVersion(t) == v2 })

	// A corrupt artifact on SIGHUP keeps the old policy and says so in the log.
	good, err := os.ReadFile(f.weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.weights, good[:len(good)/2], 0o644); err != nil {
		t.Fatal(err)
	}
	hup <- syscall.SIGHUP
	waitFor(t, "the failed reload to be logged", func() bool {
		return strings.Contains(r.stderr.String(), "policy reload failed")
	})
	if got := r.policyVersion(t); got != v2 {
		t.Errorf("policy version %q after a corrupt reload, want the old %q", got, v2)
	}
	if strings.Contains(r.stderr.String(), "frac") {
		t.Error("the log carries prompt text")
	}

	if code := r.stop(t); code != 0 {
		t.Errorf("serve exited with %d, want 0; stderr: %s", code, r.stderr.String())
	}
}

func TestServeReloadWarnsAboutIgnoredSections(t *testing.T) {
	f := newServeFixture(t, "")
	cancel, hup, _ := inject(t)
	r := startServe(t, f.cfgPath, cancel)
	if strings.Contains(r.stderr.String(), "ignored") {
		t.Fatalf("a warning before any change: %s", r.stderr.String())
	}

	// Only the policies section is hot: this change needs a restart and is said to be ignored.
	f.writeConfig(t, 2000, "")
	hup <- syscall.SIGHUP
	waitFor(t, "the warning about ignored sections", func() bool {
		return strings.Contains(r.stderr.String(), "ignored")
	})
	line := ""
	for _, l := range strings.Split(r.stderr.String(), "\n") {
		if strings.Contains(l, "ignored") {
			line = l
		}
	}
	if !strings.Contains(line, `"level":"WARN"`) || strings.Count(r.stderr.String(), "ignored") != 1 {
		t.Errorf("want exactly one WARN line about ignored sections, got: %s", r.stderr.String())
	}
	// The reload itself still succeeded.
	if got := r.policyVersion(t); got != f.slotVersion(t, serveFloor0) {
		t.Errorf("policy version %q after the reload", got)
	}
	r.stop(t)
}

func TestServeStaticPolicyDoesNotWatchForSignals(t *testing.T) {
	cancel, _, registrations := inject(t)
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	path := filepath.Join(t.TempDir(), "routepilot.yaml")
	y := fmt.Sprintf(`server:
  listen: 127.0.0.1:0
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: m
    deployments:
      - provider: mockp
        upstream_model: m
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
`, upstream.URL+"/v1", sha256Hex(serveKey))
	if err := os.WriteFile(path, []byte(y), 0o600); err != nil {
		t.Fatal(err)
	}
	r := startServe(t, path, cancel)
	if got := registrations.Load(); got != 0 {
		t.Errorf("a config without the learned policy registered for the reload signal %d times, want 0", got)
	}
	if code := r.stop(t); code != 0 {
		t.Errorf("exit %d", code)
	}
}

func TestServeLearnedStartupFailures(t *testing.T) {
	cases := map[string]struct {
		prepare func(t *testing.T, f *serveFixture)
		want    string
	}{
		"corrupt artifact": {func(t *testing.T, f *serveFixture) {
			if err := os.WriteFile(f.weights, []byte("short"), 0o644); err != nil {
				t.Fatal(err)
			}
		}, "policy"},
		"missing policy config": {func(t *testing.T, f *serveFixture) {
			if err := os.Remove(filepath.Join(f.dir, "stable.yaml")); err != nil {
				t.Fatal(err)
			}
		}, "stable.yaml"},
		"missing latency profiles": {func(t *testing.T, f *serveFixture) {
			f.writeConfig(t, 1000, "  latency_profiles: nope.yaml\n")
		}, "nope.yaml"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			f := newServeFixture(t, "")
			tc.prepare(t, f)
			var stderr syncBuffer
			if code := runServe([]string{"-config", f.cfgPath}, io.Discard, &stderr); code != 1 {
				t.Fatalf("runServe = %d, want 1; stderr: %s", code, stderr.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Errorf("stderr %q does not mention %q", stderr.String(), tc.want)
			}
		})
	}
}

func TestServeLatencyProfilesFeedTheEstimator(t *testing.T) {
	// With priors loaded from server.latency_profiles, a generous end-to-end SLO is
	// feasible. Without them there is no latency estimate for any model, so the policy
	// gives up the SLO and says so.
	slo := map[string]string{"x-routepilot-slo-e2e-ms": "100000000"}

	with := newServeFixture(t, "  latency_profiles: profiles.yaml\n")
	cancel, _, _ := inject(t)
	r := startServe(t, with.cfgPath, cancel)
	status, h := r.chat(t, slo)
	if status != http.StatusOK || h.Get("x-routepilot-relaxed") != "" {
		t.Errorf("with profiles: status %d relaxed %q, want 200 and nothing relaxed; stderr: %s", status, h.Get("x-routepilot-relaxed"), r.stderr.String())
	}
	r.stop(t)

	without := newServeFixture(t, "")
	cancel2, _, _ := inject(t)
	r2 := startServe(t, without.cfgPath, cancel2)
	status, h = r2.chat(t, slo)
	if status != http.StatusOK || h.Get("x-routepilot-relaxed") != "latency" {
		t.Errorf("without profiles: status %d relaxed %q, want 200 and latency relaxed", status, h.Get("x-routepilot-relaxed"))
	}
	r2.stop(t)
}
