package gateway

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

const (
	keyA = "test-key-tenant-a"
	keyB = "test-key-tenant-b"
)

func keyHash(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// cfgOpts are the knobs a test turns on the shared config template.
type cfgOpts struct {
	baseURL       string
	maxBodyBytes  int64
	upstreamA     string // upstream_model for model-a
	tenantAMaxMs  int
	serverDefault int
	serverMax     int
	ttftMs        int // provider ttft_timeout_ms; 0 keeps the config default
	idleMs        int // provider idle_timeout_ms; 0 keeps the config default
}

// buildConfig returns a parsed config with two models and two tenants:
// tenant-a may use model-a and model-b, tenant-b only model-a.
func buildConfig(t *testing.T, o cfgOpts) *config.Config {
	t.Helper()
	if o.baseURL == "" {
		o.baseURL = "http://127.0.0.1:1/v1"
	}
	if o.upstreamA == "" {
		o.upstreamA = "up-a"
	}
	var sb strings.Builder
	sb.WriteString("server:\n")
	if o.maxBodyBytes > 0 {
		fmt.Fprintf(&sb, "  max_body_bytes: %d\n", o.maxBodyBytes)
	}
	if o.serverDefault > 0 {
		fmt.Fprintf(&sb, "  default_timeout_ms: %d\n", o.serverDefault)
	}
	if o.serverMax > 0 {
		fmt.Fprintf(&sb, "  max_timeout_ms: %d\n", o.serverMax)
	}
	fmt.Fprintf(&sb, `providers:
  - name: mockp
    type: openaicompat
    base_url: %q
%smodels:
  - id: model-a
    deployments:
      - name: dep-a
        provider: mockp
        upstream_model: %q
  - id: model-b
    deployments:
      - name: dep-b
        provider: mockp
        upstream_model: up-b
tenants:
  - id: tenant-a
    api_key_sha256: %s
    allowed_models: ["model-a", "model-b"]
    max_timeout_ms: %d
  - id: tenant-b
    api_key_sha256: %s
    allowed_models: ["model-a"]
`, o.baseURL, providerTimeouts(o), o.upstreamA, keyHash(keyA), o.tenantAMaxMs, keyHash(keyB))
	cfg, err := config.Parse([]byte(sb.String()))
	if err != nil {
		t.Fatalf("parse test config: %v", err)
	}
	return cfg
}

// providerTimeouts renders the optional provider timeout lines.
func providerTimeouts(o cfgOpts) string {
	var sb strings.Builder
	if o.ttftMs > 0 {
		fmt.Fprintf(&sb, "    ttft_timeout_ms: %d\n", o.ttftMs)
	}
	if o.idleMs > 0 {
		fmt.Fprintf(&sb, "    idle_timeout_ms: %d\n", o.idleMs)
	}
	return sb.String()
}

// recorder captures upstream request bodies seen by a wrapped handler.
type recorder struct {
	mu     sync.Mutex
	bodies [][]byte
	next   http.Handler
}

func (r *recorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	b, _ := io.ReadAll(req.Body)
	req.Body = io.NopCloser(bytes.NewReader(b))
	r.mu.Lock()
	r.bodies = append(r.bodies, b)
	r.mu.Unlock()
	r.next.ServeHTTP(w, req)
}

func (r *recorder) last(t *testing.T) []byte {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.bodies) == 0 {
		t.Fatal("upstream saw no requests")
	}
	return r.bodies[len(r.bodies)-1]
}

// env is a gateway wired to a real mock handler through openaicompat.
type env struct {
	gw       *Server
	mock     *mock.Handler
	upstream *recorder
	logs     *bytes.Buffer
}

func newMockEnv(t *testing.T, o cfgOpts) *env {
	t.Helper()
	m := mock.NewHandler(mock.Config{})
	e := newUpstreamEnv(t, o, m)
	e.mock = m
	return e
}

// newUpstreamEnv is newMockEnv with a caller-supplied upstream handler, for
// tests that need an upstream the mock cannot imitate. env.mock is nil.
func newUpstreamEnv(t *testing.T, o cfgOpts, h http.Handler) *env {
	t.Helper()
	rec := &recorder{next: h}
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	o.baseURL = srv.URL + "/v1"
	cfg := buildConfig(t, o)
	prov := openaicompat.New(cfg.Providers[0], "", srv.Client())
	logs := &bytes.Buffer{}
	gw, err := New(cfg, map[string]provider.Provider{"mockp": prov}, Options{
		Logger: slog.New(slog.NewTextHandler(logs, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return &env{gw: gw, upstream: rec, logs: logs}
}

// fakeProvider implements provider.Provider with a function.
type fakeProvider struct {
	complete func(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error)
	stream   func(ctx context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error)
}

func (f *fakeProvider) Complete(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
	return f.complete(ctx, req)
}

func (f *fakeProvider) Stream(ctx context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error) {
	if f.stream == nil {
		return nil, fmt.Errorf("fake: stream not implemented")
	}
	return f.stream(ctx, req)
}

func newFakeGateway(t *testing.T, o cfgOpts, p provider.Provider) (*Server, *bytes.Buffer) {
	t.Helper()
	logs := &bytes.Buffer{}
	gw, err := New(buildConfig(t, o), map[string]provider.Provider{"mockp": p}, Options{
		Logger: slog.New(slog.NewTextHandler(logs, nil)),
		Now:    func() time.Time { return time.Unix(1700000000, 0) },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return gw, logs
}

func chatBody(model, content string) string {
	b, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": content}},
	})
	return string(b)
}

func do(h http.Handler, method, path, key, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

type errBody struct {
	Error struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    string  `json:"code"`
	} `json:"error"`
}

func decodeErr(t *testing.T, rec *httptest.ResponseRecorder) errBody {
	t.Helper()
	var e errBody
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("response is not an error envelope: %v\nbody: %s", err, rec.Body.String())
	}
	return e
}

func TestNewRejectsMissingProvider(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	if _, err := New(cfg, map[string]provider.Provider{}, Options{}); err == nil {
		t.Fatal("New with a missing provider succeeded, want error")
	} else if !strings.Contains(err.Error(), "mockp") {
		t.Errorf("error %q should name the missing provider", err)
	}
	if _, err := New(nil, nil, Options{}); err == nil {
		t.Fatal("New with nil config succeeded, want error")
	}
}

func TestNewRejectsMissingProviderConfig(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	// A hand-built config can omit the provider entry that a deployment uses.
	cfg.Providers = nil
	provs := map[string]provider.Provider{"mockp": &fakeProvider{}}
	_, err := New(cfg, provs, Options{})
	if err == nil {
		t.Fatal("New with a deployment whose provider has no config entry succeeded, want error")
	}
	if !strings.Contains(err.Error(), "mockp") {
		t.Errorf("error %q should name the provider", err)
	}
}

func TestHealthNoAuth(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	for path, want := range map[string]string{"/healthz": "ok", "/readyz": "ready"} {
		rec := do(e.gw.Handler(), http.MethodGet, path, "", "")
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status %d, want 200", path, rec.Code)
		}
		if got := rec.Body.String(); got != want {
			t.Errorf("%s: body %q, want %q", path, got, want)
		}
		if rec.Header().Get("x-routepilot-request-id") == "" {
			t.Errorf("%s: missing request id header", path)
		}
	}
}

func TestAuthRequired(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=3"})
	h := e.gw.Handler()
	body := chatBody("model-a", "hi")

	for name, key := range map[string]string{"missing": "", "wrong": "test-key-nobody"} {
		rec := do(h, http.MethodPost, "/v1/chat/completions", key, body)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s key: status %d, want 401", name, rec.Code)
			continue
		}
		if got := decodeErr(t, rec).Error.Code; got != "invalid_api_key" {
			t.Errorf("%s key: code %q, want invalid_api_key", name, got)
		}
		if got := rec.Header().Get("WWW-Authenticate"); got != "Bearer" {
			t.Errorf("%s key: WWW-Authenticate %q, want Bearer", name, got)
		}
		if rec.Header().Get("x-routepilot-request-id") == "" {
			t.Errorf("%s key: 401 is missing the request id header", name)
		}
	}

	// Authorization with a non-Bearer scheme is also rejected.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Basic "+keyA)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("basic scheme: status %d, want 401", rec.Code)
	}

	rec = do(h, http.MethodPost, "/v1/chat/completions", keyA, body)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("correct key got 401: %s", rec.Body.String())
	}
}

func TestBodyTooLarge(t *testing.T) {
	e := newMockEnv(t, cfgOpts{maxBodyBytes: 1024})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", strings.Repeat("x", 2048)))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d, want 413: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "request_too_large" {
		t.Errorf("code %q, want request_too_large", got)
	}
}

func TestInvalidBody(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, `{"model":`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if got := decodeErr(t, rec).Error.Type; got != "invalid_request_error" {
		t.Errorf("type %q, want invalid_request_error", got)
	}
}

func TestModelErrors(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	h := e.gw.Handler()

	tests := []struct {
		name     string
		key      string
		model    string
		status   int
		typ      string
		code     string
		param    string
		wantMsg  string
		hasParam bool
	}{
		{"auto", keyA, "auto", 400, "invalid_request_error", "invalid_request", "model", "routing policy not configured", true},
		{"unknown", keyA, "no-such-model", 404, "invalid_request_error", "model_not_found", "", "", false},
		{"disallowed", keyB, "model-b", 403, "permission_error", "model_not_allowed", "", "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(h, http.MethodPost, "/v1/chat/completions", tc.key, chatBody(tc.model, "hi"))
			if rec.Code != tc.status {
				t.Fatalf("status %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
			got := decodeErr(t, rec)
			if got.Error.Type != tc.typ || got.Error.Code != tc.code {
				t.Errorf("type/code %q/%q, want %q/%q", got.Error.Type, got.Error.Code, tc.typ, tc.code)
			}
			if tc.hasParam && (got.Error.Param == nil || *got.Error.Param != tc.param) {
				t.Errorf("param %v, want %q", got.Error.Param, tc.param)
			}
			if tc.wantMsg != "" && got.Error.Message != tc.wantMsg {
				t.Errorf("message %q, want %q", got.Error.Message, tc.wantMsg)
			}
		})
	}
	if e.mock.Stats().Requests != 0 {
		t.Errorf("rejected requests reached the upstream: %d", e.mock.Stats().Requests)
	}
}

func TestModelsListFiltersByTenant(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	h := e.gw.Handler()

	ids := func(key string) []string {
		rec := do(h, http.MethodGet, "/v1/models", key, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		var list struct {
			Object string `json:"object"`
			Data   []struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				OwnedBy string `json:"owned_by"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if list.Object != "list" {
			t.Errorf("object %q, want list", list.Object)
		}
		var out []string
		for _, m := range list.Data {
			if m.Object != "model" || m.OwnedBy != "routepilot" || m.Created != 0 {
				t.Errorf("unexpected model entry %+v", m)
			}
			out = append(out, m.ID)
		}
		return out
	}

	if got := strings.Join(ids(keyA), ","); got != "model-a,model-b" {
		t.Errorf("tenant-a models %q, want model-a,model-b", got)
	}
	if got := strings.Join(ids(keyB), ","); got != "model-a" {
		t.Errorf("tenant-b models %q, want model-a", got)
	}
	if rec := do(h, http.MethodGet, "/v1/models", "", ""); rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated list: status %d, want 401", rec.Code)
	}
}

func TestMethodAndPathErrors(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	h := e.gw.Handler()

	tests := []struct {
		method, path, allow string
		status              int
		code                string
	}{
		{http.MethodGet, "/v1/chat/completions", "POST", 405, "method_not_allowed"},
		{http.MethodPost, "/v1/models", "GET", 405, "method_not_allowed"},
		{http.MethodGet, "/v1/nope", "", 404, "not_found"},
		{http.MethodGet, "/", "", 404, "not_found"},
	}
	for _, tc := range tests {
		rec := do(h, tc.method, tc.path, keyA, "")
		if rec.Code != tc.status {
			t.Errorf("%s %s: status %d, want %d", tc.method, tc.path, rec.Code, tc.status)
			continue
		}
		got := decodeErr(t, rec)
		if got.Error.Code != tc.code || got.Error.Type != "invalid_request_error" {
			t.Errorf("%s %s: type/code %q/%q", tc.method, tc.path, got.Error.Type, got.Error.Code)
		}
		if tc.allow != "" && rec.Header().Get("Allow") != tc.allow {
			t.Errorf("%s %s: Allow %q, want %q", tc.method, tc.path, rec.Header().Get("Allow"), tc.allow)
		}
		if rec.Header().Get("x-routepilot-request-id") == "" {
			t.Errorf("%s %s: missing request id header", tc.method, tc.path)
		}
	}
}

func TestLogsContainNoPromptText(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=3"})
	const sentinel = "SENTINEL-PROMPT-7731"
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", sentinel))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	// Also log a rejected request carrying the key.
	do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("model-a", sentinel))

	out := e.logs.String()
	for _, banned := range []string{sentinel, keyA, "test-key-nobody", "Bearer"} {
		if strings.Contains(out, banned) {
			t.Errorf("log output contains %q:\n%s", banned, out)
		}
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d log lines, want 2:\n%s", len(lines), out)
	}
	for _, key := range []string{"request_id=", "tenant=tenant-a", "model=model-a", "deployment=dep-a", "status=200", "duration_ms=", fmt.Sprintf("prompt_bytes=%d", len(sentinel))} {
		if !strings.Contains(lines[0], key) {
			t.Errorf("first log line missing %q:\n%s", key, lines[0])
		}
	}
	if !strings.Contains(lines[1], "status=401") || strings.Contains(lines[1], "tenant=tenant") {
		t.Errorf("unauthenticated log line should have status 401 and no tenant:\n%s", lines[1])
	}
}

func TestNewRequestID(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 100; i++ {
		id := newRequestID()
		if len(id) != 24 {
			t.Fatalf("id %q has length %d, want 24", id, len(id))
		}
		if id != strings.ToLower(id) {
			t.Fatalf("id %q is not lowercase", id)
		}
		if _, err := hex.DecodeString(id); err != nil {
			t.Fatalf("id %q is not hex: %v", id, err)
		}
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
	}
}

// doHeaders is do with extra request headers.
func doHeaders(h http.Handler, key, body string, headers map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+key)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestHeaderRoutingControls(t *testing.T) {
	const bodyNoOpts = `{"model":"model-a","messages":[{"role":"user","content":"hi"}]}`

	t.Run("header timeout applies", func(t *testing.T) {
		e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=stall", serverDefault: 1500, serverMax: 3000})
		start := time.Now()
		rec := doHeaders(e.gw.Handler(), keyA, bodyNoOpts, map[string]string{"x-routepilot-timeout-ms": "100"})
		if elapsed := time.Since(start); elapsed > time.Second {
			t.Errorf("took %v, want under 1s (the header timeout)", elapsed)
		}
		if rec.Code != http.StatusGatewayTimeout {
			t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
		}
	})

	t.Run("body wins over header", func(t *testing.T) {
		e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=stall", serverDefault: 1500, serverMax: 3000})
		body := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":100}}`
		start := time.Now()
		rec := doHeaders(e.gw.Handler(), keyA, body, map[string]string{"x-routepilot-timeout-ms": "60000"})
		if elapsed := time.Since(start); elapsed > time.Second {
			t.Errorf("took %v, want under 1s (the body timeout wins)", elapsed)
		}
		if rec.Code != http.StatusGatewayTimeout {
			t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
		}
	})

	bad := []struct{ header, value string }{
		{"x-routepilot-min-quality", "abc"},
		{"x-routepilot-min-quality", "1.5"},
		{"x-routepilot-min-quality", "-0.1"},
		{"x-routepilot-min-quality", "NaN"},
		{"x-routepilot-min-quality", ""},
		{"x-routepilot-slo-ttft-ms", "0"},
		{"x-routepilot-slo-e2e-ms", "-5"},
		{"x-routepilot-timeout-ms", "abc"},
		{"x-routepilot-timeout-ms", "1.5"},
		{"x-routepilot-timeout-ms", "0"},
		{"x-routepilot-timeout-ms", "99999999999999999999"},
	}
	for _, tc := range bad {
		t.Run("bad "+tc.header+"="+tc.value, func(t *testing.T) {
			e := newMockEnv(t, cfgOpts{})
			rec := doHeaders(e.gw.Handler(), keyA, bodyNoOpts, map[string]string{tc.header: tc.value})
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
			}
			got := decodeErr(t, rec)
			if got.Error.Code != "invalid_request" || got.Error.Param == nil || *got.Error.Param != tc.header {
				t.Errorf("code %q param %v, want invalid_request / %q", got.Error.Code, got.Error.Param, tc.header)
			}
			if n := e.mock.Stats().Requests; n != 0 {
				t.Errorf("rejected request reached the upstream: %d", n)
			}
		})
	}
	// A header the body overrides is still validated on its own: the request
	// is the client's, and a bad header is a client error whether or not it
	// would have been used.
	overridden := []struct{ header, value, bodyOpts string }{
		{"x-routepilot-min-quality", "1.5", `{"min_quality":0.5}`},
		{"x-routepilot-slo-ttft-ms", "0", `{"slo_ttft_ms":100}`},
		{"x-routepilot-slo-e2e-ms", "-5", `{"slo_e2e_ms":100}`},
		{"x-routepilot-timeout-ms", "0", `{"timeout_ms":100}`},
	}
	for _, tc := range overridden {
		t.Run("bad "+tc.header+"="+tc.value+" even when the body sets it", func(t *testing.T) {
			e := newMockEnv(t, cfgOpts{})
			body := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":` + tc.bodyOpts + `}`
			rec := doHeaders(e.gw.Handler(), keyA, body, map[string]string{tc.header: tc.value})
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
			}
			got := decodeErr(t, rec)
			if got.Error.Code != "invalid_request" || got.Error.Param == nil || *got.Error.Param != tc.header {
				t.Errorf("code %q param %v, want invalid_request / %q", got.Error.Code, got.Error.Param, tc.header)
			}
			if n := e.mock.Stats().Requests; n != 0 {
				t.Errorf("rejected request reached the upstream: %d", n)
			}
		})
	}
}

func TestMergeRoutingHeaders(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	all := http.Header{}
	all.Set("x-routepilot-min-quality", "0.5")
	all.Set("x-routepilot-slo-ttft-ms", "200")
	all.Set("x-routepilot-slo-e2e-ms", "3000")
	all.Set("x-routepilot-timeout-ms", "9000")

	valid := func(rp *openai.RoutingOptions) *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{
			Model:      "model-a",
			Messages:   []openai.Message{{Role: "user"}},
			RoutePilot: rp,
		}
	}
	t.Run("creates the options when the body had none", func(t *testing.T) {
		req := valid(nil)
		if err := mergeRoutingHeaders(all, req); err != nil {
			t.Fatalf("merge: %v", err)
		}
		rp := req.RoutePilot
		if rp == nil || *rp.MinQuality != 0.5 || *rp.SLOTTFTMs != 200 || *rp.SLOE2EMs != 3000 || *rp.TimeoutMs != 9000 {
			t.Errorf("RoutePilot = %+v", rp)
		}
	})
	t.Run("body wins field by field", func(t *testing.T) {
		req := valid(&openai.RoutingOptions{MinQuality: f(0.9), TimeoutMs: i(100)})
		if err := mergeRoutingHeaders(all, req); err != nil {
			t.Fatalf("merge: %v", err)
		}
		rp := req.RoutePilot
		if *rp.MinQuality != 0.9 || *rp.TimeoutMs != 100 {
			t.Errorf("body values lost: %+v", rp)
		}
		if *rp.SLOTTFTMs != 200 || *rp.SLOE2EMs != 3000 {
			t.Errorf("header values not filled: %+v", rp)
		}
	})
	t.Run("no headers leaves the request alone", func(t *testing.T) {
		req := valid(nil)
		if err := mergeRoutingHeaders(http.Header{}, req); err != nil || req.RoutePilot != nil {
			t.Errorf("err=%v RoutePilot=%+v, want nil and nil", err, req.RoutePilot)
		}
	})
	t.Run("invalid header is rejected although the body sets the field", func(t *testing.T) {
		h := http.Header{}
		h.Set("x-routepilot-timeout-ms", "0")
		req := valid(&openai.RoutingOptions{TimeoutMs: i(100)})
		err := mergeRoutingHeaders(h, req)
		if err == nil || err.Param != "x-routepilot-timeout-ms" {
			t.Fatalf("err = %+v, want a 400 for x-routepilot-timeout-ms", err)
		}
		if *req.RoutePilot.TimeoutMs != 100 {
			t.Errorf("body timeout_ms changed to %d", *req.RoutePilot.TimeoutMs)
		}
	})
}

func TestSlowBodyIs408(t *testing.T) {
	gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": okProvider()}, Options{
		ReadBodyTimeout: 150 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)

	conn, err := net.Dial("tcp", srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	// Promise 1000 body bytes, send 10, then go quiet.
	fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %s\r\n"+
		"Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"model\":\"", keyA)

	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestTimeout {
		t.Fatalf("status %d, want 408", resp.StatusCode)
	}
	var got errBody
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Error.Type != "invalid_request_error" || got.Error.Code != "request_timeout" ||
		got.Error.Message != "request body read timed out" {
		t.Errorf("error = %+v", got.Error)
	}
}

func TestModelsListsAuto(t *testing.T) {
	ids := func(e *attemptEnv) (out []string, owners []string) {
		rec := do(e.gw.Handler(), http.MethodGet, "/v1/models", keyA, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		var list struct {
			Data []struct {
				ID      string `json:"id"`
				OwnedBy string `json:"owned_by"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("decode: %v", err)
		}
		for _, m := range list.Data {
			out, owners = append(out, m.ID), append(owners, m.OwnedBy)
		}
		return out, owners
	}
	models := []modelSpec{{id: "m1", in: "1", out: "1", deps: []depSpec{{"d1", "u1"}}}}

	withPolicy := newAttemptEnv(t, attemptOpts{policy: config.PolicyCheapest, models: models})
	got, owners := ids(withPolicy)
	if strings.Join(got, ",") != "auto,m1" || owners[0] != "routepilot" {
		t.Errorf("with a policy: ids %v owners %v, want auto first owned by routepilot, then m1", got, owners)
	}

	noPolicy := newAttemptEnv(t, attemptOpts{models: models})
	if got, _ := ids(noPolicy); strings.Join(got, ",") != "m1" {
		t.Errorf("without a policy: ids %v, want just m1", got)
	}
}

func TestNotFoundTruncatesModel(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	// One ASCII byte shifts the two-byte runes so byte 256 falls inside one.
	long := "x" + strings.Repeat("é", 2000)
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody(long, "hi"))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status %d, want 404", rec.Code)
	}
	msg := decodeErr(t, rec).Error.Message
	want := `model "` + long[:255] + `" does not exist`
	if msg != want {
		t.Errorf("message has %d bytes, want %d (255 bytes of model on a rune boundary)", len(msg), len(want))
	}
	if !utf8.ValidString(msg) {
		t.Error("message is not valid UTF-8")
	}
}

// deadlineRecorder is a recorder that supports http.ResponseController
// deadlines and remembers them.
type deadlineRecorder struct {
	*httptest.ResponseRecorder
	reads, writes []time.Time
}

func (d *deadlineRecorder) SetReadDeadline(t time.Time) error {
	d.reads = append(d.reads, t)
	return nil
}

func (d *deadlineRecorder) SetWriteDeadline(t time.Time) error {
	d.writes = append(d.writes, t)
	return nil
}

func TestWriteJSONSetsWriteDeadline(t *testing.T) {
	d := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	before := time.Now()
	writeJSON(d, http.StatusOK, map[string]string{"a": "b"})

	if len(d.writes) != 1 {
		t.Fatalf("SetWriteDeadline called %d times, want 1", len(d.writes))
	}
	if got := d.writes[0].Sub(before); got < 29*time.Second || got > 31*time.Second {
		t.Errorf("write deadline is %v ahead, want about 30s", got)
	}
	if d.Body.Len() == 0 {
		t.Error("nothing written")
	}
}

func TestBodyReadDeadlineSetAndCleared(t *testing.T) {
	gw, _ := newFakeGateway(t, cfgOpts{}, okProvider())
	d := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("model-a", "hi")))
	req.Header.Set("Authorization", "Bearer "+keyA)
	before := time.Now()
	gw.Handler().ServeHTTP(d, req)

	if d.Code != http.StatusOK {
		t.Fatalf("status %d: %s", d.Code, d.Body.String())
	}
	if len(d.reads) != 2 {
		t.Fatalf("SetReadDeadline called %d times, want 2 (set, then clear)", len(d.reads))
	}
	if got := d.reads[0].Sub(before); got < 29*time.Second || got > 31*time.Second {
		t.Errorf("read deadline is %v ahead, want about 30s (the default)", got)
	}
	if !d.reads[1].IsZero() {
		t.Errorf("read deadline after the body = %v, want the zero time", d.reads[1])
	}
}
