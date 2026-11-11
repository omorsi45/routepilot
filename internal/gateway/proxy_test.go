package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

func TestProxyNonStreaming(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;tokens=3"})
	body := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":5000}}`
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	reqID := rec.Header().Get("x-routepilot-request-id")
	if reqID == "" {
		t.Fatal("missing x-routepilot-request-id")
	}
	if got := rec.Header().Get("x-routepilot-model"); got != "model-a" {
		t.Errorf("x-routepilot-model %q, want model-a", got)
	}
	if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-a" {
		t.Errorf("x-routepilot-deployment %q, want dep-a", got)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type %q, want application/json", got)
	}

	var resp openai.ChatCompletion
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.ID != "chatcmpl-"+reqID {
		t.Errorf("id %q, want chatcmpl-%s", resp.ID, reqID)
	}
	if resp.Model != "model-a" {
		t.Errorf("model %q, want the catalog id model-a", resp.Model)
	}
	if resp.Object != openai.ObjectChatCompletion || len(resp.Choices) != 1 || resp.Usage == nil {
		t.Errorf("unexpected response shape: %s", rec.Body.String())
	}
	if resp.Created == 0 {
		t.Error("created is zero")
	}

	var up map[string]json.RawMessage
	if err := json.Unmarshal(e.upstream.last(t), &up); err != nil {
		t.Fatalf("decode upstream body: %v", err)
	}
	if string(up["model"]) != `"m;tokens=3"` {
		t.Errorf("upstream saw model %s, want the upstream_model", up["model"])
	}
	if _, ok := up["routepilot"]; ok {
		t.Error("upstream request still carries the routepilot key")
	}
	if got := e.mock.Stats().Completed; got != 1 {
		t.Errorf("mock completed %d, want 1", got)
	}
}

func TestProxyUpstreamHTMLError(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=html502"})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(strings.ToLower(rec.Body.String()), "<html") {
		t.Errorf("response leaks upstream HTML: %s", rec.Body.String())
	}
	got := decodeErr(t, rec)
	if got.Error.Type != "upstream_error" || got.Error.Code != "upstream_error" {
		t.Errorf("type/code %q/%q, want upstream_error", got.Error.Type, got.Error.Code)
	}
	if got.Error.Message != "upstream dep-a failed: Bad Gateway" {
		t.Errorf("message %q", got.Error.Message)
	}
	if strings.Contains(rec.Body.String(), "127.0.0.1") {
		t.Errorf("response leaks the upstream address: %s", rec.Body.String())
	}
}

func TestProxyUpstream400Passthrough(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=http400"})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
	}
	got := decodeErr(t, rec)
	if got.Error.Type != "invalid_request_error" || got.Error.Code != "invalid_request" {
		t.Errorf("type/code %q/%q", got.Error.Type, got.Error.Code)
	}
	if got.Error.Message != "mock injected invalid request" {
		t.Errorf("message %q, want the upstream message", got.Error.Message)
	}
}

func TestProxyDeadline(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=stall"})
	body := `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":100}}`
	start := time.Now()
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("took %v, want under 1s", elapsed)
	}
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec).Error.Code; got != "upstream_timeout" {
		t.Errorf("code %q, want upstream_timeout", got)
	}
}

func TestClientCancelWritesNothing(t *testing.T) {
	e := newMockEnv(t, cfgOpts{upstreamA: "m;fail=stall"})
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("model-a", "hi"))).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyA)
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		e.gw.Handler().ServeHTTP(rec, req)
		close(done)
	}()
	time.Sleep(50 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return after the client went away")
	}
	if rec.Body.Len() != 0 {
		t.Errorf("wrote a body to a gone client: %s", rec.Body.String())
	}
	if !strings.Contains(e.logs.String(), "level=INFO") || !strings.Contains(e.logs.String(), "status=499") {
		t.Errorf("want an info log with status 499, got: %s", e.logs.String())
	}
}

func TestUpstreamErrorMapping(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		status  int
		typ     string
		code    string
		message string
	}{
		{
			name:   "translation error as-is",
			err:    openai.ErrInvalidRequest("tool_choice is not supported", "tool_choice"),
			status: 400, typ: "invalid_request_error", code: "invalid_request", message: "tool_choice is not supported",
		},
		{
			name:   "provider timeout",
			err:    &provider.Error{Kind: provider.KindTimeout, Retryable: true, Message: "request timed out"},
			status: 504, typ: "upstream_error", code: "upstream_timeout", message: "upstream provider timed out",
		},
		{
			name:   "connect failure",
			err:    &provider.Error{Kind: provider.KindConnect, Retryable: true, Message: "connection failed: dial tcp 10.1.2.3:443: refused"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: connect",
		},
		{
			name:   "protocol failure",
			err:    &provider.Error{Kind: provider.KindProtocol, Message: "bad json from 10.1.2.3"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: protocol",
		},
		{
			name:   "retryable http",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 503, Retryable: true, Message: "overloaded"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: Service Unavailable",
		},
		{
			name:   "upstream auth failure is not leaked as 401",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 401, Retryable: true, Message: "bad provider key"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: Unauthorized",
		},
		{
			name:   "non-retryable 400",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 400, Message: "context too long"},
			status: 400, typ: "invalid_request_error", code: "invalid_request", message: "context too long",
		},
		{
			name:   "non-retryable 422",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 422, Message: "unprocessable"},
			status: 422, typ: "invalid_request_error", code: "upstream_rejected", message: "unprocessable",
		},
		{
			name:   "non-retryable 5xx http is a gateway error",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 501, Message: "not implemented"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: Not Implemented",
		},
		{
			name:   "canceled without client cancel",
			err:    &provider.Error{Kind: provider.KindCanceled, Message: "request canceled"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: canceled",
		},
		{
			name:   "anthropic overloaded status",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 529, Retryable: true, Message: "Overloaded"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: Overloaded",
		},
		{
			name:   "status without text",
			err:    &provider.Error{Kind: provider.KindHTTP, Status: 599, Retryable: true, Message: "odd"},
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: status 599",
		},
		{
			name:   "unknown error",
			err:    errors.New("boom at 10.1.2.3"),
			status: 502, typ: "upstream_error", code: "upstream_error", message: "upstream dep-a failed: error",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gw, _ := newFakeGateway(t, cfgOpts{}, &fakeProvider{
				complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
					return nil, tc.err
				},
			})
			rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
			if rec.Code != tc.status {
				t.Fatalf("status %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
			got := decodeErr(t, rec)
			if got.Error.Type != tc.typ || got.Error.Code != tc.code || got.Error.Message != tc.message {
				t.Errorf("got %+v, want type=%s code=%s message=%q", got.Error, tc.typ, tc.code, tc.message)
			}
			if strings.Contains(rec.Body.String(), "10.1.2.3") {
				t.Errorf("response leaks upstream detail: %s", rec.Body.String())
			}
			if rec.Header().Get("x-routepilot-request-id") == "" {
				t.Error("error response is missing the request id header")
			}
		})
	}
}

func TestDeadlineClamping(t *testing.T) {
	tests := []struct {
		name       string
		opts       cfgOpts
		timeoutMs  int // routepilot.timeout_ms, 0 = unset
		wantMillis int
	}{
		{"server default", cfgOpts{serverDefault: 30000, serverMax: 60000}, 0, 30000},
		{"request value", cfgOpts{serverDefault: 30000, serverMax: 60000}, 5000, 5000},
		{"clamped to server max", cfgOpts{serverDefault: 30000, serverMax: 60000}, 90000, 60000},
		{"clamped to tenant max", cfgOpts{serverDefault: 30000, serverMax: 60000, tenantAMaxMs: 10000}, 90000, 10000},
		{"tenant max also clamps the default", cfgOpts{serverDefault: 30000, serverMax: 60000, tenantAMaxMs: 10000}, 0, 10000},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got time.Duration
			gw, _ := newFakeGateway(t, tc.opts, &fakeProvider{
				complete: func(ctx context.Context, _ *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
					dl, ok := ctx.Deadline()
					if !ok {
						t.Error("upstream context has no deadline")
					}
					got = time.Until(dl)
					return &openai.ChatCompletion{Object: openai.ObjectChatCompletion}, nil
				},
			})
			body := chatBody("model-a", "hi")
			if tc.timeoutMs > 0 {
				body = `{"model":"model-a","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":` + itoa(tc.timeoutMs) + `}}`
			}
			rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			want := time.Duration(tc.wantMillis) * time.Millisecond
			if got > want || got < want-2*time.Second {
				t.Errorf("deadline in %v, want about %v", got, want)
			}
		})
	}
}

func TestCreatedFallbackAndPreserved(t *testing.T) {
	for _, tc := range []struct {
		name     string
		upstream int64
		want     int64
	}{
		{"zero uses Now", 0, 1700000000},
		{"upstream value kept", 1234, 1234},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gw, _ := newFakeGateway(t, cfgOpts{}, &fakeProvider{
				complete: func(_ context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
					if req.Model != "up-a" || req.RoutePilot != nil {
						t.Errorf("provider got model %q routepilot %v", req.Model, req.RoutePilot)
					}
					return &openai.ChatCompletion{ID: "up-id", Object: openai.ObjectChatCompletion, Created: tc.upstream, Model: "up-a"}, nil
				},
			})
			rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
			var resp openai.ChatCompletion
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if resp.Created != tc.want {
				t.Errorf("created %d, want %d", resp.Created, tc.want)
			}
			if resp.ID == "up-id" || resp.Model != "model-a" {
				t.Errorf("id/model not rewritten: %q %q", resp.ID, resp.Model)
			}
		})
	}
}

func TestNilCompletionIsInternalError(t *testing.T) {
	gw, _ := newFakeGateway(t, cfgOpts{}, &fakeProvider{
		complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			return nil, nil
		},
	})
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status %d, want 500", rec.Code)
	}
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestProxyRedirectFromUpstreamIs502(t *testing.T) {
	var hits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
	}))
	defer target.Close()
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	cfg := buildConfig(t, cfgOpts{baseURL: redirector.URL + "/v1"})
	prov := openaicompat.New(cfg.Providers[0], "", &http.Client{})
	gw, err := New(cfg, map[string]provider.Provider{"mockp": prov}, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec := do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
	}
	if got := decodeErr(t, rec); got.Error.Type != "upstream_error" {
		t.Errorf("error type %q, want upstream_error", got.Error.Type)
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("redirect target received %d requests, want 0", n)
	}
}
