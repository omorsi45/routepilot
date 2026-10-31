package gateway

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
)

// TestHugeTimeoutIsClampedToTheTenantMaximum: a timeout_ms too large for a duration in
// nanoseconds must come out as the tenant's longest deadline, not wrap into the past and
// cancel the request before it reaches the upstream.
func TestHugeTimeoutIsClampedToTheTenantMaximum(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models:             []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
		tenantMaxTimeoutMs: 30_000,
	})
	tenant := tenantA(e.gw.cfg)
	for _, ms := range []int{9_223_372_036_854, 9_223_372_036_854_775, math.MaxInt64 / 1000, math.MaxInt} {
		req := &openai.ChatCompletionRequest{RoutePilot: &openai.RoutingOptions{TimeoutMs: &ms}}
		if got, want := e.gw.deadline(tenant, req), 30*time.Second; got != want {
			t.Errorf("deadline for timeout_ms=%d = %v, want the tenant maximum %v", ms, got, want)
		}
	}

	t.Run("in the body", func(t *testing.T) {
		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"routepilot":{"timeout_ms":9223372036854775}}`
		if rec := e.post(body); rec.Code != http.StatusOK {
			t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
		}
	})
	t.Run("in the header", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("m", "hi")))
		req.Header.Set("Authorization", "Bearer "+keyA)
		req.Header.Set(headerTimeoutMs, strconv.Itoa(math.MaxInt))
		rec := httptest.NewRecorder()
		e.gw.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
		}
	})
	if got := e.count("ua"); got != 2 {
		t.Errorf("upstream was called %d times, want 2: the deadline must not have expired before the call", got)
	}
}
