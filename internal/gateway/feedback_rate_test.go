package gateway

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/quota"
)

// TestFeedbackIsChargedAgainstTheTenantRequestRate: a feedback call writes a line to the
// decision log, so it counts against the tenant's request rate like a chat request does,
// from the same allowance, and is refused with the same 429. A valid key is still needed.
func TestFeedbackIsChargedAgainstTheTenantRequestRate(t *testing.T) {
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		for _, tn := range []*config.TenantConfig{tenantA(cfg), tenantB(cfg)} {
			tn.RPM, tn.RPMBurst = 60, 2
		}
	}, okProvider(), quota.Limiters{})
	sink := attachLog(t, gw)
	h := gw.Handler()
	body := feedbackBody("dec_"+strings.Repeat("c3", 12), true)

	for i := 0; i < 2; i++ {
		if rec := postFeedback(h, keyA, body); rec.Code != http.StatusNoContent {
			t.Fatalf("feedback %d: status %d, want 204: %s", i+1, rec.Code, rec.Body.String())
		}
	}
	wantRateLimited(t, postFeedback(h, keyA, body), "tenant request rate exceeded", "1")

	// The allowance is shared with chat requests, so the burst feedback spent is gone.
	wantRateLimited(t, chat(gw, keyA, chatBody("model-a", "hi")), "tenant request rate exceeded", "1")

	// Another tenant has an allowance of its own, and a wrong key is a 401, not a 429.
	if rec := postFeedback(h, keyB, body); rec.Code != http.StatusNoContent {
		t.Errorf("other tenant: status %d, want 204: %s", rec.Code, rec.Body.String())
	}
	if rec := postFeedback(h, "test-key-nobody", body); rec.Code != http.StatusUnauthorized {
		t.Errorf("wrong key: status %d, want 401: %s", rec.Code, rec.Body.String())
	}

	if lines := ofType(sink.records(), "feedback"); len(lines) != 3 {
		t.Errorf("%d feedback lines, want 3: the refused call must write none", len(lines))
	}
}

// TestFeedbackWithoutARequestRateIsNotLimited: a tenant with no rpm has no allowance to
// charge, so feedback is as unlimited for it as chat is.
func TestFeedbackWithoutARequestRateIsNotLimited(t *testing.T) {
	gw, _ := newAdmissionGateway(t, nil, okProvider(), quota.Limiters{})
	attachLog(t, gw)
	body := feedbackBody("dec_"+strings.Repeat("d4", 12), false)
	for i := 0; i < 50; i++ {
		if rec := postFeedback(gw.Handler(), keyA, body); rec.Code != http.StatusNoContent {
			t.Fatalf("feedback %d: status %d, want 204: %s", i+1, rec.Code, rec.Body.String())
		}
	}
}

// TestFeedbackRateLimiterFailureIsAnOverload: when the limiter backend fails the call is
// refused with 503, as a chat request is, and nothing is written.
func TestFeedbackRateLimiterFailureIsAnOverload(t *testing.T) {
	stub := &errLimiters{mem: quota.NewMemory(), which: "requests", err: errors.New("redis down")}
	gw, logs := newAdmissionGateway(t, func(cfg *config.Config) {
		tenantA(cfg).RPM, tenantA(cfg).RPMBurst = 60, 2
	}, okProvider(), stub.limiters())
	sink := attachLog(t, gw)

	rec := postFeedback(gw.Handler(), keyA, feedbackBody("dec_"+strings.Repeat("e5", 12), true))

	if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
		t.Fatalf("status %d body %s, want 503 overloaded", rec.Code, rec.Body.String())
	}
	if !strings.Contains(logs.String(), "request rate check") {
		t.Errorf("the limiter failure was not logged:\n%s", logs.String())
	}
	if lines := ofType(sink.records(), "feedback"); len(lines) != 0 {
		t.Errorf("%d feedback lines written, want none", len(lines))
	}
}
