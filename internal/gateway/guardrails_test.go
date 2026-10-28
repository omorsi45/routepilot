package gateway

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
)

const (
	// adminKey is a fake admin key. Only its SHA-256, computed here, reaches the config.
	adminKey = "test-admin-key-0001"
	// guardPrompt is the prompt the committed tiny artifact scores in distribution.
	guardPrompt = learnedPrompt
)

// rolloutStore is a usage ledger that keeps the rollout events it is given.
type rolloutStore struct {
	*recordingStore
	mu     sync.Mutex
	events []accounting.RolloutEvent
	fail   error
}

func newRolloutStore() *rolloutStore { return &rolloutStore{recordingStore: &recordingStore{}} }

func (s *rolloutStore) RecordRollout(_ context.Context, ev accounting.RolloutEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fail != nil {
		return s.fail
	}
	s.events = append(s.events, ev)
	return nil
}

func (s *rolloutStore) got() []accounting.RolloutEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]accounting.RolloutEvent(nil), s.events...)
}

// guardEnv is a learned gateway with a canary, an admin key, a second tenant, a usage ledger
// that keeps rollout events, and a clock the test moves. It is built on learnedEnv's config,
// artifact and registry; its own Server (gw) is the one the tests talk to.
type guardEnv struct {
	*learnedEnv
	gw    *Server
	store *rolloutStore
	clock *fakeClock
}

// newGuardEnv builds the environment. A non-nil prov replaces the HTTP provider.
func newGuardEnv(t *testing.T, o learnedOpts, prov provider.Provider) *guardEnv {
	t.Helper()
	e := newLearnedEnv(t, o)
	e.cfg.Server.AdminKeySHA256 = keyHash(adminKey)
	e.cfg.Tenants = append(e.cfg.Tenants, config.TenantConfig{
		ID: "tenant-b", APIKeySHA256: keyHash(keyB), AllowedModels: []string{"*"}, DefaultMaxTokens: 4096,
	})
	if prov == nil {
		prov = openaicompat.New(e.cfg.Providers[0], "", e.upstream.Client())
	}
	g := &guardEnv{learnedEnv: e, store: newRolloutStore(), clock: &fakeClock{}}
	gw, err := New(e.cfg, map[string]provider.Provider{"mockp": prov}, Options{
		Logger:     slog.New(slog.NewTextHandler(e.logs, nil)),
		Policies:   e.reg,
		Rand:       func() float64 { return 0.5 },
		Now:        g.clock.now,
		Accounting: g.store,
		Sleep:      func(context.Context, time.Duration) error { return nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	g.gw = gw
	return g
}

func (g *guardEnv) do(key, body string, headers map[string]string) *httptest.ResponseRecorder {
	return doHeaders(g.gw.Handler(), key, body, headers)
}

// auto sends the guard prompt as tenant-a.
func (g *guardEnv) auto() *httptest.ResponseRecorder {
	return g.do(keyA, chatBody("auto", guardPrompt), nil)
}

// autoN sends the guard prompt n times and checks every answer has status want.
func (g *guardEnv) autoN(t *testing.T, n, want int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if rec := g.auto(); rec.Code != want {
			t.Fatalf("request %d: status %d, want %d: %s", i+1, rec.Code, want, rec.Body.String())
		}
	}
}

// seedStable records n clean stable requests, the baseline a canary at 100 percent has no
// traffic to build.
func (g *guardEnv) seedStable(n int) {
	for i := 0; i < n; i++ {
		g.gw.guard.Observe(policy.RoleStable, policy.Observation{Latency: 50 * time.Millisecond})
	}
}

func (g *guardEnv) stats(role string) policy.GuardStats { return g.gw.guard.Stats(role) }

// rejectCanary is a canary configuration that answers every request with 422.
var rejectCanary = policyYAML(0.99, 0, "reject")

// TestGuardObservesLearnedRequests: every request the learned policy decided is observed
// under the role of the slot that decided it, with the outcome the brief defines.
func TestGuardObservesLearnedRequests(t *testing.T) {
	t.Run("a clean answer", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
		g.autoN(t, 3, http.StatusOK)
		if s := g.stats("canary"); s.Requests != 3 || s.Errors != 0 || s.Relaxed != 0 {
			t.Errorf("canary stats = %+v, want 3 requests, no errors, none relaxed", s)
		}
		if s := g.stats("stable"); s != (policy.GuardStats{}) {
			t.Errorf("stable stats = %+v, want zero: the canary took everything", s)
		}
	})
	t.Run("the stable role at percent 0", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 0}, nil)
		g.autoN(t, 2, http.StatusOK)
		if s := g.stats("stable"); s.Requests != 2 {
			t.Errorf("stable stats = %+v, want 2 requests", s)
		}
		if s := g.stats("canary"); s.Requests != 0 {
			t.Errorf("canary stats = %+v, want none", s)
		}
	})
	t.Run("a relaxed answer", func(t *testing.T) {
		// Floor 0.99 with best_effort: the policy gives up quality and answers anyway.
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0.99, 0, "best_effort"), canaryPct: 100}, nil)
		rec := g.auto()
		if rec.Code != http.StatusOK || rec.Header().Get("x-routepilot-relaxed") != "quality" {
			t.Fatalf("status %d relaxed %q, want 200 and quality", rec.Code, rec.Header().Get("x-routepilot-relaxed"))
		}
		if s := g.stats("canary"); s.Requests != 1 || s.Relaxed != 1 || s.Errors != 0 {
			t.Errorf("canary stats = %+v, want 1 request, relaxed, no error", s)
		}
	})
	t.Run("a routing error answered 422 is an error", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
		g.autoN(t, 2, http.StatusUnprocessableEntity)
		if s := g.stats("canary"); s.Requests != 2 || s.Errors != 2 {
			t.Errorf("canary stats = %+v, want 2 requests and 2 errors", s)
		}
	})
	t.Run("a server error is an error", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{
			canary: policyYAML(0, 0, "best_effort"), canaryPct: 0,
			weakUp: "weak-up;fail=http500", strongUp: "strong-up;fail=http500",
		}, nil)
		if rec := g.auto(); rec.Code < 500 {
			t.Fatalf("status %d, want a 5xx when both models fail", rec.Code)
		}
		if s := g.stats("stable"); s.Requests != 1 || s.Errors != 1 {
			t.Errorf("stable stats = %+v, want 1 request and 1 error", s)
		}
	})
	t.Run("a client validation error is not an error", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
		// No model has the json_object capability, so the router answers 400 for the client.
		body := `{"model":"auto","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`
		rec := g.do(keyA, body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
		}
		if s := g.stats("canary"); s.Requests != 1 || s.Errors != 0 {
			t.Errorf("canary stats = %+v, want 1 request and no error: a 400 is the client's", s)
		}
	})
	t.Run("a pinned request and a refused key are not observed", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
		if rec := g.do(keyA, chatBody("tiny-weak", "hi"), nil); rec.Code != http.StatusOK {
			t.Fatalf("pinned: status %d: %s", rec.Code, rec.Body.String())
		}
		if rec := g.do("test-key-nobody", chatBody("auto", "hi"), nil); rec.Code != http.StatusUnauthorized {
			t.Fatalf("bad key: status %d", rec.Code)
		}
		if rec := g.do(keyA, `{"model":`, nil); rec.Code != http.StatusBadRequest {
			t.Fatalf("bad body: status %d", rec.Code)
		}
		for _, role := range []string{"stable", "canary"} {
			if s := g.stats(role); s != (policy.GuardStats{}) {
				t.Errorf("%s stats = %+v, want zero", role, s)
			}
		}
	})
	t.Run("the latency is the whole request", func(t *testing.T) {
		var g *guardEnv
		prov := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			g.clock.advance(150 * time.Millisecond)
			return &openai.ChatCompletion{}, nil
		}}
		g = newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, prov)
		g.autoN(t, 5, http.StatusOK)
		if p95 := g.stats("canary").P95; p95 < 150*time.Millisecond || p95 > 152*time.Millisecond {
			t.Errorf("canary p95 = %v, want about 150ms", p95)
		}
	})
}

// TestAutoRollbackSetsCanaryZero: when the guard trips, the gateway takes the canary to 0
// percent, records an auto_rollback event, counts a rollback, logs the reason once, and stays
// quiet until the canary is raised again.
func TestAutoRollbackSetsCanaryZero(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	snap := g.reg.Snapshot()
	g.seedStable(1000)

	g.autoN(t, 199, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 100 {
		t.Fatalf("canary percent = %v after 199 bad requests, want 100: the guard needs 200", pct)
	}
	if ev := g.store.got(); len(ev) != 0 {
		t.Fatalf("events = %+v before the trip, want none", ev)
	}

	g.autoN(t, 1, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()

	after := g.reg.Snapshot()
	if after.CanaryPercent != 0 {
		t.Errorf("canary percent = %v after the trip, want 0", after.CanaryPercent)
	}
	if after.Canary == nil || after.Canary.Version != snap.Canary.Version || after.Stable.Version != snap.Stable.Version {
		t.Errorf("the slots changed: canary %v stable %v, want them kept", after.Canary, after.Stable.Version)
	}
	events := g.store.got()
	if len(events) != 1 {
		t.Fatalf("%d rollout events, want 1: %+v", len(events), events)
	}
	ev := events[0]
	if ev.Action != accounting.ActionAutoRollback || ev.Actor != "guardrail" || ev.Percent != 0 ||
		ev.From != snap.Canary.Version || ev.To != snap.Stable.Version {
		t.Errorf("event = %+v, want auto_rollback by guardrail from %s to %s at 0 percent", ev, snap.Canary.Version, snap.Stable.Version)
	}
	if !strings.Contains(ev.Reason, "error rate") || ev.At.IsZero() {
		t.Errorf("event reason %q at %v, want the guard's reason and a time", ev.Reason, ev.At)
	}
	if n := g.reg.Rollbacks(); n != 1 {
		t.Errorf("Registry.Rollbacks() = %d, want 1", n)
	}
	if v := scrapeOf(t, g.gw).value(t, "rp_policy_rollbacks_total", nil); v != 1 {
		t.Errorf("rp_policy_rollbacks_total = %v, want 1", v)
	}
	logs := g.logs.String()
	if n := strings.Count(logs, "guardrails rolled back the canary"); n != 1 {
		t.Errorf("the log has %d rollback lines, want 1:\n%s", n, logs)
	}
	if !strings.Contains(logs, "level=WARN") || !strings.Contains(logs, ev.Reason[:20]) {
		t.Errorf("the rollback line is not a warning with the reason:\n%s", logs)
	}

	// Quiet: more evaluations change nothing, and the stable slot serves from here on.
	for i := 0; i < 3; i++ {
		g.gw.EvaluateGuardrails()
	}
	if n := len(g.store.got()); n != 1 {
		t.Errorf("%d events after more evaluations, want still 1", n)
	}
	if n := strings.Count(g.logs.String(), "guardrails rolled back the canary"); n != 1 {
		t.Errorf("the log has %d rollback lines after more evaluations, want 1", n)
	}
	rec := g.do(keyA, chatBody("auto", guardPrompt), map[string]string{headerMinQuality: "0"})
	if got := rec.Header().Get("x-routepilot-policy-version"); got != snap.Stable.Version {
		t.Errorf("policy version %q after the rollback, want the stable %q", got, snap.Stable.Version)
	}
}

// TestAutoRollbackWithoutAStore: the rollout event is optional. Without a usage ledger the
// canary is still stopped.
func TestAutoRollbackWithoutAStore(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.gw.usageLedger = nil
	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 0 {
		t.Errorf("canary percent = %v, want 0", pct)
	}
}

// TestAutoRollbackSurvivesAFailingStore: a rollout event that cannot be written is counted and
// logged, and the canary is stopped all the same.
func TestAutoRollbackSurvivesAFailingStore(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.store.fail = context.DeadlineExceeded
	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 0 {
		t.Errorf("canary percent = %v, want 0", pct)
	}
	if n := g.gw.AccountingErrors(); n < 1 {
		t.Errorf("AccountingErrors() = %d, want the lost event counted", n)
	}
	if !strings.Contains(g.logs.String(), "rollout event write failed") {
		t.Errorf("the failed write is not logged:\n%s", g.logs.String())
	}
}

// TestGuardIgnoresAHealthyCanary: a canary as good as stable is left alone.
func TestGuardIgnoresAHealthyCanary(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	g.seedStable(1000)
	g.autoN(t, 250, http.StatusOK)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 100 {
		t.Errorf("canary percent = %v, want 100", pct)
	}
	if ev := g.store.got(); len(ev) != 0 {
		t.Errorf("events = %+v, want none", ev)
	}
}

// TestGuardResetsWhenTheCanaryChanges: a canary that replaces another one is not judged on
// the old one's requests.
func TestGuardResetsWhenTheCanaryChanges(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.seedStable(1000)
	g.autoN(t, 250, http.StatusUnprocessableEntity)

	// A new canary configuration replaces the bad one before the next evaluation.
	if err := os.WriteFile(filepath.Join(g.dir, "canary.yaml"), []byte(policyYAML(0.98, 0, "reject")), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := g.reg.Reload(); err != nil {
		t.Fatalf("Reload: %v", err)
	}
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 100 {
		t.Errorf("canary percent = %v: the new canary was blamed for the old one's requests", pct)
	}
	if s := g.stats("canary"); s.Requests != 0 {
		t.Errorf("canary stats = %+v after the canary changed, want them cleared", s)
	}

	// The new canary is judged on its own requests.
	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 0 {
		t.Errorf("canary percent = %v, want the new bad canary rolled back", pct)
	}
}

// TestGuardRearmsWhenTheCanaryIsRaised: after a rollback the guard stays quiet; when the
// share is raised again it starts from an empty window.
func TestGuardRearmsWhenTheCanaryIsRaised(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if g.reg.Snapshot().CanaryPercent != 0 {
		t.Fatal("the first trip did not happen")
	}

	g.reg.SetCanaryPercent(100)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 100 {
		t.Fatalf("canary percent = %v: the raised canary was rolled back on the old window", pct)
	}
	if n := len(g.store.got()); n != 1 {
		t.Fatalf("%d events, want still 1", n)
	}

	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 0 {
		t.Errorf("canary percent = %v, want the second bad run rolled back too", pct)
	}
	if n := len(g.store.got()); n != 2 {
		t.Errorf("%d events, want 2", n)
	}
}

// TestGuardRearmsAfterAnAdminRollback: an operator who stops the canary through the admin API
// and raises it again before the next evaluation still gets a fresh window, so the requests the
// canary served before the rollback do not count against the new run.
func TestGuardRearmsAfterAnAdminRollback(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.seedStable(1000)
	g.autoN(t, 150, http.StatusUnprocessableEntity)
	g.gw.EvaluateGuardrails() // 150 requests: below the minimum, nothing happens, the guard is watching

	if rec := adminDo(g.gw.Handler(), http.MethodPost, "/admin/policies/rollback", adminKey); rec.Code != http.StatusOK {
		t.Fatalf("rollback: status %d: %s", rec.Code, rec.Body.String())
	}
	g.reg.SetCanaryPercent(100) // raised again before any evaluation saw the 0
	g.gw.EvaluateGuardrails()
	if s := g.stats("canary"); s.Requests != 0 {
		t.Errorf("canary stats = %+v after the canary was stopped and raised again, want an empty window", s)
	}
	if s := g.stats("stable"); s.Requests != 0 {
		t.Errorf("stable stats = %+v, want an empty window: the pair starts over", s)
	}
}

// TestGuardRearmsAfterTheOperatorStopsTheCanary: a canary stopped by hand before the guard
// tripped writes no event and rolls nothing back, and raising it again starts a fresh window.
func TestGuardRearmsAfterTheOperatorStopsTheCanary(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	g.seedStable(1000)
	g.autoN(t, 200, http.StatusUnprocessableEntity)
	g.reg.SetCanaryPercent(0) // the operator was quicker than the ten-second evaluation
	g.gw.EvaluateGuardrails()
	if ev := g.store.got(); len(ev) != 0 {
		t.Errorf("events = %+v, want none: the canary had no share left to pull", ev)
	}
	if n := g.reg.Rollbacks(); n != 0 {
		t.Errorf("Rollbacks() = %d, want 0", n)
	}

	g.reg.SetCanaryPercent(100)
	g.gw.EvaluateGuardrails()
	if s := g.stats("canary"); s.Requests != 0 {
		t.Errorf("canary stats = %+v after the raise, want an empty window", s)
	}
}

// TestEvaluateGuardrailsWithoutALearnedPolicy: nothing to evaluate, nothing to break.
func TestEvaluateGuardrailsWithoutALearnedPolicy(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	e.gw.EvaluateGuardrails()
}

// ---- feedback to the guard (Ruling V1-R31) ----

// post a request as tenant-a and return the decision ID of its answer.
func (g *guardEnv) decisionID(t *testing.T) string {
	t.Helper()
	rec := g.auto()
	id := rec.Header().Get(headerDecisionID)
	if rec.Code != http.StatusOK || id == "" {
		t.Fatalf("status %d decision id %q: %s", rec.Code, id, rec.Body.String())
	}
	return id
}

func (g *guardEnv) feedback(key, id string, success bool) int {
	return postFeedback(g.gw.Handler(), key, feedbackBody(id, success)).Code
}

func TestFeedbackFeedsTheGuardByRole(t *testing.T) {
	for _, c := range []struct {
		name, role string
		pct        float64
	}{{"canary", "canary", 100}, {"stable", "stable", 0}} {
		t.Run(c.name, func(t *testing.T) {
			g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: c.pct}, nil)
			sink := attachLog(t, g.gw)
			a, b := g.decisionID(t), g.decisionID(t)
			if code := g.feedback(keyA, a, true); code != http.StatusNoContent {
				t.Fatalf("feedback: status %d", code)
			}
			if code := g.feedback(keyA, b, false); code != http.StatusNoContent {
				t.Fatalf("feedback: status %d", code)
			}
			s := g.stats(c.role)
			if s.Labels != 2 || s.Successes != 1 {
				t.Errorf("%s stats = %+v, want 2 labels and 1 success", c.role, s)
			}
			if s.Requests != 2 {
				t.Errorf("%s requests = %d, want 2: a label is not a request", c.role, s.Requests)
			}
			if n := len(ofType(sink.records(), "feedback")); n != 2 {
				t.Errorf("%d feedback lines in the decision log, want 2", n)
			}
		})
	}
}

func TestFeedbackFromAnotherTenantIsIgnoredByTheGuard(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	sink := attachLog(t, g.gw)
	id := g.decisionID(t)

	if code := g.feedback(keyB, id, false); code != http.StatusNoContent {
		t.Fatalf("tenant-b feedback: status %d, want 204", code)
	}
	if s := g.stats("canary"); s.Labels != 0 {
		t.Errorf("canary labels = %d after another tenant's verdict, want 0", s.Labels)
	}
	// The wrong tenant did not use up the decision: its owner can still answer.
	if code := g.feedback(keyA, id, true); code != http.StatusNoContent {
		t.Fatalf("tenant-a feedback: status %d", code)
	}
	if s := g.stats("canary"); s.Labels != 1 || s.Successes != 1 {
		t.Errorf("canary stats = %+v, want the owner's one label", s)
	}
	if n := len(ofType(sink.records(), "feedback")); n != 2 {
		t.Errorf("%d feedback lines in the decision log, want both written", n)
	}
}

func TestFeedbackForAnUnknownDecisionOnlyReachesTheLog(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	sink := attachLog(t, g.gw)
	g.decisionID(t)
	if code := g.feedback(keyA, "dec_"+strings.Repeat("a", 24), false); code != http.StatusNoContent {
		t.Fatalf("status %d, want 204", code)
	}
	if s := g.stats("canary"); s.Labels != 0 {
		t.Errorf("canary labels = %d for an unknown decision, want 0", s.Labels)
	}
	lines := ofType(sink.records(), "feedback")
	if len(lines) != 1 || lines[0]["decision_id"] != "dec_"+strings.Repeat("a", 24) {
		t.Errorf("feedback lines = %v, want the unknown decision's line", lines)
	}
}

func TestFeedbackCountsOncePerDecision(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	sink := attachLog(t, g.gw)
	id := g.decisionID(t)
	for i := 0; i < 5; i++ {
		g.feedback(keyA, id, false)
	}
	if s := g.stats("canary"); s.Labels != 1 {
		t.Errorf("canary labels = %d after five verdicts on one decision, want 1", s.Labels)
	}
	if n := len(ofType(sink.records(), "feedback")); n != 5 {
		t.Errorf("%d feedback lines, want all 5 written (the offline join keeps the latest)", n)
	}
}

func TestFeedbackExpiresAfterTenMinutes(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	attachLog(t, g.gw)
	old := g.decisionID(t)
	g.clock.advance(time.Minute)
	recent := g.decisionID(t)

	g.clock.advance(9*time.Minute + 59*time.Second) // old is 10m59s old, recent 9m59s
	if code := g.feedback(keyA, old, true); code != http.StatusNoContent {
		t.Fatalf("status %d", code)
	}
	if s := g.stats("canary"); s.Labels != 0 {
		t.Errorf("canary labels = %d for a decision older than ten minutes, want 0", s.Labels)
	}
	if code := g.feedback(keyA, recent, true); code != http.StatusNoContent {
		t.Fatalf("status %d", code)
	}
	if s := g.stats("canary"); s.Labels != 1 {
		t.Errorf("canary labels = %d for a decision under ten minutes old, want 1", s.Labels)
	}
}

func TestFeedbackForPinnedDecisionsIsNotIndexed(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	attachLog(t, g.gw)
	rec := g.do(keyA, chatBody("tiny-weak", "hi"), nil)
	id := rec.Header().Get(headerDecisionID)
	if rec.Code != http.StatusOK || id == "" {
		t.Fatalf("pinned: status %d decision id %q", rec.Code, id)
	}
	g.feedback(keyA, id, false)
	for _, role := range []string{"stable", "canary"} {
		if s := g.stats(role); s != (policy.GuardStats{}) {
			t.Errorf("%s stats = %+v after feedback on a pinned decision, want zero", role, s)
		}
	}
}

// Feedback needs the decision log (the route is a 404 without it), so a gateway without one has
// no use for the index and keeps nothing in it.
func TestDecisionIndexIsOnlyKeptWithADecisionLog(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	g.autoN(t, 3, http.StatusOK)
	if n := g.gw.index.size(); n != 0 {
		t.Errorf("index holds %d decisions without a decision log, want 0", n)
	}
	attachLog(t, g.gw)
	g.autoN(t, 3, http.StatusOK)
	if n := g.gw.index.size(); n != 3 {
		t.Errorf("index holds %d decisions with a decision log, want 3", n)
	}
}

// A request the router refused has a decision ID, but there is no answer for a client to rate:
// its refusal is already counted as an error, and a verdict on it would count it twice.
func TestFeedbackForARefusedDecisionIsNotIndexed(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: rejectCanary, canaryPct: 100}, nil)
	attachLog(t, g.gw)
	rec := g.auto()
	id := rec.Header().Get(headerDecisionID)
	if rec.Code != http.StatusUnprocessableEntity || id == "" {
		t.Fatalf("status %d decision id %q, want a refusal that still has one", rec.Code, id)
	}
	if code := g.feedback(keyA, id, false); code != http.StatusNoContent {
		t.Fatalf("feedback: status %d", code)
	}
	if s := g.stats("canary"); s.Labels != 0 || s.Requests != 1 || s.Errors != 1 {
		t.Errorf("canary stats = %+v, want the refusal as one error request and no label", s)
	}
}

func TestFeedbackToTheGuardCanTripARollback(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 100}, nil)
	attachLog(t, g.gw)
	g.seedStable(1000)
	for i := 0; i < 1000; i++ { // stable's clients are happy: 95 percent success
		ok := i%20 != 0
		g.gw.guard.Observe(policy.RoleStable, policy.Observation{Feedback: &ok})
	}
	for i := 0; i < 200; i++ { // the canary answers fine as far as the gateway sees, but 200 clients say no
		id := g.decisionID(t)
		g.feedback(keyA, id, i%2 == 0)
	}
	g.gw.EvaluateGuardrails()
	if pct := g.reg.Snapshot().CanaryPercent; pct != 0 {
		t.Errorf("canary percent = %v, want the feedback to roll the canary back", pct)
	}
	events := g.store.got()
	if len(events) != 1 || !strings.Contains(events[0].Reason, "feedback") {
		t.Errorf("events = %+v, want one auto_rollback for feedback", events)
	}
}

// ---- the decision index ----

func TestDecisionIndexBoundsAndExpiry(t *testing.T) {
	clock := &fakeClock{}
	idx := newDecisionIndex(clock.now)

	idx.add("dec_first", "t", "canary")
	if role, ok := idx.take("dec_first", "t"); !ok || role != "canary" {
		t.Fatalf("take = %q, %v, want canary", role, ok)
	}
	if _, ok := idx.take("dec_first", "t"); ok {
		t.Error("a decision was taken twice")
	}
	if _, ok := idx.take("dec_unknown", "t"); ok {
		t.Error("an unknown decision was found")
	}

	idx.add("dec_other", "t", "stable")
	if _, ok := idx.take("dec_other", "someone-else"); ok {
		t.Error("another tenant's verdict took the decision")
	}
	if _, ok := idx.take("dec_other", "t"); !ok {
		t.Error("the owner could not take the decision after a stranger tried")
	}

	// At most 100,000 entries: the oldest go first.
	for i := 0; i < decisionIndexMax+5; i++ {
		idx.add(idKey(i), "t", "stable")
	}
	if n := idx.size(); n > decisionIndexMax {
		t.Errorf("index holds %d entries, want at most %d", n, decisionIndexMax)
	}
	if _, ok := idx.take(idKey(0), "t"); ok {
		t.Error("the oldest entry survived the bound")
	}
	if _, ok := idx.take(idKey(decisionIndexMax+4), "t"); !ok {
		t.Error("the newest entry was lost")
	}

	// Entries older than ten minutes are dropped: one exactly ten minutes old is not older.
	idx = newDecisionIndex(clock.now)
	idx.add("dec_a", "t", "stable")
	idx.add("dec_b", "t", "stable")
	clock.advance(decisionIndexTTL)
	if _, ok := idx.take("dec_a", "t"); !ok {
		t.Error("an entry exactly ten minutes old was dropped")
	}
	clock.advance(time.Nanosecond)
	if _, ok := idx.take("dec_b", "t"); ok {
		t.Error("an entry older than ten minutes was found")
	}
	// They also go when something new comes in, so nobody has to look them up.
	idx.add("dec_c", "t", "canary")
	clock.advance(decisionIndexTTL + time.Nanosecond)
	idx.add("dec_d", "t", "canary")
	if n := idx.size(); n != 1 {
		t.Errorf("index holds %d entries after the old ones expired, want only the fresh one", n)
	}
}

func idKey(i int) string { return fmt.Sprintf("dec_%09d", i) }
