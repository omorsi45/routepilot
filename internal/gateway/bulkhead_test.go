package gateway

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/reliability"
	"github.com/omorsi45/routepilot/internal/router"
)

// gatedUpstream holds every non-streaming request for the upstream model hold until open is
// called or the request's context ends, and passes everything else to the mock. A test uses
// it to keep one attempt in flight on a deployment. A test that builds a gated environment
// registers t.Cleanup(gate.open) after newAttemptEnv, so a failing test never leaves a
// request held (the upstream server's Close would wait for it).
type gatedUpstream struct {
	hold    string
	release chan struct{}
	once    sync.Once
	next    http.Handler
}

func newGatedUpstream(hold string) *gatedUpstream {
	return &gatedUpstream{hold: hold, release: make(chan struct{}), next: mock.NewHandler(mock.Config{})}
}

// open lets every held request through, now and later.
func (g *gatedUpstream) open() { g.once.Do(func() { close(g.release) }) }

func (g *gatedUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(bytes.NewReader(b))
	if bytes.Contains(b, []byte(`"model":"`+g.hold+`"`)) && !bytes.Contains(b, []byte(`"stream":true`)) {
		select {
		case <-g.release:
		case <-r.Context().Done():
			return
		}
	}
	g.next.ServeHTTP(w, r)
}

// startPost sends body in the background and returns the channel its response arrives on.
func (e *attemptEnv) startPost(body string) <-chan *httptest.ResponseRecorder {
	out := make(chan *httptest.ResponseRecorder, 1)
	go func() { out <- e.post(body) }()
	return out
}

// receive waits for a response started with startPost and fails the test when none arrives
// within five seconds, so a regression shows as a failure and not as a hung test.
func receive(t *testing.T, ch <-chan *httptest.ResponseRecorder) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case rec := <-ch:
		return rec
	case <-time.After(5 * time.Second):
		t.Fatal("no response within 5s: the request is stuck behind a held upstream")
		return nil
	}
}

// bulkheadRejections is the value of rp_bulkhead_rejections_total for a deployment, 0 when
// the series does not exist yet.
func bulkheadRejections(t *testing.T, gw *Server, deployment string) float64 {
	t.Helper()
	got := scrapeOf(t, gw).match("rp_bulkhead_rejections_total", map[string]string{"deployment": deployment})
	switch len(got) {
	case 0:
		return 0
	case 1:
		return got[0].value
	}
	t.Fatalf("rp_bulkhead_rejections_total{deployment=%q} has %d samples, want at most 1", deployment, len(got))
	return 0
}

// TestBulkheadSkipsFullDeployment: a deployment at its max_inflight limit is skipped like an
// unavailable one. The skip is not an attempt, takes no breaker claim and is counted in
// rp_bulkhead_rejections_total{deployment}.
func TestBulkheadSkipsFullDeployment(t *testing.T) {
	t.Run("the request moves on to the next deployment", func(t *testing.T) {
		gate := newGatedUpstream("ua")
		e := newAttemptEnv(t, attemptOpts{
			upstream:    gate,
			maxInflight: map[string]int{"dep-a": 1},
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}, {"dep-b", "ub"}}}},
		})
		t.Cleanup(gate.open)
		first := e.startPost(chatBody("m", "one"))
		waitFor(t, 2*time.Second, "the first request to hold dep-a", func() bool { return e.count("ua") == 1 })
		if got := e.gw.bulkhead.inflight("dep-a"); got != 1 {
			t.Fatalf("dep-a in flight = %d, want 1", got)
		}

		rec := receive(t, e.startPost(chatBody("m", "two")))
		if rec.Code != http.StatusOK {
			t.Fatalf("second request: status %d: %s", rec.Code, rec.Body.String())
		}
		if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-b" {
			t.Errorf("deployment header %q, want dep-b (dep-a is full)", got)
		}
		if a, b := e.count("ua"), e.count("ub"); a != 1 || b != 1 {
			t.Errorf("upstream hits ua=%d ub=%d, want 1 each", a, b)
		}
		if !strings.Contains(e.logs.String(), "attempts=1") {
			t.Errorf("the skip counted as an attempt: %s", e.logs.String())
		}
		if got := bulkheadRejections(t, e.gw, "dep-a"); got != 1 {
			t.Errorf("rp_bulkhead_rejections_total{dep-a} = %v, want 1", got)
		}
		if got := bulkheadRejections(t, e.gw, "dep-b"); got != 0 {
			t.Errorf("rp_bulkhead_rejections_total{dep-b} = %v, want 0", got)
		}

		gate.open()
		if rec := receive(t, first); rec.Code != http.StatusOK || rec.Header().Get("x-routepilot-deployment") != "dep-a" {
			t.Fatalf("first request: status %d deployment %q", rec.Code, rec.Header().Get("x-routepilot-deployment"))
		}
		if got := e.gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("dep-a in flight after the first request = %d, want 0", got)
		}
	})

	t.Run("nothing left to try is a 503 and nothing is sent", func(t *testing.T) {
		gate := newGatedUpstream("ua")
		e := newAttemptEnv(t, attemptOpts{
			upstream:    gate,
			maxInflight: map[string]int{"dep-a": 1},
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
		})
		t.Cleanup(gate.open)
		first := e.startPost(chatBody("m", "one"))
		waitFor(t, 2*time.Second, "the first request to hold dep-a", func() bool { return e.count("ua") == 1 })

		rec := receive(t, e.startPost(chatBody("m", "two")))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status %d, want 503: %s", rec.Code, rec.Body.String())
		}
		if got := decodeErr(t, rec).Error.Code; got != "no_healthy_deployment" {
			t.Errorf("code %q, want no_healthy_deployment", got)
		}
		if n := e.count("ua"); n != 1 {
			t.Errorf("dep-a saw %d requests, want 1 (the second was never sent)", n)
		}
		if !strings.Contains(e.logs.String(), "attempts=0") {
			t.Errorf("log line missing attempts=0: %s", e.logs.String())
		}
		if got := bulkheadRejections(t, e.gw, "dep-a"); got != 1 {
			t.Errorf("rp_bulkhead_rejections_total{dep-a} = %v, want 1", got)
		}
		gate.open()
		receive(t, first)
	})

	t.Run("a stream is not limited", func(t *testing.T) {
		const up = "ua;tokens=2;ttft_ms=0;tpot_ms=0"
		gate := newGatedUpstream(up)
		e := newAttemptEnv(t, attemptOpts{
			upstream:    gate,
			maxInflight: map[string]int{"dep-a": 1},
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", up}}}},
		})
		t.Cleanup(gate.open)
		first := e.startPost(chatBody("m", "one"))
		waitFor(t, 2*time.Second, "the first request to hold dep-a", func() bool { return e.count(up) == 1 })

		rec := receive(t, e.startPost(streamBody("m")))
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatalf("stream: status %d: %s", rec.Code, rec.Body.String())
		}
		if got := rec.Header().Get("x-routepilot-deployment"); got != "dep-a" {
			t.Errorf("stream deployment %q, want dep-a (streams skip the bulkhead)", got)
		}
		if got := bulkheadRejections(t, e.gw, "dep-a"); got != 0 {
			t.Errorf("rp_bulkhead_rejections_total{dep-a} = %v, want 0 for a stream", got)
		}
		gate.open()
		receive(t, first)
	})
}

// TestBulkheadReleasedOnEveryPath: the slot is returned exactly once when the attempt's
// provider call returns, whatever way it ends.
func TestBulkheadReleasedOnEveryPath(t *testing.T) {
	limited := map[string]int{"dep-a": 1, "dep-b": 1}

	t.Run("success", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{
			maxInflight: limited,
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
		})
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		if got := e.gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after a success = %d, want 0", got)
		}
	})

	t.Run("retryable failure then failover", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{
			maxInflight: limited,
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub"}}}},
		})
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		for _, dep := range []string{"dep-a", "dep-b"} {
			if got := e.gw.bulkhead.inflight(dep); got != 0 {
				t.Errorf("%s in flight after the request = %d, want 0", dep, got)
			}
		}
		// Slots are free again: a second request reaches dep-a (still closed after one failure).
		if rec := e.post(chatBody("m", "again")); rec.Code != http.StatusOK {
			t.Fatalf("second request: status %d: %s", rec.Code, rec.Body.String())
		}
		if got := bulkheadRejections(t, e.gw, "dep-a"); got != 0 {
			t.Errorf("rp_bulkhead_rejections_total{dep-a} = %v, want 0 (nothing leaked)", got)
		}
	})

	t.Run("non-retryable failure", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{
			maxInflight: limited,
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http400"}}}},
		})
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusBadRequest {
			t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
		}
		if got := e.gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after a 400 = %d, want 0", got)
		}
	})

	t.Run("deadline expiry", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{
			maxInflight:      limited,
			defaultTimeoutMs: 100,
			models:           []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;fail=stall"}}}},
		})
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusGatewayTimeout {
			t.Fatalf("status %d, want 504: %s", rec.Code, rec.Body.String())
		}
		if got := e.gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after a deadline expiry = %d, want 0", got)
		}
	})

	t.Run("client cancel", func(t *testing.T) {
		gate := newGatedUpstream("ua")
		e := newAttemptEnv(t, attemptOpts{
			upstream:    gate,
			maxInflight: limited,
			models:      []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
		})
		t.Cleanup(gate.open)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("m", "hi"))).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer "+keyA)
		done := make(chan struct{})
		go func() {
			e.gw.Handler().ServeHTTP(httptest.NewRecorder(), req)
			close(done)
		}()
		waitFor(t, 2*time.Second, "the request to hold dep-a", func() bool { return e.gw.bulkhead.inflight("dep-a") == 1 })
		cancel()
		<-done
		if got := e.gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after a client cancel = %d, want 0", got)
		}
	})

	t.Run("provider panic", func(t *testing.T) {
		cfg := buildConfig(t, cfgOpts{})
		cfg.Models[0].Deployments[0].MaxInflight = 1
		p := &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			panic("provider bug")
		}}
		gw, err := New(cfg, map[string]provider.Provider{"mockp": p}, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Error("the panic did not reach the caller")
				}
			}()
			do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
		}()
		if got := gw.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after a provider panic = %d, want 0", got)
		}
	})
}

// bulkheadPlanFixture is a server whose model-a deployment dep-a holds one slot, and a
// decision that tries model-a then model-b. clock is the server's clock.
func bulkheadPlanFixture(t *testing.T, clock func() time.Time, maxAttempts int) (*Server, router.Decision) {
	t.Helper()
	cfg := buildConfig(t, cfgOpts{})
	cfg.Models[0].Deployments[0].MaxInflight = 1
	cfg.Reliability.MaxAttempts = maxAttempts
	s, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{Now: clock})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ma, _ := s.cat.Model("model-a")
	mb, _ := s.cat.Model("model-b")
	return s, router.Decision{Candidates: []router.Candidate{
		{Model: ma, Deployments: ma.Deployments},
		{Model: mb, Deployments: mb.Deployments},
	}}
}

// walkPlan drains a plan, releasing nothing, and lists the pairs it issued.
func walkPlan(plan *attemptPlan) string {
	var got []string
	for {
		m, d, _, _, ok := plan.next()
		if !ok {
			return strings.Join(got, ",")
		}
		got = append(got, m.ID+"/"+d.Name)
	}
}

func TestAttemptPlanBulkhead(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	advance := func(d time.Duration) { mu.Lock(); now = now.Add(d); mu.Unlock() }

	t.Run("a full deployment is skipped before Allow", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		release, ok := s.bulkhead.acquire("dep-a")
		if !ok {
			t.Fatal("acquire refused a fresh deployment")
		}
		defer release()
		// The cooldown has elapsed, so Allow would claim the half-open probe.
		s.health.Trip("dep-a", time.Second)
		advance(2 * time.Second)

		plan := s.newAttemptPlan(dec)
		plan.bulkhead = s.bulkhead
		if got, want := walkPlan(plan), "model-b/dep-b"; got != want {
			t.Errorf("pairs = %s, want %s", got, want)
		}
		if got := s.health.State("dep-a"); got != reliability.Open {
			t.Errorf("dep-a breaker = %v, want Open (a full deployment must not take the probe)", got)
		}
		if got := bulkheadRejections(t, s, "dep-a"); got != 1 {
			t.Errorf("rejections = %v, want 1", got)
		}
	})

	t.Run("the skip is not an attempt", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 1)
		release, ok := s.bulkhead.acquire("dep-a")
		if !ok {
			t.Fatal("acquire refused a fresh deployment")
		}
		defer release()
		plan := s.newAttemptPlan(dec)
		plan.bulkhead = s.bulkhead
		if got, want := walkPlan(plan), "model-b/dep-b"; got != want {
			t.Errorf("pairs with max_attempts 1 = %s, want %s", got, want)
		}
	})

	t.Run("a free deployment is issued with its slot and the slot is returned once", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		plan := s.newAttemptPlan(dec)
		plan.bulkhead = s.bulkhead
		m, d, _, h, ok := plan.next()
		if !ok || m.ID != "model-a" || d.Name != "dep-a" {
			t.Fatalf("first pair = %s/%s ok=%v, want model-a/dep-a", m.ID, d.Name, ok)
		}
		if got := s.bulkhead.inflight("dep-a"); got != 1 {
			t.Fatalf("in flight after next = %d, want 1", got)
		}
		if h.ticket.Deployment() != "dep-a" {
			t.Errorf("ticket deployment %q, want dep-a", h.ticket.Deployment())
		}
		h.releaseSlot()
		h.releaseSlot() // a second release must not return a slot it no longer holds
		if got := s.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("in flight after releasing twice = %d, want 0", got)
		}
		if _, _, _, _, ok := plan.next(); !ok {
			t.Fatal("plan stopped early")
		}
	})

	t.Run("the slot is returned when the breaker refuses", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		s.health.Trip("dep-a", time.Hour)
		plan := s.newAttemptPlan(dec)
		plan.bulkhead = s.bulkhead
		if got, want := walkPlan(plan), "model-b/dep-b"; got != want {
			t.Errorf("pairs = %s, want %s", got, want)
		}
		if got := s.bulkhead.inflight("dep-a"); got != 0 {
			t.Errorf("dep-a in flight = %d, want 0 (the breaker refused it)", got)
		}
	})

	t.Run("an open and full deployment is not a bulkhead rejection", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		release, _ := s.bulkhead.acquire("dep-a")
		defer release()
		s.health.Trip("dep-a", time.Hour)
		plan := s.newAttemptPlan(dec)
		plan.bulkhead = s.bulkhead
		walkPlan(plan)
		if got := bulkheadRejections(t, s, "dep-a"); got != 0 {
			t.Errorf("rejections = %v, want 0 (the breaker, not the bulkhead, excluded dep-a)", got)
		}
	})

	t.Run("more looks past a full deployment", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		onlyA := router.Decision{Candidates: dec.Candidates[:1]}
		release, _ := s.bulkhead.acquire("dep-a")
		plan := s.newAttemptPlan(onlyA)
		plan.bulkhead = s.bulkhead
		if plan.more() {
			t.Error("more = true with the only deployment full")
		}
		release()
		if !plan.more() {
			t.Error("more = false once the slot was returned")
		}
	})

	t.Run("a plan without a bulkhead ignores the limit", func(t *testing.T) {
		s, dec := bulkheadPlanFixture(t, clock, 5)
		release, _ := s.bulkhead.acquire("dep-a")
		defer release()
		plan := s.newAttemptPlan(dec)
		if got, want := walkPlan(plan), "model-a/dep-a,model-b/dep-b"; got != want {
			t.Errorf("pairs = %s, want %s (streams do not use the bulkhead)", got, want)
		}
	})
}

// TestBulkheadCap: the limit holds under concurrent claims, an unlisted deployment is
// unlimited, and releasing twice returns one slot.
func TestBulkheadCap(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	cfg.Models[0].Deployments[0].MaxInflight = 8
	s, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b := s.bulkhead

	var admitted atomic.Int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	releases := make(chan func(), 64)
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if rel, ok := b.acquire("dep-a"); ok {
				admitted.Add(1)
				releases <- rel
			}
		}()
	}
	close(start)
	wg.Wait()
	close(releases)
	if got := admitted.Load(); got != 8 {
		t.Fatalf("%d of 64 concurrent claimers got a slot, want exactly 8", got)
	}
	if got := b.inflight("dep-a"); got != 8 {
		t.Fatalf("in flight = %d, want 8", got)
	}
	if !b.full("dep-a") {
		t.Error("full = false at the limit")
	}
	for rel := range releases {
		rel()
		rel() // idempotent
	}
	if got := b.inflight("dep-a"); got != 0 {
		t.Errorf("in flight after releasing everything = %d, want 0", got)
	}
	if b.full("dep-a") {
		t.Error("full = true with every slot returned")
	}

	// dep-b has no limit: it always admits, holds nothing, and is never full.
	for range 100 {
		rel, ok := b.acquire("dep-b")
		if !ok {
			t.Fatal("an unlimited deployment refused a claim")
		}
		if rel != nil {
			t.Fatal("an unlimited deployment handed out a slot to return")
		}
	}
	if b.full("dep-b") || b.inflight("dep-b") != 0 {
		t.Errorf("dep-b full=%v inflight=%d, want unlimited", b.full("dep-b"), b.inflight("dep-b"))
	}
	var nilBulkhead *bulkhead
	if rel, ok := nilBulkhead.acquire("dep-a"); !ok || rel != nil || nilBulkhead.full("dep-a") || nilBulkhead.inflight("dep-a") != 0 {
		t.Error("a nil bulkhead must behave as unlimited")
	}
}
