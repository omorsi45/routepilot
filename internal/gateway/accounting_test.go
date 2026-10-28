package gateway

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// recordingStore is an accounting.Store that keeps the rows it is given. It can fail, can
// block until its context ends, and notes what it saw of the context it was called with.
type recordingStore struct {
	mu        sync.Mutex
	rows      []accounting.UsageRow
	fail      error
	block     bool
	ctxErrs   []error
	deadlines []time.Duration // time left to each call's deadline, 0 when it had none
	onRecord  func()
}

func (r *recordingStore) RecordAttempt(ctx context.Context, row accounting.UsageRow) error {
	r.mu.Lock()
	r.ctxErrs = append(r.ctxErrs, ctx.Err())
	var left time.Duration
	if d, ok := ctx.Deadline(); ok {
		left = time.Until(d)
	}
	r.deadlines = append(r.deadlines, left)
	fail, block, hook := r.fail, r.block, r.onRecord
	r.mu.Unlock()
	if hook != nil {
		hook()
	}
	if block {
		<-ctx.Done()
		return ctx.Err()
	}
	if fail != nil {
		return fail
	}
	r.mu.Lock()
	r.rows = append(r.rows, row)
	r.mu.Unlock()
	return nil
}

func (r *recordingStore) RecordRollout(context.Context, accounting.RolloutEvent) error { return nil }
func (r *recordingStore) Rollouts(context.Context, int) ([]accounting.RolloutEvent, error) {
	return nil, nil
}
func (r *recordingStore) TenantSpend(context.Context, string, time.Time, time.Time) (money.PicoUSD, error) {
	return 0, nil
}
func (r *recordingStore) Close() error { return nil }

func (r *recordingStore) got() []accounting.UsageRow {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]accounting.UsageRow(nil), r.rows...)
}

// steppingClock returns an instant 10 ms after the previous call, so every attempt has a
// start and an end that differ and order.
func steppingClock() func() time.Time {
	var n atomic.Int64
	t0 := time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)
	return func() time.Time { return t0.Add(time.Duration(n.Add(1)) * 10 * time.Millisecond) }
}

// TestGatewayWritesOneRowPerAttempt: a failover writes row 1 for the failed attempt and row 2
// for the one that answered, with the fields the ledger promises.
func TestGatewayWritesOneRowPerAttempt(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc"), clock: steppingClock(),
	})
	store := &recordingStore{}
	e.gw.usageLedger = store

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusOK || modelOf(t, rec) != "pricey" {
		t.Fatalf("status %d model %q, want 200 from pricey: %s", rec.Code, modelOf(t, rec), rec.Body.String())
	}
	rows := store.got()
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want one per attempt (2): %+v", len(rows), rows)
	}
	id := rec.Header().Get(requestIDHeader)
	cheapMax := worstCase(t, 100_000, 100_000, 1000)
	priceyMax := worstCase(t, 5_000_000, 5_000_000, 2000)

	want := []accounting.UsageRow{
		{
			RequestID: id, Attempt: 1, Tenant: "tenant-a", Model: "cheap", Deployment: "cheap-a",
			CostEstimate: cheapMax, PolicyVersion: "static-cheapest", Status: "http_503",
		},
		{
			RequestID: id, Attempt: 2, Tenant: "tenant-a", Model: "pricey", Deployment: "pricey-a",
			PromptTokens: 5, CompletionTokens: 4, CostEstimate: priceyMax, CostActual: 45_000_000,
			PolicyVersion: "static-cheapest", Status: "ok",
		},
	}
	for i, w := range want {
		g := rows[i]
		g.StartedAt, g.EndedAt = time.Time{}, time.Time{}
		if g != w {
			t.Errorf("row %d = %+v, want %+v", i+1, g, w)
		}
	}
	if !rows[0].StartedAt.Before(rows[0].EndedAt) || rows[0].EndedAt.After(rows[1].StartedAt) || !rows[1].StartedAt.Before(rows[1].EndedAt) {
		t.Errorf("attempt times out of order: %v..%v then %v..%v", rows[0].StartedAt, rows[0].EndedAt, rows[1].StartedAt, rows[1].EndedAt)
	}
}

// TestGatewayWritesNothingWithoutAnAttempt: a request that never reaches a provider (bad key,
// unknown model, an unaffordable budget) writes no row.
func TestGatewayWritesNothingWithoutAnAttempt(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest})
	store := &recordingStore{}
	e.gw.usageLedger = store

	do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("auto", "hi"))
	e.post(chatBody("no-such-model", "hi"))
	e.post(`{"model":`)

	if rows := store.got(); len(rows) != 0 {
		t.Errorf("rows = %+v, want none", rows)
	}
	if n := e.gw.AccountingErrors(); n != 0 {
		t.Errorf("AccountingErrors = %d, want 0", n)
	}
}

// TestGatewayRowsForStreams: a stream writes one row for the answering attempt with the
// provider's usage, a failover before the commit writes both attempts, and a stream that dies
// after the commit writes an estimated row that is not ok.
func TestGatewayRowsForStreams(t *testing.T) {
	t.Run("answered", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", in: "1", out: "2", deps: []depSpec{{"m-a", "ua;tokens=2"}}}}, clock: steppingClock()})
		store := &recordingStore{}
		e.gw.usageLedger = store

		rec := e.post(streamBody("m"))

		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		rows := store.got()
		if len(rows) != 1 {
			t.Fatalf("rows = %+v, want 1", rows)
		}
		r := rows[0]
		if r.Status != "ok" || r.Attempt != 1 || r.PromptTokens != 5 || r.CompletionTokens != 2 || r.UsageEstimated ||
			r.CostActual != 5*stdIn+2*stdOut || r.PolicyVersion != "pinned" {
			t.Errorf("row = %+v, want an exact ok row of 5 prompt and 2 completion tokens", r)
		}
		if !r.StartedAt.Before(r.EndedAt) {
			t.Errorf("StartedAt %v is not before EndedAt %v", r.StartedAt, r.EndedAt)
		}
	})

	t.Run("failover before the commit", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
			{id: "m", deps: []depSpec{{"dep-a", "ua;fail=http500"}, {"dep-b", "ub"}}},
		}})
		store := &recordingStore{}
		e.gw.usageLedger = store

		rec := e.post(streamBody("m"))

		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		rows := store.got()
		if len(rows) != 2 || rows[0].Deployment != "dep-a" || rows[0].Status != "http_500" || rows[1].Deployment != "dep-b" || rows[1].Status != "ok" {
			t.Fatalf("rows = %+v, want dep-a http_500 then dep-b ok", rows)
		}
		if rows[0].Attempt != 1 || rows[1].Attempt != 2 {
			t.Errorf("attempt numbers = %d, %d, want 1, 2", rows[0].Attempt, rows[1].Attempt)
		}
	})

	t.Run("dies after the commit", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", in: "1", out: "2", deps: []depSpec{{"m-a", "ua;fail=midstream;fail_after=2"}}}}})
		store := &recordingStore{}
		e.gw.usageLedger = store

		e.post(streamBody("m"))

		rows := store.got()
		if len(rows) != 1 {
			t.Fatalf("rows = %+v, want 1", rows)
		}
		if rows[0].Status == "ok" || rows[0].Status == "" || !rows[0].UsageEstimated || rows[0].CostActual <= 0 {
			t.Errorf("row = %+v, want a failed, estimated row with a charge", rows[0])
		}
	})
}

// TestGatewayRowCostEstimateIsTheLearnedExpectedCost: under the learned policy the estimate is
// the expected cost the response header reports, and the policy version is the slot's.
func TestGatewayRowCostEstimateIsTheLearnedExpectedCost(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{})
	store := &recordingStore{}
	e.gw.usageLedger = store

	rec := e.post(chatBody("auto", learnedPrompt), nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	rows := store.got()
	if len(rows) != 1 {
		t.Fatalf("rows = %+v, want 1", rows)
	}
	if got, want := rows[0].CostEstimate.USD(), rec.Header().Get(headerEstimatedCost); got != want {
		t.Errorf("CostEstimate = %s USD, want the header's expected cost %s", got, want)
	}
	if got, want := rows[0].PolicyVersion, rec.Header().Get(headerPolicyVersion); got != want || got == "" {
		t.Errorf("PolicyVersion = %q, want the response header's %q", got, want)
	}
	if rows[0].Model != rec.Header().Get("x-routepilot-model") {
		t.Errorf("row model %q, header model %q", rows[0].Model, rec.Header().Get("x-routepilot-model"))
	}
}

func TestAttemptStatus(t *testing.T) {
	for _, c := range []struct {
		name string
		err  error
		want string
	}{
		{"success", nil, "ok"},
		{"http 429", &provider.Error{Kind: provider.KindHTTP, Status: 429}, "http_429"},
		{"http 500", &provider.Error{Kind: provider.KindHTTP, Status: 500}, "http_500"},
		{"timeout", &provider.Error{Kind: provider.KindTimeout}, "timeout"},
		{"connect", &provider.Error{Kind: provider.KindConnect}, "connect"},
		{"protocol", &provider.Error{Kind: provider.KindProtocol}, "protocol"},
		{"canceled", &provider.Error{Kind: provider.KindCanceled}, "canceled"},
		{"wrapped", fmt.Errorf("attempt: %w", &provider.Error{Kind: provider.KindHTTP, Status: 502}), "http_502"},
		{"translation error", openai.ErrInternal(), "error"},
		{"plain error", errors.New("boom"), "error"},
	} {
		if got := attemptStatus(c.err); got != c.want {
			t.Errorf("%s: attemptStatus = %q, want %q", c.name, got, c.want)
		}
	}
}

// TestGatewayRowWhenTheClientLeaves: a client that cancels mid-stream leaves a canceled,
// estimated row, written even though the request context is done.
func TestGatewayRowWhenTheClientLeaves(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	p := &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
		return newGatedStream(make(chan struct{}), textChunk("hello"), textChunk(" world")), nil
	}}
	gw, _ := newFakeBudgetGateway(t, budget, p, nil)
	store := &recordingStore{}
	gw.usageLedger = store
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := startStream(t, ctx, srv.URL, streamReqBody("model-a", false))
	br := bufio.NewReader(resp.Body)
	for i := 0; i < 2; i++ {
		if ev, err := readEvent(br); err != nil || !strings.Contains(ev, `"content"`) {
			t.Fatalf("event %d = %q, %v, want a content chunk", i, ev, err)
		}
	}
	cancel()

	waitFor(t, 5*time.Second, "the canceled attempt's row", func() bool { return len(store.got()) == 1 })
	r := store.got()[0]
	if r.Status != "canceled" || !r.UsageEstimated || r.CompletionTokens != 11 || r.CostActual <= 0 {
		t.Errorf("row = %+v, want a canceled, estimated row of 11 streamed bytes", r)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.ctxErrs[0] != nil {
		t.Errorf("the write ran under a done context (%v), want one detached from the request", store.ctxErrs[0])
	}
}

// brokenWriter is a response writer whose body writes fail after the first n, as when the
// client's connection has dropped but the request context has not been canceled yet.
type brokenWriter struct {
	*httptest.ResponseRecorder
	okWrites int
}

func (b *brokenWriter) Write(p []byte) (int, error) {
	if b.okWrites <= 0 {
		return 0, errors.New("write: broken pipe")
	}
	b.okWrites--
	return b.ResponseRecorder.Write(p)
}

// TestGatewayRowWhenAClientWriteFails: a stream whose client stops accepting data is a
// canceled attempt, charged for what the upstream had streamed.
func TestGatewayRowWhenAClientWriteFails(t *testing.T) {
	p := &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
		return &fakeStream{chunks: []*openai.ChatCompletionChunk{textChunk("hello"), textChunk(" world"), textChunk("!")}, err: io.EOF}, nil
	}}
	gw, _ := newFakeBudgetGateway(t, 1_000_000_000, p, nil)
	store := &recordingStore{}
	gw.usageLedger = store

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamReqBody("model-a", false)))
	req.Header.Set("Authorization", "Bearer "+keyA)
	gw.Handler().ServeHTTP(&brokenWriter{ResponseRecorder: httptest.NewRecorder(), okWrites: 1}, req)

	rows := store.got()
	if len(rows) != 1 || rows[0].Status != "canceled" || !rows[0].UsageEstimated {
		t.Fatalf("rows = %+v, want one canceled, estimated row", rows)
	}
}

// TestGatewayLedgerFailureNeverFailsTheRequest: a store that errors costs the client nothing;
// the failures are counted and one warning is logged, with neither the prompt nor a key.
func TestGatewayLedgerFailureNeverFailsTheRequest(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc"),
	})
	e.gw.usageLedger = &recordingStore{fail: errors.New("disk is full")}
	const secret = "sentinel-prompt-do-not-log"

	rec := e.post(chatBody("auto", secret))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if n := e.gw.AccountingErrors(); n != 2 {
		t.Errorf("AccountingErrors = %d, want 2, one per row that was lost", n)
	}
	logs := e.logs.String()
	if got := strings.Count(logs, "usage ledger"); got != 1 {
		t.Errorf("%d usage ledger warnings in the log, want exactly 1:\n%s", got, logs)
	}
	for _, bad := range []string{secret, keyA} {
		if strings.Contains(logs, bad) {
			t.Errorf("log contains %q:\n%s", bad, logs)
		}
	}
	if !strings.Contains(logs, "disk is full") || !strings.Contains(logs, "level=WARN") {
		t.Errorf("log should carry the store's error as a warning:\n%s", logs)
	}
}

// TestGatewayLedgerWriteIsBoundedTo250ms: a store that never answers delays the response by
// at most the bound, and the lost row is counted.
func TestGatewayLedgerWriteIsBoundedTo250ms(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"m-a", "ua"}}}}})
	store := &recordingStore{block: true}
	e.gw.usageLedger = store

	start := time.Now()
	rec := e.post(chatBody("m", "hi"))
	took := time.Since(start)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if took < 200*time.Millisecond || took > 3*time.Second {
		t.Errorf("request took %v with a blocked ledger, want about the 250ms bound", took)
	}
	if n := e.gw.AccountingErrors(); n != 1 {
		t.Errorf("AccountingErrors = %d, want 1", n)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.deadlines) != 1 || store.deadlines[0] <= 0 || store.deadlines[0] > 250*time.Millisecond {
		t.Errorf("deadline left at the call = %v, want a deadline of at most 250ms", store.deadlines)
	}
}

// TestGatewayRowsAreWrittenAfterTheBudgetCommit: when the first row is written, the budget
// counter already holds the request's cost, so the two cannot disagree for a moment that a
// reconcile run could see.
func TestGatewayRowsAreWrittenAfterTheBudgetCommit(t *testing.T) {
	e := newBudgetEnv(t, budgetOpts{budget: 1_000_000_000, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	store := &recordingStore{}
	var spentAtWrite money.PicoUSD
	store.onRecord = func() {
		spentAtWrite, _ = e.inner.Spent(context.Background(), "tenant-a", e.lim, e.now())
	}
	e.gw.usageLedger = store

	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if spentAtWrite != stdActual {
		t.Errorf("budget counter at the time of the write = %d, want the committed %d", spentAtWrite, stdActual)
	}
}

// TestLedgerSumMatchesTheBudgetCounter is the property reconcile relies on: over requests
// that succeed, fail over and fail, the ledger's sum for the tenant equals the budget counter.
func TestLedgerSumMatchesTheBudgetCounter(t *testing.T) {
	f := newFailoverBudget(t)
	e := newBudgetEnv(t, budgetOpts{
		budget: f.budget * 10, policy: config.PolicyCheapest, models: failoverModels(), upstream: upstream503For("uc"),
	})
	store, err := accounting.OpenSQLite(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	e.gw.usageLedger = store

	for i := 0; i < 3; i++ { // each fails over from cheap (503, free) to pricey
		if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, rec.Code, rec.Body.String())
		}
	}
	if rec := e.post(chatBody("cheap", "hi")); rec.Code != http.StatusServiceUnavailable && rec.Code != http.StatusBadGateway {
		t.Fatalf("pinned cheap: status %d, want a 5xx", rec.Code)
	}

	ledger, err := store.TenantSpend(context.Background(), "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if counter := e.spent(); ledger != counter || ledger == 0 {
		t.Errorf("ledger sum = %d, budget counter = %d, want them equal and non-zero", ledger, counter)
	}
}
