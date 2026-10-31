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
// block until its context ends or its unblock channel is closed, and notes what it saw of the
// context it was called with.
type recordingStore struct {
	mu       sync.Mutex
	rows     []accounting.UsageRow
	fail     error
	block    bool
	unblock  chan struct{} // ends a block when closed; nil blocks until the context ends
	ctxErrs  []error
	onRecord func()
}

func (r *recordingStore) RecordAttempt(ctx context.Context, row accounting.UsageRow) error {
	r.mu.Lock()
	r.ctxErrs = append(r.ctxErrs, ctx.Err())
	fail, block, hook := r.fail, r.block, r.onRecord
	r.mu.Unlock()
	if hook != nil {
		hook()
	}
	if block {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-r.unblock:
		}
	}
	if fail != nil {
		return fail
	}
	r.mu.Lock()
	r.rows = append(r.rows, row)
	r.mu.Unlock()
	return nil
}

// RecordAttempts takes a batch the way the batcher sends it: row by row through RecordAttempt,
// so a failing, blocking or recording store behaves the same for a batch as for a row. The
// first error ends the batch.
func (r *recordingStore) RecordAttempts(ctx context.Context, rows []accounting.UsageRow) error {
	for _, row := range rows {
		if err := r.RecordAttempt(ctx, row); err != nil {
			return err
		}
	}
	return nil
}

func (r *recordingStore) RecordRollout(context.Context, accounting.RolloutEvent) error { return nil }
func (r *recordingStore) Rollouts(context.Context, int) ([]accounting.RolloutEvent, error) {
	return nil, nil
}
func (r *recordingStore) TenantSpend(context.Context, string, time.Time, time.Time) (money.PicoUSD, error) {
	return 0, nil
}
func (r *recordingStore) TenantSpendInPeriod(context.Context, string, string) (money.PicoUSD, error) {
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
	useLedger(t, e.gw, store)

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusOK || modelOf(t, rec) != "pricey" {
		t.Fatalf("status %d model %q, want 200 from pricey: %s", rec.Code, modelOf(t, rec), rec.Body.String())
	}
	flushUsage(t, e.gw)
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
	useLedger(t, e.gw, store)

	do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("auto", "hi"))
	e.post(chatBody("no-such-model", "hi"))
	e.post(`{"model":`)

	flushUsage(t, e.gw)
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
		useLedger(t, e.gw, store)

		rec := e.post(streamBody("m"))

		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		flushUsage(t, e.gw)
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
		useLedger(t, e.gw, store)

		rec := e.post(streamBody("m"))

		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		flushUsage(t, e.gw)
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
		useLedger(t, e.gw, store)

		e.post(streamBody("m"))

		flushUsage(t, e.gw)
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
	useLedger(t, e.gw, store)

	rec := e.post(chatBody("auto", learnedPrompt), nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	flushUsage(t, e.gw)
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
	useLedger(t, gw, store)
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
	useLedger(t, gw, store)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamReqBody("model-a", false)))
	req.Header.Set("Authorization", "Bearer "+keyA)
	gw.Handler().ServeHTTP(&brokenWriter{ResponseRecorder: httptest.NewRecorder(), okWrites: 1}, req)

	flushUsage(t, gw)
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
	useLedger(t, e.gw, &recordingStore{fail: errors.New("disk is full")})
	const secret = "sentinel-prompt-do-not-log"

	rec := e.post(chatBody("auto", secret))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	flushUsage(t, e.gw)
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

// TestGatewayResponseDoesNotWaitForTheLedger: a store that never answers
// delays no response, because the rows are queued and the request returns at once. Nothing is
// lost while the queue has room, and the rows are written when the store recovers.
func TestGatewayResponseDoesNotWaitForTheLedger(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"m-a", "ua"}}}}})
	store := &recordingStore{block: true, unblock: make(chan struct{})}
	useLedger(t, e.gw, store)
	var release sync.Once
	t.Cleanup(func() { release.Do(func() { close(store.unblock) }) }) // runs before the batcher is closed

	const requests = 20 // at the old 250 ms bound these would take 5 s
	start := time.Now()
	for i := 0; i < requests; i++ {
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, rec.Code, rec.Body.String())
		}
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Errorf("%d requests took %v with a blocked ledger, want no wait for it", requests, took)
	}
	if n := e.gw.AccountingErrors(); n != 0 {
		t.Errorf("AccountingErrors = %d, want 0: the queue has room for these rows", n)
	}

	release.Do(func() { close(store.unblock) })
	flushUsage(t, e.gw)
	if got := len(store.got()); got != requests {
		t.Errorf("%d rows written after the store recovered, want %d", got, requests)
	}
}

// TestGatewayCountsRowsTheQueueHasNoRoomFor: a full queue loses the row, never fails the
// request, and the loss is counted in AccountingErrors and in the scraped counter.
func TestGatewayCountsRowsTheQueueHasNoRoomFor(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"m-a", "ua"}}}}})
	store := &recordingStore{}
	// Room for two rows and an hour before the writer would flush them: the third row is refused.
	useLedgerWith(t, e.gw, store, accounting.BatcherOptions{QueueRows: 2, MaxBatch: 1000, FlushEvery: time.Hour})

	for i := 0; i < 3; i++ {
		if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, rec.Code, rec.Body.String())
		}
	}

	if n := e.gw.AccountingErrors(); n != 1 {
		t.Errorf("AccountingErrors = %d, want 1: the row of the third request", n)
	}
	if got := scrapeOf(t, e.gw).value(t, "rp_accounting_write_errors_total", nil); got != 1 {
		t.Errorf("rp_accounting_write_errors_total = %v, want 1", got)
	}
	closeUsage(t, e.gw)
	if got := len(store.got()); got != 2 {
		t.Errorf("%d rows written, want the 2 that were queued", got)
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
	useLedger(t, e.gw, store)

	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	flushUsage(t, e.gw)
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
	useLedger(t, e.gw, store)

	for i := 0; i < 3; i++ { // each fails over from cheap (503, free) to pricey
		if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, rec.Code, rec.Body.String())
		}
	}
	if rec := e.post(chatBody("cheap", "hi")); rec.Code != http.StatusServiceUnavailable && rec.Code != http.StatusBadGateway {
		t.Fatalf("pinned cheap: status %d, want a 5xx", rec.Code)
	}
	flushUsage(t, e.gw)

	ledger, err := store.TenantSpend(context.Background(), "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if counter := e.spent(); ledger != counter || ledger == 0 {
		t.Errorf("ledger sum = %d, budget counter = %d, want them equal and non-zero", ledger, counter)
	}
}
