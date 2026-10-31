package gateway

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/provider/openaicompat"
	"github.com/omorsi45/routepilot/internal/quota"
)

// budgetNow is the fixed gateway clock of the budget tests, so a monthly period never
// changes under a test.
var budgetNow = time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)

// Prices of the standard model "m": 1 USD per million input tokens (1,000,000 picodollars
// per token) and 2 USD per million output tokens. The mock reports 5 prompt tokens for the
// one-message "hi" request and, unless a test says otherwise, 4 completion tokens.
const (
	stdIn  = money.PicoUSD(1_000_000)
	stdOut = money.PicoUSD(2_000_000)
	// stdActual is what a mock answer to "hi" costs on the standard model: 5 prompt tokens
	// and 4 completion tokens.
	stdActual = 5*stdIn + 4*stdOut
)

// countingLedger counts the calls the gateway makes, can fail them, and can run a hook
// after a successful Reserve or at the start of every upstream call.
type countingLedger struct {
	quota.BudgetLedger

	mu                                         sync.Mutex
	reserves, commits, releases, remainingCall int
	committed                                  []money.PicoUSD
	failRemaining, failReserve                 error
	// respectCtx makes every call fail once its context is done, as a networked ledger does.
	respectCtx    bool
	beforeReserve func()
	afterReserve  func(id string, amount money.PicoUSD)
}

func (c *countingLedger) Remaining(ctx context.Context, tenant string, lim quota.BudgetLimit, now time.Time) (money.PicoUSD, error) {
	c.mu.Lock()
	c.remainingCall++
	fail := c.failRemaining
	c.mu.Unlock()
	if fail != nil {
		return 0, fail
	}
	if err := c.ctxErr(ctx); err != nil {
		return 0, err
	}
	return c.BudgetLedger.Remaining(ctx, tenant, lim, now)
}

func (c *countingLedger) Reserve(ctx context.Context, tenant, id string, lim quota.BudgetLimit, amount money.PicoUSD, ttl time.Duration, now time.Time) (bool, money.PicoUSD, error) {
	c.mu.Lock()
	c.reserves++
	fail, before, hook := c.failReserve, c.beforeReserve, c.afterReserve
	c.mu.Unlock()
	if fail != nil {
		return false, 0, fail
	}
	if before != nil {
		before()
	}
	if err := c.ctxErr(ctx); err != nil {
		return false, 0, err
	}
	ok, rem, err := c.BudgetLedger.Reserve(ctx, tenant, id, lim, amount, ttl, now)
	if ok && err == nil && hook != nil {
		hook(id, amount)
	}
	return ok, rem, err
}

func (c *countingLedger) Commit(ctx context.Context, tenant, id string, actual money.PicoUSD, now time.Time) (money.PicoUSD, error) {
	c.mu.Lock()
	c.commits++
	c.committed = append(c.committed, actual)
	c.mu.Unlock()
	if err := c.ctxErr(ctx); err != nil {
		return 0, err
	}
	return c.BudgetLedger.Commit(ctx, tenant, id, actual, now)
}

func (c *countingLedger) Release(ctx context.Context, tenant, id string) error {
	c.mu.Lock()
	c.releases++
	c.mu.Unlock()
	if err := c.ctxErr(ctx); err != nil {
		return err
	}
	return c.BudgetLedger.Release(ctx, tenant, id)
}

func (c *countingLedger) ctxErr(ctx context.Context) error {
	if c.respectCtx {
		return ctx.Err()
	}
	return nil
}

func (c *countingLedger) counts() (reserves, commits, releases, remaining int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.reserves, c.commits, c.releases, c.remainingCall
}

type budgetOpts struct {
	budget      money.PicoUSD
	period      string
	models      []modelSpec
	policy      string
	routingYAML string // overrides policy: the whole routing section
	upstream    http.Handler
	inner       quota.BudgetLedger // nil: memory
	maxAttempts int
	ttftMs      int
	mockTokens  int // completion tokens of a mock answer; 0 means 4
	now         func() time.Time
}

// budgetEnv is a gateway over a mock upstream for a tenant with a budget (tenant-a) and
// one without (tenant-b).
type budgetEnv struct {
	t      *testing.T
	gw     *Server
	inner  quota.BudgetLedger
	ledger *countingLedger
	lim    quota.BudgetLimit
	now    func() time.Time
	logs   *syncBuffer

	mu         sync.Mutex
	hits       map[string]int
	maxTokens  map[string][]string
	onUpstream func(model string)
}

func newBudgetEnv(t *testing.T, o budgetOpts) *budgetEnv {
	t.Helper()
	e := &budgetEnv{t: t, logs: &syncBuffer{}, hits: map[string]int{}, maxTokens: map[string][]string{}, now: o.now}
	if e.now == nil {
		e.now = func() time.Time { return budgetNow }
	}
	next := o.upstream
	if next == nil {
		tokens := o.mockTokens
		if tokens == 0 {
			tokens = 4
		}
		next = mock.NewHandler(mock.Config{OutputTokens: tokens})
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(b))
		var body struct {
			Model     string          `json:"model"`
			MaxTokens json.RawMessage `json:"max_tokens"`
		}
		_ = json.Unmarshal(b, &body)
		e.mu.Lock()
		e.hits[body.Model]++
		e.maxTokens[body.Model] = append(e.maxTokens[body.Model], string(body.MaxTokens))
		hook := e.onUpstream
		e.mu.Unlock()
		if hook != nil {
			hook(body.Model)
		}
		next.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	period := o.period
	if period == "" {
		period = quota.PeriodMonthly
	}
	e.lim = quota.BudgetLimit{Limit: o.budget, Period: period}

	var sb strings.Builder
	fmt.Fprintf(&sb, "providers:\n  - name: mockp\n    type: openaicompat\n    base_url: %q\n", srv.URL+"/v1")
	if o.ttftMs > 0 {
		fmt.Fprintf(&sb, "    ttft_timeout_ms: %d\n", o.ttftMs)
	}
	if o.maxAttempts > 0 {
		fmt.Fprintf(&sb, "reliability:\n  max_attempts: %d\n", o.maxAttempts)
	}
	switch {
	case o.routingYAML != "":
		sb.WriteString(o.routingYAML)
	case o.policy != "":
		fmt.Fprintf(&sb, "routing:\n  policy: %s\n", o.policy)
	}
	sb.WriteString("models:\n")
	for _, m := range o.models {
		fmt.Fprintf(&sb, "  - id: %s\n    rank: %d\n", m.id, m.rank)
		if m.in != "" {
			fmt.Fprintf(&sb, "    input_price_per_mtok: %q\n    output_price_per_mtok: %q\n", m.in, m.out)
		}
		if m.maxOut > 0 {
			fmt.Fprintf(&sb, "    max_output_tokens: %d\n", m.maxOut)
		}
		sb.WriteString("    deployments:\n")
		for _, d := range m.deps {
			fmt.Fprintf(&sb, "      - name: %s\n        provider: mockp\n        upstream_model: %q\n", d.name, d.upstream)
		}
	}
	fmt.Fprintf(&sb, "tenants:\n  - id: tenant-a\n    api_key_sha256: %s\n    allowed_models: [\"*\"]\n    budget_usd: %q\n    budget_period: %s\n",
		keyHash(keyA), o.budget.USD(), period)
	fmt.Fprintf(&sb, "  - id: tenant-b\n    api_key_sha256: %s\n    allowed_models: [\"*\"]\n", keyHash(keyB))
	cfg, err := config.Parse([]byte(sb.String()))
	if err != nil {
		t.Fatalf("parse config: %v\n%s", err, sb.String())
	}

	e.inner = o.inner
	if e.inner == nil {
		e.inner = quota.NewMemoryBudget()
	}
	e.ledger = &countingLedger{BudgetLedger: e.inner}
	prov := openaicompat.New(cfg.Providers[0], "", srv.Client())
	e.gw, err = New(cfg, map[string]provider.Provider{"mockp": prov}, Options{
		Logger:   slog.New(slog.NewTextHandler(e.logs, nil)),
		Now:      e.now,
		Rand:     func() float64 { return 1 },
		Sleep:    func(context.Context, time.Duration) error { return nil },
		Limiters: quota.Limiters{Budget: e.ledger},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func (e *budgetEnv) post(body string) *httptest.ResponseRecorder {
	return do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
}

func (e *budgetEnv) postWith(body string, headers map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+keyA)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	e.gw.Handler().ServeHTTP(rec, req)
	return rec
}

func (e *budgetEnv) hit(model string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.hits[model]
}

func (e *budgetEnv) caps(model string) []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.maxTokens[model]...)
}

func (e *budgetEnv) setOnUpstream(f func(model string)) {
	e.mu.Lock()
	e.onUpstream = f
	e.mu.Unlock()
}

func (e *budgetEnv) remaining() money.PicoUSD {
	e.t.Helper()
	r, err := e.inner.Remaining(context.Background(), "tenant-a", e.lim, e.now())
	if err != nil {
		e.t.Fatalf("Remaining: %v", err)
	}
	return r
}

func (e *budgetEnv) spent() money.PicoUSD {
	e.t.Helper()
	s, err := e.inner.Spent(context.Background(), "tenant-a", e.lim, e.now())
	if err != nil {
		e.t.Fatalf("Spent: %v", err)
	}
	return s
}

// squat reserves amount for a request that is not in the gateway, as another instance
// would, so that what is left for the next raise is known exactly.
func (e *budgetEnv) squat(amount money.PicoUSD) {
	ok, _, err := e.inner.Reserve(context.Background(), "tenant-a", "squatter", e.lim, amount, time.Hour, e.now())
	if err != nil || !ok {
		e.t.Errorf("squatter reserve %d: ok=%v err=%v", amount, ok, err)
	}
}

func stdModel(upstream string) modelSpec {
	return modelSpec{id: "m", in: "1", out: "2", maxOut: 100, deps: []depSpec{{"m-a", upstream}}}
}

// worstCase is the CostMax of the standard "hi" request on a model with these prices.
func worstCase(t *testing.T, in, out money.PicoUSD, k int) money.PicoUSD {
	t.Helper()
	return in*money.PicoUSD(inUpperHi(t)) + out*money.PicoUSD(k)
}

func wantInsufficientQuota(t *testing.T, rec *httptest.ResponseRecorder, message string) {
	t.Helper()
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429: %s", rec.Code, rec.Body.String())
	}
	e := decodeErr(t, rec).Error
	if e.Code != "insufficient_quota" || e.Type != "insufficient_quota" {
		t.Errorf("error = %q / %q, want insufficient_quota", e.Code, e.Type)
	}
	if message != "" && e.Message != message {
		t.Errorf("message = %q, want %q", e.Message, message)
	}
}

// TestBudgetRejectsWhenNothingAffordable: a budget below the cheapest model's input plus one
// output token is a 429 for "auto" and for a named model, and the ledger is never written.
func TestBudgetRejectsWhenNothingAffordable(t *testing.T) {
	oneToken := worstCase(t, stdIn, stdOut, 1)
	e := newBudgetEnv(t, budgetOpts{budget: oneToken - 1, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})

	for _, model := range []string{"auto", "m"} {
		rec := e.post(chatBody(model, "hi"))
		wantInsufficientQuota(t, rec, "budget remaining below the cheapest worst-case cost")
		if rec.Header().Get("x-routepilot-decision-id") == "" {
			t.Errorf("%s: no decision id header on the 429", model)
		}
	}
	if n := e.hit("ua"); n != 0 {
		t.Errorf("upstream called %d times for an unaffordable request", n)
	}
	if res, com, rel, _ := e.ledger.counts(); res != 0 || com != 0 || rel != 0 {
		t.Errorf("ledger writes reserve=%d commit=%d release=%d, want none", res, com, rel)
	}
	if got := e.remaining(); got != oneToken-1 {
		t.Errorf("Remaining = %d, want the untouched budget %d", got, oneToken-1)
	}

	// One more picodollar buys exactly one output token.
	ok := newBudgetEnv(t, budgetOpts{budget: oneToken, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	if rec := ok.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("one output token of budget: status %d: %s", rec.Code, rec.Body.String())
	}
	if got := ok.caps("ua"); len(got) != 1 || got[0] != "1" {
		t.Errorf("upstream max_tokens = %v, want [1]", got)
	}
}

// TestAffordableKShrinksImplicitMaxTokens: without max_tokens the model is admitted, costed,
// reserved and called with the output budget the remaining budget pays for.
func TestAffordableKShrinksImplicitMaxTokens(t *testing.T) {
	const k = 300
	budget := worstCase(t, stdIn, stdOut, k) + stdOut - 1 // 300 tokens and part of a 301st
	for _, model := range []string{"auto", "m"} {
		t.Run(model, func(t *testing.T) {
			spec := stdModel("ua")
			spec.maxOut = 1000
			e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{spec}})
			var during money.PicoUSD
			e.setOnUpstream(func(string) { during, _ = e.inner.Remaining(context.Background(), "tenant-a", e.lim, e.now()) })

			rec := e.post(chatBody(model, "hi"))

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			if got := e.caps("ua"); len(got) != 1 || got[0] != fmt.Sprint(k) {
				t.Errorf("upstream max_tokens = %v, want [%d]", got, k)
			}
			if want := budget - worstCase(t, stdIn, stdOut, k); during != want {
				t.Errorf("Remaining during the call = %d, want %d: the reservation is the CostMax of K=%d", during, want, k)
			}
			if got := e.spent(); got != stdActual {
				t.Errorf("Spent = %d, want the actual usage %d", got, stdActual)
			}
		})
	}
}

// TestExplicitMaxTokensIsKeptOrRejected: a client cap is never shrunk to fit the budget.
func TestExplicitMaxTokensIsKeptOrRejected(t *testing.T) {
	spec := stdModel("ua")
	spec.maxOut = 1000
	body := func(model string) string { return bodyWithMaxTokens(model, 100, false) }
	cost := worstCase(t, stdIn, stdOut, 100)

	e := newBudgetEnv(t, budgetOpts{budget: cost - 1, policy: config.PolicyCheapest, models: []modelSpec{spec}})
	wantInsufficientQuota(t, e.post(body("auto")), "")
	wantInsufficientQuota(t, e.post(body("m")), "")

	e = newBudgetEnv(t, budgetOpts{budget: cost, policy: config.PolicyCheapest, models: []modelSpec{spec}})
	if rec := e.post(body("auto")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := e.caps("ua"); len(got) != 1 || got[0] != "100" {
		t.Errorf("upstream max_tokens = %v, want the client's [100]", got)
	}
}

// failoverBudget is a cheap model whose upstream always answers 503 and a pricey one that
// works. budget leaves room for the pricey model's worst case at the start.
type failoverBudget struct {
	cheapMax, priceyMax, budget money.PicoUSD
}

func newFailoverBudget(t *testing.T) failoverBudget {
	t.Helper()
	f := failoverBudget{
		cheapMax:  worstCase(t, 100_000, 100_000, 1000),
		priceyMax: worstCase(t, 5_000_000, 5_000_000, 2000),
	}
	f.budget = f.priceyMax + 900_000_000 // room for both at the start
	return f
}

// upstream503For answers 503 for the listed upstream models and passes everything else on.
func upstream503For(models ...string) http.Handler {
	ok := mock.NewHandler(mock.Config{OutputTokens: 4})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(b))
		for _, m := range models {
			if strings.Contains(string(b), `"model":"`+m+`"`) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = io.WriteString(w, `{"error":{"message":"overloaded","type":"server_error","code":"overloaded"}}`)
				return
			}
		}
		ok.ServeHTTP(w, r)
	})
}

func failoverModels() []modelSpec {
	return []modelSpec{
		{id: "cheap", in: "0.1", out: "0.1", maxOut: 1000, rank: 1, deps: []depSpec{{"cheap-a", "uc"}}},
		{id: "pricey", in: "5", out: "5", maxOut: 2000, rank: 3, deps: []depSpec{{"pricey-a", "up"}}},
	}
}

// TestFallbackReReserveFailureSkipsModel: the cheap model fails, and by then the rest of the
// budget is held by someone else, so raising the reservation to the pricey fallback's worst
// case fails and the fallback is never called.
func TestFallbackReReserveFailureSkipsModel(t *testing.T) {
	f := newFailoverBudget(t)
	e := newBudgetEnv(t, budgetOpts{budget: f.budget, policy: config.PolicyCheapest, models: failoverModels(), upstream: upstream503For("uc")})
	e.setOnUpstream(func(model string) {
		if model == "uc" {
			e.squat(f.budget - f.cheapMax - 1_000_000) // leaves 1,000,000 free
		}
	})

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502 from the cheap model's failure: %s", rec.Code, rec.Body.String())
	}
	if c, p := e.hit("uc"), e.hit("up"); c != 1 || p != 0 {
		t.Errorf("upstream hits cheap=%d pricey=%d, want 1 and 0: the fallback must never be called", c, p)
	}
	if got := e.spent(); got != 0 {
		t.Errorf("Spent = %d, want 0: a 503 is not billed", got)
	}
	// The request's own reservation is gone; the squatter still holds the rest.
	if got, want := e.remaining(), f.cheapMax+1_000_000; got != want {
		t.Errorf("Remaining = %d, want %d: everything but the squatter's %d is free again", got, want, f.budget-f.cheapMax-1_000_000)
	}
}

// TestFallbackReReservesWhenItFits: with room, the reservation is raised to the fallback's
// worst case before it is called, and the commit is the fallback's actual usage.
func TestFallbackReReservesWhenItFits(t *testing.T) {
	f := newFailoverBudget(t)
	e := newBudgetEnv(t, budgetOpts{budget: f.budget, policy: config.PolicyCheapest, models: failoverModels(), upstream: upstream503For("uc")})
	var cheapSaw, priceySaw money.PicoUSD
	e.setOnUpstream(func(model string) {
		r, _ := e.inner.Remaining(context.Background(), "tenant-a", e.lim, e.now())
		switch model {
		case "uc":
			cheapSaw = r
		case "up":
			priceySaw = r
		}
	})

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusOK || modelOf(t, rec) != "pricey" {
		t.Fatalf("status %d model %q, want 200 from pricey: %s", rec.Code, modelOf(t, rec), rec.Body.String())
	}
	if want := f.budget - f.cheapMax; cheapSaw != want {
		t.Errorf("Remaining during the cheap attempt = %d, want %d", cheapSaw, want)
	}
	if want := f.budget - f.priceyMax; priceySaw != want {
		t.Errorf("Remaining during the pricey attempt = %d, want %d: raised to its worst case", priceySaw, want)
	}
	if want := 5*money.PicoUSD(5_000_000) + 4*5_000_000; e.spent() != want {
		t.Errorf("Spent = %d, want %d (the pricey model's actual usage; the 503 cost nothing)", e.spent(), want)
	}
}

// TestFallbackSkipContinuesWithTheNextCandidate: a skipped candidate is not an attempt, and
// the walk goes on to a later candidate that fits the current reservation.
func TestFallbackSkipContinuesWithTheNextCandidate(t *testing.T) {
	f := newFailoverBudget(t)
	models := []modelSpec{
		{id: "cheap1", in: "0.1", out: "0.1", maxOut: 1000, rank: 2, deps: []depSpec{{"cheap1-a", "uc1"}}},
		{id: "pricey", in: "5", out: "5", maxOut: 2000, rank: 3, deps: []depSpec{{"pricey-a", "up"}}},
		{id: "cheap2", in: "0.1", out: "0.1", maxOut: 1000, rank: 1, deps: []depSpec{{"cheap2-a", "uc2"}}},
	}
	// The rule puts cheap1 first, the default model pricey second, then the rest by rank.
	routing := "routing:\n  policy: rules\n  default_model: pricey\n  rules:\n    - name: first\n      model: cheap1\n      when: {min_prompt_bytes: 1}\n"
	e := newBudgetEnv(t, budgetOpts{budget: f.budget, routingYAML: routing, models: models, upstream: upstream503For("uc1")})
	e.setOnUpstream(func(model string) {
		if model == "uc1" {
			e.squat(f.budget - f.cheapMax - 1_000_000)
		}
	})

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusOK || modelOf(t, rec) != "cheap2" {
		t.Fatalf("status %d model %q, want 200 from cheap2: %s", rec.Code, modelOf(t, rec), rec.Body.String())
	}
	if e.hit("up") != 0 || e.hit("uc1") != 1 || e.hit("uc2") != 1 {
		t.Errorf("hits cheap1=%d pricey=%d cheap2=%d, want 1, 0, 1", e.hit("uc1"), e.hit("up"), e.hit("uc2"))
	}
	if want := 5*money.PicoUSD(100_000) + 4*100_000; e.spent() != want {
		t.Errorf("Spent = %d, want %d (cheap2's usage)", e.spent(), want)
	}
	if got := e.gw.BudgetOvershoot(); got != 0 {
		t.Errorf("BudgetOvershoot = %d, want 0", got)
	}
}

// TestSecondAttemptReservesForWhatTheFirstAlreadyCost: a first attempt that is billed (it
// timed out) and a second attempt that could use its whole worst case together exceed the
// first attempt's reservation, so the reservation is raised by the first attempt's charge
// too, and the second attempt is skipped when that raise fails.
func TestSecondAttemptReservesForWhatTheFirstAlreadyCost(t *testing.T) {
	models := []modelSpec{
		{id: "m1", in: "1", out: "2", maxOut: 10, rank: 2, deps: []depSpec{{"m1-a", "u1;fail=stall"}}},
		{id: "m2", in: "1", out: "2", maxOut: 10, rank: 1, deps: []depSpec{{"m2-a", "u2"}}},
	}
	each := worstCase(t, stdIn, stdOut, 10) // the worst case of either model
	inputOnly := worstCase(t, stdIn, stdOut, 0)
	// After the first reservation and the squatter, 5,000,000 are free: not enough for the
	// 26,000,000 the first attempt's charge adds to the second attempt's worst case.
	budget := 4 * each
	e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyStrongest, models: models, ttftMs: 100})
	e.setOnUpstream(func(model string) {
		if model == "u1;fail=stall" {
			e.squat(budget - each - 5_000_000)
		}
	})

	rec := e.post(streamBody("auto"))

	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504 after the first attempt's TTFT timeout: %s", rec.Code, rec.Body.String())
	}
	if e.hit("u2") != 0 {
		t.Errorf("m2 was called %d times, want 0: its raise to %d did not fit", e.hit("u2"), each+inputOnly)
	}
	if got := e.spent(); got != inputOnly {
		t.Errorf("Spent = %d, want %d: a timed-out attempt is charged its input bound", got, inputOnly)
	}
}

// sameModelFailover is one model with two deployments whose first fails in a way that is
// billed, so the second deployment is a failover that follows a charged attempt. The
// upstream model strings are the mock directives; the first one stalls for the stream case.
type sameModelFailover struct {
	name    string
	firstUp string // upstream model string of the first deployment
	// upstream builds the upstream handler for a mock answer of this many completion tokens;
	// nil keeps the standard mock.
	upstream   func(tokens int) http.Handler
	body       string
	ttftMs     int
	wantStatus int // when the raise does not fit and the second deployment is skipped
}

// inBandErrorFirst answers the upstream model "u1" with an error object in a 200, which the
// adapter reports as a protocol failure (billed its input bound), and everyone else with the
// mock.
func inBandErrorFirst(tokens int) http.Handler {
	next := mock.NewHandler(mock.Config{OutputTokens: tokens})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(b))
		var body struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(b, &body)
		if body.Model == "u1" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"error":{"message":"engine failed"}}`)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func sameModelFailoverCases() []sameModelFailover {
	return []sameModelFailover{
		{name: "a stream that stalls", firstUp: "u1;fail=stall", body: streamBody("auto"), ttftMs: 100, wantStatus: http.StatusGatewayTimeout},
		{name: "an error object in a 200", firstUp: "u1", upstream: inBandErrorFirst, body: chatBody("auto", "hi"), wantStatus: http.StatusBadGateway},
	}
}

// TestSameModelFailoverReservesForWhatTheFirstAlreadyCost: a failover to another deployment
// of the same model is a further attempt like any other. When the first attempt is billed,
// the reservation has to be raised by that charge first, and the second deployment is not
// called when the raise does not fit, so committed spend stays inside the budget.
func TestSameModelFailoverReservesForWhatTheFirstAlreadyCost(t *testing.T) {
	each := worstCase(t, stdIn, stdOut, 10)
	inputOnly := worstCase(t, stdIn, stdOut, 0)
	for _, tc := range sameModelFailoverCases() {
		t.Run(tc.name, func(t *testing.T) {
			models := []modelSpec{{id: "m", in: "1", out: "2", maxOut: 10, deps: []depSpec{{"m-a", tc.firstUp}, {"m-b", "u2"}}}}
			// The budget is exactly one worst case: nothing is left for the first attempt's
			// charge on top of the second attempt's worst case.
			o := budgetOpts{budget: each, policy: config.PolicyCheapest, models: models, ttftMs: tc.ttftMs, mockTokens: 1000}
			if tc.upstream != nil {
				o.upstream = tc.upstream(1000)
			}
			e := newBudgetEnv(t, o)

			rec := e.post(tc.body)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status %d, want %d after the first attempt failed and the second was skipped: %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if e.hit("u2") != 0 {
				t.Errorf("the second deployment was called %d times, want 0: its raise to %d did not fit", e.hit("u2"), each+inputOnly)
			}
			if got := e.spent(); got != inputOnly {
				t.Errorf("Spent = %d, want %d: the failed attempt is charged its input bound", got, inputOnly)
			}
			if got := e.gw.BudgetOvershoot(); got != 0 {
				t.Errorf("BudgetOvershoot = %d, want 0", got)
			}
		})
	}
}

// TestSameModelFailoverRaisesTheReservationWhenItFits: with room, the reservation is raised
// to the first attempt's charge plus the worst case before the second deployment is called,
// and that deployment serves the request.
func TestSameModelFailoverRaisesTheReservationWhenItFits(t *testing.T) {
	each := worstCase(t, stdIn, stdOut, 10)
	inputOnly := worstCase(t, stdIn, stdOut, 0)
	budget := 4 * each
	for _, tc := range sameModelFailoverCases() {
		t.Run(tc.name, func(t *testing.T) {
			models := []modelSpec{{id: "m", in: "1", out: "2", maxOut: 10, deps: []depSpec{{"m-a", tc.firstUp}, {"m-b", "u2"}}}}
			o := budgetOpts{budget: budget, policy: config.PolicyCheapest, models: models, ttftMs: tc.ttftMs}
			if tc.upstream != nil {
				o.upstream = tc.upstream(4)
			}
			e := newBudgetEnv(t, o)
			var secondSaw money.PicoUSD
			e.setOnUpstream(func(model string) {
				if model == "u2" {
					secondSaw = e.remaining()
				}
			})

			rec := e.post(tc.body)

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d, want 200 from the second deployment: %s", rec.Code, rec.Body.String())
			}
			if e.hit("u2") != 1 {
				t.Errorf("the second deployment was called %d times, want 1", e.hit("u2"))
			}
			if want := budget - inputOnly - each; secondSaw != want {
				t.Errorf("Remaining during the second attempt = %d, want %d: raised by the first attempt's charge", secondSaw, want)
			}
			if want := inputOnly + 5*stdIn + 4*stdOut; e.spent() != want {
				t.Errorf("Spent = %d, want %d: the first attempt's input bound plus the second's usage", e.spent(), want)
			}
			if got := e.gw.BudgetOvershoot(); got != 0 {
				t.Errorf("BudgetOvershoot = %d, want 0", got)
			}
		})
	}
}

// TestBudgetCommitsActualUsage: the commit is the usage the provider reported, priced with
// the model's prices, for a complete answer and for a stream, and nothing stays reserved.
func TestBudgetCommitsActualUsage(t *testing.T) {
	const completion = 40
	want := 5*stdIn + completion*stdOut
	budget := money.PicoUSD(1_000_000_000)

	cases := []struct {
		name string
		body string
	}{
		{"complete", chatBody("auto", "hi")},
		{"stream", streamBody("auto")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := stdModel(fmt.Sprintf("ua;tokens=%d", completion))
			e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{spec}})

			rec := e.post(tc.body)

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			if got := e.spent(); got != want {
				t.Errorf("Spent = %d, want %d = 5 prompt tokens * %d + %d completion tokens * %d", got, want, stdIn, completion, stdOut)
			}
			if got := e.remaining(); got != budget-want {
				t.Errorf("Remaining = %d, want %d: no reservation may outlive the request", got, budget-want)
			}
			if res, com, rel, _ := e.ledger.counts(); res != 1 || com != 1 || rel != 0 {
				t.Errorf("ledger reserve=%d commit=%d release=%d, want 1, 1, 0", res, com, rel)
			}
			if got := e.gw.BudgetOvershoot(); got != 0 {
				t.Errorf("BudgetOvershoot = %d, want 0", got)
			}
		})
	}
}

// TestBudgetCommitsOnEveryExit: whatever way the request ends, the reservation is replaced by
// the cost of the attempts that were sent.
func TestBudgetCommitsOnEveryExit(t *testing.T) {
	inputOnly := stdIn * money.PicoUSD(inUpperHi(t))
	budget := money.PicoUSD(1_000_000_000)
	timeout := map[string]string{"x-routepilot-timeout-ms": "300"}

	cases := []struct {
		name       string
		upstream   string
		stream     bool
		headers    map[string]string
		wantStatus int
		wantSpent  money.PicoUSD
	}{
		{"upstream 500 is not billed", "ua;fail=http500", false, nil, http.StatusBadGateway, 0},
		{"upstream 429 is not billed", "ua;fail=http429;retry_after=0", false, nil, http.StatusTooManyRequests, 0},
		{"a connection reset is not billed", "ua;fail=reset", false, nil, http.StatusBadGateway, 0},
		{"a timeout is billed its input bound", "ua;fail=stall", false, timeout, http.StatusGatewayTimeout, inputOnly},
		{"a stream that stalls before any content is billed its input bound", "ua;fail=stall", true, timeout, http.StatusGatewayTimeout, inputOnly},
		{"a client error is not billed", "ua;fail=http400", false, nil, http.StatusBadRequest, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel(tc.upstream)}, maxAttempts: 1})
			body := chatBody("auto", "hi")
			if tc.stream {
				body = streamBody("auto")
			}

			rec := e.postWith(body, tc.headers)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status %d, want %d: %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if got := e.spent(); got != tc.wantSpent {
				t.Errorf("Spent = %d, want %d", got, tc.wantSpent)
			}
			if got := e.remaining(); got != budget-tc.wantSpent {
				t.Errorf("Remaining = %d, want %d: the reservation must be gone", got, budget-tc.wantSpent)
			}
			if res, com, rel, _ := e.ledger.counts(); res != 1 || com != 1 || rel != 0 {
				t.Errorf("ledger reserve=%d commit=%d release=%d, want 1, 1, 0 (an attempt was sent)", res, com, rel)
			}
		})
	}
}

// resetAfterStatusLine answers 200 and then loses the connection: a complete answer is cut
// after part of its declared body, and a stream is reset before its first event. Either way
// the upstream had accepted and begun to serve the request.
func resetAfterStatusLine(stream bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", "4000")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"partial`)
		}
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			return
		}
		_ = conn.(*net.TCPConn).SetLinger(0) // RST instead of a clean FIN
		_ = conn.Close()
	})
}

// TestFailureAfterTheUpstreamAnsweredIsBilled: a call the provider accepted and began to
// serve is not free just because the connection then died. A non-streaming answer cut in
// transit has an unknown size and is charged its worst case; a stream reset before any
// content is charged its input bound. Both are recorded in the budget, and a failure before
// any response (TestBudgetCommitsOnEveryExit) stays free.
func TestFailureAfterTheUpstreamAnsweredIsBilled(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	cases := []struct {
		name   string
		stream bool
		want   money.PicoUSD
	}{
		{"a body cut after the 200 is charged its worst case", false, worstCase(t, stdIn, stdOut, 100)},
		{"a stream reset after the 200 and before any content is charged its input bound", true, worstCase(t, stdIn, stdOut, 0)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}, upstream: resetAfterStatusLine(tc.stream), maxAttempts: 1})
			body := chatBody("auto", "hi")
			if tc.stream {
				body = streamBody("auto")
			}

			rec := e.post(body)

			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status %d, want 502: %s", rec.Code, rec.Body.String())
			}
			if e.hit("ua") != 1 {
				t.Errorf("upstream was called %d times, want 1", e.hit("ua"))
			}
			if got := e.spent(); got != tc.want {
				t.Errorf("Spent = %d, want %d", got, tc.want)
			}
			if got := e.remaining(); got != budget-tc.want {
				t.Errorf("Remaining = %d, want %d: the reservation must be gone", got, budget-tc.want)
			}
			if got := e.gw.BudgetOvershoot(); got != 0 {
				t.Errorf("BudgetOvershoot = %d, want 0", got)
			}
		})
	}
}

// TestBudgetReleasedWhenNothingSent: a reservation made for a request that then sends nothing
// (every deployment became unavailable between the decision and the claim) is released, not
// committed.
func TestBudgetReleasedWhenNothingSent(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	e.ledger.mu.Lock()
	e.ledger.afterReserve = func(string, money.PicoUSD) { e.gw.health.Trip("m-a", time.Hour) }
	e.ledger.mu.Unlock()

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "no_healthy_deployment" {
		t.Fatalf("status %d: %s, want 503 no_healthy_deployment", rec.Code, rec.Body.String())
	}
	if e.hit("ua") != 0 {
		t.Errorf("upstream was called %d times", e.hit("ua"))
	}
	if res, com, rel, _ := e.ledger.counts(); res != 1 || com != 0 || rel != 1 {
		t.Errorf("ledger reserve=%d commit=%d release=%d, want 1, 0, 1", res, com, rel)
	}
	if e.remaining() != budget || e.spent() != 0 {
		t.Errorf("Remaining = %d Spent = %d, want the whole budget and nothing spent", e.remaining(), e.spent())
	}
}

// newFakeBudgetGateway is a gateway over a fake provider for tenant-a with the given budget;
// model-a costs stdIn and stdOut.
func newFakeBudgetGateway(t *testing.T, budget money.PicoUSD, p provider.Provider, ledger quota.BudgetLedger) (*Server, *bytes.Buffer) {
	t.Helper()
	return newAdmissionGateway(t, func(cfg *config.Config) {
		cfg.Models[0].InputPricePerMTok, cfg.Models[0].OutputPricePerMTok = "1", "2"
		tenantA(cfg).BudgetUSD = budget.USD()
	}, p, quota.Limiters{Budget: ledger})
}

// TestBudgetCommittedOnDisconnect: a client that leaves in the middle of a stream is charged
// for the input bound plus the content bytes the gateway had received, as an estimate, and
// the reservation is gone.
func TestBudgetCommittedOnDisconnect(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	inner := quota.NewMemoryBudget()
	// A ledger that, like Redis, fails a call whose context is done: the commit must not run
	// under the context of the client that left.
	ledger := &countingLedger{BudgetLedger: inner, respectCtx: true}
	lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
	started := make(chan struct{}, 1)
	p := &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
		started <- struct{}{}
		// Two content chunks, 5 and 6 bytes, then the upstream goes quiet until it is closed.
		return newGatedStream(make(chan struct{}), textChunk("hello"), textChunk(" world")), nil
	}}
	gw, _ := newFakeBudgetGateway(t, budget, p, ledger)
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := startStream(t, ctx, srv.URL, streamReqBody("model-a", false))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the upstream stream never started")
	}
	br := bufio.NewReader(resp.Body)
	for i := 0; i < 2; i++ {
		if ev, err := readEvent(br); err != nil || !strings.Contains(ev, `"content"`) {
			t.Fatalf("event %d = %q, %v, want a content chunk", i, ev, err)
		}
	}
	cancel() // the client leaves with the upstream still streaming

	want := money.PicoUSD(inUpperHi(t))*stdIn + 11*stdOut
	waitFor(t, 5*time.Second, "the disconnect to be committed", func() bool {
		s, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow)
		return s == want
	})
	if got, _ := inner.Remaining(context.Background(), "tenant-a", lim, admissionNow); got != budget-want {
		t.Errorf("Remaining = %d, want %d: the reservation must be gone", got, budget-want)
	}
	if got := gw.BudgetOvershoot(); got != 0 {
		t.Errorf("BudgetOvershoot = %d, want 0", got)
	}
}

// TestBudgetEstimatesUsageTheProviderDoesNotReport: an answer without usage is charged the
// input bound plus its content bytes (never more than the output budget), a stream that ends
// cleanly without a usage chunk likewise, and an answer with usage is charged exactly.
func TestBudgetEstimatesUsageTheProviderDoesNotReport(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	input := money.PicoUSD(inUpperHi(t)) * stdIn
	text := "abcdefghij" // 10 bytes
	complete := func(u *openai.Usage, content string) *fakeProvider {
		return &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			return &openai.ChatCompletion{
				Choices: []openai.Choice{{Message: openai.Message{Role: "assistant", Content: openai.Content{Text: &content}}}},
				Usage:   u,
			}, nil
		}}
	}

	t.Run("complete without usage", func(t *testing.T) {
		inner := quota.NewMemoryBudget()
		gw, _ := newFakeBudgetGateway(t, budget, complete(nil, text), inner)
		if rec := chat(gw, keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
		if got, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow); got != input+10*stdOut {
			t.Errorf("Spent = %d, want %d: the input bound plus 10 content bytes", got, input+10*stdOut)
		}
	})
	t.Run("complete with usage", func(t *testing.T) {
		inner := quota.NewMemoryBudget()
		gw, _ := newFakeBudgetGateway(t, budget, complete(&openai.Usage{PromptTokens: 7, CompletionTokens: 3}, text), inner)
		if rec := chat(gw, keyA, chatBody("model-a", "hi")); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
		if got, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow); got != 7*stdIn+3*stdOut {
			t.Errorf("Spent = %d, want %d: the reported usage", got, 7*stdIn+3*stdOut)
		}
	})
	t.Run("the estimate never exceeds the output budget", func(t *testing.T) {
		inner := quota.NewMemoryBudget()
		gw, _ := newFakeBudgetGateway(t, budget, complete(nil, strings.Repeat("x", 500)), inner)
		if rec := chat(gw, keyA, bodyWithMaxTokens("model-a", 20, false)); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
		if got, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow); got != input+20*stdOut {
			t.Errorf("Spent = %d, want %d: 500 bytes are at most the 20 tokens the request allowed", got, input+20*stdOut)
		}
		if got := gw.BudgetOvershoot(); got != 0 {
			t.Errorf("BudgetOvershoot = %d, want 0", got)
		}
	})
	t.Run("stream ending cleanly without a usage chunk", func(t *testing.T) {
		inner := quota.NewMemoryBudget()
		p := &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
			return &fakeStream{chunks: []*openai.ChatCompletionChunk{textChunk("hello"), textChunk(" world")}, err: io.EOF}, nil
		}}
		gw, _ := newFakeBudgetGateway(t, budget, p, inner)
		if rec := chat(gw, keyA, streamReqBody("model-a", false)); rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
		if got, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow); got != input+11*stdOut {
			t.Errorf("Spent = %d, want %d: the input bound plus 11 streamed bytes", got, input+11*stdOut)
		}
	})
}

// TestBudgetOvershootIsCountedAndLogged: a provider that bills more than the reservation
// still gets its actual cost committed; the excess is counted and logged without the prompt.
func TestBudgetOvershootIsCountedAndLogged(t *testing.T) {
	budget := money.PicoUSD(1_000_000_000)
	inner := quota.NewMemoryBudget()
	// 5 prompt tokens and 100 completion tokens against a request capped at 10.
	p := completionWithUsage(func() *openai.Usage { return &openai.Usage{PromptTokens: 5, CompletionTokens: 100} })
	gw, logs := newFakeBudgetGateway(t, budget, p, inner)
	const secret = "sentinel-prompt-do-not-log"
	body := fmt.Sprintf(`{"model":"model-a","max_tokens":10,"messages":[{"role":"user","content":%q}]}`, secret)

	rec := chat(gw, keyA, body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	req, _ := openai.ParseChatCompletionRequest([]byte(body))
	reserved := money.PicoUSD(catalog.Derive(req).InUpper)*stdIn + 10*stdOut
	actual := 5*stdIn + 100*stdOut
	if got := gw.BudgetOvershoot(); got != actual-reserved {
		t.Errorf("BudgetOvershoot = %d, want %d = actual %d - reserved %d", got, actual-reserved, actual, reserved)
	}
	lim := quota.BudgetLimit{Limit: budget, Period: quota.PeriodMonthly}
	if got, _ := inner.Spent(context.Background(), "tenant-a", lim, admissionNow); got != actual {
		t.Errorf("Spent = %d, want the full actual %d", got, actual)
	}
	out := logs.String()
	if !strings.Contains(out, "budget overshoot") || !strings.Contains(out, "request_id=") {
		t.Errorf("log lacks the overshoot warning with a request id:\n%s", out)
	}
	if strings.Contains(out, secret) || strings.Contains(out, keyA) {
		t.Errorf("log carries the prompt or the key:\n%s", out)
	}
}

// TestBudgetLedgerFailuresAre503: a ledger that fails is an overloaded gateway, not a free
// pass and not a 429.
func TestBudgetLedgerFailuresAre503(t *testing.T) {
	boom := errors.New("ledger down")
	budget := money.PicoUSD(1_000_000_000)

	t.Run("remaining", func(t *testing.T) {
		e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
		e.ledger.failRemaining = boom
		rec := e.post(chatBody("auto", "hi"))
		if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
			t.Fatalf("status %d: %s, want 503 overloaded", rec.Code, rec.Body.String())
		}
		if e.hit("ua") != 0 {
			t.Error("upstream called although the budget could not be read")
		}
		out := e.logs.String()
		if !strings.Contains(out, "budget remaining") || !strings.Contains(out, "request_id=") {
			t.Errorf("log lacks the limiter warning with the operation and request id:\n%s", out)
		}
		if strings.Contains(out, keyA) {
			t.Errorf("log carries the API key:\n%s", out)
		}
	})
	t.Run("reserve", func(t *testing.T) {
		e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
		e.ledger.failReserve = boom
		rec := e.post(chatBody("auto", "hi"))
		if rec.Code != http.StatusServiceUnavailable || decodeErr(t, rec).Error.Code != "overloaded" {
			t.Fatalf("status %d: %s, want 503 overloaded", rec.Code, rec.Body.String())
		}
		if e.hit("ua") != 0 {
			t.Error("upstream called although nothing was reserved")
		}
		// The script may have run before the reply failed, so a release is attempted.
		if _, com, rel, _ := e.ledger.counts(); com != 0 || rel != 1 {
			t.Errorf("commit=%d release=%d, want 0 and 1", com, rel)
		}
	})
	t.Run("a refused reserve is 429", func(t *testing.T) {
		// Another instance takes the whole budget after this gateway read Remaining.
		e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
		e.ledger.mu.Lock()
		e.ledger.beforeReserve = func() { e.squat(budget) }
		e.ledger.mu.Unlock()
		rec := e.post(chatBody("auto", "hi"))
		wantInsufficientQuota(t, rec, "tenant budget exhausted")
		if e.hit("ua") != 0 {
			t.Error("upstream called although the reservation was refused")
		}
		if res, com, rel, _ := e.ledger.counts(); res != 1 || com != 0 || rel != 0 {
			t.Errorf("ledger reserve=%d commit=%d release=%d, want 1, 0, 0: a refusal holds nothing", res, com, rel)
		}
	})
}

// TestTenantWithoutBudgetNeverTouchesTheLedger: only tenants with budget_usd cost the ledger a
// call.
func TestTenantWithoutBudgetNeverTouchesTheLedger(t *testing.T) {
	e := newBudgetEnv(t, budgetOpts{budget: 1_000_000_000, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", keyB, chatBody("auto", "hi"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if res, com, rel, rem := e.ledger.counts(); res+com+rel+rem != 0 {
		t.Errorf("ledger calls reserve=%d commit=%d release=%d remaining=%d, want none", res, com, rel, rem)
	}
	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("tenant-a: status %d: %s", rec.Code, rec.Body.String())
	}
	if res, _, _, rem := e.ledger.counts(); res != 1 || rem != 1 {
		t.Errorf("tenant-a: reserve=%d remaining=%d, want 1 and 1", res, rem)
	}
}

// TestBudgetPeriodFollowsTheClock: a monthly budget starts again on the first of the month,
// and a total budget never does.
func TestBudgetPeriodFollowsTheClock(t *testing.T) {
	// 4,000,000 picodollars above the input bound: K = 2, and 13,000,000 spent by one answer.
	budget := worstCase(t, stdIn, stdOut, 2)
	for _, period := range []string{quota.PeriodMonthly, quota.PeriodTotal} {
		t.Run(period, func(t *testing.T) {
			now := time.Date(2026, 10, 31, 23, 0, 0, 0, time.UTC)
			var mu sync.Mutex
			clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
			e := newBudgetEnv(t, budgetOpts{budget: budget, period: period, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua")}, now: clock})

			if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
				t.Fatalf("first request: status %d: %s", rec.Code, rec.Body.String())
			}
			// stdActual (13,000,000) leaves 17,000,000 of 30,000,000: below the input bound.
			wantInsufficientQuota(t, e.post(chatBody("auto", "hi")), "")

			mu.Lock()
			now = time.Date(2026, 11, 1, 0, 0, 1, 0, time.UTC)
			mu.Unlock()
			rec := e.post(chatBody("auto", "hi"))
			if period == quota.PeriodMonthly && rec.Code != http.StatusOK {
				t.Errorf("monthly, next month: status %d, want 200: %s", rec.Code, rec.Body.String())
			}
			if period == quota.PeriodTotal && rec.Code != http.StatusTooManyRequests {
				t.Errorf("total, next month: status %d, want 429: %s", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestBudgetNeverOvershootsUnderConcurrency sends 64 requests at once at a budget that pays
// for only a fraction of them. Whatever the interleaving, what is committed stays within the
// budget, nothing is reported as overshoot, no reservation is left behind and every request is
// either served or told 429.
func TestBudgetNeverOvershootsUnderConcurrency(t *testing.T) {
	const clients = 64
	budget := 12 * stdActual // room for 12 answers; 64 would cost 64
	backends := []struct {
		name string
		make func(t *testing.T) quota.BudgetLedger
	}{
		{"memory", func(*testing.T) quota.BudgetLedger { return quota.NewMemoryBudget() }},
		{"redis", func(t *testing.T) quota.BudgetLedger {
			mr := miniredis.RunT(t)
			client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
			t.Cleanup(func() { _ = client.Close() })
			return quota.NewRedisBudget(client, "routepilot")
		}},
	}
	// With ROUTEPILOT_TEST_REDIS_URL set, the same test runs against a real Redis too.
	if url := os.Getenv("ROUTEPILOT_TEST_REDIS_URL"); url != "" {
		backends = append(backends, struct {
			name string
			make func(t *testing.T) quota.BudgetLedger
		}{"real-redis", func(t *testing.T) quota.BudgetLedger {
			opts, err := redis.ParseURL(url)
			if err != nil {
				t.Fatalf("parse ROUTEPILOT_TEST_REDIS_URL: %v", err)
			}
			client := redis.NewClient(opts)
			var b [8]byte
			if _, err := rand.Read(b[:]); err != nil {
				t.Fatal(err)
			}
			prefix := "routepilot-test-" + hex.EncodeToString(b[:])
			t.Cleanup(func() {
				ctx := context.Background()
				iter := client.Scan(ctx, 0, prefix+":*", 500).Iterator()
				for iter.Next(ctx) {
					client.Del(ctx, iter.Val())
				}
				_ = client.Close()
			})
			return quota.NewRedisBudget(client, prefix)
		}})
	}
	for _, be := range backends {
		t.Run(be.name, func(t *testing.T) {
			spec := stdModel("ua")
			spec.maxOut = 10 // a worst case of 46,000,000: three fit at once, then the budget shrinks them
			e := newBudgetEnv(t, budgetOpts{budget: budget, policy: config.PolicyCheapest, models: []modelSpec{spec}, inner: be.make(t)})

			recs := startTogether(clients, func(int) *httptest.ResponseRecorder {
				return e.post(chatBody("auto", "hi"))
			})

			var ok, refused int
			var served money.PicoUSD // what the served answers cost by their reported usage
			for i, rec := range recs {
				switch rec.Code {
				case http.StatusOK:
					ok++
					var resp openai.ChatCompletion
					if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil || resp.Usage == nil {
						t.Fatalf("request %d: answer without usage (%v): %s", i, err, rec.Body.String())
					}
					// The budget can shrink an answer's max_tokens below the mock's 4 tokens,
					// and the mock honors it, so each answer is priced from its own usage.
					served += stdIn*money.PicoUSD(resp.Usage.PromptTokens) + stdOut*money.PicoUSD(resp.Usage.CompletionTokens)
				case http.StatusTooManyRequests:
					refused++
					if code := decodeErr(t, rec).Error.Code; code != "insufficient_quota" {
						t.Errorf("request %d: 429 with code %q, want insufficient_quota", i, code)
					}
				default:
					t.Errorf("request %d: status %d, want 200 or 429: %s", i, rec.Code, rec.Body.String())
				}
			}
			t.Logf("%d clients at once, budget for 12 answers: served=%d refused=%d spent=%d of %d", clients, ok, refused, e.spent(), budget)
			if ok == 0 || refused == 0 {
				t.Errorf("served %d refused %d: the budget must serve some and refuse some", ok, refused)
			}
			if got := e.spent(); got != served {
				t.Errorf("Spent = %d, want %d, the reported usage of the %d served answers", got, served, ok)
			}
			if got := e.spent(); got > budget {
				t.Errorf("Spent %d is above the budget %d", got, budget)
			}
			if got := e.remaining(); got != budget-e.spent() {
				t.Errorf("Remaining = %d, want %d: a reservation was left behind", got, budget-e.spent())
			}
			if got := e.gw.BudgetOvershoot(); got != 0 {
				t.Errorf("BudgetOvershoot = %d, want 0", got)
			}
			if res, com, rel, _ := e.ledger.counts(); com != ok || res < ok || rel != 0 {
				t.Errorf("ledger reserve=%d commit=%d release=%d, want %d commits, at least as many reserves and no release", res, com, rel, ok)
			}
		})
	}
}

// TestLearnedRoutingHonorsTheBudget: the learned policy routes against the tenant's remaining
// budget too. With 20,000,000 picodollars only tiny-weak (2,600,000 for the input bound and
// 400,000 per output token) is affordable, at the 43 output tokens the rest pays for, and with
// less than one output token's worth nothing is.
func TestLearnedRoutingHonorsTheBudget(t *testing.T) {
	var mu sync.Mutex
	var caps []int
	p := &fakeProvider{complete: func(_ context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
		mu.Lock()
		defer mu.Unlock()
		if req.MaxTokens != nil {
			caps = append(caps, *req.MaxTokens)
		}
		return &openai.ChatCompletion{Usage: &openai.Usage{PromptTokens: 5, CompletionTokens: 3}}, nil
	}}
	newBudgeted := func(budget money.PicoUSD) *learnedEnv {
		e := newLearnedEnv(t, learnedOpts{provider: p})
		e.gw.budgets = map[string]quota.BudgetLimit{"tenant-a": {Limit: budget, Period: quota.PeriodMonthly}}
		return e
	}

	e := newBudgeted(20_000_000)
	rec := e.post(chatBody("auto", "hi"), nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("x-routepilot-model"); got != "tiny-weak" {
		t.Errorf("model = %q, want tiny-weak: tiny-strong's input alone costs 26,000,000", got)
	}
	mu.Lock()
	if len(caps) != 1 || caps[0] != 43 {
		t.Errorf("upstream max_tokens = %v, want [43] = (20,000,000 - 2,600,000) / 400,000", caps)
	}
	mu.Unlock()
	lim := e.gw.budgets["tenant-a"]
	if got, _ := e.gw.limits.Budget.Spent(context.Background(), "tenant-a", lim, time.Now()); got != 5*100_000+3*400_000 {
		t.Errorf("Spent = %d, want the reported usage priced on tiny-weak (1,700,000)", got)
	}

	e = newBudgeted(2_999_999) // 2,600,000 + 400,000 is one output token
	rec = e.post(chatBody("auto", "hi"), nil)
	wantInsufficientQuota(t, rec, "budget remaining below the cheapest worst-case cost")
	if rec.Header().Get("x-routepilot-policy-version") == "" {
		t.Error("the 429 carries no policy version: the learned policy did not decide it")
	}
}

// TestAttemptRecordsAreFinalWhenTheHandlerReturns pins the record the usage ledger will read:
// one per attempt sent, in order, final after handleChat returns, for a tenant with a budget
// and for one without. An attempt that ends without a signal of its own (a timeout) is the
// estimated one.
func TestAttemptRecordsAreFinalWhenTheHandlerReturns(t *testing.T) {
	handle := func(e *budgetEnv, key string, headers map[string]string) (*reqState, *httptest.ResponseRecorder) {
		rs := &reqState{id: "req-records"}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chatBody("auto", "hi")))
		req.Header.Set("Authorization", "Bearer "+key)
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		rec := httptest.NewRecorder()
		e.gw.handleChat(&statusWriter{ResponseWriter: rec}, rs, req)
		return rs, rec
	}
	for _, key := range []string{keyA, keyB} {
		t.Run("failover/"+key, func(t *testing.T) {
			spec := stdModel("uc")
			spec.deps = []depSpec{{"m-a", "uc"}, {"m-b", "ua"}} // the first answers 503
			e := newBudgetEnv(t, budgetOpts{budget: 1_000_000_000, policy: config.PolicyCheapest, models: []modelSpec{spec}, upstream: upstream503For("uc")})

			rs, rec := handle(e, key, nil)

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			if len(rs.attemptUsages) != 2 {
				t.Fatalf("%d attempt records, want 2", len(rs.attemptUsages))
			}
			first, second := rs.attemptUsages[0], rs.attemptUsages[1]
			if first.Model != "m" || first.Deployment != "m-a" || first.Prompt != 0 || first.Completion != 0 || first.Estimated || first.Cost != 0 {
				t.Errorf("first record = %+v, want the 503 on m-a: no tokens, not estimated, free", first)
			}
			if second.Model != "m" || second.Deployment != "m-b" || second.Prompt != 5 || second.Completion != 4 || second.Estimated || second.Cost != stdActual {
				t.Errorf("second record = %+v, want m-b with 5 prompt and 4 completion tokens costing %d", second, stdActual)
			}
		})
		t.Run("timeout/"+key, func(t *testing.T) {
			e := newBudgetEnv(t, budgetOpts{budget: 1_000_000_000, policy: config.PolicyCheapest, models: []modelSpec{stdModel("ua;fail=stall")}, maxAttempts: 1})

			rs, rec := handle(e, key, map[string]string{"x-routepilot-timeout-ms": "300"})

			if rec.Code != http.StatusGatewayTimeout {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			if len(rs.attemptUsages) != 1 {
				t.Fatalf("%d attempt records, want 1", len(rs.attemptUsages))
			}
			a, iu := rs.attemptUsages[0], inUpperHi(t)
			if a.Model != "m" || a.Prompt != iu || a.Completion != 0 || !a.Estimated || a.Cost != money.PicoUSD(iu)*stdIn {
				t.Errorf("record = %+v, want an estimate of %d prompt tokens (the input bound) costing %d", a, iu, money.PicoUSD(iu)*stdIn)
			}
		})
	}
}
