package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
	"github.com/omorsi45/routepilot/internal/quota"
)

// budgetSpy records, per upstream model, the output cap fields of each request
// the upstream received, then passes the request on.
type budgetSpy struct {
	next http.Handler

	mu   sync.Mutex
	seen map[string][]outputCaps
}

// outputCaps holds the raw JSON of the two output cap fields, empty when absent.
type outputCaps struct{ maxTokens, maxCompletionTokens string }

func (s *budgetSpy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(bytes.NewReader(b))
	var body struct {
		Model               string          `json:"model"`
		MaxTokens           json.RawMessage `json:"max_tokens"`
		MaxCompletionTokens json.RawMessage `json:"max_completion_tokens"`
	}
	_ = json.Unmarshal(b, &body)
	s.mu.Lock()
	if s.seen == nil {
		s.seen = map[string][]outputCaps{}
	}
	s.seen[body.Model] = append(s.seen[body.Model], outputCaps{string(body.MaxTokens), string(body.MaxCompletionTokens)})
	s.mu.Unlock()
	s.next.ServeHTTP(w, r)
}

func (s *budgetSpy) caps(upstream string) []outputCaps {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]outputCaps(nil), s.seen[upstream]...)
}

// failoverBudgetEnv is two models of different output budgets, cheapest first.
// The cheap one's upstream ("uc") always answers 503, so every request fails
// over to the pricey one ("up").
func failoverBudgetEnv(t *testing.T) (*attemptEnv, *budgetSpy) {
	t.Helper()
	ok := mock.NewHandler(mock.Config{})
	spy := &budgetSpy{next: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(b))
		if strings.Contains(string(b), `"model":"uc"`) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"overloaded","type":"server_error","code":"overloaded"}}`)
			return
		}
		ok.ServeHTTP(w, r)
	})}
	e := newAttemptEnv(t, attemptOpts{
		policy:   config.PolicyCheapest,
		upstream: spy,
		models: []modelSpec{
			{id: "cheap", in: "0.1", out: "0.1", maxOut: 1000, deps: []depSpec{{"cheap-a", "uc"}}},
			{id: "pricey", in: "5", out: "5", maxOut: 2000, deps: []depSpec{{"pricey-a", "up"}}},
		},
	})
	return e, spy
}

// TestUpstreamGetsTheAttemptBudgetWhenClientSetsNone: with no output cap in the
// request, each attempt sends the K its own model was admitted with, so a
// failover to a model with a different K forwards that model's K.
func TestUpstreamGetsTheAttemptBudgetWhenClientSetsNone(t *testing.T) {
	e, spy := failoverBudgetEnv(t)

	rec := e.post(chatBody("auto", "hi"))

	if rec.Code != http.StatusOK || modelOf(t, rec) != "pricey" {
		t.Fatalf("status %d model %q, want 200 from pricey after failover: %s", rec.Code, modelOf(t, rec), rec.Body.String())
	}
	if got, want := spy.caps("uc"), []outputCaps{{maxTokens: "1000"}}; len(got) != 1 || got[0] != want[0] {
		t.Errorf("cheap attempt caps = %+v, want max_tokens 1000 only", got)
	}
	if got, want := spy.caps("up"), []outputCaps{{maxTokens: "2000"}}; len(got) != 1 || got[0] != want[0] {
		t.Errorf("pricey attempt caps = %+v, want max_tokens 2000 only", got)
	}
}

// TestStreamUpstreamGetsTheAttemptBudget: the streaming path builds its
// upstream request the same way.
func TestStreamUpstreamGetsTheAttemptBudget(t *testing.T) {
	spy := &budgetSpy{next: mock.NewHandler(mock.Config{})}
	e := newAttemptEnv(t, attemptOpts{
		upstream: spy,
		models:   []modelSpec{{id: "m", maxOut: 777, deps: []depSpec{{"dep-a", "ua;tokens=3"}}}},
	})

	rec := e.post(streamBody("m"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := spy.caps("ua;tokens=3"); len(got) != 1 || got[0] != (outputCaps{maxTokens: "777"}) {
		t.Errorf("stream attempt caps = %+v, want max_tokens 777 only", got)
	}
}

// TestUpstreamKeepsTheClientOutputCap: a cap the client set is forwarded on every attempt,
// in the field the client used and with nothing added next to it. When both fields are set
// only the effective one (max_completion_tokens) is forwarded, since that is the cap the
// request was admitted and reserved for.
func TestUpstreamKeepsTheClientOutputCap(t *testing.T) {
	msgs := `"messages":[{"role":"user","content":"hi"}]`
	tests := []struct {
		name string
		body string
		want outputCaps
	}{
		{"max_tokens", `{"model":"auto",` + msgs + `,"max_tokens":50}`, outputCaps{maxTokens: "50"}},
		{"max_completion_tokens", `{"model":"auto",` + msgs + `,"max_completion_tokens":70}`, outputCaps{maxCompletionTokens: "70"}},
		{"both", `{"model":"auto",` + msgs + `,"max_tokens":50,"max_completion_tokens":70}`, outputCaps{maxCompletionTokens: "70"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, spy := failoverBudgetEnv(t)

			rec := e.post(tt.body)

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			for _, upstream := range []string{"uc", "up"} {
				if got := spy.caps(upstream); len(got) != 1 || got[0] != tt.want {
					t.Errorf("%s attempt caps = %+v, want %+v", upstream, got, tt.want)
				}
			}
		})
	}
}

// reserveRecorder records the size of every token reservation.
type reserveRecorder struct {
	quota.TokenLimiter
	mu sync.Mutex
	ns []int64
}

func (r *reserveRecorder) Reserve(ctx context.Context, key string, lim quota.RateLimit, n int64, now time.Time) (quota.Result, error) {
	r.mu.Lock()
	r.ns = append(r.ns, n)
	r.mu.Unlock()
	return r.TokenLimiter.Reserve(ctx, key, lim, n, now)
}

// TestTokenReservationUsesTheLargestCandidateBudget: a failover may land on any
// candidate, so the tokens reserved up front cover the largest K among them,
// not just the first candidate's.
func TestTokenReservationUsesTheLargestCandidateBudget(t *testing.T) {
	rec := &reserveRecorder{TokenLimiter: quota.NewMemory().Tokens}
	gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
		cfg.Routing.Policy = config.PolicyStrongest
		// model-a ranks first with the smaller budget; model-b follows.
		cfg.Models[0].Rank, cfg.Models[0].MaxOutputTokens = 2, 1000
		cfg.Models[1].Rank, cfg.Models[1].MaxOutputTokens = 1, 2000
		tn := tenantA(cfg)
		tn.TPM, tn.TPMBurst = 100_000, 100_000
	}, okProvider(), quota.Limiters{Tokens: rec})

	resp := chat(gw, keyA, chatBody("auto", "hi"))

	if resp.Code != 200 {
		t.Fatalf("status %d: %s", resp.Code, resp.Body.String())
	}
	if want := int64(inUpperHi(t)) + 2000; len(rec.ns) != 1 || rec.ns[0] != want {
		t.Errorf("reservations = %v, want one of InUpper + 2000 = %d (the largest K, not the first candidate's 1000)", rec.ns, want)
	}
}

// TestTPMReservesLargestModelBound pins Ruling V1-R24: a failover may land on
// any candidate, and each candidate's bound is the request bound plus its own
// hidden prompt overhead plus its own K. The reservation is the largest of
// those sums, not the largest overhead plus the largest K.
func TestTPMReservesLargestModelBound(t *testing.T) {
	tests := []struct {
		name string
		// model-a ranks first, model-b second.
		aK, aOverhead, bK, bOverhead int
		wantExtra                    int // added to InUpper
	}{
		{"the overhead model has the larger sum", 1000, 0, 800, 500, 1300},
		{"the larger K model has the larger sum", 2000, 0, 800, 500, 2000},
		{"both terms on the first candidate", 1000, 500, 800, 0, 1500},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &reserveRecorder{TokenLimiter: quota.NewMemory().Tokens}
			gw, _ := newAdmissionGateway(t, func(cfg *config.Config) {
				cfg.Routing.Policy = config.PolicyStrongest
				cfg.Models[0].Rank, cfg.Models[0].MaxOutputTokens, cfg.Models[0].PromptOverheadTokens = 2, tt.aK, tt.aOverhead
				cfg.Models[1].Rank, cfg.Models[1].MaxOutputTokens, cfg.Models[1].PromptOverheadTokens = 1, tt.bK, tt.bOverhead
				tn := tenantA(cfg)
				tn.TPM, tn.TPMBurst = 100_000, 100_000
			}, okProvider(), quota.Limiters{Tokens: rec})

			resp := chat(gw, keyA, chatBody("auto", "hi"))

			if resp.Code != 200 {
				t.Fatalf("status %d: %s", resp.Code, resp.Body.String())
			}
			if want := int64(inUpperHi(t) + tt.wantExtra); len(rec.ns) != 1 || rec.ns[0] != want {
				t.Errorf("reservations = %v, want one of %d (largest of InUpper + overhead + K)", rec.ns, want)
			}
		})
	}
}
