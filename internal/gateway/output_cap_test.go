package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"testing"

	"github.com/omorsi45/routepilot/internal/mock"
)

// TestOutputCapIsForwardedOnceInTheFieldTheClientUsed: the reservation is made for the
// effective output cap (max_completion_tokens wins over max_tokens), so that is the one cap
// the upstream gets. A request with both fields set must not reach an upstream that honors
// the larger one.
func TestOutputCapIsForwardedOnceInTheFieldTheClientUsed(t *testing.T) {
	cases := []struct {
		name string
		caps string
		// wantMaxTokens and wantMaxCompletion are the raw JSON the upstream sees, empty when
		// the field is absent.
		wantMaxTokens, wantMaxCompletion string
	}{
		{"both fields", `"max_tokens":100000,"max_completion_tokens":10`, "", "10"},
		{"both fields, the smaller one in max_tokens", `"max_tokens":10,"max_completion_tokens":20`, "", "20"},
		{"only max_completion_tokens", `"max_completion_tokens":10`, "", "10"},
		{"only max_tokens", `"max_tokens":10`, "10", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var seen map[string]json.RawMessage
			next := mock.NewHandler(mock.Config{})
			e := newAttemptEnv(t, attemptOpts{
				models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}},
				upstream: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					b, _ := io.ReadAll(r.Body)
					r.Body = io.NopCloser(bytes.NewReader(b))
					mu.Lock()
					seen = nil
					_ = json.Unmarshal(b, &seen)
					mu.Unlock()
					next.ServeHTTP(w, r)
				}),
			})

			rec := e.post(`{"model":"m","messages":[{"role":"user","content":"hi"}],` + tc.caps + `}`)

			if rec.Code != http.StatusOK {
				t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
			}
			mu.Lock()
			defer mu.Unlock()
			if got := string(seen["max_tokens"]); got != tc.wantMaxTokens {
				t.Errorf("upstream max_tokens = %q, want %q", got, tc.wantMaxTokens)
			}
			if got := string(seen["max_completion_tokens"]); got != tc.wantMaxCompletion {
				t.Errorf("upstream max_completion_tokens = %q, want %q", got, tc.wantMaxCompletion)
			}
		})
	}
}
