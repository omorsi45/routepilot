package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
)

// slowStore is a usage ledger whose every write takes delay, whatever the context says. It
// stands for any post-response work that is slow: the budget commit, the TPM refund, the
// decision log or the lease release.
type slowStore struct {
	recordingStore
	delay    time.Duration
	started  chan struct{} // closed when the first write begins
	finished atomic.Int32  // writes that ran their whole delay
}

func newSlowStore(delay time.Duration) *slowStore {
	return &slowStore{delay: delay, started: make(chan struct{})}
}

func (s *slowStore) RecordAttempt(_ context.Context, _ accounting.UsageRow) error {
	select {
	case <-s.started:
	default:
		close(s.started)
	}
	time.Sleep(s.delay)
	s.finished.Add(1)
	return nil
}

// RecordAttempts is a slow write too, one delay per batch.
func (s *slowStore) RecordAttempts(ctx context.Context, rows []accounting.UsageRow) error {
	return s.RecordAttempt(ctx, rows[0])
}

// TestCompleteResponseIsFlushedBeforeBookkeeping (Ruling V1-R30): the client has the whole
// response, status and body, long before the post-response work is done. The ledger write
// takes 300 ms; the client must not wait for it, for an answer or for an error.
func TestCompleteResponseIsFlushedBeforeBookkeeping(t *testing.T) {
	const (
		storeDelay = 300 * time.Millisecond
		clientMax  = 150 * time.Millisecond
	)
	cases := []struct {
		name     string
		upstream string // upstream model string, which carries the mock directive
		status   int
		errCode  string // the OpenAI error code, empty for a success
	}{
		{name: "answer", upstream: "ua", status: http.StatusOK},
		{name: "error", upstream: "ua;fail=http400", status: http.StatusBadRequest, errCode: "invalid_request"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newAttemptEnv(t, attemptOpts{models: []modelSpec{
				{id: "m", deps: []depSpec{{"dep-a", tc.upstream}}},
			}})
			store := newSlowStore(storeDelay)
			useLedger(t, e.gw, store)
			srv := httptest.NewServer(e.gw.Handler())
			defer srv.Close() // waits for the handler; the slow write runs on the batcher, which the test cleanup closes

			req, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(chatBody("m", "hi")))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Authorization", "Bearer "+keyA)
			req.Header.Set("Content-Type", "application/json")

			start := time.Now()
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body) // returns at the end of the body, not at the end of the handler
			took := time.Since(start)
			if err != nil {
				t.Fatalf("reading the body: %v", err)
			}

			if took >= clientMax {
				t.Errorf("the client had the complete response after %v, want under %v: it waited for the %v ledger write", took, clientMax, storeDelay)
			}
			if resp.StatusCode != tc.status {
				t.Fatalf("status %d, want %d: %s", resp.StatusCode, tc.status, body)
			}
			if !json.Valid(body) {
				t.Fatalf("body is not complete JSON: %q", body)
			}
			if resp.ContentLength != int64(len(body)) {
				t.Errorf("Content-Length %d for a %d byte body: without it the client cannot tell the body is complete before the handler returns", resp.ContentLength, len(body))
			}
			if tc.errCode != "" {
				var eb errBody
				if err := json.Unmarshal(body, &eb); err != nil || eb.Error.Code != tc.errCode {
					t.Errorf("error body %s (decode error %v), want code %q", body, err, tc.errCode)
				}
			}

			// The test measured the right thing: the slow write did start, and it was still
			// running when the client already had everything.
			select {
			case <-store.started:
			case <-time.After(2 * time.Second):
				t.Fatal("the ledger was never written, so the test proved nothing")
			}
			if n := store.finished.Load(); n != 0 {
				t.Errorf("%d ledger writes had finished when the client was done; the response was not early", n)
			}
		})
	}
}
