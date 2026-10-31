package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/reliability"
)

// TestStreamProbeResolvesAtCommit: a stream that holds the half-open probe settles it as a
// success the moment its response commits, not when the stream ends, so a
// long answer does not keep the deployment half-open for its whole length. Nothing is
// recorded again at the end of the stream.
func TestStreamProbeResolvesAtCommit(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var mu sync.Mutex
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	// 100 tokens at 50 ms each: the stream runs for about 5 s after its first chunk.
	e := newAttemptEnv(t, attemptOpts{
		clock:  clock,
		models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua;tokens=100;ttft_ms=0;tpot_ms=50"}}}},
	})
	e.gw.health.Trip("dep-a", time.Second)
	mu.Lock()
	now = now.Add(2 * time.Second) // the cooldown elapsed: the next attempt is the probe
	mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamBody("m"))).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyA)
	done := make(chan struct{})
	go func() {
		e.gw.Handler().ServeHTTP(httptest.NewRecorder(), req)
		close(done)
	}()

	waitFor(t, 2*time.Second, "the probe to resolve at commit", func() bool {
		return e.gw.health.State("dep-a") == reliability.Closed
	})
	select {
	case <-done:
		t.Fatal("the stream ended before the check: the probe must resolve while the stream is still running")
	default:
	}

	cancel() // the client leaves mid-stream
	<-done
	if h := e.gw.health.Snapshot()["dep-a"]; h.State != reliability.Closed || h.ErrorRate != 0 {
		t.Errorf("dep-a health after the stream = %+v, want closed with error rate 0 (the probe was recorded once, at commit)", h)
	}
}
