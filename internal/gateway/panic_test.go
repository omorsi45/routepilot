package gateway

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// panicStream is a stream whose Recv panics, as a buggy adapter would on unusual upstream
// data.
type panicStream struct{}

func (panicStream) Recv() (*openai.ChatCompletionChunk, error) { panic("provider bug") }
func (panicStream) Close() error                               { return nil }

// panickingProviders are the three places a provider can panic in: a whole answer, the start
// of a stream, and the read of a stream.
func panickingProviders() []struct {
	name string
	body string
	p    *fakeProvider
} {
	return []struct {
		name string
		body string
		p    *fakeProvider
	}{
		{"complete", chatBody("model-a", "hi"), &fakeProvider{complete: func(context.Context, *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
			panic("provider bug")
		}}},
		{"stream call", streamReqBody("model-a", false), &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
			panic("provider bug")
		}}},
		{"stream read", streamReqBody("model-a", false), &fakeProvider{stream: func(context.Context, *openai.ChatCompletionRequest) (provider.Stream, error) {
			return panicStream{}, nil
		}}},
	}
}

// serveExpectingPanic sends the request and checks that the panic reaches the caller, as
// net/http's server then recovers it.
func serveExpectingPanic(t *testing.T, gw *Server, body string) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Error("the panic did not reach the caller")
		}
	}()
	do(gw.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body)
}

// TestProviderPanicReleasesAHalfOpenProbe: a request that takes a half-open breaker's one
// probe and whose provider panics must hand the probe back. Otherwise nothing ever resolves
// it and the deployment stays out of rotation until the process restarts.
func TestProviderPanicReleasesAHalfOpenProbe(t *testing.T) {
	for _, tc := range panickingProviders() {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Unix(1_700_000_000, 0)
			gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": tc.p},
				Options{Now: func() time.Time { return now }})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			gw.health.Trip("dep-a", time.Second)
			now = now.Add(2 * time.Second) // the cooldown is over: the next attempt is the probe

			serveExpectingPanic(t, gw, tc.body)

			now = now.Add(24 * time.Hour)
			if !gw.health.Available("dep-a") {
				t.Errorf("dep-a is %v and takes no further probe after the panic, want the probe released", gw.health.State("dep-a"))
			}
		})
	}
}

// TestPanickingRequestIsStillLogged: the request an operator most needs to find is the one
// whose handler panicked, so it still leaves its request log line, its metrics sample and
// its trace, with the open attempt span ended as aborted.
func TestPanickingRequestIsStillLogged(t *testing.T) {
	for _, tc := range panickingProviders() {
		t.Run(tc.name, func(t *testing.T) {
			logs := &bytes.Buffer{}
			gw, err := New(buildConfig(t, cfgOpts{}), map[string]provider.Provider{"mockp": tc.p}, Options{
				Logger: slog.New(slog.NewTextHandler(logs, nil)),
				Now:    func() time.Time { return time.Unix(1_700_000_000, 0) },
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			exp := traceTo(t, gw)

			serveExpectingPanic(t, gw, tc.body)

			var lines []string
			for _, l := range strings.Split(logs.String(), "\n") {
				if strings.Contains(l, "msg=request ") {
					lines = append(lines, l)
				}
			}
			if len(lines) != 1 {
				t.Fatalf("%d request log lines, want 1:\n%s", len(lines), logs.String())
			}
			for _, want := range []string{"tenant=tenant-a", "model=model-a", "deployment=dep-a", "attempts=1", "status=500"} {
				if !strings.Contains(lines[0], want) {
					t.Errorf("request log line %q lacks %s", lines[0], want)
				}
			}

			metrics := do(gw.Handler(), http.MethodGet, "/metrics", "", "").Body.String()
			if !strings.Contains(metrics, `rp_requests_total{model="model-a",policy_role="pinned",status="500",tenant="tenant-a"} 1`) {
				t.Errorf("no rp_requests_total sample for the panicked request in:\n%s", metrics)
			}

			tr := treeOf(t, exp)
			attempts := tr.children(spanAttempt)
			if len(attempts) != 1 {
				t.Fatalf("%d attempt spans, want 1: %v", len(attempts), tr.names())
			}
			wantStr(t, attempts[0], "rp.outcome", "aborted")
		})
	}
}
