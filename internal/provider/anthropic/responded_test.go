package anthropic_test

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/omorsi45/routepilot/internal/provider"
)

// resetAfterStatusLine answers 200, writes part of a body (a Content-Length it does not
// keep to, or the start of an event stream) and resets the connection.
func resetAfterStatusLine(t *testing.T, contentType, partial string, declared string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		if declared != "" {
			w.Header().Set("Content-Length", declared)
		}
		w.WriteHeader(http.StatusOK)
		if partial != "" {
			_, _ = io.WriteString(w, partial)
		}
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		_ = conn.(*net.TCPConn).SetLinger(0) // RST instead of a clean FIN
		_ = conn.Close()
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestCompleteBodyCutAfter200IsMarkedResponded: a connection that dies while the body of a
// 200 is read is a connect failure that happened after the upstream answered.
func TestCompleteBodyCutAfter200IsMarkedResponded(t *testing.T) {
	srv := resetAfterStatusLine(t, "application/json", `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"partial`, "4000")

	_, err := newClient(srv.URL, "k").Complete(context.Background(), newReq("m"))

	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindConnect || !pe.Responded {
		t.Errorf("error = %+v, want a connect failure marked Responded", pe)
	}
}

// TestStreamResetAfter200IsMarkedResponded: a reset after the 2xx headers and before the
// first event is a failure after the upstream answered.
func TestStreamResetAfter200IsMarkedResponded(t *testing.T) {
	srv := resetAfterStatusLine(t, "text/event-stream", "", "")
	st, err := newClient(srv.URL, "k").Stream(context.Background(), newReq("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	_, err = st.Recv()

	if errors.Is(err, io.EOF) {
		t.Fatal("a reset surfaced as io.EOF")
	}
	pe := mustProviderErr(t, err)
	if pe.Kind != provider.KindConnect || !pe.Responded {
		t.Errorf("error = %+v, want a connect failure marked Responded", pe)
	}
}

// TestFailuresBeforeAnyResponseAreNotResponded: nothing was answered, so nothing is marked.
func TestFailuresBeforeAnyResponseAreNotResponded(t *testing.T) {
	down := httptest.NewServer(http.NotFoundHandler())
	downURL := down.URL
	down.Close()
	rejecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer rejecting.Close()

	for name, url := range map[string]string{"connect refused": downURL, "HTTP status": rejecting.URL} {
		t.Run(name+" complete", func(t *testing.T) {
			_, err := newClient(url, "k").Complete(context.Background(), newReq("m"))
			if pe := mustProviderErr(t, err); pe.Responded {
				t.Errorf("error = %+v, want Responded false", pe)
			}
		})
		t.Run(name+" stream", func(t *testing.T) {
			_, err := newClient(url, "k").Stream(context.Background(), newReq("m"))
			if pe := mustProviderErr(t, err); pe.Responded {
				t.Errorf("error = %+v, want Responded false", pe)
			}
		})
	}
}
