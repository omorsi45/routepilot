package gateway

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

// notFlushableWriter is a ResponseWriter with no Flush, like a wrapper that hides it.
type notFlushableWriter struct{ http.ResponseWriter }

// TestWriteJSONFlushesWithItsLength: the response is flushed and carries its
// own length, the pair that lets a client finish reading before the handler returns.
func TestWriteJSONFlushesWithItsLength(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, map[string]string{"a": "b"})

	if !rec.Flushed {
		t.Error("the response was not flushed")
	}
	if got, want := rec.Header().Get("Content-Length"), strconv.Itoa(rec.Body.Len()); got != want {
		t.Errorf("Content-Length = %q, want %q (the body length)", got, want)
	}
	if rec.Code != http.StatusOK || rec.Body.String() != `{"a":"b"}` {
		t.Errorf("status %d body %q", rec.Code, rec.Body.String())
	}
}

// TestWriteJSONWithoutFlushSupport: a writer that cannot flush still gets the whole response.
func TestWriteJSONWithoutFlushSupport(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(notFlushableWriter{rec}, http.StatusOK, map[string]string{"a": "b"})

	if rec.Code != http.StatusOK || rec.Body.String() != `{"a":"b"}` {
		t.Errorf("status %d body %q, want the complete response", rec.Code, rec.Body.String())
	}
}

// TestFailFlushesThroughTheStatusWriter: the gateway's own writer wraps the connection, and
// the flush of an error still reaches it, with the status recorded.
func TestFailFlushesThroughTheStatusWriter(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", deps: []depSpec{{"dep-a", "ua"}}}}})
	rec := do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("m", "hi"))

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status %d, want 401", rec.Code)
	}
	if !rec.Flushed {
		t.Error("the 401 was not flushed")
	}
	if got, want := rec.Header().Get("Content-Length"), strconv.Itoa(rec.Body.Len()); got != want {
		t.Errorf("Content-Length = %q, want %q", got, want)
	}
}
