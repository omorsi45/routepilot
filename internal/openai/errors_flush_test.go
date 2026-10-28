package openai

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

// notFlushable is a ResponseWriter with no Flush, like a wrapper that hides it.
type notFlushable struct{ http.ResponseWriter }

// TestWriteErrorFlushesACompleteResponse (Ruling V1-R30): the error is flushed with its
// length declared, so a client has the whole response before the handler returns.
func TestWriteErrorFlushesACompleteResponse(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteError(rec, ErrRateLimited("slow down"))

	if !rec.Flushed {
		t.Error("the error was not flushed")
	}
	if got, want := rec.Header().Get("Content-Length"), strconv.Itoa(rec.Body.Len()); got != want {
		t.Errorf("Content-Length = %q, want %q (the body length)", got, want)
	}
	if rec.Code != http.StatusTooManyRequests {
		t.Errorf("status %d, want 429", rec.Code)
	}
}

// TestWriteErrorWithoutFlushSupport: a writer that cannot flush still gets the whole error.
func TestWriteErrorWithoutFlushSupport(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteError(notFlushable{rec}, ErrInternal())

	if rec.Code != http.StatusInternalServerError || rec.Body.Len() == 0 {
		t.Errorf("status %d, body %q, want a 500 with its body", rec.Code, rec.Body.String())
	}
	if rec.Flushed {
		t.Error("flushed through a writer that hides Flush")
	}
}
