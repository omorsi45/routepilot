package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func feedbackBody(id string, success bool) string {
	return fmt.Sprintf(`{"decision_id":%q,"success":%t}`, id, success)
}

func postFeedback(h http.Handler, key, body string) *httptest.ResponseRecorder {
	return do(h, http.MethodPost, "/v1/feedback", key, body)
}

// TestFeedbackValidation: the endpoint authenticates the tenant, reads at most 4 KiB, refuses
// unknown fields and malformed decision IDs with 400, and answers 204 for a good request.
func TestFeedbackValidation(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	attachLog(t, e.gw)
	h := e.gw.Handler()
	good := "dec_" + strings.Repeat("a1", 12)

	t.Run("authentication", func(t *testing.T) {
		for name, key := range map[string]string{"missing": "", "wrong": "test-key-nobody"} {
			rec := postFeedback(h, key, feedbackBody(good, true))
			if rec.Code != http.StatusUnauthorized || decodeErr(t, rec).Error.Code != "invalid_api_key" {
				t.Errorf("%s key: status %d body %s, want 401 invalid_api_key", name, rec.Code, rec.Body.String())
			}
			if rec.Header().Get("WWW-Authenticate") != "Bearer" {
				t.Errorf("%s key: no WWW-Authenticate header", name)
			}
		}
	})

	t.Run("method", func(t *testing.T) {
		rec := do(h, http.MethodGet, "/v1/feedback", keyA, "")
		if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != "POST" {
			t.Errorf("GET: status %d Allow %q, want 405 and POST", rec.Code, rec.Header().Get("Allow"))
		}
	})

	t.Run("bad requests", func(t *testing.T) {
		for name, c := range map[string]struct{ body, param string }{
			"unknown field":           {`{"decision_id":"` + good + `","success":true,"note":"x"}`, ""},
			"tenant in the body":      {`{"decision_id":"` + good + `","success":true,"tenant":"tenant-b"}`, ""},
			"missing decision id":     {`{"success":true}`, "decision_id"},
			"null decision id":        {`{"decision_id":null,"success":true}`, "decision_id"},
			"decision id too short":   {feedbackBody("dec_abcdefg", true), "decision_id"},
			"decision id too long":    {feedbackBody("dec_"+strings.Repeat("a", 65), true), "decision_id"},
			"no dec_ prefix":          {feedbackBody(strings.Repeat("a", 24), true), "decision_id"},
			"wrong prefix":            {feedbackBody("req_"+strings.Repeat("a", 24), true), "decision_id"},
			"space in the id":         {feedbackBody("dec_abcd efgh", true), "decision_id"},
			"slash in the id":         {feedbackBody("dec_abcd/efgh", true), "decision_id"},
			"newline in the id":       {feedbackBody("dec_abcdefgh\n", true), "decision_id"},
			"non-ASCII in the id":     {feedbackBody("dec_abcdefgé", true), "decision_id"},
			"empty id":                {feedbackBody("", true), "decision_id"},
			"missing success":         {`{"decision_id":"` + good + `"}`, "success"},
			"null success":            {`{"decision_id":"` + good + `","success":null}`, "success"},
			"success as a string":     {`{"decision_id":"` + good + `","success":"yes"}`, ""},
			"success as a number":     {`{"decision_id":"` + good + `","success":1}`, ""},
			"not JSON":                {`not json`, ""},
			"truncated JSON":          {`{"decision_id":`, ""},
			"empty body":              {``, ""},
			"an array":                {`[]`, ""},
			"trailing second value":   {feedbackBody(good, true) + `{"x":1}`, ""},
			"trailing garbage":        {feedbackBody(good, true) + ` xyz`, ""},
			"decision id as a number": {`{"decision_id":12345678,"success":true}`, ""},
		} {
			rec := postFeedback(h, keyA, c.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s: status %d, want 400: %s", name, rec.Code, rec.Body.String())
				continue
			}
			e := decodeErr(t, rec).Error
			if e.Code != "invalid_request" || e.Type != "invalid_request_error" {
				t.Errorf("%s: error %q / %q, want the OpenAI invalid_request shape", name, e.Code, e.Type)
			}
			if c.param != "" && (e.Param == nil || *e.Param != c.param) {
				t.Errorf("%s: param %v, want %q", name, e.Param, c.param)
			}
		}
	})

	t.Run("body size", func(t *testing.T) {
		const limit = 4096 // the contract: a body under 4 KiB
		atLimit := `{"decision_id":"` + good + `","success":true}` + strings.Repeat(" ", limit-len(feedbackBody(good, true)))
		if rec := postFeedback(h, keyA, atLimit); rec.Code != http.StatusNoContent {
			t.Errorf("body of exactly %d bytes: status %d, want 204: %s", limit, rec.Code, rec.Body.String())
		}
		rec := postFeedback(h, keyA, atLimit+" ")
		if rec.Code != http.StatusRequestEntityTooLarge || decodeErr(t, rec).Error.Code != "request_too_large" {
			t.Errorf("body of %d bytes: status %d body %s, want 413 request_too_large", limit+1, rec.Code, rec.Body.String())
		}
	})

	t.Run("accepted", func(t *testing.T) {
		for _, id := range []string{good, "dec_" + strings.Repeat("A", 8), "dec_" + strings.Repeat("z_-", 21) + "9", "dec_" + strings.Repeat("0", 64)} {
			rec := postFeedback(h, keyA, feedbackBody(id, true))
			if rec.Code != http.StatusNoContent || rec.Body.Len() != 0 {
				t.Errorf("id %q: status %d body %q, want 204 and no body", id, rec.Code, rec.Body.String())
			}
		}
		if rec := postFeedback(h, keyA, feedbackBody(good, false)); rec.Code != http.StatusNoContent {
			t.Errorf("success false: status %d, want 204", rec.Code)
		}
	})
}

// TestFeedbackWritesTheAuthenticatedTenant: the line carries the tenant of the API key, and
// nothing from the body can change that.
func TestFeedbackWritesTheAuthenticatedTenant(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	sink := attachLog(t, e.gw)
	h := e.gw.Handler()
	id := "dec_" + strings.Repeat("b2", 12)
	before := time.Now().UTC().Add(-time.Second)

	if rec := postFeedback(h, keyA, feedbackBody(id, true)); rec.Code != http.StatusNoContent {
		t.Fatalf("tenant-a: status %d", rec.Code)
	}
	if rec := postFeedback(h, keyB, feedbackBody(id, false)); rec.Code != http.StatusNoContent {
		t.Fatalf("tenant-b: status %d", rec.Code)
	}
	postFeedback(h, keyA, `{"decision_id":"`+id+`","success":true,"tenant":"tenant-b"}`) // refused

	lines := ofType(sink.records(), "feedback")
	if len(lines) != 2 {
		t.Fatalf("%d feedback lines, want 2 (the refused request writes none): %v", len(lines), lines)
	}
	if lines[0]["tenant"] != "tenant-a" || lines[0]["success"] != true || lines[0]["decision_id"] != id {
		t.Errorf("first line = %v, want tenant-a, success true", lines[0])
	}
	if lines[1]["tenant"] != "tenant-b" || lines[1]["success"] != false {
		t.Errorf("second line = %v, want tenant-b, success false", lines[1])
	}
	at, err := time.Parse(time.RFC3339Nano, lines[0]["time"].(string))
	if err != nil || at.Before(before) || at.After(time.Now().Add(time.Second)) {
		t.Errorf("time = %v (%v), want now in RFC 3339", lines[0]["time"], err)
	}
	if len(lines[0]) != 5 {
		t.Errorf("feedback line has %d fields, want type, decision_id, tenant, success, time: %v", len(lines[0]), lines[0])
	}
}

// TestFeedbackNeedsTheDecisionLog: with no decision log there is nowhere to write, so the
// route does not exist, rather than answering 204 and dropping the feedback.
func TestFeedbackNeedsTheDecisionLog(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	for _, key := range []string{keyA, "test-key-nobody", ""} {
		rec := postFeedback(e.gw.Handler(), key, feedbackBody("dec_"+strings.Repeat("a", 24), true))
		if rec.Code != http.StatusNotFound || decodeErr(t, rec).Error.Code != "not_found" {
			t.Errorf("key %q: status %d body %s, want 404 not_found", key, rec.Code, rec.Body.String())
		}
	}
}

// TestFeedbackIDMatchesTheResponseHeader: the decision ID a response carries is accepted back.
func TestFeedbackIDMatchesTheResponseHeader(t *testing.T) {
	e := newMockEnv(t, cfgOpts{})
	attachLog(t, e.gw)
	h := e.gw.Handler()

	rec := do(h, http.MethodPost, "/v1/chat/completions", keyA, chatBody("model-a", "hi"))
	id := rec.Header().Get(headerDecisionID)
	if rec.Code != http.StatusOK || id == "" {
		t.Fatalf("status %d decision id %q", rec.Code, id)
	}
	if got := postFeedback(h, keyA, feedbackBody(id, true)); got.Code != http.StatusNoContent {
		t.Errorf("feedback with the response's decision id %q: status %d, want 204: %s", id, got.Code, got.Body.String())
	}
}
