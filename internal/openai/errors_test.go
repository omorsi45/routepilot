package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestErrorBodyShape(t *testing.T) {
	tests := []struct {
		name   string
		err    *APIError
		status int
		typ    string
		code   string
		param  string
	}{
		{"invalid request", ErrInvalidRequest("bad", "model"), 400, "invalid_request_error", "invalid_request", "model"},
		{"invalid request no param", ErrInvalidRequest("bad", ""), 400, "invalid_request_error", "invalid_request", ""},
		{"invalid api key", ErrInvalidAPIKey(), 401, "authentication_error", "invalid_api_key", ""},
		{"model not allowed", ErrModelNotAllowed("gpt-x"), 403, "permission_error", "model_not_allowed", ""},
		{"model not found", ErrModelNotFound("gpt-x"), 404, "invalid_request_error", "model_not_found", ""},
		{"request too large", ErrRequestTooLarge(1024), 413, "invalid_request_error", "request_too_large", ""},
		{"rate limited", ErrRateLimited("slow down"), 429, "rate_limit_error", "rate_limit_exceeded", ""},
		{"internal", ErrInternal(), 500, "server_error", "internal_error", ""},
		{"upstream", ErrUpstream("boom"), 502, "upstream_error", "upstream_error", ""},
		{"overloaded", ErrOverloaded(), 503, "server_error", "overloaded", ""},
		{"upstream timeout", ErrUpstreamTimeout(), 504, "upstream_error", "upstream_timeout", ""},
		{"capability unavailable", ErrCapabilityUnavailable("no model satisfies the request"), 400, "invalid_request_error", "model_capability_unavailable", "model"},
		{"no healthy deployment", ErrNoHealthyDeployment(), 503, "server_error", "no_healthy_deployment", ""},
		{"request timeout", ErrRequestTimeout(), 408, "invalid_request_error", "request_timeout", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.err.Status != tc.status || tc.err.Type != tc.typ || tc.err.Code != tc.code || tc.err.Param != tc.param {
				t.Fatalf("got status=%d type=%q code=%q param=%q", tc.err.Status, tc.err.Type, tc.err.Code, tc.err.Param)
			}
			if tc.err.Message == "" {
				t.Fatal("message must not be empty")
			}
			if tc.err.Error() == "" {
				t.Fatal("Error() must not be empty")
			}

			var body struct {
				Error map[string]any `json:"error"`
			}
			if err := json.Unmarshal(tc.err.Body(), &body); err != nil {
				t.Fatalf("body is not valid JSON: %v", err)
			}
			for _, k := range []string{"message", "type", "param", "code"} {
				if _, ok := body.Error[k]; !ok {
					t.Fatalf("body.error missing key %q: %s", k, tc.err.Body())
				}
			}
			if body.Error["type"] != tc.typ || body.Error["code"] != tc.code || body.Error["message"] != tc.err.Message {
				t.Fatalf("unexpected body: %s", tc.err.Body())
			}
			if tc.param == "" {
				if body.Error["param"] != nil {
					t.Fatalf("param must be null, got %v", body.Error["param"])
				}
			} else if body.Error["param"] != tc.param {
				t.Fatalf("param = %v, want %q", body.Error["param"], tc.param)
			}
		})
	}
}

func TestErrorMessagesMentionContext(t *testing.T) {
	if m := ErrModelNotAllowed("gpt-x").Message; !strings.Contains(m, "gpt-x") {
		t.Fatalf("message %q should mention the model", m)
	}
	if m := ErrModelNotFound("gpt-x").Message; !strings.Contains(m, "gpt-x") {
		t.Fatalf("message %q should mention the model", m)
	}
	if m := ErrRequestTooLarge(1024).Message; !strings.Contains(m, "1024") {
		t.Fatalf("message %q should mention the limit", m)
	}
}

func TestWriteError(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteError(rec, ErrRateLimited("slow down"))
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q", ct)
	}
	if !strings.Contains(rec.Body.String(), `"rate_limit_exceeded"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
}
