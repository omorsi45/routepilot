package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// APIError is an OpenAI-compatible error. Status is the HTTP status; the
// remaining fields populate the JSON error envelope.
type APIError struct {
	Status  int
	Type    string
	Code    string
	Message string
	Param   string
}

// Error implements the error interface.
func (e *APIError) Error() string {
	if e.Param != "" {
		return fmt.Sprintf("%s: %s (param %s)", e.Code, e.Message, e.Param)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

type errorEnvelope struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

// Body returns the JSON error envelope. Param marshals as null when empty.
func (e *APIError) Body() []byte {
	body := errorBody{Message: e.Message, Type: e.Type, Code: e.Code}
	if e.Param != "" {
		p := e.Param
		body.Param = &p
	}
	b, err := json.Marshal(errorEnvelope{Error: body})
	if err != nil {
		// Marshaling plain strings cannot fail; keep a valid fallback anyway.
		return []byte(`{"error":{"message":"internal error","type":"server_error","param":null,"code":"internal_error"}}`)
	}
	return b
}

// WriteError writes e as a JSON response with its HTTP status.
func WriteError(w http.ResponseWriter, e *APIError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Status)
	_, _ = w.Write(e.Body())
}

// ErrInvalidRequest is a 400 for a malformed or invalid request.
func ErrInvalidRequest(msg, param string) *APIError {
	return &APIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "invalid_request", Message: msg, Param: param}
}

// ErrInvalidAPIKey is a 401 for a missing or unknown API key.
func ErrInvalidAPIKey() *APIError {
	return &APIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "invalid_api_key", Message: "invalid or missing API key"}
}

// ErrModelNotAllowed is a 403 when the key may not use the model.
func ErrModelNotAllowed(model string) *APIError {
	return &APIError{Status: http.StatusForbidden, Type: "permission_error", Code: "model_not_allowed", Message: fmt.Sprintf("model %q is not allowed for this API key", model)}
}

// ErrModelNotFound is a 404 when the model is unknown.
func ErrModelNotFound(model string) *APIError {
	return &APIError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found", Message: fmt.Sprintf("model %q does not exist", model)}
}

// ErrCapabilityUnavailable is a 400 when no allowed model can serve the request
// (or the pinned model cannot).
func ErrCapabilityUnavailable(msg string) *APIError {
	return &APIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "model_capability_unavailable", Message: msg, Param: "model"}
}

// ErrNoHealthyDeployment is a 503 when every deployment that could serve the
// request is unavailable.
func ErrNoHealthyDeployment() *APIError {
	return &APIError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "no_healthy_deployment", Message: "no healthy deployment is available for this request"}
}

// ErrRequestTooLarge is a 413 when the body exceeds limit bytes.
func ErrRequestTooLarge(limit int64) *APIError {
	return &APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "request_too_large", Message: fmt.Sprintf("request body exceeds the %d byte limit", limit)}
}

// ErrRequestTimeout is a 408 when the client did not send the body in time.
func ErrRequestTimeout() *APIError {
	return &APIError{Status: http.StatusRequestTimeout, Type: "invalid_request_error", Code: "request_timeout", Message: "request body read timed out"}
}

// ErrRateLimited is a 429.
func ErrRateLimited(msg string) *APIError {
	return &APIError{Status: http.StatusTooManyRequests, Type: "rate_limit_error", Code: "rate_limit_exceeded", Message: msg}
}

// ErrNoFeasibleModel is a 422 when the routing policy rejects a request because no
// model meets its quality floor and latency SLO. The message names the binding
// constraints.
func ErrNoFeasibleModel(msg string) *APIError {
	return &APIError{Status: http.StatusUnprocessableEntity, Type: "invalid_request_error", Code: "no_feasible_model", Message: msg}
}

// ErrInsufficientQuota is a 429 when the remaining budget cannot cover the worst-case
// cost of any model that could serve the request.
func ErrInsufficientQuota(msg string) *APIError {
	return &APIError{Status: http.StatusTooManyRequests, Type: "insufficient_quota", Code: "insufficient_quota", Message: msg}
}

// ErrInternal is a 500.
func ErrInternal() *APIError {
	return &APIError{Status: http.StatusInternalServerError, Type: "server_error", Code: "internal_error", Message: "internal server error"}
}

// ErrUpstream is a 502 for an upstream provider failure.
func ErrUpstream(msg string) *APIError {
	return &APIError{Status: http.StatusBadGateway, Type: "upstream_error", Code: "upstream_error", Message: msg}
}

// ErrOverloaded is a 503 when the gateway cannot take more work.
func ErrOverloaded() *APIError {
	return &APIError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "overloaded", Message: "the service is overloaded, retry later"}
}

// ErrUpstreamTimeout is a 504 when the upstream did not answer in time.
func ErrUpstreamTimeout() *APIError {
	return &APIError{Status: http.StatusGatewayTimeout, Type: "upstream_error", Code: "upstream_timeout", Message: "upstream provider timed out"}
}
