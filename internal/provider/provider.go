// Package provider defines the upstream provider interface, the error model
// shared by every adapter, and the HTTP transport adapters should use.
package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/omorsi45/routepilot/internal/openai"
)

// maxErrorBodyBytes bounds how much of an upstream error body is read.
const maxErrorBodyBytes = 64 << 10

// Provider is one upstream model provider. The caller passes req.Model already
// set to the upstream model and req.RoutePilot nil.
type Provider interface {
	// Complete performs a non-streaming completion.
	Complete(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error)
	// Stream starts a streaming completion. It returns only after a 2xx status
	// and response headers; any earlier failure is returned from Stream itself.
	Stream(ctx context.Context, req *openai.ChatCompletionRequest) (Stream, error)
}

// Stream is a live streaming completion.
type Stream interface {
	// Recv returns the next chunk, or io.EOF after a clean end of stream.
	Recv() (*openai.ChatCompletionChunk, error)
	// Close releases the underlying connection. It is safe to call repeatedly.
	Close() error
}

// Kind classifies a provider failure.
type Kind string

// Failure kinds.
const (
	KindConnect  Kind = "connect"
	KindTimeout  Kind = "timeout"
	KindHTTP     Kind = "http"
	KindProtocol Kind = "protocol"
	KindCanceled Kind = "canceled"
)

// Error is a classified provider failure. Message is safe to log and surface:
// it never carries credentials or raw request or response bodies.
type Error struct {
	Kind       Kind
	Status     int
	Retryable  bool
	RetryAfter time.Duration
	Message    string
	Err        error
}

// Error implements the error interface.
func (e *Error) Error() string {
	s := fmt.Sprintf("provider %s: %s", e.Kind, e.Message)
	if e.Status != 0 {
		s += fmt.Sprintf(" (status %d)", e.Status)
	}
	return s
}

// Unwrap returns the underlying cause, if any.
func (e *Error) Unwrap() error { return e.Err }

// RetryableStatus reports whether an upstream HTTP status is worth retrying on
// another provider or attempt.
func RetryableStatus(status int) bool {
	switch status {
	case 401, 403, 408, 409, 429, 500, 502, 503, 504:
		return true
	}
	return false
}

// ParseRetryAfter parses a Retry-After header value, either delta-seconds or an
// HTTP-date relative to now. Invalid, empty, negative or past values give 0.
func ParseRetryAfter(v string, now time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// NewTransport returns an HTTP transport tuned for many concurrent requests to
// a few upstream hosts. It starts from a clone of http.DefaultTransport so the
// proxy, dial timeout and TLS handshake timeout defaults are kept.
func NewTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxIdleConnsPerHost = 256
	t.IdleConnTimeout = 90 * time.Second
	t.ForceAttemptHTTP2 = true
	return t
}

// ErrorFromResponse builds a KindHTTP error from a non-2xx response. It reads at
// most 64 KiB of the body, closes it, and uses error.message from an
// OpenAI-shaped JSON body, otherwise http.StatusText. Raw body bytes never
// reach Message.
func ErrorFromResponse(resp *http.Response, now time.Time) *Error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
	_ = resp.Body.Close()

	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := ""
	if json.Unmarshal(body, &env) == nil {
		msg = strings.TrimSpace(env.Error.Message)
	}
	if msg == "" {
		msg = http.StatusText(resp.StatusCode)
	}
	if msg == "" {
		msg = "upstream error"
	}
	return &Error{
		Kind:       KindHTTP,
		Status:     resp.StatusCode,
		Retryable:  RetryableStatus(resp.StatusCode),
		RetryAfter: ParseRetryAfter(resp.Header.Get("Retry-After"), now),
		Message:    msg,
	}
}

// ClassifyTransportError maps a transport-level failure to a provider Error.
// Cancellation is not retryable; a deadline is a retryable timeout; anything
// else is a retryable connect failure.
func ClassifyTransportError(ctx context.Context, err error) *Error {
	switch {
	case errors.Is(err, context.Canceled):
		return &Error{Kind: KindCanceled, Message: "request canceled", Err: err}
	case errors.Is(err, context.DeadlineExceeded):
		return &Error{Kind: KindTimeout, Retryable: true, Message: "request timed out", Err: err}
	}
	switch ctx.Err() {
	case context.Canceled:
		return &Error{Kind: KindCanceled, Message: "request canceled", Err: err}
	case context.DeadlineExceeded:
		return &Error{Kind: KindTimeout, Retryable: true, Message: "request timed out", Err: err}
	}
	// Drop the request URL that net/http wraps around the cause.
	cause := err
	var ue *url.Error
	if errors.As(err, &ue) && ue.Err != nil {
		cause = ue.Err
	}
	return &Error{Kind: KindConnect, Retryable: true, Message: "connection failed: " + cause.Error(), Err: err}
}
