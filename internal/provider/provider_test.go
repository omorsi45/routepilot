package provider_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/provider"
)

func TestRetryableStatus(t *testing.T) {
	cases := map[int]bool{
		200: false, 400: false, 401: true, 403: true, 404: false, 408: true,
		409: true, 422: false, 429: true, 500: true, 502: true, 503: true, 504: true,
	}
	for status, want := range cases {
		if got := provider.RetryableStatus(status); got != want {
			t.Errorf("RetryableStatus(%d) = %v, want %v", status, got, want)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		in   string
		want time.Duration
	}{
		{"delta seconds", "3", 3 * time.Second},
		{"http date ahead", now.Add(5 * time.Second).Format(http.TimeFormat), 5 * time.Second},
		{"garbage", "x", 0},
		{"past date", now.Add(-time.Minute).Format(http.TimeFormat), 0},
		{"negative", "-4", 0},
		{"empty", "", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := provider.ParseRetryAfter(tc.in, now); got != tc.want {
				t.Errorf("ParseRetryAfter(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestNewTransportSettings(t *testing.T) {
	tr := provider.NewTransport()
	if tr.MaxIdleConnsPerHost != 256 {
		t.Errorf("MaxIdleConnsPerHost = %d, want 256", tr.MaxIdleConnsPerHost)
	}
	if tr.IdleConnTimeout != 90*time.Second {
		t.Errorf("IdleConnTimeout = %v, want 90s", tr.IdleConnTimeout)
	}
	if !tr.ForceAttemptHTTP2 {
		t.Error("ForceAttemptHTTP2 = false, want true")
	}
	if tr.Proxy == nil {
		t.Error("Proxy not inherited from the default transport")
	}
}

func TestErrorString(t *testing.T) {
	e := &provider.Error{Kind: provider.KindHTTP, Status: 429, Message: "slow down"}
	if got, want := e.Error(), "provider http: slow down (status 429)"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	e = &provider.Error{Kind: provider.KindConnect, Message: "dial failed"}
	if got, want := e.Error(), "provider connect: dial failed"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	cause := errors.New("boom")
	if !errors.Is(&provider.Error{Kind: provider.KindConnect, Err: cause}, cause) {
		t.Error("Unwrap does not expose the cause")
	}
}

func TestClassifyTransportError(t *testing.T) {
	bg := context.Background()
	canceled, cancel := context.WithCancel(bg)
	cancel()
	expired, cancel2 := context.WithDeadline(bg, time.Now().Add(-time.Second))
	defer cancel2()

	cases := []struct {
		name      string
		ctx       context.Context
		err       error
		kind      provider.Kind
		retryable bool
	}{
		{"canceled err", bg, context.Canceled, provider.KindCanceled, false},
		{"canceled ctx", canceled, errors.New("read: use of closed connection"), provider.KindCanceled, false},
		{"deadline err", bg, context.DeadlineExceeded, provider.KindTimeout, true},
		{"deadline ctx", expired, errors.New("i/o timeout"), provider.KindTimeout, true},
		{"other", bg, errors.New("connection refused"), provider.KindConnect, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := provider.ClassifyTransportError(tc.ctx, tc.err)
			if got.Kind != tc.kind || got.Retryable != tc.retryable {
				t.Errorf("got kind=%s retryable=%v, want kind=%s retryable=%v", got.Kind, got.Retryable, tc.kind, tc.retryable)
			}
			if !errors.Is(got, tc.err) {
				t.Error("original error not wrapped")
			}
		})
	}
}

type trackedBody struct {
	io.Reader
	closed bool
}

func (b *trackedBody) Close() error { b.closed = true; return nil }

func TestErrorFromResponse(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	mk := func(status int, body string, hdr http.Header) (*http.Response, *trackedBody) {
		b := &trackedBody{Reader: strings.NewReader(body)}
		return &http.Response{StatusCode: status, Header: hdr, Body: b}, b
	}

	t.Run("openai shaped", func(t *testing.T) {
		resp, body := mk(429, `{"error":{"message":"rate limited","type":"x"}}`, http.Header{"Retry-After": {"2"}})
		e := provider.ErrorFromResponse(resp, now)
		if e.Kind != provider.KindHTTP || e.Status != 429 || !e.Retryable || e.RetryAfter != 2*time.Second || e.Message != "rate limited" {
			t.Errorf("unexpected error: %+v", e)
		}
		if !body.closed {
			t.Error("body not closed")
		}
	})
	t.Run("html body falls back to status text", func(t *testing.T) {
		resp, _ := mk(502, "<html>bad gateway</html>", http.Header{})
		e := provider.ErrorFromResponse(resp, now)
		if e.Message != "Bad Gateway" || strings.Contains(e.Message, "<html>") {
			t.Errorf("Message = %q", e.Message)
		}
	})
	t.Run("json without message falls back", func(t *testing.T) {
		resp, _ := mk(400, `{"error":{}}`, http.Header{})
		if e := provider.ErrorFromResponse(resp, now); e.Message != "Bad Request" || e.Retryable {
			t.Errorf("unexpected error: %+v", e)
		}
	})
	t.Run("read is bounded", func(t *testing.T) {
		huge := &countingReader{}
		resp := &http.Response{StatusCode: 500, Header: http.Header{}, Body: io.NopCloser(huge)}
		provider.ErrorFromResponse(resp, now)
		if huge.n > 64<<10+4096 {
			t.Errorf("read %d bytes, want at most about 64 KiB", huge.n)
		}
	})
}

type countingReader struct{ n int }

func (c *countingReader) Read(p []byte) (int, error) {
	c.n += len(p)
	for i := range p {
		p[i] = 'a'
	}
	return len(p), nil
}
