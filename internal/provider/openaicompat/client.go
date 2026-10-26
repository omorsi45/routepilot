// Package openaicompat implements provider.Provider for upstreams that speak
// the OpenAI chat completions protocol.
package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
)

// maxCompletionBytes caps a non-streaming success body.
const maxCompletionBytes = 16 << 20

// Client talks to one OpenAI-compatible upstream.
type Client struct {
	url    string
	apiKey string
	client *http.Client
}

var _ provider.Provider = (*Client)(nil)

// New returns a Client for cfg. apiKey may be empty, in which case no
// Authorization header is sent. A nil client selects one built on
// provider.NewTransport. The client is copied and never follows redirects.
func New(cfg config.ProviderConfig, apiKey string, client *http.Client) *Client {
	client = provider.ClientWithoutRedirects(client)
	return &Client{
		url:    strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions",
		apiKey: apiKey,
		client: client,
	}
}

// Complete performs a non-streaming completion.
func (c *Client) Complete(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
	out := req.Clone()
	out.Stream = false
	out.StreamOptions = nil
	resp, err := c.do(ctx, out)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxCompletionBytes+1))
	if err != nil {
		return nil, provider.ClassifyTransportError(ctx, err)
	}
	if len(body) > maxCompletionBytes {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "response body exceeds size limit"}
	}
	if perr := inBandError(body); perr != nil {
		return nil, perr
	}
	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "invalid completion JSON", Err: err}
	}
	if len(completion.Choices) == 0 {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "completion has no choices"}
	}
	return &completion, nil
}

// Stream starts a streaming completion with stream_options.include_usage forced on.
func (c *Client) Stream(ctx context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error) {
	out := req.Clone()
	out.Stream = true
	out.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	resp, err := c.do(ctx, out)
	if err != nil {
		return nil, err
	}
	return &stream{
		ctx:  ctx,
		body: resp.Body,
		sse:  openai.NewSSEReader(resp.Body, openai.MaxEventBytes),
	}, nil
}

// do sends the request and returns a 2xx response. Any other outcome is a
// *provider.Error with the body already drained and closed.
func (c *Client) do(ctx context.Context, body *openai.ChatCompletionRequest) (*http.Response, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "encode request", Err: err}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return nil, &provider.Error{Kind: provider.KindConnect, Message: "build request", Err: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, provider.ClassifyTransportError(ctx, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, provider.ErrorFromResponse(resp, time.Now())
	}
	return resp, nil
}

// stream is a live SSE completion stream. It is not safe for concurrent Recv
// calls; Close may be called from another goroutine.
type stream struct {
	ctx  context.Context
	body io.ReadCloser
	sse  *openai.SSEReader

	mu     sync.Mutex
	closed bool
	err    error // sticky terminal error, io.EOF after a clean end
}

// Recv returns the next chunk, io.EOF after [DONE], or a *provider.Error.
func (s *stream) Recv() (*openai.ChatCompletionChunk, error) {
	s.mu.Lock()
	closed, sticky := s.closed, s.err
	s.mu.Unlock()
	if closed {
		return nil, errStreamClosed()
	}
	if sticky != nil {
		return nil, sticky
	}

	ev, err := s.sse.Next()
	if err != nil {
		// A concurrent Close unblocks the read with a transport error; report
		// that as our own cancellation, not as an upstream connection failure.
		s.mu.Lock()
		closed = s.closed
		s.mu.Unlock()
		if closed {
			return nil, s.fail(errStreamClosed())
		}
		return nil, s.fail(s.classifyRead(err))
	}
	if ev.IsDone() {
		return nil, s.fail(io.EOF)
	}
	if perr := inBandError(ev.Data); perr != nil {
		return nil, s.fail(perr)
	}
	var chunk openai.ChatCompletionChunk
	if err := json.Unmarshal(ev.Data, &chunk); err != nil {
		return nil, s.fail(&provider.Error{Kind: provider.KindProtocol, Message: "invalid chunk JSON", Err: err})
	}
	return &chunk, nil
}

// inBandError reports an error object delivered with a 2xx status. Some
// OpenAI-compatible servers (vLLM among them) signal engine failures that way,
// either as {"error":{...}} or as {"object":"error","message":...}. It returns
// nil for any other body, including invalid JSON, which callers reject on
// their own.
func inBandError(data []byte) *provider.Error {
	var probe struct {
		Object  string          `json:"object"`
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
	}
	if json.Unmarshal(data, &probe) != nil {
		return nil
	}
	hasError := len(probe.Error) > 0 && string(probe.Error) != "null"
	if !hasError && probe.Object != "error" {
		return nil
	}
	msg := probe.Message
	if hasError {
		var obj struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(probe.Error, &obj) == nil {
			msg = obj.Message
		} else {
			var str string
			if json.Unmarshal(probe.Error, &str) == nil {
				msg = str
			}
		}
	}
	msg = provider.TruncateMessage(msg)
	if msg == "" {
		msg = "upstream reported an error in the response body"
	}
	return &provider.Error{Kind: provider.KindProtocol, Message: msg}
}

func errStreamClosed() *provider.Error {
	return &provider.Error{Kind: provider.KindCanceled, Message: "stream closed"}
}

func (s *stream) classifyRead(err error) *provider.Error {
	switch {
	case errors.Is(err, io.EOF):
		return &provider.Error{Kind: provider.KindProtocol, Message: "stream ended before [DONE]", Err: io.ErrUnexpectedEOF}
	case errors.Is(err, openai.ErrEventTooLarge):
		return &provider.Error{Kind: provider.KindProtocol, Message: "stream event exceeds size limit", Err: err}
	}
	return provider.ClassifyTransportError(s.ctx, err)
}

// fail records a terminal error and returns it. A nil *provider.Error never
// reaches here; io.EOF is stored as-is.
func (s *stream) fail(err error) error {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
	return err
}

// Close closes the response body. It is idempotent.
func (s *stream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()
	return s.body.Close()
}
