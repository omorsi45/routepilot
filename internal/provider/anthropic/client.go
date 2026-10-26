// Package anthropic implements provider.Provider for the Anthropic Messages
// API by translating OpenAI chat requests, responses and streams.
package anthropic

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

const (
	// maxCompletionBytes caps a non-streaming success body.
	maxCompletionBytes = 16 << 20
	// apiVersion is the pinned anthropic-version header value.
	apiVersion = "2023-06-01"
	// statusOverloaded is Anthropic's non-standard "overloaded" status.
	statusOverloaded = 529
)

// Client talks to the Anthropic Messages API.
type Client struct {
	url    string
	apiKey string
	client *http.Client
}

var _ provider.Provider = (*Client)(nil)

// New returns a Client for cfg. cfg.BaseURL is the API root without /v1, for
// example https://api.anthropic.com. apiKey may be empty, in which case no
// x-api-key header is sent. A nil client selects one built on
// provider.NewTransport. The client is copied and never follows redirects.
func New(cfg config.ProviderConfig, apiKey string, client *http.Client) *Client {
	client = provider.ClientWithoutRedirects(client)
	return &Client{
		url:    strings.TrimRight(cfg.BaseURL, "/") + "/v1/messages",
		apiKey: apiKey,
		client: client,
	}
}

// Complete performs a non-streaming completion. An untranslatable request is
// returned as a 400 *openai.APIError without contacting the upstream.
func (c *Client) Complete(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
	body, apiErr := translateRequest(req.Clone())
	if apiErr != nil {
		return nil, apiErr
	}
	body.Stream = false
	resp, err := c.do(ctx, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxCompletionBytes+1))
	if err != nil {
		return nil, provider.ClassifyTransportError(ctx, err)
	}
	if len(raw) > maxCompletionBytes {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "response body exceeds size limit"}
	}
	completion, perr := translateResponse(raw)
	if perr != nil {
		return nil, perr
	}
	return completion, nil
}

// Stream starts a streaming completion. An untranslatable request is returned
// as a 400 *openai.APIError without contacting the upstream.
func (c *Client) Stream(ctx context.Context, req *openai.ChatCompletionRequest) (provider.Stream, error) {
	body, apiErr := translateRequest(req.Clone())
	if apiErr != nil {
		return nil, apiErr
	}
	body.Stream = true
	resp, err := c.do(ctx, body)
	if err != nil {
		return nil, err
	}
	return &stream{
		ctx:  ctx,
		body: resp.Body,
		sse:  openai.NewSSEReader(resp.Body, openai.MaxEventBytes),
		tr:   newStreamTranslator(),
	}, nil
}

// do sends the request and returns a 2xx response. Any other outcome is a
// *provider.Error with the body already drained and closed.
func (c *Client) do(ctx context.Context, body *messagesRequest) (*http.Response, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &provider.Error{Kind: provider.KindProtocol, Message: "encode request", Err: err}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return nil, &provider.Error{Kind: provider.KindConnect, Message: "build request", Err: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", apiVersion)
	if c.apiKey != "" {
		httpReq.Header.Set("x-api-key", c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, provider.ClassifyTransportError(ctx, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		perr := provider.ErrorFromResponse(resp, time.Now())
		if perr.Status == statusOverloaded {
			perr.Retryable = true
		}
		return nil, perr
	}
	return resp, nil
}

// stream is a live Messages event stream. It is not safe for concurrent Recv
// calls; Close may be called from another goroutine.
type stream struct {
	ctx  context.Context
	body io.ReadCloser
	sse  *openai.SSEReader
	tr   *streamTranslator

	mu     sync.Mutex
	closed bool
	err    error // sticky terminal error, io.EOF after a clean end
}

// Recv returns the next chunk, io.EOF after message_stop, or a *provider.Error.
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

	for {
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
		chunk, done, perr := s.tr.handle(ev.Name, ev.Data)
		if perr != nil {
			return nil, s.fail(perr)
		}
		if done {
			// The final chunk is returned now; the next Recv reports io.EOF.
			s.mu.Lock()
			s.err = io.EOF
			s.mu.Unlock()
		}
		if chunk != nil {
			return chunk, nil
		}
	}
}

func errStreamClosed() *provider.Error {
	return &provider.Error{Kind: provider.KindCanceled, Message: "stream closed"}
}

func (s *stream) classifyRead(err error) *provider.Error {
	switch {
	case errors.Is(err, io.EOF):
		return &provider.Error{Kind: provider.KindProtocol, Message: "stream ended before message_stop", Err: io.ErrUnexpectedEOF}
	case errors.Is(err, openai.ErrEventTooLarge):
		return &provider.Error{Kind: provider.KindProtocol, Message: "stream event exceeds size limit", Err: err}
	}
	return provider.ClassifyTransportError(s.ctx, err)
}

// fail records a terminal error and returns it.
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
