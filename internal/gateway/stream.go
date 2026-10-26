package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
)

// clientWriteTimeout bounds each write to the client, so a client that stops
// reading cannot hold the handler and the upstream connection forever.
const clientWriteTimeout = 30 * time.Second

// Cancellation causes for the per-attempt context. They tell a watchdog cancel
// apart from a client disconnect and from the request deadline.
var (
	errTTFTTimeout = errors.New("gateway: upstream first token timeout")
	errIdleTimeout = errors.New("gateway: upstream idle timeout")
	errClientWrite = errors.New("gateway: client write failed")
)

// streamCall is the state of one streaming request.
type streamCall struct {
	s     *Server
	w     http.ResponseWriter
	rs    *reqState
	r     *http.Request
	rc    *http.ResponseController
	sse   *openai.SSEWriter
	model config.ModelConfig
	dep   config.DeploymentConfig

	// ctx is the request context with the deadline applied; attemptCtx is its
	// child, canceled with a cause by the watchdogs and by write failures.
	ctx        context.Context
	attemptCtx context.Context
	cancel     context.CancelCauseFunc

	started bool // the first committing chunk was written, so the status line is sent
}

// maxHeldChunks caps how many non-committing chunks (such as a role-only
// opener) are kept back while waiting for the first content.
const maxHeldChunks = 64

// proxyStream forwards a streaming completion to the model's first deployment.
//
// Exactly one chunk is in flight: read from the upstream, write and flush to
// the client, then read again. A slow client therefore slows the upstream read
// loop (TCP backpressure) instead of growing a buffer.
//
// The response is committed (headers and status 200 sent) at the first
// committing chunk: one with content text, tool calls, a finish reason, or
// usage that will be forwarded. Earlier chunks, such as a role-only opener,
// are held in order (at most maxHeldChunks) and written just before it, and
// the TTFT watchdog keeps running meanwhile, so time to first token is
// enforced for every provider.
//
// Before the commit, failures produce a normal JSON error. After it, the
// status line is already sent, so a failure becomes one SSE error event and
// the stream ends without [DONE]. The upstream is never retried.
func (s *Server) proxyStream(ctx context.Context, w http.ResponseWriter, rs *reqState, r *http.Request, req *openai.ChatCompletionRequest, model config.ModelConfig) {
	dep, up := s.prepareUpstream(rs, req, model)
	up.Stream = true
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	pcfg := s.providerCfgs[dep.Provider]

	attemptCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	c := &streamCall{
		s: s, w: w, rs: rs, r: r, model: model, dep: dep,
		ctx: ctx, attemptCtx: attemptCtx, cancel: cancel,
		rc: http.NewResponseController(w),
	}
	c.sse = openai.NewSSEWriter(w, c.rc.Flush)

	// One timer is the TTFT watchdog until the first committing chunk arrives,
	// then the idle watchdog. It starts before Stream() so it also covers connecting and
	// waiting for response headers. The idle phase times only the wait for the
	// next upstream chunk: it is stopped when Recv returns and reset just before
	// the next Recv, so a slow client blocked in a write never counts.
	var idlePhase atomic.Bool
	watchdog := time.AfterFunc(time.Duration(pcfg.TTFTTimeoutMs)*time.Millisecond, func() {
		if idlePhase.Load() {
			cancel(errIdleTimeout)
		} else {
			cancel(errTTFTTimeout)
		}
	})
	defer watchdog.Stop()
	idle := time.Duration(pcfg.IdleTimeoutMs) * time.Millisecond

	stream, err := s.providers[dep.Provider].Stream(attemptCtx, up)
	if err == nil && stream == nil {
		err = openai.ErrInternal()
	}
	if err != nil {
		c.fail(err)
		return
	}
	defer stream.Close()
	// Make a cancel unblock a Recv even if the provider ignores the context.
	stopClose := context.AfterFunc(attemptCtx, func() { _ = stream.Close() })
	defer stopClose()

	var held [][]byte // encoded non-committing chunks, in arrival order
	for {
		if c.started {
			watchdog.Reset(idle) // time only the wait for the next upstream chunk
		}
		chunk, err := stream.Recv()
		if c.started {
			watchdog.Stop()
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			c.fail(err)
			return
		}

		if !includeUsage {
			chunk.Usage = nil
			if len(chunk.Choices) == 0 {
				continue
			}
		}
		chunk.ID = "chatcmpl-" + rs.id
		chunk.Model = model.ID
		if chunk.Created == 0 {
			chunk.Created = s.now().Unix()
		}
		if chunk.Choices == nil {
			chunk.Choices = []openai.ChunkChoice{}
		}
		b, err := json.Marshal(chunk)
		if err != nil {
			c.fail(err)
			return
		}

		if !c.started && !isCommitting(chunk) {
			if len(held) >= maxHeldChunks {
				s.fail(w, rs, openai.ErrUpstream("upstream "+dep.Name+" sent too many chunks without content"))
				return
			}
			held = append(held, b)
			continue
		}
		if !c.started {
			// The first committing chunk: stop the TTFT phase, switch to the
			// idle phase, then send headers and the held chunks before it.
			watchdog.Stop()
			idlePhase.Store(true)
			c.startResponse()
			for _, h := range held {
				if err := c.send(func() error { return c.sse.WriteData(h) }); err != nil {
					c.clientGone()
					return
				}
			}
			held = nil
		}
		if err := c.send(func() error { return c.sse.WriteData(b) }); err != nil {
			c.clientGone()
			return
		}
	}

	if !c.started {
		// A clean end before any content: nothing useful was received.
		if c.timedOut() {
			s.fail(w, rs, openai.ErrUpstreamTimeout())
			return
		}
		s.fail(w, rs, openai.ErrUpstream("upstream "+dep.Name+" returned an empty stream"))
		return
	}
	if err := c.send(c.sse.WriteDone); err != nil {
		c.clientGone()
	}
}

// isCommitting reports whether a chunk carries something worth committing the
// response for: content text, tool calls, a finish reason, or usage.
func isCommitting(chunk *openai.ChatCompletionChunk) bool {
	if chunk.Usage != nil {
		return true
	}
	for _, ch := range chunk.Choices {
		if ch.FinishReason != nil || len(ch.Delta.ToolCalls) > 0 || (ch.Delta.Content != nil && *ch.Delta.Content != "") {
			return true
		}
	}
	return false
}

// startResponse writes the SSE headers and status line. It runs when the first
// chunk is ready, so earlier failures can still be a JSON error.
func (c *streamCall) startResponse() {
	h := c.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	setRouteHeaders(c.w, c.model.ID, c.dep.Name)
	c.w.WriteHeader(http.StatusOK)
	c.started = true
}

// send applies the per-write deadline, then performs one write and flush. A
// server that cannot set deadlines (such as a test recorder) is not an error.
func (c *streamCall) send(write func() error) error {
	if err := c.rc.SetWriteDeadline(time.Now().Add(clientWriteTimeout)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return err
	}
	return write()
}

// clientGone stops the upstream after a failed client write.
func (c *streamCall) clientGone() {
	c.cancel(errClientWrite)
	c.rs.canceled = true
}

// timedOut reports whether the attempt ended because a watchdog fired or the
// request deadline passed, not because of a disconnect or a plain failure.
func (c *streamCall) timedOut() bool {
	cause := context.Cause(c.attemptCtx)
	return errors.Is(cause, errTTFTTimeout) || errors.Is(cause, errIdleTimeout) ||
		errors.Is(c.ctx.Err(), context.DeadlineExceeded)
}

// fail ends the request after an upstream or local failure.
//
// Before the first byte it is a normal JSON error (or nothing, when the client
// is gone). After it, the status line is sent, so it is one SSE error event and
// no [DONE], which keeps a client from mistaking it for a complete answer.
func (c *streamCall) fail(err error) {
	if c.r.Context().Err() != nil {
		c.rs.canceled = true
		return
	}
	if !c.started {
		if c.timedOut() {
			c.s.fail(c.w, c.rs, openai.ErrUpstreamTimeout())
		} else if apiErr := mapUpstreamError(c.ctx, c.r.Context(), c.dep.Name, err); apiErr != nil {
			c.s.fail(c.w, c.rs, apiErr)
		} else {
			c.rs.canceled = true
		}
		return
	}

	apiErr := openai.ErrUpstream("upstream " + c.dep.Name + " failed mid-stream")
	if c.timedOut() {
		apiErr = openai.ErrUpstreamTimeout()
	}
	c.rs.errCode = apiErr.Code
	body := apiErr.Body()
	if err := c.send(func() error { return c.sse.WriteData(body) }); err != nil {
		c.clientGone()
	}
}
