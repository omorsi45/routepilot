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

	started bool // the first chunk was written, so the status line is sent
}

// proxyStream forwards a streaming completion to the model's first deployment.
//
// Exactly one chunk is in flight: read from the upstream, write and flush to
// the client, then read again. A slow client therefore slows the upstream read
// loop (TCP backpressure) instead of growing a buffer.
//
// Before the first chunk is written, failures produce a normal JSON error.
// After it, the status line is already sent, so a failure becomes one SSE error
// event and the stream ends without [DONE]. The upstream is never retried.
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

	// One timer is the TTFT watchdog until the first chunk arrives, then the
	// idle watchdog, reset after every chunk. It starts before Stream() so it
	// also covers connecting and waiting for response headers.
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

	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			c.fail(err)
			return
		}
		idlePhase.Store(true)
		watchdog.Reset(idle)

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
		if !c.started {
			c.startResponse()
		}
		if err := c.send(func() error { return c.sse.WriteData(b) }); err != nil {
			c.clientGone()
			return
		}
	}

	if !c.started {
		// A clean end with no chunks: nothing useful was received.
		s.fail(w, rs, openai.ErrUpstream("upstream "+dep.Name+" returned an empty stream"))
		return
	}
	if err := c.send(c.sse.WriteDone); err != nil {
		c.clientGone()
	}
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
