package gateway

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
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

// streamCall is the client-facing state of one streaming request. It outlives
// the individual attempts: only the one that commits writes to the client.
type streamCall struct {
	s   *Server
	w   http.ResponseWriter
	rs  *reqState
	r   *http.Request
	rc  *http.ResponseController
	sse *openai.SSEWriter
	ctx context.Context // the request context with the deadline applied

	started bool // the first committing chunk was written, so the status line is sent
}

// maxHeldChunks caps how many non-committing chunks (such as a role-only
// opener) are kept back while waiting for the first content.
const maxHeldChunks = 64

// maxHeldBytes caps the marshaled size of those held chunks in total, so a
// few chunks that carry large unknown fields cannot grow the buffer either.
const maxHeldBytes = 1 << 20

// proxyStream forwards a streaming completion to the decision's deployments in
// order until one commits, like proxyComplete does for a whole response. One
// deadline, already on ctx, covers every attempt and wait.
//
// Exactly one chunk is in flight: read from the upstream, write and flush to
// the client, then read again. A slow client therefore slows the upstream read
// loop (TCP backpressure) instead of growing a buffer.
//
// An attempt commits the response (headers and status 200 sent) at its first
// committing chunk: one with content text, tool calls, a finish reason, or
// usage that will be forwarded, or another delta field with a real value such
// as reasoning_content. Earlier chunks, such as a role-only opener, are held
// in order (at most maxHeldChunks and maxHeldBytes) and written just before it, and the TTFT
// watchdog keeps running meanwhile, so time to first token is enforced for
// every provider.
//
// Until the commit nothing has reached the client, so a retryable failure
// discards the attempt's held chunks and moves on to the next deployment, and
// a final failure is a normal JSON error. After the commit the status line is
// already sent: a failure becomes one SSE error event, the stream ends without
// [DONE], and the upstream is never retried.
func (s *Server) proxyStream(ctx context.Context, w http.ResponseWriter, rs *reqState, r *http.Request, req *openai.ChatCompletionRequest, dec router.Decision) {
	c := &streamCall{s: s, w: w, rs: rs, r: r, ctx: ctx, rc: http.NewResponseController(w)}
	c.sse = openai.NewSSEWriter(w, c.rc.Flush)

	plan := s.newAttemptPlan(dec)
	var (
		last     attemptResult
		lastDep  catalog.Deployment
		timedOut bool // a retry wait would have passed the deadline
	)
	for {
		model, dep, up, h, outcome := s.claimNext(ctx, rs, plan, req, last.err)
		if outcome == claimClientGone {
			return
		}
		if outcome == claimDeadline {
			timedOut = true
		}
		if outcome != claimOK {
			break
		}
		up.Stream = true

		last, lastDep = c.attempt(model.ID, dep, h.ticket, up, req.StreamOptions != nil && req.StreamOptions.IncludeUsage), dep
		if last.done {
			return
		}
		if !retryable(last.err, last.ctx) {
			break
		}
	}

	s.failAttempts(ctx, w, r, rs, lastDep, last.err, timedOut || last.timedOut)
}

// isCommitting reports whether a chunk carries something worth committing the
// response for: content text, tool calls, a finish reason, usage, or another
// real delta field such as reasoning_content (a reasoning model can stream
// many of those before its first answer token).
func isCommitting(chunk *openai.ChatCompletionChunk) bool {
	if chunk.Usage != nil {
		return true
	}
	for _, ch := range chunk.Choices {
		if ch.FinishReason != nil || len(ch.Delta.ToolCalls) > 0 ||
			(ch.Delta.Content != nil && *ch.Delta.Content != "") || ch.Delta.HasExtraContent() {
			return true
		}
	}
	return false
}

// startResponse writes the SSE headers and status line. It runs at the first
// committing chunk (see isCommitting), so any earlier failure can still be a
// JSON error.
func (c *streamCall) startResponse(modelID, deployment string) {
	h := c.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	c.rs.setRouteHeaders(c.w, modelID, deployment)
	c.rs.commitAt = c.s.now()
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
