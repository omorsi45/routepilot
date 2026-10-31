package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/reliability"
)

// attemptResult is how one streaming attempt ended, for proxyStream's loop.
type attemptResult struct {
	// done means the request is finished and the loop has nothing left to do:
	// the response was committed (and has ended, cleanly or not), or the client
	// is gone. Otherwise the attempt failed before the commit and err says why.
	done bool
	err  error
	// ctx is the context the attempt ran under, which retryable needs.
	ctx context.Context
	// timedOut reports that the attempt ended because a watchdog fired or the
	// request deadline passed.
	timedOut bool
}

// streamAttempt is one try at one deployment. Everything the watchdogs and the
// held chunks need is fresh for each attempt, so a failed one leaves nothing
// behind for the next.
type streamAttempt struct {
	c      *streamCall
	model  string // catalog model ID served
	dep    catalog.Deployment
	ticket reliability.Ticket // the breaker claim the attempt owes one Record or Abort for

	// ctx is the request context's child, canceled with a cause by the
	// watchdogs and by write failures.
	ctx    context.Context
	cancel context.CancelCauseFunc

	watchdog  *time.Timer
	idlePhase atomic.Bool
	start     time.Time // just before the provider call, for settle's elapsed time
	settled   bool      // the attempt's health outcome was reported

	// usage is the attempt's record for the budget: the usage it reports, the bytes it
	// streams, and how it fails.
	usage *attemptUsage
}

// attempt streams from one deployment, whose breaker claim is ticket. It returns once the
// response has ended or the attempt failed before committing it, having reported the
// attempt's outcome to deployment health either way.
func (c *streamCall) attempt(modelID string, dep catalog.Deployment, ticket reliability.Ticket, up *openai.ChatCompletionRequest, includeUsage bool) attemptResult {
	pcfg := c.s.providerCfgs[dep.Provider]
	ctx, cancel := context.WithCancelCause(c.ctx)
	defer cancel(nil)
	a := &streamAttempt{c: c, model: modelID, dep: dep, ticket: ticket, ctx: ctx, cancel: cancel, usage: c.rs.lastAttempt()}
	// A panic in the provider leaves the breaker claim unresolved: a half-open probe would
	// stay claimed for good. Hand it back and let the panic go on.
	defer func() {
		if v := recover(); v != nil {
			a.abort()
			panic(v)
		}
	}()

	// One timer is the TTFT watchdog until the first committing chunk arrives,
	// then the idle watchdog. It starts before Stream() so it also covers
	// connecting and waiting for response headers. The idle phase times only
	// the wait for the next upstream chunk: it is stopped when Recv returns and
	// reset just before the next Recv, so a slow client blocked in a write
	// never counts.
	a.watchdog = time.AfterFunc(time.Duration(pcfg.TTFTTimeoutMs)*time.Millisecond, func() {
		if a.idlePhase.Load() {
			cancel(errIdleTimeout)
		} else {
			cancel(errTTFTTimeout)
		}
	})
	defer a.watchdog.Stop()
	idle := time.Duration(pcfg.IdleTimeoutMs) * time.Millisecond

	c.rs.usage = nil // only the attempt that answers reports usage
	a.start = c.s.now()
	a.usage.begin(a.start)
	stream, err := c.s.providers[dep.Provider].Stream(ctx, up)
	if err == nil && stream == nil {
		err = openai.ErrInternal()
	}
	if err != nil {
		return a.fail(err)
	}
	defer stream.Close()
	// Make a cancel unblock a Recv even if the provider ignores the context.
	stopClose := context.AfterFunc(ctx, func() { _ = stream.Close() })
	defer stopClose()

	var held [][]byte // encoded non-committing chunks, in arrival order
	var heldBytes int // their total size, capped at maxHeldBytes
	var commitAt, endAt time.Time
	for {
		if c.started {
			a.watchdog.Reset(idle) // time only the wait for the next upstream chunk
		}
		chunk, err := stream.Recv()
		if c.started {
			a.watchdog.Stop()
		}
		if errors.Is(err, io.EOF) {
			endAt = c.s.now()
			break
		}
		if err != nil {
			return a.fail(err)
		}

		a.usage.addStreamed(chunkBytes(chunk))
		if chunk.Usage != nil {
			// Capture before a client that did not ask for usage has it stripped.
			u := *chunk.Usage
			c.rs.usage = &u
			a.usage.reportUsage(&u)
		}
		if !includeUsage {
			chunk.Usage = nil
			if len(chunk.Choices) == 0 {
				continue
			}
		}
		chunk.ID = "chatcmpl-" + c.rs.id
		chunk.Object = openai.ObjectChatCompletionChunk
		chunk.Model = modelID
		if chunk.Created == 0 {
			chunk.Created = c.s.now().Unix()
		}
		if chunk.Choices == nil {
			chunk.Choices = []openai.ChunkChoice{}
		}
		b, err := json.Marshal(chunk)
		if err != nil {
			return a.fail(err)
		}

		if !c.started && !isCommitting(chunk) {
			if len(held) >= maxHeldChunks {
				return a.fail(&provider.Error{Kind: provider.KindProtocol, Message: "too many chunks without content"})
			}
			if heldBytes+len(b) > maxHeldBytes {
				return a.fail(&provider.Error{Kind: provider.KindProtocol, Message: "too many bytes without content"})
			}
			held = append(held, b)
			heldBytes += len(b)
			continue
		}
		if !c.started {
			// The first committing chunk. If the TTFT watchdog got there
			// first, this attempt timed out before committing, whatever the
			// chunk holds.
			if a.ttftFired() {
				return a.fail(&provider.Error{Kind: provider.KindTimeout, Message: "first token timeout"})
			}
			a.idlePhase.Store(true)
			// The time to first token is how long the deployment took to produce the
			// first committing chunk, counted from just before the provider call.
			commitAt = c.s.now()
			c.s.latency.ObserveTTFT(dep.Name, commitAt.Sub(a.start))
			c.startResponse(modelID, dep.Name)
			a.resolveProbeAtCommit()
			for _, h := range held {
				if err := c.send(func() error { return c.sse.WriteData(h) }); err != nil {
					return a.clientGone()
				}
			}
			held = nil
			heldBytes = 0
		}
		if err := c.send(func() error { return c.sse.WriteData(b) }); err != nil {
			return a.clientGone()
		}
	}

	if !c.started {
		// A clean end before any content: nothing useful was received.
		return a.fail(&provider.Error{Kind: provider.KindProtocol, Message: "stream ended without content"})
	}
	a.settle(nil) // the upstream finished cleanly, whatever the client does next
	a.usage.finish(endAt, statusOK)
	a.observeDecode(commitAt, endAt)
	if err := c.send(c.sse.WriteDone); err != nil {
		return a.clientGone()
	}
	return attemptResult{done: true}
}

// observeDecode feeds the latency estimator the time per output token of a stream that
// ended cleanly: the time from the commit to the end of the upstream stream, over the
// completion tokens the upstream reported. Without a usage report, or with at most one
// completion token, there is nothing to divide by and nothing is observed.
func (a *streamAttempt) observeDecode(commitAt, endAt time.Time) {
	u := a.c.rs.usage
	if u == nil || u.CompletionTokens <= 1 {
		return
	}
	a.c.s.latency.ObserveTPOT(a.dep.Name, endAt.Sub(commitAt)/time.Duration(u.CompletionTokens))
}

// ttftFired stops the TTFT watchdog and reports whether it had already fired.
// A watchdog that cannot be stopped has run out its budget, so the first token
// came too late whatever else canceled the attempt first (a disconnect or the
// request deadline): the caller treats it as a pre-commit timeout. The cancel
// makes the cause certain when the timer func has not run yet, and is a no-op
// if something else canceled first.
func (a *streamAttempt) ttftFired() bool {
	if a.watchdog.Stop() {
		return false
	}
	a.cancel(errTTFTTimeout)
	return true
}

// timedOut reports whether the attempt ended because a watchdog fired or the
// request deadline passed, not because of a disconnect or a plain failure.
func (a *streamAttempt) timedOut() bool {
	cause := context.Cause(a.ctx)
	return errors.Is(cause, errTTFTTimeout) || errors.Is(cause, errIdleTimeout) ||
		errors.Is(a.c.ctx.Err(), context.DeadlineExceeded)
}

// settle reports the attempt's outcome to deployment health, once. A nil err
// is a success. An attempt that already resolved its half-open probe at the commit
// (resolveProbeAtCommit) has nothing left to report.
func (a *streamAttempt) settle(err error) {
	if a.settled {
		return
	}
	a.settled = true
	a.c.s.settle(a.ctx, a.c.rs, a.ticket, err, a.c.s.now().Sub(a.start))
}

// resolveProbeAtCommit settles a half-open probe as a success the moment the response
// commits (Ruling V1-R13): the deployment has produced its first committing chunk, which is
// all a probe asks, and a long answer must not keep the deployment half-open for its whole
// length. The ticket is spent, so neither the end of the stream nor a later failure or
// disconnect reports for this attempt again. A closed-state attempt keeps reporting at the
// end of the stream.
func (a *streamAttempt) resolveProbeAtCommit() {
	if a.settled || !a.ticket.Probe() {
		return
	}
	a.settled = true
	a.c.s.health.Record(a.ticket, true)
}

// abort releases the attempt's claim without an outcome, once, because the
// client went away.
func (a *streamAttempt) abort() {
	if a.settled {
		return
	}
	a.settled = true
	a.c.s.health.Abort(a.ticket)
}

// clientGone stops the upstream after a failed client write.
func (a *streamAttempt) clientGone() attemptResult {
	a.cancel(errClientWrite)
	a.c.rs.canceled = true
	a.usage.finish(a.c.s.now(), statusCanceled)
	a.abort()
	return attemptResult{done: true}
}

// fail handles an upstream or local failure of the attempt.
//
// Before the commit it only reports the outcome and returns the error: the
// caller decides between another attempt and a JSON error, and any chunks the
// attempt held are dropped with it. After the commit the status line is sent,
// so it is one SSE error event and no [DONE], which keeps a client from
// mistaking it for a complete answer.
func (a *streamAttempt) fail(err error) attemptResult {
	c := a.c
	a.usage.failed(err)
	a.usage.finish(c.s.now(), attemptEndStatus(err, c.r.Context().Err() != nil, a.timedOut() && timeoutOrCancel(err)))
	if !c.started {
		// No attempt has answered, so usage seen on this one is not the
		// usage of a response and must not drive a refund (M2-R18).
		c.rs.usage = nil
	}
	if c.r.Context().Err() != nil {
		c.rs.canceled = true
		a.abort()
		return attemptResult{done: true}
	}
	if !c.started {
		a.settle(err)
		return attemptResult{err: err, ctx: a.ctx, timedOut: a.timedOut()}
	}

	// The deployment produced output before the request deadline ran out, so
	// that is not its failure.
	if deadlineExpired(a.ctx) {
		a.settle(nil)
	} else {
		a.settle(err)
	}
	apiErr := openai.ErrUpstream("upstream " + a.dep.Name + " failed mid-stream")
	if a.timedOut() {
		apiErr = openai.ErrUpstreamTimeout()
	}
	c.rs.errCode = apiErr.Code
	body := apiErr.Body()
	if err := c.send(func() error { return c.sse.WriteData(body) }); err != nil {
		return a.clientGone()
	}
	return attemptResult{done: true}
}
