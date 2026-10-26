package gateway

import (
	"context"
	"errors"
	"time"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/reliability"
	"github.com/omorsi45/routepilot/internal/router"
)

const (
	// maxInlineRetryAfter is the longest Retry-After the gateway waits out.
	// A longer one trips the deployment's breaker and the request moves on.
	maxInlineRetryAfter = 2 * time.Second
	// authQuarantine is how long a deployment is skipped after an upstream
	// 401 or 403: the credential is wrong, so retrying it soon is pointless.
	authQuarantine = 120 * time.Second
)

// attemptPlan walks the (model, deployment) pairs of a routing decision in
// order. Each pair it returns has been claimed with Health.Allow, so the
// caller owes exactly one Record or Abort for it. It is used by one request
// goroutine and is not safe for concurrent use.
type attemptPlan struct {
	health *reliability.Health
	cands  []router.Candidate
	max    int

	ci, di int // cursor: the next pair is cands[ci].Deployments[di]
	issued int // pairs returned so far
}

func (s *Server) newAttemptPlan(d router.Decision) *attemptPlan {
	return &attemptPlan{health: s.health, cands: d.Candidates, max: s.cfg.Reliability.MaxAttempts}
}

// next returns the next pair whose breaker admits an attempt. Pairs the
// breaker refuses are skipped and do not count toward the attempt limit.
func (p *attemptPlan) next() (catalog.Model, catalog.Deployment, bool) {
	for p.issued < p.max && p.ci < len(p.cands) {
		c := p.cands[p.ci]
		if p.di >= len(c.Deployments) {
			p.ci, p.di = p.ci+1, 0
			continue
		}
		dep := c.Deployments[p.di]
		p.di++
		if p.health.Allow(dep.Name) {
			p.issued++
			return *c.Model, dep, true
		}
	}
	return catalog.Model{}, catalog.Deployment{}, false
}

// more reports whether next could return a pair, judged by Available so that
// nothing is claimed. The answer can still turn out wrong if another request
// takes a half-open probe first; next then simply returns false.
func (p *attemptPlan) more() bool {
	if p.issued >= p.max {
		return false
	}
	for ci, di := p.ci, p.di; ci < len(p.cands); ci, di = ci+1, 0 {
		for _, dep := range p.cands[ci].Deployments[min(di, len(p.cands[ci].Deployments)):] {
			if p.health.Available(dep.Name) {
				return true
			}
		}
	}
	return false
}

// retryableStatus lists the upstream HTTP statuses worth another attempt. It
// is deliberately the gateway's own list rather than provider.Error.Retryable.
func retryableStatus(status int) bool {
	switch status {
	case 408, 409, 429, 500, 502, 503, 504, 529, 401, 403:
		return true
	}
	return false
}

// retryable reports whether a failed attempt may be followed by another.
// attemptCtx is the context the attempt ran under.
//
// A translation error from the provider adapter (*openai.APIError) is never
// retryable, and neither is a client disconnect. A timeout is retryable only
// while the request deadline has not passed, and a cancel caused by the TTFT
// watchdog counts as a timeout.
func retryable(err error, attemptCtx context.Context) bool {
	var apiErr *openai.APIError
	if err == nil || errors.As(err, &apiErr) {
		return false
	}
	var perr *provider.Error
	if !errors.As(err, &perr) {
		return false
	}
	switch perr.Kind {
	case provider.KindConnect, provider.KindProtocol:
		return true
	case provider.KindTimeout:
		return !errors.Is(attemptCtx.Err(), context.DeadlineExceeded)
	case provider.KindCanceled:
		return errors.Is(context.Cause(attemptCtx), errTTFTTimeout)
	case provider.KindHTTP:
		return retryableStatus(perr.Status)
	}
	return false
}

// healthyOutcome reports whether a failed attempt still proves the deployment
// works: it answered with a non-retryable client error, or the failure was a
// translation error that never depended on the deployment's health.
func healthyOutcome(err error) bool {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return true
	}
	var perr *provider.Error
	return errors.As(err, &perr) && perr.Kind == provider.KindHTTP &&
		perr.Status >= 400 && perr.Status < 500 && !retryableStatus(perr.Status)
}

// deadlineExpired reports whether the attempt ended because the request
// deadline passed, as opposed to a provider watchdog or a disconnect.
func deadlineExpired(ctx context.Context) bool {
	if !errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return false
	}
	cause := context.Cause(ctx)
	return !errors.Is(cause, errTTFTTimeout) && !errors.Is(cause, errIdleTimeout)
}

// settle reports the outcome of an attempt claimed with Allow. A nil err is a
// success. ctx is the context the attempt ran under and elapsed is how long
// the attempt took, from just before the provider call to its return.
//
// A failure caused by the request deadline blames the deployment only when
// the attempt ran for at least the server's default timeout. A shorter run
// was cut off by a client-chosen deadline, which says nothing about the
// deployment, so the claim is released instead.
//
// An upstream 401 or 403 also quarantines the deployment, and a Retry-After
// longer than maxInlineRetryAfter trips its breaker for that long.
func (s *Server) settle(ctx context.Context, dep string, err error, elapsed time.Duration) {
	if err == nil || healthyOutcome(err) {
		s.health.Record(dep, true)
		return
	}
	if deadlineExpired(ctx) {
		if elapsed >= time.Duration(s.cfg.Server.DefaultTimeoutMs)*time.Millisecond {
			s.health.Record(dep, false)
		} else {
			s.health.Abort(dep)
		}
		return
	}
	s.health.Record(dep, false)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		return
	}
	if perr.RetryAfter > maxInlineRetryAfter {
		s.health.Trip(dep, perr.RetryAfter)
	}
	if perr.Kind == provider.KindHTTP && (perr.Status == 401 || perr.Status == 403) {
		s.health.Trip(dep, authQuarantine)
	}
}

// waitBeforeRetry waits before the next attempt after err, the failure of
// attempt number attempt (0 for the first). It honors a Retry-After of at
// most maxInlineRetryAfter and does not wait for a longer one (settle already
// tripped that deployment's breaker), and otherwise sleeps a jittered backoff.
// A wait that would end after the context deadline is not started, and
// neither is a zero wait once the deadline has passed: both return
// context.DeadlineExceeded.
func (s *Server) waitBeforeRetry(ctx context.Context, attempt int, err error) error {
	var wait time.Duration
	var perr *provider.Error
	if errors.As(err, &perr) && perr.RetryAfter > 0 {
		if perr.RetryAfter <= maxInlineRetryAfter {
			wait = perr.RetryAfter
		}
	} else {
		wait = reliability.Backoff(attempt, s.rand)
	}
	// Context deadlines are wall-clock times, so compare with the real clock
	// even when the server clock is injected.
	if deadline, ok := ctx.Deadline(); ok && time.Now().Add(wait).After(deadline) {
		return context.DeadlineExceeded
	}
	if wait <= 0 {
		return nil
	}
	return s.sleep(ctx, wait)
}
