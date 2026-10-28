package gateway

import (
	"context"
	"errors"
	"net/http"
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

	// gate, when set, rules once on each candidate, just before the first of its
	// deployments that looks available is claimed. A candidate it refuses is skipped with all
	// its deployments, and the skip is not an attempt. The budget uses it to raise the
	// request's reservation before a pricier fallback is called.
	gate  func(router.Candidate) bool
	ruled int // index of the last candidate the gate ruled on, -1 before the first

	ci, di int // cursor: the next pair is cands[ci].Deployments[di]
	issued int // pairs returned so far
}

func (s *Server) newAttemptPlan(d router.Decision) *attemptPlan {
	return &attemptPlan{health: s.health, cands: d.Candidates, max: s.cfg.Reliability.MaxAttempts, ruled: -1}
}

// next returns the next pair whose breaker admits an attempt, with the output
// token budget K its model was admitted with. Pairs the breaker refuses, and the
// candidates the gate refuses, are skipped and do not count toward the attempt limit.
func (p *attemptPlan) next() (catalog.Model, catalog.Deployment, int, bool) {
	for p.issued < p.max && p.ci < len(p.cands) {
		c := p.cands[p.ci]
		if p.di >= len(c.Deployments) {
			p.ci, p.di = p.ci+1, 0
			continue
		}
		dep := c.Deployments[p.di]
		p.di++
		if p.gate != nil && p.ruled < p.ci && p.health.Available(dep.Name) {
			p.ruled = p.ci
			if !p.gate(c) {
				p.ci, p.di = p.ci+1, 0
				continue
			}
		}
		if p.health.Allow(dep.Name) {
			p.issued++
			return *c.Model, dep, c.K, true
		}
	}
	return catalog.Model{}, catalog.Deployment{}, 0, false
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

// timeoutOrCancel reports whether err is a timeout or a cancel: a provider
// KindTimeout or KindCanceled, or a context error. Only these can be blamed on
// the request deadline rather than on the deployment.
func timeoutOrCancel(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return true
	}
	var perr *provider.Error
	return errors.As(err, &perr) && (perr.Kind == provider.KindTimeout || perr.Kind == provider.KindCanceled)
}

// deadlineBlameNum and deadlineBlameDen make the share of the request's blame
// budget (9/10) an attempt must have used for a deadline expiry to count
// against its deployment.
const deadlineBlameNum, deadlineBlameDen = 9, 10

// settle reports the outcome of an attempt claimed with Allow. A nil err is a
// success. ctx is the context the attempt ran under and elapsed is how long
// the attempt took, from just before the provider call to its return.
//
// A timeout or cancel caused by the request deadline blames the deployment
// only when the attempt ran for at least 90% of rs.blameBudget, the deadline
// the request would have had without a client timeout_ms (see blameBudget).
// A shorter run was cut off by a client-chosen deadline, which says nothing
// about the deployment, so the claim is released instead. Any other failure
// that happens to arrive near the deadline, such as a 401 or a long
// Retry-After, is still the deployment's.
//
// An upstream 401 or 403 also quarantines the deployment, and a Retry-After
// longer than maxInlineRetryAfter trips its breaker for that long, capped at
// reliability.MaxOpenFor so one upstream value cannot park it for hours.
func (s *Server) settle(ctx context.Context, rs *reqState, dep string, err error, elapsed time.Duration) {
	if err != nil {
		rs.noteUpstreamError(err)
	}
	if err == nil || healthyOutcome(err) {
		s.health.Record(dep, true)
		return
	}
	if deadlineExpired(ctx) && timeoutOrCancel(err) {
		if elapsed*deadlineBlameDen >= rs.blameBudget*deadlineBlameNum {
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
		s.health.Trip(dep, min(perr.RetryAfter, reliability.MaxOpenFor))
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

// claimOutcome says how claimNext ended.
type claimOutcome int

const (
	claimOK         claimOutcome = iota // a pair was claimed and the upstream request built
	claimNone                           // no deployment is left to try
	claimDeadline                       // the request deadline passed or would pass during the wait
	claimClientGone                     // the client disconnected during the wait
)

// claimNext owns the retry gate shared by the streaming and non-streaming
// paths. For the first attempt it just claims the next pair. After a failure
// (lastErr) it checks that another pair can be tried, waits out the backoff or
// Retry-After, and rechecks the deadline before claiming anything, so a wait
// that ran past the deadline never takes a half-open probe. For a tenant with a budget
// it also raises the reservation before claiming a pair of a pricier model, and skips the
// model when the raise is refused. On claimOK it has counted the attempt and recorded it
// for the request log and the budget (rs.attemptUsages); the caller owes the pair exactly
// one Record or Abort. A claimClientGone has already marked the request canceled.
func (s *Server) claimNext(ctx context.Context, rs *reqState, plan *attemptPlan, req *openai.ChatCompletionRequest, lastErr error) (catalog.Model, catalog.Deployment, *openai.ChatCompletionRequest, claimOutcome) {
	if rs.attempts > 0 {
		if !plan.more() {
			return catalog.Model{}, catalog.Deployment{}, nil, claimNone
		}
		waitErr := s.waitBeforeRetry(ctx, rs.attempts-1, lastErr)
		// Check the context even when the wait succeeded (a zero wait never
		// looks at it): claim nothing once the deadline passed or the client
		// left.
		if ctxErr := ctx.Err(); waitErr != nil || ctxErr != nil {
			if errors.Is(waitErr, context.DeadlineExceeded) || errors.Is(ctxErr, context.DeadlineExceeded) {
				return catalog.Model{}, catalog.Deployment{}, nil, claimDeadline
			}
			rs.canceled = true // the client went away while we waited
			return catalog.Model{}, catalog.Deployment{}, nil, claimClientGone
		}
	}
	if rs.budget != nil {
		plan.gate = rs.budget.gate(ctx)
	}
	model, dep, k, ok := plan.next()
	if !ok {
		return catalog.Model{}, catalog.Deployment{}, nil, claimNone
	}
	rs.attempts++
	rs.beginAttempt(model, dep, k)
	return model, dep, s.prepareUpstream(rs, req, model.ID, dep, k), claimOK
}

// failAttempts writes the error for a request whose attempts are over without
// a response: 503 if nothing was sent upstream, 504 if the deadline passed or
// timedOut says the last attempt (or the next wait) ran out of time, 429 if every
// attempt that was sent ended in an upstream 429 (the client is told to back off for as
// long as the shortest Retry-After an upstream gave), else the mapped error of lastErr,
// which lastDep produced. Nothing is written when the client is gone.
func (s *Server) failAttempts(ctx context.Context, w http.ResponseWriter, r *http.Request, rs *reqState, lastDep catalog.Deployment, lastErr error, timedOut bool) {
	if r.Context().Err() != nil {
		rs.canceled = true
		return
	}
	switch {
	case rs.attempts == 0:
		s.fail(w, rs, openai.ErrNoHealthyDeployment())
	case timedOut || errors.Is(ctx.Err(), context.DeadlineExceeded):
		s.fail(w, rs, openai.ErrUpstreamTimeout())
	case rs.allUpstream429():
		s.failUpstreamRateLimited(w, rs)
	default:
		if apiErr := mapUpstreamError(ctx, r.Context(), lastDep.Name, lastErr); apiErr != nil {
			s.fail(w, rs, apiErr)
		} else {
			rs.canceled = true
		}
	}
}
