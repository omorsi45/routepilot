package gateway

import (
	"context"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/quota"
)

const (
	// leaseGrace is added to the request deadline to get a concurrency lease
	// TTL, so a lease outlives its request but a crashed gateway's leases
	// still expire.
	leaseGrace = 60 * time.Second
	// releaseTimeout bounds the cleanup calls to the limiters, which run
	// after the request context may already be canceled.
	releaseTimeout = 2 * time.Second
	// limiterCallTimeout bounds each Acquire, Allow and Reserve call, so a
	// stalled limiter backend costs a request at most this long per call
	// instead of holding it (and its global slot) for the whole deadline.
	limiterCallTimeout = 250 * time.Millisecond
)

// limiterCtx derives the context of one limiter call from the request
// context. The caller must call the cancel function once the call returns.
// A call the bound cuts off returns an error while the request context is
// still live, which admit treats as a limiter error.
func limiterCtx(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, limiterCallTimeout)
}

// fillLimiters returns lim with every nil limiter replaced by its in-process
// memory implementation.
func fillLimiters(lim quota.Limiters) quota.Limiters {
	if lim.Requests != nil && lim.Tokens != nil && lim.Concurrency != nil && lim.Budget != nil {
		return lim
	}
	mem := quota.NewMemory()
	if lim.Requests == nil {
		lim.Requests = mem.Requests
	}
	if lim.Tokens == nil {
		lim.Tokens = mem.Tokens
	}
	if lim.Concurrency == nil {
		lim.Concurrency = mem.Concurrency
	}
	if lim.Budget == nil {
		lim.Budget = mem.Budget
	}
	return lim
}

// acquireSlot takes a place in the global in-flight limit, waiting at most
// server.queue_wait_ms. It returns false when the request is over: the wait
// ran out (503 overloaded written) or the client left (nothing written). On
// true the caller owes one releaseSlot. A server without a limit always
// succeeds.
func (s *Server) acquireSlot(w http.ResponseWriter, rs *reqState, r *http.Request) bool {
	if s.sem == nil {
		return true
	}
	select {
	case s.sem <- struct{}{}:
		return true
	default:
	}
	t := time.NewTimer(time.Duration(s.cfg.Server.QueueWaitMs) * time.Millisecond)
	defer t.Stop()
	select {
	case s.sem <- struct{}{}:
		return true
	case <-r.Context().Done():
		rs.canceled = true
		return false
	case <-t.C:
		s.fail(w, rs, openai.ErrOverloaded())
		return false
	}
}

// releaseSlot returns the place taken by acquireSlot.
func (s *Server) releaseSlot() {
	if s.sem != nil {
		<-s.sem
	}
}

// concLease is a request's place in its tenant's concurrency limit. The lease
// ID is the request ID. It is taken right after the global slot, before the
// body is read (acquireLease), and refreshed once the request's real deadline
// is known (admit). release hands it back at most once, so every exit path may
// call it.
type concLease struct {
	s      *Server
	rs     *reqState
	r      *http.Request
	tenant *config.TenantConfig
	// held is set when the limiter may hold the lease: also after an Acquire
	// that failed, because a reply timeout can follow a script that ran.
	held bool
}

// acquire takes the lease, or refreshes it when already held, with the given
// TTL. It does nothing for a tenant without a concurrency limit. It returns
// nil when the lease is held afterwards, else the error to answer with and its
// Retry-After value. A limiter failure is logged and answered as 503.
func (l *concLease) acquire(ctx context.Context, ttl time.Duration) (*openai.APIError, string) {
	if l.tenant.MaxConcurrency <= 0 {
		return nil, ""
	}
	cctx, cancel := limiterCtx(ctx)
	got, err := l.s.limits.Concurrency.Acquire(cctx, l.tenant.ID, l.rs.id, l.tenant.MaxConcurrency, ttl, l.s.now())
	cancel()
	if err != nil {
		// The script may have run before the reply failed (a timeout),
		// leaving a lease that would block the tenant for its whole TTL.
		// Release it best effort: an absent lease is a no-op.
		l.held = true
		if ctx.Err() == nil {
			l.s.limiterWarn(l.rs, "concurrency acquire", err)
		}
		return openai.ErrOverloaded(), ""
	}
	if !got {
		return openai.ErrRateLimited("tenant concurrency limit reached"), "1"
	}
	l.held = true
	return nil, ""
}

// release hands the lease back. It is a no-op when not held or already
// released.
func (l *concLease) release() {
	if !l.held {
		return
	}
	l.held = false
	cctx, cancel := context.WithTimeout(context.WithoutCancel(l.r.Context()), releaseTimeout)
	defer cancel()
	if err := l.s.limits.Concurrency.Release(cctx, l.tenant.ID, l.rs.id); err != nil {
		l.s.limiterWarn(l.rs, "concurrency release", err)
	}
}

// acquireLease takes the tenant's concurrency lease before the body is read, so
// a tenant cannot hold more global slots than its own limit by sending slow
// bodies. The TTL covers the longest deadline the tenant may be given, since
// the request's own is not known yet; admit shortens it. It always returns the
// lease, so the caller can defer its release. On false the request is answered
// (or the client is gone) and nothing is held.
func (s *Server) acquireLease(w http.ResponseWriter, rs *reqState, r *http.Request, tenant *config.TenantConfig) (*concLease, bool) {
	l := &concLease{s: s, rs: rs, r: r, tenant: tenant}
	apiErr, retryAfter := l.acquire(r.Context(), s.maxDeadline(tenant)+leaseGrace)
	if apiErr == nil {
		return l, true
	}
	l.release()
	if r.Context().Err() != nil { // the client left; there is nobody to answer
		rs.canceled = true
		return l, false
	}
	if retryAfter != "" {
		w.Header().Set("Retry-After", retryAfter)
	}
	s.fail(w, rs, apiErr)
	return l, false
}

// admit applies the tenant's remaining limits in order: the concurrency lease
// (refreshed from the one acquireLease took before the body was read), request
// rate, token rate. n is the token reservation (the largest of each candidate's
// model InUpper plus its K) and ttl the new lifetime of the lease. Each
// limiter call is bounded by limiterCallTimeout; one that runs out of time is
// a limiter error (503).
//
// On false the request is answered (or the client is gone) and nothing is
// held. On true the caller must run finish exactly once, on every exit path:
// it releases the concurrency lease and refunds unused reserved tokens.
func (s *Server) admit(w http.ResponseWriter, rs *reqState, r *http.Request, tenant *config.TenantConfig, lease *concLease, n int64, ttl time.Duration) (finish func(), ok bool) {
	ctx, now := r.Context(), s.now()
	tpm := quota.RateLimit{PerMinute: int64(tenant.TPM), Burst: int64(tenant.TPMBurst)}
	var reserved bool

	finish = func() {
		if reserved {
			cctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), releaseTimeout)
			defer cancel()
			if refund := refundAmount(n, rs); refund > 0 {
				if err := s.limits.Tokens.Refund(cctx, tenant.ID, tpm, refund, s.now()); err != nil {
					s.limiterWarn(rs, "token refund", err)
				}
			}
		}
		lease.release()
	}
	reject := func(e *openai.APIError, retryAfter string) (func(), bool) {
		finish()
		if ctx.Err() != nil { // the client left; there is nobody to answer
			rs.canceled = true
			return nil, false
		}
		if retryAfter != "" {
			w.Header().Set("Retry-After", retryAfter)
		}
		s.fail(w, rs, e)
		return nil, false
	}
	failed := func(what string, err error) (func(), bool) {
		if ctx.Err() == nil {
			s.limiterWarn(rs, what, err)
		}
		return reject(openai.ErrOverloaded(), "")
	}

	if apiErr, retryAfter := lease.acquire(ctx, ttl); apiErr != nil {
		return reject(apiErr, retryAfter)
	}
	if tenant.RPM > 0 {
		cctx, cancel := limiterCtx(ctx)
		res, err := s.limits.Requests.Allow(cctx, tenant.ID, quota.RateLimit{PerMinute: int64(tenant.RPM), Burst: int64(tenant.RPMBurst)}, now)
		cancel()
		if err != nil {
			return failed("request rate check", err)
		}
		if !res.Allowed {
			return reject(openai.ErrRateLimited("tenant request rate exceeded"), retryAfterSeconds(res.RetryAfter))
		}
	}
	if tenant.TPM > 0 {
		cctx, cancel := limiterCtx(ctx)
		res, err := s.limits.Tokens.Reserve(cctx, tenant.ID, tpm, n, now)
		cancel()
		if err != nil {
			return failed("token reserve", err)
		}
		if !res.Allowed {
			if res.RetryAfter <= 0 {
				// The reservation can never fit the burst: waiting will not help.
				return reject(openai.ErrRateLimited("request exceeds tenant token burst; lower max_tokens"), "")
			}
			return reject(openai.ErrRateLimited("tenant token rate exceeded"), retryAfterSeconds(res.RetryAfter))
		}
		reserved = true
	}
	return finish, true
}

// limiterWarn logs a limiter failure with its cause. It never logs the tenant
// key hash or API key: only the request ID and the failing operation.
func (s *Server) limiterWarn(rs *reqState, what string, err error) {
	s.logger.LogAttrs(context.Background(), slog.LevelWarn, "quota limiter error",
		slog.String("request_id", rs.id),
		slog.String("operation", what),
		slog.String("error", err.Error()),
	)
}

// refundAmount is how many of the n reserved tokens to hand back once the
// request is over. With no attempt sent nothing was spent, so all of them.
// With attempts sent but no usage known the tokens may have been spent, so
// none. Otherwise the difference to the tokens the answering attempt used.
func refundAmount(n int64, rs *reqState) int64 {
	if rs.attempts == 0 {
		return n
	}
	if rs.usage == nil {
		return 0
	}
	if used := int64(rs.usage.PromptTokens) + int64(rs.usage.CompletionTokens); used < n {
		return n - used
	}
	return 0
}

// retryAfterSeconds renders d as a Retry-After value: whole seconds rounded up,
// at least 1.
func retryAfterSeconds(d time.Duration) string {
	secs := int64(d / time.Second)
	if d%time.Second != 0 {
		secs++
	}
	return strconv.FormatInt(max(secs, 1), 10)
}
