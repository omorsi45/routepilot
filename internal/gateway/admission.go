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
	if lim.Requests != nil && lim.Tokens != nil && lim.Concurrency != nil {
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

// admit applies the tenant's limits in order: concurrency, request rate, token
// rate. n is the token reservation (InUpper plus the first candidate's K) and
// ttl the lifetime of the concurrency lease. Each limiter call is bounded by
// limiterCallTimeout; one that runs out of time is a limiter error (503).
//
// On false the request is answered (or the client is gone) and nothing is
// held. On true the caller must run finish exactly once, on every exit path:
// it releases the concurrency lease and refunds unused reserved tokens.
func (s *Server) admit(w http.ResponseWriter, rs *reqState, r *http.Request, tenant *config.TenantConfig, n int64, ttl time.Duration) (finish func(), ok bool) {
	ctx, now := r.Context(), s.now()
	tpm := quota.RateLimit{PerMinute: int64(tenant.TPM), Burst: int64(tenant.TPMBurst)}
	var leased, reserved bool

	finish = func() {
		cctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), releaseTimeout)
		defer cancel()
		if reserved {
			if refund := refundAmount(n, rs); refund > 0 {
				if err := s.limits.Tokens.Refund(cctx, tenant.ID, tpm, refund, s.now()); err != nil {
					s.limiterWarn(rs, "token refund", err)
				}
			}
		}
		if leased {
			if err := s.limits.Concurrency.Release(cctx, tenant.ID, rs.id); err != nil {
				s.limiterWarn(rs, "concurrency release", err)
			}
		}
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

	if tenant.MaxConcurrency > 0 {
		cctx, cancel := limiterCtx(ctx)
		got, err := s.limits.Concurrency.Acquire(cctx, tenant.ID, rs.id, tenant.MaxConcurrency, ttl, now)
		cancel()
		if err != nil {
			// The script may have run before the reply failed (a timeout),
			// leaving a lease that would block the tenant for deadline + 60 s.
			// Release it best effort: an absent lease is a no-op.
			leased = true
			return failed("concurrency acquire", err)
		}
		if !got {
			return reject(openai.ErrRateLimited("tenant concurrency limit reached"), "1")
		}
		leased = true
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
