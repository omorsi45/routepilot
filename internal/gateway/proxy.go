package gateway

import (
	"context"
	"errors"
	"net/http"
	"strconv"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/router"
)

// proxyComplete sends a non-streaming request to the decision's deployments in
// order until one answers, then writes the response or the mapped error of the
// last attempt. One deadline, already on ctx, covers every attempt and wait.
func (s *Server) proxyComplete(ctx context.Context, w http.ResponseWriter, rs *reqState, r *http.Request, req *openai.ChatCompletionRequest, dec router.Decision) {
	plan := s.newAttemptPlan(dec)
	plan.bulkhead = s.bulkhead // only non-streaming attempts count against max_inflight
	var (
		lastErr  error
		lastDep  catalog.Deployment
		timedOut bool // a retry wait would have passed the deadline
	)
	for {
		model, dep, up, h, outcome := s.claimNext(ctx, rs, plan, req, lastErr)
		if outcome == claimClientGone {
			return
		}
		if outcome == claimDeadline {
			timedOut = true
		}
		if outcome != claimOK {
			break
		}

		started := s.now()
		resp, err := s.complete(ctx, dep, up, h)
		finished := s.now()
		elapsed := finished.Sub(started)
		if err == nil && resp == nil {
			err = openai.ErrInternal()
		}
		att := rs.lastAttempt() // the record of what this attempt is charged
		att.wholeAnswer = true
		att.begin(started)
		att.finish(finished, attemptEndStatus(err, r.Context().Err() != nil, deadlineExpired(ctx) && timeoutOrCancel(err)))
		att.failed(err)
		switch {
		case err == nil:
			att.observeResponse(resp)
			s.settle(ctx, rs, h.ticket, nil, elapsed)
			rs.usage = resp.Usage
			resp.ID = "chatcmpl-" + rs.id
			resp.Object = openai.ObjectChatCompletion
			resp.Model = model.ID
			if resp.Created == 0 {
				resp.Created = s.now().Unix()
			}
			rs.setRouteHeaders(w, model.ID, dep.Name)
			writeJSON(w, http.StatusOK, resp)
			return
		case r.Context().Err() != nil:
			s.health.Abort(h.ticket)
			rs.canceled = true
			return
		}

		s.settle(ctx, rs, h.ticket, err, elapsed)
		lastErr, lastDep = err, dep
		if !retryable(err, ctx) {
			break
		}
	}

	s.failAttempts(ctx, w, r, rs, lastDep, lastErr, timedOut)
}

// complete makes the provider call of one non-streaming attempt and gives the attempt's
// bulkhead slot back when the call returns, whatever way it ends, a panic included.
func (s *Server) complete(ctx context.Context, dep catalog.Deployment, up *openai.ChatCompletionRequest, h hold) (*openai.ChatCompletion, error) {
	defer h.releaseSlot()
	return s.providers[dep.Provider].Complete(ctx, up)
}

// prepareUpstream records the attempt for the request log and builds the
// request to send upstream. Both the streaming and non-streaming paths use it.
// k is the output token budget the attempt's model was admitted with.
func (s *Server) prepareUpstream(rs *reqState, req *openai.ChatCompletionRequest, modelID string, dep catalog.Deployment, k int) *openai.ChatCompletionRequest {
	rs.model = modelID
	rs.deployment = dep.Name

	up := req.Clone()
	up.Model = dep.UpstreamModel
	up.RoutePilot = nil
	// A request without an output cap is admitted and costed with k, so it is
	// sent upstream with k as its cap: the provider then cannot produce more
	// than was reserved, and an adapter that needs a cap (Anthropic) uses it
	// instead of its own default.
	if _, set := req.EffectiveMaxTokens(); !set {
		up.MaxTokens = &k
	}
	return up
}

// mapUpstreamError converts a provider failure into the error to send. It
// returns nil when the client is gone and nothing should be written. ctx is
// the request context with the deadline applied; clientCtx is the original
// request context.
//
// The result never carries the upstream URL or raw body: provider.Error
// messages are only used for non-retryable 4xx responses, where the upstream's
// own error message is the useful part.
func mapUpstreamError(ctx, clientCtx context.Context, deployment string, err error) *openai.APIError {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return apiErr
	}

	var perr *provider.Error
	if errors.As(err, &perr) {
		if perr.Kind == provider.KindTimeout {
			return openai.ErrUpstreamTimeout()
		}
		if perr.Kind == provider.KindCanceled && clientCtx.Err() != nil {
			return nil
		}
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return openai.ErrUpstreamTimeout()
	}

	reason := "error"
	if perr != nil {
		reason = string(perr.Kind)
		if perr.Kind == provider.KindHTTP {
			if perr.Status >= 400 && perr.Status < 500 && !perr.Retryable {
				if perr.Status == http.StatusBadRequest {
					return openai.ErrInvalidRequest(perr.Message, "")
				}
				return &openai.APIError{Status: perr.Status, Type: "invalid_request_error", Code: "upstream_rejected", Message: perr.Message}
			}
			reason = statusReason(perr.Status)
		}
	}
	return openai.ErrUpstream("upstream " + deployment + " failed: " + reason)
}

// statusReason names an upstream HTTP status for the client-facing message.
// 529 is Anthropic's "overloaded"; any other status without standard text is
// shown as "status <N>".
func statusReason(status int) string {
	if text := http.StatusText(status); text != "" {
		return text
	}
	if status == 529 {
		return "Overloaded"
	}
	return "status " + strconv.Itoa(status)
}
