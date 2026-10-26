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
	var (
		lastErr  error
		lastDep  catalog.Deployment
		timedOut bool // a retry wait would have passed the deadline
	)
	for {
		if rs.attempts > 0 {
			if !plan.more() {
				break
			}
			if err := s.waitBeforeRetry(ctx, rs.attempts-1, lastErr, lastDep.Name); err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					timedOut = true
					break
				}
				rs.canceled = true // the client went away while we waited
				return
			}
		}
		model, dep, ok := plan.next()
		if !ok {
			break
		}
		rs.attempts++
		up := s.prepareUpstream(rs, req, model.ID, dep)

		resp, err := s.providers[dep.Provider].Complete(ctx, up)
		if err == nil && resp == nil {
			err = openai.ErrInternal()
		}
		switch {
		case err == nil:
			s.settle(dep.Name, nil)
			resp.ID = "chatcmpl-" + rs.id
			resp.Object = openai.ObjectChatCompletion
			resp.Model = model.ID
			if resp.Created == 0 {
				resp.Created = s.now().Unix()
			}
			setRouteHeaders(w, model.ID, dep.Name)
			writeJSON(w, http.StatusOK, resp)
			return
		case r.Context().Err() != nil:
			s.health.Abort(dep.Name)
			rs.canceled = true
			return
		}

		s.settle(dep.Name, err)
		lastErr, lastDep = err, dep
		if !retryable(err, ctx) {
			break
		}
	}

	switch {
	case rs.attempts == 0:
		s.fail(w, rs, openai.ErrNoHealthyDeployment())
	case timedOut || errors.Is(ctx.Err(), context.DeadlineExceeded):
		s.fail(w, rs, openai.ErrUpstreamTimeout())
	default:
		if apiErr := mapUpstreamError(ctx, r.Context(), lastDep.Name, lastErr); apiErr != nil {
			s.fail(w, rs, apiErr)
		} else {
			rs.canceled = true
		}
	}
}

// prepareUpstream records the attempt for the request log and builds the
// request to send upstream. Both the streaming and non-streaming paths use it.
func (s *Server) prepareUpstream(rs *reqState, req *openai.ChatCompletionRequest, modelID string, dep catalog.Deployment) *openai.ChatCompletionRequest {
	rs.model = modelID
	rs.deployment = dep.Name

	up := req.Clone()
	up.Model = dep.UpstreamModel
	up.RoutePilot = nil
	return up
}

// setRouteHeaders tells the client which model and deployment served it.
func setRouteHeaders(w http.ResponseWriter, modelID, deployment string) {
	w.Header().Set("x-routepilot-model", modelID)
	w.Header().Set("x-routepilot-deployment", deployment)
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
