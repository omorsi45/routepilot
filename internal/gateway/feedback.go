package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/omorsi45/routepilot/internal/decisionlog"
	"github.com/omorsi45/routepilot/internal/openai"
)

// maxFeedbackBytes is the largest feedback body the endpoint reads.
const maxFeedbackBytes = 4096

// decisionIDPattern is the shape of a decision ID: "dec_" and a request ID. Any ID of this
// shape is accepted. Whether it names a decision of the caller is settled offline, where
// feedback is joined to decisions of the same tenant only (Ruling V1-R17).
var decisionIDPattern = regexp.MustCompile(`^dec_[A-Za-z0-9_-]{8,64}$`)

// maxEchoedFieldBytes caps how much of a client-supplied field name an error repeats.
const maxEchoedFieldBytes = 64

// handleFeedback serves POST /v1/feedback: the client says whether the answer of a decision
// worked out, and the verdict is appended to the decision log under the caller's tenant. The
// route exists only when a decision log is configured; without one there is nowhere to
// write, and answering 204 would silently drop the feedback.
func (s *Server) handleFeedback(w http.ResponseWriter, rs *reqState, r *http.Request) {
	if r.Method != http.MethodPost {
		s.methodNotAllowed(w, rs, http.MethodPost)
		return
	}
	if s.decisions == nil {
		s.fail(w, rs, &openai.APIError{
			Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found",
			Message: "no route for /v1/feedback",
		})
		return
	}
	tenant, ok := s.authenticate(r)
	if !ok {
		s.fail(w, rs, openai.ErrInvalidAPIKey())
		return
	}
	rs.tenant = tenant.ID

	body, apiErr := s.readBodyLimit(w, r, maxFeedbackBytes)
	if apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	in, apiErr := parseFeedback(body)
	if apiErr != nil {
		s.fail(w, rs, apiErr)
		return
	}
	s.decisions.Write(decisionlog.Feedback{
		Type: "feedback", DecisionID: in.decisionID, Tenant: tenant.ID, Success: in.success, Time: s.now().UTC(),
	})
	w.WriteHeader(http.StatusNoContent)
}

type feedbackInput struct {
	decisionID string
	success    bool
}

// parseFeedback reads a feedback body: exactly one JSON object with a well-formed decision_id
// and a boolean success, and no other field. An error never repeats the body.
func parseFeedback(body []byte) (feedbackInput, *openai.APIError) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var raw struct {
		DecisionID *string `json:"decision_id"`
		Success    *bool   `json:"success"`
	}
	if err := dec.Decode(&raw); err != nil {
		return feedbackInput{}, openai.ErrInvalidRequest(feedbackDecodeMessage(err), "")
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return feedbackInput{}, openai.ErrInvalidRequest("the body must be one JSON object", "")
	}
	switch {
	case raw.DecisionID == nil:
		return feedbackInput{}, openai.ErrInvalidRequest("decision_id is required", "decision_id")
	case !decisionIDPattern.MatchString(*raw.DecisionID):
		return feedbackInput{}, openai.ErrInvalidRequest("decision_id must look like dec_ followed by 8 to 64 letters, digits, '-' or '_'", "decision_id")
	case raw.Success == nil:
		return feedbackInput{}, openai.ErrInvalidRequest("success is required and must be true or false", "success")
	}
	return feedbackInput{decisionID: *raw.DecisionID, success: *raw.Success}, nil
}

// feedbackDecodeMessage describes a decode failure without echoing the body, except for the
// name of an unknown field, which is cut short.
func feedbackDecodeMessage(err error) string {
	if name, ok := strings.CutPrefix(err.Error(), "json: unknown field "); ok {
		name = strings.Trim(name, `"`)
		return "unknown field " + `"` + truncateBytes(name, maxEchoedFieldBytes) + `"`
	}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) && typeErr.Field != "" {
		return typeErr.Field + " has the wrong type"
	}
	return "the body must be a JSON object with decision_id and success"
}
