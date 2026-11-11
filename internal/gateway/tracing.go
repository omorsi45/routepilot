package gateway

import (
	"context"
	"net/http"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
	"github.com/omorsi45/routepilot/internal/telemetry"
)

// Tracing. A chat completion request is one root span, chat.completions, with a child span
// for each stage that can take time or fail:
//
//	admission         the global slot and the tenant's concurrency lease (before the body is
//	                  read), and then the rate limits and token reservation (after the decision)
//	decide            the router; rp.model, rp.policy_version, rp.relaxed, and for the learned
//	                  policy rp.p, rp.l and rp.ood
//	budget            the budget ledger: the read of what is left, and each reservation or raise
//	upstream.attempt  one per attempt sent: rp.deployment, rp.attempt (from 1) and rp.outcome
//
// A span carries identifiers (tenant, request ID, model, deployment, policy version),
// numbers, booleans and outcome names, and nothing else: never message text, a system
// prompt, a tool argument, a message name, a header or key material. A failed span has the
// status Error with the OpenAI error code, or the attempt outcome, as its description, never
// an error message, which could quote upstream text. No span has events or links.
//
// Without a tracer provider every call below lands on a noop span.

const tracerName = "github.com/omorsi45/routepilot/internal/gateway"

// Span names.
const (
	spanRequest   = "chat.completions"
	spanAdmission = "admission"
	spanDecide    = "decide"
	spanBudget    = "budget"
	spanAttempt   = "upstream.attempt"
)

// noSpan is what startSpan returns when tracing is off: a span that records nothing.
var noSpan = trace.SpanFromContext(context.Background())

// setTracerProvider makes tp the provider spans are recorded with; nil, and the noop
// provider, mean none are.
func (s *Server) setTracerProvider(tp trace.TracerProvider) {
	if tp == nil {
		tp = noop.NewTracerProvider()
	}
	_, off := tp.(noop.TracerProvider)
	s.tracing = !off
	s.tracer = tp.Tracer(tracerName)
}

// startTrace begins the root span of a chat completion request.
func (s *Server) startTrace(r *http.Request, rs *reqState) {
	if !s.tracing {
		rs.root = noSpan
		return
	}
	rs.traceCtx, rs.root = s.tracer.Start(r.Context(), spanRequest,
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(attribute.String("rp.request_id", rs.id)))
}

// tagTenant puts the authenticated tenant on the root span.
func (s *Server) tagTenant(rs *reqState) {
	if rs.root != nil { // handleChat can be driven without serve, which starts the trace
		rs.root.SetAttributes(attribute.String("rp.tenant", rs.tenant))
	}
}

// startSpan begins a child of the root span.
func (s *Server) startSpan(rs *reqState, name string, attrs ...attribute.KeyValue) trace.Span {
	if !s.tracing || rs.traceCtx == nil {
		return noSpan
	}
	_, span := s.tracer.Start(rs.traceCtx, name, trace.WithAttributes(attrs...))
	return span
}

// endSpan ends span. A non-empty code marks it failed, with the code as the description.
func endSpan(span trace.Span, code string) {
	if code != "" {
		span.SetStatus(codes.Error, code)
	}
	span.End()
}

// endSpanErr ends span, failed when apiErr is not nil.
func endSpanErr(span trace.Span, apiErr *openai.APIError) {
	if apiErr != nil {
		endSpan(span, apiErr.Code)
		return
	}
	span.End()
}

// traceDecision puts the router's answer on the decide span and ends it. The learned policy
// adds the expected probability of success and the latency of the chosen model, and whether
// the request was out of distribution; a static or pinned decision has neither.
func (s *Server) traceDecision(span trace.Span, rs *reqState, d router.Decision, apiErr *openai.APIError) {
	if span.IsRecording() {
		var attrs []attribute.KeyValue
		chosen := ""
		if len(d.Candidates) > 0 {
			chosen = d.Candidates[0].Model.ID
			attrs = append(attrs, attribute.String("rp.model", chosen))
		}
		if rs.policyVersion != "" {
			attrs = append(attrs, attribute.String("rp.policy_version", rs.policyVersion))
		}
		attrs = append(attrs, attribute.String("rp.relaxed", relaxedLabel(d.Relaxed)))
		if d.PolicyVersion != "" {
			attrs = append(attrs, attribute.Bool("rp.ood", d.OOD))
			for _, sc := range d.Scores {
				if sc.Model != chosen {
					continue
				}
				if p := finite(sc.P); p != nil {
					attrs = append(attrs, attribute.Float64("rp.p", *p))
				}
				if l := finite(sc.L); l != nil {
					attrs = append(attrs, attribute.Float64("rp.l", *l))
				}
			}
		}
		span.SetAttributes(attrs...)
	}
	endSpanErr(span, apiErr)
}

// relaxedLabel lists the constraints a decision relaxed, comma separated, or "none".
func relaxedLabel(relaxed []string) string {
	if len(relaxed) == 0 {
		return telemetry.NoRelaxation
	}
	return strings.Join(relaxed, ",")
}

// endSpan ends the attempt's span with how the attempt ended. It acts once: an attempt that
// was left without an outcome is ended by endTrace as aborted.
func (a *attemptUsage) endSpan(outcome string) {
	if a == nil || a.span == nil {
		return
	}
	span := a.span
	a.span = nil
	span.SetAttributes(attribute.String("rp.outcome", outcome))
	if outcome != statusOK {
		span.SetStatus(codes.Error, outcome)
	}
	span.End()
}

// endTrace ends the root span with the status the client got. Attempt spans a panic left
// open are ended as aborted first, so no child outlives its parent.
func (s *Server) endTrace(sw *statusWriter, rs *reqState) {
	for _, a := range rs.attemptUsages {
		a.endSpan(statusAborted)
	}
	rs.root.SetAttributes(attribute.Int("http.response.status_code", finalStatus(sw, rs)))
	endSpan(rs.root, rs.errCode)
}
