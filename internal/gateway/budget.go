package gateway

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/trace"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/provider"
	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/router"
)

// tenantBudgets returns the budget limit of every tenant that has one, by tenant ID. Tenants
// without budget_usd are absent, and never touch the budget ledger.
func tenantBudgets(cfg *config.Config) (map[string]quota.BudgetLimit, error) {
	budgets := map[string]quota.BudgetLimit{}
	for _, t := range cfg.Tenants {
		lim, ok, err := t.BudgetLimit()
		if err != nil {
			return nil, err
		}
		if ok {
			budgets[t.ID] = lim
		}
	}
	return budgets, nil
}

// BudgetOvershoot is the total, in picodollars, by which committed costs have exceeded their
// reservations since the server started. A reservation is a request's worst case, so it
// stays zero unless a provider bills more than the request allowed (more tokens than
// max_tokens, or a longer prompt than the input bound). The metrics export it as
// rp_budget_overshoot_picousd_total.
func (s *Server) BudgetOvershoot() money.PicoUSD {
	return money.PicoUSD(s.budgetOvershoot.Load())
}

// addOvershoot adds n to the overshoot total, saturating.
func (s *Server) addOvershoot(n money.PicoUSD) {
	for {
		cur := s.budgetOvershoot.Load()
		next := int64(money.AddSat(money.PicoUSD(cur), n))
		if s.budgetOvershoot.CompareAndSwap(cur, next) {
			return
		}
	}
}

// budgetRemaining reads what the tenant may still spend, under the limiter call bound.
func (s *Server) budgetRemaining(ctx context.Context, tenant string, lim quota.BudgetLimit) (money.PicoUSD, error) {
	cctx, cancel := limiterCtx(ctx)
	defer cancel()
	return s.limits.Budget.Remaining(cctx, tenant, lim, s.now())
}

// budgetHold is a request's place in its tenant's budget. The reservation is held under the
// request ID. reserve and the gate raise it, and finish ends it: the cost of the attempts
// that were sent is committed, or the reservation is released when none was. finish may be
// called on every exit path and acts once.
type budgetHold struct {
	s      *Server
	rs     *reqState
	r      *http.Request
	tenant string
	lim    quota.BudgetLimit
	ttl    time.Duration

	// reserved is the amount the ledger holds for the request, as far as this request knows.
	reserved money.PicoUSD
	// held is set when the ledger may hold a reservation: also after a Reserve that failed,
	// because a reply timeout can follow a script that ran.
	held bool
}

// reserve asks the ledger to hold amount for the request, raising any reservation it has.
// It reports whether the ledger accepted. The call is bounded by limiterCallTimeout.
func (h *budgetHold) reserve(ctx context.Context, amount money.PicoUSD) (bool, error) {
	span := h.s.startSpan(h.rs, spanBudget)
	cctx, cancel := limiterCtx(ctx)
	ok, _, err := h.s.limits.Budget.Reserve(cctx, h.tenant, h.rs.id, h.lim, amount, h.ttl, h.s.now())
	cancel()
	// The span fails with the code the client would get: the ledger failing is "overloaded"
	// and a refusal is "insufficient_quota". A client that left is no failure of the ledger.
	switch {
	case err != nil && ctx.Err() == nil:
		endSpan(span, "overloaded")
	case err == nil && !ok:
		endSpan(span, "insufficient_quota")
	default:
		endSpan(span, "")
	}
	if err != nil {
		h.held = true
		return false, err
	}
	if ok {
		h.held = true
		h.reserved = max(h.reserved, amount)
	}
	return ok, nil
}

// gate is the attempt plan's check before it claims a deployment of candidate c. A failover
// can end up charged for every attempt that was sent, so the reservation has to cover what
// those attempts cost so far plus the worst case of this one. When it already does, the
// candidate is approved; otherwise the reservation is raised first, and a candidate whose
// raise is refused (or fails) is skipped.
func (h *budgetHold) gate(ctx context.Context) func(router.Candidate) bool {
	return func(c router.Candidate) bool {
		need := money.AddSat(h.rs.chargedSoFar(), c.CostMax)
		if need <= h.reserved {
			return true
		}
		ok, err := h.reserve(ctx, need)
		if err != nil {
			if ctx.Err() == nil {
				h.s.limiterWarn(h.rs, "budget re-reserve", err)
			}
			return false
		}
		return ok
	}
}

// finish ends the reservation. With no attempt sent it is released; otherwise the summed
// cost of the attempts (Ruling V1-R28) is committed. It runs after the response, under
// releaseTimeout and detached from the request context, so a client that left does not
// leave the reservation behind. A commit above the reservation is counted and logged.
func (h *budgetHold) finish() {
	if !h.held {
		return
	}
	h.held = false
	ctx, cancel := context.WithTimeout(context.WithoutCancel(h.r.Context()), releaseTimeout)
	defer cancel()

	if len(h.rs.attemptUsages) == 0 {
		if err := h.s.limits.Budget.Release(ctx, h.tenant, h.rs.id); err != nil {
			h.s.limiterWarn(h.rs, "budget release", err)
		}
		return
	}
	actual := h.rs.settleAttempts()
	overshoot, err := h.s.limits.Budget.Commit(ctx, h.tenant, h.rs.id, actual, h.s.now())
	if err != nil {
		h.s.limiterWarn(h.rs, "budget commit", err)
		return
	}
	if overshoot > 0 {
		h.s.addOvershoot(overshoot)
		h.s.logger.LogAttrs(context.Background(), slog.LevelWarn, "budget overshoot",
			slog.String("request_id", h.rs.id),
			slog.Int64("reserved_picousd", int64(h.reserved)),
			slog.Int64("actual_picousd", int64(actual)),
			slog.Int64("overshoot_picousd", int64(overshoot)),
		)
	}
}

// attemptUsage is what one sent attempt cost (Ruling V1-R28). The exported fields are the
// record the usage ledger writes once the request is over (resolve fills them in).
//
//   - Reported usage, when the provider gave it: the tokens it counted, priced exactly.
//   - An attempt that failed with an HTTP status or a connect error before any output
//     costs nothing: providers do not bill rejected requests.
//   - Any other attempt without usage (a timeout, a cancel, a failure in the middle of a
//     stream, an answer or a stream that ended without usage) is charged the input bound of
//     its model as prompt tokens and the bytes of content and tool-call deltas it received
//     as completion tokens, never more than the output budget the attempt was sent with
//     (the request carried that cap, so it cannot have produced more tokens). Estimated
//     says so.
type attemptUsage struct {
	Model      string // catalog model ID
	Deployment string
	Prompt     int
	Completion int
	// Estimated is true when the tokens are the bounds above and not the provider's count.
	Estimated bool
	Cost      money.PicoUSD

	priceIn, priceOut money.PicoUSD
	inUpper, maxOut   int
	streamed          int  // bytes of content and tool-call deltas received
	settled           bool // Prompt, Completion, Estimated and Cost are final

	// The fields below feed the usage ledger (ledger.go).

	// costEstimate is what the routing decision expected the model to cost: its expected
	// cost under the learned policy, else its worst case for the request.
	costEstimate money.PicoUSD
	// status is how the attempt ended (see attemptStatus), empty while it is in flight.
	status         string
	started, ended time.Time

	// span is the attempt's trace span, ended with its outcome (tracing.go); nil once ended.
	span trace.Span
}

// newAttemptUsage starts the record of an attempt at model m on dep with output budget k.
func newAttemptUsage(m catalog.Model, dep catalog.Deployment, reqs catalog.Requirements, k int) *attemptUsage {
	return &attemptUsage{
		Model: m.ID, Deployment: dep.Name,
		priceIn: m.PriceIn, priceOut: m.PriceOut,
		inUpper: m.InUpper(reqs), maxOut: k,
	}
}

func (a *attemptUsage) price(prompt, completion int) money.PicoUSD {
	return money.AddSat(money.MulSat(a.priceIn, int64(prompt)), money.MulSat(a.priceOut, int64(completion)))
}

// reportUsage records the usage the provider reported. It is final.
func (a *attemptUsage) reportUsage(u *openai.Usage) {
	if a == nil || u == nil {
		return
	}
	a.Prompt, a.Completion = max(u.PromptTokens, 0), max(u.CompletionTokens, 0)
	a.Estimated = false
	a.Cost = a.price(a.Prompt, a.Completion)
	a.settled = true
}

// addStreamed counts n more bytes of content or tool-call deltas received.
func (a *attemptUsage) addStreamed(n int) {
	if a == nil || n <= 0 {
		return
	}
	a.streamed = min(a.streamed, math.MaxInt-n) + n
}

// failed records that the attempt ended with err. An HTTP status or a connect error before
// any output makes the attempt free; any other failure leaves it to be estimated.
func (a *attemptUsage) failed(err error) {
	if a == nil || a.settled || a.streamed > 0 || err == nil {
		return
	}
	var perr *provider.Error
	if !errors.As(err, &perr) || (perr.Kind != provider.KindHTTP && perr.Kind != provider.KindConnect) {
		return
	}
	a.Prompt, a.Completion, a.Estimated, a.Cost = 0, 0, false, 0
	a.settled = true
}

// charge is what the attempt costs as things stand, without fixing the record.
func (a *attemptUsage) charge() money.PicoUSD {
	if a.settled {
		return a.Cost
	}
	return a.price(a.inUpper, a.estimatedCompletion())
}

func (a *attemptUsage) estimatedCompletion() int {
	return min(a.streamed, max(a.maxOut, 0))
}

// resolve fixes the record: an attempt nothing settled becomes an estimate.
func (a *attemptUsage) resolve() {
	if a == nil || a.settled {
		return
	}
	a.Prompt, a.Completion = a.inUpper, a.estimatedCompletion()
	a.Estimated = true
	a.Cost = a.price(a.Prompt, a.Completion)
	a.settled = true
}

// observeResponse records the answer of a non-streaming attempt: its usage when it has one,
// else the bytes of the answer.
func (a *attemptUsage) observeResponse(resp *openai.ChatCompletion) {
	if resp == nil {
		return
	}
	if resp.Usage != nil {
		a.reportUsage(resp.Usage)
		return
	}
	a.addStreamed(completionBytes(resp))
}

// beginAttempt records an attempt at m on dep with output budget k. Every attempt the plan
// claims has a record, so one that ends in a way nothing accounts for is still charged.
func (rs *reqState) beginAttempt(m catalog.Model, dep catalog.Deployment, k int) *attemptUsage {
	a := newAttemptUsage(m, dep, rs.reqs, k)
	if est, ok := rs.estCost[m.ID]; ok {
		a.costEstimate = est
	} else {
		a.costEstimate = m.CostMax(rs.reqs, k)
	}
	rs.attemptUsages = append(rs.attemptUsages, a)
	return a
}

// lastAttempt is the record of the attempt in flight, nil before the first.
func (rs *reqState) lastAttempt() *attemptUsage {
	if len(rs.attemptUsages) == 0 {
		return nil
	}
	return rs.attemptUsages[len(rs.attemptUsages)-1]
}

// chargedSoFar is what the attempts recorded so far cost, estimates included.
func (rs *reqState) chargedSoFar() money.PicoUSD {
	var sum money.PicoUSD
	for _, a := range rs.attemptUsages {
		sum = money.AddSat(sum, a.charge())
	}
	return sum
}

// settleAttempts fixes every attempt's record and returns what they cost together.
func (rs *reqState) settleAttempts() money.PicoUSD {
	var sum money.PicoUSD
	for _, a := range rs.attemptUsages {
		a.resolve()
		sum = money.AddSat(sum, a.Cost)
	}
	return sum
}

// chunkBytes is the size of the content and tool-call deltas of a streamed chunk: the text
// the model produced, not its framing.
func chunkBytes(chunk *openai.ChatCompletionChunk) int {
	n := 0
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != nil {
			n += len(*ch.Delta.Content)
		}
		n += toolCallBytes(ch.Delta.ToolCalls)
	}
	return n
}

// completionBytes is the size of the content and tool calls of a complete answer.
func completionBytes(resp *openai.ChatCompletion) int {
	if resp == nil {
		return 0
	}
	n := 0
	for _, ch := range resp.Choices {
		n += len(ch.Message.Content.PlainText())
		n += toolCallBytes(ch.Message.ToolCalls)
	}
	return n
}

func toolCallBytes(calls []openai.ToolCall) int {
	n := 0
	for _, tc := range calls {
		n += len(tc.Function.Name) + len(tc.Function.Arguments)
	}
	return n
}
