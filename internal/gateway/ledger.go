package gateway

import (
	"context"
	"errors"
	"strconv"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/provider"
)

// Attempt statuses other than a provider failure kind or an HTTP status.
const (
	statusOK       = "ok"
	statusCanceled = "canceled"
	statusTimeout  = "timeout"
	// statusAborted marks an attempt the handler left before it recorded how it ended, which
	// only a panic can do.
	statusAborted = "aborted"
	statusError   = "error"
)

// attemptStatus names how an attempt ended: "ok" for success, "http_<status>" for an upstream
// HTTP status, the provider failure kind ("timeout", "connect", "protocol", "canceled") for
// the others, and "error" for anything else.
func attemptStatus(err error) string {
	if err == nil {
		return statusOK
	}
	var perr *provider.Error
	if !errors.As(err, &perr) {
		return statusError
	}
	if perr.Kind == provider.KindHTTP {
		return "http_" + strconv.Itoa(perr.Status)
	}
	return string(perr.Kind)
}

// attemptEndStatus is the status of an attempt that ended with err: "ok" without one, and
// otherwise "canceled" when the client left or "timeout" when a watchdog or the request
// deadline cut the attempt off, whatever error the provider adapter made of that, and else
// attemptStatus.
func attemptEndStatus(err error, clientGone, timedOut bool) string {
	switch {
	case err == nil:
		return statusOK
	case clientGone:
		return statusCanceled
	case timedOut:
		return statusTimeout
	}
	return attemptStatus(err)
}

// begin records when the attempt's provider call started.
func (a *attemptUsage) begin(at time.Time) {
	if a != nil {
		a.started = at
	}
}

// finish records when and how the attempt ended. Only the first call counts, so the outcome
// the upstream gave is not overwritten by a client that leaves afterwards.
func (a *attemptUsage) finish(at time.Time, status string) {
	if a == nil || a.status != "" {
		return
	}
	a.ended, a.status = at, status
	a.endSpan(status)
}

// AccountingErrors is the number of usage ledger writes that were lost since the server
// started: rows the ledger writer's queue had no room for, rows of a batch the store failed to
// write or did not answer within the write bound, rows still queued when shutdown gave up on
// them, and rollout events the store refused. The metrics export it as
// rp_accounting_write_errors_total.
func (s *Server) AccountingErrors() int64 { return s.accountingErrors.Load() }

// attachLedger makes store the usage ledger: the rollout events go to it directly and the
// attempt rows of the requests go through a batcher that writes them off the request path.
// tune sets the batcher's sizes and intervals; the zero value takes the defaults.
func (s *Server) attachLedger(store accounting.Store, tune accounting.BatcherOptions) {
	tune.OnLost = func(n int) { s.accountingErrors.Add(int64(n)) }
	tune.Logger = s.logger
	s.usageLedger = store
	s.usage = accounting.NewBatcher(store, tune)
}

// CloseUsage stops taking usage rows and writes the ones still queued, giving up when ctx
// ends (the rows left are then counted by AccountingErrors). The caller runs it once the HTTP
// server has drained, so no request is still adding rows, and before it closes the store.
// Without a ledger it does nothing, and a second call is harmless.
func (s *Server) CloseUsage(ctx context.Context) error {
	if s.usage == nil {
		return nil
	}
	return s.usage.Close(ctx)
}

// usageRows is the ledger rows of the request: one per attempt sent, numbered from 1 in send
// order. It must run after settleAttempts, so that tokens and costs are final.
func (s *Server) usageRows(rs *reqState) []accounting.UsageRow {
	now := s.now()
	rows := make([]accounting.UsageRow, 0, len(rs.attemptUsages))
	for i, a := range rs.attemptUsages {
		status, ended, started := a.status, a.ended, a.started
		if status == "" {
			status, ended = statusAborted, now
		}
		if started.IsZero() {
			started = ended
		}
		rows = append(rows, accounting.UsageRow{
			RequestID: rs.id, Attempt: i + 1, Tenant: rs.tenant,
			Model: a.Model, Deployment: a.Deployment,
			PromptTokens: a.Prompt, CompletionTokens: a.Completion, UsageEstimated: a.Estimated,
			CostEstimate: a.costEstimate, CostActual: a.Cost,
			PolicyVersion: rs.policyVersion, Status: status,
			StartedAt: started, EndedAt: ended,
		})
	}
	return rows
}

// recordUsage queues the request's rows for the usage ledger, if there is one. It runs once
// the handler has returned, so the budget commit is done and every cost is final. It only
// appends to the ledger writer's queue and never waits for the store: the rows reach the
// store within the writer's flush interval, whether or not the client is still there. A row
// the queue has no room for, or that the store later fails to write, never reaches the
// client; the writer counts it (AccountingErrors) and logs the cause once per failed batch.
func (s *Server) recordUsage(rs *reqState) {
	if s.usage == nil || len(rs.attemptUsages) == 0 {
		return
	}
	s.usage.Enqueue(s.usageRows(rs))
}
