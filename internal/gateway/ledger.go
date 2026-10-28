package gateway

import (
	"context"
	"errors"
	"log/slog"
	"strconv"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/provider"
)

// usageWriteTimeout bounds the ledger write of one request: all its rows together.
const usageWriteTimeout = 250 * time.Millisecond

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

// AccountingErrors is the number of usage ledger rows that could not be written since the
// server started: the store failed or did not answer within the write bound. The metrics
// export it as rp_accounting_errors_total.
func (s *Server) AccountingErrors() int64 { return s.accountingErrors.Load() }

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

// recordUsage writes the request's rows to the usage ledger, if there is one. It runs once
// the handler has returned, so the budget commit is done and every cost is final, and it is
// detached from the request context: a client that left still gets its attempts recorded.
// All rows share one usageWriteTimeout. A failure never reaches the client; it is counted
// and logged once.
func (s *Server) recordUsage(rs *reqState) {
	if s.usageLedger == nil || len(rs.attemptUsages) == 0 {
		return
	}
	rows := s.usageRows(rs)
	ctx, cancel := context.WithTimeout(context.Background(), usageWriteTimeout)
	defer cancel()
	var (
		lost  int
		first error
	)
	for _, row := range rows {
		if err := s.usageLedger.RecordAttempt(ctx, row); err != nil {
			lost++
			if first == nil {
				first = err
			}
		}
	}
	if lost == 0 {
		return
	}
	s.accountingErrors.Add(int64(lost))
	s.logger.LogAttrs(context.Background(), slog.LevelWarn, "usage ledger write failed",
		slog.String("request_id", rs.id),
		slog.Int("rows_lost", lost),
		slog.String("error", first.Error()),
	)
}
