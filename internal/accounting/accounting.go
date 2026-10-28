// Package accounting is the usage ledger: one row per attempt the gateway sent to a provider,
// kept in SQLite or PostgreSQL behind one Store interface, plus the log of policy rollout
// events. The ledger is the audit trail the budget counters are reconciled against
// (routepilot reconcile); it never takes part in a request's outcome.
package accounting

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/omorsi45/routepilot/internal/money"
)

// UsageRow is one attempt: a request that failed over has one row per deployment it tried.
// (RequestID, Attempt) identifies the row, and Attempt counts from 1 in send order.
type UsageRow struct {
	RequestID  string
	Attempt    int
	Tenant     string
	Model      string
	Deployment string
	// PromptTokens and CompletionTokens are what the attempt was charged for. UsageEstimated
	// is true when they are the gateway's estimate and not the provider's count.
	PromptTokens, CompletionTokens int
	UsageEstimated                 bool
	// CostEstimate is what the routing decision expected the attempt's model to cost and
	// CostActual what the attempt was charged, both in picodollars.
	CostEstimate, CostActual money.PicoUSD
	PolicyVersion            string
	// Status is "ok" for the attempt that answered and otherwise the way it failed
	// ("http_429", "http_500", "timeout", "connect", "protocol", "canceled", ...).
	Status    string
	StartedAt time.Time
	EndedAt   time.Time
}

// Rollout actions accepted in RolloutEvent.Action.
const (
	ActionReload       = "reload"
	ActionCanary       = "canary"
	ActionRollback     = "rollback"
	ActionAutoRollback = "auto_rollback"
	ActionPromote      = "promote"
)

// RolloutEvent is one change to which policy serves traffic.
type RolloutEvent struct {
	At time.Time
	// Action is one of the Action constants.
	Action   string
	From, To string
	// Percent is the share of traffic the event concerns, from 0 to 100.
	Percent float64
	Reason  string
	Actor   string
}

// Store is the usage ledger. Implementations are safe for concurrent use and pass the
// conformance suite in package accountingtest.
type Store interface {
	// RecordAttempt writes a row. It is idempotent on (RequestID, Attempt): a second row with
	// the same key is ignored whole, whatever else it says, and is not an error.
	RecordAttempt(ctx context.Context, row UsageRow) error
	// RecordRollout appends a rollout event.
	RecordRollout(ctx context.Context, event RolloutEvent) error
	// TenantSpend is the sum of CostActual over the tenant's rows with EndedAt in
	// [from, to), in picodollars.
	TenantSpend(ctx context.Context, tenant string, from, to time.Time) (money.PicoUSD, error)
	// Rollouts returns up to limit rollout events, newest first by At, and for equal times
	// the one recorded last first. A limit below 1 is an error.
	Rollouts(ctx context.Context, limit int) ([]RolloutEvent, error)
	// Close releases the store. Closing twice is not an error; using a closed store is.
	Close() error
}

func (r UsageRow) validate() error {
	switch {
	case r.RequestID == "":
		return errors.New("accounting: usage row has no request id")
	case r.Attempt < 1:
		return fmt.Errorf("accounting: usage row attempt %d, want 1 or more", r.Attempt)
	case r.Tenant == "":
		return errors.New("accounting: usage row has no tenant")
	case r.CostActual < 0 || r.CostEstimate < 0:
		return errors.New("accounting: usage row has a negative cost")
	case r.PromptTokens < 0 || r.CompletionTokens < 0:
		return errors.New("accounting: usage row has negative tokens")
	}
	return nil
}

func (e RolloutEvent) validate() error {
	switch e.Action {
	case ActionReload, ActionCanary, ActionRollback, ActionAutoRollback, ActionPromote:
	default:
		return fmt.Errorf("accounting: unknown rollout action %q", e.Action)
	}
	if math.IsNaN(e.Percent) || math.IsInf(e.Percent, 0) {
		return errors.New("accounting: rollout percent is not a finite number")
	}
	return nil
}
