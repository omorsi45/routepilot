package quota

import (
	"context"
	"time"

	"github.com/omorsi45/routepilot/internal/money"
)

// Budget periods accepted in BudgetLimit.Period.
const (
	// PeriodMonthly counts spend per UTC calendar month and starts again on the first.
	PeriodMonthly = "monthly"
	// PeriodTotal counts spend since the tenant's budget began and never starts again.
	PeriodTotal = "total"
)

// budgetRetention is how long a ledger remembers a request after it ended: the result of
// its commit (so a repeated commit returns it) and, past the expiry of its lease, which
// tenant and period the reservation belonged to (so a late commit is charged where the
// reservation was made).
const budgetRetention = 48 * time.Hour

// BudgetLimit is a tenant's spending limit. Limit is in picodollars; a Limit of zero allows
// only requests that cost nothing. Period is PeriodMonthly or PeriodTotal; anything else,
// the empty string included, is treated as PeriodMonthly.
type BudgetLimit struct {
	Limit  money.PicoUSD
	Period string
}

// PeriodKey names the budget period that contains now: the UTC year and month ("2026-10")
// for a monthly budget, and "total" for a total one.
func PeriodKey(period string, now time.Time) string {
	if period == PeriodTotal {
		return PeriodTotal
	}
	return now.UTC().Format("2006-01")
}

// BudgetLedger keeps what a tenant has spent and what its in-flight requests may still
// spend, so that the sum of both never passes the limit. A request reserves its worst-case
// cost before it is sent, may raise the reservation if it moves to a pricier model, and
// ends by committing what it actually cost or releasing the reservation if nothing was sent.
//
// Every operation is atomic and every backend passes the same conformance suite (see the
// quotatest package). All amounts are non-negative picodollars: a negative amount is read as
// zero, and sums saturate rather than wrap. The caller supplies now explicitly.
//
// A reservation is a lease: Reserve gives it an expiry, and from that instant the lease
// protects nothing. Reserve, Remaining and Commit ignore an expired lease whether or not
// Sweep has removed it yet, so a crashed gateway's reservations stop blocking the tenant
// when they expire. A lease is live while its expiry is after now.
type BudgetLedger interface {
	// Reserve holds amount for request requestID of tenant. It succeeds if and only if
	//
	//	spent + live reserved - (this request's live reservation) + amount <= lim.Limit
	//
	// and then the request's reservation is the larger of its live reservation and amount
	// (a reservation is raised, never lowered) and its lease expires at now+ttl; ttl <= 0
	// means one second. A reservation stays in the period it was first made in, whatever
	// now is by the time it is raised; a new one is made in lim's period at now. It returns
	// whether the amount was reserved and the budget left afterwards, as Remaining would
	// report it; a refusal changes nothing.
	Reserve(ctx context.Context, tenant, requestID string, lim BudgetLimit, amount money.PicoUSD, ttl time.Duration, now time.Time) (ok bool, remaining money.PicoUSD, err error)

	// Commit ends the request by adding actual to the spend of the period its reservation
	// belongs to and removing the reservation. It returns the overshoot, the part of actual
	// above the live reservation: max(0, actual - reserved), where an expired or missing
	// reservation counts as zero. Commit is idempotent by request ID: a second commit
	// returns the first result and changes nothing, for 48 hours.
	//
	// A commit for a request that never reserved (or whose record is gone) cannot know the
	// period, so it adds actual to tenant's monthly period at now and reports all of it as
	// overshoot.
	Commit(ctx context.Context, tenant, requestID string, actual money.PicoUSD, now time.Time) (overshoot money.PicoUSD, err error)

	// Release removes the request's reservation without spending anything. A request with
	// no reservation, or one already committed or released, is a no-op.
	Release(ctx context.Context, tenant, requestID string) error

	// Remaining is max(0, lim.Limit - spent - live reserved) for the period that contains now.
	Remaining(ctx context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error)

	// Spent is the committed spend of the period that contains now.
	Spent(ctx context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error)

	// Sweep removes every lease that expired at or before now, in every tenant and period,
	// and returns how many it removed. Run it periodically: the ledger is correct without
	// it, but it keeps expired leases from piling up. What a request's lease said (its
	// tenant and period) outlives the lease by 48 hours so that a late Commit is charged
	// to the right period; it is forgotten after that.
	Sweep(ctx context.Context, now time.Time) (int, error)
}
