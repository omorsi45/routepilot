package quota

import (
	"context"
	"sync"
	"time"

	"github.com/omorsi45/routepilot/internal/money"
)

// NewMemoryBudget returns an in-process BudgetLedger. One mutex guards all of its state, so
// every operation is atomic. State is kept for every tenant and period ever seen (a small
// number) and for the requests of the last 48 hours; Sweep forgets the rest.
func NewMemoryBudget() BudgetLedger {
	return &memBudget{
		spent:   map[budgetKey]money.PicoUSD{},
		held:    map[budgetKey]map[string]*memLease{},
		leases:  map[string]*memLease{},
		commits: map[string]memCommit{},
	}
}

// budgetKey is one tenant's budget in one period.
type budgetKey struct{ tenant, period string }

// memLease is a request's reservation. It stays in the leases map after its lease expires
// and is swept (held false), so a late commit still finds the period it belongs to.
type memLease struct {
	id     string
	key    budgetKey
	amount money.PicoUSD
	expiry time.Time
	held   bool // counted in held[key]: not released and not swept
}

// live reports whether the lease still protects the budget at now.
func (l *memLease) live(now time.Time) bool { return l.held && l.expiry.After(now) }

// memCommit is the remembered result of a commit.
type memCommit struct {
	overshoot money.PicoUSD
	at        time.Time
}

type memBudget struct {
	mu      sync.Mutex
	spent   map[budgetKey]money.PicoUSD
	held    map[budgetKey]map[string]*memLease // request ID -> lease, per period
	leases  map[string]*memLease               // request ID -> lease, held or not
	commits map[string]memCommit
}

// nonNeg reads a negative amount as zero.
func nonNeg(a money.PicoUSD) money.PicoUSD { return max(a, 0) }

// budgetLeft is max(0, limit - spent - reserved), without overflow.
func budgetLeft(limit, spent, reserved money.PicoUSD) money.PicoUSD {
	used := money.AddSat(spent, reserved)
	if used >= limit {
		return 0
	}
	return limit - used
}

// liveReserved is the sum of the live reservations of one budget.
func (m *memBudget) liveReserved(k budgetKey, now time.Time) money.PicoUSD {
	var sum money.PicoUSD
	for _, l := range m.held[k] {
		if l.live(now) {
			sum = money.AddSat(sum, l.amount)
		}
	}
	return sum
}

// dropHeld stops l from counting against its budget.
func (m *memBudget) dropHeld(l *memLease) {
	if !l.held {
		return
	}
	l.held = false
	if set := m.held[l.key]; set != nil {
		delete(set, l.id)
		if len(set) == 0 {
			delete(m.held, l.key)
		}
	}
}

func (m *memBudget) Reserve(_ context.Context, tenant, requestID string, lim BudgetLimit, amount money.PicoUSD, ttl time.Duration, now time.Time) (bool, money.PicoUSD, error) {
	amount, limit := nonNeg(amount), nonNeg(lim.Limit)
	if ttl <= 0 {
		ttl = time.Second
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	key := budgetKey{tenant, PeriodKey(lim.Period, now)}
	old := m.leases[requestID]
	var existing money.PicoUSD
	if old != nil && old.key.tenant == tenant && old.live(now) {
		key, existing = old.key, old.amount // a live reservation stays in its own period
	}

	spent, reserved := m.spent[key], m.liveReserved(key, now)
	others := reserved - existing
	used := money.AddSat(spent, others)
	if used > limit || amount > limit-used {
		return false, budgetLeft(limit, spent, reserved), nil
	}

	held := max(existing, amount)
	l := old
	if l == nil {
		l = &memLease{id: requestID}
		m.leases[requestID] = l
	} else {
		m.dropHeld(l)
	}
	l.key, l.amount, l.expiry, l.held = key, held, now.Add(ttl), true
	if m.held[key] == nil {
		m.held[key] = map[string]*memLease{}
	}
	m.held[key][requestID] = l
	return true, budgetLeft(limit, spent, money.AddSat(others, held)), nil
}

func (m *memBudget) Commit(_ context.Context, tenant, requestID string, actual money.PicoUSD, now time.Time) (money.PicoUSD, error) {
	actual = nonNeg(actual)

	m.mu.Lock()
	defer m.mu.Unlock()
	if c, ok := m.commits[requestID]; ok {
		return c.overshoot, nil
	}
	key := budgetKey{tenant, PeriodKey(PeriodMonthly, now)}
	var reserved money.PicoUSD
	if l := m.leases[requestID]; l != nil {
		key = l.key
		if l.live(now) {
			reserved = l.amount
		}
		m.dropHeld(l)
		delete(m.leases, requestID)
	}
	overshoot := max(actual-reserved, 0)
	m.spent[key] = money.AddSat(m.spent[key], actual)
	m.commits[requestID] = memCommit{overshoot: overshoot, at: now}
	return overshoot, nil
}

func (m *memBudget) Release(_ context.Context, _, requestID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if l := m.leases[requestID]; l != nil {
		m.dropHeld(l)
		delete(m.leases, requestID)
	}
	return nil
}

func (m *memBudget) Remaining(_ context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error) {
	key := budgetKey{tenant, PeriodKey(lim.Period, now)}
	m.mu.Lock()
	defer m.mu.Unlock()
	return budgetLeft(nonNeg(lim.Limit), m.spent[key], m.liveReserved(key, now)), nil
}

func (m *memBudget) Spent(_ context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error) {
	key := budgetKey{tenant, PeriodKey(lim.Period, now)}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.spent[key], nil
}

func (m *memBudget) Sweep(_ context.Context, now time.Time) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	swept := 0
	for id, l := range m.leases {
		if l.held && !l.expiry.After(now) {
			m.dropHeld(l)
			swept++
		}
		if !l.held && now.Sub(l.expiry) >= budgetRetention {
			delete(m.leases, id)
		}
	}
	for id, c := range m.commits {
		if now.Sub(c.at) >= budgetRetention {
			delete(m.commits, id)
		}
	}
	return swept, nil
}
