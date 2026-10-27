package quota

import (
	"context"
	"sync"
	"time"
)

// NewMemory returns in-process limiters. Each limiter kind has its own state
// and its own mutex; keys are opaque strings. The memory backend never
// returns an error: the error results exist for networked backends.
//
// State is kept for every key ever seen and is not evicted; callers use a
// bounded key space (tenant, model).
func NewMemory() Limiters {
	newStore := func() *gcraStore { return &gcraStore{entries: map[string]*gcraEntry{}} }
	return Limiters{
		Requests:    memRequests{newStore()},
		Tokens:      memTokens{newStore()},
		Concurrency: &memConcurrency{leases: map[string]map[string]time.Time{}},
	}
}

// gcraEntry is the state for one key.
type gcraEntry struct {
	tat      int64 // theoretical arrival time, unix microseconds
	lastSeen int64 // latest effective now this key was evaluated at
}

type gcraStore struct {
	mu      sync.Mutex
	entries map[string]*gcraEntry
}

// reserve applies the GCRA decision for cost n as described in the package
// comment. The effective clock is max(now, lastSeen) so a backwards now is
// never more permissive than the latest one.
func (s *gcraStore) reserve(key string, lim RateLimit, n int64, now time.Time) Result {
	cost, limit, unlimited := gcraParams(lim, n)
	if unlimited || n <= 0 {
		return Result{Allowed: true}
	}
	if n > max(lim.Burst, 1) {
		return Result{}
	}
	nowUS := now.UnixMicro()

	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		e = &gcraEntry{tat: nowUS, lastSeen: nowUS}
	}
	eff := max(nowUS, e.lastSeen)
	tat0 := max(e.tat, eff)
	// newTAT - eff, computed as (tat0 - eff) + cost so it stays in range.
	span := satAdd(satSub(tat0, eff), cost)
	if span > limit {
		e.lastSeen = eff
		return Result{RetryAfter: microsToDuration(span - limit)}
	}
	e.tat = satAdd(tat0, cost)
	e.lastSeen = eff
	s.entries[key] = e
	return Result{Allowed: true}
}

// refund moves the key's TAT back by the cost of n, never to before the
// effective now.
func (s *gcraStore) refund(key string, lim RateLimit, n int64, now time.Time) {
	cost, _, unlimited := gcraParams(lim, n)
	if unlimited || n <= 0 {
		return
	}
	nowUS := now.UnixMicro()

	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		return
	}
	eff := max(nowUS, e.lastSeen)
	e.lastSeen = eff
	if e.tat <= eff || satSub(e.tat, eff) <= cost {
		e.tat = eff
		return
	}
	e.tat -= cost
}

type memRequests struct{ s *gcraStore }

func (m memRequests) Allow(_ context.Context, key string, lim RateLimit, now time.Time) (Result, error) {
	return m.s.reserve(key, lim, 1, now), nil
}

type memTokens struct{ s *gcraStore }

func (m memTokens) Reserve(_ context.Context, key string, lim RateLimit, n int64, now time.Time) (Result, error) {
	return m.s.reserve(key, lim, n, now), nil
}

func (m memTokens) Refund(_ context.Context, key string, lim RateLimit, n int64, now time.Time) error {
	m.s.refund(key, lim, n, now)
	return nil
}

type memConcurrency struct {
	mu     sync.Mutex
	leases map[string]map[string]time.Time // key -> lease -> expiry
}

func (m *memConcurrency) Acquire(_ context.Context, key, lease string, limit int, ttl time.Duration, now time.Time) (bool, error) {
	if limit <= 0 {
		return true, nil
	}
	if ttl <= 0 {
		ttl = time.Second
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	held := m.leases[key]
	for id, exp := range held {
		if !exp.After(now) {
			delete(held, id)
		}
	}
	if _, ok := held[lease]; ok {
		held[lease] = now.Add(ttl)
		return true, nil
	}
	if len(held) >= limit {
		return false, nil
	}
	if held == nil {
		held = map[string]time.Time{}
		m.leases[key] = held
	}
	held[lease] = now.Add(ttl)
	return true, nil
}

func (m *memConcurrency) Release(_ context.Context, key, lease string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	held := m.leases[key]
	delete(held, lease)
	if len(held) == 0 {
		delete(m.leases, key)
	}
	return nil
}
