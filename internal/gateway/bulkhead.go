package gateway

import (
	"sync"
	"sync/atomic"

	"github.com/omorsi45/routepilot/internal/catalog"
)

// bulkhead caps the non-streaming attempts in flight on each deployment that sets
// max_inflight, so one slow deployment cannot absorb every goroutine of the server.
// A deployment without a limit has no counter and is never full. A nil *bulkhead, which is
// what a config without any limit gets, behaves as no limit anywhere.
//
// It lives in the gateway, not in reliability: it is about capacity, not about health, and
// it takes no part in the breaker's bookkeeping.
type bulkhead struct {
	slots map[string]*deploymentSlots
	// onReject is told about every attempt the bulkhead turned away, with the deployment.
	onReject func(deployment string)
}

// deploymentSlots is the counter of one limited deployment.
type deploymentSlots struct {
	limit    int64
	inflight atomic.Int64
}

// newBulkhead builds the bulkhead of the catalog's limited deployments, nil when none is
// limited. onReject may be nil.
func newBulkhead(cat *catalog.Catalog, onReject func(deployment string)) *bulkhead {
	slots := make(map[string]*deploymentSlots)
	for _, m := range cat.Models() {
		for _, d := range m.Deployments {
			if d.MaxInflight > 0 {
				slots[d.Name] = &deploymentSlots{limit: int64(d.MaxInflight)}
			}
		}
	}
	if len(slots) == 0 {
		return nil
	}
	return &bulkhead{slots: slots, onReject: onReject}
}

// slot is the counter of the deployment, nil for one without a limit.
func (b *bulkhead) slot(deployment string) *deploymentSlots {
	if b == nil {
		return nil
	}
	return b.slots[deployment]
}

// full reports whether the deployment is at its limit. It takes nothing.
func (b *bulkhead) full(deployment string) bool {
	s := b.slot(deployment)
	return s != nil && s.inflight.Load() >= s.limit
}

// inflight is the number of slots the deployment has handed out.
func (b *bulkhead) inflight(deployment string) int {
	s := b.slot(deployment)
	if s == nil {
		return 0
	}
	return int(s.inflight.Load())
}

// acquire takes a slot on the deployment. It returns the func that gives the slot back, which
// is safe to call more than once and returns the slot only the first time, and true. A
// deployment at its limit returns false. A deployment without a limit returns a nil func and
// true: there is nothing to give back.
func (b *bulkhead) acquire(deployment string) (release func(), ok bool) {
	s := b.slot(deployment)
	if s == nil {
		return nil, true
	}
	for {
		n := s.inflight.Load()
		if n >= s.limit {
			return nil, false
		}
		if s.inflight.CompareAndSwap(n, n+1) {
			var once sync.Once
			return func() { once.Do(func() { s.inflight.Add(-1) }) }, true
		}
	}
}

// reject counts an attempt turned away because the deployment was full.
func (b *bulkhead) reject(deployment string) {
	if b != nil && b.onReject != nil {
		b.onReject(deployment)
	}
}
