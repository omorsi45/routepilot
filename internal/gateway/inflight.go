package gateway

import (
	"context"
	"fmt"
	"sync"
)

// inflight counts the requests the server is handling, so that a caller can wait for them to
// return. A sync.WaitGroup does not fit: a request may arrive while a waiter waits, and
// WaitGroup forbids an Add from zero concurrent with a Wait.
type inflight struct {
	mu sync.Mutex
	n  int
	// idle is closed when n drops to zero; it exists only while n > 0.
	idle chan struct{}
}

// enter counts one request in.
func (f *inflight) enter() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.n == 0 {
		f.idle = make(chan struct{})
	}
	f.n++
}

// leave counts one request out.
func (f *inflight) leave() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.n--
	if f.n == 0 {
		close(f.idle)
	}
}

// wait blocks until no request is being handled, or ctx ends. It reports how many were
// still running in that case.
func (f *inflight) wait(ctx context.Context) error {
	f.mu.Lock()
	n, idle := f.n, f.idle
	f.mu.Unlock()
	if n == 0 {
		return nil
	}
	select {
	case <-idle:
		return nil
	case <-ctx.Done():
		f.mu.Lock()
		left := f.n
		f.mu.Unlock()
		return fmt.Errorf("%d requests still running: %w", left, ctx.Err())
	}
}

// WaitIdle blocks until every request the server is handling has returned, which includes
// its budget commit and the queueing of its usage rows and decision record, or until ctx
// ends. A server that has stopped taking connections (http.Server.Close) is idle once the
// requests it cut have unwound. The caller runs it before closing the usage queue, the
// decision log and the quota backend, so that a request cut at shutdown still leaves its
// records.
func (s *Server) WaitIdle(ctx context.Context) error {
	return s.inflight.wait(ctx)
}
