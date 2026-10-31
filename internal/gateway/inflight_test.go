package gateway

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestInflightWaitReturnsAtOnceWhenIdle(t *testing.T) {
	var f inflight
	if err := f.wait(context.Background()); err != nil {
		t.Fatalf("wait on an idle tracker = %v, want nil", err)
	}
	f.enter()
	f.leave()
	if err := f.wait(context.Background()); err != nil {
		t.Fatalf("wait after enter and leave = %v, want nil", err)
	}
}

func TestInflightWaitBlocksUntilTheLastRequestLeaves(t *testing.T) {
	var f inflight
	f.enter()
	f.enter()
	done := make(chan error, 1)
	go func() { done <- f.wait(context.Background()) }()

	f.leave()
	select {
	case err := <-done:
		t.Fatalf("wait returned %v with one request still running", err)
	case <-time.After(50 * time.Millisecond):
	}
	f.leave()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("wait = %v, want nil once the last request left", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("wait did not return after the last request left")
	}
}

func TestInflightWaitGivesUpWithTheContext(t *testing.T) {
	var f inflight
	f.enter()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err := f.wait(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("wait = %v, want the context error", err)
	}
	f.leave()
}

// TestInflightEnterDuringWait: a request that arrives while a waiter waits is counted, and
// the waiter returns only when it has left too.
func TestInflightEnterDuringWait(t *testing.T) {
	var f inflight
	f.enter()
	done := make(chan error, 1)
	go func() { done <- f.wait(context.Background()) }()
	time.Sleep(10 * time.Millisecond)
	f.enter()
	f.leave()
	select {
	case err := <-done:
		t.Fatalf("wait returned %v with a request still running", err)
	case <-time.After(30 * time.Millisecond):
	}
	f.leave()
	if err := <-done; err != nil {
		t.Fatalf("wait = %v, want nil", err)
	}
}
