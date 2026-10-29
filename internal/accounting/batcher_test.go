package accounting_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
)

// fakeStore is the store behind a batcher under test. It embeds a nil Store because the
// batcher calls only RecordAttempts; any other call would panic, which is the point.
type fakeStore struct {
	accounting.Store

	mu        sync.Mutex
	batches   [][]accounting.UsageRow
	arrived   []time.Time // when each batch was written
	calls     int
	deadlines []time.Duration // time left to the deadline of each call, 0 when it had none
	fail      error
	// hold, when set, makes RecordAttempts wait until release is called or the context ends.
	hold    chan struct{}
	relOnce sync.Once
}

// release lets the writes held by hold proceed. It is safe to call more than once.
func (f *fakeStore) release() { f.relOnce.Do(func() { close(f.hold) }) }

func (f *fakeStore) RecordAttempts(ctx context.Context, rows []accounting.UsageRow) error {
	f.mu.Lock()
	f.calls++
	var left time.Duration
	if d, ok := ctx.Deadline(); ok {
		left = time.Until(d)
	}
	f.deadlines = append(f.deadlines, left)
	hold, fail := f.hold, f.fail
	f.mu.Unlock()

	if hold != nil {
		select {
		case <-hold:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if fail != nil {
		return fail
	}
	f.mu.Lock()
	f.batches = append(f.batches, slices.Clone(rows))
	f.arrived = append(f.arrived, time.Now())
	f.mu.Unlock()
	return nil
}

func (f *fakeStore) setFail(err error) {
	f.mu.Lock()
	f.fail = err
	f.mu.Unlock()
}

func (f *fakeStore) sizes() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := []int{}
	for _, b := range f.batches {
		out = append(out, len(b))
	}
	return out
}

func (f *fakeStore) ids() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := []string{}
	for _, b := range f.batches {
		for _, r := range b {
			out = append(out, r.RequestID)
		}
	}
	return out
}

func (f *fakeStore) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *fakeStore) deadlineLeft(i int) time.Duration {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i >= len(f.deadlines) {
		return -1
	}
	return f.deadlines[i]
}

// hungStore is a store whose writes wait until the test ends (or their context does).
func hungStore(t *testing.T) *fakeStore {
	t.Helper()
	f := &fakeStore{hold: make(chan struct{})}
	t.Cleanup(f.release)
	return f
}

// usageRows makes n rows with request ids "<prefix>0" .. "<prefix>n-1".
func usageRows(prefix string, n int) []accounting.UsageRow {
	rows := make([]accounting.UsageRow, n)
	for i := range rows {
		rows[i] = accounting.UsageRow{
			RequestID: fmt.Sprintf("%s%d", prefix, i), Attempt: 1, Tenant: "tenant-a",
			Model: "m", Deployment: "m-a", Status: "ok", StartedAt: t0, EndedAt: t0,
		}
	}
	return rows
}

func wantIDs(prefix string, from, to int) []string {
	var ids []string
	for i := from; i < to; i++ {
		ids = append(ids, fmt.Sprintf("%s%d", prefix, i))
	}
	return ids
}

func waitUntil(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %v waiting for: %s", d, what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

func closeBatcher(t *testing.T, b *accounting.Batcher) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := b.Close(ctx); err != nil {
		t.Errorf("Close error = %v", err)
	}
}

type lostCounter struct{ n atomic.Int64 }

func (l *lostCounter) onLost(n int) { l.n.Add(int64(n)) }
func (l *lostCounter) get() int64   { return l.n.Load() }

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// TestBatcherWritesWhenMaxBatchRowsArePending: the time bound is an hour, so only the size
// can make the batcher write.
func TestBatcherWritesWhenMaxBatchRowsArePending(t *testing.T) {
	store := &fakeStore{}
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 3, FlushEvery: time.Hour})
	t.Cleanup(func() { closeBatcher(t, b) })

	if !b.Enqueue(usageRows("r", 2)) {
		t.Fatal("Enqueue of 2 rows = false, want true")
	}
	time.Sleep(100 * time.Millisecond)
	if n := store.callCount(); n != 0 {
		t.Fatalf("the store was called %d times with 2 of 3 rows pending and an hour to flush", n)
	}
	if got := b.Pending(); got != 2 {
		t.Errorf("Pending = %d, want 2", got)
	}

	if !b.Enqueue(usageRows("s", 1)) {
		t.Fatal("Enqueue of the third row = false, want true")
	}
	waitUntil(t, 5*time.Second, "the batch of 3 to be written", func() bool { return len(store.ids()) == 3 })
	if got, want := store.ids(), []string{"r0", "r1", "s0"}; !slices.Equal(got, want) {
		t.Errorf("rows written = %v, want %v in enqueue order", got, want)
	}
	if got := store.sizes(); !slices.Equal(got, []int{3}) {
		t.Errorf("batch sizes = %v, want one batch of 3", got)
	}
}

// TestBatcherCutsABacklogIntoBatchesOfAtMostMaxBatch: 10 rows at once with MaxBatch 4 are
// written as 4, 4 and the remaining 2, in order.
func TestBatcherCutsABacklogIntoBatchesOfAtMostMaxBatch(t *testing.T) {
	store := &fakeStore{}
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 4, FlushEvery: 30 * time.Millisecond})
	t.Cleanup(func() { closeBatcher(t, b) })

	if !b.Enqueue(usageRows("r", 10)) {
		t.Fatal("Enqueue = false, want true")
	}
	waitUntil(t, 5*time.Second, "all 10 rows to be written", func() bool { return len(store.ids()) == 10 })
	if got, want := store.sizes(), []int{4, 4, 2}; !slices.Equal(got, want) {
		t.Errorf("batch sizes = %v, want %v", got, want)
	}
	if got, want := store.ids(), wantIDs("r", 0, 10); !slices.Equal(got, want) {
		t.Errorf("rows written = %v, want %v", got, want)
	}
}

// TestBatcherWritesAfterFlushEvery: a single row, far below MaxBatch, is written once
// FlushEvery has passed and not before.
func TestBatcherWritesAfterFlushEvery(t *testing.T) {
	const every = 200 * time.Millisecond
	store := &fakeStore{}
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 100, FlushEvery: every})
	t.Cleanup(func() { closeBatcher(t, b) })

	start := time.Now()
	b.Enqueue(usageRows("r", 1))
	waitUntil(t, 5*time.Second, "the row to be written", func() bool { return len(store.ids()) == 1 })

	store.mu.Lock()
	elapsed := store.arrived[0].Sub(start)
	store.mu.Unlock()
	if elapsed < every*3/4 {
		t.Errorf("the row was written after %v, want at least about FlushEvery (%v)", elapsed, every)
	}
}

// TestBatcherFlushIsMeasuredFromTheOldestRow: rows that keep arriving more often than
// FlushEvery must not postpone the write. A timer restarted by each row would never fire.
func TestBatcherFlushIsMeasuredFromTheOldestRow(t *testing.T) {
	store := &fakeStore{}
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 100000, FlushEvery: 200 * time.Millisecond})
	t.Cleanup(func() { closeBatcher(t, b) })

	start := time.Now()
	for i := 0; len(store.sizes()) == 0; i++ {
		if time.Since(start) > 3*time.Second {
			t.Fatal("no batch was written in 3 s of steady arrivals, want one after about FlushEvery")
		}
		b.Enqueue(usageRows(fmt.Sprintf("r%d-", i), 1))
		time.Sleep(20 * time.Millisecond)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("the first batch came after %v of steady arrivals, want about 200ms", elapsed)
	}
	if got := store.sizes()[0]; got < 2 {
		t.Errorf("first batch holds %d rows, want the rows that arrived meanwhile too", got)
	}
}

// TestBatcherEnqueueNeverBlocksWhenTheStoreHangs: with the store stuck, a thousand Enqueue
// calls all return at once; what does not fit is refused and counted, and nothing is lost
// from what was accepted once the store recovers.
func TestBatcherEnqueueNeverBlocksWhenTheStoreHangs(t *testing.T) {
	store := hungStore(t)
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{
		QueueRows: 100, MaxBatch: 10, FlushEvery: 5 * time.Millisecond, WriteTimeout: time.Minute, OnLost: lost.onLost,
	})

	const calls = 1000
	var accepted atomic.Int64
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range calls {
			if b.Enqueue(usageRows(fmt.Sprintf("r%d-", i), 1)) {
				accepted.Add(1)
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Enqueue blocked while the store was hung")
	}

	got := accepted.Load()
	if got < 100 || got > 110 {
		t.Errorf("accepted %d rows, want 100 queued plus the one batch (at most 10) in the hung write", got)
	}
	if lost.get() != calls-got {
		t.Errorf("OnLost counted %d rows, want %d: every refused row", lost.get(), calls-got)
	}

	store.release()
	closeBatcher(t, b)
	if n := int64(len(store.ids())); n != got {
		t.Errorf("the store received %d rows after it recovered, want the %d accepted", n, got)
	}
}

// TestBatcherQueueMustHoldTheWholeSet: Enqueue takes all the rows of a call or none.
func TestBatcherQueueMustHoldTheWholeSet(t *testing.T) {
	store := &fakeStore{}
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{
		QueueRows: 5, MaxBatch: 1000, FlushEvery: time.Hour, OnLost: lost.onLost,
	})

	if !b.Enqueue(usageRows("a", 3)) {
		t.Fatal("Enqueue of 3 rows into an empty queue of 5 = false")
	}
	if b.Enqueue(usageRows("b", 3)) {
		t.Error("Enqueue of 3 rows with room for 2 = true, want false")
	}
	if lost.get() != 3 || b.Pending() != 3 {
		t.Errorf("after the refusal: lost %d, pending %d, want 3 and 3: none of the refused set is kept", lost.get(), b.Pending())
	}
	if !b.Enqueue(usageRows("c", 2)) {
		t.Error("Enqueue of 2 rows with room for 2 = false, want true")
	}
	if b.Enqueue(usageRows("d", 1)) {
		t.Error("Enqueue into a full queue = true, want false")
	}
	if lost.get() != 4 || b.Pending() != 5 {
		t.Errorf("after the second refusal: lost %d, pending %d, want 4 and 5", lost.get(), b.Pending())
	}

	closeBatcher(t, b)
	if got, want := store.ids(), []string{"a0", "a1", "a2", "c0", "c1"}; !slices.Equal(got, want) {
		t.Errorf("rows written = %v, want %v", got, want)
	}
}

// TestBatcherCloseDrainsAndStopsIntake: Close writes everything pending however far off
// FlushEvery is, and after it Enqueue refuses (and counts) rows.
func TestBatcherCloseDrainsAndStopsIntake(t *testing.T) {
	store := &fakeStore{}
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 100, FlushEvery: time.Hour, OnLost: lost.onLost})

	b.Enqueue(usageRows("r", 7))
	closeBatcher(t, b)

	if got, want := store.ids(), wantIDs("r", 0, 7); !slices.Equal(got, want) {
		t.Errorf("rows written by Close = %v, want %v", got, want)
	}
	if b.Pending() != 0 {
		t.Errorf("Pending after Close = %d, want 0", b.Pending())
	}
	if b.Enqueue(usageRows("late", 1)) {
		t.Error("Enqueue after Close = true, want false")
	}
	if lost.get() != 1 {
		t.Errorf("lost = %d, want 1: a row refused after Close is a lost row", lost.get())
	}
	closeBatcher(t, b) // closing twice is not an error
}

// TestBatcherWithoutOnLost: the callback is optional.
func TestBatcherWithoutOnLost(t *testing.T) {
	b := accounting.NewBatcher(&fakeStore{}, accounting.BatcherOptions{QueueRows: 1, FlushEvery: time.Hour})
	b.Enqueue(usageRows("a", 1))
	if b.Enqueue(usageRows("b", 1)) {
		t.Error("Enqueue into a full queue = true, want false")
	}
	closeBatcher(t, b)
	if b.Enqueue(usageRows("c", 1)) {
		t.Error("Enqueue after Close = true, want false")
	}
}

// TestBatcherCloseGivesUpWhenTheContextEnds: a store that hangs cannot hold shutdown past
// the caller's deadline, and the rows still queued then are counted as lost.
func TestBatcherCloseGivesUpWhenTheContextEnds(t *testing.T) {
	store := hungStore(t)
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{
		MaxBatch: 2, FlushEvery: time.Hour, WriteTimeout: time.Minute, OnLost: lost.onLost,
	})
	b.Enqueue(usageRows("a", 2)) // taken at once, stuck in the store
	waitUntil(t, 5*time.Second, "the first write to start", func() bool { return store.callCount() == 1 })
	b.Enqueue(usageRows("b", 1)) // stays queued

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := b.Close(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Close error = %v, want one wrapping context.DeadlineExceeded", err)
	}
	if took := time.Since(start); took > 3*time.Second {
		t.Errorf("Close took %v, want it to return soon after its context ended", took)
	}
	if lost.get() != 1 {
		t.Errorf("lost = %d, want 1: the queued row that was never written", lost.get())
	}

	store.release() // the write in flight finishes normally
	waitUntil(t, 5*time.Second, "the batcher to settle", func() bool { return b.Pending() == 0 })
	if lost.get() != 1 {
		t.Errorf("lost = %d after the hung write recovered, want still 1", lost.get())
	}
}

// TestBatcherCountsAFailedWriteAsLost: every row of a failed transaction is lost, the cause
// is logged once for the batch, and the batcher goes on to the next batch.
func TestBatcherCountsAFailedWriteAsLost(t *testing.T) {
	store := &fakeStore{fail: errors.New("disk is full")}
	var (
		lost lostCounter
		logs syncBuffer
	)
	b := accounting.NewBatcher(store, accounting.BatcherOptions{
		MaxBatch: 3, FlushEvery: time.Hour, OnLost: lost.onLost, Logger: slog.New(slog.NewTextHandler(&logs, nil)),
	})

	b.Enqueue(usageRows("r", 3))
	waitUntil(t, 5*time.Second, "the failed batch to settle", func() bool { return store.callCount() == 1 && b.Pending() == 0 })
	if lost.get() != 3 {
		t.Errorf("lost = %d, want 3: every row of the failed batch", lost.get())
	}
	out := logs.String()
	if strings.Count(out, "usage ledger write failed") != 1 || !strings.Contains(out, "disk is full") ||
		!strings.Contains(out, "rows_lost=3") || !strings.Contains(out, "level=WARN") {
		t.Errorf("log should hold one warning naming the rows and the cause:\n%s", out)
	}

	store.setFail(nil)
	b.Enqueue(usageRows("s", 3))
	waitUntil(t, 5*time.Second, "the next batch to be written", func() bool { return len(store.ids()) == 3 })
	closeBatcher(t, b)
	if lost.get() != 3 {
		t.Errorf("lost = %d after a good batch, want still 3", lost.get())
	}
}

// TestBatcherWriteIsBoundedByWriteTimeout: a store that never answers is given up on after
// WriteTimeout, and its rows are lost.
func TestBatcherWriteIsBoundedByWriteTimeout(t *testing.T) {
	store := hungStore(t)
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{
		MaxBatch: 2, FlushEvery: time.Hour, WriteTimeout: 100 * time.Millisecond, OnLost: lost.onLost,
	})
	t.Cleanup(func() { closeBatcher(t, b) })

	b.Enqueue(usageRows("r", 2))
	waitUntil(t, 5*time.Second, "the write to time out", func() bool { return lost.get() == 2 })
	if left := store.deadlineLeft(0); left <= 0 || left > 100*time.Millisecond {
		t.Errorf("deadline left to the write = %v, want a deadline of at most 100ms", left)
	}
}

// TestBatcherPendingCountsRowsBeingWritten: a row is pending until its write has ended, so
// a caller waiting for Pending() == 0 knows the store has everything.
func TestBatcherPendingCountsRowsBeingWritten(t *testing.T) {
	store := hungStore(t)
	b := accounting.NewBatcher(store, accounting.BatcherOptions{MaxBatch: 2, FlushEvery: 30 * time.Millisecond, WriteTimeout: time.Minute})
	t.Cleanup(func() { closeBatcher(t, b) })

	b.Enqueue(usageRows("a", 2))
	waitUntil(t, 5*time.Second, "the write to start", func() bool { return store.callCount() == 1 })
	b.Enqueue(usageRows("b", 1))
	if got := b.Pending(); got != 3 {
		t.Errorf("Pending = %d, want 3: two being written and one queued", got)
	}

	store.release()
	waitUntil(t, 5*time.Second, "everything to be written", func() bool { return b.Pending() == 0 })
	if got, want := store.ids(), []string{"a0", "a1", "b0"}; !slices.Equal(got, want) {
		t.Errorf("rows written = %v, want %v", got, want)
	}
}

// TestBatcherDefaults: the zero options hold 65,536 rows, write 512 at a time, flush every
// 50 ms and bound a write to 5 s.
func TestBatcherDefaults(t *testing.T) {
	t.Run("queue holds 65536 rows", func(t *testing.T) {
		b := accounting.NewBatcher(hungStore(t), accounting.BatcherOptions{})
		if b.Enqueue(usageRows("r", 65537)) {
			t.Error("Enqueue of 65537 rows = true, want false: over the default queue")
		}
		closeBatcher(t, b)
		b2 := accounting.NewBatcher(&fakeStore{}, accounting.BatcherOptions{})
		if !b2.Enqueue(usageRows("r", 65536)) {
			t.Error("Enqueue of 65536 rows = false, want true")
		}
		closeBatcher(t, b2)
	})

	t.Run("batches of 512 and a flush within 50ms", func(t *testing.T) {
		store := &fakeStore{}
		b := accounting.NewBatcher(store, accounting.BatcherOptions{})
		t.Cleanup(func() { closeBatcher(t, b) })
		start := time.Now()
		b.Enqueue(usageRows("r", 1100))
		waitUntil(t, 5*time.Second, "all rows to be written", func() bool { return len(store.ids()) == 1100 })
		if got, want := store.sizes(), []int{512, 512, 76}; !slices.Equal(got, want) {
			t.Errorf("batch sizes = %v, want %v", got, want)
		}
		store.mu.Lock()
		tail := store.arrived[2].Sub(start)
		store.mu.Unlock()
		if tail < 35*time.Millisecond || tail > 3*time.Second {
			t.Errorf("the short tail batch was written after %v, want about the default 50ms", tail)
		}
	})

	t.Run("write is bounded to 5s", func(t *testing.T) {
		store := &fakeStore{}
		b := accounting.NewBatcher(store, accounting.BatcherOptions{})
		t.Cleanup(func() { closeBatcher(t, b) })
		b.Enqueue(usageRows("r", 1))
		waitUntil(t, 5*time.Second, "the write", func() bool { return store.callCount() == 1 })
		if left := store.deadlineLeft(0); left < 4*time.Second || left > 5*time.Second {
			t.Errorf("deadline left to the write = %v, want about 5s", left)
		}
	})
}

// TestBatcherConcurrentEnqueue: many goroutines enqueue while the writer runs (this is the
// test the race detector watches); every row is written exactly once.
func TestBatcherConcurrentEnqueue(t *testing.T) {
	store := &fakeStore{}
	var lost lostCounter
	b := accounting.NewBatcher(store, accounting.BatcherOptions{FlushEvery: 5 * time.Millisecond, OnLost: lost.onLost})

	const writers, perWriter = 32, 100
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range perWriter {
				if !b.Enqueue(usageRows(fmt.Sprintf("w%d-r%d-", w, i), 2)) {
					t.Errorf("writer %d: Enqueue refused", w)
				}
			}
		}()
	}
	wg.Wait()
	closeBatcher(t, b)

	seen := map[string]int{}
	for _, id := range store.ids() {
		seen[id]++
	}
	if len(seen) != writers*perWriter*2 {
		t.Errorf("%d distinct rows written, want %d", len(seen), writers*perWriter*2)
	}
	for id, n := range seen {
		if n != 1 {
			t.Errorf("row %s written %d times", id, n)
		}
	}
	if lost.get() != 0 {
		t.Errorf("lost = %d, want 0", lost.get())
	}
}
