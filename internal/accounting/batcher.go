package accounting

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Defaults of BatcherOptions.
const (
	defaultBatcherQueueRows    = 65536
	defaultBatcherMaxBatch     = 512
	defaultBatcherFlushEvery   = 50 * time.Millisecond
	defaultBatcherWriteTimeout = 5 * time.Second
	// queueFullWarnEvery spaces the warnings about a full queue, which a stuck store would
	// otherwise repeat for every request.
	queueFullWarnEvery = time.Second
)

// BatcherOptions configure a Batcher. A zero field takes its default.
type BatcherOptions struct {
	// QueueRows is how many rows may wait to be written (default 65536). Rows being written
	// are not counted, so up to MaxBatch more are in memory while a write is under way.
	QueueRows int
	// MaxBatch is the most rows one write carries (default 512).
	MaxBatch int
	// FlushEvery is the longest a row waits before its batch is written even when the batch
	// is not full, counted from the oldest waiting row (default 50 ms).
	FlushEvery time.Duration
	// WriteTimeout bounds one RecordAttempts call (default 5 s). A write that exceeds it
	// fails, and its rows are lost.
	WriteTimeout time.Duration
	// OnLost, when set, is called with the number of rows that were not written: refused by
	// Enqueue, or in a batch whose write failed or timed out, or still queued when Close gave
	// up. It must not block and must not call back into the Batcher.
	OnLost func(n int)
	// Logger, when set, receives a warning for each failed batch write, and for a full queue
	// at most once a second. Rows are never logged.
	Logger *slog.Logger
}

// Batcher moves usage rows to a Store off the caller's path: Enqueue only appends to a
// bounded in-memory queue, and one goroutine writes the queue to the Store in transactions of
// up to MaxBatch rows. A request that records its attempts therefore never waits for the
// database, whatever its speed; the cost is that rows reach the Store up to FlushEvery late
// and that a crash of the process loses the rows still queued. Rows that cannot be written
// are counted through OnLost and never retried.
type Batcher struct {
	store        Store
	queueRows    int
	maxBatch     int
	flushEvery   time.Duration
	writeTimeout time.Duration
	onLost       func(n int)
	logger       *slog.Logger

	mu     sync.Mutex
	queue  []UsageRow
	oldest time.Time // when the oldest row now queued was queued; meaningful while queue is not empty
	// pending is the rows queued plus the rows in the write under way.
	pending int
	closed  bool
	// lastWarn and droppedSince rate-limit the full-queue warning.
	lastWarn     time.Time
	droppedSince int

	wake chan struct{} // capacity 1: tells the writer that the queue changed
	done chan struct{} // closed when the writer has returned
}

// NewBatcher starts a Batcher that writes to store. Close stops it.
func NewBatcher(store Store, opts BatcherOptions) *Batcher {
	b := &Batcher{
		store:        store,
		queueRows:    opts.QueueRows,
		maxBatch:     opts.MaxBatch,
		flushEvery:   opts.FlushEvery,
		writeTimeout: opts.WriteTimeout,
		onLost:       opts.OnLost,
		logger:       opts.Logger,
		wake:         make(chan struct{}, 1),
		done:         make(chan struct{}),
	}
	if b.queueRows <= 0 {
		b.queueRows = defaultBatcherQueueRows
	}
	if b.maxBatch <= 0 {
		b.maxBatch = defaultBatcherMaxBatch
	}
	if b.flushEvery <= 0 {
		b.flushEvery = defaultBatcherFlushEvery
	}
	if b.writeTimeout <= 0 {
		b.writeTimeout = defaultBatcherWriteTimeout
	}
	go b.run()
	return b
}

// Enqueue queues rows to be written and reports whether it did. It never blocks. It takes
// all the rows or none: when the queue lacks room for all of them, or the Batcher is closed,
// it returns false and reports len(rows) rows through OnLost.
func (b *Batcher) Enqueue(rows []UsageRow) bool {
	if len(rows) == 0 {
		return true
	}
	b.mu.Lock()
	if b.closed || len(b.queue)+len(rows) > b.queueRows {
		warn, dropped := b.noteDropLocked(len(rows))
		b.mu.Unlock()
		b.lose(len(rows))
		if warn {
			b.logger.LogAttrs(context.Background(), slog.LevelWarn, "usage ledger queue full, rows dropped",
				slog.Int("rows_dropped", dropped), slog.Int("queue_rows", b.queueRows))
		}
		return false
	}
	wasEmpty := len(b.queue) == 0
	if wasEmpty {
		b.oldest = time.Now()
	}
	b.queue = append(b.queue, rows...)
	b.pending += len(rows)
	notify := wasEmpty || len(b.queue) >= b.maxBatch
	b.mu.Unlock()
	if notify {
		b.signal()
	}
	return true
}

// noteDropLocked counts n refused rows toward the next full-queue warning and reports
// whether to log one now, with the rows dropped since the last. b.mu must be held.
func (b *Batcher) noteDropLocked(n int) (warn bool, dropped int) {
	b.droppedSince += n
	if b.logger == nil || b.closed || time.Since(b.lastWarn) < queueFullWarnEvery {
		return false, 0
	}
	b.lastWarn = time.Now()
	dropped, b.droppedSince = b.droppedSince, 0
	return true, dropped
}

// Pending is the number of rows accepted by Enqueue whose write has not ended: queued, or in
// the write under way. It reaches zero once every accepted row has been written or counted
// lost.
func (b *Batcher) Pending() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.pending
}

// Close stops intake and writes every row still queued, then returns. It gives up when ctx
// ends: the rows not yet written are counted lost and ctx's error is returned. It does not
// close the Store. Closing twice is not an error.
func (b *Batcher) Close(ctx context.Context) error {
	b.mu.Lock()
	b.closed = true
	b.mu.Unlock()
	b.signal()

	select {
	case <-b.done:
		return nil
	case <-ctx.Done():
	}
	b.mu.Lock()
	n := len(b.queue)
	b.queue = nil
	b.pending -= n
	b.mu.Unlock()
	b.lose(n)
	return fmt.Errorf("accounting: close usage batcher: %d queued rows not written: %w", n, ctx.Err())
}

func (b *Batcher) signal() {
	select {
	case b.wake <- struct{}{}:
	default: // a wake-up is already waiting
	}
}

func (b *Batcher) lose(n int) {
	if n > 0 && b.onLost != nil {
		b.onLost(n)
	}
}

// run is the writer goroutine.
func (b *Batcher) run() {
	defer close(b.done)
	for {
		batch, wait, stop := b.next()
		switch {
		case batch != nil:
			b.write(batch)
		case stop:
			return
		case wait < 0:
			<-b.wake
		default:
			timer := time.NewTimer(wait)
			select {
			case <-b.wake:
			case <-timer.C:
			}
			timer.Stop()
		}
	}
}

// next takes the rows to write now, if any. Otherwise it says how long to wait for the oldest
// queued row to be FlushEvery old (a negative wait means until the queue changes), or that
// the Batcher is closed and empty. A full batch, and everything once closed, is due at once.
func (b *Batcher) next() (batch []UsageRow, wait time.Duration, stop bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(b.queue)
	switch {
	case n == 0 && b.closed:
		return nil, 0, true
	case n == 0:
		return nil, -1, false
	case n < b.maxBatch && !b.closed:
		if wait = b.flushEvery - time.Since(b.oldest); wait > 0 {
			return nil, wait, false
		}
	}
	take := min(n, b.maxBatch)
	// The capacity is capped so that appending to the rest of the queue cannot overwrite the
	// batch while it is written.
	batch = b.queue[:take:take]
	if b.queue = b.queue[take:]; len(b.queue) == 0 {
		b.queue = nil
	}
	return batch, 0, false
}

// write sends one batch to the store. A failure loses every row of the batch.
func (b *Batcher) write(batch []UsageRow) {
	ctx, cancel := context.WithTimeout(context.Background(), b.writeTimeout)
	err := b.store.RecordAttempts(ctx, batch)
	cancel()
	if err != nil {
		b.lose(len(batch))
		if b.logger != nil {
			b.logger.LogAttrs(context.Background(), slog.LevelWarn, "usage ledger write failed",
				slog.Int("rows_lost", len(batch)), slog.String("error", err.Error()))
		}
	}
	b.mu.Lock()
	b.pending -= len(batch)
	b.mu.Unlock()
}
