// Package decisionlog writes the gateway's append-only JSONL log of routing decisions and of
// the feedback clients send about them. The log is the data off-policy evaluation reads
// (python/routepilot_train/logs.py): what the policy could choose, what it chose and with
// what probability, and what came of it.
//
// A record never holds prompt text. A prompt is represented by the first 16 hex digits of its
// SHA-256 and its size.
package decisionlog

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// queueSize is how many encoded records wait for the writer goroutine. A full queue drops
// the record that did not fit, so a slow disk never slows a request.
const queueSize = 4096

// rotatedTimeLayout is the UTC time in the name of a rotated file.
const rotatedTimeLayout = "20060102T150405Z"

// Record is one routing decision, written as one JSON line with "type": "decision". The
// primary decision of a request carries the outcome (Outcome); a shadow decision, which was
// never acted on, does not. Fields that do not apply to the policy that decided are null.
type Record struct {
	Type          string    `json:"type"`
	DecisionID    string    `json:"decision_id"`
	Time          time.Time `json:"time"` // UTC
	Tenant        string    `json:"tenant"`
	PolicyVersion string    `json:"policy_version"`
	// PolicyRole is stable, canary, shadow, static or pinned.
	PolicyRole string `json:"policy_role"`
	// PromptSHA16 is the first 16 hex digits of the SHA-256 of the message texts joined by
	// newlines: the only trace of the prompt.
	PromptSHA16 string `json:"prompt_sha256_16"`
	PromptBytes int    `json:"prompt_bytes"`
	// Required names the capabilities the request needs.
	Required   []string    `json:"required"`
	OODScore   *float64    `json:"ood_score"`
	OOD        *bool       `json:"ood"`
	Candidates []Candidate `json:"candidates"`
	// Chosen is the model the policy picked first; empty when it picked none. A failover can
	// serve the request from another model, which Attempts shows.
	Chosen     string   `json:"chosen"`
	Propensity *float64 `json:"propensity"`
	Epsilon    *float64 `json:"epsilon"`
	Explored   *bool    `json:"explored"`
	Relaxed    []string `json:"relaxed"`

	// Outcome is nil for a shadow decision, and its fields are then left out.
	*Outcome
}

// Candidate is one model the policy considered. P, L, CostExpPicoUSD and QMs are null for a
// static policy, and for a model that was ruled out before it was scored; CostMaxPicoUSD is
// null for a model ruled out before it was costed.
type Candidate struct {
	Model          string   `json:"model"`
	P              *float64 `json:"p"`
	L              *float64 `json:"l"`
	CostExpPicoUSD *int64   `json:"cost_exp_picousd"`
	CostMaxPicoUSD *int64   `json:"cost_max_picousd"`
	QMs            *float64 `json:"q_ms"`
	Feasible       bool     `json:"feasible"`
	Reasons        []string `json:"reasons"`
}

// Outcome is what came of a decision.
type Outcome struct {
	// Status is the HTTP status the client got.
	Status    int   `json:"status"`
	LatencyMs int64 `json:"latency_ms"`
	// TTFTMs is the time to the first chunk sent to the client, for a stream; null otherwise.
	TTFTMs           *int64 `json:"ttft_ms"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	// CostActualPicoUSD is what every attempt together cost.
	CostActualPicoUSD int64     `json:"cost_actual_picousd"`
	UsageEstimated    bool      `json:"usage_estimated"`
	Attempts          []Attempt `json:"attempts"`
}

// Attempt is one try at a deployment and how it ended ("ok", "http_503", "timeout", ...).
type Attempt struct {
	Deployment string `json:"deployment"`
	Outcome    string `json:"outcome"`
}

// Feedback is a client's verdict on a decision, written as one JSON line.
type Feedback struct {
	Type       string    `json:"type"`
	DecisionID string    `json:"decision_id"`
	Tenant     string    `json:"tenant"`
	Success    bool      `json:"success"`
	Time       time.Time `json:"time"`
}

// Writer appends records to a file. Write never blocks: records go through a bounded queue to
// one goroutine that owns the file, and a full queue drops the new record and counts it.
type Writer struct {
	path     string
	maxBytes int64
	now      func() time.Time
	// beforeWrite, when set, runs in the writer goroutine before each record is written. Tests
	// use it to hold the goroutine and fill the queue.
	beforeWrite func()

	// mu makes a send and Close mutually exclusive, so a Write never sends on a closed queue.
	mu     sync.RWMutex
	closed bool
	queue  chan []byte
	done   chan struct{}

	dropped atomic.Int64
	errs    atomic.Int64

	// The fields below belong to the writer goroutine (and to open, before it starts).
	f        *os.File
	bw       *bufio.Writer
	size     int64
	closeErr error
}

// Open opens the log at path for appending, creating it and its directory if needed, and
// starts the writer goroutine. maxBytes must be positive: once the file is larger than that
// it is renamed to <path without extension>.<UTC yyyymmddTHHMMSSZ>.jsonl and a new file is
// started. Records already in the file are kept; an unfinished last line is closed off with a
// newline so the next record starts on its own line.
func Open(path string, maxBytes int64) (*Writer, error) {
	return open(path, maxBytes, queueSize, time.Now, nil)
}

func open(path string, maxBytes int64, queue int, now func() time.Time, beforeWrite func()) (*Writer, error) {
	switch {
	case path == "":
		return nil, errors.New("decisionlog: path is empty")
	case maxBytes <= 0:
		return nil, fmt.Errorf("decisionlog: max bytes %d, want a positive number", maxBytes)
	}
	w := &Writer{
		path: path, maxBytes: maxBytes, now: now, beforeWrite: beforeWrite,
		queue: make(chan []byte, queue), done: make(chan struct{}),
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return nil, fmt.Errorf("decisionlog: create directory for %s: %w", path, err)
	}
	if err := w.openFile(); err != nil {
		return nil, err
	}
	if w.size > w.maxBytes {
		if err := w.rotate(); err != nil {
			_ = w.f.Close()
			return nil, err
		}
	}
	go w.run()
	return w, nil
}

// openFile opens the active file for appending and repairs a missing final newline.
func (w *Writer) openFile() error {
	f, err := os.OpenFile(w.path, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0o600)
	if err != nil {
		return fmt.Errorf("decisionlog: open %s: %w", w.path, err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("decisionlog: stat %s: %w", w.path, err)
	}
	if info.IsDir() {
		_ = f.Close()
		return fmt.Errorf("decisionlog: %s is a directory", w.path)
	}
	size := info.Size()
	if size > 0 {
		var last [1]byte
		if _, err := f.ReadAt(last[:], size-1); err == nil && last[0] != '\n' {
			if _, err := f.Write([]byte{'\n'}); err != nil {
				_ = f.Close()
				return fmt.Errorf("decisionlog: repair %s: %w", w.path, err)
			}
			size++
		}
	}
	w.f, w.size = f, size
	if w.bw == nil {
		w.bw = bufio.NewWriter(f)
	} else {
		w.bw.Reset(f)
	}
	return nil
}

// Write queues rec as one JSON line. It never blocks: a full queue, or a writer that has been
// closed, drops the record and counts it in Dropped, and a record that cannot be encoded to
// JSON is counted in Errors. rec is encoded before Write returns, so the caller may change or
// reuse it at once.
func (w *Writer) Write(rec any) {
	b, err := json.Marshal(rec)
	if err != nil {
		w.errs.Add(1)
		return
	}
	b = append(b, '\n')
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		w.dropped.Add(1)
		return
	}
	select {
	case w.queue <- b:
	default:
		w.dropped.Add(1)
	}
}

// Dropped is the number of records dropped because the queue was full or the writer closed.
func (w *Writer) Dropped() int64 { return w.dropped.Load() }

// Errors is the number of records lost to an encoding or file error.
func (w *Writer) Errors() int64 { return w.errs.Load() }

// Close stops accepting records, writes what is queued, flushes and closes the file. It is
// safe to call more than once.
func (w *Writer) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		<-w.done
		return nil
	}
	w.closed = true
	close(w.queue)
	w.mu.Unlock()
	<-w.done
	return w.closeErr
}

// run is the writer goroutine.
func (w *Writer) run() {
	defer close(w.done)
	for line := range w.queue {
		if w.beforeWrite != nil {
			w.beforeWrite()
		}
		w.write(line)
		if len(w.queue) == 0 {
			w.flush()
		}
	}
	w.flush()
	if w.f != nil {
		w.closeErr = w.f.Close()
		w.f = nil
	}
}

func (w *Writer) write(line []byte) {
	if w.f == nil && w.openFile() != nil {
		w.errs.Add(1)
		return
	}
	n, err := w.bw.Write(line)
	w.size += int64(n)
	if err != nil {
		w.errs.Add(1)
		w.bw.Reset(w.f) // drop the sticky error and what was buffered, and go on
		return
	}
	if w.size > w.maxBytes {
		if err := w.rotate(); err != nil {
			w.errs.Add(1)
		}
	}
}

func (w *Writer) flush() {
	if w.f == nil {
		return
	}
	if err := w.bw.Flush(); err != nil {
		w.errs.Add(1)
		w.bw.Reset(w.f)
	}
}

// rotate moves the active file aside and starts a new one. The name carries the UTC time of
// the rotation; if that name is taken (two rotations in one second) the time moves on a
// second until it is free, so no file is ever overwritten.
func (w *Writer) rotate() error {
	if err := w.bw.Flush(); err != nil {
		w.bw.Reset(w.f)
		w.errs.Add(1)
	}
	closeErr := w.f.Close()
	w.f = nil
	base := strings.TrimSuffix(w.path, filepath.Ext(w.path))
	at := w.now().UTC()
	var target string
	for i := 0; ; i++ {
		target = base + "." + at.Format(rotatedTimeLayout) + ".jsonl"
		if _, err := os.Lstat(target); os.IsNotExist(err) {
			break
		}
		if i >= 3600 {
			return fmt.Errorf("decisionlog: no free name to rotate %s to", w.path)
		}
		at = at.Add(time.Second)
	}
	renameErr := os.Rename(w.path, target)
	reopenErr := w.openFile() // a log with nowhere to go would lose every later record
	switch {
	case renameErr != nil:
		return fmt.Errorf("decisionlog: rotate %s: %w", w.path, renameErr)
	case reopenErr != nil:
		return reopenErr
	case closeErr != nil:
		return fmt.Errorf("decisionlog: close %s before rotating: %w", w.path, closeErr)
	}
	return nil
}
