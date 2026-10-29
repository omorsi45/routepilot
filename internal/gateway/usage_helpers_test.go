package gateway

import (
	"context"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
)

// useLedger installs store as the gateway's usage ledger, replacing the one the gateway was
// built with. Its rows are written by the gateway's batcher, which is closed when the test
// ends; a test that reads the store right after a response calls flushUsage first.
func useLedger(t *testing.T, gw *Server, store accounting.Store) {
	t.Helper()
	useLedgerWith(t, gw, store, accounting.BatcherOptions{})
}

// useLedgerWith is useLedger with the batcher's sizes and intervals set by tune.
func useLedgerWith(t *testing.T, gw *Server, store accounting.Store, tune accounting.BatcherOptions) {
	t.Helper()
	closeUsage(t, gw)
	gw.attachLedger(store, tune)
	t.Cleanup(func() { closeUsage(t, gw) })
}

// flushUsage waits until every row the gateway has queued so far is written to its store, or
// counted lost. It must run after the responses it waits for have been returned in full.
func flushUsage(t *testing.T, gw *Server) {
	t.Helper()
	if gw.usage == nil {
		return
	}
	waitFor(t, 10*time.Second, "the usage rows to be written", func() bool { return gw.usage.Pending() == 0 })
}

// closeUsage ends the gateway's writes to the usage ledger, after writing what is queued.
func closeUsage(t *testing.T, gw *Server) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := gw.CloseUsage(ctx); err != nil {
		t.Errorf("CloseUsage error = %v", err)
	}
}
