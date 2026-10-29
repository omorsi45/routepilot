package gateway

import (
	"database/sql"
	"net/http"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/omorsi45/routepilot/internal/accounting"
)

// usageRowCount counts the rows of the SQLite ledger at path through a connection of its own,
// the way `routepilot reconcile` reads it.
func usageRowCount(t *testing.T, path string) int {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM usage").Scan(&n); err != nil {
		t.Fatalf("count the ledger rows: %v", err)
	}
	return n
}

// TestGatewayKeepsEveryUsageRowAt64Clients (Ruling V1-R38): 64 clients send requests back to
// back against a SQLite ledger on disk, and once the gateway has closed its ledger writer every
// request has its row and none is counted lost. When each request wrote its own row
// synchronously, under a 250 ms bound, on SQLite's single connection with a sync per row, the
// writers queued past the bound at this concurrency and the ledger lost about a tenth of its
// rows (the quota experiment of batch E-5).
func TestGatewayKeepsEveryUsageRowAt64Clients(t *testing.T) {
	const clients, perClient = 64, 8
	e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "m", in: "1", out: "2", deps: []depSpec{{"m-a", "ua"}}}}})
	path := filepath.Join(t.TempDir(), "usage.db")
	store, err := accounting.OpenSQLite(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	useLedger(t, e.gw, store)

	var (
		wg     sync.WaitGroup
		failed atomic.Int64
	)
	for range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range perClient {
				if rec := e.post(chatBody("m", "hi")); rec.Code != http.StatusOK {
					failed.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	closeUsage(t, e.gw)

	if n := failed.Load(); n != 0 {
		t.Fatalf("%d requests failed, want none", n)
	}
	if got, want := usageRowCount(t, path), clients*perClient; got != want {
		t.Errorf("the ledger holds %d rows after %d requests, want %d: %d rows lost", got, want, want, want-got)
	}
	if n := e.gw.AccountingErrors(); n != 0 {
		t.Errorf("AccountingErrors = %d, want 0", n)
	}
}
