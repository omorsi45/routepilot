package accounting_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
)

// usageTableBeforeThePeriodColumn is the usage table of a ledger written by a gateway that
// did not yet record the budget period of its rows.
const usageTableBeforeThePeriodColumn = `CREATE TABLE usage (
		request_id        TEXT    NOT NULL,
		attempt           INTEGER NOT NULL,
		tenant            TEXT    NOT NULL,
		model             TEXT    NOT NULL,
		deployment        TEXT    NOT NULL,
		prompt_tokens     INTEGER NOT NULL,
		completion_tokens INTEGER NOT NULL,
		usage_estimated   INTEGER NOT NULL,
		cost_estimate     INTEGER NOT NULL,
		cost_actual       INTEGER NOT NULL,
		policy_version    TEXT    NOT NULL,
		status            TEXT    NOT NULL,
		started_at        TEXT    NOT NULL,
		ended_at          TEXT    NOT NULL,
		PRIMARY KEY (request_id, attempt)
	)`

// TestSQLiteUpgradesALedgerWithoutThePeriodColumn: opening a database written before the
// budget_period column existed adds the column, keeps the old rows (with the empty period,
// as they were written for no known budget) and takes new rows with a period; opening it
// again is harmless.
func TestSQLiteUpgradesALedgerWithoutThePeriodColumn(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(usageTableBeforeThePeriodColumn); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`INSERT INTO usage VALUES ('old-req', 1, 'tenant-a', 'm', 'm-a', 5, 4, 0, 10, 45000000,
		'pinned', 'ok', '2026-10-15T12:00:00.000000000Z', '2026-10-15T12:00:01.000000000Z')`); err != nil {
		t.Fatal(err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	for pass := 1; pass <= 2; pass++ {
		s, err := accounting.OpenSQLite(path)
		if err != nil {
			t.Fatalf("OpenSQLite pass %d on a database without budget_period: %v", pass, err)
		}
		if pass == 1 {
			fresh := sampleRow("new-req", 1)
			fresh.BudgetPeriod = "total"
			fresh.CostActual = 7
			if err := s.RecordAttempt(ctx, fresh); err != nil {
				t.Fatalf("record a row with a period: %v", err)
			}
		}
		if got, err := s.TenantSpendInPeriod(ctx, "tenant-a", ""); err != nil || got != 45_000_000 {
			t.Errorf("pass %d: spend of the old row in the empty period = %d, %v, want 45000000", pass, got, err)
		}
		if got, err := s.TenantSpendInPeriod(ctx, "tenant-a", "total"); err != nil || got != 7 {
			t.Errorf("pass %d: spend of the new row in period total = %d, %v, want 7", pass, got, err)
		}
		if got, err := s.TenantSpend(ctx, "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC)); err != nil || got != 45_000_007 {
			t.Errorf("pass %d: spend over all time = %d, %v, want 45000007", pass, got, err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
