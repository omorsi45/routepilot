package accounting_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "modernc.org/sqlite"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/accounting/accountingtest"
	"github.com/omorsi45/routepilot/internal/money"
)

var t0 = time.Date(2026, 10, 15, 12, 0, 0, 123456789, time.UTC)

func openTemp(t *testing.T) (accounting.Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "usage.db")
	s, err := accounting.OpenSQLite(path)
	if err != nil {
		t.Fatalf("OpenSQLite(%q) error = %v", path, err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, path
}

func TestSQLiteConformance(t *testing.T) {
	accountingtest.Run(t, func(t *testing.T) accounting.Store {
		s, _ := openTemp(t)
		return s
	})
}

func sampleRow(request string, attempt int) accounting.UsageRow {
	return accounting.UsageRow{
		RequestID: request, Attempt: attempt, Tenant: "tenant-a", Model: "m", Deployment: "m-a",
		PromptTokens: 5, CompletionTokens: 4, UsageEstimated: true,
		CostEstimate: 1_234_567_890_123, CostActual: 45_000_000,
		PolicyVersion: "static-cheapest", Status: "http_503",
		StartedAt: t0, EndedAt: t0.Add(1500 * time.Millisecond),
	}
}

// TestSQLiteStoresTypedColumns pins the on-disk shape: money as INTEGER picodollars, the
// estimated flag as 0/1, and timestamps as fixed-width UTC RFC 3339 text with nanoseconds,
// which sorts the way time does.
func TestSQLiteStoresTypedColumns(t *testing.T) {
	s, path := openTemp(t)
	row := sampleRow("req-1", 2)
	if err := s.RecordAttempt(context.Background(), row); err != nil {
		t.Fatal(err)
	}

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	var (
		request, tenant, model, dep, policy, status, started, ended string
		attempt, prompt, completion, estimated, estCost, actual     int64
		typCost, typEst                                             string
	)
	err = raw.QueryRow(`SELECT request_id, attempt, tenant, model, deployment, prompt_tokens, completion_tokens,
		usage_estimated, cost_estimate, cost_actual, policy_version, status, started_at, ended_at,
		typeof(cost_actual), typeof(cost_estimate) FROM usage`).Scan(
		&request, &attempt, &tenant, &model, &dep, &prompt, &completion, &estimated, &estCost, &actual,
		&policy, &status, &started, &ended, &typCost, &typEst)
	if err != nil {
		t.Fatalf("read the row back: %v", err)
	}
	if request != "req-1" || attempt != 2 || tenant != "tenant-a" || model != "m" || dep != "m-a" ||
		prompt != 5 || completion != 4 || estimated != 1 || estCost != 1_234_567_890_123 || actual != 45_000_000 ||
		policy != "static-cheapest" || status != "http_503" {
		t.Errorf("row = %q %d %q %q %q %d %d %d %d %d %q %q", request, attempt, tenant, model, dep, prompt, completion,
			estimated, estCost, actual, policy, status)
	}
	if typCost != "integer" || typEst != "integer" {
		t.Errorf("money column types = %s, %s, want integer picodollars", typCost, typEst)
	}
	if started != "2026-10-15T12:00:00.123456789Z" || ended != "2026-10-15T12:00:01.623456789Z" {
		t.Errorf("timestamps = %q, %q, want fixed-width UTC RFC 3339 with nanoseconds", started, ended)
	}
}

// TestSQLiteKeepsNanoseconds: SQLite's text timestamps keep all nine digits, so a window can
// split two instants a nanosecond apart (PostgreSQL's timestamptz cannot).
func TestSQLiteKeepsNanoseconds(t *testing.T) {
	s, _ := openTemp(t)
	row := sampleRow("req-ns", 1)
	row.EndedAt = t0 // ...:00.123456789
	if err := s.RecordAttempt(context.Background(), row); err != nil {
		t.Fatal(err)
	}
	far := time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC)
	for _, c := range []struct {
		name     string
		from, to time.Time
		want     money.PicoUSD
	}{
		{"to is exclusive at the nanosecond", time.Time{}, t0, 0},
		{"one nanosecond later is inside", time.Time{}, t0.Add(time.Nanosecond), 45_000_000},
		{"from is inclusive at the nanosecond", t0, far, 45_000_000},
		{"one nanosecond later is outside", t0.Add(time.Nanosecond), far, 0},
	} {
		if got, err := s.TenantSpend(context.Background(), "tenant-a", c.from, c.to); err != nil || got != c.want {
			t.Errorf("%s: TenantSpend = %d, %v, want %d", c.name, got, err, c.want)
		}
	}
}

func TestSQLiteUsesWAL(t *testing.T) {
	_, path := openTemp(t)
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	var mode string
	if err := raw.QueryRow("PRAGMA journal_mode").Scan(&mode); err != nil || mode != "wal" {
		t.Errorf("journal_mode = %q, %v, want wal", mode, err)
	}
}

// TestSQLiteWaitsForALockedDatabase: another connection holds the write lock for a while, and
// the write waits (busy_timeout 5000) instead of failing at once with "database is locked".
func TestSQLiteWaitsForALockedDatabase(t *testing.T) {
	s, path := openTemp(t)
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	raw.SetMaxOpenConns(1)
	conn, err := raw.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := conn.ExecContext(context.Background(), "BEGIN IMMEDIATE"); err != nil {
		t.Fatal(err)
	}

	const hold = 300 * time.Millisecond
	time.AfterFunc(hold, func() { _, _ = conn.ExecContext(context.Background(), "COMMIT") })
	start := time.Now()
	if err := s.RecordAttempt(context.Background(), sampleRow("req-locked", 1)); err != nil {
		t.Fatalf("RecordAttempt while another connection holds the write lock: %v, want it to wait", err)
	}
	if waited := time.Since(start); waited < hold/2 {
		t.Errorf("RecordAttempt returned after %v, before the lock was released", waited)
	}
}

func TestSQLitePersistsAcrossOpen(t *testing.T) {
	s, path := openTemp(t)
	if err := s.RecordAttempt(context.Background(), sampleRow("req-1", 1)); err != nil {
		t.Fatal(err)
	}
	if err := s.RecordRollout(context.Background(), accounting.RolloutEvent{At: t0, Action: accounting.ActionReload}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	again, err := accounting.OpenSQLite(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer again.Close()
	if got, err := again.TenantSpend(context.Background(), "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC)); err != nil || got != 45_000_000 {
		t.Errorf("TenantSpend after reopen = %d, %v, want 45000000", got, err)
	}
	if got, err := again.Rollouts(context.Background(), 5); err != nil || len(got) != 1 {
		t.Errorf("Rollouts after reopen = %v, %v, want one event", got, err)
	}
	// A repeated attempt after a restart is still ignored.
	if err := again.RecordAttempt(context.Background(), sampleRow("req-1", 1)); err != nil {
		t.Fatal(err)
	}
	if got, _ := again.TenantSpend(context.Background(), "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC)); got != 45_000_000 {
		t.Errorf("TenantSpend after repeating a stored attempt = %d, want it unchanged", got)
	}
}

func TestOpenSQLiteCreatesTheParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a", "b", "usage.db")
	s, err := accounting.OpenSQLite(path)
	if err != nil {
		t.Fatalf("OpenSQLite in a missing directory: %v", err)
	}
	_ = s.Close()
}

func TestOpenSQLiteRejects(t *testing.T) {
	dir := t.TempDir()
	for name, path := range map[string]string{
		"empty path":              "",
		"path with a query mark":  filepath.Join(dir, "usage.db?mode=ro"),
		"a directory, not a file": dir,
	} {
		s, err := accounting.OpenSQLite(path)
		if err == nil {
			_ = s.Close()
			t.Errorf("OpenSQLite(%s) succeeded, want an error", name)
		}
	}
}

// TestSQLiteOverflowingSumIsAnError documents the one place the sum is not saturating: SQLite
// refuses an integer sum that overflows, and the store reports that instead of a wrong number.
func TestSQLiteOverflowingSumIsAnError(t *testing.T) {
	s, _ := openTemp(t)
	for i, cost := range []int64{1 << 62, 1 << 62, 1 << 62} {
		r := sampleRow("req-big", i+1)
		r.CostActual = money.PicoUSD(cost)
		if err := s.RecordAttempt(context.Background(), r); err != nil {
			t.Fatal(err)
		}
	}
	_, err := s.TenantSpend(context.Background(), "tenant-a", time.Time{}, time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC))
	if err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Errorf("TenantSpend over a sum above int64 = %v, want an overflow error", err)
	}
}
