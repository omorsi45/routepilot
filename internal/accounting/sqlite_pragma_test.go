package accounting

import (
	"path/filepath"
	"testing"
)

// TestSQLiteConnectionPragmas: the connection the store writes through is in WAL mode with
// synchronous NORMAL. The default (FULL) fsyncs the log on every commit, which is what made
// one write per request too slow for 64 concurrent clients. Both are
// per-connection settings, so they are read from the store's own connection.
func TestSQLiteConnectionPragmas(t *testing.T) {
	s, err := OpenSQLite(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	db := s.(*sqlStore).db

	var synchronous int
	if err := db.QueryRow("PRAGMA synchronous").Scan(&synchronous); err != nil || synchronous != 1 {
		t.Errorf("PRAGMA synchronous = %d, %v, want 1 (NORMAL)", synchronous, err)
	}
	var mode string
	if err := db.QueryRow("PRAGMA journal_mode").Scan(&mode); err != nil || mode != "wal" {
		t.Errorf("PRAGMA journal_mode = %q, %v, want wal", mode, err)
	}
	var busy int
	if err := db.QueryRow("PRAGMA busy_timeout").Scan(&busy); err != nil || busy != 5000 {
		t.Errorf("PRAGMA busy_timeout = %d, %v, want 5000", busy, err)
	}
}
