package accounting

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// sqliteTimeLayout is UTC RFC 3339 with all nine fractional digits. The width is fixed on
// purpose: time.RFC3339Nano trims trailing zeros, and as text "...:00Z" sorts after
// "...:00.5Z", so a range query over trimmed timestamps would be wrong. Fixed-width text
// compares the way the instants do.
const sqliteTimeLayout = "2006-01-02T15:04:05.000000000Z07:00"

// sqliteDSNQuery sets the pragmas every connection gets: wait up to 5 s for a lock instead of
// failing at once, write-ahead logging so a reader (routepilot reconcile) does not block the
// gateway's writes, and synchronous NORMAL, which in WAL mode syncs the log at checkpoints
// and not at every commit. A crash of the process loses nothing; a power cut can lose the
// last transactions but cannot corrupt the database.
const sqliteDSNQuery = "?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)"

var sqliteDDL = []string{
	`CREATE TABLE IF NOT EXISTS usage (
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
		budget_period     TEXT    NOT NULL DEFAULT '',
		PRIMARY KEY (request_id, attempt)
	)`,
	`CREATE INDEX IF NOT EXISTS usage_tenant_ended ON usage (tenant, ended_at)`,
	`CREATE TABLE IF NOT EXISTS rollouts (
		id           INTEGER PRIMARY KEY AUTOINCREMENT,
		occurred_at  TEXT    NOT NULL,
		action       TEXT    NOT NULL,
		from_version TEXT    NOT NULL,
		to_version   TEXT    NOT NULL,
		percent      REAL    NOT NULL,
		reason       TEXT    NOT NULL,
		actor        TEXT    NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS rollouts_occurred ON rollouts (occurred_at)`,
}

var sqliteDialect = dialect{
	name:   "sqlite",
	rebind: func(q string) string { return q },
	encodeBool: func(b bool) any {
		if b {
			return int64(1)
		}
		return int64(0)
	},
	encodeTime: func(t time.Time) any { return t.UTC().Format(sqliteTimeLayout) },
	ddl:        sqliteDDL,
	// SQLite refuses a sum of integers that overflows int64 ("integer overflow") and the store
	// reports that error: a tenant would need over 9 million USD in one window.
	sumSQL: `SELECT COALESCE(SUM(cost_actual), 0) FROM usage WHERE tenant = ? AND ended_at >= ? AND ended_at < ?`,
	// The same overflow error applies.
	periodSumSQL: `SELECT COALESCE(SUM(cost_actual), 0) FROM usage WHERE tenant = ? AND budget_period = ?`,
	upgrade:      upgradeSQLite,
}

// upgradeSQLite adds the budget_period column to a usage table created before it existed
// (CREATE TABLE IF NOT EXISTS leaves such a table as it is), giving the rows already there the
// empty period, and then indexes it. SQLite has no ADD COLUMN IF NOT EXISTS, so the column is
// looked up first; a second process that adds it between the lookup and the ALTER is not an
// error.
func upgradeSQLite(ctx context.Context, db *sql.DB) error {
	var have int
	err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pragma_table_info('usage') WHERE name = 'budget_period'`).Scan(&have)
	if err != nil {
		return err
	}
	if have == 0 {
		_, err := db.ExecContext(ctx, `ALTER TABLE usage ADD COLUMN budget_period TEXT NOT NULL DEFAULT ''`)
		if err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return err
		}
	}
	_, err = db.ExecContext(ctx, `CREATE INDEX IF NOT EXISTS usage_tenant_period ON usage (tenant, budget_period)`)
	return err
}

// OpenSQLite opens (creating it and its directory if needed) the SQLite ledger at path. A
// single connection serves the store, so writes queue in the driver instead of contending
// for the file lock; the database is in WAL mode, which lets another process read it
// meanwhile.
func OpenSQLite(path string) (Store, error) {
	switch {
	case path == "":
		return nil, fmt.Errorf("accounting: sqlite path is empty")
	case strings.Contains(path, "?"):
		return nil, fmt.Errorf("accounting: sqlite path %q must not contain '?'", path)
	case strings.HasPrefix(path, "file:"):
		return nil, fmt.Errorf("accounting: sqlite path %q must be a file path, not a URI", path)
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return nil, fmt.Errorf("accounting: create directory for %s: %w", path, err)
		}
	}
	db, err := sql.Open("sqlite", path+sqliteDSNQuery)
	if err != nil {
		return nil, fmt.Errorf("accounting: open sqlite %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)
	s := &sqlStore{db: db, d: sqliteDialect}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.migrate(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("%w (path %s)", err, path)
	}
	return s, nil
}
