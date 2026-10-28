package accounting

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib" // registers the "pgx" database/sql driver
)

// postgresMaxConns bounds the pool: the gateway writes a handful of small rows per request
// and each write is bounded to 250 ms, so a few connections are enough.
const postgresMaxConns = 8

// postgresMigrationLock is the advisory lock that serializes schema creation, so two
// gateways starting together do not race in CREATE TABLE IF NOT EXISTS.
const postgresMigrationLock = 727274301

var postgresDDL = []string{
	`CREATE TABLE IF NOT EXISTS usage (
		request_id        TEXT             NOT NULL,
		attempt           INTEGER          NOT NULL,
		tenant            TEXT             NOT NULL,
		model             TEXT             NOT NULL,
		deployment        TEXT             NOT NULL,
		prompt_tokens     BIGINT           NOT NULL,
		completion_tokens BIGINT           NOT NULL,
		usage_estimated   BOOLEAN          NOT NULL,
		cost_estimate     BIGINT           NOT NULL,
		cost_actual       BIGINT           NOT NULL,
		policy_version    TEXT             NOT NULL,
		status            TEXT             NOT NULL,
		started_at        TIMESTAMPTZ      NOT NULL,
		ended_at          TIMESTAMPTZ      NOT NULL,
		PRIMARY KEY (request_id, attempt)
	)`,
	`CREATE INDEX IF NOT EXISTS usage_tenant_ended ON usage (tenant, ended_at)`,
	`CREATE TABLE IF NOT EXISTS rollouts (
		id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
		occurred_at  TIMESTAMPTZ      NOT NULL,
		action       TEXT             NOT NULL,
		from_version TEXT             NOT NULL,
		to_version   TEXT             NOT NULL,
		percent      DOUBLE PRECISION NOT NULL,
		reason       TEXT             NOT NULL,
		actor        TEXT             NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS rollouts_occurred ON rollouts (occurred_at)`,
}

var postgresDialect = dialect{
	name:       "postgres",
	rebind:     numbered,
	encodeBool: func(b bool) any { return b },
	encodeTime: func(t time.Time) any { return t.UTC() },
	ddl:        postgresDDL,
	// SUM of a bigint is a numeric in PostgreSQL, so it cannot overflow; it is clamped to the
	// int64 range, which saturates the way the Go side does.
	sumSQL: `SELECT LEAST(COALESCE(SUM(cost_actual), 0), 9223372036854775807)::bigint
		FROM usage WHERE tenant = ? AND ended_at >= ? AND ended_at < ?`,
}

// OpenPostgres connects to the PostgreSQL ledger at dsn (a postgres:// URL or key/value
// string), creating the tables if they are missing. The DSN carries the password, so no
// error returned here repeats it.
func OpenPostgres(ctx context.Context, dsn string) (Store, error) {
	if dsn == "" {
		return nil, errors.New("accounting: postgres DSN is empty")
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, redacted(fmt.Errorf("accounting: open postgres: %w", err), dsn)
	}
	db.SetMaxOpenConns(postgresMaxConns)
	db.SetConnMaxIdleTime(5 * time.Minute)
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, redacted(fmt.Errorf("accounting: connect to postgres: %w", err), dsn)
	}
	if err := migratePostgres(ctx, db); err != nil {
		_ = db.Close()
		return nil, redacted(err, dsn)
	}
	return &sqlStore{db: db, d: postgresDialect}, nil
}

// migratePostgres creates the schema in one transaction under an advisory lock.
func migratePostgres(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("accounting: create postgres schema: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock($1)", postgresMigrationLock); err != nil {
		return fmt.Errorf("accounting: create postgres schema: %w", err)
	}
	for _, stmt := range postgresDDL {
		if _, err := tx.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("accounting: create postgres schema: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("accounting: create postgres schema: %w", err)
	}
	return nil
}

// redacted returns err with the DSN's password removed from its text, should a driver ever
// quote the connection string. The error chain is kept for errors.Is and errors.As.
func redacted(err error, dsn string) error {
	var secrets []string
	if u, perr := url.Parse(dsn); perr == nil && u.User != nil {
		if pw, ok := u.User.Password(); ok && pw != "" {
			secrets = append(secrets, pw)
		}
	}
	for _, field := range strings.Fields(dsn) {
		if pw, ok := strings.CutPrefix(field, "password="); ok && pw != "" {
			secrets = append(secrets, strings.Trim(pw, "'"))
		}
	}
	if len(secrets) == 0 {
		return err
	}
	msg := err.Error()
	for _, s := range secrets {
		msg = strings.ReplaceAll(msg, s, "***")
	}
	return &redactedError{msg: msg, err: err}
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.err }
