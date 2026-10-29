package accounting

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/omorsi45/routepilot/internal/money"
)

// dialect is what differs between the SQL backends: placeholder syntax, how booleans and
// instants are stored, the DDL and the saturating sum. Everything else is shared in sqlStore
// so the two backends cannot drift apart.
type dialect struct {
	// name labels errors.
	name string
	// rebind turns the "?" placeholders of the shared queries into the backend's own.
	rebind     func(query string) string
	encodeBool func(bool) any
	encodeTime func(time.Time) any
	// ddl creates the tables and indexes if they are missing.
	ddl []string
	// sumSQL selects the sum of cost_actual for tenant, from and to, in that order.
	sumSQL string
}

const (
	insertUsageSQL = `INSERT INTO usage (request_id, attempt, tenant, model, deployment, prompt_tokens,
		completion_tokens, usage_estimated, cost_estimate, cost_actual, policy_version, status, started_at, ended_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (request_id, attempt) DO NOTHING`
	insertRolloutSQL = `INSERT INTO rollouts (occurred_at, action, from_version, to_version, percent, reason, actor)
		VALUES (?, ?, ?, ?, ?, ?, ?)`
	selectRolloutsSQL = `SELECT occurred_at, action, from_version, to_version, percent, reason, actor
		FROM rollouts ORDER BY occurred_at DESC, id DESC LIMIT ?`
)

// sqlStore is the Store over a database/sql handle.
type sqlStore struct {
	db *sql.DB
	d  dialect
}

// migrate creates the schema. It is safe to run on every start.
func (s *sqlStore) migrate(ctx context.Context) error {
	for _, stmt := range s.d.ddl {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("accounting: create %s schema: %w", s.d.name, err)
		}
	}
	return nil
}

func (s *sqlStore) RecordAttempt(ctx context.Context, r UsageRow) error {
	if err := r.validate(); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, s.d.rebind(insertUsageSQL), s.usageArgs(r)...)
	if err != nil {
		return fmt.Errorf("accounting: record attempt %s/%d: %w", r.RequestID, r.Attempt, err)
	}
	return nil
}

// RecordAttempts writes the rows in one transaction with one prepared insert, so the commit
// (and with it the log sync) is paid once per batch and not once per row. Every row is
// validated before the transaction starts, so a bad row rejects the batch whole.
func (s *sqlStore) RecordAttempts(ctx context.Context, rows []UsageRow) error {
	if len(rows) == 0 {
		return nil
	}
	for i, r := range rows {
		if err := r.validate(); err != nil {
			return fmt.Errorf("accounting: batch row %d: %w", i, err)
		}
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("accounting: record %d attempts: %w", len(rows), err)
	}
	defer func() { _ = tx.Rollback() }() // a no-op once the transaction is committed
	stmt, err := tx.PrepareContext(ctx, s.d.rebind(insertUsageSQL))
	if err != nil {
		return fmt.Errorf("accounting: record %d attempts: %w", len(rows), err)
	}
	defer stmt.Close()
	for _, r := range rows {
		if _, err := stmt.ExecContext(ctx, s.usageArgs(r)...); err != nil {
			return fmt.Errorf("accounting: record attempt %s/%d: %w", r.RequestID, r.Attempt, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("accounting: record %d attempts: %w", len(rows), err)
	}
	return nil
}

// usageArgs are the values of insertUsageSQL for r, in the order of its placeholders.
func (s *sqlStore) usageArgs(r UsageRow) []any {
	return []any{
		r.RequestID, r.Attempt, r.Tenant, r.Model, r.Deployment, r.PromptTokens, r.CompletionTokens,
		s.d.encodeBool(r.UsageEstimated), int64(r.CostEstimate), int64(r.CostActual), r.PolicyVersion, r.Status,
		s.d.encodeTime(r.StartedAt), s.d.encodeTime(r.EndedAt),
	}
}

func (s *sqlStore) RecordRollout(ctx context.Context, e RolloutEvent) error {
	if err := e.validate(); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, s.d.rebind(insertRolloutSQL),
		s.d.encodeTime(e.At), e.Action, e.From, e.To, e.Percent, e.Reason, e.Actor)
	if err != nil {
		return fmt.Errorf("accounting: record rollout: %w", err)
	}
	return nil
}

func (s *sqlStore) TenantSpend(ctx context.Context, tenant string, from, to time.Time) (money.PicoUSD, error) {
	var sum int64
	err := s.db.QueryRowContext(ctx, s.d.rebind(s.d.sumSQL), tenant, s.d.encodeTime(from), s.d.encodeTime(to)).Scan(&sum)
	if err != nil {
		return 0, fmt.Errorf("accounting: sum spend of %s: %w", tenant, err)
	}
	return money.PicoUSD(sum), nil
}

func (s *sqlStore) Rollouts(ctx context.Context, limit int) ([]RolloutEvent, error) {
	if limit < 1 {
		return nil, fmt.Errorf("accounting: rollouts limit %d, want 1 or more", limit)
	}
	rows, err := s.db.QueryContext(ctx, s.d.rebind(selectRolloutsSQL), limit)
	if err != nil {
		return nil, fmt.Errorf("accounting: read rollouts: %w", err)
	}
	defer rows.Close()
	out := []RolloutEvent{}
	for rows.Next() {
		var (
			e  RolloutEvent
			at instant
		)
		if err := rows.Scan(&at, &e.Action, &e.From, &e.To, &e.Percent, &e.Reason, &e.Actor); err != nil {
			return nil, fmt.Errorf("accounting: read rollouts: %w", err)
		}
		e.At = at.Time
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("accounting: read rollouts: %w", err)
	}
	return out, nil
}

func (s *sqlStore) Close() error { return s.db.Close() }

// instant scans a stored timestamp, whether the driver hands it over as a time.Time
// (PostgreSQL) or as the fixed-width text SQLite keeps, and yields it in UTC.
type instant struct{ time.Time }

func (i *instant) Scan(src any) error {
	switch v := src.(type) {
	case time.Time:
		i.Time = v.UTC()
	case string:
		return i.parse(v)
	case []byte:
		return i.parse(string(v))
	default:
		return fmt.Errorf("accounting: cannot read a timestamp from %T", src)
	}
	return nil
}

func (i *instant) parse(s string) error {
	t, err := time.Parse(sqliteTimeLayout, s)
	if err != nil {
		return fmt.Errorf("accounting: bad stored timestamp %q: %w", s, err)
	}
	i.Time = t.UTC()
	return nil
}

// numbered rewrites the "?" placeholders of query as $1, $2, ... in order.
func numbered(query string) string {
	var b strings.Builder
	n := 0
	for _, r := range query {
		if r == '?' {
			n++
			fmt.Fprintf(&b, "$%d", n)
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}
