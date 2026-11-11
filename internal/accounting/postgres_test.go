package accounting_test

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/accounting/accountingtest"
)

// postgresDSNEnv names the variable that points the Postgres tests at a server. Without it
// they skip, so the default suite needs no Docker.
const postgresDSNEnv = "ROUTEPILOT_TEST_POSTGRES_DSN"

// newSchema creates a schema of its own, dropped when the test ends, and returns an admin
// handle, the schema name and a DSN whose search_path is that schema, so subtests and reruns
// never see each other's rows.
func newSchema(t *testing.T) (admin *sql.DB, schema, dsn string) {
	t.Helper()
	base := os.Getenv(postgresDSNEnv)
	if base == "" {
		t.Skipf("%s is not set", postgresDSNEnv)
	}
	admin, err := sql.Open("pgx", base)
	if err != nil {
		t.Fatalf("open admin connection: %v", err)
	}
	t.Cleanup(func() { _ = admin.Close() })

	var b [6]byte
	_, _ = rand.Read(b[:])
	schema = "rp_test_" + hex.EncodeToString(b[:])
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+schema); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	t.Cleanup(func() { _, _ = admin.Exec("DROP SCHEMA " + schema + " CASCADE") })

	u, err := url.Parse(base)
	if err != nil {
		t.Fatalf("%s must be a postgres:// URL: %v", postgresDSNEnv, err)
	}
	q := u.Query()
	q.Set("search_path", schema)
	u.RawQuery = q.Encode()
	return admin, schema, u.String()
}

// openPostgres opens a store in a schema of its own.
func openPostgres(t *testing.T) (accounting.Store, *sql.DB, string) {
	t.Helper()
	admin, schema, dsn := newSchema(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	s, err := accounting.OpenPostgres(ctx, dsn)
	if err != nil {
		t.Fatalf("OpenPostgres: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, admin, schema
}

// TestPostgresConcurrentOpensCreateTheSchemaOnce: gateways that start together all create the
// tables, and the advisory lock keeps CREATE TABLE IF NOT EXISTS from racing.
func TestPostgresConcurrentOpensCreateTheSchemaOnce(t *testing.T) {
	_, _, dsn := newSchema(t)
	const opens = 8
	errs := make(chan error, opens)
	stores := make(chan accounting.Store, opens)
	for range opens {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			s, err := accounting.OpenPostgres(ctx, dsn)
			errs <- err
			stores <- s
		}()
	}
	for range opens {
		if err := <-errs; err != nil {
			t.Errorf("concurrent OpenPostgres: %v", err)
		}
		if s := <-stores; s != nil {
			_ = s.Close()
		}
	}
}

func TestPostgresConformance(t *testing.T) {
	accountingtest.Run(t, func(t *testing.T) accounting.Store {
		s, _, _ := openPostgres(t)
		return s
	})
}

// TestPostgresColumnTypes pins the shape: BIGINT picodollars, timestamptz instants and a
// primary key on (request_id, attempt).
func TestPostgresColumnTypes(t *testing.T) {
	_, admin, schema := openPostgres(t)
	types := map[string]string{}
	rows, err := admin.Query(`SELECT table_name || '.' || column_name, data_type
		FROM information_schema.columns WHERE table_schema = $1`, schema)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var name, typ string
		if err := rows.Scan(&name, &typ); err != nil {
			t.Fatal(err)
		}
		types[name] = typ
	}
	for col, want := range map[string]string{
		"usage.cost_actual":       "bigint",
		"usage.cost_estimate":     "bigint",
		"usage.prompt_tokens":     "bigint",
		"usage.usage_estimated":   "boolean",
		"usage.started_at":        "timestamp with time zone",
		"usage.ended_at":          "timestamp with time zone",
		"rollouts.occurred_at":    "timestamp with time zone",
		"rollouts.percent":        "double precision",
		"usage.attempt":           "integer",
		"usage.request_id":        "text",
		"rollouts.id":             "bigint",
		"rollouts.from_version":   "text",
		"rollouts.to_version":     "text",
		"usage.deployment":        "text",
		"usage.completion_tokens": "bigint",
		"usage.budget_period":     "text",
	} {
		if types[col] != want {
			t.Errorf("column %s has type %q, want %q", col, types[col], want)
		}
	}
	var pk string
	err = admin.QueryRow(`SELECT string_agg(kcu.column_name, ',' ORDER BY kcu.ordinal_position)
		FROM information_schema.table_constraints tc
		JOIN information_schema.key_column_usage kcu
		  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
		WHERE tc.table_schema = $1 AND tc.table_name = 'usage' AND tc.constraint_type = 'PRIMARY KEY'`, schema).Scan(&pk)
	if err != nil || pk != "request_id,attempt" {
		t.Errorf("primary key of usage = %q, %v, want request_id,attempt", pk, err)
	}
}

// TestOpenPostgresErrorOmitsTheDSN: a failed connection reports the cause without echoing the
// DSN, which carries the password.
func TestOpenPostgresErrorOmitsTheDSN(t *testing.T) {
	const password = "test-pass-not-real"
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s, err := accounting.OpenPostgres(ctx, "postgres://rp:"+password+"@127.0.0.1:1/rp?sslmode=disable&connect_timeout=2")
	if err == nil {
		_ = s.Close()
		t.Fatal("OpenPostgres to a closed port succeeded, want an error")
	}
	if strings.Contains(err.Error(), password) {
		t.Errorf("error %q contains the password", err)
	}
	if _, err := accounting.OpenPostgres(ctx, ""); err == nil {
		t.Error("OpenPostgres with an empty DSN succeeded, want an error")
	}
}

// TestPostgresUpgradesALedgerWithoutThePeriodColumn: opening a database written before the
// budget_period column existed adds the column, keeps the old rows (with the empty period)
// and takes new rows with a period; opening it again is harmless.
func TestPostgresUpgradesALedgerWithoutThePeriodColumn(t *testing.T) {
	admin, schema, dsn := newSchema(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	for _, stmt := range []string{
		`CREATE TABLE ` + schema + `.usage (
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
		`INSERT INTO ` + schema + `.usage VALUES ('old-req', 1, 'tenant-a', 'm', 'm-a', 5, 4, false, 10, 45000000,
			'pinned', 'ok', '2026-10-15T12:00:00Z', '2026-10-15T12:00:01Z')`,
	} {
		if _, err := admin.ExecContext(ctx, stmt); err != nil {
			t.Fatal(err)
		}
	}

	for pass := 1; pass <= 2; pass++ {
		s, err := accounting.OpenPostgres(ctx, dsn)
		if err != nil {
			t.Fatalf("OpenPostgres pass %d on a database without budget_period: %v", pass, err)
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
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
