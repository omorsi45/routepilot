package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/mock"
)

// accountingConfig is budgetServeConfig plus an accounting section.
func accountingConfig(t *testing.T, upstreamURL, accountingYAML string) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
providers:
  - name: mockp
    type: openaicompat
    base_url: %q
models:
  - id: m
    input_price_per_mtok: "1"
    output_price_per_mtok: "2"
    deployments:
      - provider: mockp
        upstream_model: m
tenants:
  - id: free
    api_key_sha256: %s
    allowed_models: ["*"]
%s`, upstreamURL+"/v1", sha256Hex("test-key-free"), accountingYAML)))
	if err != nil {
		t.Fatalf("config.Parse() error = %v", err)
	}
	return cfg
}

func farFuture() time.Time { return time.Date(9999, 1, 1, 0, 0, 0, 0, time.UTC) }

// TestServeWritesTheUsageLedger: with accounting.driver sqlite the gateway records the attempt
// of a request, and shutdown closes the store so the file can be read afterwards.
func TestServeWritesTheUsageLedger(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	db := filepath.Join(t.TempDir(), "usage.db")
	cfg := accountingConfig(t, upstream.URL, fmt.Sprintf("accounting:\n  driver: sqlite\n  sqlite_path: %q\n", filepath.ToSlash(db)))
	url, stop := startBudgetServe(t, cfg)

	if code, body := postChat(t, url, "test-key-free"); code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	stop()

	store, err := accounting.OpenSQLite(db)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	spent, err := store.TenantSpend(context.Background(), "free", time.Time{}, farFuture())
	if err != nil || spent <= 0 {
		t.Errorf("TenantSpend(free) = %d, %v, want the request's cost", spent, err)
	}
}

// TestServeWithoutAccountingWritesNoDatabase: the default is driver none.
func TestServeWithoutAccountingWritesNoDatabase(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	dir := t.TempDir()
	t.Chdir(dir)
	url, stop := startBudgetServe(t, accountingConfig(t, upstream.URL, ""))

	if code, body := postChat(t, url, "test-key-free"); code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	stop()

	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Errorf("working directory holds %v (err %v), want no file written", entries, err)
	}
}

func TestOpenAccounting(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	t.Run("none", func(t *testing.T) {
		s, err := openAccounting(ctx, accountingConfig(t, "http://127.0.0.1:1", ""), os.Getenv)
		if s != nil || err != nil {
			t.Errorf("openAccounting(none) = %v, %v, want nil, nil", s, err)
		}
	})

	t.Run("sqlite path is relative to the config file", func(t *testing.T) {
		cfgPath := filepath.Join(dir, "config.yaml")
		y := fmt.Sprintf(`
providers:
  - name: p
    type: openaicompat
    base_url: http://127.0.0.1:1/v1
models:
  - id: m
    deployments:
      - provider: p
        upstream_model: m
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
accounting:
  driver: sqlite
  sqlite_path: nested/usage.db
`, sha256Hex("k"))
		if err := os.WriteFile(cfgPath, []byte(y), 0o600); err != nil {
			t.Fatal(err)
		}
		cfg, err := config.Load(cfgPath)
		if err != nil {
			t.Fatal(err)
		}
		s, err := openAccounting(ctx, cfg, os.Getenv)
		if err != nil || s == nil {
			t.Fatalf("openAccounting(sqlite) = %v, %v", s, err)
		}
		_ = s.Close()
		if _, err := os.Stat(filepath.Join(dir, "nested", "usage.db")); err != nil {
			t.Errorf("database not created next to the config: %v", err)
		}
	})

	t.Run("postgres needs its variable and never echoes the DSN", func(t *testing.T) {
		cfg := accountingConfig(t, "http://127.0.0.1:1", "accounting:\n  driver: postgres\n  postgres_dsn_env: ROUTEPILOT_TEST_PG_DSN\n")
		_, err := openAccounting(ctx, cfg, func(string) string { return "" })
		if err == nil || !strings.Contains(err.Error(), "ROUTEPILOT_TEST_PG_DSN") {
			t.Errorf("openAccounting with an unset variable = %v, want an error naming it", err)
		}

		const password = "test-pass-not-real"
		dsn := "postgres://rp:" + password + "@127.0.0.1:1/rp?sslmode=disable&connect_timeout=2"
		_, err = openAccounting(ctx, cfg, func(string) string { return dsn })
		if err == nil {
			t.Fatal("openAccounting to a closed port succeeded")
		}
		if strings.Contains(err.Error(), password) {
			t.Errorf("error %q contains the DSN password", err)
		}
	})
}

// TestServeFailsWhenTheLedgerCannotBeOpened: a bad ledger is a startup error, not a gateway
// that silently runs without one.
func TestServeFailsWhenTheLedgerCannotBeOpened(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	notADir := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(notADir, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := accountingConfig(t, upstream.URL, fmt.Sprintf("accounting:\n  driver: sqlite\n  sqlite_path: %q\n", filepath.ToSlash(filepath.Join(notADir, "usage.db"))))

	ln := listenLocal(t)
	err := serve(context.Background(), cfg, ln, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err == nil || !strings.Contains(err.Error(), "accounting") {
		t.Errorf("serve() = %v, want an accounting startup error", err)
	}
}
