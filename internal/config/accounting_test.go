package config

import (
	"fmt"
	"strings"
	"testing"
)

// accountingYAML is a valid config with the given extra top-level YAML appended.
func accountingYAML(extra string) string {
	return fmt.Sprintf(`
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
%s`, strings.Repeat("a", 64), extra)
}

func TestAccountingDefaultsToNone(t *testing.T) {
	cfg, err := Parse([]byte(accountingYAML("")))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if cfg.Accounting.Driver != AccountingNone {
		t.Errorf("Accounting.Driver = %q, want %q: a default run must not write a database", cfg.Accounting.Driver, AccountingNone)
	}
}

func TestAccountingParse(t *testing.T) {
	for _, c := range []struct {
		name, yaml string
		want       AccountingConfig
	}{
		{"none", "accounting:\n  driver: none\n", AccountingConfig{Driver: AccountingNone}},
		{"empty driver is none", "accounting: {}\n", AccountingConfig{Driver: AccountingNone}},
		{"sqlite", "accounting:\n  driver: sqlite\n  sqlite_path: data/usage.db\n",
			AccountingConfig{Driver: AccountingSQLite, SQLitePath: "data/usage.db"}},
		{"postgres", "accounting:\n  driver: postgres\n  postgres_dsn_env: ROUTEPILOT_PG\n",
			AccountingConfig{Driver: AccountingPostgres, PostgresDSNEnv: "ROUTEPILOT_PG"}},
	} {
		cfg, err := Parse([]byte(accountingYAML(c.yaml)))
		if err != nil {
			t.Errorf("%s: Parse() error = %v", c.name, err)
			continue
		}
		if cfg.Accounting != c.want {
			t.Errorf("%s: Accounting = %+v, want %+v", c.name, cfg.Accounting, c.want)
		}
	}
}

func TestAccountingRejects(t *testing.T) {
	for _, c := range []struct{ name, yaml, want string }{
		{"unknown driver", "accounting:\n  driver: mysql\n", "accounting.driver"},
		{"sqlite without a path", "accounting:\n  driver: sqlite\n", "accounting.sqlite_path"},
		{"postgres without a dsn variable", "accounting:\n  driver: postgres\n", "accounting.postgres_dsn_env"},
		{"sqlite with a dsn variable", "accounting:\n  driver: sqlite\n  sqlite_path: a.db\n  postgres_dsn_env: X\n", "accounting.postgres_dsn_env"},
		{"postgres with a sqlite path", "accounting:\n  driver: postgres\n  postgres_dsn_env: X\n  sqlite_path: a.db\n", "accounting.sqlite_path"},
		{"none with a path", "accounting:\n  driver: none\n  sqlite_path: a.db\n", "accounting.sqlite_path"},
		{"path without a driver", "accounting:\n  sqlite_path: a.db\n", "accounting.sqlite_path"},
		{"unknown key", "accounting:\n  driver: none\n  dsn: x\n", "dsn"},
	} {
		_, err := Parse([]byte(accountingYAML(c.yaml)))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Parse() error = %v, want it to mention %q", c.name, err, c.want)
		}
	}
}
