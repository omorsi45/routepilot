package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
)

const usd = money.PicoUSD(1_000_000_000_000)

var reconcileAt = time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)

// reconcileEnv is a config on disk with three tenants (alpha: monthly 10 USD, beta: total
// 5 USD, gamma: no budget), a SQLite ledger next to it and a miniredis for the counters.
type reconcileEnv struct {
	t       *testing.T
	cfgPath string
	dbPath  string
	redis   *miniredis.Miniredis
	store   accounting.Store
}

func newReconcileEnv(t *testing.T, mutate func(yaml string) string) *reconcileEnv {
	t.Helper()
	dir := t.TempDir()
	e := &reconcileEnv{t: t, cfgPath: filepath.Join(dir, "config.yaml"), dbPath: filepath.Join(dir, "usage.db"), redis: miniredis.RunT(t)}
	t.Setenv("ROUTEPILOT_TEST_REDIS_URL", "redis://"+e.redis.Addr())
	yaml := fmt.Sprintf(`
providers:
  - name: p
    type: openaicompat
    base_url: http://127.0.0.1:1/v1
models:
  - id: m
    deployments:
      - provider: p
        upstream_model: m
quota:
  backend: redis
  redis_url_env: ROUTEPILOT_TEST_REDIS_URL
accounting:
  driver: sqlite
  sqlite_path: usage.db
tenants:
  - id: alpha
    api_key_sha256: %s
    allowed_models: ["*"]
    budget_usd: "10"
  - id: beta
    api_key_sha256: %s
    allowed_models: ["*"]
    budget_usd: "5"
    budget_period: total
  - id: gamma
    api_key_sha256: %s
    allowed_models: ["*"]
`, sha256Hex("key-alpha"), sha256Hex("key-beta"), sha256Hex("key-gamma"))
	if mutate != nil {
		yaml = mutate(yaml)
	}
	if err := os.WriteFile(e.cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return e
}

// openStore opens the SQLite ledger the config names.
func (e *reconcileEnv) openStore() accounting.Store {
	e.t.Helper()
	if e.store == nil {
		s, err := accounting.OpenSQLite(e.dbPath)
		if err != nil {
			e.t.Fatal(err)
		}
		e.store = s
	}
	return e.store
}

// ledger records a finished attempt of tenant that cost cost and ended at ended, charged to
// the budget period the tenant's budget has at that instant (none for a tenant without one).
func (e *reconcileEnv) ledger(tenant, request string, cost money.PicoUSD, ended time.Time) {
	e.t.Helper()
	period := ""
	switch tenant {
	case "alpha":
		period = quota.PeriodKey(alphaLim.Period, ended)
	case "beta":
		period = quota.PeriodKey(betaLim.Period, ended)
	}
	e.ledgerIn(tenant, request, cost, ended, period)
}

// ledgerIn records a finished attempt that was charged to the budget period named period.
func (e *reconcileEnv) ledgerIn(tenant, request string, cost money.PicoUSD, ended time.Time, period string) {
	e.t.Helper()
	err := e.openStore().RecordAttempt(context.Background(), accounting.UsageRow{
		RequestID: request, Attempt: 1, Tenant: tenant, Model: "m", Deployment: "d", Status: "ok",
		CostActual: cost, StartedAt: ended.Add(-time.Second), EndedAt: ended, BudgetPeriod: period,
	})
	if err != nil {
		e.t.Fatal(err)
	}
}

// counter charges the budget counter of tenant, as the gateway's reserve and commit do.
func (e *reconcileEnv) counter(tenant string, lim quota.BudgetLimit, cost money.PicoUSD) {
	e.t.Helper()
	e.counterAt(tenant, lim, cost, reconcileAt)
}

// counterAt is counter for a request that reserved and committed at instant at.
func (e *reconcileEnv) counterAt(tenant string, lim quota.BudgetLimit, cost money.PicoUSD, at time.Time) {
	e.t.Helper()
	l := quota.NewRedisBudget(redisClientFor(e.t, e.redis), redisKeyPrefix)
	id := "counter-" + tenant + at.Format(time.RFC3339)
	ok, _, err := l.Reserve(context.Background(), tenant, id, lim, cost, time.Hour, at)
	if err != nil || !ok {
		e.t.Fatalf("Reserve(%s): ok=%v err=%v", tenant, ok, err)
	}
	if _, err := l.Commit(context.Background(), tenant, id, cost, at); err != nil {
		e.t.Fatal(err)
	}
}

func (e *reconcileEnv) run(extra ...string) (code int, stdout, stderr string) {
	e.t.Helper()
	if e.store != nil { // release the SQLite file before reconcile opens it
		_ = e.store.Close()
		e.store = nil
	}
	var out, errOut bytes.Buffer
	args := append([]string{"--config", e.cfgPath}, extra...)
	code = runReconcile(args, &out, &errOut)
	return code, out.String(), errOut.String()
}

var (
	alphaLim = quota.BudgetLimit{Limit: 10 * usd, Period: quota.PeriodMonthly}
	betaLim  = quota.BudgetLimit{Limit: 5 * usd, Period: quota.PeriodTotal}
)

// TestReconcileReportsDifference: the ledger and the counter disagree for alpha, so the line
// shows both and the difference, the exit code is 1, and beta, which agrees, prints a zero.
// Rows outside alpha's month do not count, and beta's total period counts them all.
func TestReconcileReportsDifference(t *testing.T) {
	e := newReconcileEnv(t, nil)
	e.ledger("alpha", "a1", 2*usd, reconcileAt.Add(-24*time.Hour))
	e.ledger("alpha", "a2", usd/2, reconcileAt)
	e.ledger("alpha", "a-september", 7*usd, time.Date(2026, 9, 30, 23, 59, 59, 0, time.UTC))
	e.ledger("alpha", "a-november", 9*usd, time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC))
	e.counter("alpha", alphaLim, 2*usd)

	e.ledger("beta", "b-old", usd/2, time.Date(2026, 1, 3, 0, 0, 0, 0, time.UTC))
	e.ledger("beta", "b-now", usd/2, reconcileAt)
	e.counter("beta", betaLim, usd)

	e.ledger("gamma", "g1", usd, reconcileAt) // gamma has no budget: not reported

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 1 {
		t.Errorf("exit code = %d, want 1 for a difference; stderr: %s", code, stderr)
	}
	if want := "alpha 2.5 2 0.5\nbeta 1 1 0\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
}

func TestReconcileExitsZeroWhenEverythingMatches(t *testing.T) {
	e := newReconcileEnv(t, nil)
	e.ledger("alpha", "a1", 3*usd, reconcileAt)
	e.counter("alpha", alphaLim, 3*usd)
	e.counter("beta", betaLim, 0)

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 0 {
		t.Errorf("exit code = %d, want 0; stderr: %s", code, stderr)
	}
	if want := "alpha 3 3 0\nbeta 0 0 0\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
}

// TestReconcileShowsALostCommit: the ledger holds less than the counter (rows were dropped),
// which prints a negative difference and still exits 1.
func TestReconcileShowsALedgerBehindTheCounter(t *testing.T) {
	e := newReconcileEnv(t, nil)
	e.ledger("alpha", "a1", usd, reconcileAt)
	e.counter("alpha", alphaLim, 3*usd/2)
	e.counter("beta", betaLim, 0)

	code, stdout, _ := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 1 || !strings.HasPrefix(stdout, "alpha 1 1.5 -0.5\n") {
		t.Errorf("exit %d stdout %q, want exit 1 and the line %q first", code, stdout, "alpha 1 1.5 -0.5")
	}
}

// TestReconcileDefaultsToNow: without --at the monthly window is the month of the clock.
func TestReconcileDefaultsToNow(t *testing.T) {
	old := reconcileClock
	reconcileClock = func() time.Time { return reconcileAt }
	t.Cleanup(func() { reconcileClock = old })
	e := newReconcileEnv(t, nil)
	e.ledger("alpha", "a1", usd, reconcileAt)
	e.counter("alpha", alphaLim, usd)
	e.counter("beta", betaLim, 0)

	code, stdout, stderr := e.run()

	if code != 0 || !strings.HasPrefix(stdout, "alpha 1 1 0\n") {
		t.Errorf("exit %d stdout %q stderr %q, want exit 0 and alpha 1 1 0", code, stdout, stderr)
	}
}

// TestReconcileMemoryBackendCannotBeRead: the memory ledger lives in the gateway process.
func TestReconcileMemoryBackendCannotBeRead(t *testing.T) {
	e := newReconcileEnv(t, func(y string) string {
		return strings.Replace(y, "quota:\n  backend: redis\n  redis_url_env: ROUTEPILOT_TEST_REDIS_URL\n", "", 1)
	})
	e.ledger("alpha", "a1", usd, reconcileAt)

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 2 {
		t.Errorf("exit code = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "memory") || !strings.Contains(stderr, "gateway process") {
		t.Errorf("stderr = %q, want it to say the memory ledger lives in the gateway process", stderr)
	}
}

func TestReconcileNeedsAnAccountingStore(t *testing.T) {
	e := newReconcileEnv(t, func(y string) string {
		return strings.Replace(y, "accounting:\n  driver: sqlite\n  sqlite_path: usage.db\n", "", 1)
	})

	code, _, stderr := e.run()

	if code != 2 || !strings.Contains(stderr, "accounting") {
		t.Errorf("exit %d stderr %q, want exit 2 naming accounting", code, stderr)
	}
}

// TestReconcileDoesNotCreateTheDatabase: reading a ledger that was never written is an error,
// not an empty database created as a side effect.
func TestReconcileDoesNotCreateTheDatabase(t *testing.T) {
	e := newReconcileEnv(t, nil)

	code, _, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 2 || !strings.Contains(stderr, "usage.db") {
		t.Errorf("exit %d stderr %q, want exit 2 naming the missing database", code, stderr)
	}
	if _, err := os.Stat(e.dbPath); !os.IsNotExist(err) {
		t.Errorf("reconcile created %s (stat error %v)", e.dbPath, err)
	}
}

func TestReconcileWithoutBudgets(t *testing.T) {
	e := newReconcileEnv(t, func(y string) string {
		y = strings.Replace(y, "    budget_usd: \"10\"\n", "", 1)
		y = strings.Replace(y, "    budget_usd: \"5\"\n    budget_period: total\n", "", 1)
		return y
	})
	e.ledger("alpha", "a1", usd, reconcileAt)

	code, stdout, stderr := e.run("--at", reconcileAt.Format(time.RFC3339))

	if code != 0 || stdout != "" || !strings.Contains(stderr, "no tenant has a budget") {
		t.Errorf("exit %d stdout %q stderr %q, want exit 0, no lines and a note", code, stdout, stderr)
	}
}

func TestReconcileUsageErrors(t *testing.T) {
	e := newReconcileEnv(t, nil)
	for name, args := range map[string][]string{
		"no arguments":      nil,
		"bad --at":          {"--config", e.cfgPath, "--at", "yesterday"},
		"extra positional":  {"--config", e.cfgPath, "extra"},
		"unknown flag":      {"--config", e.cfgPath, "--nope"},
		"missing file":      {"--config", filepath.Join(t.TempDir(), "absent.yaml")},
		"--at without zone": {"--config", e.cfgPath, "--at", "2026-10-15T12:00:00"},
	} {
		var out, errOut bytes.Buffer
		if code := runReconcile(args, &out, &errOut); code != 2 {
			t.Errorf("%s: exit code = %d, want 2; stderr: %s", name, code, errOut.String())
		}
		if out.Len() != 0 {
			t.Errorf("%s: stdout = %q, want nothing", name, out.String())
		}
	}
}

// TestReconcileUsageSaysToRunItOnAQuietGateway: against a gateway that is serving, a
// difference can be requests in flight, and the usage says how to tell.
func TestReconcileUsageSaysToRunItOnAQuietGateway(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runReconcile(nil, &out, &errOut); code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	for _, want := range []string{"quiet gateway", "in flight", "a few seconds later"} {
		if !strings.Contains(errOut.String(), want) {
			t.Errorf("usage %q does not contain %q", errOut.String(), want)
		}
	}
}

// TestRunDispatchesReconcile: the subcommand is reachable from the CLI and listed in its usage.
func TestRunDispatchesReconcile(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"reconcile"}, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "--config") {
		t.Errorf("run(reconcile) = %d stderr %q, want the reconcile usage and exit 2", code, stderr.String())
	}
	stderr.Reset()
	run(nil, &stdout, &stderr)
	if !strings.Contains(stderr.String(), "reconcile") {
		t.Errorf("usage %q does not list reconcile", stderr.String())
	}
}
