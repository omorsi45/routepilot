package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/quota"
)

const reconcileUsage = "usage: routepilot reconcile --config FILE [--at RFC3339]\n" +
	"\n" +
	"For each tenant with a budget, compares what the usage ledger says it spent in the budget\n" +
	"period with the budget counter, and prints one line per tenant:\n" +
	"\n" +
	"    tenant ledger_usd counter_usd diff_usd\n" +
	"\n" +
	"diff is ledger minus counter. A monthly budget is compared over the UTC month that contains\n" +
	"--at (default: now); a total budget over all time. The counter is read from the quota\n" +
	"backend the config names, so it needs quota.backend redis: the memory counter lives in the\n" +
	"gateway process.\n" +
	"\n" +
	"exit codes: 0 every tenant matches, 1 at least one differs, 2 nothing could be compared\n"

// reconcileClock is the clock behind the default of --at. Tests replace it.
var reconcileClock = time.Now

// reconcileTimeout bounds all the reads of one run.
const reconcileTimeout = 60 * time.Second

// runReconcile implements `routepilot reconcile`. It returns 0 when the ledger and the budget
// counter agree for every tenant with a budget, 1 when any differ, and 2 when they could not
// be compared: a usage error, an unreadable config, no ledger, a memory counter, or a read
// that failed.
func runReconcile(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("routepilot reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "path to the YAML config file (required)")
	atFlag := fs.String("at", "", "RFC 3339 instant that picks the budget period (default: now)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *configPath == "" || fs.NArg() > 0 {
		fmt.Fprintf(stderr, "routepilot reconcile: --config is required\n%s", reconcileUsage)
		return 2
	}
	at := reconcileClock()
	if *atFlag != "" {
		var err error
		if at, err = time.Parse(time.RFC3339, *atFlag); err != nil {
			fmt.Fprintf(stderr, "routepilot reconcile: --at must be an RFC 3339 time such as 2026-10-15T12:00:00Z: %v\n", err)
			return 2
		}
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot reconcile: %v\n", err)
		return 2
	}
	if cfg.Accounting.Driver == config.AccountingNone {
		fmt.Fprintln(stderr, "routepilot reconcile: accounting.driver is none, so there is no usage ledger to compare")
		return 2
	}
	if cfg.Quota.Backend != config.QuotaRedis {
		fmt.Fprintln(stderr, "routepilot reconcile: quota.backend is memory, and the memory budget counter lives in the gateway process, "+
			"so it cannot be read from here; use quota.backend redis")
		return 2
	}

	type budgeted struct {
		id  string
		lim quota.BudgetLimit
	}
	var tenants []budgeted
	for _, t := range cfg.Tenants {
		lim, ok, err := t.BudgetLimit()
		if err != nil {
			fmt.Fprintf(stderr, "routepilot reconcile: %v\n", err)
			return 2
		}
		if ok {
			tenants = append(tenants, budgeted{t.ID, lim})
		}
	}
	if len(tenants) == 0 {
		fmt.Fprintln(stderr, "routepilot reconcile: no tenant has a budget, nothing to reconcile")
		return 0
	}

	ctx, cancel := context.WithTimeout(context.Background(), reconcileTimeout)
	defer cancel()
	limiters, closeLimiters, err := buildLimiters(ctx, cfg, os.Getenv)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot reconcile: %v\n", err)
		return 2
	}
	defer closeLimiters()
	store, err := openLedgerForReading(ctx, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot reconcile: %v\n", err)
		return 2
	}
	defer store.Close()

	differ := false
	for _, t := range tenants {
		from, to := reconcileWindow(t.lim.Period, at)
		ledger, err := store.TenantSpend(ctx, t.id, from, to)
		if err != nil {
			fmt.Fprintf(stderr, "routepilot reconcile: %v\n", err)
			return 2
		}
		counter, err := limiters.Budget.Spent(ctx, t.id, t.lim, at)
		if err != nil {
			fmt.Fprintf(stderr, "routepilot reconcile: read the budget counter of %s: %v\n", t.id, err)
			return 2
		}
		fmt.Fprintf(stdout, "%s %s %s %s\n", t.id, ledger.USD(), counter.USD(), (ledger - counter).USD())
		if ledger != counter {
			differ = true
		}
	}
	if differ {
		return 1
	}
	return 0
}

// reconcileWindow is the span of ended_at the ledger is summed over for a budget period: the
// UTC month that contains at for a monthly budget, and all time for a total one.
func reconcileWindow(period string, at time.Time) (from, to time.Time) {
	if period == quota.PeriodTotal {
		return time.Time{}, time.Date(9999, 12, 31, 0, 0, 0, 0, time.UTC)
	}
	at = at.UTC()
	from = time.Date(at.Year(), at.Month(), 1, 0, 0, 0, 0, time.UTC)
	return from, from.AddDate(0, 1, 0)
}

// openLedgerForReading opens the usage ledger the config names for a reader. A SQLite ledger
// that does not exist is an error and is not created: reconcile only reads.
func openLedgerForReading(ctx context.Context, cfg *config.Config) (accounting.Store, error) {
	if cfg.Accounting.Driver == config.AccountingSQLite {
		path := config.ResolvePath(cfg.BaseDir(), cfg.Accounting.SQLitePath)
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("the usage ledger %s cannot be read: %w", path, err)
		}
	}
	return openAccounting(ctx, cfg, os.Getenv)
}
