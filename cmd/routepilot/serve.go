package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/decisionlog"
	"github.com/omorsi45/routepilot/internal/gateway"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider/factory"
	"github.com/omorsi45/routepilot/internal/quota"
	"github.com/omorsi45/routepilot/internal/telemetry"
)

const (
	// redisKeyPrefix namespaces every key the Redis limiters write.
	redisKeyPrefix = "routepilot"
	// redisPingTimeout bounds the startup connectivity check.
	redisPingTimeout = 5 * time.Second
	// accountingOpenTimeout bounds connecting to the usage ledger and creating its tables.
	accountingOpenTimeout = 15 * time.Second
	// usageCloseTimeout bounds writing the usage rows still queued at shutdown.
	usageCloseTimeout = 15 * time.Second
	// tracingShutdownTimeout bounds the final flush of the trace exporter at shutdown.
	tracingShutdownTimeout = 5 * time.Second
	// forcedDrainTimeout bounds the wait for the requests still running when the shutdown
	// grace period ended and their connections were cut. A request that is cut finishes in
	// milliseconds: its budget commit is bounded by two seconds.
	forcedDrainTimeout = 5 * time.Second
)

// notifyContext is signal.NotifyContext; tests replace it to inject signals.
var notifyContext = signal.NotifyContext

// notifyReload registers for the signal that reloads the learned policies (SIGHUP) and
// returns the channel it arrives on and a function that unregisters. Tests replace it to
// inject signals. On Windows the signal is never delivered, and the poll covers it.
var notifyReload = func() (<-chan os.Signal, func()) {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGHUP)
	return ch, func() { signal.Stop(ch) }
}

// policyPollEvery is how often serve checks the learned policy files for changes. Tests
// replace it.
var policyPollEvery = 5 * time.Second

// budgetSweepEvery is how often serve drops the expired leases of the budget ledger. Tests
// replace it.
var budgetSweepEvery = 30 * time.Second

// guardEvery is how often serve evaluates the guardrails of the learned policy's canary
// (every 10 seconds). Tests replace it.
var guardEvery = 10 * time.Second

const serveUsage = "usage: routepilot serve -config <path>\n"

// runServe implements `routepilot serve`. It returns 0 after a clean
// shutdown, 1 for startup errors and 2 for usage errors.
func runServe(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("routepilot serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "path to the YAML config file (required)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *configPath == "" || fs.NArg() > 0 {
		fmt.Fprintf(stderr, "routepilot serve: -config is required\n%s", serveUsage)
		return 2
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot serve: %v\n", err)
		return 1
	}
	ln, err := net.Listen("tcp", cfg.Server.Listen)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot serve: %v\n", err)
		return 1
	}

	ctx, stopSignals := notifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	stop := sync.OnceFunc(stopSignals)
	defer stop()
	// After the first signal, restore the default handling so a second
	// SIGINT or SIGTERM ends the process instead of being swallowed while the
	// graceful shutdown runs.
	finished := make(chan struct{})
	defer close(finished)
	go func() {
		select {
		case <-ctx.Done():
			stop()
		case <-finished:
		}
	}()
	logger := slog.New(slog.NewJSONHandler(stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if err := serve(ctx, cfg, ln, logger); err != nil {
		fmt.Fprintf(stderr, "routepilot serve: %v\n", err)
		return 1
	}
	return 0
}

// serve builds the providers and gateway from cfg and serves on ln until ctx
// is done, then shuts down gracefully within cfg.Server.ShutdownGraceMs. It
// closes ln. A clean shutdown returns nil.
func serve(ctx context.Context, cfg *config.Config, ln net.Listener, logger *slog.Logger) error {
	defer ln.Close()

	providers, err := factory.Build(cfg.Providers, os.Getenv, nil)
	if err != nil {
		return err
	}
	limiters, closeLimiters, err := buildLimiters(ctx, cfg, os.Getenv)
	if err != nil {
		return err
	}
	defer closeLimiters()
	if limiters.Budget == nil { // the memory backend: serve owns the ledger so it can sweep it
		limiters.Budget = quota.NewMemoryBudget()
		warnMemoryBudgets(cfg, logger)
	}
	opts := gateway.Options{Logger: logger, Limiters: limiters, Registry: newMetricsRegistry()}
	tp, shutdownTracing, err := telemetry.NewTracerProvider(ctx, cfg.Tracing)
	if err != nil {
		return fmt.Errorf("tracing: %w", err)
	}
	// Flushed when serve returns, which is after the HTTP server has drained, so the spans
	// of the last requests are sent. A collector that is down costs at most the timeout.
	defer func() {
		flush, cancel := context.WithTimeout(context.Background(), tracingShutdownTimeout)
		defer cancel()
		if err := shutdownTracing(flush); err != nil {
			logger.Warn("flushing the traces failed", slog.String("error", err.Error()))
		}
	}()
	opts.TracerProvider = tp
	store, err := openAccounting(ctx, cfg, os.Getenv)
	if err != nil {
		return err
	}
	if store != nil {
		// Closed when serve returns, after the HTTP server has drained and the gateway has
		// written the rows it still queued (closeUsageQueue), so the last requests keep theirs.
		defer func() {
			if err := store.Close(); err != nil {
				logger.Warn("closing the usage ledger failed", slog.String("error", err.Error()))
			}
		}()
		opts.Accounting = store
	}
	if cfg.DecisionLog.Path != "" {
		dl, err := decisionlog.Open(config.ResolvePath(cfg.BaseDir(), cfg.DecisionLog.Path), cfg.DecisionLog.MaxBytes)
		if err != nil {
			return fmt.Errorf("decision log: %w", err)
		}
		// Closed when serve returns, after the HTTP server has drained, so the records of the
		// last requests are written before the process exits.
		defer closeDecisionLog(dl, logger)
		opts.DecisionLog = dl
	}
	opts.Latency, err = buildEstimator(cfg)
	if err != nil {
		return err
	}
	var reg *policy.Registry
	if cfg.Routing.Policy == config.PolicyLearned {
		reg, err = buildRegistry(cfg, logger)
		if err != nil {
			return err
		}
		opts.Policies = reg
	}
	gw, err := gateway.New(cfg, providers, opts)
	if err != nil {
		return err
	}
	// Deferred after the store's close, so it runs before it: when serve returns the HTTP
	// server has drained, so no request adds rows any more, and the gateway writes the rows
	// still queued before the store goes away.
	defer closeUsageQueue(gw, logger)

	// Background work stops with the server, and serve waits for it so nothing logs after
	// it returns.
	bgCtx, stopBG := context.WithCancel(ctx)
	var bg sync.WaitGroup
	defer bg.Wait()
	defer stopBG()

	// Drop the budget leases of requests that never ended, but only when some tenant has a
	// budget: the Redis sweep scans the keyspace.
	if hasBudget(cfg) {
		bg.Add(1)
		go func() {
			defer bg.Done()
			sweepBudgets(bgCtx, limiters.Budget, budgetSweepEvery, logger)
		}()
	}

	// Reload the learned policies on SIGHUP and when their files change, and judge the canary
	// against the stable policy every guardEvery.
	if reg != nil {
		hup, stopHup := notifyReload()
		defer stopHup()
		bg.Add(3)
		go func() {
			defer bg.Done()
			reg.Watch(bgCtx, policyPollEvery)
		}()
		go func() {
			defer bg.Done()
			evaluateGuardrails(bgCtx, gw, guardEvery)
		}()
		go func() {
			defer bg.Done()
			for {
				select {
				case <-bgCtx.Done():
					return
				case <-hup:
					_ = reg.Reload() // the registry logs the outcome
				}
			}
		}()
	}

	// No ReadTimeout or WriteTimeout: streams are long-lived, and the stream
	// path sets its own per-write deadlines.
	srv := &http.Server{
		Handler:           gw.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()
	logger.Info("listening", slog.String("addr", ln.Addr().String()))

	select {
	case err := <-errc:
		return fmt.Errorf("serve: %w", err)
	case <-ctx.Done():
	}

	grace := time.Duration(cfg.Server.ShutdownGraceMs) * time.Millisecond
	shutdownCtx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		// The grace period ended with requests still open. Close cuts their connections but
		// does not wait for their handlers, which are still unwinding: each one commits its
		// budget and queues its usage rows and decision record on the way out. Wait for them,
		// within forcedDrainTimeout, before the deferred closes run, or those rows are refused
		// and the commits race the closing of the quota backend.
		_ = srv.Close()
		drainCtx, cancelDrain := context.WithTimeout(context.Background(), forcedDrainTimeout)
		defer cancelDrain()
		if derr := gw.WaitIdle(drainCtx); derr != nil {
			logger.Warn("requests were still running when the server closed", slog.String("error", derr.Error()))
		}
		return fmt.Errorf("shutdown: %w", err)
	}
	if err := <-errc; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}

// newMetricsRegistry is the registry GET /metrics serves: the gateway adds its own series, and
// the Go runtime and process collectors are added here, for the real server only. Each server
// gets a registry of its own, so tests can start many.
func newMetricsRegistry() *prometheus.Registry {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	return reg
}

// warnMemoryBudgets logs one warning for each tenant of cfg that has a budget while the quota
// backend is memory: that counter starts at zero with every start and is not shared with a
// second process, so a restart (or a replica) allows the whole budget again. A budget that
// is durable and shared needs quota.backend redis.
func warnMemoryBudgets(cfg *config.Config, logger *slog.Logger) {
	for _, t := range cfg.Tenants {
		if t.BudgetUSD != "" {
			logger.Warn("the budget counter is in this process's memory: it is per process and resets on restart, so the tenant can spend its whole budget again; set quota.backend to redis for a durable budget shared by every replica",
				slog.String("tenant", t.ID), slog.String("budget_usd", t.BudgetUSD))
		}
	}
}

// hasBudget reports whether any tenant of cfg has a budget.
func hasBudget(cfg *config.Config) bool {
	for _, t := range cfg.Tenants {
		if t.BudgetUSD != "" {
			return true
		}
	}
	return false
}

// sweepBudgets drops the expired leases of l every interval until ctx is done. A lease
// expires at the deadline of its request plus a minute, so one left behind by a request
// that never ended stops holding budget even before the sweep; the sweep keeps them from
// piling up.
func sweepBudgets(ctx context.Context, l quota.BudgetLedger, every time.Duration, logger *slog.Logger) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			n, err := l.Sweep(ctx, time.Now())
			switch {
			case err != nil && ctx.Err() == nil:
				logger.Warn("budget sweep failed", slog.String("error", err.Error()))
			case n > 0:
				logger.Info("budget leases expired", slog.Int("count", n))
			}
		}
	}
}

// evaluateGuardrails runs the gateway's guardrail evaluation every interval until ctx ends.
func evaluateGuardrails(ctx context.Context, gw *gateway.Server, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			gw.EvaluateGuardrails() // the gateway logs and records a rollback
		}
	}
}

// buildEstimator returns the latency estimator the gateway learns from. Its priors come
// from server.latency_profiles when that is set (a path relative to the config file); with
// none, it knows only what the server observes.
func buildEstimator(cfg *config.Config) (*latency.Estimator, error) {
	var priors map[string]latency.Profile
	if p := cfg.Server.LatencyProfiles; p != "" {
		var err error
		priors, err = latency.LoadProfiles(config.ResolvePath(cfg.BaseDir(), p))
		if err != nil {
			return nil, err
		}
	}
	return latency.NewEstimator(priors, nil), nil
}

// buildRegistry loads the learned policies named by cfg. When cfg came from a file, a
// reload re-reads that file but uses only its policies section; the other sections need a
// restart, and a change to any of them is logged once per reload as ignored.
func buildRegistry(cfg *config.Config, logger *slog.Logger) (*policy.Registry, error) {
	load := func() (config.PoliciesConfig, error) { return cfg.Policies, nil }
	if path := cfg.Path(); path != "" {
		load = func() (config.PoliciesConfig, error) {
			fresh, err := config.Load(path)
			if err != nil {
				return config.PoliciesConfig{}, err
			}
			if otherSectionsDiffer(cfg, fresh) {
				logger.Warn("config sections other than policies changed and are ignored until restart",
					slog.String("config", path))
			}
			return fresh.Policies, nil
		}
	}
	catalog := make([]string, len(cfg.Models))
	for i, m := range cfg.Models {
		catalog[i] = m.ID
	}
	reg, err := policy.NewRegistry(load, cfg.BaseDir(), policy.WithCatalog(catalog), policy.WithLogger(logger))
	if err != nil {
		return nil, err
	}
	if path := cfg.Path(); path != "" {
		reg.SetConfigFile(path)
	}
	snap := reg.Snapshot()
	attrs := []slog.Attr{slog.String("stable", snap.Stable.Version), slog.Int("shadow", len(snap.Shadow))}
	if snap.Canary != nil {
		attrs = append(attrs, slog.String("canary", snap.Canary.Version), slog.Float64("canary_percent", snap.CanaryPercent))
	}
	logger.LogAttrs(context.Background(), slog.LevelInfo, "policies loaded", attrs...)
	return reg, nil
}

// otherSectionsDiffer reports whether the reloaded config differs from the running one in
// anything but its policies section.
func otherSectionsDiffer(running, reloaded *config.Config) bool {
	a, b := *running, *reloaded
	a.Policies, b.Policies = config.PoliciesConfig{}, config.PoliciesConfig{}
	return !reflect.DeepEqual(a, b)
}

// closeUsageQueue writes the usage rows the gateway still holds in memory, within
// usageCloseTimeout, and logs when it gave up on some. Without a usage ledger it does nothing.
func closeUsageQueue(gw *gateway.Server, logger *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), usageCloseTimeout)
	defer cancel()
	if err := gw.CloseUsage(ctx); err != nil {
		logger.Warn("writing the queued usage rows failed", slog.String("error", err.Error()))
	}
}

// closeDecisionLog drains and closes the decision log, and says so when records were lost.
func closeDecisionLog(dl *decisionlog.Writer, logger *slog.Logger) {
	if err := dl.Close(); err != nil {
		logger.Warn("closing the decision log failed", slog.String("error", err.Error()))
	}
	if dropped, lost := dl.Dropped(), dl.Errors(); dropped > 0 || lost > 0 {
		logger.Warn("decision log records were lost", slog.Int64("dropped", dropped), slog.Int64("errors", lost))
	}
}

// openAccounting opens the usage ledger that cfg.Accounting names. It returns a nil Store for
// the none driver. A SQLite path is relative to the config file's directory; a PostgreSQL
// DSN is read from the environment variable the config names and never appears in an error.
func openAccounting(ctx context.Context, cfg *config.Config, getenv func(string) string) (accounting.Store, error) {
	a := cfg.Accounting
	switch a.Driver {
	case config.AccountingSQLite:
		return accounting.OpenSQLite(config.ResolvePath(cfg.BaseDir(), a.SQLitePath))
	case config.AccountingPostgres:
		dsn := getenv(a.PostgresDSNEnv)
		if dsn == "" {
			return nil, fmt.Errorf("accounting: environment variable %s is empty or unset", a.PostgresDSNEnv)
		}
		ctx, cancel := context.WithTimeout(ctx, accountingOpenTimeout)
		defer cancel()
		return accounting.OpenPostgres(ctx, dsn)
	}
	return nil, nil
}

// buildLimiters returns the quota limiters selected by cfg.Quota and a func
// that releases their resources. The memory backend is the zero Limiters (the
// gateway fills it in). The Redis backend reads its URL from the environment
// variable named by quota.redis_url_env and pings the server, so a bad URL or
// an unreachable Redis fails startup instead of the first request.
func buildLimiters(ctx context.Context, cfg *config.Config, getenv func(string) string) (quota.Limiters, func(), error) {
	if cfg.Quota.Backend != config.QuotaRedis {
		return quota.Limiters{}, func() {}, nil
	}
	envName := cfg.Quota.RedisURLEnv
	raw := getenv(envName)
	if raw == "" {
		return quota.Limiters{}, nil, fmt.Errorf("quota backend redis: environment variable %s is empty or unset", envName)
	}
	opts, err := redis.ParseURL(raw)
	if err != nil {
		// The parse error can quote the URL, which may hold a password.
		return quota.Limiters{}, nil, fmt.Errorf("quota backend redis: %s is not a valid Redis URL", envName)
	}
	client := redis.NewClient(opts)
	pingCtx, cancel := context.WithTimeout(ctx, redisPingTimeout)
	defer cancel()
	if err := client.Ping(pingCtx).Err(); err != nil {
		_ = client.Close()
		return quota.Limiters{}, nil, fmt.Errorf("quota backend redis: ping %s: %w", opts.Addr, err)
	}
	return quota.NewRedis(client, redisKeyPrefix), func() { _ = client.Close() }, nil
}
