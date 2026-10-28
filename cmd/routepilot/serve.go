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

	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/gateway"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/provider/factory"
	"github.com/omorsi45/routepilot/internal/quota"
)

const (
	// redisKeyPrefix namespaces every key the Redis limiters write.
	redisKeyPrefix = "routepilot"
	// redisPingTimeout bounds the startup connectivity check.
	redisPingTimeout = 5 * time.Second
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
	}
	opts := gateway.Options{Logger: logger, Limiters: limiters}
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

	// Reload the learned policies on SIGHUP and when their files change.
	if reg != nil {
		hup, stopHup := notifyReload()
		defer stopHup()
		bg.Add(2)
		go func() {
			defer bg.Done()
			reg.Watch(bgCtx, policyPollEvery)
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
		_ = srv.Close()
		return fmt.Errorf("shutdown: %w", err)
	}
	if err := <-errc; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
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
	reg, err := policy.NewRegistry(load, cfg.BaseDir())
	if err != nil {
		return nil, err
	}
	if path := cfg.Path(); path != "" {
		reg.SetConfigFile(path)
	}
	reg.SetLogger(logger)
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
