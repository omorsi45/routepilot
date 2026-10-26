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
	"syscall"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/gateway"
	"github.com/omorsi45/routepilot/internal/provider/factory"
)

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

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
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
	gw, err := gateway.New(cfg, providers, gateway.Options{Logger: logger})
	if err != nil {
		return err
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
