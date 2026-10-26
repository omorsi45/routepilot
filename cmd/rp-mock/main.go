// Command rp-mock runs the deterministic OpenAI-compatible mock provider used
// for RoutePilot gateway tests and load tests.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/omorsi45/routepilot/internal/mock"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stderr))
}

func run(args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("rp-mock", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		listen    = fs.String("listen", "127.0.0.1:9090", "address to listen on; the API is served under /v1")
		seed      = fs.Int64("seed", 1, "seed for latency samples and failure decisions")
		ttftMed   = fs.Float64("ttft-median-ms", 0, "median time to first token in milliseconds")
		ttftSigma = fs.Float64("ttft-sigma", 0, "log-normal shape of the time to first token; 0 means constant")
		tpotMed   = fs.Float64("tpot-median-ms", 0, "median time per output token in milliseconds")
		tpotSigma = fs.Float64("tpot-sigma", 0, "log-normal shape of the time per output token; 0 means constant")
		tokens    = fs.Int("tokens", 16, "default completion length in tokens")
		rate500   = fs.Float64("fail-rate-http500", 0, "probability of an injected HTTP 500")
		rateReset = fs.Float64("fail-rate-reset", 0, "probability of an injected connection reset")
	)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if *rate500 < 0 || *rateReset < 0 || *rate500+*rateReset > 1 {
		fmt.Fprintln(stderr, "rp-mock: fail rates must be non-negative and sum to at most 1")
		return 2
	}
	failRates := map[string]float64{}
	if *rate500 > 0 {
		failRates["http500"] = *rate500
	}
	if *rateReset > 0 {
		failRates["reset"] = *rateReset
	}

	handler := mock.NewHandler(mock.Config{
		Seed:         *seed,
		TTFT:         mock.Dist{MedianMs: *ttftMed, Sigma: *ttftSigma},
		TPOT:         mock.Dist{MedianMs: *tpotMed, Sigma: *tpotSigma},
		OutputTokens: *tokens,
		FailRates:    failRates,
	})
	srv := &http.Server{Addr: *listen, Handler: handler, ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		fmt.Fprintf(stderr, "rp-mock: %v\n", err)
		return 1
	}
	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()
	fmt.Fprintf(stderr, "rp-mock listening on %s\n", ln.Addr())

	select {
	case err := <-errc:
		fmt.Fprintf(stderr, "rp-mock: %v\n", err)
		return 1
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		// Stalled requests never finish on their own; force them closed.
		_ = srv.Close()
	}
	return 0
}
