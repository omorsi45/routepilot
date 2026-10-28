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

// options are what the command line asks for.
type options struct {
	listen string
	config mock.Config
}

// parseFlags reads the command line. It returns the options and -1 when the server should
// start, or the process exit code (0 for --help, 2 for a usage error) when it should not.
func parseFlags(args []string, stderr io.Writer) (options, int) {
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
		rate429   = fs.Float64("fail-rate-http429", 0, "probability of an injected HTTP 429 with Retry-After 1 second")
		rateReset = fs.Float64("fail-rate-reset", 0, "probability of an injected connection reset")
	)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return options{}, 0
		}
		return options{}, 2
	}

	if *rate500 < 0 || *rate429 < 0 || *rateReset < 0 || *rate500+*rate429+*rateReset > 1 {
		fmt.Fprintln(stderr, "rp-mock: fail rates must be non-negative and sum to at most 1")
		return options{}, 2
	}
	failRates := map[string]float64{}
	if *rate500 > 0 {
		failRates["http500"] = *rate500
	}
	if *rate429 > 0 {
		failRates["http429"] = *rate429
	}
	if *rateReset > 0 {
		failRates["reset"] = *rateReset
	}

	return options{
		listen: *listen,
		config: mock.Config{
			Seed:         *seed,
			TTFT:         mock.Dist{MedianMs: *ttftMed, Sigma: *ttftSigma},
			TPOT:         mock.Dist{MedianMs: *tpotMed, Sigma: *tpotSigma},
			OutputTokens: *tokens,
			FailRates:    failRates,
		},
	}, -1
}

func run(args []string, stderr io.Writer) int {
	opts, code := parseFlags(args, stderr)
	if code >= 0 {
		return code
	}

	handler := mock.NewHandler(opts.config)
	srv := &http.Server{Addr: opts.listen, Handler: handler, ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ln, err := net.Listen("tcp", opts.listen)
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
