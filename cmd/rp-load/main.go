// Command rp-load is an open-loop, coordinated-omission-safe load generator for
// OpenAI-compatible chat completion endpoints. It sends requests at Poisson-spaced times,
// measures each latency from its scheduled time and prints one JSON summary.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/HdrHistogram/hdrhistogram-go"

	"github.com/omorsi45/routepilot/internal/loadgen"
)

const defaultKeyEnv = "RP_LOAD_API_KEY"

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, os.Getenv))
}

// percentiles summarizes a histogram in milliseconds.
type percentiles struct {
	Count int64   `json:"count"`
	Min   float64 `json:"min"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P90   float64 `json:"p90"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
	P999  float64 `json:"p99.9"`
	Max   float64 `json:"max"`
}

// output is the JSON result. Latency, TTFT and LaunchLag are null when they hold no sample.
type output struct {
	URL           string         `json:"url"`
	Model         string         `json:"model"`
	Stream        bool           `json:"stream"`
	RPSTarget     float64        `json:"rps_target"`
	WarmupS       float64        `json:"warmup_s"`
	DurationS     float64        `json:"duration_s"`
	Seed          int64          `json:"seed"`
	MaxInflight   int            `json:"max_inflight"`
	MaxTokens     int            `json:"max_tokens"`
	Scheduled     int            `json:"scheduled"`
	Completed     int            `json:"completed"`
	Errors        int            `json:"errors"`
	Dropped       int            `json:"dropped"`
	ErrorsByKind  map[string]int `json:"errors_by_kind"`
	ElapsedS      float64        `json:"elapsed_s"`
	OfferedRPS    float64        `json:"offered_rps"`
	ThroughputRPS float64        `json:"throughput_rps"`
	LatencyMS     *percentiles   `json:"latency_ms"`
	TTFTMS        *percentiles   `json:"ttft_ms"`
	LaunchLagMS   *percentiles   `json:"launch_lag_ms"`
}

func run(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	fs := flag.NewFlagSet("rp-load", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		url         = fs.String("url", "", "full chat completions URL, for example http://127.0.0.1:8080/v1/chat/completions (required)")
		keyEnv      = fs.String("api-key-env", defaultKeyEnv, "name of the environment variable that holds the API key; the key is never printed")
		model       = fs.String("model", "", "model field of every request (required)")
		stream      = fs.Bool("stream", false, "request a server-sent event stream and record the time to first token")
		rps         = fs.Float64("rps", 0, "mean arrival rate in requests per second (required)")
		warmup      = fs.Duration("warmup", 0, "time at the start whose requests are sent but not recorded")
		duration    = fs.Duration("duration", 60*time.Second, "length of the measurement window")
		seed        = fs.Int64("seed", 1, "seed of the arrival schedule")
		maxInflight = fs.Int("max-inflight", 4096, "most requests in flight; further arrivals are dropped and counted")
		maxTokens   = fs.Int("max-tokens", 16, "max_tokens of every request; 0 omits it")
	)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "rp-load: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	cfg := loadgen.Config{
		URL: *url, APIKey: getenv(*keyEnv), Model: *model, Stream: *stream, RPS: *rps,
		Warmup: *warmup, Duration: *duration, Seed: *seed, MaxInflight: *maxInflight, MaxTokens: *maxTokens,
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(stderr, "rp-load: %v\n", err)
		return 2
	}
	if cfg.APIKey == "" {
		fmt.Fprintf(stderr, "rp-load: %s is empty, sending no Authorization header\n", *keyEnv)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Callers that scrape a server at the end of the warmup read this line: the first
	// arrival is scheduled within microseconds of it.
	fmt.Fprintln(stderr, "rp-load: started")
	res, err := loadgen.Run(ctx, loadgen.NewClient(cfg.MaxInflight), cfg)
	if err != nil {
		fmt.Fprintf(stderr, "rp-load: %v\n", err)
		return 1
	}

	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(summarize(cfg, res)); err != nil {
		fmt.Fprintf(stderr, "rp-load: write the result: %v\n", err)
		return 1
	}
	return 0
}

func summarize(cfg loadgen.Config, res loadgen.Result) output {
	out := output{
		URL: cfg.URL, Model: cfg.Model, Stream: cfg.Stream, RPSTarget: cfg.RPS,
		WarmupS: cfg.Warmup.Seconds(), DurationS: cfg.Duration.Seconds(), Seed: cfg.Seed,
		MaxInflight: cfg.MaxInflight, MaxTokens: cfg.MaxTokens,
		Scheduled: res.Scheduled, Completed: res.Completed, Errors: res.Errors, Dropped: res.Dropped,
		ErrorsByKind: res.ErrorsByKind, ElapsedS: res.Elapsed.Seconds(),
		// Rates are over the measurement window the schedule covers, not over the drain.
		OfferedRPS:    float64(res.Scheduled) / cfg.Duration.Seconds(),
		ThroughputRPS: float64(res.Completed) / cfg.Duration.Seconds(),
		LatencyMS:     percentilesMS(res.Latency),
		TTFTMS:        percentilesMS(res.TTFT),
		LaunchLagMS:   percentilesMS(res.LaunchLag),
	}
	if out.ErrorsByKind == nil {
		out.ErrorsByKind = map[string]int{}
	}
	return out
}

// percentilesMS converts a histogram of microseconds, or returns nil when it is empty.
func percentilesMS(h *hdrhistogram.Histogram) *percentiles {
	if h == nil || h.TotalCount() == 0 {
		return nil
	}
	ms := func(us int64) float64 { return float64(us) / 1000 }
	return &percentiles{
		Count: h.TotalCount(), Min: ms(h.Min()), Mean: h.Mean() / 1000,
		P50: ms(h.ValueAtQuantile(50)), P90: ms(h.ValueAtQuantile(90)), P95: ms(h.ValueAtQuantile(95)),
		P99: ms(h.ValueAtQuantile(99)), P999: ms(h.ValueAtQuantile(99.9)), Max: ms(h.Max()),
	}
}
