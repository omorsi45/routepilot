package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/latency"
	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/replay"
)

const replayUsage = "usage: routepilot replay --config FILE --input FILE --split SPLIT --policy NAME [--seed N] [--model ID] --out FILE\n" +
	"                        [--latency-profiles FILE [--latency-seed N] [--slo-ttft-ms MS | --slo-e2e-ms MS]]\n" +
	"\n" +
	"policies: cheapest, strongest, rules, random, oracle, fixed (fixed needs --model)\n" +
	"split: train, calib, test, heldout or all\n" +
	"latency: with --latency-profiles every routed item gets a simulated latency drawn from the\n" +
	"model's profile; an SLO marks the items whose simulated latency exceeds it\n"

// runReplay implements `routepilot replay`. It returns 0 on success, 1 for
// input errors and 2 for usage errors. Routing errors are recorded in the
// output, not returned.
func runReplay(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("routepilot replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, replayUsage) }
	configPath := fs.String("config", "", "path to the YAML config file (required)")
	inputPath := fs.String("input", "", "path to the replay JSONL input (required)")
	split := fs.String("split", "", "split to replay: train, calib, test, heldout or all (required)")
	policy := fs.String("policy", "", "policy: cheapest, strongest, rules, random, oracle or fixed (required)")
	seed := fs.Uint64("seed", 0, "seed for the random policy")
	model := fs.String("model", "", "model the fixed policy always asks for")
	outPath := fs.String("out", "", "path of the JSONL output (required)")
	profilesPath := fs.String("latency-profiles", "", "path to the simulated latency profiles YAML")
	latencySeed := fs.Uint64("latency-seed", 0, "seed of the simulated latency draws")
	sloTTFT := fs.Int("slo-ttft-ms", 0, "time to first token SLO in milliseconds, judged on the simulated latency")
	sloE2E := fs.Int("slo-e2e-ms", 0, "end-to-end SLO in milliseconds, judged on the simulated latency")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	opts := replay.Options{
		Split: *split, Policy: *policy, Seed: *seed, Model: *model,
		LatencySeed: *latencySeed, SLOTTFTMs: *sloTTFT, SLOE2EMs: *sloE2E,
	}
	var usageErr error
	switch {
	case fs.NArg() > 0:
		usageErr = fmt.Errorf("unexpected argument %q", fs.Arg(0))
	case *configPath == "" || *inputPath == "" || *outPath == "":
		usageErr = errors.New("--config, --input and --out are required")
	case (*sloTTFT > 0 || *sloE2E > 0) && *profilesPath == "":
		usageErr = errors.New("--slo-ttft-ms and --slo-e2e-ms need --latency-profiles")
	default:
		usageErr = opts.Validate()
	}
	if usageErr != nil {
		fmt.Fprintf(stderr, "routepilot replay: %v\n%s", usageErr, replayUsage)
		return 2
	}

	if *profilesPath != "" {
		profiles, err := latency.LoadProfiles(*profilesPath)
		if err != nil {
			fmt.Fprintf(stderr, "routepilot replay: %v\n", err)
			return 1
		}
		opts.Profiles = profiles
	}

	sum, err := replayFiles(*configPath, *inputPath, *outPath, opts)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot replay: %v\n", err)
		if errors.Is(err, replay.ErrUsage) {
			fmt.Fprint(stderr, replayUsage)
			return 2
		}
		return 1
	}
	fmt.Fprintf(stdout, "items=%d errors=%d success_rate=%.4f mean_cost_usd=%s\n",
		sum.Items, sum.Errors, sum.SuccessRate(), money.PicoUSD(sum.MeanCostPicoUSD()).USD())
	return 0
}

// replayFiles opens the files and runs the replay. The output file is created
// only after the config and the input open, and is removed when the run fails.
func replayFiles(configPath, inputPath, outPath string, opts replay.Options) (sum replay.Summary, err error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return replay.Summary{}, err
	}
	in, err := os.Open(inputPath)
	if err != nil {
		return replay.Summary{}, fmt.Errorf("open input: %w", err)
	}
	defer in.Close()
	out, err := os.Create(outPath)
	if err != nil {
		return replay.Summary{}, fmt.Errorf("create output: %w", err)
	}
	defer func() {
		if cerr := out.Close(); err == nil && cerr != nil {
			err = fmt.Errorf("close output: %w", cerr)
		}
		if err != nil {
			_ = os.Remove(outPath)
		}
	}()
	return replay.Run(context.Background(), cfg, in, out, opts)
}
