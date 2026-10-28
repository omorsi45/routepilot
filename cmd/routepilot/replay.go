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
	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/replay"
	"github.com/omorsi45/routepilot/internal/router"
)

const replayUsage = "usage: routepilot replay --config FILE --input FILE --split SPLIT --policy NAME [--seed N] [--model ID] --out FILE\n" +
	"                        [--latency-profiles FILE [--latency-seed N] [--slo-ttft-ms MS | --slo-e2e-ms MS]]\n" +
	"                        [--artifact DIR --policy-config FILE [--mode M] [--lambda X] [--floor Q] [--epsilon E]\n" +
	"                         [--emit-candidates]]\n" +
	"                        [--mf FILE --threshold T]\n" +
	"\n" +
	"policies: cheapest, strongest, rules, random, oracle, fixed (fixed needs --model),\n" +
	"          learned (needs --artifact and --policy-config; --mode, --lambda, --floor and\n" +
	"          --epsilon override the policy config, --floor sets its default floor;\n" +
	"          --emit-candidates adds every model's scores to each record),\n" +
	"          routellm_mf (needs --mf, an exported matrix factorization scorer, and --threshold in\n" +
	"          [0, 1]: items scoring at least T go to the scorer's strong model, the rest to its weak one)\n" +
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
	policyName := fs.String("policy", "", "policy: cheapest, strongest, rules, random, oracle, fixed, learned or routellm_mf (required)")
	seed := fs.Uint64("seed", 0, "seed for the random policy")
	model := fs.String("model", "", "model the fixed policy always asks for")
	outPath := fs.String("out", "", "path of the JSONL output (required)")
	profilesPath := fs.String("latency-profiles", "", "path to the simulated latency profiles YAML")
	latencySeed := fs.Uint64("latency-seed", 0, "seed of the simulated latency draws")
	sloTTFT := fs.Int("slo-ttft-ms", 0, "time to first token SLO in milliseconds, judged on the simulated latency")
	sloE2E := fs.Int("slo-e2e-ms", 0, "end-to-end SLO in milliseconds, judged on the simulated latency")
	artifactDir := fs.String("artifact", "", "policy artifact directory (learned policy)")
	policyConfigPath := fs.String("policy-config", "", "policy config YAML (learned policy)")
	mode := fs.String("mode", "", "override the policy config mode: min_cost or dual")
	lambda := fs.Float64("lambda", 0, "override the policy config lambda, per USD of expected cost")
	floor := fs.Float64("floor", 0, "override the policy config default floor, in [0, 1]")
	epsilon := fs.Float64("epsilon", 0, "override the policy config exploration rate, in [0, 1)")
	mfPath := fs.String("mf", "", "exported matrix factorization scorer (routellm_mf policy)")
	threshold := fs.Float64("threshold", 0, "score at or above which routellm_mf picks the strong model, in [0, 1]")
	emitCandidates := fs.Bool("emit-candidates", false, "add the learned policy's per-model scores to every record (policy learned)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	given := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { given[f.Name] = true })
	learnedFlags := []string{"artifact", "policy-config", "mode", "lambda", "floor", "epsilon"}
	mfFlags := []string{"mf", "threshold"}

	opts := replay.Options{
		Split: *split, Policy: *policyName, Seed: *seed, Model: *model, Threshold: *threshold,
		LatencySeed: *latencySeed, SLOTTFTMs: *sloTTFT, SLOE2EMs: *sloE2E, EmitCandidates: *emitCandidates,
	}
	var usageErr error
	switch {
	case fs.NArg() > 0:
		usageErr = fmt.Errorf("unexpected argument %q", fs.Arg(0))
	case *configPath == "" || *inputPath == "" || *outPath == "":
		usageErr = errors.New("--config, --input and --out are required")
	case (*sloTTFT > 0 || *sloE2E > 0) && *profilesPath == "":
		usageErr = errors.New("--slo-ttft-ms and --slo-e2e-ms need --latency-profiles")
	case *policyName == replay.PolicyLearned && (*artifactDir == "" || *policyConfigPath == ""):
		usageErr = errors.New("policy learned needs --artifact and --policy-config")
	case *policyName != replay.PolicyLearned && anyGiven(given, learnedFlags):
		usageErr = errors.New("--artifact, --policy-config, --mode, --lambda, --floor and --epsilon are only valid with --policy learned")
	case *policyName == replay.PolicyRouteLLMMF && (*mfPath == "" || !given["threshold"]):
		usageErr = errors.New("policy routellm_mf needs --mf and --threshold")
	case *policyName != replay.PolicyRouteLLMMF && anyGiven(given, mfFlags):
		usageErr = errors.New("--mf and --threshold are only valid with --policy routellm_mf")
	case *emitCandidates && *policyName != replay.PolicyLearned:
		usageErr = errors.New("--emit-candidates is only valid with --policy learned")
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

	if *policyName == replay.PolicyLearned {
		learned, err := buildLearned(*artifactDir, *policyConfigPath, opts.Profiles, func(c *policy.Config) {
			if given["mode"] {
				c.Mode = *mode
			}
			if given["lambda"] {
				c.Lambda = *lambda
			}
			if given["floor"] {
				c.DefaultFloor = *floor
			}
			if given["epsilon"] {
				c.Epsilon = *epsilon
			}
		})
		if err != nil {
			fmt.Fprintf(stderr, "routepilot replay: %v\n", err)
			if errors.Is(err, errBadOverride) {
				fmt.Fprint(stderr, replayUsage)
				return 2
			}
			return 1
		}
		opts.Learned = learned
	}

	if *policyName == replay.PolicyRouteLLMMF {
		mf, err := replay.LoadMF(*mfPath)
		if err != nil {
			fmt.Fprintf(stderr, "routepilot replay: %v\n", err)
			return 1
		}
		opts.MF = mf
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

func anyGiven(given map[string]bool, names []string) bool {
	for _, n := range names {
		if given[n] {
			return true
		}
	}
	return false
}

// errBadOverride marks a policy config that is invalid after the command line overrides,
// which is the caller's mistake and not a bad file.
var errBadOverride = errors.New("invalid override")

// buildLearned loads the artifact and the policy config, applies the overrides and
// returns the constrained optimizer. With latency profiles the optimizer gets a prior-only
// estimator, so a latency SLO is judged on the same simulated profiles the replay draws
// the latencies from.
func buildLearned(artifactDir, policyConfigPath string, profiles map[string]latency.Profile, override func(*policy.Config)) (router.Learned, error) {
	art, err := policy.Load(artifactDir)
	if err != nil {
		return nil, fmt.Errorf("load artifact: %w", err)
	}
	cfg, err := policy.LoadConfig(policyConfigPath)
	if err != nil {
		return nil, err
	}
	override(&cfg)
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", errBadOverride, err)
	}
	var est *latency.Estimator
	if len(profiles) > 0 {
		est = latency.NewEstimator(profiles, nil)
	}
	return router.NewOptimizer(art, cfg, est), nil
}
