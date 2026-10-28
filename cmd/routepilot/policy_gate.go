package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/omorsi45/routepilot/internal/policy"
	"github.com/omorsi45/routepilot/internal/replay"
	"github.com/omorsi45/routepilot/internal/router"
)

const policyGateUsage = "usage: routepilot policy gate --config FILE --input FILE [--split calib]\n" +
	"                             --incumbent DIR --incumbent-config FILE\n" +
	"                             --candidate DIR --candidate-config FILE\n" +
	"                             [--epsilon 0.1] [--seed 0] [--python PATH] --out DIR\n" +
	"\n" +
	"gate replays the split three times with the real router: the incumbent with exploration\n" +
	"(the logging policy, with its per-model scores), the candidate and the incumbent, both\n" +
	"deterministic. The three files and the verdict (gate.json) are written to --out. The\n" +
	"verdict comes from the off-policy estimators in python/routepilot_train/ope.py, run with\n" +
	"--python (default .venv/Scripts/python.exe when it exists, else .venv/bin/python).\n" +
	"exit codes: 0 the candidate passes, 1 it fails the gate, 2 usage or runtime error\n"

// The files the gate writes under --out, besides gate.json.
const (
	gateLoggedFile    = "logged.jsonl"
	gateCandidateFile = "candidate.jsonl"
	gateIncumbentFile = "incumbent.jsonl"
)

// runGateCommand runs the Python side of the gate and returns its exit code. An error means
// the command could not be started or waited for. Tests replace it.
var runGateCommand = func(name string, args []string, stdout, stderr io.Writer) (int, error) {
	cmd := exec.Command(name, args...)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	var exit *exec.ExitError
	switch {
	case err == nil:
		return 0, nil
	case errors.As(err, &exit):
		return exit.ExitCode(), nil
	}
	return -1, err
}

// defaultPython is the interpreter of the repository's virtual environment under root:
// Scripts/python.exe on Windows, bin/python elsewhere (the Makefile's rule for $(PY)).
func defaultPython(root string) string {
	windows := filepath.Join(root, ".venv", "Scripts", "python.exe")
	if _, err := os.Stat(windows); err == nil {
		return windows
	}
	return filepath.Join(root, ".venv", "bin", "python")
}

// runPolicyGate implements `routepilot policy gate`. It returns 0 when the candidate passes,
// 1 when it fails the gate, and 2 for usage errors and for anything that stopped the gate
// from reaching a verdict.
func runPolicyGate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("routepilot policy gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, policyGateUsage) }
	configPath := fs.String("config", "", "RoutePilot config for the replay (required)")
	inputPath := fs.String("input", "", "replay JSONL input (required)")
	split := fs.String("split", "calib", "split to replay")
	incumbent := fs.String("incumbent", "", "incumbent policy artifact directory (required)")
	incumbentCfg := fs.String("incumbent-config", "", "incumbent policy config YAML (required)")
	candidate := fs.String("candidate", "", "candidate policy artifact directory (required)")
	candidateCfg := fs.String("candidate-config", "", "candidate policy config YAML (required)")
	epsilon := fs.Float64("epsilon", 0.1, "exploration rate of the logging policy, in [0, 1)")
	seed := fs.Uint64("seed", 0, "seed of the logging policy's exploration")
	python := fs.String("python", defaultPython("."), "Python interpreter that runs the estimators")
	outDir := fs.String("out", "", "output directory (required)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	var usageErr error
	switch {
	case fs.NArg() > 0:
		usageErr = fmt.Errorf("unexpected argument %q", fs.Arg(0))
	case *configPath == "" || *inputPath == "" || *outDir == "":
		usageErr = errors.New("--config, --input and --out are required")
	case *incumbent == "" || *incumbentCfg == "" || *candidate == "" || *candidateCfg == "":
		usageErr = errors.New("--incumbent, --incumbent-config, --candidate and --candidate-config are required")
	}
	if usageErr != nil {
		fmt.Fprintf(stderr, "routepilot policy gate: %v\n%s", usageErr, policyGateUsage)
		return 2
	}

	// The logging policy explores; the two policies under test never do. Every policy is
	// built, and so every override checked, before any replay runs.
	replays := []struct {
		name, artifact, policyConfig, file string
		epsilon                            float64
		emitCandidates                     bool
		learned                            router.Learned
	}{
		{name: "logged", artifact: *incumbent, policyConfig: *incumbentCfg, file: gateLoggedFile, epsilon: *epsilon, emitCandidates: true},
		{name: "candidate", artifact: *candidate, policyConfig: *candidateCfg, file: gateCandidateFile},
		{name: "incumbent", artifact: *incumbent, policyConfig: *incumbentCfg, file: gateIncumbentFile},
	}
	for i := range replays {
		r := &replays[i]
		learned, err := buildLearned(r.artifact, r.policyConfig, nil, func(c *policy.Config) { c.Epsilon = r.epsilon })
		if err != nil {
			fmt.Fprintf(stderr, "routepilot policy gate: %s: %v\n", r.name, err)
			if errors.Is(err, errBadOverride) {
				fmt.Fprint(stderr, policyGateUsage)
			}
			return 2
		}
		r.learned = learned
	}
	if err := os.MkdirAll(*outDir, 0o750); err != nil {
		fmt.Fprintf(stderr, "routepilot policy gate: %v\n", err)
		return 2
	}
	for _, r := range replays {
		sum, err := replayFiles(*configPath, *inputPath, filepath.Join(*outDir, r.file), replay.Options{
			Split: *split, Policy: replay.PolicyLearned, Seed: *seed,
			Learned: r.learned, EmitCandidates: r.emitCandidates,
		})
		if err != nil {
			fmt.Fprintf(stderr, "routepilot policy gate: replay %s: %v\n", r.name, err)
			return 2
		}
		fmt.Fprintf(stdout, "replayed %s: items=%d errors=%d\n", r.name, sum.Items, sum.Errors)
	}

	code, err := runGateCommand(*python, []string{
		"-m", "routepilot_train.ope", "gate",
		"--logged", filepath.Join(*outDir, gateLoggedFile),
		"--candidate", filepath.Join(*outDir, gateCandidateFile),
		"--incumbent", filepath.Join(*outDir, gateIncumbentFile),
		"--out", *outDir,
	}, stdout, stderr)
	switch {
	case err != nil:
		fmt.Fprintf(stderr, "routepilot policy gate: run %s: %v\n", *python, err)
		return 2
	case code == 0 || code == 1:
		return code
	}
	fmt.Fprintf(stderr, "routepilot policy gate: the estimators exited with %d\n", code)
	return 2
}
