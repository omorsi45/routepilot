package main

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/omorsi45/routepilot/internal/policy"
)

const policyUsage = "usage: routepilot policy verify DIR\n" +
	"       routepilot policy inspect DIR\n" +
	"       routepilot policy gate --config FILE --input FILE --incumbent DIR --incumbent-config FILE\n" +
	"                              --candidate DIR --candidate-config FILE --out DIR [--split calib]\n" +
	"                              [--epsilon 0.1] [--seed 0] [--python PATH]\n" +
	"\n" +
	"DIR is a policy artifact directory such as policies/flagship/<version>.\n" +
	"verify checks the content hash, spec version and golden predictions, then prints the version.\n" +
	"inspect does the same checks, then prints a JSON summary of the artifact.\n" +
	"gate decides whether a candidate policy may replace the incumbent, from replays\n" +
	"(routepilot policy gate --help).\n"

// runPolicy implements `routepilot policy`. It returns 0 on success, 1 when the artifact is
// refused or unreadable and 2 for usage errors. gate has its own exit codes.
func runPolicy(args []string, stdout, stderr io.Writer) int {
	if len(args) == 1 && (args[0] == "-h" || args[0] == "--help") {
		fmt.Fprint(stderr, policyUsage)
		return 0
	}
	if len(args) > 0 && args[0] == "gate" {
		return runPolicyGate(args[1:], stdout, stderr)
	}
	if len(args) != 2 || (args[0] != "verify" && args[0] != "inspect") {
		fmt.Fprint(stderr, policyUsage)
		return 2
	}
	command, dir := args[0], args[1]

	a, err := policy.Load(dir)
	if err != nil {
		fmt.Fprintf(stderr, "routepilot policy %s: %v\n", command, err)
		return 1
	}
	if command == "verify" {
		fmt.Fprintln(stdout, a.Version())
		return 0
	}
	summary, err := json.MarshalIndent(a.Info(), "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "routepilot policy inspect: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "%s\n", summary)
	return 0
}
