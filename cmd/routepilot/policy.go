package main

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/omorsi45/routepilot/internal/policy"
)

const policyUsage = "usage: routepilot policy verify DIR\n" +
	"       routepilot policy inspect DIR\n" +
	"\n" +
	"DIR is a policy artifact directory such as policies/flagship/<version>.\n" +
	"verify checks the content hash, spec version and golden predictions, then prints the version.\n" +
	"inspect does the same checks, then prints a JSON summary of the artifact.\n"

// runPolicy implements `routepilot policy`. It returns 0 on success, 1 when the artifact is
// refused or unreadable and 2 for usage errors.
func runPolicy(args []string, stdout, stderr io.Writer) int {
	if len(args) == 1 && (args[0] == "-h" || args[0] == "--help") {
		fmt.Fprint(stderr, policyUsage)
		return 0
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
