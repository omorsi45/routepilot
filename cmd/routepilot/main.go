// Command routepilot is the RoutePilot gateway CLI.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/omorsi45/routepilot/internal/version"
)

const usage = "usage: routepilot <command>\n\ncommands:\n  serve     run the gateway (routepilot serve -config <path>)\n  version   print the version and exit\n"

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run dispatches a subcommand and returns the process exit code. args excludes
// the program name.
func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}

	switch args[0] {
	case "serve":
		return runServe(args[1:], stdout, stderr)
	case "version":
		fmt.Fprintln(stdout, version.String())
		return 0
	default:
		fmt.Fprintf(stderr, "routepilot: unknown command %q\n\n%s", args[0], usage)
		return 2
	}
}
