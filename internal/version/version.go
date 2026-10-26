// Package version reports the RoutePilot build version.
package version

import (
	"fmt"
	"runtime"
)

// Version is the release version. It defaults to "dev" and is overridden at
// build time with:
//
//	-ldflags "-X github.com/omorsi45/routepilot/internal/version.Version=1.2.3"
var Version = "dev"

// String returns the human-readable version line, for example
// "routepilot dev (go1.27.2)".
func String() string {
	return fmt.Sprintf("routepilot %s (%s)", Version, runtime.Version())
}
