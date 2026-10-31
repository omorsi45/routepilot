package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPolicyGateNeedsAVerdictThatAgreesWithTheExitCode: an interpreter that cannot even import
// the package exits 1 before the estimators run, which is the code for "fails the gate". The
// gate takes an exit code of 0 or 1 as a verdict only when gate.json is there and says the
// same; anything else is a broken run (2), not a candidate that failed.
func TestPolicyGateNeedsAVerdictThatAgreesWithTheExitCode(t *testing.T) {
	for _, c := range []struct {
		name    string
		code    int
		leave   func(t *testing.T, outDir string)
		want    int
		mention string
	}{
		{"fail with no gate.json", 1, func(*testing.T, string) {}, 2, "gate.json"},
		{"pass with no gate.json", 0, func(*testing.T, string) {}, 2, "gate.json"},
		{"fail but gate.json says pass", 1, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `{"pass": true}`) }, 2, "disagrees"},
		{"pass but gate.json says fail", 0, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `{"pass": false}`) }, 2, "disagrees"},
		{"gate.json is not JSON", 0, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `not json`) }, 2, "gate.json"},
		{"gate.json has no pass", 1, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `{}`) }, 2, "gate.json"},
		{"fail with a matching gate.json", 1, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `{"pass": false}`) }, 1, ""},
		{"pass with a matching gate.json", 0, func(t *testing.T, d string) { writeFile(t, filepath.Join(d, "gate.json"), `{"pass": true}`) }, 0, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			fakePythonWriting(t, c.code, nil, "", func(d string) { c.leave(t, d) })
			var stderr bytes.Buffer

			got := run(gateArgs(t, filepath.Join(t.TempDir(), "o")), io.Discard, &stderr)

			if got != c.want {
				t.Errorf("exit = %d, want %d; stderr = %q", got, c.want, stderr.String())
			}
			if c.mention != "" && !strings.Contains(stderr.String(), c.mention) {
				t.Errorf("stderr = %q, want it to mention %q", stderr.String(), c.mention)
			}
		})
	}
}

// TestPolicyGateDropsAStaleVerdict: a gate.json left by an earlier run is removed before the
// replays, so a run whose estimators leave no verdict cannot be read as the previous pass, by
// the exit code or by a consumer of the file.
func TestPolicyGateDropsAStaleVerdict(t *testing.T) {
	out := filepath.Join(t.TempDir(), "o")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(out, "gate.json")
	writeFile(t, stale, `{"pass": true}`)
	fakePythonWriting(t, 1, nil, "", func(string) {})

	var stderr bytes.Buffer
	if got := run(gateArgs(t, out), io.Discard, &stderr); got != 2 {
		t.Errorf("exit = %d, want 2: the stale pass must not stand for this run; stderr = %q", got, stderr.String())
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("the stale gate.json is still there (stat error %v)", err)
	}
}
