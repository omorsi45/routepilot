package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/policy"
)

// gateRun records what the gate asked the Python side to run.
type gateRun struct {
	calls  int
	name   string
	args   []string
	code   int
	err    error
	stdout string
}

// fakePython replaces the command runner for the test and returns what it was asked to run.
// It answers with the exit code and error it is given and writes stdout to the gate's stdout.
func fakePython(t *testing.T, code int, err error, stdout string) *gateRun {
	t.Helper()
	g := &gateRun{code: code, err: err, stdout: stdout}
	orig := runGateCommand
	runGateCommand = func(name string, args []string, out, _ io.Writer) (int, error) {
		g.calls++
		g.name, g.args = name, slices.Clone(args)
		_, _ = io.WriteString(out, g.stdout)
		return g.code, g.err
	}
	t.Cleanup(func() { runGateCommand = orig })
	return g
}

// gateArgs is a complete gate command line over the learned replay fixture: the incumbent and
// the candidate are the tiny artifact under their own policy configs.
func gateArgs(t *testing.T, outDir string, extra ...string) []string {
	t.Helper()
	candidateCfg := filepath.Join(t.TempDir(), "candidate.yaml")
	writeFile(t, candidateCfg, "mode: min_cost\nlambda: 0\ndefault_floor: 0.4\nepsilon: 0\nood_action: conservative\non_infeasible: best_effort\n")
	args := []string{
		"policy", "gate",
		"--config", filepath.Join(learnedFixtureDir, "config.yaml"),
		"--input", filepath.Join(learnedFixtureDir, "items.jsonl"),
		"--split", "calib",
		"--incumbent", tinyArtifactDir,
		"--incumbent-config", filepath.Join(learnedFixtureDir, "policy.yaml"),
		"--candidate", tinyArtifactDir,
		"--candidate-config", candidateCfg,
		"--python", "python-under-test",
		"--out", outDir,
	}
	return append(args, extra...)
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

type gateRecord struct {
	ItemID        string            `json:"item_id"`
	Split         string            `json:"split"`
	PolicyVersion *string           `json:"policy_version"`
	Model         *string           `json:"model"`
	Explored      bool              `json:"explored"`
	Propensity    *float64          `json:"propensity"`
	Candidates    []json.RawMessage `json:"candidates"`
}

func readGateRecords(t *testing.T, path string) []gateRecord {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the gate did not write %s: %v", filepath.Base(path), err)
	}
	var recs []gateRecord
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var r gateRecord
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("decode %q: %v", line, err)
		}
		recs = append(recs, r)
	}
	return recs
}

// TestPolicyGateRunsThreeReplaysThenThePythonGate: the incumbent explores for the logged file,
// the candidate and the incumbent are deterministic, and the Python gate is asked to compare
// the three files, writing next to them.
func TestPolicyGateRunsThreeReplaysThenThePythonGate(t *testing.T) {
	py := fakePython(t, 0, nil, "gate: PASS\n")
	out := filepath.Join(t.TempDir(), "gate-out")
	var stdout, stderr bytes.Buffer
	args := gateArgs(t, out, "--epsilon", "0.9", "--seed", "3")
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("run(policy gate) = %d, want 0; stderr = %q", code, stderr.String())
	}

	wantArgs := []string{
		"-m", "routepilot_train.ope", "gate",
		"--logged", filepath.Join(out, "logged.jsonl"),
		"--candidate", filepath.Join(out, "candidate.jsonl"),
		"--incumbent", filepath.Join(out, "incumbent.jsonl"),
		"--out", out,
	}
	if py.calls != 1 || py.name != "python-under-test" || !slices.Equal(py.args, wantArgs) {
		t.Errorf("python run = %d calls, name %q, args %v; want 1 call of python-under-test with %v", py.calls, py.name, py.args, wantArgs)
	}
	if !strings.Contains(stdout.String(), "gate: PASS") {
		t.Errorf("stdout = %q, want the Python verdict passed through", stdout.String())
	}

	// The files hold the calib items only (two of the fixture's six).
	logged := readGateRecords(t, filepath.Join(out, "logged.jsonl"))
	cand := readGateRecords(t, filepath.Join(out, "candidate.jsonl"))
	inc := readGateRecords(t, filepath.Join(out, "incumbent.jsonl"))
	for name, recs := range map[string][]gateRecord{"logged": logged, "candidate": cand, "incumbent": inc} {
		if len(recs) != 2 {
			t.Fatalf("%s has %d records, want the 2 calib items", name, len(recs))
		}
		for _, r := range recs {
			if r.Split != "calib" || r.Model == nil || r.Propensity == nil {
				t.Errorf("%s record %+v, want a calib item with a model and a propensity", name, r)
			}
		}
	}

	// The logging policy is the incumbent with exploration, and carries its candidates; the
	// two replays under test are deterministic.
	base := policy.Config{Mode: "min_cost", OODAction: "conservative", OnInfeasible: "best_effort"}
	explore := base
	explore.Epsilon = 0.9
	candidate := policy.Config{Mode: "min_cost", DefaultFloor: 0.4, OODAction: "conservative", OnInfeasible: "best_effort"}
	for _, c := range []struct {
		name string
		recs []gateRecord
		want string
	}{
		{"logged", logged, "0782d801a5a3+" + explore.Hash8()},
		{"incumbent", inc, "0782d801a5a3+" + base.Hash8()},
		{"candidate", cand, "0782d801a5a3+" + candidate.Hash8()},
	} {
		for _, r := range c.recs {
			if r.PolicyVersion == nil || *r.PolicyVersion != c.want {
				t.Errorf("%s version = %v, want %s", c.name, r.PolicyVersion, c.want)
			}
		}
	}
	for _, r := range logged {
		if len(r.Candidates) != 2 {
			t.Errorf("logged %s has %d candidates, want 2: the logging policy's outcome model", r.ItemID, len(r.Candidates))
		}
	}
	for name, recs := range map[string][]gateRecord{"candidate": cand, "incumbent": inc} {
		for _, r := range recs {
			if r.Explored || *r.Propensity != 1 {
				t.Errorf("%s %s explored %v propensity %v, want a deterministic replay", name, r.ItemID, r.Explored, *r.Propensity)
			}
		}
	}
}

// TestPolicyGateSeedReachesTheLoggedReplay: the logged file is the exploring replay, so the
// seed and the epsilon reach it and some item is explored.
func TestPolicyGateSeedReachesTheLoggedReplay(t *testing.T) {
	fakePython(t, 0, nil, "")
	read := func(seed string) []gateRecord {
		out := filepath.Join(t.TempDir(), "o")
		if code := run(gateArgs(t, out, "--epsilon", "0.9", "--seed", seed), io.Discard, io.Discard); code != 0 {
			t.Fatalf("seed %s: exit %d", seed, code)
		}
		return readGateRecords(t, filepath.Join(out, "logged.jsonl"))
	}
	explored := false
	for _, seed := range []string{"0", "1", "2", "3", "4", "5", "6", "7"} {
		for _, r := range read(seed) {
			explored = explored || r.Explored
		}
	}
	if !explored {
		t.Error("no logged record explored at epsilon 0.9 over eight seeds: the epsilon or the seed does not reach the logged replay")
	}
}

// TestPolicyGateExitCodes: 0 passes, 1 fails the gate, and everything else is an error (2).
func TestPolicyGateExitCodes(t *testing.T) {
	for _, c := range []struct {
		name string
		code int
		err  error
		want int
	}{
		{"pass", 0, nil, 0},
		{"fail", 1, nil, 1},
		{"python usage error", 2, nil, 2},
		{"python crashed", 134, nil, 2},
		{"python could not start", -1, os.ErrNotExist, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			fakePython(t, c.code, c.err, "")
			var stderr bytes.Buffer
			if got := run(gateArgs(t, filepath.Join(t.TempDir(), "o")), io.Discard, &stderr); got != c.want {
				t.Errorf("exit = %d, want %d; stderr = %q", got, c.want, stderr.String())
			}
			if c.want == 2 && stderr.Len() == 0 {
				t.Error("an error exit says nothing on stderr")
			}
		})
	}
}

// TestPolicyGateStopsBeforePythonWhenAReplayFails: a replay that cannot run is an error and
// the Python gate is not asked to judge half a result.
func TestPolicyGateStopsBeforePythonWhenAReplayFails(t *testing.T) {
	py := fakePython(t, 0, nil, "")
	corrupt := filepath.Join(t.TempDir(), "corrupt")
	if err := os.MkdirAll(corrupt, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		b, err := os.ReadFile(filepath.Join(tinyArtifactDir, name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "weights.bin" {
			b = b[:len(b)/2]
		}
		writeFile(t, filepath.Join(corrupt, name), string(b))
	}
	args := gateArgs(t, filepath.Join(t.TempDir(), "o"))
	for i, a := range args {
		if a == "--candidate" {
			args[i+1] = corrupt
		}
	}
	var stderr bytes.Buffer
	if code := run(args, io.Discard, &stderr); code != 2 {
		t.Errorf("exit = %d, want 2 for a corrupt candidate artifact; stderr = %q", code, stderr.String())
	}
	if py.calls != 0 {
		t.Errorf("python was run %d times after a failed replay, want 0", py.calls)
	}
}

// TestPolicyGateReplayInputErrors: an input the replay cannot read is an error and the Python
// gate is not run on the files that did get written.
func TestPolicyGateReplayInputErrors(t *testing.T) {
	py := fakePython(t, 0, nil, "")
	args := gateArgs(t, filepath.Join(t.TempDir(), "o"))
	args[slices.Index(args, "--input")+1] = filepath.Join(t.TempDir(), "missing.jsonl")
	var stderr bytes.Buffer
	if code := run(args, io.Discard, &stderr); code != 2 {
		t.Errorf("exit = %d, want 2; stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "replay logged") {
		t.Errorf("stderr = %q, want it to name the replay that failed", stderr.String())
	}
	if py.calls != 0 {
		t.Errorf("python was run %d times after a failed replay, want 0", py.calls)
	}
}

// TestPolicyGateExploresAtTenPercentByDefault: without --epsilon the logging policy explores
// at 0.1.
func TestPolicyGateExploresAtTenPercentByDefault(t *testing.T) {
	fakePython(t, 0, nil, "")
	out := filepath.Join(t.TempDir(), "o")
	if code := run(gateArgs(t, out), io.Discard, io.Discard); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	explore := policy.Config{Mode: "min_cost", Epsilon: 0.1, OODAction: "conservative", OnInfeasible: "best_effort"}
	want := "0782d801a5a3+" + explore.Hash8()
	for _, r := range readGateRecords(t, filepath.Join(out, "logged.jsonl")) {
		if r.PolicyVersion == nil || *r.PolicyVersion != want {
			t.Errorf("logged version = %v, want %s (epsilon 0.1)", r.PolicyVersion, want)
		}
	}
}

func TestPolicyGateUsageErrors(t *testing.T) {
	py := fakePython(t, 0, nil, "")
	valid := gateArgs(t, filepath.Join(t.TempDir(), "o"))
	without := func(flag string) []string {
		var out []string
		for i := 0; i < len(valid); i++ {
			if valid[i] == "--"+flag {
				i++
				continue
			}
			out = append(out, valid[i])
		}
		return out
	}
	cases := map[string][]string{
		"no config":           without("config"),
		"no input":            without("input"),
		"no incumbent":        without("incumbent"),
		"no incumbent config": without("incumbent-config"),
		"no candidate":        without("candidate"),
		"no candidate config": without("candidate-config"),
		"no out":              without("out"),
		"epsilon one":         append(slices.Clone(valid), "--epsilon", "1"),
		"epsilon negative":    append(slices.Clone(valid), "--epsilon", "-0.1"),
		"extra argument":      append(slices.Clone(valid), "stray"),
		"unknown flag":        append(slices.Clone(valid), "--nope"),
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			var stderr bytes.Buffer
			if code := run(args, io.Discard, &stderr); code != 2 {
				t.Errorf("exit = %d, want 2; stderr = %q", code, stderr.String())
			}
			if !strings.Contains(stderr.String(), "usage: routepilot policy gate") {
				t.Errorf("stderr = %q, want the gate usage", stderr.String())
			}
		})
	}
	if py.calls != 0 {
		t.Errorf("python was run %d times for usage errors, want 0", py.calls)
	}
}

// TestPolicyGateDefaultsToCalib: --split is optional and means calib.
func TestPolicyGateDefaultsToCalib(t *testing.T) {
	fakePython(t, 0, nil, "")
	args := gateArgs(t, filepath.Join(t.TempDir(), "o"))
	for i, a := range args {
		if a == "--split" {
			args = append(args[:i], args[i+2:]...)
			break
		}
	}
	out := args[slices.Index(args, "--out")+1]
	if code := run(args, io.Discard, io.Discard); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	for _, r := range readGateRecords(t, filepath.Join(out, "logged.jsonl")) {
		if r.Split != "calib" {
			t.Errorf("split %q, want calib", r.Split)
		}
	}
}

func TestDefaultPythonPrefersTheWindowsVenv(t *testing.T) {
	root := t.TempDir()
	if got, want := defaultPython(root), filepath.Join(root, ".venv", "bin", "python"); got != want {
		t.Errorf("defaultPython(empty) = %q, want %q", got, want)
	}
	if err := os.MkdirAll(filepath.Join(root, ".venv", "Scripts"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, ".venv", "Scripts", "python.exe"), "")
	if got, want := defaultPython(root), filepath.Join(root, ".venv", "Scripts", "python.exe"); got != want {
		t.Errorf("defaultPython(with Scripts/python.exe) = %q, want %q", got, want)
	}
}

func TestPolicyUsageListsGate(t *testing.T) {
	var stderr bytes.Buffer
	if code := run([]string{"policy"}, io.Discard, &stderr); code != 2 || !strings.Contains(stderr.String(), "routepilot policy gate") {
		t.Errorf("exit %d stderr %q, want 2 and the gate in the usage", code, stderr.String())
	}
}
