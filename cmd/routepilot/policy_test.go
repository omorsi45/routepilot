package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var tinyPolicyRoot = filepath.Join("..", "..", "testdata", "policy", "tiny")

func tinyPolicyDir(t *testing.T) string {
	t.Helper()
	entries, err := os.ReadDir(tinyPolicyRoot)
	if err != nil {
		t.Fatalf("read %s: %v", tinyPolicyRoot, err)
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	if len(dirs) != 1 {
		t.Fatalf("%s holds %d version directories, want exactly 1", tinyPolicyRoot, len(dirs))
	}
	return filepath.Join(tinyPolicyRoot, dirs[0])
}

func TestPolicyVerifyCLI(t *testing.T) {
	dir := tinyPolicyDir(t)
	var stdout, stderr bytes.Buffer

	code := run([]string{"policy", "verify", dir}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(policy verify) = %d, want 0; stderr = %q", code, stderr.String())
	}
	if want := filepath.Base(dir) + "\n"; stdout.String() != want {
		t.Errorf("stdout = %q, want the version %q", stdout.String(), want)
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q, want empty", stderr.String())
	}
}

func TestPolicyVerifyCLIRefusesACorruptArtifact(t *testing.T) {
	src := tinyPolicyDir(t)
	dir := filepath.Join(t.TempDir(), filepath.Base(src))
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		b, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "weights.bin" {
			b[len(b)/3] ^= 0xff
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var stdout, stderr bytes.Buffer

	code := run([]string{"policy", "verify", dir}, &stdout, &stderr)

	if code != 1 {
		t.Fatalf("run(policy verify corrupt) = %d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty (no version for a refused artifact)", stdout.String())
	}
	if !strings.Contains(stderr.String(), "hash check failed") {
		t.Errorf("stderr = %q, want it to name the hash check", stderr.String())
	}
}

func TestPolicyVerifyCLIMissingDirectory(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run([]string{"policy", "verify", filepath.Join(t.TempDir(), "absent")}, &stdout, &stderr)

	if code != 1 || stdout.Len() != 0 || stderr.Len() == 0 {
		t.Errorf("run = %d, stdout = %q, stderr = %q; want 1, empty, an error", code, stdout.String(), stderr.String())
	}
}

func TestPolicyInspectCLI(t *testing.T) {
	dir := tinyPolicyDir(t)
	var stdout, stderr bytes.Buffer

	code := run([]string{"policy", "inspect", dir}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run(policy inspect) = %d, want 0; stderr = %q", code, stderr.String())
	}
	var got struct {
		Version string `json:"version"`
		Dim     int    `json:"dim"`
		Models  []struct {
			ID         string `json:"id"`
			Calibrator string `json:"calibrator"`
		} `json:"models"`
		OODThreshold float64 `json:"ood_threshold"`
		Training     struct {
			Pool       string   `json:"pool"`
			TrainedOn  []string `json:"trained_on"`
			Calibrated []string `json:"calibrated_on"`
		} `json:"training"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, stdout.String())
	}
	if got.Version != filepath.Base(dir) || got.Dim != 1024 || !(got.OODThreshold > 0) {
		t.Errorf("summary = %+v", got)
	}
	if len(got.Models) != 2 || got.Models[0].ID != "tiny-weak" || got.Models[0].Calibrator != "isotonic" ||
		got.Models[1].ID != "tiny-strong" || got.Models[1].Calibrator != "platt" {
		t.Errorf("models = %+v", got.Models)
	}
	if got.Training.Pool != "tiny" || len(got.Training.TrainedOn) != 1 || got.Training.TrainedOn[0] != "train" ||
		len(got.Training.Calibrated) != 1 || got.Training.Calibrated[0] != "calib" {
		t.Errorf("training = %+v", got.Training)
	}
	if strings.Contains(stdout.String(), "messages") {
		t.Error("inspect output carries golden prompt messages")
	}
}

func TestPolicyUsageErrors(t *testing.T) {
	dir := tinyPolicyDir(t)
	for name, args := range map[string][]string{
		"no subcommand":     {"policy"},
		"unknown":           {"policy", "rebuild", dir},
		"verify no dir":     {"policy", "verify"},
		"verify two dirs":   {"policy", "verify", dir, dir},
		"inspect no dir":    {"policy", "inspect"},
		"inspect extra arg": {"policy", "inspect", dir, "extra"},
	} {
		t.Run(name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer

			code := run(args, &stdout, &stderr)

			if code != 2 {
				t.Errorf("run(%v) = %d, want 2", args, code)
			}
			if stdout.Len() != 0 || !strings.Contains(stderr.String(), "usage: routepilot policy") {
				t.Errorf("stdout = %q, stderr = %q; want empty stdout and the policy usage", stdout.String(), stderr.String())
			}
		})
	}
}

func TestMainUsageListsPolicy(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := run(nil, &stdout, &stderr)

	if code != 2 || !strings.Contains(stderr.String(), "\n  policy ") {
		t.Errorf("run() = %d, stderr = %q; want 2 and a usage line for the policy command", code, stderr.String())
	}
}
