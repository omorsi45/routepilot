package main

import (
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// serveRejectAll is a policy configuration whose quality floor no model reaches and that
// refuses instead of relaxing, so every request it decides is answered 422.
const serveRejectAll = "mode: min_cost\nlambda: 0\ndefault_floor: 0.99\nepsilon: 0\nood_action: conservative\non_infeasible: reject\n"

// withCanary rewrites the fixture's config with a canary slot that uses canaryYAML and takes
// percent of the traffic. The canary is the same artifact under another policy configuration.
func (f *serveFixture) withCanary(t *testing.T, canaryYAML string, percent int) {
	t.Helper()
	writeFileAtomic(t, filepath.Join(f.dir, "canary.yaml"), canaryYAML)
	b, err := os.ReadFile(f.cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	const stable = "    config: stable.yaml\n"
	if !strings.Contains(string(b), stable) {
		t.Fatal("the fixture config has no stable slot to attach a canary to")
	}
	canary := stable + "  canary:\n    name: tiny\n    version: " + f.version +
		"\n    config: canary.yaml\n    percent: " + strconv.Itoa(percent) + "\n"
	writeFileAtomic(t, f.cfgPath, strings.Replace(string(b), stable, canary, 1))
}

// TestServeEvaluatesGuardrails: serve runs the guard on a timer, so a canary that fails every
// request is pulled without anyone calling the evaluation, and the stable policy serves from
// then on.
func TestServeEvaluatesGuardrails(t *testing.T) {
	f := newServeFixture(t, "")
	f.withCanary(t, serveRejectAll, 50)
	cancel, _, _ := inject(t)
	origEvery := guardEvery
	guardEvery = 20 * time.Millisecond
	t.Cleanup(func() { guardEvery = origEvery })
	r := startServe(t, f.cfgPath, cancel)

	stable := f.slotVersion(t, serveFloor0)
	rolledBack := func() bool { return strings.Contains(r.stderr.String(), "guardrails rolled back the canary") }
	var canary422, stable200 int
	for i := 0; i < 3000 && !rolledBack(); i++ {
		status, h := r.chat(t, nil)
		switch {
		case status == http.StatusUnprocessableEntity:
			canary422++
		case status == http.StatusOK && h.Get("x-routepilot-policy-version") == stable:
			stable200++
		default:
			t.Fatalf("request %d: status %d version %q; stderr: %s", i, status, h.Get("x-routepilot-policy-version"), r.stderr.String())
		}
	}
	waitFor(t, "the guardrails to pull the canary", rolledBack)
	if canary422 < 200 || stable200 == 0 {
		t.Errorf("%d canary refusals and %d stable answers before the rollback, want at least 200 and some", canary422, stable200)
	}

	// From here on only the stable policy answers.
	for i := 0; i < 40; i++ {
		if status, h := r.chat(t, nil); status != http.StatusOK || h.Get("x-routepilot-policy-version") != stable {
			t.Fatalf("after the rollback: status %d version %q, want 200 from the stable %q", status, h.Get("x-routepilot-policy-version"), stable)
		}
	}
	if n := strings.Count(r.stderr.String(), "guardrails rolled back the canary"); n != 1 {
		t.Errorf("%d rollback log lines, want 1", n)
	}
	if code := r.stop(t); code != 0 {
		t.Errorf("serve exited with %d, want 0; stderr: %s", code, r.stderr.String())
	}
}
