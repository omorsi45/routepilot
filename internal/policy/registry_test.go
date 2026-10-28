package policy

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
)

// tinySource returns the committed tiny artifact directory and its version.
func tinySource(t *testing.T) (dir, version string) {
	t.Helper()
	root := filepath.Join("..", "..", "testdata", "policy", "tiny")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read %s: %v", root, err)
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	if len(names) != 1 {
		t.Fatalf("want exactly one artifact under %s, found %v", root, names)
	}
	return filepath.Join(root, names[0]), names[0]
}

func copyDir(t *testing.T, src, dst string) {
	t.Helper()
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// bumpClock hands out strictly increasing modification times, so a rewrite is always a
// visible change however coarse the file system's timestamps are.
var bumpClock atomic.Int64

// writeAtomic replaces path with data in one rename, so a poller never reads a half-written
// file, and gives it a modification time no earlier write had.
func writeAtomic(t *testing.T, path string, data []byte) {
	t.Helper()
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		t.Fatal(err)
	}
	mod := time.Now().Add(time.Duration(bumpClock.Add(1)) * time.Second)
	if err := os.Chtimes(tmp, mod, mod); err != nil {
		t.Fatal(err)
	}
	var err error
	for range 100 { // a reader may hold the target open for a moment on Windows
		if err = os.Rename(tmp, path); err == nil {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal(err)
}

// fixture is a base directory holding a copy of the tiny artifact under
// policies/tiny/<version> and policy configuration files next to it.
type fixture struct {
	t       *testing.T
	base    string
	version string // artifact version, also its directory name
	srcDir  string
	cfg     atomic.Pointer[config.PoliciesConfig]
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	src, ver := tinySource(t)
	f := &fixture{t: t, base: t.TempDir(), version: ver, srcDir: src}
	copyDir(t, src, f.artifactDir())
	f.writePolicy("stable.yaml", 0.1)
	f.set(config.PoliciesConfig{Root: "policies", Stable: f.ref("stable.yaml")})
	return f
}

func (f *fixture) artifactDir() string {
	return filepath.Join(f.base, "policies", "tiny", f.version)
}

func (f *fixture) ref(policyFile string) config.PolicyRef {
	return config.PolicyRef{Name: "tiny", Version: f.version, Config: policyFile}
}

func (f *fixture) set(c config.PoliciesConfig) { f.cfg.Store(&c) }

func (f *fixture) load() (config.PoliciesConfig, error) { return *f.cfg.Load(), nil }

func (f *fixture) registry() *Registry {
	f.t.Helper()
	r, err := NewRegistry(f.load, f.base)
	if err != nil {
		f.t.Fatalf("NewRegistry: %v", err)
	}
	return r
}

// writePolicy writes a policy configuration whose default floor is floor, so two floors
// give two configuration hashes and so two slot versions.
func (f *fixture) writePolicy(name string, floor float64) {
	f.t.Helper()
	y := fmt.Sprintf("mode: min_cost\nlambda: 0\ndefault_floor: %v\nepsilon: 0\nood_action: conservative\non_infeasible: best_effort\n", floor)
	writeAtomic(f.t, filepath.Join(f.base, name), []byte(y))
}

// version is the slot version the registry must report for the policy file name: the
// artifact version and the configuration hash, computed here without the registry.
func (f *fixture) slotVersion(name string) string {
	f.t.Helper()
	art, err := Load(f.srcDir)
	if err != nil {
		f.t.Fatal(err)
	}
	c, err := LoadConfig(filepath.Join(f.base, name))
	if err != nil {
		f.t.Fatal(err)
	}
	return art.Version() + "+" + c.Hash8()
}

func TestNewRegistryLoadsStable(t *testing.T) {
	f := newFixture(t)
	before := time.Now()
	reg := f.registry()
	snap := reg.Snapshot()

	if snap == nil {
		t.Fatal("Snapshot() = nil")
	}
	if snap.Stable.Role != "stable" {
		t.Errorf("stable role = %q", snap.Stable.Role)
	}
	if want := f.slotVersion("stable.yaml"); snap.Stable.Version != want {
		t.Errorf("stable version = %q, want %q", snap.Stable.Version, want)
	}
	if snap.Stable.Artifact == nil || snap.Stable.Artifact.Version() != f.version {
		t.Errorf("stable artifact = %v, want version %s", snap.Stable.Artifact, f.version)
	}
	if snap.Stable.Config.DefaultFloor != 0.1 {
		t.Errorf("stable default floor = %v, want 0.1", snap.Stable.Config.DefaultFloor)
	}
	if snap.Canary != nil || len(snap.Shadow) != 0 || snap.CanaryPercent != 0 {
		t.Errorf("unexpected canary or shadow: %+v", snap)
	}
	if snap.Generation != 1 {
		t.Errorf("generation = %d, want 1", snap.Generation)
	}
	if snap.LoadedAt.Before(before) || snap.LoadedAt.After(time.Now()) {
		t.Errorf("LoadedAt = %v, want between %v and now", snap.LoadedAt, before)
	}
	if reg.ReloadErrors() != 0 || reg.LastError() != nil {
		t.Errorf("fresh registry has errors: %d, %v", reg.ReloadErrors(), reg.LastError())
	}
}

func TestNewRegistryLoadsCanaryAndShadow(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("canary.yaml", 0.3)
	f.writePolicy("shadow-1.yaml", 0.5)
	f.writePolicy("shadow-2.yaml", 0.7)
	c := *f.cfg.Load()
	c.Canary = &config.CanaryRef{PolicyRef: f.ref("canary.yaml"), Percent: 12.5}
	c.Shadow = []config.PolicyRef{f.ref("shadow-1.yaml"), f.ref("shadow-2.yaml")}
	f.set(c)

	snap := f.registry().Snapshot()
	if snap.Canary == nil || snap.Canary.Role != "canary" || snap.Canary.Version != f.slotVersion("canary.yaml") {
		t.Errorf("canary = %+v", snap.Canary)
	}
	if snap.CanaryPercent != 12.5 {
		t.Errorf("canary percent = %v, want 12.5", snap.CanaryPercent)
	}
	if len(snap.Shadow) != 2 {
		t.Fatalf("shadow slots = %d, want 2", len(snap.Shadow))
	}
	for i, name := range []string{"shadow-1.yaml", "shadow-2.yaml"} {
		if got := snap.Shadow[i]; got.Role != "shadow" || got.Version != f.slotVersion(name) {
			t.Errorf("shadow[%d] = role %q version %q, want shadow %q", i, got.Role, got.Version, f.slotVersion(name))
		}
	}
}

func TestNewRegistryRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(f *fixture, c *config.PoliciesConfig)
		load   func() (config.PoliciesConfig, error)
		want   string
	}{
		{name: "no stable", mutate: func(_ *fixture, c *config.PoliciesConfig) { c.Stable = config.PolicyRef{} }, want: "stable"},
		{name: "load error", load: func() (config.PoliciesConfig, error) { return config.PoliciesConfig{}, errors.New("disk on fire") }, want: "disk on fire"},
		{name: "missing artifact", mutate: func(_ *fixture, c *config.PoliciesConfig) { c.Stable.Version = "ffffffffffff" }, want: "ffffffffffff"},
		{name: "missing policy config", mutate: func(_ *fixture, c *config.PoliciesConfig) { c.Stable.Config = "nope.yaml" }, want: "nope.yaml"},
		{name: "name escapes the root", mutate: func(_ *fixture, c *config.PoliciesConfig) { c.Stable.Name = ".." }, want: "name"},
		{name: "version with a separator", mutate: func(_ *fixture, c *config.PoliciesConfig) { c.Stable.Version = "a/b" }, want: "version"},
		{name: "canary percent too high", mutate: func(f *fixture, c *config.PoliciesConfig) {
			c.Canary = &config.CanaryRef{PolicyRef: f.ref("stable.yaml"), Percent: 100.5}
		}, want: "percent"},
		{name: "canary percent negative", mutate: func(f *fixture, c *config.PoliciesConfig) {
			c.Canary = &config.CanaryRef{PolicyRef: f.ref("stable.yaml"), Percent: -1}
		}, want: "percent"},
		{name: "bad canary artifact", mutate: func(f *fixture, c *config.PoliciesConfig) {
			ref := f.ref("stable.yaml")
			ref.Version = "ffffffffffff"
			c.Canary = &config.CanaryRef{PolicyRef: ref, Percent: 5}
		}, want: "canary"},
		{name: "bad shadow policy config", mutate: func(f *fixture, c *config.PoliciesConfig) {
			c.Shadow = []config.PolicyRef{f.ref("nope.yaml")}
		}, want: "shadow"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t)
			c := *f.cfg.Load()
			if tc.mutate != nil {
				tc.mutate(f, &c)
			}
			f.set(c)
			load := f.load
			if tc.load != nil {
				load = tc.load
			}
			reg, err := NewRegistry(load, f.base)
			if err == nil {
				t.Fatalf("NewRegistry succeeded: %+v", reg.Snapshot())
			}
			if reg != nil {
				t.Errorf("NewRegistry returned a registry with its error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

func TestRegistryPickCanaryFraction(t *testing.T) {
	snap := &Snapshot{
		Stable:        Slot{Role: "stable", Version: "s"},
		Canary:        &Slot{Role: "canary", Version: "c"},
		CanaryPercent: 5,
	}
	rng := rand.New(rand.NewPCG(7, 11))
	const n = 100_000
	canary := 0
	for range n {
		id := fmt.Sprintf("%016x%08x", rng.Uint64(), rng.Uint32())
		if snap.Pick("tenant-a", id).Role == "canary" {
			canary++
		}
	}
	if frac := float64(canary) / n; frac < 0.048 || frac > 0.052 {
		t.Errorf("canary fraction = %.4f over %d requests at 5%%, want 0.048 to 0.052", frac, n)
	}
}

func TestRegistryPickFollowsTheDocumentedHash(t *testing.T) {
	snap := &Snapshot{Stable: Slot{Role: "stable"}, Canary: &Slot{Role: "canary"}, CanaryPercent: 37.5}
	rng := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		tenant := fmt.Sprintf("tenant-%d", rng.IntN(5))
		id := fmt.Sprintf("%024x", rng.Uint64())
		h := fnv.New64a()
		_, _ = h.Write([]byte(tenant + id))
		want := "stable"
		if h.Sum64()%10000 < uint64(37.5*100) {
			want = "canary"
		}
		if got := snap.Pick(tenant, id).Role; got != want {
			t.Fatalf("Pick(%q, %q) = %s, want %s", tenant, id, got, want)
		}
		if again := snap.Pick(tenant, id).Role; again != want {
			t.Fatalf("Pick is not deterministic for %q %q", tenant, id)
		}
	}
}

func TestRegistryPickEdges(t *testing.T) {
	canary := &Slot{Role: "canary"}
	stable := Slot{Role: "stable"}
	cases := []struct {
		name    string
		snap    *Snapshot
		wantAll string
	}{
		{"zero percent", &Snapshot{Stable: stable, Canary: canary, CanaryPercent: 0}, "stable"},
		{"hundred percent", &Snapshot{Stable: stable, Canary: canary, CanaryPercent: 100}, "canary"},
		{"percent without a canary slot", &Snapshot{Stable: stable, CanaryPercent: 100}, "stable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for i := range 1000 {
				if got := tc.snap.Pick("t", fmt.Sprintf("id-%d", i)).Role; got != tc.wantAll {
					t.Fatalf("Pick = %s, want %s for every request", got, tc.wantAll)
				}
			}
		})
	}
}

func TestReloadSwapsAtomically(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("a.yaml", 0.1)
	f.writePolicy("b.yaml", 0.2)
	vA, vB := f.slotVersion("a.yaml"), f.slotVersion("b.yaml")
	if vA == vB {
		t.Fatal("fixture policies share a version")
	}
	// Two states that differ in stable and canary together. A reader that saw a stable
	// slot of one state with the canary of the other would see equal versions.
	state := func(stable, canary string) config.PoliciesConfig {
		return config.PoliciesConfig{
			Root:   "policies",
			Stable: f.ref(stable),
			Canary: &config.CanaryRef{PolicyRef: f.ref(canary), Percent: 10},
		}
	}
	f.set(state("a.yaml", "b.yaml"))
	reg := f.registry()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var last uint64
			for {
				select {
				case <-stop:
					return
				default:
				}
				s := reg.Snapshot()
				if s == nil || s.Canary == nil {
					t.Error("Snapshot() returned a nil or canary-less snapshot")
					return
				}
				pair := s.Stable.Version + "/" + s.Canary.Version
				if pair != vA+"/"+vB && pair != vB+"/"+vA {
					t.Errorf("torn snapshot: stable %s, canary %s", s.Stable.Version, s.Canary.Version)
					return
				}
				if got := s.Stable.Artifact.Version() + "+" + s.Stable.Config.Hash8(); got != s.Stable.Version {
					t.Errorf("slot mixes parts: version %s, parts %s", s.Stable.Version, got)
					return
				}
				if s.Generation < last {
					t.Errorf("generation went back from %d to %d", last, s.Generation)
					return
				}
				last = s.Generation
			}
		}()
	}

	const reloads = 40
	for i := range reloads {
		if i%2 == 0 {
			f.set(state("b.yaml", "a.yaml"))
		} else {
			f.set(state("a.yaml", "b.yaml"))
		}
		if err := reg.Reload(); err != nil {
			close(stop)
			wg.Wait()
			t.Fatalf("Reload %d: %v", i, err)
		}
	}
	close(stop)
	wg.Wait()
	if got := reg.Snapshot().Generation; got != reloads+1 {
		t.Errorf("generation = %d, want %d", got, reloads+1)
	}
}

func TestReloadKeepsOldOnCorruptArtifact(t *testing.T) {
	f := newFixture(t)
	reg := f.registry()
	before := reg.Snapshot()

	weights := filepath.Join(f.artifactDir(), "weights.bin")
	good, err := os.ReadFile(weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weights, good[:len(good)/2], 0o644); err != nil {
		t.Fatal(err)
	}

	err = reg.Reload()
	if err == nil {
		t.Fatal("Reload accepted a truncated weights.bin")
	}
	if !errors.Is(err, ErrHashMismatch) {
		t.Errorf("Reload error = %v, want it to wrap ErrHashMismatch", err)
	}
	if got := reg.ReloadErrors(); got != 1 {
		t.Errorf("ReloadErrors() = %d, want 1", got)
	}
	if got := reg.LastError(); got == nil || got.Error() != err.Error() {
		t.Errorf("LastError() = %v, want the Reload error %v", got, err)
	}
	if reg.Snapshot() != before {
		t.Error("the snapshot changed after a failed reload")
	}
	if got := reg.Snapshot().Stable.Version; got != f.slotVersion("stable.yaml") {
		t.Errorf("served version = %s, want the old one", got)
	}
	// The old artifact still scores.
	if sc := reg.Snapshot().Stable.Artifact.Score(nil); len(sc.Models) == 0 {
		t.Error("the kept artifact no longer scores")
	}

	// Once the file is whole again the next reload succeeds and clears the error.
	if err := os.WriteFile(weights, good, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := reg.Reload(); err != nil {
		t.Fatalf("Reload after repair: %v", err)
	}
	if reg.LastError() != nil {
		t.Errorf("LastError() = %v after a good reload, want nil", reg.LastError())
	}
	if got := reg.ReloadErrors(); got != 1 {
		t.Errorf("ReloadErrors() = %d after a good reload, want it to stay 1", got)
	}
	if got := reg.Snapshot().Generation; got != before.Generation+1 {
		t.Errorf("generation = %d, want %d", got, before.Generation+1)
	}
}

func TestReloadKeepsOldOnAnyFailure(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(f *fixture)
	}{
		{"unparseable policy config", func(f *fixture) {
			writeAtomic(f.t, filepath.Join(f.base, "stable.yaml"), []byte("mode: [unterminated\n"))
		}},
		{"invalid policy config value", func(f *fixture) {
			writeAtomic(f.t, filepath.Join(f.base, "stable.yaml"), []byte("mode: nonsense\nood_action: conservative\non_infeasible: reject\n"))
		}},
		{"manifest removed", func(f *fixture) {
			if err := os.Remove(filepath.Join(f.artifactDir(), "manifest.json")); err != nil {
				f.t.Fatal(err)
			}
		}},
		{"canary artifact missing", func(f *fixture) {
			ref := f.ref("stable.yaml")
			ref.Version = "ffffffffffff"
			c := *f.cfg.Load()
			c.Canary = &config.CanaryRef{PolicyRef: ref, Percent: 5}
			f.set(c)
		}},
		{"stable dropped", func(f *fixture) {
			c := *f.cfg.Load()
			c.Stable = config.PolicyRef{}
			f.set(c)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t)
			reg := f.registry()
			before := reg.Snapshot()
			tc.break_(f)
			if err := reg.Reload(); err == nil {
				t.Fatal("Reload succeeded")
			}
			if reg.Snapshot() != before {
				t.Error("the snapshot changed after a failed reload")
			}
			if reg.ReloadErrors() != 1 || reg.LastError() == nil {
				t.Errorf("errors = %d, last = %v", reg.ReloadErrors(), reg.LastError())
			}
		})
	}
}

func TestReloadKeepsOldWhenTheLoaderFails(t *testing.T) {
	f := newFixture(t)
	var fail atomic.Bool
	reg, err := NewRegistry(func() (config.PoliciesConfig, error) {
		if fail.Load() {
			return config.PoliciesConfig{}, errors.New("config no longer parses")
		}
		return f.load()
	}, f.base)
	if err != nil {
		t.Fatal(err)
	}
	before := reg.Snapshot()
	fail.Store(true)
	if err := reg.Reload(); err == nil || !strings.Contains(err.Error(), "config no longer parses") {
		t.Fatalf("Reload error = %v", err)
	}
	if reg.Snapshot() != before || reg.ReloadErrors() != 1 {
		t.Errorf("snapshot changed or error not counted: errors %d", reg.ReloadErrors())
	}
}

func TestReloadPicksUpAChange(t *testing.T) {
	f := newFixture(t)
	reg := f.registry()
	old := reg.Snapshot().Stable.Version

	f.writePolicy("stable.yaml", 0.4)
	if err := reg.Reload(); err != nil {
		t.Fatal(err)
	}
	snap := reg.Snapshot()
	if snap.Stable.Version == old || snap.Stable.Version != f.slotVersion("stable.yaml") {
		t.Errorf("version = %s, want the new %s (old %s)", snap.Stable.Version, f.slotVersion("stable.yaml"), old)
	}
	if snap.Generation != 2 {
		t.Errorf("generation = %d, want 2", snap.Generation)
	}
}

func TestRollbackStable(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("a.yaml", 0.1)
	f.writePolicy("b.yaml", 0.2)
	f.writePolicy("canary.yaml", 0.3)
	vA, vB := f.slotVersion("a.yaml"), f.slotVersion("b.yaml")

	f.set(config.PoliciesConfig{Root: "policies", Stable: f.ref("a.yaml")})
	reg := f.registry()

	if _, _, err := reg.RollbackStable(); err == nil {
		t.Fatal("RollbackStable succeeded with no previous stable slot")
	}

	f.set(config.PoliciesConfig{Root: "policies", Stable: f.ref("b.yaml")})
	if err := reg.Reload(); err != nil {
		t.Fatal(err)
	}
	// A reload that leaves the stable version alone must not replace the previous slot.
	f.set(config.PoliciesConfig{
		Root: "policies", Stable: f.ref("b.yaml"),
		Canary: &config.CanaryRef{PolicyRef: f.ref("canary.yaml"), Percent: 10},
	})
	if err := reg.Reload(); err != nil {
		t.Fatal(err)
	}
	genBefore := reg.Snapshot().Generation

	from, to, err := reg.RollbackStable()
	if err != nil {
		t.Fatalf("RollbackStable: %v", err)
	}
	if from != vB || to != vA {
		t.Errorf("RollbackStable = (%s, %s), want (%s, %s)", from, to, vB, vA)
	}
	snap := reg.Snapshot()
	if snap.Stable.Version != vA || snap.Stable.Role != "stable" {
		t.Errorf("stable after rollback = %s (role %s), want %s", snap.Stable.Version, snap.Stable.Role, vA)
	}
	if snap.Canary == nil || snap.CanaryPercent != 10 {
		t.Errorf("rollback dropped the canary: %+v", snap)
	}
	if snap.Generation != genBefore+1 {
		t.Errorf("generation = %d, want %d", snap.Generation, genBefore+1)
	}

	// Exactly one previous slot is kept, and rolling back spends it.
	if _, _, err := reg.RollbackStable(); err == nil {
		t.Error("a second RollbackStable succeeded, want an error (one previous slot only)")
	}
	if reg.Snapshot().Stable.Version != vA {
		t.Error("the failed rollback changed the stable slot")
	}
}

func TestSetCanaryPercent(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("canary.yaml", 0.3)
	c := *f.cfg.Load()
	c.Canary = &config.CanaryRef{PolicyRef: f.ref("canary.yaml"), Percent: 10}
	f.set(c)
	reg := f.registry()
	gen := reg.Snapshot().Generation

	if old := reg.SetCanaryPercent(25); old != 10 {
		t.Errorf("SetCanaryPercent returned %v, want the old 10", old)
	}
	snap := reg.Snapshot()
	if snap.CanaryPercent != 25 || snap.Generation != gen+1 {
		t.Errorf("percent %v generation %d, want 25 and %d", snap.CanaryPercent, snap.Generation, gen+1)
	}
	if snap.Canary == nil || snap.Stable.Version == "" {
		t.Errorf("slots lost: %+v", snap)
	}
	if old := reg.SetCanaryPercent(500); old != 25 || reg.Snapshot().CanaryPercent != 100 {
		t.Errorf("above range: old %v, now %v, want 25 and 100", old, reg.Snapshot().CanaryPercent)
	}
	if reg.SetCanaryPercent(-3); reg.Snapshot().CanaryPercent != 0 {
		t.Errorf("below range: now %v, want 0", reg.Snapshot().CanaryPercent)
	}
}

// watchConfigFile writes a RoutePilot config whose policies section names the stable
// policy file, and returns its path.
func (f *fixture) writeConfigFile(stablePolicy string) string {
	f.t.Helper()
	y := fmt.Sprintf(`providers:
  - name: p
    type: openaicompat
    base_url: http://127.0.0.1:1/v1
models:
  - id: tiny-weak
    deployments:
      - provider: p
        upstream_model: w
tenants:
  - id: t
    api_key_sha256: %s
    allowed_models: ["*"]
routing:
  policy: learned
policies:
  root: policies
  stable:
    name: tiny
    version: %s
    config: %s
`, strings.Repeat("a", 64), f.version, stablePolicy)
	path := filepath.Join(f.base, "routepilot.yaml")
	writeAtomic(f.t, path, []byte(y))
	return path
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestWatchReloadsOnChange(t *testing.T) {
	f := newFixture(t)
	f.writePolicy("other.yaml", 0.9)
	cfgPath := f.writeConfigFile("stable.yaml")
	load := func() (config.PoliciesConfig, error) {
		c, err := config.Load(cfgPath)
		if err != nil {
			return config.PoliciesConfig{}, err
		}
		return c.Policies, nil
	}
	loaded, err := config.Load(cfgPath)
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	reg, err := NewRegistry(load, loaded.BaseDir())
	if err != nil {
		t.Fatal(err)
	}
	reg.SetConfigFile(cfgPath)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); reg.Watch(ctx, 5*time.Millisecond) }()
	t.Cleanup(func() { cancel(); <-done })

	// Nothing changed: no reload however many polls pass.
	time.Sleep(60 * time.Millisecond)
	if g := reg.Snapshot().Generation; g != 1 {
		t.Fatalf("generation = %d after idle polls, want 1", g)
	}

	// 1. The policy configuration file changes.
	f.writePolicy("stable.yaml", 0.4)
	want := f.slotVersion("stable.yaml")
	eventually(t, "reload after the policy config changed", func() bool { return reg.Snapshot().Stable.Version == want })

	// 2. An artifact file is corrupted: the old policy keeps serving, the error is counted
	// once and the poller does not retry the same bytes.
	weights := filepath.Join(f.artifactDir(), "weights.bin")
	good, err := os.ReadFile(weights)
	if err != nil {
		t.Fatal(err)
	}
	errsBefore := reg.ReloadErrors()
	writeAtomic(t, weights, good[:len(good)/3])
	eventually(t, "the error from a truncated weights.bin", func() bool { return reg.ReloadErrors() == errsBefore+1 })
	time.Sleep(80 * time.Millisecond)
	if got := reg.ReloadErrors(); got != errsBefore+1 {
		t.Errorf("ReloadErrors() = %d after more polls, want it to stay %d", got, errsBefore+1)
	}
	if got := reg.Snapshot().Stable.Version; got != want {
		t.Errorf("served version = %s while the artifact is corrupt, want %s", got, want)
	}

	// 3. The artifact is repaired.
	gen := reg.Snapshot().Generation
	writeAtomic(t, weights, good)
	eventually(t, "reload after the repair", func() bool { return reg.Snapshot().Generation > gen && reg.LastError() == nil })

	// 4. The config file itself changes to name another policy file.
	f.writeConfigFile("other.yaml")
	want = f.slotVersion("other.yaml")
	eventually(t, "reload after the config file changed", func() bool { return reg.Snapshot().Stable.Version == want })

	// Cancelling stops the loop.
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Watch did not return after its context was canceled")
	}
}
