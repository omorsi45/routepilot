package policy

import (
	"log/slog"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

// foreignArtifact copies the tiny artifact under policies/foreign/<version> with its models
// renamed, so it covers none of the models the tiny artifact covers, and returns the
// reference to it.
func (f *fixture) foreignArtifact() config.PolicyRef {
	f.t.Helper()
	d := readDoc(f.t, f.srcDir)
	rename := map[string]string{"tiny-weak": "foreign-weak", "tiny-strong": "foreign-strong"}
	for i, m := range objectAt(f.t, d, "models").([]any) {
		id := objectAt(f.t, m, "id").(string)
		setAt(f.t, d, rename[id], "models", i, "id")
	}
	for i, g := range objectAt(f.t, d, "golden").([]any) {
		models := objectAt(f.t, g, "models").(map[string]any)
		renamed := make(map[string]any, len(models))
		for id, v := range models {
			renamed[rename[id]] = v
		}
		setAt(f.t, d, renamed, "golden", i, "models")
	}
	src := rewrite(f.t, f.srcDir, d)
	version := filepath.Base(src)
	copyDir(f.t, src, filepath.Join(f.base, "policies", "foreign", version))
	return config.PolicyRef{Name: "foreign", Version: version, Config: "stable.yaml"}
}

func (f *fixture) registryFor(catalog ...string) (*Registry, error) {
	return NewRegistry(f.load, f.base, WithCatalog(catalog))
}

// TestNewRegistryRefusesASlotThatCoversNoCatalogModel: an artifact trained for other models
// than the catalog's would answer every automatic request with a capability error, so each
// slot is refused when none of the catalog's models is one of its models.
func TestNewRegistryRefusesASlotThatCoversNoCatalogModel(t *testing.T) {
	cases := []struct {
		name  string
		slot  string
		apply func(f *fixture, c *config.PoliciesConfig, foreign config.PolicyRef)
	}{
		{"stable", "stable", func(_ *fixture, c *config.PoliciesConfig, foreign config.PolicyRef) { c.Stable = foreign }},
		{"canary", "canary", func(_ *fixture, c *config.PoliciesConfig, foreign config.PolicyRef) {
			c.Canary = &config.CanaryRef{PolicyRef: foreign, Percent: 5}
		}},
		{"shadow", "shadow[0]", func(_ *fixture, c *config.PoliciesConfig, foreign config.PolicyRef) {
			c.Shadow = []config.PolicyRef{foreign}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t)
			c := *f.cfg.Load()
			tc.apply(f, &c, f.foreignArtifact())
			f.set(c)

			reg, err := f.registryFor("tiny-weak", "tiny-strong")

			if err == nil {
				t.Fatalf("NewRegistry accepted a %s artifact that covers none of the catalog: %+v", tc.name, reg.Snapshot())
			}
			for _, want := range []string{tc.slot, "none of the catalog's models", "foreign-weak"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// TestReloadKeepsTheOldPolicyWhenAnArtifactCoversNoCatalogModel: pointing the stable slot at
// an artifact of another model pool is a failed reload, and the old policy keeps serving.
func TestReloadKeepsTheOldPolicyWhenAnArtifactCoversNoCatalogModel(t *testing.T) {
	f := newFixture(t)
	reg, err := f.registryFor("tiny-weak", "tiny-strong")
	if err != nil {
		t.Fatal(err)
	}
	before := reg.Snapshot()

	c := *f.cfg.Load()
	c.Stable = f.foreignArtifact()
	f.set(c)
	err = reg.Reload()

	if err == nil || !strings.Contains(err.Error(), "none of the catalog's models") {
		t.Fatalf("Reload = %v, want the coverage error", err)
	}
	if got := reg.ReloadErrors(); got != 1 {
		t.Errorf("ReloadErrors() = %d, want 1", got)
	}
	if reg.Snapshot() != before {
		t.Error("the snapshot changed after a refused reload")
	}
	if got := reg.Snapshot().Stable.Version; got != f.slotVersion("stable.yaml") {
		t.Errorf("served version = %s, want the old one", got)
	}
}

// TestRegistryWarnsAboutCatalogModelsASlotDoesNotCover: partial coverage loads, because a
// model the artifact does not know is simply never chosen automatically, and says which ones.
func TestRegistryWarnsAboutCatalogModelsASlotDoesNotCover(t *testing.T) {
	f := newFixture(t)
	var logs strings.Builder
	logger := slog.New(slog.NewTextHandler(&logs, nil))

	reg, err := NewRegistry(f.load, f.base, WithCatalog([]string{"tiny-weak", "mid-b"}), WithLogger(logger))

	if err != nil {
		t.Fatalf("NewRegistry with partial coverage = %v, want it to load", err)
	}
	if reg.Snapshot().Stable.Role != RoleStable {
		t.Error("the stable slot did not load")
	}
	out := logs.String()
	if !strings.Contains(out, "mid-b") || !strings.Contains(out, "stable") {
		t.Errorf("log %q does not name the uncovered model mid-b and the slot", out)
	}
	if strings.Contains(out, "tiny-weak") {
		t.Errorf("log %q names tiny-weak, which the artifact covers", out)
	}
}

// TestRegistryWithoutACatalogChecksNoCoverage keeps the registry usable on its own: without
// the catalog's model ids there is nothing to compare an artifact with.
func TestRegistryWithoutACatalogChecksNoCoverage(t *testing.T) {
	f := newFixture(t)
	c := *f.cfg.Load()
	c.Stable = f.foreignArtifact()
	f.set(c)

	if _, err := NewRegistry(f.load, f.base); err != nil {
		t.Fatalf("NewRegistry without a catalog = %v, want it to load", err)
	}
	if _, err := NewRegistry(f.load, f.base, WithCatalog(nil)); err != nil {
		t.Fatalf("NewRegistry with an empty catalog = %v, want it to load", err)
	}
}
