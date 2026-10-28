package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/provider"
)

func adminDo(h http.Handler, method, path, key string) *httptest.ResponseRecorder {
	return do(h, method, path, key, "")
}

// adminSnapshot decodes the JSON the admin routes answer with.
type adminView struct {
	StableVersion  string    `json:"stable_version"`
	CanaryVersion  *string   `json:"canary_version"`
	ShadowVersions []string  `json:"shadow_versions"`
	CanaryPercent  float64   `json:"canary_percent"`
	Generation     uint64    `json:"generation"`
	LoadedAt       time.Time `json:"loaded_at"`
	ReloadErrors   uint64    `json:"reload_errors"`
	LastError      *string   `json:"last_error"`
}

func decodeAdmin(t *testing.T, rec *httptest.ResponseRecorder) adminView {
	t.Helper()
	var v adminView
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatalf("admin response is not the snapshot: %v\n%s", err, rec.Body.String())
	}
	return v
}

// TestAdminRequiresKey: without server.admin_key_sha256 the routes do not exist; with it, only
// the admin key opens them, and a tenant key does not.
func TestAdminRequiresKey(t *testing.T) {
	routes := []struct{ method, path string }{
		{http.MethodGet, "/admin/policies"},
		{http.MethodPost, "/admin/policies/rollback"},
	}

	t.Run("not configured answers 404 to everyone", func(t *testing.T) {
		e := newLearnedEnv(t, learnedOpts{})
		for _, r := range routes {
			for _, key := range []string{"", adminKey, keyA, keyHash(adminKey)} {
				rec := adminDo(e.gw.Handler(), r.method, r.path, key)
				if rec.Code != http.StatusNotFound || decodeErr(t, rec).Error.Code != "not_found" {
					t.Errorf("%s %s key %q: status %d body %s, want 404 not_found", r.method, r.path, key, rec.Code, rec.Body.String())
				}
				if want := "no route for " + r.path; decodeErr(t, rec).Error.Message != want {
					t.Errorf("%s %s: message %q, want the unknown-route message %q", r.method, r.path, decodeErr(t, rec).Error.Message, want)
				}
			}
		}
		if s := e.reg.Snapshot(); s.Generation != 1 {
			t.Errorf("generation = %d: a refused request changed the policies", s.Generation)
		}
	})

	t.Run("configured", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 10}, nil)
		h := g.gw.Handler()
		for _, r := range routes {
			for name, key := range map[string]string{
				"no key":                 "",
				"a wrong key":            "test-admin-key-0002",
				"a tenant key":           keyA,
				"the other tenant":       keyB,
				"the hash from the file": keyHash(adminKey),
				"a prefix of the key":    adminKey[:len(adminKey)-1],
				"the key and more":       adminKey + "x",
			} {
				rec := adminDo(h, r.method, r.path, key)
				if rec.Code != http.StatusUnauthorized || decodeErr(t, rec).Error.Code != "invalid_api_key" {
					t.Errorf("%s %s with %s: status %d body %s, want 401 invalid_api_key", r.method, r.path, name, rec.Code, rec.Body.String())
				}
				if rec.Header().Get("WWW-Authenticate") != "Bearer" {
					t.Errorf("%s %s with %s: no WWW-Authenticate header", r.method, r.path, name)
				}
			}
		}
		if s := g.reg.Snapshot(); s.CanaryPercent != 10 || s.Generation != 1 {
			t.Errorf("snapshot = percent %v generation %d: a refused request changed the policies", s.CanaryPercent, s.Generation)
		}
		if rec := adminDo(h, http.MethodGet, "/admin/policies", adminKey); rec.Code != http.StatusOK {
			t.Errorf("GET with the admin key: status %d: %s", rec.Code, rec.Body.String())
		}
		// The scheme is case-insensitive, like the tenants'.
		req := httptest.NewRequest(http.MethodGet, "/admin/policies", nil)
		req.Header.Set("Authorization", "bearer "+adminKey)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("GET with a lowercase scheme: status %d", rec.Code)
		}
	})

	t.Run("methods", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0, 0, "best_effort"), canaryPct: 10}, nil)
		h := g.gw.Handler()
		for _, c := range []struct{ method, path, allow string }{
			{http.MethodPost, "/admin/policies", "GET"},
			{http.MethodPut, "/admin/policies", "GET"},
			{http.MethodGet, "/admin/policies/rollback", "POST"},
			{http.MethodDelete, "/admin/policies/rollback", "POST"},
		} {
			rec := adminDo(h, c.method, c.path, adminKey)
			if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != c.allow {
				t.Errorf("%s %s: status %d Allow %q, want 405 and %q", c.method, c.path, rec.Code, rec.Header().Get("Allow"), c.allow)
			}
		}
		if s := g.reg.Snapshot(); s.CanaryPercent != 10 {
			t.Errorf("canary percent = %v: a wrong-method request changed the policies", s.CanaryPercent)
		}
	})
}

func TestNewRejectsABadAdminKeyHash(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	cfg.Server.AdminKeySHA256 = "not-a-hash"
	if _, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{}); err == nil {
		t.Fatal("New accepted a malformed admin key hash")
	} else if strings.Contains(err.Error(), "not-a-hash") {
		t.Errorf("the error repeats the configured value: %v", err)
	}
}

func TestAdminNeedsALearnedPolicy(t *testing.T) {
	cfg := buildConfig(t, cfgOpts{})
	cfg.Server.AdminKeySHA256 = keyHash(adminKey)
	gw, err := New(cfg, map[string]provider.Provider{"mockp": &fakeProvider{}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []struct{ method, path string }{{http.MethodGet, "/admin/policies"}, {http.MethodPost, "/admin/policies/rollback"}} {
		rec := adminDo(gw.Handler(), r.method, r.path, adminKey)
		if rec.Code != http.StatusNotFound || !strings.Contains(decodeErr(t, rec).Error.Message, "learned") {
			t.Errorf("%s %s: status %d body %s, want 404 saying the policy is not learned", r.method, r.path, rec.Code, rec.Body.String())
		}
	}
	if rec := adminDo(gw.Handler(), http.MethodGet, "/admin/policies", keyA); rec.Code != http.StatusUnauthorized {
		t.Errorf("a tenant key on a static gateway: status %d, want 401", rec.Code)
	}
}

func TestAdminSnapshot(t *testing.T) {
	g := newGuardEnv(t, learnedOpts{
		canary: policyYAML(0.2, 0, "best_effort"), canaryPct: 12.5,
		shadow: []string{policyYAML(0.3, 0, "best_effort"), policyYAML(0.4, 0, "best_effort")},
	}, nil)
	h := g.gw.Handler()
	snap := g.reg.Snapshot()

	rec := adminDo(h, http.MethodGet, "/admin/policies", adminKey)
	if rec.Code != http.StatusOK || !strings.HasPrefix(rec.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("status %d content type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	if rec.Header().Get("Cache-Control") != "no-store" {
		t.Errorf("Cache-Control = %q, want no-store", rec.Header().Get("Cache-Control"))
	}
	v := decodeAdmin(t, rec)
	if v.StableVersion != snap.Stable.Version || v.CanaryVersion == nil || *v.CanaryVersion != snap.Canary.Version {
		t.Errorf("versions = stable %q canary %v, want %q and %q", v.StableVersion, v.CanaryVersion, snap.Stable.Version, snap.Canary.Version)
	}
	if len(v.ShadowVersions) != 2 || v.ShadowVersions[0] != snap.Shadow[0].Version || v.ShadowVersions[1] != snap.Shadow[1].Version {
		t.Errorf("shadow versions = %v, want %q and %q", v.ShadowVersions, snap.Shadow[0].Version, snap.Shadow[1].Version)
	}
	if v.CanaryPercent != 12.5 || v.Generation != snap.Generation || !v.LoadedAt.Equal(snap.LoadedAt) {
		t.Errorf("percent %v generation %d loaded_at %v, want 12.5, %d, %v", v.CanaryPercent, v.Generation, v.LoadedAt, snap.Generation, snap.LoadedAt)
	}
	if v.ReloadErrors != 0 || v.LastError != nil {
		t.Errorf("reload errors %d last error %v, want none", v.ReloadErrors, v.LastError)
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"stable_version", "canary_version", "shadow_versions", "canary_percent", "generation", "loaded_at", "reload_errors", "last_error"} {
		if _, ok := raw[key]; !ok {
			t.Errorf("the snapshot has no %q: %s", key, rec.Body.String())
		}
	}

	// A reload that fails is reported, and the old policies keep serving.
	if err := os.WriteFile(filepath.Join(g.dir, "canary.yaml"), []byte("mode: nonsense\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := g.reg.Reload(); err == nil {
		t.Fatal("Reload accepted a broken policy configuration")
	}
	v = decodeAdmin(t, adminDo(h, http.MethodGet, "/admin/policies", adminKey))
	if v.ReloadErrors != 1 || v.LastError == nil || *v.LastError == "" {
		t.Errorf("reload errors %d last error %v, want 1 and the message", v.ReloadErrors, v.LastError)
	}
	if v.Generation != snap.Generation || v.StableVersion != snap.Stable.Version {
		t.Errorf("generation %d stable %q after a failed reload, want the old %d and %q", v.Generation, v.StableVersion, snap.Generation, snap.Stable.Version)
	}

	// No canary: its version is null.
	plain := newGuardEnv(t, learnedOpts{}, nil)
	var rawPlain map[string]any
	if err := json.Unmarshal(adminDo(plain.gw.Handler(), http.MethodGet, "/admin/policies", adminKey).Body.Bytes(), &rawPlain); err != nil {
		t.Fatal(err)
	}
	if rawPlain["canary_version"] != nil || rawPlain["canary_percent"] != float64(0) {
		t.Errorf("canary_version %v canary_percent %v without a canary, want null and 0", rawPlain["canary_version"], rawPlain["canary_percent"])
	}
	if sv, ok := rawPlain["shadow_versions"].([]any); !ok || len(sv) != 0 {
		t.Errorf("shadow_versions = %v without shadows, want an empty list", rawPlain["shadow_versions"])
	}
}

// TestAdminRollback: with a canary taking traffic the rollback stops it; with none it puts the
// previous stable policy back. Each writes a rollback event by the admin and answers with the
// new snapshot.
func TestAdminRollback(t *testing.T) {
	t.Run("the canary first", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0.2, 0, "best_effort"), canaryPct: 25}, nil)
		h := g.gw.Handler()
		before := g.reg.Snapshot()

		rec := adminDo(h, http.MethodPost, "/admin/policies/rollback", adminKey)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		v := decodeAdmin(t, rec)
		if v.CanaryPercent != 0 || v.StableVersion != before.Stable.Version || v.CanaryVersion == nil || *v.CanaryVersion != before.Canary.Version {
			t.Errorf("snapshot = %+v, want canary percent 0 and the slots kept", v)
		}
		if after := g.reg.Snapshot(); after.CanaryPercent != 0 || v.Generation != after.Generation {
			t.Errorf("registry percent %v generation %d, response generation %d", after.CanaryPercent, after.Generation, v.Generation)
		}
		events := g.store.got()
		if len(events) != 1 {
			t.Fatalf("%d rollout events, want 1: %+v", len(events), events)
		}
		ev := events[0]
		if ev.Action != accounting.ActionRollback || ev.Actor != "admin" || ev.Percent != 0 ||
			ev.From != before.Canary.Version || ev.To != before.Stable.Version || ev.At.IsZero() || ev.Reason == "" {
			t.Errorf("event = %+v, want rollback by admin from the canary to stable at 0 percent, with a reason", ev)
		}
		if n := g.reg.Rollbacks(); n != 1 {
			t.Errorf("Rollbacks() = %d, want 1", n)
		}

		// Nothing left to roll back: the canary has no share and stable has no predecessor.
		rec = adminDo(h, http.MethodPost, "/admin/policies/rollback", adminKey)
		if rec.Code != http.StatusConflict || decodeErr(t, rec).Error.Code != "no_rollback_target" {
			t.Errorf("second rollback: status %d body %s, want 409 no_rollback_target", rec.Code, rec.Body.String())
		}
		if n := len(g.store.got()); n != 1 {
			t.Errorf("%d events after a refused rollback, want still 1", n)
		}
		if n := g.reg.Rollbacks(); n != 1 {
			t.Errorf("Rollbacks() = %d after a refused rollback, want still 1", n)
		}
		if strings.Contains(g.logs.String(), adminKey) {
			t.Error("the admin key is in the log")
		}
		if !strings.Contains(g.logs.String(), "rolled back through the admin API") {
			t.Errorf("the rollback is not logged:\n%s", g.logs.String())
		}
	})

	t.Run("then the stable policy", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{policy: policyYAML(0.1, 0, "best_effort")}, nil)
		h := g.gw.Handler()
		first := g.reg.Snapshot().Stable.Version

		// A reload brings in a new stable configuration.
		if err := os.WriteFile(filepath.Join(g.dir, "stable.yaml"), []byte(policyYAML(0.3, 0, "best_effort")), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := g.reg.Reload(); err != nil {
			t.Fatal(err)
		}
		second := g.reg.Snapshot().Stable.Version
		if second == first {
			t.Fatal("the reload did not change the stable version")
		}

		rec := adminDo(h, http.MethodPost, "/admin/policies/rollback", adminKey)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		if v := decodeAdmin(t, rec); v.StableVersion != first {
			t.Errorf("stable version = %q, want the previous %q", v.StableVersion, first)
		}
		events := g.store.got()
		if len(events) != 1 || events[0].Action != accounting.ActionRollback || events[0].Actor != "admin" ||
			events[0].From != second || events[0].To != first {
			t.Errorf("events = %+v, want one rollback by admin from %q to %q", events, second, first)
		}
		if n := g.reg.Rollbacks(); n != 1 {
			t.Errorf("Rollbacks() = %d, want 1", n)
		}
		// GET shows the same state.
		if v := decodeAdmin(t, adminDo(h, http.MethodGet, "/admin/policies", adminKey)); v.StableVersion != first {
			t.Errorf("GET after the rollback shows stable %q, want %q", v.StableVersion, first)
		}
	})

	t.Run("a canary with no share rolls the stable policy back", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0.2, 0, "best_effort"), canaryPct: 0}, nil)
		rec := adminDo(g.gw.Handler(), http.MethodPost, "/admin/policies/rollback", adminKey)
		if rec.Code != http.StatusConflict {
			t.Errorf("status %d, want 409: there is no previous stable policy and no canary share", rec.Code)
		}
	})

	t.Run("without a ledger the rollback still happens", func(t *testing.T) {
		g := newGuardEnv(t, learnedOpts{canary: policyYAML(0.2, 0, "best_effort"), canaryPct: 25}, nil)
		g.gw.usageLedger = nil
		rec := adminDo(g.gw.Handler(), http.MethodPost, "/admin/policies/rollback", adminKey)
		if rec.Code != http.StatusOK || g.reg.Snapshot().CanaryPercent != 0 {
			t.Errorf("status %d percent %v, want 200 and 0", rec.Code, g.reg.Snapshot().CanaryPercent)
		}
	})
}
