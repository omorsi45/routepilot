package config

import (
	"strings"
	"testing"
)

// adminYAML is the base config with server.admin_key_sha256 set to value. The hash is computed
// by the caller from a fake key: no key hash is committed.
func adminYAML(t *testing.T, value string) string {
	t.Helper()
	return baseYAML(t, "max_timeout_ms: 2000\n", "max_timeout_ms: 2000\n  admin_key_sha256: "+value+"\n")
}

func TestAdminKeyIsOffByDefault(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Server.AdminKeySHA256 != "" {
		t.Errorf("AdminKeySHA256 = %q, want empty: the admin routes are off unless a key is configured", cfg.Server.AdminKeySHA256)
	}
}

func TestAdminKeyParses(t *testing.T) {
	want := keyHash("test-admin-key")
	cfg, err := Parse([]byte(adminYAML(t, want)))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Server.AdminKeySHA256 != want {
		t.Errorf("AdminKeySHA256 = %q, want %q", cfg.Server.AdminKeySHA256, want)
	}
}

func TestAdminKeyRejects(t *testing.T) {
	good := keyHash("test-admin-key")
	for _, c := range []struct{ name, value string }{
		{"too short", good[:63]},
		{"too long", good + "0"},
		{"uppercase", strings.ToUpper(good)},
		{"not hex", strings.Repeat("g", 64)},
		{"a tenant's key hash", keyHash("test-key-tenant-a")},
	} {
		_, err := Parse([]byte(adminYAML(t, c.value)))
		if err == nil {
			t.Errorf("%s: Parse accepted %q", c.name, c.value)
			continue
		}
		if !strings.Contains(err.Error(), "server.admin_key_sha256") {
			t.Errorf("%s: error = %v, want it to name server.admin_key_sha256", c.name, err)
		}
		if strings.Contains(err.Error(), c.value) && c.value != "" {
			t.Errorf("%s: the error repeats the configured value: %v", c.name, err)
		}
	}
}
