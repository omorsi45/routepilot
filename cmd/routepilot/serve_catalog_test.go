package main

import (
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
)

// TestServeRefusesAPolicyThatCoversNoCatalogModel: the committed tiny artifact predicts for
// tiny-weak and tiny-strong, so a catalog of other models would answer every automatic
// request with a 400. serve refuses to start instead.
func TestServeRefusesAPolicyThatCoversNoCatalogModel(t *testing.T) {
	f := newServeFixture(t, "")
	raw, err := os.ReadFile(f.cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	other := strings.NewReplacer("tiny-weak", "cheap-a", "tiny-strong", "top-c").Replace(string(raw))
	// The policy reference "name: tiny" stays; only the model ids change.
	if err := os.WriteFile(f.cfgPath, []byte(other), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(f.cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serve(ctx, cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil))) }()

	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "none of the catalog's models") {
			t.Fatalf("serve() = %v, want a startup error saying the policy covers none of the catalog's models", err)
		}
	case <-time.After(5 * time.Second):
		cancel()
		t.Fatal("serve() started with a policy that covers none of the catalog's models")
	}
}
