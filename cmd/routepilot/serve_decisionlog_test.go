package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/mock"
)

// TestServeWritesTheDecisionLog: serve opens the decision log the config names, a request
// and a feedback call land in it, and shutdown drains the queue after the server has stopped,
// so the last records are on disk when serve returns.
func TestServeWritesTheDecisionLog(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	path := filepath.Join(t.TempDir(), "logs", "decisions.jsonl")
	cfg := accountingConfig(t, upstream.URL, fmt.Sprintf("decision_log:\n  path: %q\n", filepath.ToSlash(path)))
	url, stop := startBudgetServe(t, cfg)

	req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer test-key-free")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	id := resp.Header.Get("x-routepilot-decision-id")
	if resp.StatusCode != http.StatusOK || id == "" {
		t.Fatalf("status %d decision id %q", resp.StatusCode, id)
	}

	req, _ = http.NewRequest(http.MethodPost, url+"/v1/feedback", strings.NewReader(fmt.Sprintf(`{"decision_id":%q,"success":true}`, id)))
	req.Header.Set("Authorization", "Bearer test-key-free")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("feedback status %d, want 204", resp.StatusCode)
	}
	stop()

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("decision log not written: %v", err)
	}
	defer f.Close()
	var types []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("line is not JSON: %v: %s", err, sc.Text())
		}
		if m["decision_id"] != id {
			t.Errorf("record %v has decision_id %v, want %q", m["type"], m["decision_id"], id)
		}
		types = append(types, m["type"].(string))
	}
	if fmt.Sprint(types) != "[decision feedback]" {
		t.Errorf("record types = %v, want [decision feedback]", types)
	}
}

func TestServeWithoutADecisionLogHasNoFeedbackRoute(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	url, stop := startBudgetServe(t, accountingConfig(t, upstream.URL, ""))
	defer stop()

	req, _ := http.NewRequest(http.MethodPost, url+"/v1/feedback", strings.NewReader(`{"decision_id":"dec_abcdefgh","success":true}`))
	req.Header.Set("Authorization", "Bearer test-key-free")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("feedback status %d, want 404 without a decision log", resp.StatusCode)
	}
}

func TestServeFailsWhenTheDecisionLogCannotBeOpened(t *testing.T) {
	upstream := httptest.NewServer(mock.NewHandler(mock.Config{}))
	t.Cleanup(upstream.Close)
	notADir := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(notADir, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := accountingConfig(t, upstream.URL, fmt.Sprintf("decision_log:\n  path: %q\n", filepath.ToSlash(filepath.Join(notADir, "d.jsonl"))))

	err := serve(context.Background(), cfg, listenLocal(t), slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err == nil || !strings.Contains(err.Error(), "decision log") {
		t.Errorf("serve() = %v, want a decision log startup error", err)
	}
}
