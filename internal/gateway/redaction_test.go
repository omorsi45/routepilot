package gateway

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/omorsi45/routepilot/internal/decisionlog"
)

// redactionSentinel is planted in every place a request can carry text. If it turns up in
// anything the gateway writes down, a prompt has leaked.
const redactionSentinel = "SENTINEL-PROMPT-1d9c"

// secretsIn returns which of the secrets appear in output: the sentinel and every 12-byte
// piece of it, the API key, its SHA-256 and the first 16 hex digits of that hash.
func secretsIn(output string) []string {
	secrets := []string{redactionSentinel, keyA, keyHash(keyA), keyHash(keyA)[:16]}
	for i := 0; i+12 <= len(redactionSentinel); i++ {
		secrets = append(secrets, redactionSentinel[i:i+12])
	}
	var found []string
	for _, s := range secrets {
		if strings.Contains(output, s) {
			found = append(found, s)
		}
	}
	return found
}

// redactionBody is a request whose system message, user message, message name, tool-call
// argument and user field all carry the sentinel.
func redactionBody(model string, stream bool) string {
	s := redactionSentinel
	return fmt.Sprintf(`{"model":%q,"stream":%v,"user":%q,"messages":[
		{"role":"system","content":%q},
		{"role":"user","name":%q,"content":%q},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":%q}}]},
		{"role":"tool","tool_call_id":"call_1","content":"done"}],
		"tools":[{"type":"function","function":{"name":"lookup","description":"find things","parameters":{"type":"object"}}}]}`,
		model, stream, s+"-user-field", s+"-system", s+"-name", s+"-user", `{"q":"`+s+`-args"}`)
}

// dumpSpans writes down everything an exported span holds: names, attributes, status,
// events, links, scope and resource.
func dumpSpans(spans tracetest.SpanStubs) string {
	var sb strings.Builder
	for _, s := range spans {
		fmt.Fprintf(&sb, "span %s kind=%v trace=%s id=%s parent=%s scope=%s/%s status=%v:%s\n",
			s.Name, s.SpanKind, s.SpanContext.TraceID(), s.SpanContext.SpanID(), s.Parent.SpanID(),
			s.InstrumentationScope.Name, s.InstrumentationScope.Version, s.Status.Code, s.Status.Description)
		for _, kv := range s.Attributes {
			fmt.Fprintf(&sb, "  attr %s=%s\n", kv.Key, kv.Value.Emit())
		}
		for _, ev := range s.Events {
			fmt.Fprintf(&sb, "  event %s\n", ev.Name)
			for _, kv := range ev.Attributes {
				fmt.Fprintf(&sb, "    attr %s=%s\n", kv.Key, kv.Value.Emit())
			}
		}
		for _, l := range s.Links {
			fmt.Fprintf(&sb, "  link %v\n", l)
		}
		if s.Resource != nil {
			fmt.Fprintf(&sb, "  resource %v\n", s.Resource.Attributes())
		}
		fmt.Fprintf(&sb, "  raw %+v\n", s)
	}
	return sb.String()
}

// TestNoSecretsOrPromptsAnywhere sends a non-streaming and a streaming request that carry
// the sentinel and the key through a gateway that logs, traces, counts, records decisions
// and keeps a usage ledger, and checks that none of what was written down holds the
// sentinel, the key or the key's hash.
func TestNoSecretsOrPromptsAnywhere(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{
		canary: policyYAML(0.1, 0, "best_effort"), canaryPct: 50,
		shadow: []string{policyYAML(0.2, 0, "best_effort")},
	})
	// Every feature on: tools, a budget, rate limits and a concurrency limit, so the
	// admission, budget and ledger code all handle the request.
	for i := range e.cfg.Models {
		e.cfg.Models[i].Capabilities = []string{"streaming", "tools"}
	}
	tn := &e.cfg.Tenants[0]
	tn.BudgetUSD, tn.BudgetPeriod = "10", "monthly"
	tn.RPM, tn.RPMBurst, tn.MaxConcurrency = 6000, 100, 8

	logs := &syncLog{}
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(t.Context()) })
	logPath := filepath.Join(t.TempDir(), "decisions.jsonl")
	dlog, err := decisionlog.Open(logPath, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dlog.Close() })
	store := &recordingStore{}
	gw := rebuild(t, e, Options{
		Logger:         slog.New(slog.NewJSONHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})),
		TracerProvider: tp,
		DecisionLog:    dlog,
		Accounting:     store,
	})
	t.Cleanup(func() { closeUsage(t, gw) })

	// One request that fails before the body is read, and one of each kind that is served.
	if rec := doHeaders(gw.Handler(), "test-key-wrong", redactionBody("auto", false), nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key: status %d", rec.Code)
	}
	for _, stream := range []bool{false, true} {
		rec := doHeaders(gw.Handler(), keyA, redactionBody("auto", stream), nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("stream=%v: status %d: %s", stream, rec.Code, rec.Body.String())
		}
	}
	if rec := doHeaders(gw.Handler(), keyA, redactionBody("tiny-weak", false), nil); rec.Code != http.StatusOK {
		t.Fatalf("pinned: status %d: %s", rec.Code, rec.Body.String())
	}

	if err := dlog.Close(); err != nil {
		t.Fatal(err)
	}
	flushUsage(t, gw)
	decisions, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	scrape := do(gw.Handler(), http.MethodGet, "/metrics", "", "").Body.String()
	var ledger strings.Builder
	for _, row := range store.got() {
		fmt.Fprintf(&ledger, "%+v\n", row)
	}

	outputs := []struct {
		name, text, marker string // a marker that proves the output holds what it should
	}{
		{"slog JSON", logs.String(), `"msg":"request"`},
		{"spans", dumpSpans(exp.GetSpans()), "span chat.completions"},
		{"metrics", scrape, "rp_requests_total"},
		{"decision log", string(decisions), `"type":"decision"`},
		{"usage ledger", ledger.String(), "Attempt:1"},
	}
	for _, o := range outputs {
		if !strings.Contains(o.text, o.marker) {
			t.Errorf("%s does not hold %q, so checking it proves nothing:\n%s", o.name, o.marker, o.text)
		}
		if found := secretsIn(o.text); len(found) > 0 {
			t.Errorf("%s contains %q:\n%s", o.name, found, o.text)
		}
	}

	// The three kinds of requests that were served each left their spans.
	if n := strings.Count(outputs[1].text, "span chat.completions"); n != 4 {
		t.Errorf("%d root spans, want 4 (one refused, three served)", n)
	}
	if n := strings.Count(outputs[3].text, `"type":"decision"`); n < 3 {
		t.Errorf("%d decision records, want at least one per served request", n)
	}
}

// TestSecretsInFindsWhatIsPlanted: the detector the test above relies on does see each
// kind of secret, including a piece of the sentinel and the start of the key hash.
func TestSecretsInFindsWhatIsPlanted(t *testing.T) {
	for name, text := range map[string]string{
		"sentinel":      "prompt: " + redactionSentinel,
		"sentinel part": "x " + redactionSentinel[3:17] + " y",
		"key":           "Authorization: Bearer " + keyA,
		"key hash":      "tenant " + keyHash(keyA),
		"key hash part": "k=" + keyHash(keyA)[:16],
	} {
		if len(secretsIn(text)) == 0 {
			t.Errorf("%s: not detected in %q", name, text)
		}
	}
	for _, clean := range []string{"", `{"msg":"request","tenant":"tenant-a","prompt_bytes":123}`, "rp_requests_total{tenant=\"tenant-a\"} 1"} {
		if found := secretsIn(clean); len(found) != 0 {
			t.Errorf("%q flagged as holding %q", clean, found)
		}
	}
}
