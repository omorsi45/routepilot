package gateway

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/decisionlog"
)

// logSink is a decision log attached to a gateway, read back whole.
type logSink struct {
	t    *testing.T
	w    *decisionlog.Writer
	path string
}

// attachLog opens a decision log in a temp dir and hands it to gw.
func attachLog(t *testing.T, gw *Server) *logSink {
	t.Helper()
	path := filepath.Join(t.TempDir(), "decisions.jsonl")
	w, err := decisionlog.Open(path, 1<<20)
	if err != nil {
		t.Fatalf("decisionlog.Open: %v", err)
	}
	t.Cleanup(func() { _ = w.Close() })
	gw.decisions = w
	return &logSink{t: t, w: w, path: path}
}

// raw closes the log, which drains it, and returns the file's bytes.
func (l *logSink) raw() []byte {
	l.t.Helper()
	if err := l.w.Close(); err != nil {
		l.t.Fatalf("Close: %v", err)
	}
	b, err := os.ReadFile(l.path)
	if err != nil {
		l.t.Fatal(err)
	}
	return b
}

// records closes the log and parses every line.
func (l *logSink) records() []map[string]any { return parseLog(l.t, l.raw()) }

func parseLog(t *testing.T, raw []byte) []map[string]any {
	t.Helper()
	var out []map[string]any
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("log line is not JSON: %v: %s", err, sc.Text())
		}
		out = append(out, m)
	}
	return out
}

// ofType keeps the records of one type.
func ofType(recs []map[string]any, typ string) []map[string]any {
	var out []map[string]any
	for _, r := range recs {
		if r["type"] == typ {
			out = append(out, r)
		}
	}
	return out
}

func str(t *testing.T, m map[string]any, key string) string {
	t.Helper()
	v, ok := m[key]
	if !ok {
		t.Fatalf("record has no %q: %v", key, m)
	}
	s, _ := v.(string)
	return s
}

func num(t *testing.T, m map[string]any, key string) float64 {
	t.Helper()
	v, ok := m[key]
	if !ok {
		t.Fatalf("record has no %q: %v", key, m)
	}
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("record %q = %v (%T), want a number", key, v, v)
	}
	return f
}

// candidate finds the candidate entry for model.
func candidate(t *testing.T, rec map[string]any, model string) map[string]any {
	t.Helper()
	for _, c := range rec["candidates"].([]any) {
		if m := c.(map[string]any); m["model"] == model {
			return m
		}
	}
	t.Fatalf("no candidate %q in %v", model, rec["candidates"])
	return nil
}

// TestDecisionRecordHasNoPromptText plants a sentinel in every place a prompt can carry text
// and checks that no byte run of it reaches the log, while the hash prefix and the byte count
// the log does keep are the right ones.
func TestDecisionRecordHasNoPromptText(t *testing.T) {
	const sentinel = "zq7-sentinel-prompt-do-not-log-zq7"
	bodyFor := func(model string) string {
		return fmt.Sprintf(`{"model":%q,"user":%q,"messages":[
		{"role":"system","content":%q},
		{"role":"user","content":[{"type":"text","text":%q},{"type":"text","text":%q}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":%q}}]},
		{"role":"tool","tool_call_id":"call_1","content":%q}],
		"tools":[{"type":"function","function":{"name":"lookup","description":%q,"parameters":{"type":"object"}}}]}`,
			model, sentinel+"-user", sentinel+"-system", sentinel+"-part1", sentinel+"-part2",
			`{"q":"`+sentinel+`-args"}`, sentinel+"-tool", sentinel+"-tooldesc")
	}
	e := newAttemptEnv(t, attemptOpts{
		models: []modelSpec{{id: "m", in: "1", out: "2", caps: []string{"streaming", "tools"}, deps: []depSpec{{"m-a", "ua"}}}},
		policy: config.PolicyCheapest,
	})
	sink := attachLog(t, e.gw)

	for _, model := range []string{"auto", "m"} {
		if rec := e.post(bodyFor(model)); rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", model, rec.Code, rec.Body.String())
		}
	}
	raw := sink.raw()

	if len(raw) == 0 {
		t.Fatal("decision log is empty")
	}
	for _, secret := range []string{sentinel, keyA, "lookup", "call_1"} {
		if bytes.Contains(raw, []byte(secret)) {
			t.Errorf("decision log contains %q:\n%s", secret, raw)
		}
	}
	for i := 0; i+12 <= len(sentinel); i++ { // no fragment of the sentinel either
		if frag := sentinel[i : i+12]; bytes.Contains(raw, []byte(frag)) {
			t.Errorf("decision log contains the fragment %q of the sentinel", frag)
		}
	}

	// What the log keeps of the prompt: the hash prefix of the joined message texts and the
	// number of text bytes.
	texts := []string{sentinel + "-system", sentinel + "-part1\n" + sentinel + "-part2", "", sentinel + "-tool"}
	sum := sha256.Sum256([]byte(strings.Join(texts, "\n")))
	want := hex.EncodeToString(sum[:])[:16]
	total := 0
	for _, s := range texts {
		total += len(s)
	}
	decisions := ofType(parseLog(t, raw), "decision")
	if len(decisions) != 2 {
		t.Fatalf("%d decision records, want 2", len(decisions))
	}
	for _, r := range decisions {
		if got := str(t, r, "prompt_sha256_16"); got != want {
			t.Errorf("prompt_sha256_16 = %q, want %q (SHA-256 of the message texts joined by newlines)", got, want)
		}
		if got := num(t, r, "prompt_bytes"); int(got) != total {
			t.Errorf("prompt_bytes = %v, want %d", got, total)
		}
		if got := r["required"]; fmt.Sprint(got) != "[tools]" {
			t.Errorf("required = %v, want [tools]", got)
		}
	}
}

// TestDecisionLogRecordsShadowAndOutcome: one learned request writes the primary decision
// with its outcome and then the shadow decision without one, both under the response's
// decision ID.
func TestDecisionLogRecordsShadowAndOutcome(t *testing.T) {
	shadowPolicy := policyYAML(0.9, 0, "best_effort")
	e := newLearnedEnv(t, learnedOpts{shadow: []string{shadowPolicy}, now: steppingClock()})
	sink := attachLog(t, e.gw)

	rec := e.post(chatBody("auto", learnedPrompt), nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	decisions := ofType(sink.records(), "decision")
	if len(decisions) != 2 {
		t.Fatalf("%d decision records, want the primary and one shadow: %v", len(decisions), decisions)
	}
	primary, shadow := decisions[0], decisions[1]
	id := rec.Header().Get(headerDecisionID)

	if str(t, primary, "decision_id") != id || str(t, shadow, "decision_id") != id {
		t.Errorf("decision ids %q and %q, want both %q", primary["decision_id"], shadow["decision_id"], id)
	}
	if got := str(t, primary, "policy_role"); got != "stable" {
		t.Errorf("primary policy_role = %q, want stable", got)
	}
	if got, want := str(t, primary, "policy_version"), rec.Header().Get(headerPolicyVersion); got != want {
		t.Errorf("primary policy_version = %q, want the header's %q", got, want)
	}
	if str(t, primary, "tenant") != "tenant-a" {
		t.Errorf("tenant = %v", primary["tenant"])
	}
	if got, want := str(t, primary, "chosen"), rec.Header().Get("x-routepilot-model"); got != want || got == "" {
		t.Errorf("chosen = %q, want the served model %q", got, want)
	}
	if p := num(t, primary, "propensity"); p != 1 {
		t.Errorf("propensity = %v, want 1 without exploration", p)
	}
	if primary["explored"] != false || num(t, primary, "epsilon") != 0 || primary["ood"] != false {
		t.Errorf("explored/epsilon/ood = %v/%v/%v, want false/0/false", primary["explored"], primary["epsilon"], primary["ood"])
	}
	if _, ok := primary["ood_score"].(float64); !ok {
		t.Errorf("ood_score = %v, want a number", primary["ood_score"])
	}
	if relaxed, ok := primary["relaxed"].([]any); !ok || len(relaxed) != 0 {
		t.Errorf("relaxed = %v, want an empty list", primary["relaxed"])
	}
	for _, model := range []string{"tiny-weak", "tiny-strong"} {
		c := candidate(t, primary, model)
		for _, key := range []string{"p", "l", "cost_exp_picousd", "cost_max_picousd"} {
			if _, ok := c[key].(float64); !ok {
				t.Errorf("candidate %s %s = %v, want a number", model, key, c[key])
			}
		}
		if c["q_ms"] != nil {
			t.Errorf("candidate %s q_ms = %v, want null without an SLO", model, c["q_ms"])
		}
		if _, ok := c["feasible"].(bool); !ok {
			t.Errorf("candidate %s feasible = %v", model, c["feasible"])
		}
	}

	// The outcome of the primary decision.
	if num(t, primary, "status") != 200 {
		t.Errorf("status = %v, want 200", primary["status"])
	}
	if num(t, primary, "latency_ms") < 0 {
		t.Errorf("latency_ms = %v", primary["latency_ms"])
	}
	if primary["ttft_ms"] != nil {
		t.Errorf("ttft_ms = %v, want null for a non-streaming request", primary["ttft_ms"])
	}
	if num(t, primary, "prompt_tokens") <= 0 || num(t, primary, "completion_tokens") <= 0 {
		t.Errorf("tokens = %v/%v, want the answering attempt's usage", primary["prompt_tokens"], primary["completion_tokens"])
	}
	if num(t, primary, "cost_actual_picousd") <= 0 || primary["usage_estimated"] != false {
		t.Errorf("cost %v estimated %v", primary["cost_actual_picousd"], primary["usage_estimated"])
	}
	attempts := primary["attempts"].([]any)
	if len(attempts) != 1 || attempts[0].(map[string]any)["outcome"] != "ok" {
		t.Fatalf("attempts = %v, want one ok attempt", attempts)
	}
	if dep := attempts[0].(map[string]any)["deployment"]; dep != rec.Header().Get("x-routepilot-deployment") {
		t.Errorf("attempt deployment = %v, want %s", dep, rec.Header().Get("x-routepilot-deployment"))
	}

	// The shadow decision: its own slot, no outcome.
	if got := str(t, shadow, "policy_role"); got != "shadow" {
		t.Errorf("shadow policy_role = %q", got)
	}
	if got, want := str(t, shadow, "policy_version"), e.slotVersion(t, shadowPolicy); got != want {
		t.Errorf("shadow policy_version = %q, want %q", got, want)
	}
	for _, key := range []string{"status", "latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "cost_actual_picousd", "usage_estimated", "attempts"} {
		if _, ok := shadow[key]; ok {
			t.Errorf("shadow record has outcome field %q", key)
		}
	}
	if len(shadow["candidates"].([]any)) != 2 {
		t.Errorf("shadow candidates = %v, want both models", shadow["candidates"])
	}
	if str(t, shadow, "prompt_sha256_16") != str(t, primary, "prompt_sha256_16") {
		t.Error("shadow and primary disagree on the prompt hash")
	}
}

// TestDecisionLogRecordsAStream: a stream records the time to the first committed chunk.
func TestDecisionLogRecordsAStream(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{now: steppingClock()})
	sink := attachLog(t, e.gw)

	rec := e.post(fmt.Sprintf(`{"model":"auto","stream":true,"messages":[{"role":"user","content":%q}]}`, learnedPrompt), nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	decisions := ofType(sink.records(), "decision")
	if len(decisions) != 1 {
		t.Fatalf("%d decision records, want 1", len(decisions))
	}
	d := decisions[0]
	ttft, ok := d["ttft_ms"].(float64)
	if !ok || ttft <= 0 {
		t.Fatalf("ttft_ms = %v, want a positive number for a stream", d["ttft_ms"])
	}
	if latency := num(t, d, "latency_ms"); latency < ttft {
		t.Errorf("latency_ms %v is below ttft_ms %v", latency, ttft)
	}
	if num(t, d, "status") != 200 || num(t, d, "completion_tokens") <= 0 {
		t.Errorf("status/completion_tokens = %v/%v", d["status"], d["completion_tokens"])
	}
}

// TestDecisionLogRecordsRoutingErrors: a request that reached the router and was refused
// still has a record, with the status the client got and the reasons models were out.
func TestDecisionLogRecordsRoutingErrors(t *testing.T) {
	t.Run("learned reject", func(t *testing.T) {
		e := newLearnedEnv(t, learnedOpts{policy: policyYAML(0.99, 0, "reject")})
		sink := attachLog(t, e.gw)

		rec := e.post(chatBody("auto", learnedPrompt), nil)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		d := ofType(sink.records(), "decision")
		if len(d) != 1 {
			t.Fatalf("%d decision records, want 1", len(d))
		}
		if num(t, d[0], "status") != 422 || str(t, d[0], "chosen") != "" {
			t.Errorf("status/chosen = %v/%q, want 422 and nothing chosen", d[0]["status"], d[0]["chosen"])
		}
		if d[0]["propensity"] != nil {
			t.Errorf("propensity = %v, want null: no action was taken", d[0]["propensity"])
		}
		if attempts := d[0]["attempts"].([]any); len(attempts) != 0 || num(t, d[0], "cost_actual_picousd") != 0 {
			t.Errorf("attempts %v cost %v, want none and zero", attempts, d[0]["cost_actual_picousd"])
		}
		for _, c := range d[0]["candidates"].([]any) {
			if c.(map[string]any)["feasible"] != false || len(c.(map[string]any)["reasons"].([]any)) == 0 {
				t.Errorf("candidate %v, want infeasible with reasons", c)
			}
		}
	})

	t.Run("no model has the capability", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: []modelSpec{{id: "plain", deps: []depSpec{{"plain-a", "ua"}}}}, policy: config.PolicyCheapest})
		sink := attachLog(t, e.gw)

		rec := e.post(`{"model":"auto","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		d := ofType(sink.records(), "decision")
		if len(d) != 1 || num(t, d[0], "status") != 400 {
			t.Fatalf("records %v, want one with status 400", d)
		}
		c := candidate(t, d[0], "plain")
		reasons := c["reasons"].([]any)
		if c["feasible"] != false || len(reasons) != 1 || !strings.Contains(reasons[0].(string), "missing capability tools") {
			t.Errorf("candidate %v, want infeasible for the missing capability", c)
		}
		if c["cost_max_picousd"] != nil || c["p"] != nil {
			t.Errorf("an unscored candidate has cost/p %v/%v, want null", c["cost_max_picousd"], c["p"])
		}
	})

	t.Run("unknown model", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest})
		sink := attachLog(t, e.gw)

		if rec := e.post(chatBody("nope", "hi")); rec.Code != http.StatusNotFound {
			t.Fatalf("status %d", rec.Code)
		}
		d := ofType(sink.records(), "decision")
		if len(d) != 1 || num(t, d[0], "status") != 404 || str(t, d[0], "policy_role") != "pinned" {
			t.Errorf("records %v, want one pinned record with status 404", d)
		}
	})

	t.Run("no router, no record", func(t *testing.T) {
		e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest})
		sink := attachLog(t, e.gw)

		do(e.gw.Handler(), http.MethodPost, "/v1/chat/completions", "test-key-nobody", chatBody("auto", "hi")) // 401
		e.post(`{"model":`)                                                                                    // 400 before routing
		do(e.gw.Handler(), http.MethodGet, "/v1/models", keyA, "")

		if raw := sink.raw(); len(raw) != 0 {
			t.Errorf("log = %s, want empty: none of these requests reached the router", raw)
		}
	})
}

// TestDecisionLogStaticAndPinnedCandidates: a static policy has no probabilities, so those
// fields are null, and a failover shows every attempt while "chosen" stays the policy's pick.
func TestDecisionLogStaticAndPinnedCandidates(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{
		models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc"), clock: steppingClock(),
	})
	sink := attachLog(t, e.gw)

	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK || modelOf(t, rec) != "pricey" {
		t.Fatalf("auto: status %d model %q", rec.Code, modelOf(t, rec))
	}
	if rec := e.post(chatBody("pricey", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("pinned: status %d", rec.Code)
	}
	d := ofType(sink.records(), "decision")
	if len(d) != 2 {
		t.Fatalf("%d records, want 2", len(d))
	}

	auto := d[0]
	if str(t, auto, "policy_role") != "static" || str(t, auto, "policy_version") != "static-cheapest" || str(t, auto, "chosen") != "cheap" {
		t.Errorf("auto role/version/chosen = %v/%v/%v, want static/static-cheapest/cheap", auto["policy_role"], auto["policy_version"], auto["chosen"])
	}
	for _, key := range []string{"propensity", "epsilon", "explored", "ood", "ood_score"} {
		if auto[key] != nil {
			t.Errorf("static decision %s = %v, want null", key, auto[key])
		}
	}
	cheap := candidate(t, auto, "cheap")
	if cheap["feasible"] != true || cheap["p"] != nil || cheap["l"] != nil || cheap["cost_exp_picousd"] != nil || cheap["q_ms"] != nil {
		t.Errorf("static candidate = %v, want feasible with null p, l, cost_exp, q_ms", cheap)
	}
	if got, want := num(t, cheap, "cost_max_picousd"), float64(worstCase(t, 100_000, 100_000, 1000)); got != want {
		t.Errorf("cost_max_picousd = %v, want %v", got, want)
	}
	attempts := auto["attempts"].([]any)
	if len(attempts) != 2 {
		t.Fatalf("attempts = %v, want the failed cheap attempt and the answering pricey one", attempts)
	}
	if a := attempts[0].(map[string]any); a["deployment"] != "cheap-a" || a["outcome"] != "http_503" {
		t.Errorf("first attempt = %v, want cheap-a http_503", a)
	}
	if a := attempts[1].(map[string]any); a["deployment"] != "pricey-a" || a["outcome"] != "ok" {
		t.Errorf("second attempt = %v, want pricey-a ok", a)
	}
	if num(t, auto, "cost_actual_picousd") != 45_000_000 || num(t, auto, "prompt_tokens") != 5 || num(t, auto, "completion_tokens") != 4 {
		t.Errorf("cost/tokens = %v/%v/%v, want the answering attempt's 45000000, 5, 4",
			auto["cost_actual_picousd"], auto["prompt_tokens"], auto["completion_tokens"])
	}

	pinned := d[1]
	if str(t, pinned, "policy_role") != "pinned" || str(t, pinned, "policy_version") != "pinned" || str(t, pinned, "chosen") != "pricey" {
		t.Errorf("pinned role/version/chosen = %v/%v/%v", pinned["policy_role"], pinned["policy_version"], pinned["chosen"])
	}
}

// TestDecisionLogUnscoredModelsHaveNoNumbers: a model the policy artifact does not cover is a
// candidate with its reason and null numbers, not zeros that look like a score.
func TestDecisionLogUnscoredModelsHaveNoNumbers(t *testing.T) {
	e := newLearnedEnv(t, learnedOpts{extraModel: `  - id: unlisted
    rank: 3
    input_price_per_mtok: "9"
    output_price_per_mtok: "9"
    deployments:
      - name: unlisted-dep
        provider: mockp
        upstream_model: "unlisted-up"
`})
	sink := attachLog(t, e.gw)

	if rec := e.post(chatBody("auto", learnedPrompt), nil); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	d := ofType(sink.records(), "decision")
	if len(d) != 1 {
		t.Fatalf("%d records, want 1", len(d))
	}
	unlisted := candidate(t, d[0], "unlisted")
	reasons := unlisted["reasons"].([]any)
	if unlisted["feasible"] != false || len(reasons) != 1 || reasons[0] != "not in the policy artifact" {
		t.Errorf("unlisted candidate = %v, want infeasible because it is not in the policy artifact", unlisted)
	}
	for _, key := range []string{"p", "l", "cost_exp_picousd", "cost_max_picousd", "q_ms"} {
		if unlisted[key] != nil {
			t.Errorf("unlisted candidate %s = %v, want null", key, unlisted[key])
		}
	}
	if _, ok := candidate(t, d[0], "tiny-strong")["p"].(float64); !ok {
		t.Error("a scored candidate has no p")
	}
}

// TestDecisionLogCostIsTheBudgetCommit: the cost a record shows is what the budget committed.
func TestDecisionLogCostIsTheBudgetCommit(t *testing.T) {
	f := newFailoverBudget(t)
	e := newBudgetEnv(t, budgetOpts{
		budget: f.budget, policy: config.PolicyCheapest, models: failoverModels(), upstream: upstream503For("uc"),
	})
	sink := attachLog(t, e.gw)

	if rec := e.post(chatBody("auto", "hi")); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	d := ofType(sink.records(), "decision")
	if len(d) != 1 {
		t.Fatalf("%d records", len(d))
	}
	if got, want := num(t, d[0], "cost_actual_picousd"), float64(e.spent()); got != want || got == 0 {
		t.Errorf("cost_actual_picousd = %v, budget committed %v, want them equal and non-zero", got, want)
	}
}

// TestDecisionLogCountersAreExposed: the metrics read these.
func TestDecisionLogCountersAreExposed(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest})
	if e.gw.DecisionLogDropped() != 0 || e.gw.DecisionLogErrors() != 0 {
		t.Error("a gateway without a decision log reports drops or errors")
	}
	sink := attachLog(t, e.gw)
	e.post(chatBody("auto", "hi"))
	_ = sink.raw()
	if e.gw.DecisionLogDropped() != 0 || e.gw.DecisionLogErrors() != 0 {
		t.Errorf("dropped/errors = %d/%d, want 0/0", e.gw.DecisionLogDropped(), e.gw.DecisionLogErrors())
	}
}
