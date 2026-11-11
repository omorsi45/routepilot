package replay

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/config"
	"github.com/omorsi45/routepilot/internal/openai"
)

var fixtureDir = filepath.Join("..", "..", "testdata", "replay")

func loadFixtureConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Load(filepath.Join(fixtureDir, "config.yaml"))
	if err != nil {
		t.Fatalf("load fixture config: %v", err)
	}
	return cfg
}

func fixtureInput(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir, "items.jsonl"))
	if err != nil {
		t.Fatalf("read fixture items: %v", err)
	}
	return b
}

func fixtureItems(t *testing.T) []Item {
	t.Helper()
	var items []Item
	for _, line := range bytes.Split(bytes.TrimSpace(fixtureInput(t)), []byte("\n")) {
		var it Item
		if err := json.Unmarshal(line, &it); err != nil {
			t.Fatalf("decode fixture item: %v", err)
		}
		items = append(items, it)
	}
	return items
}

func decodeRecords(t *testing.T, raw []byte) []Record {
	t.Helper()
	var recs []Record
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var r Record
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatalf("decode record %q: %v", line, err)
		}
		recs = append(recs, r)
	}
	return recs
}

// replayInput runs Run over input and returns the decoded records, the
// summary and the raw output bytes.
func replayInput(t *testing.T, cfg *config.Config, input []byte, opts Options) ([]Record, Summary, []byte) {
	t.Helper()
	var out bytes.Buffer
	sum, err := Run(context.Background(), cfg, bytes.NewReader(input), &out, opts)
	if err != nil {
		t.Fatalf("Run(%+v): %v", opts, err)
	}
	return decodeRecords(t, out.Bytes()), sum, out.Bytes()
}

func replayFixture(t *testing.T, opts Options) ([]Record, Summary, []byte) {
	t.Helper()
	return replayInput(t, loadFixtureConfig(t), fixtureInput(t), opts)
}

func modelOf(r Record) string {
	if r.Model == nil {
		return ""
	}
	return *r.Model
}

// picks maps item_id to the chosen model.
func picks(recs []Record) map[string]string {
	m := make(map[string]string, len(recs))
	for _, r := range recs {
		m[r.ItemID] = modelOf(r)
	}
	return m
}

func TestReplayCheapestAndStrongest(t *testing.T) {
	recs, sum, _ := replayFixture(t, Options{Split: "all", Policy: "cheapest"})
	if len(recs) != 5 || sum.Items != 5 || sum.Errors != 0 {
		t.Fatalf("cheapest: %d records, summary %+v; want 5 items and no errors", len(recs), sum)
	}
	for _, r := range recs {
		if got := modelOf(r); got != "cheap-a" {
			t.Errorf("cheapest picked %q for %s, want cheap-a (lowest CostMax)", got, r.ItemID)
		}
		if r.Policy != "cheapest" {
			t.Errorf("record policy = %q, want cheapest", r.Policy)
		}
	}
	// cheap-a recorded: fx-a/1 ok 300, fx-a/2 fail 150, fx-b/3 fail 100, fx-b/4 ok 250, fx-b/5 ok 120.
	if sum.Successes != 3 || sum.CostPicoUSD != 920 {
		t.Errorf("cheapest summary = %+v, want 3 successes and cost 920", sum)
	}

	recs, sum, _ = replayFixture(t, Options{Split: "all", Policy: "strongest"})
	for _, r := range recs {
		if got := modelOf(r); got != "top-c" {
			t.Errorf("strongest picked %q for %s, want top-c (highest rank)", got, r.ItemID)
		}
	}
	// top-c recorded: ok 900, fail 90, ok 700, ok 250 (negative tokens kept as recorded), fail 140.
	if sum.Successes != 3 || sum.CostPicoUSD != 2080 {
		t.Errorf("strongest summary = %+v, want 3 successes and cost 2080", sum)
	}
}

func TestReplayRules(t *testing.T) {
	recs, _, _ := replayFixture(t, Options{Split: "all", Policy: "rules"})
	want := map[string]string{
		"fx-a/1": "top-c", // "def " in the text
		"fx-a/2": "mid-b", // default model
		"fx-b/3": "mid-b",
		"fx-b/4": "mid-b",
		"fx-b/5": "top-c",
	}
	if got := picks(recs); !mapsEqual(got, want) {
		t.Errorf("rules picks = %v, want %v", got, want)
	}
}

func TestReplayOracle(t *testing.T) {
	recs, sum, _ := replayFixture(t, Options{Split: "all", Policy: "oracle"})
	want := map[string]string{
		"fx-a/1": "mid-b",   // cheapest successful recorded cost (200), not the cheapest price
		"fx-a/2": "top-c",   // none succeeded: lowest recorded cost (90)
		"fx-b/3": "top-c",   // only successful model
		"fx-b/4": "cheap-a", // all succeeded at 250: ties go to the lower model ID
		"fx-b/5": "cheap-a", // only successful model
	}
	if got := picks(recs); !mapsEqual(got, want) {
		t.Errorf("oracle picks = %v, want %v", got, want)
	}
	if sum.Successes != 4 || sum.CostPicoUSD != 200+90+700+250+120 {
		t.Errorf("oracle summary = %+v, want 4 successes and cost 1360", sum)
	}
	for _, r := range recs {
		if r.Policy != "oracle" {
			t.Errorf("record policy = %q, want oracle", r.Policy)
		}
	}
}

func TestReplayFixed(t *testing.T) {
	recs, sum, _ := replayFixture(t, Options{Split: "all", Policy: "fixed", Model: "mid-b"})
	for _, r := range recs {
		if got := modelOf(r); got != "mid-b" {
			t.Errorf("fixed picked %q for %s, want mid-b", got, r.ItemID)
		}
		if r.Policy != "fixed" || r.Error != "" {
			t.Errorf("record = policy %q error %q, want fixed and no error", r.Policy, r.Error)
		}
	}
	// mid-b recorded: ok 200, fail 400, fail 200, ok 250, fail 130.
	if sum.Successes != 2 || sum.CostPicoUSD != 1180 {
		t.Errorf("fixed summary = %+v, want 2 successes and cost 1180", sum)
	}

	// An unknown model is a routing error recorded per item, a failure at zero cost.
	recs, sum, _ = replayFixture(t, Options{Split: "all", Policy: "fixed", Model: "ghost"})
	if sum.Items != 5 || sum.Errors != 5 || sum.Successes != 0 || sum.CostPicoUSD != 0 {
		t.Fatalf("unknown fixed model summary = %+v, want 5 items, 5 errors, no success, no cost", sum)
	}
	for _, r := range recs {
		if r.Error != "model_not_found" || r.Model != nil || r.Success || r.CostPicoUSD != 0 {
			t.Errorf("record %+v: want error model_not_found, null model, failure, zero cost", r)
		}
		if r.InUpper == 0 {
			t.Errorf("record %s: in_upper must be set even when routing fails", r.ItemID)
		}
		if r.K != nil || r.CostMaxPicoUSD != nil {
			t.Errorf("record %s: k and cost_max_picousd must be null on a routing error", r.ItemID)
		}
	}
}

func TestReplayRandomDeterministic(t *testing.T) {
	cfg, input := loadFixtureConfig(t), fixtureInput(t)
	a, _, rawA := replayInput(t, cfg, input, Options{Split: "all", Policy: "random", Seed: 1})
	if len(a) != 5 {
		t.Fatalf("random produced %d records, want 5", len(a))
	}
	_, _, rawB := replayInput(t, cfg, input, Options{Split: "all", Policy: "random", Seed: 1})
	if !bytes.Equal(rawA, rawB) {
		t.Fatal("same seed produced different output")
	}
	c, _, _ := replayInput(t, cfg, input, Options{Split: "all", Policy: "random", Seed: 2})
	if mapsEqual(picks(a), picks(c)) {
		t.Errorf("seeds 1 and 2 picked the same models on all 5 items: %v", picks(a))
	}

	// A seed and an item fix the pick, whatever else is in the input.
	subset, _, _ := replayInput(t, cfg, input, Options{Split: "test", Policy: "random", Seed: 1})
	for _, r := range subset {
		if got, want := modelOf(r), picks(a)[r.ItemID]; got != want {
			t.Errorf("%s: pick %q on the test split but %q on all items", r.ItemID, got, want)
		}
	}
	for _, r := range a {
		if r.Policy != "random" {
			t.Errorf("record policy = %q, want random", r.Policy)
		}
	}
}

func TestReplaySplitFilter(t *testing.T) {
	cases := []struct {
		split string
		want  []string
	}{
		{"test", []string{"fx-b/4", "fx-b/5"}},
		{"train", []string{"fx-a/1", "fx-a/2"}},
		{"calib", []string{"fx-b/3"}},
		{"all", []string{"fx-a/1", "fx-a/2", "fx-b/3", "fx-b/4", "fx-b/5"}},
	}
	for _, tc := range cases {
		recs, sum, _ := replayFixture(t, Options{Split: tc.split, Policy: "cheapest"})
		var ids []string
		for _, r := range recs {
			ids = append(ids, r.ItemID)
		}
		if !slices.Equal(ids, tc.want) || sum.Items != len(tc.want) {
			t.Errorf("split %q kept %v (summary items %d), want %v", tc.split, ids, sum.Items, tc.want)
		}
	}

	var out bytes.Buffer
	_, err := Run(context.Background(), loadFixtureConfig(t), bytes.NewReader(fixtureInput(t)), &out,
		Options{Split: "heldout", Policy: "cheapest"})
	if err == nil || !strings.Contains(err.Error(), "heldout") {
		t.Errorf("a split with no items: err = %v, want an error naming the split", err)
	}
}

func TestReplayOutputHasNoPromptText(t *testing.T) {
	const sentinel = "SENTINEL-PROMPT-7f3a"
	if !bytes.Contains(fixtureInput(t), []byte(sentinel)) {
		t.Fatal("fixture input does not carry the sentinel, the test proves nothing")
	}
	for _, opts := range []Options{
		{Split: "all", Policy: "cheapest"},
		{Split: "all", Policy: "rules"},
		{Split: "all", Policy: "oracle"},
		{Split: "all", Policy: "random", Seed: 3},
		{Split: "all", Policy: "fixed", Model: "ghost"},
	} {
		_, _, raw := replayFixture(t, opts)
		if bytes.Contains(raw, []byte(sentinel)) || bytes.Contains(raw, []byte("messages")) {
			t.Errorf("policy %s: output carries prompt text or the messages key", opts.Policy)
		}
	}
}

func TestReplayRecordFields(t *testing.T) {
	cfg := loadFixtureConfig(t)
	cat, err := catalog.New(cfg)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	cheapA, _ := cat.Model("cheap-a")
	items := fixtureItems(t)
	recs, _, raw := replayInput(t, cfg, fixtureInput(t), Options{Split: "all", Policy: "cheapest"})
	if len(recs) != len(items) {
		t.Fatalf("%d records for %d items", len(recs), len(items))
	}
	for i, r := range recs {
		it := items[i]
		req := &openai.ChatCompletionRequest{Model: "auto", Messages: it.Messages}
		reqs := catalog.Derive(req)
		if r.ItemID != it.ItemID || r.Dataset != it.Dataset || r.Split != it.Split {
			t.Errorf("record %d identity = %s/%s/%s, want %s/%s/%s",
				i, r.ItemID, r.Dataset, r.Split, it.ItemID, it.Dataset, it.Split)
		}
		if r.PromptSHA25616 != it.PromptSHA256[:16] {
			t.Errorf("%s: prompt_sha256_16 = %q, want %q", r.ItemID, r.PromptSHA25616, it.PromptSHA256[:16])
		}
		if r.InUpper != reqs.InUpper {
			t.Errorf("%s: in_upper = %d, want catalog.Derive InUpper %d", r.ItemID, r.InUpper, reqs.InUpper)
		}
		if r.K == nil || *r.K != 1024 {
			t.Errorf("%s: k = %v, want the tenant default 1024", r.ItemID, r.K)
		}
		wantMax := int64(cheapA.CostMax(reqs, 1024))
		if r.CostMaxPicoUSD == nil || *r.CostMaxPicoUSD != wantMax {
			t.Errorf("%s: cost_max_picousd = %v, want %d", r.ItemID, r.CostMaxPicoUSD, wantMax)
		}
		if r.Relaxed == nil || len(r.Relaxed) != 0 {
			t.Errorf("%s: relaxed = %v, want an empty list", r.ItemID, r.Relaxed)
		}
		if r.Error != "" || r.Infeasible || r.OOD || r.Explored {
			t.Errorf("%s: unexpected flags in %+v", r.ItemID, r)
		}
	}
	// Literal check so the test does not only mirror the code under test: fx-a/2 is
	// 42 bytes of text, one message: 42 + 8 + 16 = 66 input tokens upper bound, and
	// cheap-a costs 100000 pico per input token and 400000 per output token:
	// 66*100000 + 1024*400000.
	if recs[1].InUpper != 66 || recs[1].CostMaxPicoUSD == nil || *recs[1].CostMaxPicoUSD != 416200000 {
		t.Errorf("fx-a/2: in_upper %d, cost_max %v; want 66 and 416200000", recs[1].InUpper, recs[1].CostMaxPicoUSD)
	}

	// Fields that do not apply to a static baseline are present and null.
	var generic map[string]json.RawMessage
	line := bytes.SplitN(raw, []byte("\n"), 2)[0]
	if err := json.Unmarshal(line, &generic); err != nil {
		t.Fatalf("decode first line: %v", err)
	}
	for _, key := range []string{
		"policy_version", "p", "l", "cost_exp_picousd", "floor", "ood_score", "propensity",
		"ttft_ms_sim", "e2e_ms_sim", "slo_ms", "slo_violated",
	} {
		v, ok := generic[key]
		if !ok || string(v) != "null" {
			t.Errorf("key %q = %s (present %v), want null", key, v, ok)
		}
	}
	for key, want := range map[string]string{
		"relaxed": "[]", "ood": "false", "explored": "false", "infeasible": "false", "error": `""`,
	} {
		if got := string(generic[key]); got != want {
			t.Errorf("key %q = %s, want %s", key, got, want)
		}
	}
	wantKeys := 28
	if len(generic) != wantKeys {
		t.Errorf("record has %d keys, want %d", len(generic), wantKeys)
	}
}

// TestReplayRecordsPromptOverhead pins that in_upper
// stays the request-level bound, prompt_overhead is the chosen model's own
// value, and cost_max_picousd is the model's CostMax with that overhead.
func TestReplayRecordsPromptOverhead(t *testing.T) {
	cfg := loadFixtureConfig(t)
	for i := range cfg.Models {
		if cfg.Models[i].ID == "cheap-a" {
			cfg.Models[i].PromptOverheadTokens = 40
		}
	}

	recs, _, _ := replayInput(t, cfg, fixtureInput(t), Options{Split: "all", Policy: "cheapest"})
	for _, r := range recs {
		if r.PromptOverhead == nil || *r.PromptOverhead != 40 {
			t.Errorf("%s: prompt_overhead = %v, want cheap-a's 40", r.ItemID, r.PromptOverhead)
		}
	}
	// fx-a/2 has the request bound 66 (see TestReplayRecordFields). cheap-a costs
	// 100000 pico per input token and 400000 per output token: with the overhead
	// the input bound is 66+40, so (66+40)*100000 + 1024*400000.
	if recs[1].InUpper != 66 || recs[1].CostMaxPicoUSD == nil || *recs[1].CostMaxPicoUSD != 420200000 {
		t.Errorf("fx-a/2: in_upper %d, cost_max %v; want 66 (request bound) and 420200000", recs[1].InUpper, recs[1].CostMaxPicoUSD)
	}

	// Another model reports its own value, which is 0 here and still present.
	recs, _, _ = replayInput(t, cfg, fixtureInput(t), Options{Split: "all", Policy: "strongest"})
	for _, r := range recs {
		if r.PromptOverhead == nil || *r.PromptOverhead != 0 {
			t.Errorf("%s: prompt_overhead = %v, want top-c's 0", r.ItemID, r.PromptOverhead)
		}
	}

	// No chosen model, no value: the key is null.
	_, _, raw := replayInput(t, cfg, fixtureInput(t), Options{Split: "all", Policy: "fixed", Model: "ghost"})
	if !bytes.Contains(raw, []byte(`"prompt_overhead":null`)) {
		t.Error("a routing error must write prompt_overhead as null")
	}
}

// TestReplayOverheadCanMakeAModelInfeasible: the context check uses the model's
// bound, so an overhead that fills the window removes the model from routing.
func TestReplayOverheadCanMakeAModelInfeasible(t *testing.T) {
	cfg := loadFixtureConfig(t)
	for i := range cfg.Models {
		if cfg.Models[i].ID == "cheap-a" {
			cfg.Models[i].PromptOverheadTokens = cfg.Models[i].ContextWindow
		}
	}
	recs, _, _ := replayInput(t, cfg, fixtureInput(t), Options{Split: "all", Policy: "cheapest"})
	for _, r := range recs {
		if got := modelOf(r); got != "mid-b" {
			t.Errorf("%s: cheapest picked %q, want mid-b (cheap-a no longer fits its context window)", r.ItemID, got)
		}
	}
}

func TestReplayMissingOutcomeIsHardError(t *testing.T) {
	line := `{"item_id":"fx-z/9","dataset":"fx-z","split":"test","prompt_sha256":"0123456789abcdef",` +
		`"messages":[{"role":"user","content":"hello"}],` +
		`"outcomes":{"cheap-a":{"success":true,"cost_picousd":1,"prompt_tokens":1,"completion_tokens":1}}}` + "\n"
	var out bytes.Buffer
	_, err := Run(context.Background(), loadFixtureConfig(t), strings.NewReader(line), &out,
		Options{Split: "all", Policy: "strongest"})
	if err == nil || !strings.Contains(err.Error(), "fx-z/9") || !strings.Contains(err.Error(), "top-c") {
		t.Errorf("err = %v, want an error naming the item fx-z/9 and the model top-c", err)
	}
	if errors.Is(err, ErrUsage) {
		t.Error("a missing outcome is an input error, not a usage error")
	}
}

func TestReplayReadsLongLines(t *testing.T) {
	// A line well past bufio.Scanner's 64 KiB default.
	long := strings.Repeat("x", 200<<10)
	line := `{"item_id":"fx-l/1","dataset":"fx-l","split":"test","prompt_sha256":"0123456789abcdef0123",` +
		`"messages":[{"role":"user","content":"` + long + `"}],` +
		`"outcomes":{"cheap-a":{"success":true,"cost_picousd":5,"prompt_tokens":1,"completion_tokens":1},` +
		`"mid-b":{"success":true,"cost_picousd":6,"prompt_tokens":1,"completion_tokens":1},` +
		`"top-c":{"success":true,"cost_picousd":7,"prompt_tokens":1,"completion_tokens":1}}}` + "\n\n"
	recs, sum, _ := replayInput(t, loadFixtureConfig(t), []byte(line), Options{Split: "all", Policy: "cheapest"})
	if len(recs) != 1 || sum.Items != 1 {
		t.Fatalf("got %d records, summary %+v; want exactly one item (blank lines are skipped)", len(recs), sum)
	}
	if recs[0].InUpper < 200<<10 {
		t.Errorf("in_upper = %d, want at least the %d prompt bytes", recs[0].InUpper, 200<<10)
	}
}

func TestReplayInputWithoutTrailingNewline(t *testing.T) {
	input := bytes.TrimRight(fixtureInput(t), "\n")
	recs, sum, _ := replayInput(t, loadFixtureConfig(t), input, Options{Split: "all", Policy: "cheapest"})
	if len(recs) != 5 || sum.Items != 5 {
		t.Errorf("got %d records, summary %+v; want all 5 items", len(recs), sum)
	}
}

func TestReplayBadInputIsAnInputError(t *testing.T) {
	for name, input := range map[string]string{
		"not json":      "this is not json\n",
		"no messages":   `{"item_id":"fx-q/1","dataset":"d","split":"test","prompt_sha256":"abcdef","messages":[],"outcomes":{}}` + "\n",
		"negative cost": `{"item_id":"fx-q/2","dataset":"d","split":"test","prompt_sha256":"abcdef","messages":[{"role":"user","content":"hi"}],"outcomes":{"cheap-a":{"success":true,"cost_picousd":-1,"prompt_tokens":1,"completion_tokens":1}}}` + "\n",
		"empty input":   "",
	} {
		var out bytes.Buffer
		_, err := Run(context.Background(), loadFixtureConfig(t), strings.NewReader(input), &out,
			Options{Split: "all", Policy: "cheapest"})
		if err == nil {
			t.Errorf("%s: want an error", name)
			continue
		}
		if errors.Is(err, ErrUsage) {
			t.Errorf("%s: err = %v, want an input error, not a usage error", name, err)
		}
	}
}

func TestOptionsValidate(t *testing.T) {
	bad := map[string]Options{
		"no split":            {Policy: "cheapest"},
		"no policy":           {Split: "all"},
		"unknown policy":      {Split: "all", Policy: "best"},
		"fixed without model": {Split: "all", Policy: "fixed"},
		"model without fixed": {Split: "all", Policy: "cheapest", Model: "cheap-a"},
		"unknown tenant":      {Split: "all", Policy: "cheapest", Tenant: "nobody"},
		"oracle with a model": {Split: "all", Policy: "oracle", Model: "cheap-a"},
	}
	for name, opts := range bad {
		var out bytes.Buffer
		_, err := Run(context.Background(), loadFixtureConfig(t), bytes.NewReader(fixtureInput(t)), &out, opts)
		if !errors.Is(err, ErrUsage) {
			t.Errorf("%s: err = %v, want ErrUsage", name, err)
		}
		if out.Len() != 0 {
			t.Errorf("%s: wrote output before failing", name)
		}
	}
	for _, policy := range []string{"cheapest", "strongest", "rules", "random", "oracle"} {
		if err := (Options{Split: "all", Policy: policy}).Validate(); err != nil {
			t.Errorf("policy %s: Validate = %v, want nil", policy, err)
		}
	}
	if err := (Options{Split: "all", Policy: "fixed", Model: "cheap-a"}).Validate(); err != nil {
		t.Errorf("fixed with a model: Validate = %v, want nil", err)
	}
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
