package replay

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/features"
)

// The fixture config has cheap-a, mid-b and top-c; the MF files below route between the
// cheapest (weak) and the strongest (strong) of them.
const (
	mfStrong = "top-c"
	mfWeak   = "cheap-a"
)

func encodeU(u []float32) string {
	buf := make([]byte, 4*len(u))
	for i, v := range u {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// mfDocument is a valid MF file body as a map, so a test can break one key at a time.
func mfDocument(dim int, u []float32, mean, std []float64) map[string]any {
	return map[string]any{
		"strong": mfStrong, "weak": mfWeak, "dim": dim,
		"dense_mean": mean, "dense_std": std, "u": encodeU(u),
	}
}

func unitStd() []float64 {
	std := make([]float64, features.NumDense)
	for i := range std {
		std[i] = 1
	}
	return std
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func mustParseMF(t *testing.T, doc map[string]any) *MF {
	t.Helper()
	mf, err := ParseMF(mustMarshal(t, doc))
	if err != nil {
		t.Fatalf("ParseMF: %v", err)
	}
	return mf
}

// lengthMF scores only the standardized log1p(bytes of the text): the score is
// sigmoid((log1p(n) - log1p(40)) / 0.1), so items of different lengths spread over (0, 1).
func lengthMF(t *testing.T) *MF {
	t.Helper()
	const dim = 16
	u := make([]float32, dim+features.NumDense)
	u[dim] = 1
	mean := make([]float64, features.NumDense)
	mean[0] = math.Log1p(40)
	std := unitStd()
	std[0] = 0.1
	return mustParseMF(t, mfDocument(dim, u, mean, std))
}

func lengthScore(item Item) float64 {
	texts := make([]string, len(item.Messages))
	for i, m := range item.Messages {
		texts[i] = m.Content.PlainText()
	}
	n := len(strings.Join(texts, "\n"))
	z := (math.Log1p(float64(n)) - math.Log1p(40)) / 0.1
	return 1 / (1 + math.Exp(-z))
}

var mfLengths = []int{30, 36, 40, 44, 60}

// mfInput is five items whose prompts are mfLengths bytes long, so lengthMF gives them five
// different scores. The prompts are invented text behind a sentinel prefix.
func mfInput() []byte {
	var b strings.Builder
	for i, n := range mfLengths {
		text := "SENTINEL-PROMPT-7f3a" + strings.Repeat("x", n-20)
		outcome := func(ok bool, cost int) string {
			return fmt.Sprintf(`{"success":%t,"cost_picousd":%d,"prompt_tokens":9,"completion_tokens":7}`, ok, cost)
		}
		fmt.Fprintf(&b, `{"item_id":"mf/%d","dataset":"mf","split":"test","prompt_sha256":"%064d","messages":[{"role":"user","content":%q}],`+
			`"outcomes":{"cheap-a":%s,"mid-b":%s,"top-c":%s}}`+"\n",
			i, i+1, text, outcome(i%2 == 0, 100+i), outcome(i%3 == 0, 200+i), outcome(i != 2, 900+i))
	}
	return []byte(b.String())
}

func mfItems(t *testing.T) []Item {
	t.Helper()
	var items []Item
	for _, line := range bytes.Split(bytes.TrimSpace(mfInput()), []byte("\n")) {
		var it Item
		if err := json.Unmarshal(line, &it); err != nil {
			t.Fatalf("decode item: %v", err)
		}
		items = append(items, it)
	}
	return items
}

func replayMF(t *testing.T, mf *MF, threshold float64) []Record {
	t.Helper()
	opts := Options{Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: threshold}
	recs, _, _ := replayInput(t, loadFixtureConfig(t), mfInput(), opts)
	return recs
}

func TestParseMF(t *testing.T) {
	const dim = 4
	u := make([]float32, dim+features.NumDense)
	for i := range u {
		u[i] = float32(i) - 3.5
	}
	mean := make([]float64, features.NumDense)
	mean[3] = 1.25
	std := unitStd()
	std[5] = 2

	mf := mustParseMF(t, mfDocument(dim, u, mean, std))

	if mf.Strong != mfStrong || mf.Weak != mfWeak || mf.Dim != dim {
		t.Errorf("got strong %q weak %q dim %d", mf.Strong, mf.Weak, mf.Dim)
	}
	if !slices.Equal(mf.U, u) {
		t.Errorf("U = %v, want %v", mf.U, u)
	}
	if mf.DenseMean[3] != 1.25 || mf.DenseStd[5] != 2 {
		t.Errorf("dense mean %v std %v", mf.DenseMean, mf.DenseStd)
	}
}

func TestParseMFRejects(t *testing.T) {
	const dim = 4
	valid := func() map[string]any {
		return mfDocument(dim, make([]float32, dim+features.NumDense), make([]float64, features.NumDense), unitStd())
	}
	nan := make([]float32, dim+features.NumDense)
	nan[2] = float32(math.NaN())
	inf := make([]float32, dim+features.NumDense)
	inf[7] = float32(math.Inf(-1))
	cases := map[string]func(map[string]any){
		"empty strong":  func(d map[string]any) { d["strong"] = "" },
		"empty weak":    func(d map[string]any) { d["weak"] = "" },
		"same model":    func(d map[string]any) { d["weak"] = d["strong"] },
		"zero dim":      func(d map[string]any) { d["dim"] = 0 },
		"negative dim":  func(d map[string]any) { d["dim"] = -1 },
		"dim too large": func(d map[string]any) { d["dim"] = math.MaxInt32 + 1 },
		"short mean":    func(d map[string]any) { d["dense_mean"] = []float64{0} },
		"short std":     func(d map[string]any) { d["dense_std"] = []float64{1} },
		"zero std":      func(d map[string]any) { s := unitStd(); s[4] = 0; d["dense_std"] = s },
		"negative std":  func(d map[string]any) { s := unitStd(); s[4] = -1; d["dense_std"] = s },
		"u not base64":  func(d map[string]any) { d["u"] = "!!not base64!!" },
		"u too short":   func(d map[string]any) { d["u"] = encodeU(make([]float32, dim+features.NumDense-1)) },
		"u too long":    func(d map[string]any) { d["u"] = encodeU(make([]float32, dim+features.NumDense+1)) },
		"u not a multiple": func(d map[string]any) {
			d["u"] = base64.StdEncoding.EncodeToString(make([]byte, 4*(dim+features.NumDense)+1))
		},
		"u holds NaN":       func(d map[string]any) { d["u"] = encodeU(nan) },
		"u holds infinity":  func(d map[string]any) { d["u"] = encodeU(inf) },
		"unknown key":       func(d map[string]any) { d["extra"] = 1 },
		"missing u":         func(d map[string]any) { delete(d, "u") },
		"missing dense_std": func(d map[string]any) { delete(d, "dense_std") },
	}
	for name, mutate := range cases {
		doc := valid()
		mutate(doc)
		if _, err := ParseMF(mustMarshal(t, doc)); err == nil {
			t.Errorf("%s: ParseMF succeeded, want an error", name)
		}
	}
	for name, raw := range map[string]string{
		"not json":      "weights",
		"empty":         "",
		"trailing data": string(mustMarshal(t, valid())) + " {}",
	} {
		if _, err := ParseMF([]byte(raw)); err == nil {
			t.Errorf("%s: ParseMF succeeded, want an error", name)
		}
	}
}

func TestLoadMF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mf.json")
	const dim = 2
	doc := mfDocument(dim, make([]float32, dim+features.NumDense), make([]float64, features.NumDense), unitStd())
	if err := os.WriteFile(path, mustMarshal(t, doc), 0o600); err != nil {
		t.Fatal(err)
	}
	mf, err := LoadMF(path)
	if err != nil || mf.Dim != dim {
		t.Fatalf("LoadMF = %v, %v", mf, err)
	}
	if _, err := LoadMF(filepath.Join(t.TempDir(), "none.json")); err == nil {
		t.Error("LoadMF of a missing file succeeded")
	}
}

func TestMFScoreMatchesTheLinearScore(t *testing.T) {
	const dim = 16
	u := make([]float32, dim+features.NumDense)
	for i := range u[:dim] {
		u[i] = float32(i%5)*0.25 - 0.5
	}
	mean := make([]float64, features.NumDense)
	std := make([]float64, features.NumDense)
	for d := range features.NumDense {
		u[dim+d] = float32(d)*0.1 - 0.4
		mean[d] = float64(d) * 0.05
		std[d] = 1 + float64(d)*0.5
	}
	mf := mustParseMF(t, mfDocument(dim, u, mean, std))

	cases := map[string][]features.Message{
		"plain":        {{Role: "user", Text: "alpha beta alpha def x"}},
		"empty":        {{Role: "user", Text: ""}},
		"non ascii":    {{Role: "user", Text: "caf\u00e9 \u4e2d\u6587 pi\u00f1ata"}},
		"system":       {{Role: "system", Text: "be brief"}, {Role: "user", Text: "A. one\nB. two ```x``` $$ 12?"}},
		"no messages":  nil,
		"long repeats": {{Role: "user", Text: strings.Repeat("token ", 5000)}},
	}
	for name, msgs := range cases {
		v := features.Extract(msgs, dim)
		var logit float64
		for k, i := range v.Idx {
			logit += float64(u[i]) * v.Val[k]
		}
		for d := range v.Dense {
			logit += float64(u[dim+d]) * ((v.Dense[d] - mean[d]) / std[d])
		}
		want := 1 / (1 + math.Exp(-logit))
		if got := mf.Score(msgs); math.Abs(got-want) > 1e-12 {
			t.Errorf("%s: Score = %v, want %v", name, got, want)
		}
	}
}

func TestMFPolicyThreshold(t *testing.T) {
	mf := lengthMF(t)
	items := mfItems(t)
	scores := make(map[string]float64, len(items))
	for _, it := range items {
		scores[it.ItemID] = lengthScore(it)
	}
	sorted := make([]float64, 0, len(scores))
	for _, s := range scores {
		sorted = append(sorted, s)
	}
	slices.Sort(sorted)
	if sorted[0] < 0.01 || sorted[len(sorted)-1] > 0.99 || sorted[0] == sorted[len(sorted)-1] {
		t.Fatalf("fixture scores %v do not spread over (0, 1); the test would prove nothing", sorted)
	}

	// Threshold 0 sends everything to the strong model and records each item's score in p.
	recs := replayMF(t, mf, 0)
	for _, r := range recs {
		if modelOf(r) != mfStrong {
			t.Errorf("threshold 0: %s went to %q, want %s", r.ItemID, modelOf(r), mfStrong)
		}
		if r.P == nil || math.Abs(*r.P-scores[r.ItemID]) > 1e-12 {
			t.Errorf("%s: p = %v, want the MF score %v", r.ItemID, r.P, scores[r.ItemID])
		}
	}

	// A threshold equal to a score routes that item to the strong model; the next float
	// above it routes the item to the weak model.
	pivot := sorted[2]
	for _, tc := range []struct {
		name      string
		threshold float64
	}{
		{"at the third score", pivot},
		{"just above the third score", math.Nextafter(pivot, 1)},
		{"above every score", 1},
		{"between two scores", (sorted[1] + sorted[2]) / 2},
	} {
		for _, r := range replayMF(t, mf, tc.threshold) {
			want := mfWeak
			if *r.P >= tc.threshold {
				want = mfStrong
			}
			if modelOf(r) != want {
				t.Errorf("%s: %s (p %v) went to %q, want %q", tc.name, r.ItemID, *r.P, modelOf(r), want)
			}
		}
	}
	strongCount := func(threshold float64) int {
		n := 0
		for _, r := range replayMF(t, mf, threshold) {
			if modelOf(r) == mfStrong {
				n++
			}
		}
		return n
	}
	if got := strongCount(pivot); got != 3 {
		t.Errorf("threshold at the third lowest score: %d strong, want 3 (the third lowest and above)", got)
	}
	if got := strongCount(math.Nextafter(pivot, 1)); got != 2 {
		t.Errorf("threshold just above the third lowest score: %d strong, want 2", got)
	}
	if got := strongCount(1); got != 0 {
		t.Errorf("threshold 1: %d strong, want none", got)
	}
}

func TestMFPolicyRecordFields(t *testing.T) {
	mf := lengthMF(t)
	items := mfItems(t)
	byID := make(map[string]Item, len(items))
	for _, it := range items {
		byID[it.ItemID] = it
	}
	for _, r := range replayMF(t, mf, 0.5) {
		it := byID[r.ItemID]
		model := modelOf(r)
		o := it.Outcomes[model]
		if r.Policy != PolicyRouteLLMMF || r.PolicyVersion != nil {
			t.Errorf("%s: policy %q version %v, want %q and null", r.ItemID, r.Policy, r.PolicyVersion, PolicyRouteLLMMF)
		}
		if r.Success != o.Success || r.CostPicoUSD != o.CostPicoUSD {
			t.Errorf("%s: success %v cost %d, want the recorded outcome of %s: %v %d",
				r.ItemID, r.Success, r.CostPicoUSD, model, o.Success, o.CostPicoUSD)
		}
		if r.K == nil || r.CostMaxPicoUSD == nil || r.PromptOverhead == nil {
			t.Errorf("%s: k, cost_max_picousd or prompt_overhead is null for a routed item", r.ItemID)
		}
		if r.L != nil || r.CostExpPicoUSD != nil || r.Floor != nil || r.Propensity != nil || r.OOD || r.Explored {
			t.Errorf("%s: learned-policy fields set for the MF baseline: %+v", r.ItemID, r)
		}
		if r.Error != "" || r.Infeasible {
			t.Errorf("%s: error %q infeasible %v", r.ItemID, r.Error, r.Infeasible)
		}
	}
}

func TestMFPolicyFallsBackWhenTheRoutedModelCannotServe(t *testing.T) {
	mf := lengthMF(t)
	run := func(narrow ...string) []Record {
		cfg := loadFixtureConfig(t)
		for i := range cfg.Models {
			if slices.Contains(narrow, cfg.Models[i].ID) {
				cfg.Models[i].ContextWindow = 4 // below every request's token bound
			}
		}
		var out bytes.Buffer
		opts := Options{Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 0}
		_, err := Run(context.Background(), cfg, bytes.NewReader(mfInput()), &out, opts)
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		return decodeRecords(t, out.Bytes())
	}

	// Threshold 0 prefers the strong model, which cannot serve: every item falls back.
	for _, r := range run(mfStrong) {
		if modelOf(r) != mfWeak || r.Error != "" {
			t.Errorf("strong infeasible: %s went to %q (error %q), want %s", r.ItemID, modelOf(r), r.Error, mfWeak)
		}
		if r.P == nil {
			t.Errorf("%s: p is null after a fallback, want the MF score", r.ItemID)
		}
	}

	// Threshold 1 would prefer the weak model; with it infeasible the strong one serves.
	cfg := loadFixtureConfig(t)
	for i := range cfg.Models {
		if cfg.Models[i].ID == mfWeak {
			cfg.Models[i].ContextWindow = 4
		}
	}
	var out bytes.Buffer
	opts := Options{Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 1}
	if _, err := Run(context.Background(), cfg, bytes.NewReader(mfInput()), &out, opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, r := range decodeRecords(t, out.Bytes()) {
		if modelOf(r) != mfStrong || r.Error != "" {
			t.Errorf("weak infeasible: %s went to %q (error %q), want %s", r.ItemID, modelOf(r), r.Error, mfStrong)
		}
	}

	// Neither can serve: a recorded routing error, a failure at zero cost, infeasible.
	recs := run(mfStrong, mfWeak)
	for _, r := range recs {
		if r.Error != "model_capability_unavailable" || !r.Infeasible || r.Model != nil || r.Success || r.CostPicoUSD != 0 {
			t.Errorf("both infeasible: %s = %+v", r.ItemID, r)
		}
		if r.P == nil {
			t.Errorf("%s: p is null on a routing error, want the MF score", r.ItemID)
		}
	}
}

func TestMFPolicyRejectsBadOptions(t *testing.T) {
	mf := lengthMF(t)
	cfg := loadFixtureConfig(t)
	cases := map[string]Options{
		"no mf":             {Split: SplitAll, Policy: PolicyRouteLLMMF, Threshold: 0.5},
		"nan threshold":     {Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: math.NaN()},
		"negative":          {Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: -0.1},
		"above one":         {Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 1.1},
		"with a model":      {Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 0.5, Model: mfStrong},
		"mf with cheapest":  {Split: SplitAll, Policy: PolicyCheapest, MF: mf},
		"mf with fixed":     {Split: SplitAll, Policy: PolicyFixed, Model: mfStrong, MF: mf},
		"threshold alone":   {Split: SplitAll, Policy: PolicyCheapest, Threshold: 0.5},
		"threshold on rule": {Split: SplitAll, Policy: PolicyRules, Threshold: 0.5},
	}
	for name, opts := range cases {
		_, err := Run(context.Background(), cfg, bytes.NewReader(mfInput()), &bytes.Buffer{}, opts)
		if !errors.Is(err, ErrUsage) {
			t.Errorf("%s: Run error = %v, want a usage error", name, err)
		}
	}
}

func TestMFPolicyNamesAModelMissingFromTheConfig(t *testing.T) {
	const dim = 2
	for _, missing := range []string{"strong", "weak"} {
		doc := mfDocument(dim, make([]float32, dim+features.NumDense), make([]float64, features.NumDense), unitStd())
		doc[missing] = "ghost-model"
		mf := mustParseMF(t, doc)
		opts := Options{Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 0.5}
		_, err := Run(context.Background(), loadFixtureConfig(t), bytes.NewReader(mfInput()), &bytes.Buffer{}, opts)
		if err == nil || errors.Is(err, ErrUsage) || !strings.Contains(err.Error(), "ghost-model") {
			t.Errorf("%s missing: error = %v, want an input error naming ghost-model", missing, err)
		}
	}
}

func TestMFPolicyOutputHasNoPromptText(t *testing.T) {
	mf := lengthMF(t)
	_, _, raw := replayInput(t, loadFixtureConfig(t), mfInput(), Options{Split: SplitAll, Policy: PolicyRouteLLMMF, MF: mf, Threshold: 0.5})
	if bytes.Contains(raw, []byte("SENTINEL-PROMPT")) {
		t.Fatal("the output carries prompt text")
	}
	for _, key := range []string{"messages", "content", "prompt\""} {
		if bytes.Contains(raw, []byte(key)) {
			t.Errorf("the output carries the key or word %q", key)
		}
	}
}
