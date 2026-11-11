package policy

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/openai"
)

var parityPath = filepath.Join("..", "..", "testdata", "policy", "parity", "cases.json")

// The cases are written by python/tests/export/make_tiny.py: invented prompts and the
// reference predictor's output for each, computed on the committed tiny artifact.
type parityFile struct {
	ArtifactVersion string       `json:"artifact_version"`
	Cases           []parityCase `json:"cases"`
}

type parityCase struct {
	Name     string                       `json:"name"`
	Messages []parityMessage              `json:"messages"`
	InUpper  int                          `json:"in_upper"`
	OODScore float64                      `json:"ood_score"`
	OOD      bool                         `json:"ood"`
	Models   map[string]parityModelResult `json:"models"`
}

// A message has either a plain text or a repeat spec (text = pattern * count + suffix).
type parityMessage struct {
	Role   string      `json:"role"`
	Text   *string     `json:"text"`
	Repeat *repeatSpec `json:"repeat"`
}

type repeatSpec struct {
	Pattern string `json:"pattern"`
	Count   int    `json:"count"`
	Suffix  string `json:"suffix"`
}

type parityModelResult struct {
	P          float64    `json:"p"`
	L          float64    `json:"l"`
	PooledRate float64    `json:"pooled_rate"`
	PooledLCB  float64    `json:"pooled_lcb"`
	EOut       float64    `json:"e_out"`
	OutQ       [4]float64 `json:"out_q"`
	Q99        float64    `json:"q99"`
	InRatio    float64    `json:"in_ratio"`
}

func loadParity(t *testing.T) parityFile {
	t.Helper()
	var f parityFile
	if err := json.Unmarshal(readFile(t, parityPath), &f); err != nil {
		t.Fatalf("decode %s: %v", parityPath, err)
	}
	return f
}

func (c parityCase) messages(t *testing.T) []features.Message {
	t.Helper()
	msgs := make([]features.Message, 0, len(c.Messages))
	for _, m := range c.Messages {
		switch {
		case m.Text != nil && m.Repeat == nil:
			msgs = append(msgs, features.Message{Role: m.Role, Text: *m.Text})
		case m.Repeat != nil && m.Text == nil:
			text := strings.Repeat(m.Repeat.Pattern, m.Repeat.Count) + m.Repeat.Suffix
			msgs = append(msgs, features.Message{Role: m.Role, Text: text})
		default:
			t.Fatalf("case %q: a message needs exactly one of text and repeat", c.Name)
		}
	}
	return msgs
}

func loadTiny(t *testing.T) *Artifact {
	t.Helper()
	a, err := Load(tinyDir(t))
	if err != nil {
		t.Fatalf("Load tiny artifact: %v", err)
	}
	return a
}

func relTol(want float64) float64 { return 1e-3 * math.Max(1, math.Abs(want)) }

func TestScoreParity(t *testing.T) {
	a := loadTiny(t)
	f := loadParity(t)
	if f.ArtifactVersion != a.Version() {
		t.Fatalf("cases.json was generated for artifact %s but the committed artifact is %s; rerun make_tiny.py",
			f.ArtifactVersion, a.Version())
	}
	if len(f.Cases) != 30 {
		t.Fatalf("cases.json holds %d cases, want 30", len(f.Cases))
	}
	var sawOOD, sawInDistribution bool

	for _, c := range f.Cases {
		t.Run(c.Name, func(t *testing.T) {
			got := a.Score(c.messages(t))

			if math.Abs(got.OODScore-c.OODScore) > 1e-5 {
				t.Errorf("OODScore = %.12g, want %.12g", got.OODScore, c.OODScore)
			}
			if got.OOD != c.OOD {
				t.Errorf("OOD = %v, want %v (score %.12g, threshold %.12g)", got.OOD, c.OOD, got.OODScore, a.OODThreshold())
			}
			if len(got.Models) != len(c.Models) || len(got.Models) != len(a.Models()) {
				t.Fatalf("scored %d models, want %d", len(got.Models), len(c.Models))
			}
			for id, want := range c.Models {
				s, ok := got.Models[id]
				if !ok {
					t.Fatalf("no score for model %q", id)
				}
				if math.Abs(s.P-want.P) > 1e-5 {
					t.Errorf("%s: P = %.12g, want %.12g", id, s.P, want.P)
				}
				if math.Abs(s.L-want.L) > 1e-9 {
					t.Errorf("%s: L = %.12g, want %.12g", id, s.L, want.L)
				}
				if math.Abs(s.EOut-want.EOut) > relTol(want.EOut) {
					t.Errorf("%s: EOut = %.12g, want %.12g", id, s.EOut, want.EOut)
				}
				for q := range want.OutQ {
					if math.Abs(s.OutQ[q]-want.OutQ[q]) > relTol(want.OutQ[q]) {
						t.Errorf("%s: OutQ[%d] = %.12g, want %.12g", id, q, s.OutQ[q], want.OutQ[q])
					}
				}
				if math.Abs(s.OutQ[3]-want.Q99) > relTol(want.Q99) {
					t.Errorf("%s: OutQ[3] = %.12g, want q99 %.12g", id, s.OutQ[3], want.Q99)
				}
				if s.PooledRate != want.PooledRate || s.PooledLCB != want.PooledLCB || s.InRatio != want.InRatio {
					t.Errorf("%s: pooled %v %v in_ratio %v, want %v %v %v",
						id, s.PooledRate, s.PooledLCB, s.InRatio, want.PooledRate, want.PooledLCB, want.InRatio)
				}
				if !slices.IsSorted(s.OutQ[:]) {
					t.Errorf("%s: OutQ = %v is not non-decreasing", id, s.OutQ)
				}
			}
			if c.OOD {
				sawOOD = true
			} else {
				sawInDistribution = true
			}
		})
	}
	if !sawOOD || !sawInDistribution {
		t.Errorf("the cases exercise OOD=%v and in-distribution=%v; both must occur", sawOOD, sawInDistribution)
	}
}

func TestParityCasesCoverTheEdgeCases(t *testing.T) {
	f := loadParity(t)
	names := make(map[string]bool, len(f.Cases))
	var empty, midRune, nonASCII, system, multiTurn bool
	for _, c := range f.Cases {
		if names[c.Name] {
			t.Errorf("duplicate case name %q", c.Name)
		}
		names[c.Name] = true
		msgs := c.messages(t)
		texts := make([]string, len(msgs))
		for i, m := range msgs {
			texts[i] = m.Text
			switch m.Role {
			case "system":
				system = true
			case "assistant":
				multiTurn = true
			}
		}
		joined := strings.Join(texts, "\n")
		switch {
		case joined == "":
			empty = true
		case len(joined) > features.PrefixBytes && !utf8Boundary(joined, features.PrefixBytes):
			midRune = true
		}
		for _, b := range []byte(joined) {
			if b >= 0x80 {
				nonASCII = true
			}
		}
	}
	if !empty || !midRune || !nonASCII || !system || !multiTurn {
		t.Errorf("edge cases present: empty=%v cut-mid-rune=%v non-ASCII=%v system=%v multi-turn=%v; want all",
			empty, midRune, nonASCII, system, multiTurn)
	}
}

// utf8Boundary reports whether cutting s at byte n leaves whole UTF-8 sequences.
func utf8Boundary(s string, n int) bool {
	return n >= len(s) || s[n]&0xC0 != 0x80
}

func TestInUpperMatchesPython(t *testing.T) {
	f := loadParity(t)
	checkedSingle := 0
	for _, c := range f.Cases {
		t.Run(c.Name, func(t *testing.T) {
			msgs := c.messages(t)
			req := &openai.ChatCompletionRequest{Model: "tiny-weak"}
			textBytes := 0
			for _, m := range msgs {
				text := m.Text
				req.Messages = append(req.Messages, openai.Message{Role: m.Role, Content: openai.Content{Text: &text}})
				textBytes += len(m.Text)
			}

			got := catalog.Derive(req).InUpper

			if got != c.InUpper {
				t.Errorf("catalog.Derive InUpper = %d, the Python side computed %d", got, c.InUpper)
			}
			if len(msgs) == 1 && msgs[0].Role == "user" {
				if want := textBytes + 24; got != want {
					t.Errorf("single user message: InUpper = %d, want len(utf8)+24 = %d", got, want)
				}
				checkedSingle++
			}
		})
	}
	if checkedSingle == 0 {
		t.Error("no single-user-message case was checked")
	}
}

func TestScoreZeroVectorIsOOD(t *testing.T) {
	a := loadTiny(t)
	for name, msgs := range map[string][]features.Message{
		"empty text":   {{Role: "user", Text: ""}},
		"no messages":  nil,
		"only symbols": {{Role: "user", Text: "?! ... --- \n\t"}},
	} {
		t.Run(name, func(t *testing.T) {
			got := a.Score(msgs)

			if got.OODScore != 1 {
				t.Errorf("OODScore = %v, want exactly 1 for a request with no hashed features", got.OODScore)
			}
			if !got.OOD {
				t.Errorf("OOD = false, want true (threshold %v)", a.OODThreshold())
			}
			for id, s := range got.Models {
				for field, v := range map[string]float64{"P": s.P, "L": s.L, "EOut": s.EOut, "Q99": s.OutQ[3]} {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Errorf("%s: %s = %v, want a finite number", id, field, v)
					}
				}
				if s.P < 0 || s.P > 1 || s.L < 0 || s.L > 1 {
					t.Errorf("%s: P=%v L=%v outside [0, 1]", id, s.P, s.L)
				}
			}
		})
	}
}

func TestScoreCoversEveryModelAndIsDeterministic(t *testing.T) {
	a := loadTiny(t)
	msgs := []features.Message{{Role: "system", Text: "Be brief."}, {Role: "user", Text: "Why is the sky blue?"}}

	first := a.Score(msgs)
	second := a.Score(msgs)

	if len(first.Models) != 2 || first.OODScore != second.OODScore ||
		first.Models["tiny-weak"] != second.Models["tiny-weak"] ||
		first.Models["tiny-strong"] != second.Models["tiny-strong"] {
		t.Errorf("Score is not deterministic or misses a model: %+v vs %+v", first, second)
	}
}

func TestScoreIsSafeForConcurrentUse(t *testing.T) {
	a := loadTiny(t)
	msgs := []features.Message{{Role: "user", Text: "What is 17 times 23?"}}
	want := a.Score(msgs)
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				got := a.Score(msgs)
				if got.OODScore != want.OODScore || got.Models["tiny-weak"] != want.Models["tiny-weak"] {
					t.Error("concurrent Score returned a different result")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestIsotonicApplyFollowsNumpyInterp(t *testing.T) {
	// xs and ys are in probability space: apply(s) = interp(sigmoid(s), xs, ys).
	c, err := newCalibrator(calibratorJSON{Type: "isotonic", X: []float64{0.2, 0.5, 0.8}, Y: []float64{0.1, 0.4, 0.9}})
	if err != nil {
		t.Fatal(err)
	}
	logit := func(p float64) float64 { return math.Log(p / (1 - p)) }
	tests := []struct {
		name string
		s    float64
		want float64
	}{
		{"below the first knot", logit(0.05), 0.1},
		{"at the first knot", logit(0.2), 0.1},
		{"between knots", logit(0.35), 0.1 + (0.4-0.1)*(0.35-0.2)/(0.5-0.2)},
		{"at an inner knot", logit(0.5), 0.4},
		{"between the last knots", logit(0.65), 0.4 + (0.9-0.4)*(0.65-0.5)/(0.8-0.5)},
		{"at the last knot", logit(0.8), 0.9},
		{"above the last knot", logit(0.99), 0.9},
		{"huge positive score", 1e9, 0.9},
		{"huge negative score", -1e9, 0.1},
	}
	for _, tt := range tests {
		if got := c.apply(tt.s); math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("%s: apply(%v) = %.15g, want %.15g", tt.name, tt.s, got, tt.want)
		}
	}
	if got := c.apply(math.NaN()); got != 0.1 {
		t.Errorf("apply(NaN) = %v, want the first value 0.1 and no panic", got)
	}

	one, err := newCalibrator(calibratorJSON{Type: "isotonic", X: []float64{0.5}, Y: []float64{0.3}})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []float64{-5, 0, 5} {
		if got := one.apply(s); got != 0.3 {
			t.Errorf("single knot: apply(%v) = %v, want 0.3", s, got)
		}
	}
}

func TestPlattApply(t *testing.T) {
	c, err := newCalibrator(calibratorJSON{Type: "platt", A: 2, B: -1})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []float64{-3, 0, 0.5, 4} {
		want := 1 / (1 + math.Exp(-(2*s - 1)))
		if got := c.apply(s); math.Abs(got-want) > 1e-15 {
			t.Errorf("apply(%v) = %.17g, want %.17g", s, got, want)
		}
	}
}

func TestLCBLookupUsesTheFirstBinAtLeastP(t *testing.T) {
	m := &modelParams{lcbUpper: []float64{0.3, 0.3, 0.6, 1.0}, lcbLower: []float64{0.1, 0.15, 0.4, 0.7}}
	tests := []struct {
		p    float64
		want float64
	}{
		{0, 0.1}, {0.3, 0.1}, {0.30000001, 0.4}, {0.6, 0.4}, {0.61, 0.7}, {1, 0.7}, {1.5, 0.7},
		{math.NaN(), 0.1}, // the most conservative bound
	}
	for _, tt := range tests {
		if got := m.lookup(tt.p); got != tt.want {
			t.Errorf("lookup(%v) = %v, want %v", tt.p, got, tt.want)
		}
	}
}

func TestOODScoreAgainstHandMadeCentroids(t *testing.T) {
	o := oodParams{centroids: []centroid{
		newTestCentroid([]int32{1, 4}, []float64{0.6, 0.8}),
		newTestCentroid([]int32{2}, []float64{1}),
		{}, // a zero centroid has cosine 0
	}}
	// x equal to the first centroid.
	if got := o.score([]int32{1, 4}, []float64{0.6, 0.8}); math.Abs(got) > 1e-15 {
		t.Errorf("score of a centroid = %v, want 0", got)
	}
	// x equal to the second centroid.
	if got := o.score([]int32{2}, []float64{1}); math.Abs(got) > 1e-15 {
		t.Errorf("score of a centroid = %v, want 0", got)
	}
	// x = e1 overlaps the first centroid by 0.6.
	if got := o.score([]int32{1}, []float64{1}); math.Abs(got-0.4) > 1e-15 {
		t.Errorf("score = %v, want 0.4", got)
	}
	// Disjoint support and the empty vector are maximally far.
	if got := o.score([]int32{9}, []float64{1}); got != 1 {
		t.Errorf("disjoint score = %v, want 1", got)
	}
	if got := o.score(nil, nil); got != 1 {
		t.Errorf("empty score = %v, want 1", got)
	}
}

func newTestCentroid(idx []int32, val []float64) centroid {
	var sq float64
	for _, v := range val {
		sq += v * v
	}
	return centroid{idx: idx, val: val, norm: math.Sqrt(sq)}
}

// syntheticArtifact builds an in-memory artifact of the production shape (dim 65536,
// 12 models, 32 centroids of 512 entries) for the benchmark. Its numbers are random.
func syntheticArtifact(dim, nModels int) *Artifact {
	rng := rand.New(rand.NewPCG(1, 2))
	a := &Artifact{version: "bench", dim: dim, byID: map[string]int{}}
	for i := range a.std {
		a.std[i] = 1
	}
	weights := func() []float32 {
		w := make([]float32, dim+features.NumDense)
		for i := range w {
			w[i] = float32(rng.NormFloat64() * 0.1)
		}
		return w
	}
	for i := range nModels {
		xs, ys := make([]float64, 200), make([]float64, 200)
		for k := range xs {
			xs[k] = float64(k+1) / 201
			ys[k] = xs[k]
		}
		id := string(rune('a' + i))
		a.byID[id] = i
		a.models = append(a.models, modelParams{
			id:       id,
			qualityW: weights(),
			lengthW:  weights(),
			cal:      calibrator{kind: "isotonic", xs: xs, ys: ys},
			lcbUpper: []float64{0.2, 0.4, 0.6, 0.8, 1},
			lcbLower: []float64{0.1, 0.3, 0.5, 0.7, 0.9},
			smear:    1.1,
			resid:    [4]float64{0, 0.5, 0.8, 1.2},
			inRatio:  0.3,
		})
	}
	for range 32 {
		idx := make([]int32, 0, 512)
		for _, j := range rng.Perm(dim)[:512] {
			idx = append(idx, int32(j))
		}
		slices.Sort(idx)
		val := make([]float64, len(idx))
		for k := range val {
			val[k] = rng.Float64()
		}
		a.ood.centroids = append(a.ood.centroids, newTestCentroid(idx, val))
	}
	a.ood.threshold = 0.9
	return a
}

// BenchmarkScore times scoring one request of about 1 KiB against a production-shaped
// artifact. It is reported, not asserted.
func BenchmarkScore(b *testing.B) {
	a := syntheticArtifact(features.DefaultDim, 12)
	text := strings.Repeat("Explain how a binary search tree stays balanced after many insertions. ", 15)
	msgs := []features.Message{{Role: "system", Text: "Be concise."}, {Role: "user", Text: text}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = a.Score(msgs)
	}
}
