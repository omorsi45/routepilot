package features

import (
	"encoding/json"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/openai"
)

const tolerance = 1e-12

var testdataDir = filepath.Join("..", "..", "testdata", "features")

// The case file is shared with the Python tests: a message has either a plain
// text or a repeat spec (text = pattern * count + suffix).
type caseFile struct {
	SpecVersion int        `json:"spec_version"`
	Cases       []testCase `json:"cases"`
}

type testCase struct {
	Name     string        `json:"name"`
	Dim      *int          `json:"dim"`
	Messages []caseMessage `json:"messages"`
}

type caseMessage struct {
	Role   string      `json:"role"`
	Text   *string     `json:"text"`
	Repeat *repeatSpec `json:"repeat"`
}

type repeatSpec struct {
	Pattern string `json:"pattern"`
	Count   int    `json:"count"`
	Suffix  string `json:"suffix"`
}

type goldenFile struct {
	SpecVersion int          `json:"spec_version"`
	Cases       []goldenCase `json:"cases"`
}

type goldenCase struct {
	Name  string    `json:"name"`
	Dim   int       `json:"dim"`
	Idx   []int32   `json:"idx"`
	Val   []float64 `json:"val"`
	Dense []float64 `json:"dense"`
}

func readJSON(t *testing.T, name string, into any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(testdataDir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
}

func loadCases(t *testing.T) []testCase {
	t.Helper()
	var f caseFile
	readJSON(t, "cases.json", &f)
	if f.SpecVersion != SpecVersion {
		t.Fatalf("cases.json spec_version = %d, want %d", f.SpecVersion, SpecVersion)
	}
	return f.Cases
}

func (c testCase) dim() int {
	if c.Dim != nil {
		return *c.Dim
	}
	return DefaultDim
}

func (c testCase) messages(t *testing.T) []Message {
	t.Helper()
	msgs := make([]Message, 0, len(c.Messages))
	for _, m := range c.Messages {
		switch {
		case m.Text != nil && m.Repeat == nil:
			msgs = append(msgs, Message{Role: m.Role, Text: *m.Text})
		case m.Repeat != nil && m.Text == nil:
			text := strings.Repeat(m.Repeat.Pattern, m.Repeat.Count) + m.Repeat.Suffix
			msgs = append(msgs, Message{Role: m.Role, Text: text})
		default:
			t.Fatalf("case %q: a message needs exactly one of text and repeat", c.Name)
		}
	}
	return msgs
}

func caseByName(t *testing.T, name string) testCase {
	t.Helper()
	for _, c := range loadCases(t) {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no case named %q", name)
	return testCase{}
}

func denseOf(text string) [NumDense]float64 {
	return Extract([]Message{{Role: "user", Text: text}}, DefaultDim).Dense
}

func TestGoldenVectors(t *testing.T) {
	cases := loadCases(t)
	var golden goldenFile
	readJSON(t, "golden.json", &golden)
	if golden.SpecVersion != SpecVersion {
		t.Fatalf("golden.json spec_version = %d, want %d", golden.SpecVersion, SpecVersion)
	}
	if len(golden.Cases) != len(cases) {
		t.Fatalf("golden.json has %d cases, cases.json has %d", len(golden.Cases), len(cases))
	}
	for i, c := range cases {
		want := golden.Cases[i]
		t.Run(c.Name, func(t *testing.T) {
			if want.Name != c.Name {
				t.Fatalf("golden case %d is %q, want %q", i, want.Name, c.Name)
			}
			if want.Dim != c.dim() {
				t.Fatalf("golden dim = %d, want %d", want.Dim, c.dim())
			}
			got := Extract(c.messages(t), c.dim())
			if !slices.Equal(got.Idx, want.Idx) {
				t.Fatalf("Idx differ: got %d entries, want %d\n got %v\nwant %v", len(got.Idx), len(want.Idx), got.Idx, want.Idx)
			}
			if len(got.Val) != len(want.Val) {
				t.Fatalf("Val has %d entries, want %d", len(got.Val), len(want.Val))
			}
			for j := range got.Val {
				if d := math.Abs(got.Val[j] - want.Val[j]); d > tolerance {
					t.Errorf("Val[%d] (idx %d) = %v, want %v (diff %g)", j, got.Idx[j], got.Val[j], want.Val[j], d)
				}
			}
			if len(want.Dense) != NumDense {
				t.Fatalf("golden dense has %d entries, want %d", len(want.Dense), NumDense)
			}
			for j := range got.Dense {
				if d := math.Abs(got.Dense[j] - want.Dense[j]); d > tolerance {
					t.Errorf("Dense[%d] = %v, want %v (diff %g)", j, got.Dense[j], want.Dense[j], d)
				}
			}
		})
	}
}

func TestFNV1a64KnownValues(t *testing.T) {
	// Published FNV-1a 64 test vectors.
	tests := []struct {
		in   string
		want uint64
	}{
		{"", 0xcbf29ce484222325},
		{"a", 0xaf63dc4c8601ec8c},
		{"foobar", 0x85944171f73967e8},
	}
	for _, tc := range tests {
		if got := fnv1a64([]byte(tc.in)); got != tc.want {
			t.Errorf("fnv1a64(%q) = %#x, want %#x", tc.in, got, tc.want)
		}
	}
}

func TestFNV1a64MatchesStdlib(t *testing.T) {
	inputs := []string{"u:hello", "b:go go", "u:\xc3\xa9", "b:caf \xc3\xa9", strings.Repeat("xyz", 1000)}
	for _, in := range inputs {
		h := fnv.New64a()
		_, _ = h.Write([]byte(in))
		if got, want := fnv1a64([]byte(in)), h.Sum64(); got != want {
			t.Errorf("fnv1a64(%q) = %#x, stdlib %#x", in, got, want)
		}
	}
}

func TestMessagesFromRequest(t *testing.T) {
	body := `{"model":"auto","messages":[
		{"role":"system","content":"be brief"},
		{"role":"user","content":[
			{"type":"text","text":"look at this"},
			{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}},
			{"type":"text","text":"and answer"}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"user","content":"thanks"}]}`
	var req openai.ChatCompletionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	want := []Message{
		{Role: "system", Text: "be brief"},
		{Role: "user", Text: "look at this\nand answer"},
		{Role: "assistant", Text: ""},
		{Role: "user", Text: "thanks"},
	}
	got := Messages(&req)
	if !slices.Equal(got, want) {
		t.Fatalf("Messages = %#v, want %#v", got, want)
	}
	if got := Messages(&openai.ChatCompletionRequest{}); len(got) != 0 {
		t.Fatalf("Messages of an empty request = %#v, want none", got)
	}
}

func TestExtractEmptyInputs(t *testing.T) {
	for name, msgs := range map[string][]Message{
		"no messages": nil,
		"empty text":  {{Role: "user", Text: ""}},
		"separators":  {{Role: "user", Text: " \n\t ... "}},
	} {
		v := Extract(msgs, DefaultDim)
		if len(v.Idx) != 0 || len(v.Val) != 0 {
			t.Errorf("%s: got %d indices, want none", name, len(v.Idx))
		}
	}
	if got := Extract(nil, DefaultDim).Dense; got != [NumDense]float64{} {
		t.Errorf("dense of no messages = %v, want all zero", got)
	}
	want := [NumDense]float64{0, math.Log1p(1)}
	if got := denseOf(""); got != want {
		t.Errorf("dense of empty text = %v, want %v", got, want)
	}
}

func TestExtractInvariants(t *testing.T) {
	for _, c := range loadCases(t) {
		v := Extract(c.messages(t), c.dim())
		if len(v.Idx) != len(v.Val) {
			t.Errorf("%s: %d indices but %d values", c.Name, len(v.Idx), len(v.Val))
		}
		var sum float64
		for i, ix := range v.Idx {
			if ix < 0 || int(ix) >= c.dim() {
				t.Errorf("%s: index %d out of range for dim %d", c.Name, ix, c.dim())
			}
			if i > 0 && v.Idx[i-1] >= ix {
				t.Errorf("%s: indices not strictly ascending at %d", c.Name, i)
			}
			sum += v.Val[i] * v.Val[i]
		}
		if len(v.Idx) > 0 && math.Abs(sum-1) > tolerance {
			t.Errorf("%s: squared norm = %v, want 1", c.Name, sum)
		}
	}
}

func TestTokenize(t *testing.T) {
	got := tokenize([]byte("Caf\xc3\xa9 42-x \xc3\xa9, \xc3\x80B"))
	want := []string{"caf", "\xc3\xa9", "42", "x", "\xc3\xa9", "\xc3\x80", "b"}
	if len(got) != len(want) {
		t.Fatalf("tokenize = %q, want %q", got, want)
	}
	for i := range want {
		if string(got[i]) != want[i] {
			t.Fatalf("tokenize = %q, want %q", got, want)
		}
	}
	if got := tokenize(nil); len(got) != 0 {
		t.Errorf("tokenize(nil) = %q, want none", got)
	}
	if got := tokenize([]byte("  ... \n")); len(got) != 0 {
		t.Errorf("tokenize of separators = %q, want none", got)
	}
}

func TestCaseFoldingIsASCIIOnly(t *testing.T) {
	a := Extract([]Message{{Role: "user", Text: "Hello WORLD"}}, DefaultDim)
	b := Extract([]Message{{Role: "user", Text: "hello world"}}, DefaultDim)
	if !slices.Equal(a.Idx, b.Idx) || !slices.Equal(a.Val, b.Val) {
		t.Error("ASCII case must not change the hashed part")
	}
	c := Extract([]Message{{Role: "user", Text: "\xc3\x89"}}, DefaultDim)
	d := Extract([]Message{{Role: "user", Text: "\xc3\xa9"}}, DefaultDim)
	if slices.Equal(c.Idx, d.Idx) {
		t.Error("non-ASCII bytes must be kept raw, not case folded")
	}
}

func TestHashedPartIgnoresBytesPastPrefix(t *testing.T) {
	inside := strings.Repeat("x ", PrefixBytes/2)
	long := inside + strings.Repeat("tail words beyond the prefix ", 50)
	a := Extract([]Message{{Role: "user", Text: inside}}, DefaultDim)
	b := Extract([]Message{{Role: "user", Text: long}}, DefaultDim)
	if !slices.Equal(a.Idx, b.Idx) || !slices.Equal(a.Val, b.Val) {
		t.Error("bytes past the prefix changed the hashed part")
	}
	if !(b.Dense[0] > a.Dense[0]) {
		t.Error("the length feature must still see the whole text")
	}
}

func TestLongCaseCutsMidRuneAndIgnoresTailMarkers(t *testing.T) {
	c := caseByName(t, "long_cut_mid_rune")
	msgs := c.messages(t)
	if len(msgs) != 1 || len(msgs[0].Text) != 40000 {
		t.Fatalf("expanded case has %d messages, first of %d bytes; want one of 40000", len(msgs), len(msgs[0].Text))
	}
	if b := msgs[0].Text[PrefixBytes]; b < 0x80 || b > 0xbf {
		t.Fatalf("byte %d is %#x, want a UTF-8 continuation byte so the cut splits a rune", PrefixBytes, b)
	}
	dense := Extract(msgs, c.dim()).Dense
	if dense[0] != math.Log1p(40000) {
		t.Errorf("dense[0] = %v, want log1p(40000)", dense[0])
	}
	if dense[8] != 0 || dense[11] != 0 {
		t.Errorf("markers past the prefix leaked: dense[8]=%v dense[11]=%v", dense[8], dense[11])
	}
	if dense[2] != math.Log1p(2) {
		t.Errorf("dense[2] = %v, want log1p(2): lines are counted over the full text", dense[2])
	}
}

func TestCancellingCollisionsAreDropped(t *testing.T) {
	c := caseByName(t, "small_dim_collisions")
	msgs := c.messages(t)
	counts := bucketCounts([]byte(msgs[0].Text), c.dim())
	var want []int32
	cancelled := 0
	for ix, n := range counts {
		if n == 0 {
			cancelled++
		} else {
			want = append(want, ix)
		}
	}
	slices.Sort(want)
	if cancelled == 0 {
		t.Fatal("the case must contain a bucket whose signed counts cancel")
	}
	if got := Extract(msgs, c.dim()).Idx; !slices.Equal(got, want) {
		t.Fatalf("Idx = %v, want %v", got, want)
	}
}

func TestChoicesFlag(t *testing.T) {
	tests := []struct {
		text string
		want float64
	}{
		{"A. x", 1},
		{"(b) x", 1},
		{"C: x", 1},
		{"  \tE) x", 1},
		{"(e) x", 1},
		{"intro\nB) yes", 1},
		{"F. x", 0},
		{"A.x", 0},
		{"A.\nB.", 0},
		{"((a) x", 0},
		{"xA. y", 0},
		{"the answer is A. maybe", 0},
		{"a]x", 0},
	}
	for _, tc := range tests {
		if got := denseOf(tc.text)[9]; got != tc.want {
			t.Errorf("choices flag for %q = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func TestMathMarkerFlag(t *testing.T) {
	for _, marker := range []string{`\frac`, `\sum`, `$$`, `\(`, `\[`} {
		if got := denseOf("a " + marker + " b")[8]; got != 1 {
			t.Errorf("marker %q: flag = %v, want 1", marker, got)
		}
	}
	if got := denseOf(`frac sum $ ( [ \ `)[8]; got != 0 {
		t.Errorf("no marker present: flag = %v, want 0", got)
	}
}

func TestSystemFlagFollowsRole(t *testing.T) {
	tests := []struct {
		msgs []Message
		want float64
	}{
		{[]Message{{"system", "x"}}, 1},
		{[]Message{{"user", "x"}, {"system", "y"}}, 1},
		{[]Message{{"user", "x"}, {"assistant", "y"}}, 0},
	}
	for _, tc := range tests {
		if got := Extract(tc.msgs, DefaultDim).Dense[10]; got != tc.want {
			t.Errorf("system flag for %v = %v, want %v", tc.msgs, got, tc.want)
		}
	}
}

func TestDenseCountsAndRatios(t *testing.T) {
	d := denseOf("a1B!\xc3\xa9") // 6 bytes: a 1 B ! 0xC3 0xA9
	want := [NumDense]float64{math.Log1p(6), math.Log1p(1), math.Log1p(1), 1.0 / 6, 1.0 / 6, 2.0 / 6, 1.0 / 6}
	if d != want {
		t.Errorf("dense = %v, want %v", d, want)
	}
}

func TestLinesFencesAndQuestions(t *testing.T) {
	checks := []struct {
		name string
		got  float64
		want float64
	}{
		{"three lines", denseOf("a\nb\nc")[2], math.Log1p(3)},
		{"lone newline", denseOf("\n")[2], math.Log1p(2)},
		{"two fences", denseOf("```a```")[7], math.Log1p(2)},
		{"non-overlapping fence run", denseOf("`````")[7], math.Log1p(1)},
		{"question marks", denseOf("why? how??")[11], math.Log1p(3)},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestAll32PunctuationBytesAreCounted(t *testing.T) {
	var sb strings.Builder
	for b := byte(0x21); b < 0x7f; b++ {
		if !(b >= '0' && b <= '9' || b >= 'A' && b <= 'Z' || b >= 'a' && b <= 'z') {
			sb.WriteByte(b)
		}
	}
	if sb.Len() != 32 {
		t.Fatalf("built %d punctuation bytes, want 32", sb.Len())
	}
	if got := denseOf(sb.String())[6]; got != 1 {
		t.Errorf("punctuation ratio = %v, want 1", got)
	}
}

func TestMessagesJoinWithNewline(t *testing.T) {
	two := Extract([]Message{{"user", "ab"}, {"assistant", "cd"}}, DefaultDim)
	one := Extract([]Message{{"user", "ab\ncd"}}, DefaultDim)
	if !slices.Equal(one.Idx, two.Idx) || !slices.Equal(one.Val, two.Val) {
		t.Error("hashed part must only depend on the joined bytes")
	}
	if two.Dense[0] != math.Log1p(5) || two.Dense[1] != math.Log1p(2) || two.Dense[2] != math.Log1p(2) {
		t.Errorf("dense = %v", two.Dense)
	}
}

func TestInvalidUTF8IsTreatedAsHighBytes(t *testing.T) {
	v := Extract([]Message{{Role: "user", Text: "ok \xff\xfe bad"}}, DefaultDim)
	if len(v.Idx) == 0 {
		t.Fatal("expected hashed features for text with invalid UTF-8")
	}
	if v.Dense[5] != 2.0/9 { // 9 bytes, 2 of them >= 0x80
		t.Errorf("non-ASCII ratio = %v, want 2/9", v.Dense[5])
	}
}

func TestOutOfRangeDimFallsBackToDefault(t *testing.T) {
	msgs := []Message{{Role: "user", Text: "hello world again"}}
	want := Extract(msgs, DefaultDim)
	for _, dim := range []int{0, -5, math.MaxInt32 + 1} {
		got := Extract(msgs, dim)
		if !slices.Equal(got.Idx, want.Idx) || !slices.Equal(got.Val, want.Val) {
			t.Errorf("dim %d: expected the default dimension", dim)
		}
	}
}
