package decisionlog

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// line is a small record whose encoded size is predictable.
type line struct {
	Type string `json:"type"`
	N    int    `json:"n"`
	Pad  string `json:"pad"`
}

func rec(n int) line { return line{Type: "decision", N: n, Pad: strings.Repeat("x", 40)} }

// readLines returns the names of the rotated files and the lines of every file of the log in
// write order: rotated files by name (their names sort by time), then the active file.
func readLines(t *testing.T, dir, active string) (rotated []string, lines []string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		if e.Name() != active {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	for _, name := range append(append([]string(nil), names...), active) {
		f, err := os.Open(filepath.Join(dir, name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			lines = append(lines, sc.Text())
		}
		_ = f.Close()
	}
	return names, lines
}

func counters(t *testing.T, lines []string) []int {
	t.Helper()
	out := make([]int, len(lines))
	for i, l := range lines {
		var r line
		if err := json.Unmarshal([]byte(l), &r); err != nil {
			t.Fatalf("line %d is not JSON: %v: %q", i, err, l)
		}
		out[i] = r.N
	}
	return out
}

func TestWriterWritesOneJSONLinePerRecord(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(filepath.Join(dir, "decisions.jsonl"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 5 {
		w.Write(rec(i))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	rotated, lines := readLines(t, dir, "decisions.jsonl")
	if len(rotated) != 0 {
		t.Errorf("rotated files = %v, want none under the limit", rotated)
	}
	if got := counters(t, lines); fmt.Sprint(got) != "[0 1 2 3 4]" {
		t.Errorf("records = %v, want 0..4 in order", got)
	}
	if w.Dropped() != 0 || w.Errors() != 0 {
		t.Errorf("Dropped = %d, Errors = %d, want 0, 0", w.Dropped(), w.Errors())
	}
}

// TestWriterRotates: once the file passes maxBytes it is renamed to <base>.<UTC time>.jsonl
// and a new file starts, with no record split or lost, also when the clock does not move
// between rotations (the names must still be distinct, so nothing is overwritten).
func TestWriterRotates(t *testing.T) {
	dir := t.TempDir()
	const (
		max   = 300
		total = 40
	)
	frozen := time.Date(2026, 10, 28, 1, 2, 3, 0, time.UTC)
	w, err := open(filepath.Join(dir, "decisions.jsonl"), max, queueSize, func() time.Time { return frozen }, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range total {
		w.Write(rec(i))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	rotated, lines := readLines(t, dir, "decisions.jsonl")
	if len(rotated) < 3 {
		t.Fatalf("rotated files = %v, want several for %d records at max %d", rotated, total, max)
	}
	name := regexp.MustCompile(`^decisions\.\d{8}T\d{6}Z\.jsonl$`)
	for _, n := range rotated {
		if !name.MatchString(n) {
			t.Errorf("rotated file %q does not match decisions.<yyyymmddTHHMMSSZ>.jsonl", n)
		}
	}
	if !contains(rotated, "decisions.20261028T010203Z.jsonl") {
		t.Errorf("rotated files = %v, want the first named for the clock, decisions.20261028T010203Z.jsonl", rotated)
	}
	var want []int
	for i := range total {
		want = append(want, i)
	}
	if got := counters(t, lines); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("records across files = %v, want 0..%d once each, in order", got, total-1)
	}
	recordSize := int64(len(lines[0]) + 1)
	for _, n := range rotated {
		info, err := os.Stat(filepath.Join(dir, n))
		if err != nil {
			t.Fatal(err)
		}
		// A file is rotated once it exceeds the limit: past it, by less than one more record.
		if info.Size() <= max || info.Size() > max+recordSize {
			t.Errorf("rotated file %s is %d bytes, want just over %d", n, info.Size(), max)
		}
	}
	if w.Errors() != 0 {
		t.Errorf("Errors = %d, want 0", w.Errors())
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// TestWriterRotatedNamesFollowTheClock: rotations at different seconds are named for them.
func TestWriterRotatedNamesFollowTheClock(t *testing.T) {
	dir := t.TempDir()
	var mu sync.Mutex
	now := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	clock := func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		now = now.Add(time.Minute)
		return now
	}
	w, err := open(filepath.Join(dir, "log.jsonl"), 100, queueSize, clock, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 6 {
		w.Write(rec(i))
	}
	_ = w.Close()

	rotated, _ := readLines(t, dir, "log.jsonl")
	if len(rotated) < 2 || rotated[0] != "log.20260102T030505Z.jsonl" || rotated[1] != "log.20260102T030605Z.jsonl" {
		t.Errorf("rotated = %v, want log.20260102T030505Z.jsonl then log.20260102T030605Z.jsonl first", rotated)
	}
}

// TestWriterDropsWhenFull: with the drain goroutine stuck, the queue holds exactly
// queueSize records and every further Write is dropped and counted, without blocking.
func TestWriterDropsWhenFull(t *testing.T) {
	dir := t.TempDir()
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	hook := func() {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
	}
	w, err := open(filepath.Join(dir, "decisions.jsonl"), 1<<30, queueSize, time.Now, hook)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(rec(0))
	<-entered // the drain goroutine holds record 0 and is stuck before writing it

	const extra = 7
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 1; i <= queueSize+extra; i++ {
			w.Write(rec(i))
		}
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Write blocked on a full queue")
	}
	if got := w.Dropped(); got != extra {
		t.Errorf("Dropped = %d, want %d: the queue holds %d", got, extra, queueSize)
	}

	close(release)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	_, lines := readLines(t, dir, "decisions.jsonl")
	if len(lines) != 1+queueSize {
		t.Fatalf("%d records on disk, want %d: the stuck one and the %d that fit", len(lines), 1+queueSize, queueSize)
	}
	if got := counters(t, lines); got[0] != 0 || got[len(got)-1] != queueSize {
		t.Errorf("records on disk run %d..%d, want 0..%d: the dropped ones are the newest", got[0], got[len(got)-1], queueSize)
	}
}

func TestQueueSizeIs4096(t *testing.T) {
	if queueSize != 4096 {
		t.Errorf("queueSize = %d, want 4096", queueSize)
	}
}

func TestWriterCloseDrainsAndFlushes(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(filepath.Join(dir, "decisions.jsonl"), 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	const n = 3000
	for i := range n {
		w.Write(rec(i))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	_, lines := readLines(t, dir, "decisions.jsonl")
	if w.Dropped() != 0 || len(lines) != n {
		t.Errorf("%d lines on disk and %d dropped, want all %d records written", len(lines), w.Dropped(), n)
	}
}

func TestWriterCloseIsIdempotentAndLaterWritesAreDropped(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(filepath.Join(dir, "decisions.jsonl"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(rec(1))
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
	w.Write(rec(2)) // must not panic on the closed queue
	if w.Dropped() != 1 {
		t.Errorf("Dropped after a Write on a closed writer = %d, want 1", w.Dropped())
	}
	_, lines := readLines(t, dir, "decisions.jsonl")
	if len(lines) != 1 {
		t.Errorf("%d lines, want only the record written before Close", len(lines))
	}
}

// TestWriterConcurrentWritesAndClose runs under the race detector: many writers racing a
// Close lose or drop records, never panic, and everything accepted is on disk whole.
func TestWriterConcurrentWritesAndClose(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(filepath.Join(dir, "decisions.jsonl"), 2000)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	const writers, each = 16, 200
	for g := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range each {
				w.Write(rec(g*each + i))
			}
		}()
	}
	time.Sleep(2 * time.Millisecond)
	_ = w.Close()
	wg.Wait()

	_, lines := readLines(t, dir, "decisions.jsonl")
	for i, l := range lines {
		var r line
		if err := json.Unmarshal([]byte(l), &r); err != nil {
			t.Fatalf("line %d is not whole JSON: %v: %q", i, err, l)
		}
	}
	if got := len(lines) + int(w.Dropped()); got != writers*each {
		t.Errorf("%d written + %d dropped = %d, want %d", len(lines), w.Dropped(), got, writers*each)
	}
}

// TestWriterAppendsAndRepairsATornLine: an existing log is appended to, and a last line that
// a crash left without its newline does not swallow the first new record.
func TestWriterAppendsAndRepairsATornLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.jsonl")
	if err := os.WriteFile(path, []byte(`{"type":"decision","n":1,"pad":""}`+"\n"+`{"type":"deci`), 0o600); err != nil {
		t.Fatal(err)
	}
	w, err := Open(path, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(rec(2))
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	_, lines := readLines(t, dir, "decisions.jsonl")
	if len(lines) != 3 || lines[1] != `{"type":"deci` {
		t.Fatalf("lines = %q, want the old line, the torn one kept apart, and the new record", lines)
	}
	var r line
	if err := json.Unmarshal([]byte(lines[2]), &r); err != nil || r.N != 2 {
		t.Errorf("new record = %q (%v), want whole JSON with n=2", lines[2], err)
	}
}

func TestOpenRotatesAnOversizedExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.jsonl")
	old := strings.Repeat(`{"type":"decision","n":0,"pad":""}`+"\n", 20)
	if err := os.WriteFile(path, []byte(old), 0o600); err != nil {
		t.Fatal(err)
	}
	w, err := open(path, 100, queueSize, func() time.Time { return time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC) }, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The oversized file is moved aside by Open itself, before any record is written.
	if info, err := os.Stat(filepath.Join(dir, "decisions.20260506T070809Z.jsonl")); err != nil || info.Size() != int64(len(old)) {
		t.Fatalf("after Open the old file should be rotated whole: %v (%v)", info, err)
	}
	if info, err := os.Stat(path); err != nil || info.Size() != 0 {
		t.Fatalf("after Open the active file should be new and empty: %v (%v)", info, err)
	}
	w.Write(rec(1))
	_ = w.Close()

	rotated, lines := readLines(t, dir, "decisions.jsonl")
	if len(rotated) != 1 || rotated[0] != "decisions.20260506T070809Z.jsonl" {
		t.Errorf("rotated = %v, want the oversized old file moved aside", rotated)
	}
	if len(lines) != 21 {
		t.Errorf("%d lines, want the 20 old and 1 new", len(lines))
	}
	active, _ := os.ReadFile(path)
	if n := strings.Count(string(active), "\n"); n != 1 {
		t.Errorf("active file has %d lines, want only the new record", n)
	}
}

func TestOpenRejects(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "afile")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		path string
		max  int64
	}{
		"empty path":          {"", 100},
		"zero max bytes":      {filepath.Join(dir, "a.jsonl"), 0},
		"negative max bytes":  {filepath.Join(dir, "a.jsonl"), -5},
		"parent is a file":    {filepath.Join(file, "a.jsonl"), 100},
		"path is a directory": {dir, 100},
	} {
		if w, err := Open(c.path, c.max); err == nil {
			_ = w.Close()
			t.Errorf("Open(%s) succeeded, want an error", name)
		}
	}
}

func TestOpenCreatesTheParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a", "b", "decisions.jsonl")
	w, err := Open(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(rec(1))
	_ = w.Close()
	if _, err := os.Stat(path); err != nil {
		t.Errorf("log not created: %v", err)
	}
}

// TestWriteOfAnUnencodableRecordIsCounted: a record that cannot be encoded is an error counted
// by Errors, not a panic and not a drop.
func TestWriteOfAnUnencodableRecordIsCounted(t *testing.T) {
	w, err := Open(filepath.Join(t.TempDir(), "decisions.jsonl"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	w.Write(make(chan int))
	w.Write(struct{ V float64 }{math.NaN()})
	if w.Errors() != 2 || w.Dropped() != 0 {
		t.Errorf("Errors = %d, Dropped = %d, want 2, 0", w.Errors(), w.Dropped())
	}
}

func TestRecordJSONShape(t *testing.T) {
	p, l, qms := 0.75, 0.5, 120.5
	cost := int64(42)
	ood, exp := false, true
	prop, eps := 0.95, 0.05
	r := Record{
		Type: "decision", DecisionID: "dec_abc12345", Time: time.Date(2026, 10, 28, 1, 2, 3, 400000000, time.UTC),
		Tenant: "t", PolicyVersion: "v1", PolicyRole: "stable", PromptSHA16: "0123456789abcdef", PromptBytes: 12,
		Required: []string{"tools"}, OODScore: &l, OOD: &ood,
		Candidates: []Candidate{{Model: "m", P: &p, L: &l, CostExpPicoUSD: &cost, CostMaxPicoUSD: &cost, QMs: &qms, Feasible: true, Reasons: []string{}}},
		Chosen:     "m", Propensity: &prop, Epsilon: &eps, Explored: &exp, Relaxed: []string{"latency"},
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"type", "decision_id", "time", "tenant", "policy_version", "policy_role", "prompt_sha256_16",
		"prompt_bytes", "required", "ood_score", "ood", "candidates", "chosen", "propensity", "epsilon", "explored", "relaxed"} {
		if _, ok := got[key]; !ok {
			t.Errorf("decision record has no %q: %s", key, b)
		}
	}
	outcomeKeys := []string{"status", "latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "cost_actual_picousd", "usage_estimated", "attempts"}
	for _, key := range outcomeKeys {
		if _, ok := got[key]; ok {
			t.Errorf("a record without an outcome (a shadow decision) has %q: %s", key, b)
		}
	}
	cand := got["candidates"].([]any)[0].(map[string]any)
	for _, key := range []string{"model", "p", "l", "cost_exp_picousd", "cost_max_picousd", "q_ms", "feasible", "reasons"} {
		if _, ok := cand[key]; !ok {
			t.Errorf("candidate has no %q: %s", key, b)
		}
	}
	if got["time"] != "2026-10-28T01:02:03.4Z" {
		t.Errorf("time = %v, want RFC 3339 UTC", got["time"])
	}

	r.Outcome = &Outcome{Status: 200, LatencyMs: 12, PromptTokens: 5, CompletionTokens: 4, CostActualPicoUSD: 9, Attempts: []Attempt{{Deployment: "d", Outcome: "ok"}}}
	b, _ = json.Marshal(r)
	got = map[string]any{}
	_ = json.Unmarshal(b, &got)
	for _, key := range outcomeKeys {
		if _, ok := got[key]; !ok {
			t.Errorf("a record with an outcome has no %q: %s", key, b)
		}
	}
	if got["ttft_ms"] != nil {
		t.Errorf("ttft_ms = %v, want null when unset", got["ttft_ms"])
	}
}

// TestRecordFieldsMatchThePythonLoader guards against drift between the record written here and
// the columns python/routepilot_train/logs.py reads: every key of a decision record is a column
// of the loader, and every column but the two joined from feedback is a key.
func TestRecordFieldsMatchThePythonLoader(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "python", "routepilot_train", "logs.py"))
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)\nCOLUMNS = \[(.*?)\n\]`).FindSubmatch(src)
	if block == nil {
		t.Fatal("COLUMNS list not found in logs.py")
	}
	columns := map[string]bool{}
	for _, m := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllSubmatch(block[1], -1) {
		columns[string(m[1])] = true
	}
	delete(columns, "feedback_success")
	delete(columns, "feedback_time")

	f, i, b := 0.5, int64(1), true
	full := Record{
		Type: "decision", OODScore: &f, OOD: &b, Propensity: &f, Epsilon: &f, Explored: &b,
		Required: []string{}, Candidates: []Candidate{}, Relaxed: []string{},
		Outcome: &Outcome{TTFTMs: &i, Attempts: []Attempt{}},
	}
	raw, err := json.Marshal(full)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]any
	if err := json.Unmarshal(raw, &keys); err != nil {
		t.Fatal(err)
	}
	delete(keys, "type")
	for k := range keys {
		if !columns[k] {
			t.Errorf("record key %q is not a column of logs.py", k)
		}
	}
	for c := range columns {
		if _, ok := keys[c]; !ok {
			t.Errorf("logs.py column %q is not a key of a decision record", c)
		}
	}
}

func TestFeedbackJSONShape(t *testing.T) {
	b, err := json.Marshal(Feedback{Type: "feedback", DecisionID: "dec_abc12345", Tenant: "t", Success: true, Time: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"feedback","decision_id":"dec_abc12345","tenant":"t","success":true,"time":"2026-01-02T03:04:05Z"}`; string(b) != want {
		t.Errorf("feedback = %s, want %s", b, want)
	}
}
