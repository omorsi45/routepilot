package openai

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"
)

// readAll drains an SSEReader until it returns an error and reports that error.
func readAll(t *testing.T, r *SSEReader) ([]Event, error) {
	t.Helper()
	var events []Event
	for i := 0; i < 1000; i++ {
		ev, err := r.Next()
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	t.Fatal("reader did not terminate")
	return nil, nil
}

func TestSSEReaderBasic(t *testing.T) {
	r := NewSSEReader(strings.NewReader("data: {\"a\":1}\n\ndata: [DONE]\n\n"), 0)

	ev, err := r.Next()
	if err != nil {
		t.Fatalf("first Next: %v", err)
	}
	if string(ev.Data) != `{"a":1}` || ev.Name != "" || ev.IsDone() {
		t.Fatalf("first event = %+v", ev)
	}

	ev, err = r.Next()
	if err != nil {
		t.Fatalf("second Next: %v", err)
	}
	if !ev.IsDone() {
		t.Fatalf("second event should be done, got %+v", ev)
	}

	if _, err = r.Next(); err != io.EOF {
		t.Fatalf("third Next err = %v, want io.EOF", err)
	}
	if _, err = r.Next(); err != io.EOF {
		t.Fatalf("Next after EOF err = %v, want io.EOF", err)
	}
}

func TestSSEReaderOneByteReads(t *testing.T) {
	const stream = "data: {\"a\":1}\n\nevent: x\ndata: p\ndata: q\n\ndata: [DONE]\n\n"

	want, wantErr := readAll(t, NewSSEReader(strings.NewReader(stream), 0))
	got, gotErr := readAll(t, NewSSEReader(iotest.OneByteReader(strings.NewReader(stream)), 0))

	if wantErr != io.EOF || gotErr != io.EOF {
		t.Fatalf("errors = %v, %v; want io.EOF both", wantErr, gotErr)
	}
	if len(want) != 3 || len(got) != len(want) {
		t.Fatalf("event counts: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		if want[i].Name != got[i].Name || !bytes.Equal(want[i].Data, got[i].Data) {
			t.Fatalf("event %d differs: %+v vs %+v", i, want[i], got[i])
		}
	}
}

func TestSSEReaderCRLFCommentsAndNames(t *testing.T) {
	r := NewSSEReader(strings.NewReader(": ping\r\nevent: message_start\r\ndata: x\r\n\r\n"), 0)
	ev, err := r.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if ev.Name != "message_start" || string(ev.Data) != "x" {
		t.Fatalf("event = %+v", ev)
	}
	if _, err = r.Next(); err != io.EOF {
		t.Fatalf("err = %v, want io.EOF", err)
	}
}

func TestSSEReaderMultiLineData(t *testing.T) {
	r := NewSSEReader(strings.NewReader("data: a\ndata: b\n\n"), 0)
	ev, err := r.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if string(ev.Data) != "a\nb" {
		t.Fatalf("data = %q, want %q", ev.Data, "a\nb")
	}
}

func TestSSEReaderFieldParsingRules(t *testing.T) {
	tests := []struct {
		name   string
		in     string
		events []Event
	}{
		{"one leading space stripped only", "data:  two\n\n", []Event{{Data: []byte(" two")}}},
		{"no space after colon", "data:x\n\n", []Event{{Data: []byte("x")}}},
		{"data without colon is empty line", "data\ndata: b\n\n", []Event{{Data: []byte("\nb")}}},
		{"empty data value", "data:\n\n", []Event{{Data: []byte("")}}},
		{"id and retry ignored", "id: 7\nretry: 100\ndata: x\n\n", []Event{{Data: []byte("x")}}},
		{"unknown field ignored", "foo: bar\ndata: x\n\n", []Event{{Data: []byte("x")}}},
		{"comment-only block not dispatched", ": hi\n\ndata: x\n\n", []Event{{Data: []byte("x")}}},
		{"blank lines alone not dispatched", "\n\n\ndata: x\n\n", []Event{{Data: []byte("x")}}},
		{"name without data is dispatched", "event: ping\n\n", []Event{{Name: "ping"}}},
		{"final event without blank line", "data: x", []Event{{Data: []byte("x")}}},
		{"final event with line end but no blank line", "data: x\n", []Event{{Data: []byte("x")}}},
		{"trailing comment-only block at EOF", "data: x\n\n: bye", []Event{{Data: []byte("x")}}},
		{"trailing name-only block at EOF dropped", "data: x\n\nevent: y", []Event{{Data: []byte("x")}}},
		{"empty input", "", nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := readAll(t, NewSSEReader(strings.NewReader(tc.in), 0))
			if err != io.EOF {
				t.Fatalf("err = %v, want io.EOF", err)
			}
			if len(got) != len(tc.events) {
				t.Fatalf("got %d events %+v, want %d", len(got), got, len(tc.events))
			}
			for i := range got {
				if got[i].Name != tc.events[i].Name || !bytes.Equal(got[i].Data, tc.events[i].Data) {
					t.Fatalf("event %d = %+v (data %q), want %+v", i, got[i], got[i].Data, tc.events[i])
				}
			}
		})
	}
}

func TestSSEReaderReturnsReadErrorAsIs(t *testing.T) {
	boom := errors.New("boom")
	src := io.MultiReader(strings.NewReader("data: x\n\ndata: y"), iotest.ErrReader(boom))
	r := NewSSEReader(src, 0)

	if ev, err := r.Next(); err != nil || string(ev.Data) != "x" {
		t.Fatalf("first = %+v, %v", ev, err)
	}
	if _, err := r.Next(); err != boom {
		t.Fatalf("err = %v, want boom", err)
	}
}

// endlessReader yields the same byte forever and counts how much was read.
type endlessReader struct {
	b    byte
	read int
}

func (e *endlessReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = e.b
	}
	e.read += len(p)
	return len(p), nil
}

// prefixThenEndless yields a fixed prefix, then an endless stream of one byte.
func prefixThenEndless(prefix string, b byte) (io.Reader, *endlessReader) {
	e := &endlessReader{b: b}
	return io.MultiReader(strings.NewReader(prefix), e), e
}

func TestSSEReaderEventTooLarge(t *testing.T) {
	src, e := prefixThenEndless("data: ", 'x')
	r := NewSSEReader(src, 1024)

	_, err := r.Next()
	if !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("err = %v, want ErrEventTooLarge", err)
	}
	// Must stop reading soon after the budget: limit plus a few buffers at most.
	if e.read > 1024+3*4096 {
		t.Fatalf("read %d bytes from source, expected a bound near the 1024 limit", e.read)
	}
	// The failure is sticky.
	if _, err = r.Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("second err = %v, want ErrEventTooLarge", err)
	}
}

func TestSSEReaderEventTooLargeAcrossLines(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 100; i++ {
		sb.WriteString("data: 0123456789\n")
	}
	sb.WriteString("\n")
	r := NewSSEReader(strings.NewReader(sb.String()), 512)
	if _, err := r.Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("err = %v, want ErrEventTooLarge", err)
	}
}

func TestSSEReaderBudgetBoundary(t *testing.T) {
	// "data: " (6) + 10 payload + "\n" (1) + "\n" (1) = 18 bytes.
	const stream = "data: 0123456789\n\n"
	if _, err := NewSSEReader(strings.NewReader(stream), 18).Next(); err != nil {
		t.Fatalf("event at exactly the limit: %v", err)
	}
	if _, err := NewSSEReader(strings.NewReader(stream), 17).Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("event one byte over the limit: err = %v, want ErrEventTooLarge", err)
	}
}

func TestSSEReaderBudgetResetsPerEvent(t *testing.T) {
	const one = "data: 0123456789\n\n" // 18 bytes
	r := NewSSEReader(strings.NewReader(strings.Repeat(one, 50)), 18)
	events, err := readAll(t, r)
	if err != io.EOF || len(events) != 50 {
		t.Fatalf("got %d events, err %v; want 50, io.EOF", len(events), err)
	}
}

func TestSSEReaderEndlessCommentsBounded(t *testing.T) {
	src, _ := prefixThenEndless(": ", 'c')
	if _, err := NewSSEReader(src, 1024).Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("err = %v, want ErrEventTooLarge", err)
	}
}

func TestSSEReaderDefaultLimit(t *testing.T) {
	src, _ := prefixThenEndless("data: ", 'x')
	if _, err := NewSSEReader(src, 0).Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("zero limit: err = %v, want ErrEventTooLarge", err)
	}
	src, _ = prefixThenEndless("data: ", 'x')
	if _, err := NewSSEReader(src, -5).Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("negative limit: err = %v, want ErrEventTooLarge", err)
	}
	// A payload larger than a small explicit limit but smaller than MaxEventBytes passes with the default.
	big := "data: " + strings.Repeat("y", 100_000) + "\n\n"
	ev, err := NewSSEReader(strings.NewReader(big), 0).Next()
	if err != nil || len(ev.Data) != 100_000 {
		t.Fatalf("default limit rejected 100000-byte event: len %d, err %v", len(ev.Data), err)
	}
}

func TestEventIsDone(t *testing.T) {
	if !(Event{Data: []byte("[DONE]")}).IsDone() {
		t.Fatal("[DONE] should be done")
	}
	for _, d := range []string{"", "[DONE] ", "done", "{\"a\":1}"} {
		if (Event{Data: []byte(d)}).IsDone() {
			t.Fatalf("%q should not be done", d)
		}
	}
}

type flushRecorder struct {
	buf     bytes.Buffer
	flushes int
	// writesAtFlush records how many bytes had been written at each flush.
	writesAtFlush []int
}

func (f *flushRecorder) Write(p []byte) (int, error) { return f.buf.Write(p) }
func (f *flushRecorder) flush() error {
	f.flushes++
	f.writesAtFlush = append(f.writesAtFlush, f.buf.Len())
	return nil
}

func TestSSEWriterFormatsAndFlushes(t *testing.T) {
	rec := &flushRecorder{}
	w := NewSSEWriter(rec, rec.flush)

	if err := w.WriteData([]byte(`{"a":1}`)); err != nil {
		t.Fatalf("WriteData: %v", err)
	}
	if rec.flushes != 1 {
		t.Fatalf("flushes after WriteData = %d, want 1", rec.flushes)
	}
	if err := w.WriteDone(); err != nil {
		t.Fatalf("WriteDone: %v", err)
	}
	if rec.flushes != 2 {
		t.Fatalf("flushes after WriteDone = %d, want 2", rec.flushes)
	}

	const want = "data: {\"a\":1}\n\ndata: [DONE]\n\n"
	if rec.buf.String() != want {
		t.Fatalf("output = %q, want %q", rec.buf.String(), want)
	}
	if rec.writesAtFlush[0] != len("data: {\"a\":1}\n\n") {
		t.Fatalf("first flush happened after %d bytes", rec.writesAtFlush[0])
	}
}

func TestSSEWriterRejectsNewlines(t *testing.T) {
	for _, in := range []string{"a\nb", "a\r\nb", "a\rb", "\n"} {
		rec := &flushRecorder{}
		w := NewSSEWriter(rec, rec.flush)
		if err := w.WriteData([]byte(in)); err == nil {
			t.Fatalf("WriteData(%q) should fail", in)
		}
		if rec.buf.Len() != 0 || rec.flushes != 0 {
			t.Fatalf("WriteData(%q) wrote %q and flushed %d times", in, rec.buf.String(), rec.flushes)
		}
	}
}

type failingWriter struct{ err error }

func (f failingWriter) Write([]byte) (int, error) { return 0, f.err }

func TestSSEWriterPropagatesErrors(t *testing.T) {
	boom := errors.New("boom")

	w := NewSSEWriter(failingWriter{boom}, func() error { t.Fatal("flush must not run after a failed write"); return nil })
	if err := w.WriteData([]byte("x")); !errors.Is(err, boom) {
		t.Fatalf("write err = %v, want boom", err)
	}

	var sink bytes.Buffer
	w = NewSSEWriter(&sink, func() error { return boom })
	if err := w.WriteData([]byte("x")); !errors.Is(err, boom) {
		t.Fatalf("flush err = %v, want boom", err)
	}
	if err := w.WriteDone(); !errors.Is(err, boom) {
		t.Fatalf("WriteDone flush err = %v, want boom", err)
	}
}

func TestSSEWriterNilFlush(t *testing.T) {
	var sink bytes.Buffer
	w := NewSSEWriter(&sink, nil)
	if err := w.WriteData([]byte("x")); err != nil {
		t.Fatalf("WriteData: %v", err)
	}
	if sink.String() != "data: x\n\n" {
		t.Fatalf("output = %q", sink.String())
	}
}

func TestSSEWriterRoundTripsThroughReader(t *testing.T) {
	var sink bytes.Buffer
	w := NewSSEWriter(&sink, nil)
	_ = w.WriteData([]byte(`{"n":1}`))
	_ = w.WriteData([]byte(`{"n":2}`))
	_ = w.WriteDone()

	events, err := readAll(t, NewSSEReader(&sink, 0))
	if err != io.EOF || len(events) != 3 {
		t.Fatalf("got %d events, err %v", len(events), err)
	}
	if string(events[1].Data) != `{"n":2}` || !events[2].IsDone() {
		t.Fatalf("events = %+v", events)
	}
}
