package decisionlog

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// rotatedName matches <base>.<UTC yyyymmddTHHMMSS.uuuuuuZ>.<sequence>.jsonl.
var rotatedName = regexp.MustCompile(`^log\.(\d{8}T\d{6}\.\d{6}Z)\.(\d+)\.jsonl$`)

// TestWriterRotatesManyTimesInOneSecond: at a high request rate and a small max_bytes the log
// rotates several times a second. Every record lands, in order across the files, and no
// rotated file is named for an instant that has not happened yet: the name is the time of
// the rotation with a sequence number, and is never searched for forward in time.
func TestWriterRotatesManyTimesInOneSecond(t *testing.T) {
	const total = 5000
	dir := t.TempDir()
	frozen := time.Date(2026, 10, 28, 1, 2, 3, 456789000, time.UTC)
	// A max of 1 byte rotates after every record.
	w, err := open(filepath.Join(dir, "log.jsonl"), 1, total, func() time.Time { return frozen }, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range total {
		w.Write(rec(i))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	if w.Errors() != 0 || w.Dropped() != 0 {
		t.Errorf("Errors = %d, Dropped = %d, want 0, 0", w.Errors(), w.Dropped())
	}
	rotated, lines := readLines(t, dir, "log.jsonl")
	if len(rotated) != total {
		t.Fatalf("%d rotated files, want one per record (%d)", len(rotated), total)
	}
	got := counters(t, lines)
	if len(got) != total {
		t.Fatalf("%d records across the files, want %d", len(got), total)
	}
	for i, n := range got {
		if n != i {
			t.Fatalf("record %d is n=%d: the files in name order are not in write order", i, n)
		}
	}
	for _, name := range rotated {
		m := rotatedName.FindStringSubmatch(name)
		if m == nil {
			t.Fatalf("rotated file %q does not match log.<yyyymmddTHHMMSS.uuuuuuZ>.<sequence>.jsonl", name)
		}
		at, err := time.Parse("20060102T150405.000000Z", m[1])
		if err != nil {
			t.Fatalf("rotated file %q: %v", name, err)
		}
		if at.After(frozen) {
			t.Fatalf("rotated file %q is named for %v, later than the clock %v", name, at, frozen)
		}
	}
}

// TestWriterRotationNeverOverwritesAFile: if the name a rotation would take is already
// there, from an earlier run with the same clock reading, the rotation takes the next
// sequence number and the older file keeps its content.
func TestWriterRotationNeverOverwritesAFile(t *testing.T) {
	dir := t.TempDir()
	frozen := time.Date(2026, 10, 28, 1, 2, 3, 456789000, time.UTC)
	stamp := frozen.Format("20060102T150405.000000Z")
	taken := filepath.Join(dir, fmt.Sprintf("log.%s.%06d.jsonl", stamp, 1))
	if err := os.WriteFile(taken, []byte("earlier run\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	w, err := open(filepath.Join(dir, "log.jsonl"), 1, queueSize, func() time.Time { return frozen }, nil)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(rec(7))
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	if b, err := os.ReadFile(taken); err != nil || string(b) != "earlier run\n" {
		t.Errorf("the existing file now holds %q (%v), want it untouched", b, err)
	}
	if w.Errors() != 0 {
		t.Errorf("Errors = %d, want 0", w.Errors())
	}
	rotated, lines := readLines(t, dir, "log.jsonl")
	if len(rotated) != 2 || !strings.HasSuffix(rotated[1], ".000002.jsonl") {
		t.Errorf("rotated = %v, want the earlier file and the new one with the next sequence number", rotated)
	}
	if len(lines) != 2 || lines[0] != "earlier run" {
		t.Errorf("lines = %q, want the earlier run's line and the new record", lines)
	}
}
