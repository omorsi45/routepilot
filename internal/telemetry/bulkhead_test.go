package telemetry

import "testing"

// TestBulkheadRejectionCounter: rp_bulkhead_rejections_total counts, per deployment, the
// attempts a full deployment turned away. The series appears with the first rejection, so a
// server that never rejects exposes nothing extra.
func TestBulkheadRejectionCounter(t *testing.T) {
	m, reg := newTestMetrics()
	if got, _ := gather(t, reg); len(got["rp_bulkhead_rejections_total"]) != 0 {
		t.Fatalf("series before any rejection = %+v, want none", got["rp_bulkhead_rejections_total"])
	}

	m.BulkheadRejection("dep-a")
	m.BulkheadRejection("dep-a")
	m.BulkheadRejection("dep-b")

	got, types := gather(t, reg)
	if types["rp_bulkhead_rejections_total"] != "COUNTER" {
		t.Errorf("rp_bulkhead_rejections_total has type %q, want COUNTER", types["rp_bulkhead_rejections_total"])
	}
	if n := len(got["rp_bulkhead_rejections_total"]); n != 2 {
		t.Errorf("series = %+v, want one per deployment", got["rp_bulkhead_rejections_total"])
	}
	if s := find(t, got, "rp_bulkhead_rejections_total", map[string]string{"deployment": "dep-a"}); s.value != 2 {
		t.Errorf("dep-a rejections = %v, want 2", s.value)
	}
	if s := find(t, got, "rp_bulkhead_rejections_total", map[string]string{"deployment": "dep-b"}); s.value != 1 {
		t.Errorf("dep-b rejections = %v, want 1", s.value)
	}
}
