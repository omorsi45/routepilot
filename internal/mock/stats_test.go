package mock

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

// TestMockStatsEndpoint: GET /mock/stats serves the handler's own request counters as JSON, so
// an experiment can read what an upstream really received without trusting the gateway's view.
func TestMockStatsEndpoint(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})

	read := func() map[string]int64 {
		t.Helper()
		resp, err := http.Get(srv.URL + "/mock/stats")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("stats status = %d", resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("stats content-type = %q", ct)
		}
		b, _ := io.ReadAll(resp.Body)
		var got map[string]int64
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("stats body %q is not a JSON object of integers: %v", b, err)
		}
		return got
	}

	got := read()
	for _, key := range []string{"requests", "completed", "canceled", "events_written"} {
		if v, ok := got[key]; !ok || v != 0 {
			t.Fatalf("fresh stats[%q] = %v (present %v), want 0", key, v, ok)
		}
	}

	// Two answered requests and one injected failure: every one of them was received.
	mustPost(t, srv.URL, chatBody("m;tokens=2;ttft_ms=0;tpot_ms=0", ""), nil)
	mustPost(t, srv.URL, chatBody("m;tokens=2;ttft_ms=0;tpot_ms=0", ""), nil)
	mustPost(t, srv.URL, chatBody("m;fail=http500", ""), nil)
	waitFor(t, "two completed responses", func() bool { return h.Stats().Completed == 2 })

	got = read()
	if got["requests"] != 3 || got["completed"] != 2 || got["canceled"] != 0 {
		t.Fatalf("stats after three requests = %v, want requests 3, completed 2, canceled 0", got)
	}
	if want := h.Stats(); got["requests"] != want.Requests || got["events_written"] != want.EventsWritten {
		t.Fatalf("stats %v differ from Handler.Stats() %+v", got, want)
	}
}

// TestMockStatsEndpointMethod: only GET reads the counters, and reading them is not a request.
func TestMockStatsEndpointMethod(t *testing.T) {
	h, srv := newServer(t, Config{Seed: 1})

	resp, err := http.Post(srv.URL+"/mock/stats", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed || resp.Header.Get("Allow") != http.MethodGet {
		t.Fatalf("POST stats: status %d, Allow %q, want 405 and GET", resp.StatusCode, resp.Header.Get("Allow"))
	}
	if r, err := http.Get(srv.URL + "/mock/stats"); err == nil {
		_ = r.Body.Close()
	}
	if n := h.Stats().Requests; n != 0 {
		t.Fatalf("reading the stats counted %d chat requests, want 0", n)
	}
}
