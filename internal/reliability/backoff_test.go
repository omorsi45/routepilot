package reliability

import (
	"math"
	"testing"
	"time"
)

func TestBackoffBounds(t *testing.T) {
	one := func() float64 { return 1 }
	zero := func() float64 { return 0 }
	want := []time.Duration{
		50 * time.Millisecond,
		100 * time.Millisecond,
		200 * time.Millisecond,
		400 * time.Millisecond,
		800 * time.Millisecond,
		time.Second,
		time.Second,
	}
	for attempt, w := range want {
		if got := Backoff(attempt, one); got != w {
			t.Errorf("Backoff(%d, 1) = %v, want %v", attempt, got, w)
		}
		if got := Backoff(attempt, zero); got != 0 {
			t.Errorf("Backoff(%d, 0) = %v, want 0", attempt, got)
		}
	}
}

func TestBackoffJitterScales(t *testing.T) {
	half := func() float64 { return 0.5 }
	if got := Backoff(2, half); got != 100*time.Millisecond {
		t.Fatalf("Backoff(2, 0.5) = %v, want 100ms", got)
	}
}

func TestBackoffExtremeAttemptsDoNotOverflow(t *testing.T) {
	one := func() float64 { return 1 }
	for _, attempt := range []int{-5, -1, 62, 63, 64, 100, math.MaxInt32, math.MaxInt} {
		got := Backoff(attempt, one)
		if got < 0 || got > time.Second {
			t.Errorf("Backoff(%d, 1) = %v, want within [0, 1s]", attempt, got)
		}
	}
	if got := Backoff(-3, one); got != 50*time.Millisecond {
		t.Errorf("negative attempt = %v, want 50ms", got)
	}
}

func TestBackoffClampsRandom(t *testing.T) {
	cases := map[string]func() float64{
		"above one": func() float64 { return 7 },
		"negative":  func() float64 { return -3 },
		"nan":       func() float64 { return math.NaN() },
	}
	for name, rnd := range cases {
		got := Backoff(3, rnd)
		if got < 0 || got > 400*time.Millisecond {
			t.Errorf("%s: Backoff(3) = %v, want within [0, 400ms]", name, got)
		}
	}
	if got := Backoff(3, cases["above one"]); got != 400*time.Millisecond {
		t.Errorf("rnd above 1 = %v, want clamped to 400ms", got)
	}
	if got := Backoff(3, cases["negative"]); got != 0 {
		t.Errorf("rnd below 0 = %v, want 0", got)
	}
}
