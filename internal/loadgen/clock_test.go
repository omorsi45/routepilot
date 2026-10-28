package loadgen

import (
	"testing"
	"time"
)

func TestClockStepIsPositiveAndTheFractionIsAShare(t *testing.T) {
	step, zero := ClockStep(20 * time.Millisecond)
	if step <= 0 {
		t.Errorf("step = %v, want a positive duration", step)
	}
	if zero < 0 || zero > 1 {
		t.Errorf("zero fraction = %v, want a value in [0, 1]", zero)
	}
}

func TestClockStepNeverExceedsTheTimeItWatched(t *testing.T) {
	// The smallest non-zero step between two reads cannot be longer than the whole probe, plus
	// whatever the scheduler took away from the last read. A generous bound still catches a
	// step that is computed from the wrong quantity.
	d := 20 * time.Millisecond
	step, _ := ClockStep(d)
	if step > 10*d {
		t.Errorf("step = %v after watching %v", step, d)
	}
}
