package loadgen

import "time"

// ClockStep reads time.Now in a tight loop for d and reports how finely the clock advances.
//
// step is the smallest non-zero difference between two consecutive reads, measured on the
// monotonic reading that latency measurements use. On a clock that ticks every few
// microseconds or finer it is about the cost of one read (tens of nanoseconds on Linux); on a
// clock that only advances in coarse ticks it is the tick. zeroFraction is the share of
// consecutive reads that returned the same instant: near 0 for a fine clock, near 1 for a
// coarse one. Anything timed below step is quantized to a multiple of it.
//
// When no read differs from the one before it for the whole of d, the clock step is at least
// d, and step is reported as d.
func ClockStep(d time.Duration) (step time.Duration, zeroFraction float64) {
	start := time.Now()
	prev := start
	var reads, zeros int64
	for {
		now := time.Now()
		delta := now.Sub(prev)
		reads++
		if delta == 0 {
			zeros++
		} else if step == 0 || delta < step {
			step = delta
		}
		prev = now
		if now.Sub(start) >= d {
			break
		}
	}
	if step == 0 {
		step = d
	}
	return step, float64(zeros) / float64(reads)
}
