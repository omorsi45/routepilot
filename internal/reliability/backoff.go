package reliability

import "time"

const (
	backoffBase = 50 * time.Millisecond
	backoffCap  = time.Second
	// backoffMaxShift is the first shift whose base (50ms << 5 = 1.6s) already
	// exceeds backoffCap, so larger attempts cannot change the result.
	backoffMaxShift = 5
)

// Backoff returns the jittered wait before retry number attempt+1:
// rnd() * min(1s, 50ms * 2^attempt). Attempt 0 is the wait before the second
// attempt. rnd is expected to return a value in [0,1) and is clamped to [0,1].
func Backoff(attempt int, rnd func() float64) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	if attempt > backoffMaxShift {
		attempt = backoffMaxShift
	}
	base := min(backoffBase<<uint(attempt), backoffCap)

	r := rnd()
	if !(r > 0) { // also catches NaN
		r = 0
	}
	if r > 1 {
		r = 1
	}
	return time.Duration(r * float64(base))
}
