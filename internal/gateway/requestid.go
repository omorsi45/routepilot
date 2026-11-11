package gateway

import (
	"crypto/rand"
	"encoding/hex"
)

// requestIDHeader carries the per-request ID on every response.
const requestIDHeader = "x-routepilot-request-id"

// newRequestID returns 24 lowercase hex characters from crypto/rand.
func newRequestID() string {
	var b [12]byte
	// crypto/rand.Read does not return an error; it aborts the process if the
	// system source of randomness is unavailable.
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
