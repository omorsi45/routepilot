package replay

import (
	"encoding/binary"
	"hash/fnv"
	"io"

	"github.com/omorsi45/routepilot/internal/latency"
)

// stream is a deterministic source of uniform draws. It is a SplitMix64 generator seeded
// with the FNV-1a 64 hash of a seed (8 bytes, big endian) followed by the given parts, so
// a seed and a few IDs fix the whole sequence on every Go version.
type stream struct{ state uint64 }

func newStream(seed uint64, parts ...string) *stream {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], seed)
	h := fnv.New64a()
	// A hash.Hash64 write never fails.
	_, _ = h.Write(b[:])
	for _, p := range parts {
		_, _ = io.WriteString(h, p)
	}
	return &stream{state: h.Sum64()}
}

func (s *stream) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// float64 returns a draw strictly inside (0, 1).
func (s *stream) float64() float64 {
	return (float64(s.next()>>11) + 0.5) / (1 << 53)
}

// simulateLatency draws the simulated latency of one item on one model: a time to first
// token and a time per token from the model's profile, seeded by the latency seed, the
// item and the model. The end-to-end time is the first token plus the recorded completion
// tokens (none when the record is negative) times the time per token. Both results are
// simulated figures in milliseconds.
func simulateLatency(p latency.Profile, seed uint64, itemID, model string, completionTokens int) (ttftMs, e2eMs float64) {
	s := newStream(seed, itemID, model)
	ttft, tpot := p.Draw(s.float64(), s.float64())
	return ttft, ttft + float64(max(completionTokens, 0))*tpot
}
