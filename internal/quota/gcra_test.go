package quota

import (
	"math"
	"math/big"
	"testing"
)

// exactMicros is the oracle for gcraCost and gcraLimit: n*60_000_000/perMinute
// in arbitrary precision, rounded up or down, capped at math.MaxInt64.
func exactMicros(n, perMinute int64, roundUp bool) int64 {
	num := new(big.Int).Mul(big.NewInt(n), big.NewInt(microsPerMinute))
	q, r := new(big.Int).QuoRem(num, big.NewInt(perMinute), new(big.Int))
	if roundUp && r.Sign() != 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsInt64() {
		return math.MaxInt64
	}
	return q.Int64()
}

func TestGCRACostAndLimitMatchExactArithmetic(t *testing.T) {
	cases := []struct{ n, perMinute int64 }{
		{1, 60}, {1, 7}, {3, 7}, {1, 40_000_000}, {20_000_000, 40_000_000},
		{1, 600_000_000}, {123_456_789, 987_654_321}, {1 << 40, 7},
		{1 << 62, 1}, {1 << 62, 60_000_000}, {1 << 62, 59_999_999},
		{math.MaxInt64, 1}, {math.MaxInt64, 2}, {math.MaxInt64, 59_999_999},
		{math.MaxInt64, 60_000_000}, {math.MaxInt64, 60_000_001},
		{math.MaxInt64, math.MaxInt64},
	}
	for _, c := range cases {
		if got, want := gcraCost(c.n, c.perMinute), exactMicros(c.n, c.perMinute, true); got != want {
			t.Errorf("gcraCost(%d, %d) = %d, want %d", c.n, c.perMinute, got, want)
		}
		if got, want := gcraLimit(c.n, c.perMinute), exactMicros(c.n, c.perMinute, false); got != want {
			t.Errorf("gcraLimit(%d, %d) = %d, want %d", c.n, c.perMinute, got, want)
		}
	}
}

func TestGCRACostKnownValues(t *testing.T) {
	cases := []struct {
		name         string
		n, perMinute int64
		want         int64
	}{
		{"exact interval", 1, 60, 1_000_000},
		{"half microsecond rounds up", 1, 40_000_000, 2},
		{"exact at high rate", 20_000_000, 40_000_000, 30_000_000},
		{"inexact interval rounds up", 3, 7, 25_714_286},
		{"above one per microsecond costs 1", 1, 600_000_000, 1},
		{"zero units", 0, 7, 0},
		{"negative units", -5, 7, 0},
		{"unlimited rate", 5, 0, 0},
		{"128-bit quotient overflow saturates", math.MaxInt64, 1, math.MaxInt64},
		{"quotient above int64 saturates", math.MaxInt64, 59_999_999, math.MaxInt64},
		{"quotient exactly int64 max", math.MaxInt64, 60_000_000, math.MaxInt64},
	}
	for _, c := range cases {
		if got := gcraCost(c.n, c.perMinute); got != c.want {
			t.Errorf("%s: gcraCost(%d, %d) = %d, want %d", c.name, c.n, c.perMinute, got, c.want)
		}
	}
}

func TestGCRALimitKnownValues(t *testing.T) {
	cases := []struct {
		name             string
		burst, perMinute int64
		want             int64
	}{
		{"exact", 40_000_000, 40_000_000, 60_000_000},
		{"inexact rounds down", 3, 7, 25_714_285},
		{"burst below one counts as one", 0, 7, 8_571_428},
		{"unlimited rate", 10, 0, 0},
		{"saturates", math.MaxInt64, 1, math.MaxInt64},
	}
	for _, c := range cases {
		if got := gcraLimit(c.burst, c.perMinute); got != c.want {
			t.Errorf("%s: gcraLimit(%d, %d) = %d, want %d", c.name, c.burst, c.perMinute, got, c.want)
		}
	}
}

func TestWindowMicros(t *testing.T) {
	const day = 86_400_000_000
	cases := []struct {
		burst, perMinute, want int64
	}{
		{1440, 1, day},
		{1441, 1, day + 60_000_000},
		// 1.5us per unit: 57_600_000_000 units are exactly a day, and the
		// next one is past it (the old floored interval of 1us allowed ~86e9).
		{57_600_000_000, 40_000_000, day},
		{57_600_000_001, 40_000_000, day + 1},
	}
	for _, c := range cases {
		if got := WindowMicros(c.burst, c.perMinute); got != c.want {
			t.Errorf("WindowMicros(%d, %d) = %d, want %d", c.burst, c.perMinute, got, c.want)
		}
	}
}

// A reservation of at most Burst must fit an idle key at any rate, including
// ones where Burst*60_000_000/PerMinute is not a whole number.
func TestGCRAParamsLimitCoversAFullBurst(t *testing.T) {
	for _, perMinute := range []int64{1, 3, 7, 59, 999, 40_000_000, 600_000_000, 1_000_000_007} {
		for _, burst := range []int64{0, 1, 2, 3, 7, 100, 12_345_679} {
			cost, limit, unlimited := gcraParams(RateLimit{PerMinute: perMinute, Burst: burst}, max(burst, 1))
			if unlimited {
				t.Fatalf("perMinute %d reported unlimited", perMinute)
			}
			if cost > limit {
				t.Errorf("perMinute %d burst %d: full-burst cost %d exceeds limit %d", perMinute, burst, cost, limit)
			}
		}
	}
	if _, _, unlimited := gcraParams(RateLimit{PerMinute: 0, Burst: 5}, 1); !unlimited {
		t.Error("PerMinute 0 must be unlimited")
	}
}
