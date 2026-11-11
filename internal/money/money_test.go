package money

import (
	"math"
	"testing"
)

func TestParseUSDPerMTok(t *testing.T) {
	good := []struct {
		in   string
		want PicoUSD
	}{
		{"1.25", 1250000},
		{"0", 0},
		{"0.000001", 1},
		{"10", 10000000},
		{"0.15", 150000},
		{"2.50", 2500000},
		{".5", 500000},
		{"1.", 1000000},
	}
	for _, tc := range good {
		got, err := ParseUSDPerMTok(tc.in)
		if err != nil || got != tc.want {
			t.Errorf("ParseUSDPerMTok(%q) = %d, %v; want %d", tc.in, got, err, tc.want)
		}
	}
	bad := []string{
		"1.2345678", "-1", "+1", "abc", "1e3", "", " 1", "1 ", "1.2.3", ".", "1,5",
		"9223372036854",              // overflow after scaling
		"99999999999999999999999999", // overflow while parsing
	}
	for _, in := range bad {
		if got, err := ParseUSDPerMTok(in); err == nil {
			t.Errorf("ParseUSDPerMTok(%q) = %d, want error", in, got)
		}
	}
}

func TestParseUSD(t *testing.T) {
	good := []struct {
		in   string
		want PicoUSD
	}{
		{"0", 0},
		{"1", 1_000_000_000_000},
		{"25.50", 25_500_000_000_000},
		{"0.000000000001", 1}, // twelve fractional digits: one picodollar
		{"0.5", 500_000_000_000},
		{".5", 500_000_000_000},
		{"1.", 1_000_000_000_000},
		{"007", 7_000_000_000_000},
		{"9000", 9_000_000_000_000_000},
		{"9223372.036854775807", PicoUSD(math.MaxInt64)}, // the largest amount that fits
	}
	for _, tc := range good {
		got, err := ParseUSD(tc.in)
		if err != nil || got != tc.want {
			t.Errorf("ParseUSD(%q) = %d, %v; want %d", tc.in, got, err, tc.want)
		}
	}
	bad := []string{
		"", " 1", "1 ", "-1", "-0", "+1", "abc", "1e3", "1,5", "1.2.3", ".", "$1",
		"0.0000000000001",             // thirteen fractional digits would not be exact
		"9223372.036854775808",        // one picodollar over the int64 range
		"9223373",                     // whole dollars over the range
		"99999999999999999999999999",  // overflow while parsing
		"1.0000000000000000000000001", // many fractional digits
		"\u0661",                      // a non-ASCII digit
	}
	for _, in := range bad {
		if got, err := ParseUSD(in); err == nil {
			t.Errorf("ParseUSD(%q) = %d, want error", in, got)
		}
	}
}

// TestParseUSDRoundTripsUSD pins that ParseUSD reads back exactly what USD prints.
func TestParseUSDRoundTripsUSD(t *testing.T) {
	for _, p := range []PicoUSD{0, 1, 999, 1_000_000_000_000, 1_234_567_890_123, PicoUSD(math.MaxInt64)} {
		got, err := ParseUSD(p.USD())
		if err != nil || got != p {
			t.Errorf("ParseUSD(%q) = %d, %v; want %d", p.USD(), got, err, p)
		}
	}
}

func TestMulSatAndAddSat(t *testing.T) {
	if got := MulSat(1250000, 1000); got != 1250000000 {
		t.Errorf("MulSat exact = %d", got)
	}
	if got := MulSat(0, 5); got != 0 {
		t.Errorf("MulSat zero = %d", got)
	}
	if got := MulSat(PicoUSD(math.MaxInt64), 2); got != PicoUSD(math.MaxInt64) {
		t.Errorf("MulSat overflow = %d", got)
	}
	if got := MulSat(PicoUSD(math.MaxInt64/2+1), 2); got != PicoUSD(math.MaxInt64) {
		t.Errorf("MulSat boundary = %d", got)
	}
	if got := MulSat(-5, 3); got != 0 {
		t.Errorf("MulSat negative price = %d, want 0", got)
	}
	if got := MulSat(5, -3); got != 0 {
		t.Errorf("MulSat negative count = %d, want 0", got)
	}
	if got := AddSat(1, 2); got != 3 {
		t.Errorf("AddSat = %d", got)
	}
	if got := AddSat(PicoUSD(math.MaxInt64), 1); got != PicoUSD(math.MaxInt64) {
		t.Errorf("AddSat overflow = %d", got)
	}
}

func TestUSDString(t *testing.T) {
	cases := []struct {
		in   PicoUSD
		want string
	}{
		{1250000, "0.00000125"},
		{0, "0"},
		{1e12, "1"},
		{1, "0.000000000001"},
		{1500000000000, "1.5"},
		{123456789012345, "123.456789012345"},
	}
	for _, tc := range cases {
		if got := tc.in.USD(); got != tc.want {
			t.Errorf("PicoUSD(%d).USD() = %q, want %q", int64(tc.in), got, tc.want)
		}
	}
}
