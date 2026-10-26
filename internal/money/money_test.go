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
