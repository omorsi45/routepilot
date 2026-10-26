// Package money provides exact integer money arithmetic for cost accounting.
//
// Amounts are counted in picodollars (1e-12 USD) so that per-token prices,
// which are tiny fractions of a cent, stay exact with no floating point.
package money

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// PicoUSD is an amount of money in picodollars (1e-12 USD).
type PicoUSD int64

const (
	// picoPerUSD is the number of picodollars in one US dollar.
	picoPerUSD = 1_000_000_000_000
	// perMTokScale converts USD per million tokens into picodollars per
	// token: USD/Mtok * 1e12 picodollars/USD / 1e6 tokens/Mtok.
	perMTokScale     = 1_000_000
	maxFractionDigit = 6
	// maxPerToken caps a parsed per-token price at half of MaxInt64 so the sum
	// of an input and an output price can never overflow.
	maxPerToken = math.MaxInt64 / 2
)

// ParseUSDPerMTok parses a price in US dollars per million tokens, such as
// "1.25", and returns the exact price in picodollars per token.
//
// The accepted syntax is digits with an optional '.' and at most six
// fractional digits. Signs, exponents, spaces, empty input, and values too
// large to represent are errors.
func ParseUSDPerMTok(s string) (PicoUSD, error) {
	if s == "" {
		return 0, errors.New("empty price")
	}
	whole, frac, _ := strings.Cut(s, ".")
	if whole == "" && frac == "" {
		return 0, fmt.Errorf("invalid price %q", s)
	}
	if len(frac) > maxFractionDigit {
		return 0, fmt.Errorf("invalid price %q: at most %d fractional digits", s, maxFractionDigit)
	}

	var value int64
	for _, r := range whole {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("invalid price %q: want digits and an optional '.'", s)
		}
		d := int64(r - '0')
		if value > (maxPerToken/perMTokScale-d)/10 {
			return 0, fmt.Errorf("invalid price %q: too large", s)
		}
		value = value*10 + d
	}
	value *= perMTokScale

	var fracValue int64
	for _, r := range frac {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("invalid price %q: want digits and an optional '.'", s)
		}
		fracValue = fracValue*10 + int64(r-'0')
	}
	for i := len(frac); i < maxFractionDigit; i++ {
		fracValue *= 10
	}
	value += fracValue
	if value > maxPerToken {
		return 0, fmt.Errorf("invalid price %q: too large", s)
	}
	return PicoUSD(value), nil
}

// MulSat returns a*n, saturating at the maximum PicoUSD on overflow. A
// negative a or n is treated as zero, so the result is never negative.
func MulSat(a PicoUSD, n int64) PicoUSD {
	if a <= 0 || n <= 0 {
		return 0
	}
	if int64(a) > math.MaxInt64/n {
		return PicoUSD(math.MaxInt64)
	}
	return a * PicoUSD(n)
}

// AddSat returns a+b, saturating at the maximum PicoUSD on overflow. It is
// intended for non-negative amounts.
func AddSat(a, b PicoUSD) PicoUSD {
	if b > 0 && a > PicoUSD(math.MaxInt64)-b {
		return PicoUSD(math.MaxInt64)
	}
	return a + b
}

// USD renders the amount as an exact decimal number of US dollars with up to
// twelve fractional digits, trailing zeros trimmed (for example "0.00000125").
func (p PicoUSD) USD() string {
	sign := ""
	// Work in unsigned space so the minimum int64 does not overflow on negation.
	u := uint64(p)
	if p < 0 {
		sign = "-"
		u = -u
	}
	whole := u / picoPerUSD
	frac := fmt.Sprintf("%012d", u%picoPerUSD)
	frac = strings.TrimRight(frac, "0")
	if frac == "" {
		return fmt.Sprintf("%s%d", sign, whole)
	}
	return fmt.Sprintf("%s%d.%s", sign, whole, frac)
}
