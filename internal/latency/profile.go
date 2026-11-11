// Package latency models the time a model takes to answer: sourced prior profiles per
// model, an online per-deployment estimator, and the union-bound quantile the router
// checks a latency SLO against.
//
// Every number that comes from a profile file is a simulated figure: the profiles are
// medians read from public benchmark pages, with assumed spreads, and are not
// measurements of this gateway. Reports and tables that show them must say "simulated".
package latency

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Profile is the prior latency of one model: log-normal time to first token (TTFT) and
// log-normal time per output token (TPOT), both in milliseconds. Source and Retrieved say
// where the medians come from; the sigmas are assumptions unless the Note says otherwise.
type Profile struct {
	Model        string  `yaml:"model"`
	TTFTMedianMs float64 `yaml:"ttft_median_ms"`
	TTFTSigma    float64 `yaml:"ttft_sigma"`
	TPOTMedianMs float64 `yaml:"tpot_median_ms"`
	TPOTSigma    float64 `yaml:"tpot_sigma"`
	Source       string  `yaml:"source"`
	Retrieved    string  `yaml:"retrieved"`
	Note         string  `yaml:"note"`
}

// Draw returns a TTFT and a TPOT, in milliseconds, for the uniform draws u1 and u2 in
// (0, 1): each is the log-normal quantile of the profile at its draw.
func (p Profile) Draw(u1, u2 float64) (ttftMs, tpotMs float64) {
	return LogNormalQuantile(p.TTFTMedianMs, p.TTFTSigma, u1), LogNormalQuantile(p.TPOTMedianMs, p.TPOTSigma, u2)
}

// profileFile is the YAML document. Fields are pointers so an absent key is an error and
// not a silent zero.
type profileFile struct {
	Profiles []rawProfile `yaml:"profiles"`
}

type rawProfile struct {
	Model        *string  `yaml:"model"`
	TTFTMedianMs *float64 `yaml:"ttft_median_ms"`
	TTFTSigma    *float64 `yaml:"ttft_sigma"`
	TPOTMedianMs *float64 `yaml:"tpot_median_ms"`
	TPOTSigma    *float64 `yaml:"tpot_sigma"`
	Source       *string  `yaml:"source"`
	Retrieved    *string  `yaml:"retrieved"`
	Note         string   `yaml:"note"`
}

// LoadProfiles reads a profile file: a top-level "profiles" list whose entries need every
// field except note. Medians must be positive and finite, sigmas non-negative and finite,
// retrieved a YYYY-MM-DD date, and a model may appear once. Unknown keys are errors.
func LoadProfiles(path string) (map[string]Profile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load latency profiles: %w", err)
	}
	profiles, err := parseProfiles(b)
	if err != nil {
		return nil, fmt.Errorf("load latency profiles %s: %w", path, err)
	}
	return profiles, nil
}

func parseProfiles(b []byte) (map[string]Profile, error) {
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var file profileFile
	if err := dec.Decode(&file); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("file is empty")
		}
		return nil, err
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, err
		}
		return nil, errors.New("multiple YAML documents are not supported")
	}
	if len(file.Profiles) == 0 {
		return nil, errors.New("no profiles")
	}

	out := make(map[string]Profile, len(file.Profiles))
	for i, raw := range file.Profiles {
		p, err := raw.validate()
		if err != nil {
			return nil, fmt.Errorf("profiles[%d]: %w", i, err)
		}
		if _, dup := out[p.Model]; dup {
			return nil, fmt.Errorf("profiles[%d]: duplicate model %q", i, p.Model)
		}
		out[p.Model] = p
	}
	return out, nil
}

func (r rawProfile) validate() (Profile, error) {
	if r.Model == nil || *r.Model == "" {
		return Profile{}, errors.New("model is required")
	}
	p := Profile{Model: *r.Model, Note: r.Note}
	fail := func(format string, args ...any) (Profile, error) {
		return Profile{}, fmt.Errorf("%s: %s", p.Model, fmt.Sprintf(format, args...))
	}

	for _, f := range []struct {
		name     string
		value    *float64
		dst      *float64
		positive bool
	}{
		{"ttft_median_ms", r.TTFTMedianMs, &p.TTFTMedianMs, true},
		{"ttft_sigma", r.TTFTSigma, &p.TTFTSigma, false},
		{"tpot_median_ms", r.TPOTMedianMs, &p.TPOTMedianMs, true},
		{"tpot_sigma", r.TPOTSigma, &p.TPOTSigma, false},
	} {
		switch {
		case f.value == nil:
			return fail("%s is required", f.name)
		case math.IsNaN(*f.value) || math.IsInf(*f.value, 0):
			return fail("%s must be finite, got %v", f.name, *f.value)
		case f.positive && *f.value <= 0:
			return fail("%s must be positive, got %v", f.name, *f.value)
		case !f.positive && *f.value < 0:
			return fail("%s must not be negative, got %v", f.name, *f.value)
		}
		*f.dst = *f.value
	}

	if r.Source == nil || *r.Source == "" {
		return fail("source is required")
	}
	p.Source = *r.Source
	if r.Retrieved == nil {
		return fail("retrieved is required")
	}
	if _, err := time.Parse(time.DateOnly, *r.Retrieved); err != nil {
		return fail("retrieved must be a YYYY-MM-DD date, got %q", *r.Retrieved)
	}
	p.Retrieved = *r.Retrieved
	return p, nil
}

// LogNormalQuantile is the u-quantile of a log-normal distribution with the given median
// and sigma (the standard deviation of the logarithm): median * exp(sigma * Phi^-1(u)).
// It returns 0 for u <= 0, +Inf for u >= 1 and NaN for a NaN u.
func LogNormalQuantile(median, sigma, u float64) float64 {
	switch {
	case math.IsNaN(u):
		return math.NaN()
	case u <= 0:
		return 0
	case u >= 1:
		return math.Inf(1)
	}
	return median * math.Exp(sigma*normalQuantile(u))
}

// UnionBound bounds the end-to-end time: first token plus output tokens times time per
// token. With each of the three inputs at its 1-delta/3 quantile, the sum is at or above
// the 1-delta quantile of the real end-to-end time by the union bound, whatever the
// dependence between the three.
func UnionBound(ttftQ, outTokensQ, tpotQ float64) float64 {
	return ttftQ + outTokensQ*tpotQ
}

// Coefficients of Acklam's rational approximation of the inverse normal CDF.
var (
	acklamA = [6]float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	acklamB = [5]float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	acklamC = [6]float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	acklamD = [4]float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
)

// acklamLow is where Acklam's central region ends and the tail polynomials start.
const acklamLow = 0.02425

// normalQuantile is Phi^-1(p) for p in (0, 1): Acklam's approximation (relative error
// about 1e-9) followed by one Halley step, which brings it to full double precision. The
// upper half uses the symmetry Phi^-1(p) = -Phi^-1(1-p), where 1-p is exact, because the
// refinement subtracts p from a CDF value and that loses precision near 1.
func normalQuantile(p float64) float64 {
	if p > 0.5 {
		return -normalQuantile(1 - p)
	}
	var x float64
	if p < acklamLow {
		q := math.Sqrt(-2 * math.Log(p))
		x = (((((acklamC[0]*q+acklamC[1])*q+acklamC[2])*q+acklamC[3])*q+acklamC[4])*q + acklamC[5]) /
			((((acklamD[0]*q+acklamD[1])*q+acklamD[2])*q+acklamD[3])*q + 1)
	} else {
		q := p - 0.5
		r := q * q
		x = (((((acklamA[0]*r+acklamA[1])*r+acklamA[2])*r+acklamA[3])*r+acklamA[4])*r + acklamA[5]) * q /
			(((((acklamB[0]*r+acklamB[1])*r+acklamB[2])*r+acklamB[3])*r+acklamB[4])*r + 1)
	}
	// Halley step on f(x) = Phi(x) - p.
	e := 0.5*math.Erfc(-x/math.Sqrt2) - p
	u := e * math.Sqrt(2*math.Pi) * math.Exp(x*x/2)
	return x - u/(1+x*u/2)
}
