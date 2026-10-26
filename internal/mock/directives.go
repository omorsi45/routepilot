package mock

import (
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/omorsi45/routepilot/internal/openai"
)

// Failure modes accepted by the fail directive and used as FailRates keys.
const (
	failHTTP500   = "http500"
	failHTTP429   = "http429"
	failHTTP400   = "http400"
	failReset     = "reset"
	failStall     = "stall"
	failMidstream = "midstream"
	failHTML502   = "html502"
)

const (
	defaultFailAfter  = 1
	defaultRetryAfter = 1
	maxTokens         = 100000
	headerPrefix      = "x-mock-"
)

func validFail(v string) bool {
	switch v {
	case failHTTP500, failHTTP429, failHTTP400, failReset, failStall, failMidstream, failHTML502:
		return true
	}
	return false
}

// directives are the per-request behavior overrides. Pointer fields are nil
// when the request did not set them, so the handler falls back to its config.
type directives struct {
	fail       string
	tokens     *int
	ttftMs     *float64
	tpotMs     *float64
	failAfter  int
	retryAfter int
}

// parseDirectives reads x-mock-<key> headers first, then the key=value
// segments of the model string, so the model wins. It returns the model name
// (the first segment). Unknown keys and unparseable values are a 400 so test
// typos surface instead of silently doing nothing.
func parseDirectives(model string, hdr http.Header) (string, directives, *openai.APIError) {
	d := directives{failAfter: defaultFailAfter, retryAfter: defaultRetryAfter}

	for name, values := range hdr {
		lower := strings.ToLower(name)
		if !strings.HasPrefix(lower, headerPrefix) || len(values) == 0 {
			continue
		}
		if err := d.set(strings.TrimPrefix(lower, headerPrefix), values[0]); err != nil {
			return "", d, err
		}
	}

	segments := strings.Split(model, ";")
	for _, seg := range segments[1:] {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		key, value, ok := strings.Cut(seg, "=")
		if !ok {
			return "", d, openai.ErrInvalidRequest(fmt.Sprintf("mock directive %q must be key=value", seg), "model")
		}
		if err := d.set(strings.TrimSpace(key), strings.TrimSpace(value)); err != nil {
			return "", d, err
		}
	}
	return strings.TrimSpace(segments[0]), d, nil
}

func (d *directives) set(key, value string) *openai.APIError {
	bad := func(want string) *openai.APIError {
		return openai.ErrInvalidRequest(fmt.Sprintf("mock directive %q: %q is not %s", key, value, want), "model")
	}
	switch key {
	case "fail":
		if !validFail(value) {
			return bad("a known failure mode")
		}
		d.fail = value
	case "tokens":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 || n > maxTokens {
			return bad(fmt.Sprintf("an integer in [0,%d]", maxTokens))
		}
		d.tokens = &n
	case "ttft_ms", "tpot_ms":
		f, err := strconv.ParseFloat(value, 64)
		if err != nil || f < 0 || math.IsNaN(f) || f > 3.6e6 {
			return bad("a number of milliseconds in [0,3600000]")
		}
		if key == "ttft_ms" {
			d.ttftMs = &f
		} else {
			d.tpotMs = &f
		}
	case "fail_after":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			return bad("a non-negative integer")
		}
		d.failAfter = n
	case "retry_after":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			return bad("a non-negative integer")
		}
		d.retryAfter = n
	default:
		return openai.ErrInvalidRequest(fmt.Sprintf("unknown mock directive %q", key), "model")
	}
	return nil
}
