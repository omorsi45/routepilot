// Package features implements feature spec v1: the hashed text features the
// router's quality predictor reads at request time. The offline trainer
// (python/routepilot_train/features.py) implements the same spec, and both are
// tested against the golden vectors in testdata/features, so a prediction
// made here equals the one the trainer evaluated.
//
// The spec, in order:
//
//  1. full is the message texts joined by "\n" and prefix is its first
//     PrefixBytes bytes. The cut is on bytes and may split a UTF-8 sequence.
//  2. Tokens are maximal runs of ASCII letters or digits (ASCII-lowercased)
//     and maximal runs of bytes >= 0x80 (kept raw). Every other byte separates.
//  3. Each token t contributes the key "u:"+t and each adjacent token pair
//     the key "b:"+t1+" "+t2. A key hashes with FNV-1a 64 to h; the bucket is
//     h % dim and the sign is -1 when the top bit of h is set, else +1.
//     Signed counts accumulate per bucket.
//  4. A bucket with net count c != 0 gets sign(c)*log1p(|c|). The values are
//     divided by their L2 norm, summed in ascending bucket order.
//  5. Twelve dense features, raw and unstandardized (see Vector.Dense).
package features

import (
	"bytes"
	"math"
	"slices"
	"strings"

	"github.com/omorsi45/routepilot/internal/openai"
)

const (
	// SpecVersion identifies this feature spec. A change to any step above
	// bumps it and regenerates the golden vectors.
	SpecVersion = 1
	// DefaultDim is the hashed feature dimension.
	DefaultDim = 65536
	// PrefixBytes is how many leading bytes of the joined text feed the
	// hashed features and the content-shape dense features.
	PrefixBytes = 32768
	// NumDense is the number of dense features.
	NumDense = 12
)

const (
	fnvOffset64 uint64 = 14695981039346656037
	fnvPrime64  uint64 = 1099511628211
)

// Message is one chat message reduced to what the features read.
type Message struct {
	Role string
	Text string
}

// Vector is the extracted feature vector. Idx and Val are the sparse hashed
// part: Idx is ascending and unique, Val is L2-normalized. Dense is raw and
// unstandardized:
//
//	0 log1p(bytes of full)         6 ASCII punctuation share of prefix
//	1 log1p(message count)         7 log1p(count of "```" in prefix)
//	2 log1p(lines of full)         8 1 if prefix has \frac \sum $$ \( or \[
//	3 digit share of prefix        9 1 if a prefix line looks like a choice
//	4 ASCII uppercase share        10 1 if any message role is "system"
//	5 non-ASCII byte share         11 log1p(count of "?" in prefix)
//
// A choice line matches ^[ \t]*\(?[A-Ea-e][).:] (option text follows).
type Vector struct {
	Idx   []int32
	Val   []float64
	Dense [NumDense]float64
}

// Messages reduces a chat request to its feature inputs: each message's role
// and plain text, in order.
func Messages(req *openai.ChatCompletionRequest) []Message {
	if req == nil {
		return nil
	}
	msgs := make([]Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = Message{Role: m.Role, Text: m.Content.PlainText()}
	}
	return msgs
}

// Extract computes the feature vector of msgs with a hashed dimension of dim.
// A dim outside [1, MaxInt32] falls back to DefaultDim instead of panicking.
// The text is read as raw bytes, so invalid UTF-8 is just high bytes.
func Extract(msgs []Message, dim int) Vector {
	if dim <= 0 || dim > math.MaxInt32 {
		dim = DefaultDim
	}
	texts := make([]string, len(msgs))
	for i, m := range msgs {
		texts[i] = m.Text
	}
	full := strings.Join(texts, "\n")
	prefix := []byte(full)
	if len(prefix) > PrefixBytes {
		prefix = prefix[:PrefixBytes]
	}

	var v Vector
	v.Idx, v.Val = hashedPart(prefix, dim)
	v.Dense = densePart(msgs, full, prefix)
	return v
}

// fnv1a64 is the 64-bit FNV-1a hash of b.
func fnv1a64(b []byte) uint64 {
	return fnvAdd(fnvOffset64, b)
}

// fnvAdd continues an FNV-1a hash with more bytes, so a key built from
// several pieces hashes without being concatenated first.
func fnvAdd(h uint64, b []byte) uint64 {
	for _, c := range b {
		h = (h ^ uint64(c)) * fnvPrime64
	}
	return h
}

func isASCIIAlnum(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// tokenize splits prefix into ASCII-lowercased alphanumeric runs and raw
// high-byte runs. The tokens alias one lowercased copy of prefix.
func tokenize(prefix []byte) [][]byte {
	lower := slices.Clone(prefix)
	for i, c := range lower {
		if c >= 'A' && c <= 'Z' {
			lower[i] = c + ('a' - 'A')
		}
	}
	var tokens [][]byte
	for i := 0; i < len(lower); {
		start := i
		switch c := lower[i]; {
		case isASCIIAlnum(c):
			for i < len(lower) && isASCIIAlnum(lower[i]) {
				i++
			}
		case c >= 0x80:
			for i < len(lower) && lower[i] >= 0x80 {
				i++
			}
		default:
			i++
			continue
		}
		tokens = append(tokens, lower[start:i:i])
	}
	return tokens
}

// bucketCounts returns the signed token counts per bucket, zero-net buckets
// included.
func bucketCounts(prefix []byte, dim int) map[int32]int {
	tokens := tokenize(prefix)
	counts := make(map[int32]int, 2*len(tokens))
	add := func(h uint64) {
		bucket := int32(h % uint64(dim))
		if h>>63 == 0 {
			counts[bucket]++
		} else {
			counts[bucket]--
		}
	}
	for i, t := range tokens {
		add(fnvAdd(fnvAdd(fnvOffset64, []byte("u:")), t))
		if i > 0 {
			h := fnvAdd(fnvOffset64, []byte("b:"))
			h = fnvAdd(h, tokens[i-1])
			h = fnvAdd(h, []byte(" "))
			add(fnvAdd(h, t))
		}
	}
	return counts
}

func hashedPart(prefix []byte, dim int) ([]int32, []float64) {
	counts := bucketCounts(prefix, dim)
	idx := make([]int32, 0, len(counts))
	for bucket, c := range counts {
		if c != 0 {
			idx = append(idx, bucket)
		}
	}
	slices.Sort(idx)
	val := make([]float64, len(idx))
	var squares float64
	for i, bucket := range idx {
		c := counts[bucket]
		v := math.Log1p(float64(c))
		if c < 0 {
			v = -math.Log1p(float64(-c))
		}
		val[i] = v
		// The explicit conversion forbids a fused multiply-add, which would
		// round differently from the Python side.
		squares += float64(v * v)
	}
	norm := math.Sqrt(squares)
	for i := range val {
		val[i] /= norm
	}
	return idx, val
}

// isASCIIPunct reports the 32 bytes of Python's string.punctuation.
func isASCIIPunct(c byte) bool {
	return c >= 0x21 && c <= 0x2f || c >= 0x3a && c <= 0x40 || c >= 0x5b && c <= 0x60 || c >= 0x7b && c <= 0x7e
}

// isChoiceLine reports whether line matches ^[ \t]*\(?[A-Ea-e][).:] as a
// byte-level check. The greedy parse is deterministic: backtracking cannot
// help because neither a space, a tab nor "(" is a choice letter.
func isChoiceLine(line []byte) bool {
	i := 0
	for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
		i++
	}
	if i < len(line) && line[i] == '(' {
		i++
	}
	if i+2 >= len(line) {
		return false
	}
	letter, delim, space := line[i], line[i+1], line[i+2]
	return (letter >= 'A' && letter <= 'E' || letter >= 'a' && letter <= 'e') &&
		(delim == ')' || delim == '.' || delim == ':') &&
		space == ' '
}

func hasChoiceLine(prefix []byte) bool {
	for line := range bytes.SplitSeq(prefix, []byte("\n")) {
		if isChoiceLine(line) {
			return true
		}
	}
	return false
}

func densePart(msgs []Message, full string, prefix []byte) [NumDense]float64 {
	var d [NumDense]float64
	d[0] = math.Log1p(float64(len(full)))
	d[1] = math.Log1p(float64(len(msgs)))
	if len(full) > 0 {
		d[2] = math.Log1p(float64(strings.Count(full, "\n") + 1))
	}

	if n := len(prefix); n > 0 {
		var digits, upper, high, punct int
		for _, c := range prefix {
			switch {
			case c >= '0' && c <= '9':
				digits++
			case c >= 'A' && c <= 'Z':
				upper++
			case c >= 0x80:
				high++
			case isASCIIPunct(c):
				punct++
			}
		}
		d[3] = float64(digits) / float64(n)
		d[4] = float64(upper) / float64(n)
		d[5] = float64(high) / float64(n)
		d[6] = float64(punct) / float64(n)
	}

	d[7] = math.Log1p(float64(bytes.Count(prefix, []byte("```"))))
	for _, marker := range []string{`\frac`, `\sum`, `$$`, `\(`, `\[`} {
		if bytes.Contains(prefix, []byte(marker)) {
			d[8] = 1
			break
		}
	}
	if hasChoiceLine(prefix) {
		d[9] = 1
	}
	for _, m := range msgs {
		if m.Role == "system" {
			d[10] = 1
			break
		}
	}
	d[11] = math.Log1p(float64(bytes.Count(prefix, []byte("?"))))
	return d
}
