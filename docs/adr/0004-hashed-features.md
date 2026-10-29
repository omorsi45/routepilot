# ADR 0004: Hashed text features, identical in Go and Python

Status: Accepted, 2026-10-26.

## Context

The predictor needs a numeric view of the request text. It has to be computed per request in microseconds, identically in the Go gateway and in the Python trainer, over arbitrary multilingual text with no fixed vocabulary. An embedding model would add a network hop or a large dependency to the request path, and the Go side cannot run one at the latency target (see [ADR 0001](0001-go-data-plane-python-training.md) and [ADR 0007](0007-predictor-choice.md)).

## Decision

Feature spec v1, implemented in `python/routepilot_train/features.py` and `internal/features`:

- Tokens: maximal runs of ASCII letters and digits, lowercased. A maximal run of non-ASCII bytes that are not ASCII whitespace is one token, kept as raw bytes. Every other byte separates tokens.
- Hashed part: unigrams and adjacent bigrams are hashed with FNV-1a 64 into 65,536 buckets. The top bit of the hash gives the sign, the count per bucket is transformed to `sign * log1p(|count|)`, and the vector is L2-normalized.
- Dense part: 12 values (log byte count, message count, line count, digit, uppercase, non-ASCII and punctuation fractions, code fence count, a math marker, a multiple-choice marker, a system message flag, question mark count), standardized with a mean and standard deviation stored in the artifact.
- Only the first 32 KiB of bytes, cut on a rune boundary, feed the hashed part and the fractions. Length features use the full text.
- The spec version is stored in the artifact and Go refuses another version. Golden vectors in `testdata/features/` (36 invented cases, among them empty text, a long text cut in the middle of a rune, non-ASCII runs and every dense marker) are checked by both languages.

## Consequences

- There is no vocabulary file to ship or to keep in sync, unseen tokens still land in a bucket, and the extractor is a single pass over the bytes. The price is collisions, and a representation that sees words and word pairs but not meaning.
- Both parts of the feature vector carry signal, and the width matters a little. In the ablation, each variant was retrained and its dual operating point retuned to the same calibration target, so the test success changes little by construction and the cost carries the comparison. Against the shipped configuration (both parts, 65,536 buckets), the run's own statements are: hashed part only, mean cost 33.3% dearer (worse); dense part only, 123.3% dearer (worse); 16,384 buckets, 14.8% dearer (worse); 262,144 buckets, 8.1% dearer with an interval that includes zero (no clear difference). The wider vector multiplies the artifact size by four (25,281,768 bytes of weights against 6,407,616), so the shipped width was kept ([ablations](../results.md#ablations)).
- A representation without meaning limits what the predictor can learn. The failure analysis shows the learned policy losing most against Best Single on arc-agi (-20.5 points), hle (-12.5), gpqa (-12.1) and livecodebench (-10.7) ([failure analysis](../results.md#failure-analysis)). The data do not say whether a richer representation would close that gap.
- Changing any rule of the spec (tokenization, hash, transform, dense list, prefix length) is a new spec version: the golden vectors are regenerated in both languages and every artifact is retrained.
- Byte-level handling means invalid UTF-8 and non-ASCII scripts are handled consistently, and the prefix cut cannot split a rune. The 32 KiB prefix means that features of a very long prompt describe its beginning.
