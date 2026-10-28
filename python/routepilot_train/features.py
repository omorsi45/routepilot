"""Feature spec v1: the hashed text features shared by the trainer and the router.

The Go implementation in internal/features must produce the same indices and
(within 1e-12) the same values for the same text. testdata/features holds the
hand-written cases and the golden vectors generated here; both sides test
against them.

Algorithm, in order:

1. ``full`` is the UTF-8 bytes of the message texts joined by ``"\\n"`` and
   ``prefix = full[:PREFIX_BYTES]``. The cut is on bytes and may split a
   UTF-8 sequence. Every feature below except the length, message-count and
   line-count features reads the prefix only.
2. Tokens are maximal runs of ASCII letters or digits (ASCII-lowercased) and
   maximal runs of bytes >= 0x80 (kept raw). Every other byte separates.
3. Each token ``t`` contributes the key ``b"u:" + t`` and each adjacent token
   pair the key ``b"b:" + t1 + b" " + t2``. A key hashes with FNV-1a 64 to
   ``h``; the bucket is ``h % dim`` and the sign is -1 when the top bit of
   ``h`` is set, else +1. Signed counts accumulate per bucket.
4. A bucket with net count ``c != 0`` gets ``sign(c) * log1p(|c|)``. The
   values are divided by the L2 norm, summed in ascending bucket order.
5. Twelve dense features, raw and unstandardized (see ``extract``).

Text must be valid Unicode (no lone surrogates); the Go side takes raw bytes,
so parity is defined for valid UTF-8 input.
"""

import argparse
import json
import math
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse

SPEC_VERSION = 1
DEFAULT_DIM = 65536
PREFIX_BYTES = 32768
NUM_DENSE = 12
MAX_DIM = 2**31 - 1

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK64 = 0xFFFFFFFFFFFFFFFF

# After ASCII lowercasing, a token is a run of [a-z0-9] or a run of high bytes.
_TOKEN = re.compile(rb"[a-z0-9]+|[\x80-\xff]+")
_CHOICE_LINE = re.compile(rb"[ \t]*\(?[A-Ea-e][).:] ")
_DIGITS = b"0123456789"
_UPPERCASE = string.ascii_uppercase.encode("ascii")
_HIGH_BYTES = bytes(range(0x80, 0x100))
_PUNCTUATION = string.punctuation.encode("ascii")
_FENCE = b"```"
_MATH_MARKERS = (b"\\frac", b"\\sum", b"$$", b"\\(", b"\\[")


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash of ``data``."""
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return h


def tokenize(prefix: bytes) -> list[bytes]:
    """Spec step 2: ASCII-lowercased alphanumeric runs and raw high-byte runs.

    ``bytes.lower`` only touches ASCII, so high bytes pass through unchanged.
    """
    return _TOKEN.findall(prefix.lower())


def bucket_counts(prefix: bytes, dim: int) -> dict[int, int]:
    """Spec step 3: signed token counts per bucket, zero-net buckets included."""
    tokens = tokenize(prefix)
    counts: dict[int, int] = {}

    def add(key: bytes) -> None:
        h = fnv1a64(key)
        bucket = h % dim
        counts[bucket] = counts.get(bucket, 0) + (1 if h >> 63 == 0 else -1)

    for token in tokens:
        add(b"u:" + token)
    for first, second in zip(tokens, tokens[1:], strict=False):
        add(b"b:" + first + b" " + second)
    return counts


def _check_dim(dim: int) -> None:
    if not 1 <= dim <= MAX_DIM:
        raise ValueError(f"dim must be in [1, {MAX_DIM}], got {dim}")


def _hashed_part(prefix: bytes, dim: int) -> tuple[np.ndarray, np.ndarray]:
    counts = bucket_counts(prefix, dim)
    idx = sorted(bucket for bucket, count in counts.items() if count != 0)
    values = [
        math.log1p(counts[bucket]) if counts[bucket] > 0 else -math.log1p(-counts[bucket])
        for bucket in idx
    ]
    # Plain left-to-right summation in ascending bucket order, as the Go side does.
    squares = 0.0
    for value in values:
        squares += value * value
    norm = math.sqrt(squares)
    return (
        np.array(idx, dtype=np.int32),
        np.array([value / norm for value in values], dtype=np.float64),
    )


def _fraction(prefix: bytes, selected: bytes) -> float:
    """Share of prefix bytes that are in ``selected``; 0 for an empty prefix."""
    if not prefix:
        return 0.0
    return (len(prefix) - len(prefix.translate(None, selected))) / len(prefix)


def _has_choice_line(prefix: bytes) -> bool:
    return any(_CHOICE_LINE.match(line) for line in prefix.split(b"\n"))


def _dense_part(roles: Sequence[str], full: bytes, prefix: bytes) -> np.ndarray:
    lines = full.count(b"\n") + 1 if full else 0
    return np.array(
        [
            math.log1p(len(full)),
            math.log1p(len(roles)),
            math.log1p(lines),
            _fraction(prefix, _DIGITS),
            _fraction(prefix, _UPPERCASE),
            _fraction(prefix, _HIGH_BYTES),
            _fraction(prefix, _PUNCTUATION),
            math.log1p(prefix.count(_FENCE)),
            1.0 if any(marker in prefix for marker in _MATH_MARKERS) else 0.0,
            1.0 if _has_choice_line(prefix) else 0.0,
            1.0 if "system" in roles else 0.0,
            math.log1p(prefix.count(b"?")),
        ],
        dtype=np.float64,
    )


def extract(
    messages: Sequence[tuple[str, str]], dim: int = DEFAULT_DIM
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from ``(role, text)`` messages.

    Returns ``(idx, val, dense)``: int32 ascending unique bucket indices, the
    L2-normalized float64 hashed values, and 12 raw float64 dense features:

    0. log1p(bytes of full), 1. log1p(message count), 2. log1p(lines of full),
    3. digit share, 4. ASCII uppercase share, 5. non-ASCII byte share,
    6. ASCII punctuation share (all over the prefix), 7. log1p(count of
    triple backticks), 8. any of ``\\frac \\sum $$ \\( \\[``, 9. any line
    starting like a multiple-choice option (``A. ``, ``(b) ``, ``C: ``),
    10. any ``system`` role, 11. log1p(count of ``?``).
    """
    _check_dim(dim)
    roles = [role for role, _ in messages]
    full = "\n".join(text for _, text in messages).encode("utf-8")
    prefix = full[:PREFIX_BYTES]
    idx, val = _hashed_part(prefix, dim)
    return idx, val, _dense_part(roles, full, prefix)


def extract_texts(
    texts: Sequence[str], dim: int = DEFAULT_DIM
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Extract features for texts that are each one user message.

    Row ``i`` of the returned CSR matrix and of the dense matrix equals
    ``extract([("user", texts[i])], dim)``.
    """
    _check_dim(dim)
    indptr = [0]
    indices: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dense = np.zeros((len(texts), NUM_DENSE), dtype=np.float64)
    for row, text in enumerate(texts):
        idx, val, dense[row] = extract([("user", text)], dim)
        indices.append(idx)
        data.append(val)
        indptr.append(indptr[-1] + len(idx))
    matrix = scipy.sparse.csr_matrix(
        (
            np.concatenate(data) if data else np.empty(0, dtype=np.float64),
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int32),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(len(texts), dim),
    )
    matrix.has_sorted_indices = True
    return matrix, dense


@dataclass(frozen=True)
class GoldenCase:
    """One hand-written input from cases.json, with repeat specs expanded."""

    name: str
    dim: int
    messages: list[tuple[str, str]]


def load_cases(path: Path) -> list[GoldenCase]:
    """Read cases.json. A message has ``text`` or ``repeat``
    (``pattern * count + suffix``); ``dim`` defaults to ``DEFAULT_DIM``.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document["spec_version"] != SPEC_VERSION:
        raise ValueError(f"cases.json spec_version {document['spec_version']} != {SPEC_VERSION}")
    cases = []
    for raw in document["cases"]:
        messages = []
        for message in raw["messages"]:
            if ("text" in message) == ("repeat" in message):
                raise ValueError(
                    f"case {raw['name']!r}: a message needs exactly one of text, repeat"
                )
            if "text" in message:
                text = message["text"]
            else:
                spec = message["repeat"]
                text = spec["pattern"] * spec["count"] + spec["suffix"]
            messages.append((message["role"], text))
        cases.append(GoldenCase(raw["name"], raw.get("dim", DEFAULT_DIM), messages))
    return cases


def write_golden(directory: Path) -> None:
    """Write ``golden.json`` next to ``cases.json`` in ``directory`` (LF line endings)."""
    directory = Path(directory)
    golden_cases = []
    for case in load_cases(directory / "cases.json"):
        idx, val, dense = extract(case.messages, case.dim)
        golden_cases.append(
            {
                "name": case.name,
                "dim": case.dim,
                "idx": idx.tolist(),
                "val": val.tolist(),
                "dense": dense.tolist(),
            }
        )
    document = {"spec_version": SPEC_VERSION, "cases": golden_cases}
    with open(directory / "golden.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m routepilot_train.features",
        description="Regenerate the feature spec v1 golden vectors from cases.json.",
    )
    parser.add_argument(
        "--write-golden",
        metavar="DIR",
        type=Path,
        required=True,
        help="directory holding cases.json; golden.json is written next to it",
    )
    args = parser.parse_args(argv)
    write_golden(args.write_golden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
