"""Build the tiny committed policy artifact for the Go tests (a script, not a test).

    .venv/Scripts/python.exe python/tests/export/make_tiny.py

It writes ``testdata/policy/tiny/<version>/`` from a seeded synthetic dataset of invented
texts: 2 models, 1500 train and 1200 calibration items, dim 1024, fitted with the real
train functions. The first model keeps the calibrator that ``fit_calibrator`` picks
(isotonic, 1200 samples); the second is forced to Platt on the same calibration scores, so
both calibrator types are covered. ``created_at`` is a fixed constant, so the whole
artifact is reproducible and ``test_tiny.py`` checks that it is.

It also writes ``testdata/policy/parity/cases.json``: 30 invented prompts with the reference
predictor's output on that artifact, which the Go tests must reproduce.

The file name has no ``test_`` prefix, so pytest does not collect it; the tests import
``build_tiny``.
"""

import json
import shutil
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from routepilot_train import features
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root
from routepilot_train.train import calibrate, design, lcb, length, ood, quality

SEED = 0
DIM = 1024
N_TRAIN = 1500
N_CALIB = 1200
MODELS = ("tiny-weak", "tiny-strong")
CREATED_AT = "2026-10-28T00:00:00Z"
SYNTHETIC = "none (synthetic data)"
N_TOPICS = 4
WORDS_PER_TOPIC = 25
SHARED_WORDS = 30
HARD_WORDS = 5  # the first words of a topic make a prompt harder

_SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
_FENCE = "```"


def _word(rng: np.random.Generator) -> str:
    count = int(rng.integers(2, 4))
    return "".join(str(rng.choice(_SYLLABLES)) for _ in range(count))


def make_vocab(rng: np.random.Generator) -> tuple[list[list[str]], list[str]]:
    """Invented pseudo-words: one list per topic and one list shared by all topics."""
    topics = [[_word(rng) for _ in range(WORDS_PER_TOPIC)] for _ in range(N_TOPICS)]
    return topics, [_word(rng) for _ in range(SHARED_WORDS)]


def make_items(
    rng: np.random.Generator, topics: list[list[str]], shared: list[str], n: int
) -> dict[str, np.ndarray | list[str]]:
    """``n`` invented prompts with a quality outcome and a length per model.

    A prompt draws its words mostly from its topic's list. Some get a question mark, a
    multiple-choice block, digits, a code fence, a LaTeX marker or non-ASCII text, so every
    dense feature varies. The outcomes depend on the topic, the number of hard words and
    whether the prompt is a question, with noise.
    """
    texts: list[str] = []
    success = np.zeros((n, len(MODELS)), dtype=np.int64)
    out_tokens = np.zeros((n, len(MODELS)), dtype=np.int64)
    prompt_tokens = np.zeros((n, len(MODELS)), dtype=np.int64)
    prompt_bytes = np.zeros(n, dtype=np.int64)
    for row in range(n):
        topic = int(rng.integers(0, N_TOPICS))
        words = [
            str(rng.choice(topics[topic] if rng.random() < 0.6 else shared))
            for _ in range(int(rng.integers(4, 80)))
        ]
        hard = sum(word in topics[topic][:HARD_WORDS] for word in words)
        question = rng.random() < 0.3
        text = " ".join(words) + (" ?" if question else "")
        if rng.random() < 0.2:
            text = "A. one\nB. two\nC. three\nD. four\n" + text
        if rng.random() < 0.15:
            text += " 12 + 7 = 19 and 408 / 4"
        if rng.random() < 0.1:
            text += f"\n{_FENCE}\nx = 1\n{_FENCE}"
        if rng.random() < 0.1:
            text += r" \frac{1}{2}"
        if rng.random() < 0.1:
            text += " caf\u00e9"
        texts.append(text)
        prompt_bytes[row] = len(text.encode("utf-8"))
        for m in range(len(MODELS)):
            logit = (0.2, 1.0)[m] + (0.4, -0.2, 0.1, -0.3)[topic] - 0.25 * hard + 0.5 * question
            success[row, m] = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            mean_log = 3.0 + 0.4 * topic + 0.01 * len(words) + 0.4 * m
            out_tokens[row, m] = max(1, int(np.exp(rng.normal(mean_log, 0.6))))
            prompt_tokens[row, m] = max(1, int(prompt_bytes[row] / 4 * rng.uniform(0.8, 1.2)))
    return {
        "texts": texts,
        "success": success,
        "out": out_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_bytes": prompt_bytes,
    }


def build_tiny() -> tuple[dict, dict[str, np.ndarray]]:
    """Fit the tiny pool and return the ``(bundle, weights)`` that ``export_artifact`` takes."""
    rng = np.random.default_rng(SEED)
    topics, shared = make_vocab(rng)
    train = make_items(rng, topics, shared, N_TRAIN)
    calib = make_items(rng, topics, shared, N_CALIB)

    h_train, d_train = features.extract_texts(train["texts"], DIM)
    h_calib, d_calib = features.extract_texts(calib["texts"], DIM)
    mean, std = design.standardizer(d_train)
    x_train = design.stack(h_train, d_train, mean, std)
    x_calib = design.stack(h_calib, d_calib, mean, std)
    in_upper = train["prompt_bytes"] + 24  # one user message: bytes + 8 + 16

    models = []
    for m, model_id in enumerate(MODELS):
        fitted = quality.fit_quality(x_train, train["success"][:, m])
        s_calib = quality.raw_score(fitted, x_calib)
        y_calib = calib["success"][:, m]
        if m == 1:
            # Same scores, but the Platt branch: pretend there are too few samples for isotonic.
            with mock.patch.object(calibrate, "ISOTONIC_MIN", len(s_calib) + 1):
                calibrator = calibrate.fit_calibrator(s_calib, y_calib)
        else:
            calibrator = calibrate.fit_calibrator(s_calib, y_calib)
        table = lcb.fit_lcb(calibrate.apply_calibrator(calibrator, s_calib), y_calib)
        fitted_length = length.fit_length(
            x_train,
            train["out"][:, m],
            x_calib,
            calib["out"][:, m],
            train["prompt_tokens"][:, m],
            in_upper,
        )
        models.append(
            {
                "id": model_id,
                "quality": fitted,
                "calibrator": calibrator,
                "lcb": table,
                "length": fitted_length,
            }
        )
    detector = ood.fit_ood(h_train, h_calib)
    return artifact.assemble_bundle(
        dim=DIM,
        dense_mean=mean,
        dense_std=std,
        models=models,
        ood=detector,
        training={
            "pool": "tiny",
            "dataset_sha256": SYNTHETIC,
            "split_salt": SYNTHETIC,
            "git_sha": None,
            "created_at": CREATED_AT,
        },
    )


def tiny_dir(root: Path) -> Path:
    """``<root>/testdata/policy/tiny``, the parent of the single version directory."""
    return Path(root) / "testdata" / "policy" / "tiny"


def write_tiny(root: Path) -> Path:
    """Rebuild ``testdata/policy/tiny/<version>/`` under ``root`` and return that directory."""
    target = tiny_dir(root)
    if target.exists():
        shutil.rmtree(target)
    bundle, weights = build_tiny()
    return artifact.export_artifact(target.parent, "tiny", bundle, weights)


_PROSE = "The quick brown fox jumps over the lazy dog near the old stone bridge. "
_PUNCTUATION = "!!! ??? ... ;;; ::: --- ((( ))) [[[ ]]] {{{ }}} @@@ ### %%% ^^^ &&& *** +++ ==="
_CJK = (
    "\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f"
    "\u8bf7\u7528\u4e00\u53e5\u8bdd\u56de\u7b54\u3002"
)
_ARABIC = "\u0645\u0627 \u0647\u064a \u0639\u0627\u0635\u0645\u0629 \u0645\u0635\u0631\u061f"


def _text(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "text": text}


def _repeat(role: str, pattern: str, count: int, suffix: str = "") -> dict[str, Any]:
    return {"role": role, "repeat": {"pattern": pattern, "count": count, "suffix": suffix}}


def expand_messages(specs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Expand message specs (``text`` or ``repeat``: pattern * count + suffix) to (role, text)."""
    messages = []
    for spec in specs:
        if ("text" in spec) == ("repeat" in spec):
            raise ValueError(f"a message needs exactly one of text, repeat: {spec!r}")
        if "text" in spec:
            text = spec["text"]
        else:
            rep = spec["repeat"]
            text = rep["pattern"] * rep["count"] + rep["suffix"]
        messages.append((spec["role"], text))
    return messages


def parity_specs() -> list[tuple[str, list[dict[str, Any]]]]:
    """30 invented prompts: edge cases for feature drift plus four in-distribution texts.

    The edge cases are an empty text, a text of only separators, a 40 KiB text cut inside a
    UTF-8 sequence by the 32768-byte prefix, non-ASCII runs and every dense marker. Nothing
    here is dataset text.
    """
    user = "user"
    fixed: list[tuple[str, list[dict[str, Any]]]] = [
        ("empty-text", [_text(user, "")]),
        ("whitespace-only", [_text(user, "  \n\t  \n")]),
        ("short-question", [_text(user, "What is the capital of Narvonia?")]),
        ("digits-heavy", [_text(user, "3141592653589793 2718281828459045 1618033988749895")]),
        (
            "uppercase-heavy",
            [_text(user, "PLEASE SUMMARIZE THE FOLLOWING NOTICE IN CAPITAL LETTERS")],
        ),
        (
            "punctuation-heavy",
            [_text(user, _PUNCTUATION)],
        ),
        ("code-fence", [_text(user, "```python\nprint('hello')\n```")]),
        ("two-fences", [_text(user, "First:\n```\na = 1\n```\nSecond:\n```\nb = 2\n```")]),
        ("latex-frac", [_text(user, r"Simplify \frac{3}{4} + \frac{1}{6}.")]),
        ("latex-sum", [_text(user, r"Compute $$\sum_{i=1}^{10} i^2$$ exactly.")]),
        ("latex-paren", [_text(user, r"Is \( x^2 + 1 \) ever zero?")]),
        ("latex-bracket", [_text(user, r"Solve \[ 2x + 3 = 11 \] for x.")]),
        (
            "choice-upper",
            [_text(user, "Pick the heaviest.\nA. feather\nB. brick\nC. cloud\nD. whisper")],
        ),
        ("choice-paren-lower", [_text(user, "Which one?\n(a) red\n(b) green\n(c) blue")]),
        (
            "choice-colon-indented",
            [_text(user, "Select:\n  C: the third option\n  D: the fourth option")],
        ),
        ("not-a-choice-line", [_text(user, "A.B. is an abbreviation, and e.g. so is this")]),
        (
            "system-and-user",
            [_text("system", "Answer as a pirate."), _text(user, "How do tides work?")],
        ),
        (
            "multi-turn",
            [
                _text("system", "Be brief."),
                _text(user, "Name a primary colour."),
                _text("assistant", "Red."),
                _text(user, "And another?"),
            ],
        ),
        (
            "latin-accents",
            [_text(user, "Un caf\u00e9 cr\u00e8me s'il vous pla\u00eet, na\u00efve fa\u00e7ade")],
        ),
        (
            "cjk",
            [_text(user, _CJK)],
        ),
        (
            "arabic",
            [_text(user, _ARABIC)],
        ),
        ("emoji-mixed", [_text(user, "ship it \U0001f680 then rest \U0001f634 and 2 more")]),
        ("long-prose-3kb", [_repeat(user, _PROSE, 45)]),
        # 13654 three-byte characters are 40962 bytes; 32768 = 3 * 10922 + 2, so the prefix
        # ends after two of the three bytes of a character.
        ("long-40kib-cut-mid-rune", [_repeat(user, "\u20ac", 13654, "?")]),
        ("long-ascii-over-prefix", [_repeat(user, "zorb quill mep ", 3000, " tail?")]),
        ("many-lines", [_repeat(user, "line of text\n", 300)]),
    ]
    topics, shared = make_vocab(np.random.default_rng(SEED))
    in_distribution = make_items(np.random.default_rng(SEED + 1), topics, shared, 4)["texts"]
    synthetic = [
        (f"synthetic-{i + 1}", [_text(user, text)]) for i, text in enumerate(in_distribution)
    ]
    return fixed + synthetic


def parity_document(manifest: dict[str, Any], blob: bytes) -> dict[str, Any]:
    """The parity cases with the reference predictor's output for ``manifest`` and ``blob``."""
    cases = []
    for name, specs in parity_specs():
        messages = expand_messages(specs)
        predicted = artifact.predict_bundle(manifest, blob, messages)
        in_upper = sum(len(text.encode("utf-8")) for _, text in messages) + 8 * len(messages) + 16
        cases.append(
            {
                "name": name,
                "messages": specs,
                "in_upper": in_upper,
                "ood_score": predicted["ood_score"],
                "ood": predicted["ood"],
                "models": predicted["models"],
            }
        )
    return {"artifact_version": manifest["version"], "cases": cases}


def write_parity(root: Path, artifact_dir: Path) -> Path:
    """Write ``testdata/policy/parity/cases.json`` for the artifact in ``artifact_dir``."""
    manifest, blob = artifact.load_artifact(artifact_dir)
    document = parity_document(manifest, blob)
    text = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False)
    path = Path(root) / "testdata" / "policy" / "parity" / "cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((text + "\n").encode("ascii"))
    return path


def main() -> int:
    root = repo_root()
    out = write_tiny(root)
    parity = write_parity(root, out)
    print(f"tiny artifact: {out.relative_to(root).as_posix()}")
    print(f"parity cases: {parity.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
