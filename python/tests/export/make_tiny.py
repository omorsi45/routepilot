"""Build the tiny committed policy artifact for the Go tests (a script, not a test).

    .venv/Scripts/python.exe python/tests/export/make_tiny.py

It writes ``testdata/policy/tiny/<version>/`` from a seeded synthetic dataset of invented
texts: 2 models, 1500 train and 1200 calibration items, dim 1024, fitted with the real
train functions. The first model keeps the calibrator that ``fit_calibrator`` picks
(isotonic, 1200 samples); the second is forced to Platt on the same calibration scores, so
both calibrator types are covered. ``created_at`` is a fixed constant, so the whole
artifact is reproducible and ``test_tiny.py`` checks that it is.

The file name has no ``test_`` prefix, so pytest does not collect it; the tests import
``build_tiny``.
"""

import shutil
from pathlib import Path
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


def main() -> int:
    root = repo_root()
    out = write_tiny(root)
    print(f"tiny artifact: {out.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
