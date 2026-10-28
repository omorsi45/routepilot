"""Tests for the output length model (ridge on log1p tokens, Duan smearing)."""

import json
import math

import numpy as np
import pytest

from routepilot_train import features
from routepilot_train.train import design, length

DIM = 1024
FILLER = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
]
LONG_BOOST = 2.0  # log1p tokens added when the text asks for an essay
BASE = 4.0


def synthetic_split(n: int, seed: int, shift: float = 0.0, noise: float = 0.1):
    """Texts whose log1p(completion tokens) is BASE + LONG_BOOST * [essay] + noise + shift."""
    rng = np.random.default_rng(seed)
    long = rng.integers(0, 2, size=n)
    texts = []
    for flag in long:
        words = [str(w) for w in rng.choice(FILLER, size=6)]
        if flag:
            words[int(rng.integers(0, 6))] = "essay"
        texts.append(" ".join(words))
    log_out = BASE + LONG_BOOST * long + shift + rng.normal(0.0, noise, size=n)
    return texts, np.rint(np.expm1(log_out)).astype(np.int64), long


def build(train, calib):
    _, dense = features.extract_texts(train[0], DIM)
    mean, std = design.standardizer(dense)
    return (
        design.design_matrix(train[0], mean, std, DIM),
        design.design_matrix(calib[0], mean, std, DIM),
    )


def fit_synthetic(shift_calib: float = 0.0, n_train: int = 400, n_calib: int = 300):
    train = synthetic_split(n_train, seed=0)
    calib = synthetic_split(n_calib, seed=1, shift=shift_calib)
    X_train, X_calib = build(train, calib)
    prompt_tokens = np.full(n_train, 50)
    in_upper = np.full(n_train, 100)
    params = length.fit_length(X_train, train[1], X_calib, calib[1], prompt_tokens, in_upper)
    return params, X_train, X_calib, train, calib


def test_constants():
    assert length.ALPHA_GRID == (0.1, 1.0, 10.0, 100.0)
    assert length.QUANTILES == (0.5, 0.9, 0.95, 0.99)


def test_length_recovers_log_linear_signal():
    params, _, X_calib, _, calib = fit_synthetic()
    assert params["alpha"] in length.ALPHA_GRID
    assert params["coef"].shape == (X_calib.shape[1],)
    pred = length.predict_log(params, X_calib)
    long = calib[2].astype(bool)
    gap = pred[long].mean() - pred[~long].mean()
    assert gap == pytest.approx(LONG_BOOST, abs=0.15)
    assert np.corrcoef(pred, np.log1p(calib[1]))[0, 1] > 0.99


def test_duan_smearing_uses_calibration_residuals():
    # The calibration outputs sit 0.5 above the train relationship, so a smearing factor
    # taken from the train residuals (about 1.005) would differ from the calibration
    # one (about exp(0.5) = 1.65).
    params, X_train, X_calib, train, calib = fit_synthetic(shift_calib=0.5)
    r_calib = np.log1p(calib[1]) - length.predict_log(params, X_calib)
    r_train = np.log1p(train[1]) - length.predict_log(params, X_train)
    assert params["smear"] == pytest.approx(float(np.mean(np.exp(r_calib))), abs=1e-12)
    assert params["smear"] > 1.5
    assert float(np.mean(np.exp(r_train))) < 1.1
    for q in length.QUANTILES:
        assert params["residual_quantiles"][str(q)] == pytest.approx(
            float(np.quantile(r_calib, q)), abs=1e-12
        )


def test_quantiles_ordered():
    params, _, X_calib, _, _ = fit_synthetic()
    quantiles = [params["residual_quantiles"][str(q)] for q in length.QUANTILES]
    assert list(params["residual_quantiles"]) == [str(q) for q in length.QUANTILES]
    assert quantiles == sorted(quantiles)
    assert quantiles[0] < quantiles[-1]
    curve = np.stack([length.out_quantile(params, X_calib, q) for q in length.QUANTILES])
    assert (np.diff(curve, axis=0) >= 0).all()
    np.testing.assert_allclose(
        length.out_quantile(params, X_calib, 0.9),
        np.exp(length.predict_log(params, X_calib) + quantiles[1]) - 1,
    )


def test_expected_out_is_smeared_exponential():
    params, _, X_calib, _, _ = fit_synthetic()
    np.testing.assert_allclose(
        length.expected_out(params, X_calib),
        np.exp(length.predict_log(params, X_calib)) * params["smear"] - 1,
    )


def test_out_quantile_rejects_an_unfitted_level():
    params, _, X_calib, _, _ = fit_synthetic()
    with pytest.raises(ValueError, match="quantile"):
        length.out_quantile(params, X_calib, 0.75)


def test_in_ratio_median():
    train = synthetic_split(60, seed=0)
    calib = synthetic_split(40, seed=1)
    X_train, X_calib = build(train, calib)
    # Ratios 0.4, 0.5, 0.6, 0.8 are used (median 0.55); the zero-prompt row is skipped
    # (counting it as 0 would give 0.5).
    prompt_tokens = np.array([40, 50, 60, 80, 0] + [0] * 55)
    in_upper = np.array([100, 100, 100, 100, 100] + [100] * 55)
    params = length.fit_length(X_train, train[1], X_calib, calib[1], prompt_tokens, in_upper)
    assert params["in_ratio"] == pytest.approx(0.55)
    # Everything is skipped when no row has prompt tokens.
    with pytest.raises(ValueError, match="prompt_tokens"):
        length.fit_length(X_train, train[1], X_calib, calib[1], np.zeros(60), np.full(60, 100))


def test_negative_completion_rows_are_dropped():
    train = synthetic_split(120, seed=0)
    calib = synthetic_split(80, seed=1)
    X_train, X_calib = build(train, calib)
    prompt_tokens = np.full(120, 30)
    in_upper = np.full(120, 60)
    clean = length.fit_length(
        X_train[3:], train[1][3:], X_calib[2:], calib[1][2:], prompt_tokens[3:], in_upper[3:]
    )
    bad_train, bad_calib = train[1].copy(), calib[1].copy()
    bad_train[:3] = -1
    bad_calib[:2] = -1
    dirty = length.fit_length(X_train, bad_train, X_calib, bad_calib, prompt_tokens, in_upper)
    assert math.isfinite(dirty["smear"])
    np.testing.assert_array_equal(dirty["coef"], clean["coef"])
    for key in ("intercept", "alpha", "smear", "residual_quantiles", "in_ratio"):
        assert dirty[key] == clean[key]


def test_fit_is_deterministic_and_json_friendly():
    first = fit_synthetic()[0]
    second = fit_synthetic()[0]
    np.testing.assert_array_equal(first["coef"], second["coef"])
    for key in ("intercept", "alpha", "smear", "residual_quantiles", "in_ratio"):
        assert first[key] == second[key]
    scalars = {k: v for k, v in first.items() if k != "coef"}
    assert json.loads(json.dumps(scalars)) == scalars


def test_fit_length_rejects_bad_input():
    train = synthetic_split(30, seed=0)
    calib = synthetic_split(20, seed=1)
    X_train, X_calib = build(train, calib)
    ones = np.full(30, 10)
    with pytest.raises(ValueError, match="rows"):
        length.fit_length(X_train, train[1][:-1], X_calib, calib[1], ones, ones)
    with pytest.raises(ValueError, match="rows"):
        length.fit_length(X_train, train[1], X_calib, calib[1][:-1], ones, ones)
    with pytest.raises(ValueError, match="prompt_tokens"):
        length.fit_length(X_train, train[1], X_calib, calib[1], ones[:-1], ones)
    with pytest.raises(ValueError, match="calibration"):
        length.fit_length(X_train, train[1], X_calib, np.full(20, -1), ones, ones)
    with pytest.raises(ValueError, match="in_upper"):
        length.fit_length(X_train, train[1], X_calib, calib[1], ones, np.zeros(30))
