"""Tests for the shared design matrix and the per-model quality regression."""

import math

import numpy as np
import pytest
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from routepilot_train import features
from routepilot_train.train import design, quality

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
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
]


def synthetic_texts(n: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Short texts of filler words; the token ``zephyr`` is present iff the label is 1."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    texts = []
    for label in y:
        words = [str(w) for w in rng.choice(FILLER, size=8)]
        if label:
            words[int(rng.integers(0, 8))] = "zephyr"
        texts.append(" ".join(words))
    return texts, y.astype(np.int64)


def synthetic_design(n: int = 300, seed: int = 0) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    texts, y = synthetic_texts(n, seed)
    _, dense = features.extract_texts(texts, DIM)
    mean, std = design.standardizer(dense)
    return design.design_matrix(texts, mean, std, DIM), y


def test_standardizer_zero_std_becomes_one():
    dense = np.array([[1.0, 5.0, 0.0], [3.0, 5.0, 2.0], [5.0, 5.0, 4.0]])
    mean, std = design.standardizer(dense)
    np.testing.assert_allclose(mean, [3.0, 5.0, 2.0])
    np.testing.assert_allclose(std, [math.sqrt(8 / 3), 1.0, math.sqrt(8 / 3)])


def test_design_matrix_is_hashed_then_standardized_dense():
    texts, _ = synthetic_texts(20)
    hashed, dense = features.extract_texts(texts, DIM)
    mean, std = design.standardizer(dense)
    X = design.design_matrix(texts, mean, std, DIM)
    assert scipy.sparse.isspmatrix_csr(X)
    assert X.dtype == np.float64
    assert X.shape == (20, DIM + features.NUM_DENSE)
    np.testing.assert_array_equal(X[:, :DIM].toarray(), hashed.toarray())
    np.testing.assert_allclose(X[:, DIM:].toarray(), (dense - mean) / std)


def test_design_matrix_default_dim_is_the_feature_default():
    mean, std = np.zeros(features.NUM_DENSE), np.ones(features.NUM_DENSE)
    X = design.design_matrix(["hello world"], mean, std)
    assert X.shape == (1, features.DEFAULT_DIM + features.NUM_DENSE)


def test_fit_quality_learns_signal():
    X, y = synthetic_design()
    params = quality.fit_quality(X, y)
    assert params["coef"].shape == (X.shape[1],)
    assert params["coef"].dtype == np.float64
    assert roc_auc_score(y, quality.raw_score(params, X)) > 0.95


def test_cv_picks_from_grid_deterministically():
    X, y = synthetic_design(n=120)
    first = quality.fit_quality(X, y)
    second = quality.fit_quality(X.copy(), y.copy())
    assert first["C"] in quality.C_GRID
    assert math.isfinite(first["cv_log_loss"]) and first["cv_log_loss"] > 0
    assert first["C"] == second["C"]
    assert first["cv_log_loss"] == second["cv_log_loss"]
    assert first["intercept"] == second["intercept"]
    np.testing.assert_array_equal(first["coef"], second["coef"])


def test_cv_tie_picks_the_smaller_c(monkeypatch: pytest.MonkeyPatch):
    X, y = synthetic_design(n=60)
    monkeypatch.setattr(quality, "log_loss", lambda *args, **kwargs: 0.5)
    params = quality.fit_quality(X, y)
    assert params["C"] == min(quality.C_GRID)
    assert params["cv_log_loss"] == 0.5


def test_raw_score_matches_sklearn_decision_function():
    X, y = synthetic_design(n=150)
    params = quality.fit_quality(X, y)
    reference = LogisticRegression(solver="liblinear", max_iter=1000, C=params["C"], l1_ratio=0)
    reference.fit(X, y)
    np.testing.assert_allclose(
        quality.raw_score(params, X), reference.decision_function(X), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(params["coef"], reference.coef_[0], rtol=0, atol=1e-12)
    assert params["intercept"] == pytest.approx(float(reference.intercept_[0]), abs=1e-12)


def test_sigmoid_is_stable_at_the_extremes():
    s = np.array([-1000.0, 0.0, 1000.0])
    np.testing.assert_allclose(quality.sigmoid(s), [0.0, 0.5, 1.0])


@pytest.mark.parametrize(("label", "p"), [(1, 0.999), (0, 0.001)])
def test_single_class_returns_the_constant_model(label: int, p: float):
    X, _ = synthetic_design(n=40)
    y = np.full(40, label, dtype=np.int64)
    params = quality.fit_quality(X, y)
    assert params["C"] is None
    np.testing.assert_array_equal(params["coef"], np.zeros(X.shape[1]))
    assert params["intercept"] == pytest.approx(math.log(p / (1 - p)))
    np.testing.assert_allclose(quality.sigmoid(quality.raw_score(params, X)), p)
    want_loss = -math.log(p) if label == 1 else -math.log(1 - p)
    assert params["cv_log_loss"] == pytest.approx(want_loss)


def test_a_cv_fold_without_both_classes_does_not_crash():
    # One positive: the fold that holds it out trains on a single class.
    X, _ = synthetic_design(n=50)
    y = np.zeros(50, dtype=np.int64)
    y[7] = 1
    params = quality.fit_quality(X, y)
    assert params["C"] in quality.C_GRID
    assert math.isfinite(params["cv_log_loss"])


def test_fit_quality_rejects_bad_input():
    X, y = synthetic_design(n=30)
    with pytest.raises(ValueError, match="0 or 1"):
        quality.fit_quality(X, y + 1)
    with pytest.raises(ValueError, match="rows"):
        quality.fit_quality(X, y[:-1])
    with pytest.raises(ValueError, match="at least"):
        quality.fit_quality(X[:4], y[:4])
