"""Per-model quality regression: an L2 logistic model of P(success | features).

One model is fitted per pool model, all on the same design matrix rows. The inputs
of ``fit_quality`` must come from the train split only: it tunes ``C`` by 5-fold
cross validation inside them, and the calibration and test splits stay unread. The
pipeline enforces the split with ``require_splits`` before calling it.
"""

import math

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
# A single-class model predicts the observed rate, kept away from 0 and 1 so the
# score stays finite.
RATE_CLIP = 1e-3


def sigmoid(s) -> np.ndarray:
    """The logistic function, stable for large positive and negative scores."""
    return expit(np.asarray(s, dtype=np.float64))


def raw_score(params: dict, X) -> np.ndarray:
    """The linear score ``s = X @ coef + intercept``, one value per row of ``X``."""
    return np.asarray(X @ params["coef"], dtype=np.float64).ravel() + params["intercept"]


def _constant_intercept(y: np.ndarray) -> float:
    rate = float(np.clip(y.mean(), RATE_CLIP, 1.0 - RATE_CLIP))
    return math.log(rate / (1.0 - rate))


def _fit(X, y: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Coefficients and intercept at ``C``; the constant model when ``y`` has one class."""
    if y.min() == y.max():
        return np.zeros(X.shape[1], dtype=np.float64), _constant_intercept(y)
    model = LogisticRegression(solver="liblinear", max_iter=1000, C=C, l1_ratio=0)
    model.fit(X, y)
    return model.coef_[0].astype(np.float64), float(model.intercept_[0])


def _check(X, y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has shape {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0 or 1")
    if y.shape[0] < N_FOLDS:
        raise ValueError(f"need at least {N_FOLDS} rows for {N_FOLDS}-fold CV, got {y.shape[0]}")
    return y.astype(np.int64)


def fit_quality(X, y: np.ndarray, groups_seed: int = 0) -> dict:
    """Fit the quality model on train rows and pick ``C`` from ``C_GRID`` by CV log-loss.

    ``X`` is the design matrix and ``y`` the 0/1 success of one model on the same rows;
    both must come from the train split. The folds are
    ``KFold(5, shuffle=True, random_state=groups_seed)``; the chosen ``C`` has the lowest
    mean fold log-loss, the smaller ``C`` on a tie. A fold whose training rows hold one
    class predicts the constant rate. When ``y`` itself has one class no model is fitted:
    ``coef`` is zero, ``C`` is ``None`` and the intercept is the logit of the clipped rate.

    Returns ``{"coef": float64 (n_features,), "intercept": float, "C": float | None,
    "cv_log_loss": float}``.
    """
    y = _check(X, y)
    if y.min() == y.max():
        intercept = _constant_intercept(y)
        constant = np.full(y.shape[0], float(sigmoid(intercept)))
        return {
            "coef": np.zeros(X.shape[1], dtype=np.float64),
            "intercept": intercept,
            "C": None,
            "cv_log_loss": float(log_loss(y, constant, labels=[0, 1])),
        }
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=groups_seed).split(np.arange(len(y))))
    best_c, best_loss = C_GRID[0], math.inf
    for C in C_GRID:
        losses = []
        for train, valid in folds:
            coef, intercept = _fit(X[train], y[train], C)
            p = sigmoid(np.asarray(X[valid] @ coef, dtype=np.float64).ravel() + intercept)
            losses.append(float(log_loss(y[valid], p, labels=[0, 1])))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:  # strict: the first (smallest) C wins a tie
            best_c, best_loss = C, mean_loss
    coef, intercept = _fit(X, y, best_c)
    return {"coef": coef, "intercept": intercept, "C": best_c, "cv_log_loss": best_loss}
