"""Score calibration: map a raw linear score ``s`` to a probability of success.

``fit_calibrator`` takes the scores and 0/1 outcomes of one model on the calibration
split only. With 1000 or more samples it fits an isotonic map on ``sigmoid(s)``, below
that a two-parameter Platt map on ``s``, which has far less variance on small samples.
The Task 13 pipeline enforces the split with ``require_splits`` before calling it.

``method`` selects the calibrator instead of that rule, for the calibration study: ``switch``
is the rule above (the default and the only one a shipped artifact uses), ``isotonic`` and
``platt`` force one of its two branches at any sample size, and ``histogram`` is 15
equal-mass bins on the raw score, each holding its empirical success rate. The histogram
calibrator is an experiment only: the artifact format has no place for it.
"""

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from routepilot_train.train.quality import RATE_CLIP, sigmoid

ISOTONIC_MIN = 1000
HISTOGRAM_BINS = 15
METHODS = ("isotonic", "platt", "histogram", "switch")


def _check(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"s and y must be 1-D arrays of the same length, got {s.shape}, {y.shape}")
    if s.size == 0:
        raise ValueError("s and y are empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0 or 1")
    return s, y.astype(np.int64)


def fit_calibrator(s: np.ndarray, y: np.ndarray, method: str = "switch") -> dict:
    """Fit a calibrator on calibration-split scores ``s`` and outcomes ``y``.

    The default ``method="switch"``: ``n >= ISOTONIC_MIN`` gives
    ``{"type": "isotonic", "x": [...], "y": [...]}``: the
    knots of an isotonic regression (clipped to [0, 1]) of ``y`` on ``sigmoid(s)``.
    Otherwise ``{"type": "platt", "a": a, "b": b}`` with ``p = sigmoid(a * s + b)`` from an
    unregularized logistic fit; a single-class ``y`` gives ``a = 0`` and ``b`` the logit of
    the observed rate clipped to ``[0.001, 0.999]``.

    ``method`` may also be ``"isotonic"`` or ``"platt"`` (that branch at any sample size) or
    ``"histogram"`` (``{"type": "histogram", "edges": [...], "rate": [...]}``, experiment
    only). Raises ``ValueError`` for another name.
    """
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method!r}, want one of {METHODS}")
    s, y = _check(s, y)
    if method == "histogram":
        return _fit_histogram(s, y)
    if method == "isotonic" or (method == "switch" and s.size >= ISOTONIC_MIN):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(sigmoid(s), y)
        return {
            "type": "isotonic",
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
        }
    if y.min() == y.max():
        rate = float(np.clip(y.mean(), RATE_CLIP, 1.0 - RATE_CLIP))
        return {"type": "platt", "a": 0.0, "b": math.log(rate / (1.0 - rate))}
    model = LogisticRegression(C=np.inf, solver="lbfgs")
    model.fit(s.reshape(-1, 1), y)
    return {"type": "platt", "a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}


def _histogram_index(edges: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The bin of each score: the number of edges at or below it (scores on an edge go up)."""
    return np.searchsorted(edges, s, side="right")


def _fit_histogram(s: np.ndarray, y: np.ndarray) -> dict:
    """``HISTOGRAM_BINS`` equal-mass bins on ``s``, each holding its empirical success rate.

    A bin starts at the first score of its group in sorted order. Tied scores stay in one
    bin, and edges that would leave a bin empty are dropped, so there are fewer bins than
    ``HISTOGRAM_BINS`` for few or heavily tied scores and every bin holds at least one sample.
    """
    ordered = np.sort(s)
    groups = np.array_split(np.arange(ordered.size), HISTOGRAM_BINS)
    edges = np.unique(ordered[[g[0] for g in groups[1:] if g.size]])
    edges = edges[edges > ordered[0]]
    index = _histogram_index(edges, s)
    rate = [float(y[index == b].mean()) for b in range(edges.size + 1)]
    return {"type": "histogram", "edges": [float(v) for v in edges], "rate": rate}


def apply_calibrator(cal: dict, s) -> np.ndarray:
    """Calibrated probabilities for scores ``s``.

    Isotonic is ``np.interp(sigmoid(s), x, y)``, which holds the end values outside
    the fitted range; Platt is ``sigmoid(a * s + b)``; histogram is the rate of the score's
    bin, the end bins for scores outside the fitted range.
    """
    s = np.asarray(s, dtype=np.float64)
    if cal["type"] == "isotonic":
        return np.interp(sigmoid(s), cal["x"], cal["y"])
    if cal["type"] == "platt":
        return sigmoid(cal["a"] * s + cal["b"])
    if cal["type"] == "histogram":
        return np.asarray(cal["rate"], dtype=np.float64)[
            _histogram_index(np.asarray(cal["edges"], dtype=np.float64), s)
        ]
    raise ValueError(f"unknown calibrator type {cal['type']!r}")
