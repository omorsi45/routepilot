"""Calibration metrics for predicted success probabilities.

``p`` is a forecast in [0, 1] and ``y`` the observed 0/1 outcome. Binning is by equal
mass (each bin holds about the same number of forecasts, ties broken by input order)
or by equal width on [0, 1].
"""

import numpy as np
import pandas as pd

SCHEMES = ("mass", "width")


def _check(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError(f"p and y must be 1-D arrays of the same length, got {p.shape}, {y.shape}")
    if p.size == 0:
        raise ValueError("p and y are empty")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("p must be finite and between 0 and 1")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0 or 1")
    return p, y.astype(float)


def reliability_table(p, y, bins: int = 15, scheme: str = "mass") -> pd.DataFrame:
    """Per-bin count, mean forecast and observed rate; empty bins are left out.

    Columns: ``bin`` (0-based index), ``n``, ``mean_p``, ``mean_y``.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    p, y = _check(p, y)
    if scheme == "mass":
        order = np.argsort(p, kind="stable")
        groups = [g for g in np.array_split(order, bins) if g.size]
        index = np.concatenate([np.full(g.size, i) for i, g in enumerate(groups)])
        members = np.empty(p.size, dtype=int)
        members[order] = index
    else:
        members = np.minimum((p * bins).astype(int), bins - 1)
    frame = pd.DataFrame({"bin": members, "p": p, "y": y})
    grouped = frame.groupby("bin").agg(n=("p", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
    return grouped.reset_index()


def ece(p, y, bins: int = 15, scheme: str = "mass") -> float:
    """Expected calibration error: the count-weighted mean of |observed rate - mean forecast|."""
    table = reliability_table(p, y, bins=bins, scheme=scheme)
    gap = (table["mean_y"] - table["mean_p"]).abs()
    return float((table["n"] * gap).sum() / table["n"].sum())


def brier(p, y) -> float:
    """Mean squared error of the forecast."""
    p, y = _check(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    """Mean negative log likelihood, with forecasts clipped to [eps, 1 - eps]."""
    p, y = _check(p, y)
    clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))
