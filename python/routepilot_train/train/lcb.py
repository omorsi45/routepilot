"""Lower confidence bound tables: a conservative success probability per calibrated bin.

``fit_lcb`` takes calibrated probabilities ``p`` and 0/1 outcomes ``y`` from the
calibration split only (the pipeline enforces the split with ``require_splits``).
It cuts the samples into equal-mass bins by ``p``, merges bins that are too small, and
stores the one-sided Clopper-Pearson lower bound of each bin's success rate. The router
looks up the bound for a request's ``p`` and uses it as ``l``.
"""

from bisect import bisect_left

import numpy as np
from scipy.stats import beta

BINS = 15
MIN_BIN = 30
ALPHA = 0.05


def clopper_pearson_lower(k: int, n: int, alpha: float = ALPHA) -> float:
    """The one-sided lower ``1 - alpha`` confidence bound for ``k`` successes in ``n`` trials.

    It is 0 when ``k == 0`` and otherwise the ``alpha`` quantile of Beta(k, n - k + 1).
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def _merged_sizes(n: int) -> list[int]:
    """Bin sizes after ``np.array_split`` into ``BINS`` and merging the small bins.

    While more than one bin remains and some bin holds fewer than ``MIN_BIN`` samples,
    the leftmost such bin merges into its right neighbor, or into its left neighbor when
    it is the last bin. So fewer than ``MIN_BIN`` samples in total leave a single bin.
    """
    sizes = [len(part) for part in np.array_split(np.arange(n), BINS)]
    while len(sizes) > 1:
        small = next((i for i, size in enumerate(sizes) if size < MIN_BIN), None)
        if small is None:
            break
        merged = sizes.pop(small)
        # After the pop the right neighbor sits at index ``small``; the last bin has none.
        sizes[min(small, len(sizes) - 1)] += merged
    return sizes


def fit_lcb(p: np.ndarray, y: np.ndarray) -> dict:
    """Build the lower-bound table from calibration-split probabilities and outcomes.

    Returns ``{"upper": [...], "lower": [...], "n": [...], "pooled_rate": float,
    "pooled_lcb": float}``. ``upper[i]`` is the largest ``p`` in bin ``i`` (the last
    entry is 1.0 so every ``p`` finds a bin), ``lower`` is each bin's bound turned into a
    running maximum from low to high ``p``, and ``pooled_*`` describe all samples at once.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError(f"p and y must be 1-D arrays of the same length, got {p.shape}, {y.shape}")
    if p.size == 0:
        raise ValueError("p and y are empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0 or 1")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("p must be finite and between 0 and 1")
    order = np.argsort(p, kind="stable")
    p_sorted, y_sorted = p[order], y[order].astype(np.int64)
    upper, lower, counts = [], [], []
    start = 0
    for size in _merged_sizes(p.size):
        stop = start + size
        upper.append(float(p_sorted[stop - 1]))
        lower.append(clopper_pearson_lower(int(y_sorted[start:stop].sum()), size))
        counts.append(size)
        start = stop
    upper[-1] = 1.0
    return {
        "upper": upper,
        "lower": [float(v) for v in np.maximum.accumulate(lower)],
        "n": counts,
        "pooled_rate": float(y_sorted.mean()),
        "pooled_lcb": clopper_pearson_lower(int(y_sorted.sum()), p.size),
    }


def lookup_lcb(table: dict, p: float) -> float:
    """The lower bound of the first bin with ``p <= upper[i]``, else of the last bin."""
    i = bisect_left(table["upper"], p)
    return table["lower"][min(i, len(table["lower"]) - 1)]
