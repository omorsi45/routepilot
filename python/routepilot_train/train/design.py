"""The shared design matrix: hashed text features plus standardized dense features.

Every model of a pool is fitted on the same rows of this matrix, so the quality,
length and OOD models all see one feature definition. The dense columns are
standardized with the train-split mean and standard deviation (population
standard deviation, ``ddof=0``); the Go router applies the same two vectors.
"""

import numpy as np
import scipy.sparse

from routepilot_train import features

# Which columns a model is fitted on (the ablation): both blocks, the hashed block or the dense one.
FEATURE_SETS = ("both", "hashed", "dense")


def standardizer(dense_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each dense column over the train rows.

    A column with no variation (every row holds the same value) gets the exact constant as
    its mean and 1.0 as its std, so standardizing leaves it at exactly zero. Computing them
    would not: summing many equal values rounds, and the leftover std is noise-sized, which
    would blow up the first request that differs (for instance a two-message request when
    every training prompt had one message).
    """
    dense = np.asarray(dense_train, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] == 0:
        raise ValueError(f"dense_train must be a non-empty 2-D array, got shape {dense.shape}")
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    constant = np.ptp(dense, axis=0) == 0.0
    mean[constant] = dense[0, constant]
    std[constant] = 1.0
    return mean, std


def stack(
    hashed: scipy.sparse.csr_matrix, dense: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> scipy.sparse.csr_matrix:
    """``[hashed | (dense - mean) / std]`` from already extracted features.

    Lets a caller that needs the raw dense block first (to fit ``standardizer``) extract
    the features once and still build the same matrix as ``design_matrix``.
    """
    standardized = scipy.sparse.csr_matrix((dense - mean) / std)
    return scipy.sparse.hstack([hashed, standardized], format="csr", dtype=np.float64)


def check_feature_set(feature_set: str) -> None:
    """Raise ``ValueError`` unless ``feature_set`` is one of ``FEATURE_SETS``."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, want one of {FEATURE_SETS}")


def select_features(
    hashed: scipy.sparse.csr_matrix,
    dense: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_set: str = "both",
) -> scipy.sparse.csr_matrix:
    """The design matrix restricted to a ``feature_set``: ``both``, ``hashed`` or ``dense``.

    ``both`` is ``stack(hashed, dense, mean, std)``, ``hashed`` the hashed block alone and
    ``dense`` the standardized dense block alone, so a restricted model is fitted on exactly
    the columns it keeps. ``expand_weights`` puts its coefficients back into full length.
    """
    check_feature_set(feature_set)
    if feature_set == "both":
        return stack(hashed, dense, mean, std)
    if feature_set == "hashed":
        return scipy.sparse.csr_matrix(hashed)
    return scipy.sparse.csr_matrix((dense - mean) / std)


def expand_weights(coef: np.ndarray, dim: int, feature_set: str = "both") -> np.ndarray:
    """Coefficients fitted on ``select_features(..., feature_set)`` as a full-length vector.

    The artifact always holds ``dim + NUM_DENSE`` weights per model (hashed block first), so the
    Go router serves any feature set unchanged: the dropped block is exactly zero.
    """
    check_feature_set(feature_set)
    coef = np.asarray(coef, dtype=np.float64)
    kept = {
        "both": dim + features.NUM_DENSE,
        "hashed": dim,
        "dense": features.NUM_DENSE,
    }[feature_set]
    if coef.shape != (kept,):
        raise ValueError(
            f"feature set {feature_set!r} expects {kept} coefficients, got shape {coef.shape}"
        )
    full = np.zeros(dim + features.NUM_DENSE, dtype=np.float64)
    if feature_set == "dense":
        full[dim:] = coef
    else:
        full[:kept] = coef
    return full


def design_matrix(
    texts: list[str], mean: np.ndarray, std: np.ndarray, dim: int = features.DEFAULT_DIM
) -> scipy.sparse.csr_matrix:
    """``[hashed | (dense - mean) / std]`` for one user-message text per row.

    Shape ``(len(texts), dim + 12)``; the hashed block is the L2-normalized output of
    ``features.extract_texts`` and ``mean`` and ``std`` come from ``standardizer``.
    """
    hashed, dense = features.extract_texts(texts, dim)
    return stack(hashed, dense, mean, std)
