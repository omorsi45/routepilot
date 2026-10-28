"""The shared design matrix: hashed text features plus standardized dense features.

Every model of a pool is fitted on the same rows of this matrix, so the quality,
length and OOD models all see one feature definition. The dense columns are
standardized with the train-split mean and standard deviation (population
standard deviation, ``ddof=0``); the Go router applies the same two vectors.
"""

import numpy as np
import scipy.sparse

from routepilot_train import features


def standardizer(dense_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each dense column over the train rows.

    A column with zero standard deviation gets 1.0 so standardizing leaves it at zero.
    """
    dense = np.asarray(dense_train, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] == 0:
        raise ValueError(f"dense_train must be a non-empty 2-D array, got shape {dense.shape}")
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def design_matrix(
    texts: list[str], mean: np.ndarray, std: np.ndarray, dim: int = features.DEFAULT_DIM
) -> scipy.sparse.csr_matrix:
    """``[hashed | (dense - mean) / std]`` for one user-message text per row.

    Shape ``(len(texts), dim + 12)``; the hashed block is the L2-normalized output of
    ``features.extract_texts`` and ``mean`` and ``std`` come from ``standardizer``.
    """
    hashed, dense = features.extract_texts(texts, dim)
    standardized = scipy.sparse.csr_matrix((dense - mean) / std)
    return scipy.sparse.hstack([hashed, standardized], format="csr", dtype=np.float64)
