"""Out-of-distribution detector: distance from a request to the nearest k-means centroid.

K-means runs on the L2-normalized hashed features of the train split. Each centroid is
cut to its ``TOP`` largest-magnitude entries and renormalized, because the router stores
and scans those sparse centroids, and the score is computed against the stored values
so Python and Go agree. A request scores ``1 - max_j cos(x, c_j)``; the threshold is the
``Q`` quantile of the calibration split's scores, so about 1% of in-distribution
requests are flagged. ``H_train`` must be train-split rows and ``H_calib`` calibration-split
rows; the pipeline enforces this with ``require_splits`` before calling ``fit_ood``.
"""

import numpy as np
import scipy.sparse
from sklearn.cluster import MiniBatchKMeans

K = 32
TOP = 512
Q = 0.99


def _truncate(centroid: np.ndarray, top: int = TOP) -> dict:
    """Keep the ``top`` largest-magnitude nonzero entries (ties go to the lower index).

    Returns ``{"idx": int32 ascending, "val": float32}`` with the kept values scaled to
    unit L2 norm before the float32 cast. An all-zero centroid gives empty arrays.
    """
    nonzero = np.flatnonzero(centroid)
    if nonzero.size > top:
        order = np.lexsort((nonzero, -np.abs(centroid[nonzero])))[:top]
        nonzero = np.sort(nonzero[order])
    val = centroid[nonzero]
    norm = np.linalg.norm(val)
    if norm > 0:
        val = val / norm
    return {"idx": nonzero.astype(np.int32), "val": val.astype(np.float32)}


def _centroid_matrix(centroids: list[dict], dim: int) -> scipy.sparse.csr_matrix:
    indptr, indices, data = [0], [], []
    for centroid in centroids:
        indices.append(np.asarray(centroid["idx"], dtype=np.int64))
        data.append(np.asarray(centroid["val"], dtype=np.float64))
        indptr.append(indptr[-1] + len(centroid["idx"]))
    return scipy.sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=(len(centroids), dim),
    )


def ood_score(params: dict, H) -> np.ndarray:
    """``1 - max_j cos(x, c_j)`` for each row ``x`` of ``H`` against the stored centroids.

    ``H`` holds the hashed features only (CSR, one row per request). A zero row, or a
    zero centroid, has cosine 0, so an empty request scores 1.0.
    """
    H = scipy.sparse.csr_matrix(H, dtype=np.float64)
    C = _centroid_matrix(params["centroids"], H.shape[1])
    x_norm = np.sqrt(np.asarray(H.multiply(H).sum(axis=1)).ravel())
    c_norm = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    dots = (H @ C.T).toarray()
    denom = np.outer(x_norm, c_norm)
    cos = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
    return 1.0 - cos.max(axis=1)


def fit_ood(H_train, H_calib, seed: int = 0) -> dict:
    """Fit the detector: k-means on ``H_train``, threshold from ``H_calib``.

    Both matrices are the hashed parts only, already L2-normalized, with the same number
    of columns. ``seed`` is the k-means ``random_state``. Returns ``{"k", "threshold",
    "centroids": [{"idx": int32 array, "val": float32 array}, ...]}``.
    """
    H_train = scipy.sparse.csr_matrix(H_train, dtype=np.float64)
    H_calib = scipy.sparse.csr_matrix(H_calib, dtype=np.float64)
    if H_train.shape[1] != H_calib.shape[1]:
        raise ValueError(
            f"H_train and H_calib need the same columns, got {H_train.shape[1]} and "
            f"{H_calib.shape[1]}"
        )
    if H_train.shape[0] < K:
        raise ValueError(f"need at least {K} train rows for {K} clusters, got {H_train.shape[0]}")
    if H_calib.shape[0] == 0:
        raise ValueError("H_calib is empty")
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3, batch_size=1024)
    kmeans.fit(H_train)
    params = {
        "k": K,
        "threshold": 0.0,
        "centroids": [_truncate(centroid) for centroid in kmeans.cluster_centers_],
    }
    params["threshold"] = float(np.quantile(ood_score(params, H_calib), Q))
    return params
