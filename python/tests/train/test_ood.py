"""Tests for the k-means out-of-distribution detector."""

import numpy as np
import pytest
import scipy.sparse

from routepilot_train.train import ood

DIM = 8192
TOPICS = 8
VOCAB = 700  # features per topic; topic t owns [t * VOCAB, (t + 1) * VOCAB)
NNZ = 100
OOD_LOW = 6000  # far rows draw from [6000, 8192), disjoint from every topic


def normalized_rows(columns: list[np.ndarray], rng: np.random.Generator) -> scipy.sparse.csr_matrix:
    indptr, indices, data = [0], [], []
    for cols in columns:
        order = np.argsort(cols)
        weights = rng.random(cols.size) + 0.5
        weights /= np.linalg.norm(weights)
        indices.append(cols[order])
        data.append(weights[order])
        indptr.append(indptr[-1] + cols.size)
    return scipy.sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=(len(columns), DIM),
    )


def topic_rows(n: int, seed: int) -> scipy.sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    columns = [
        rng.integers(0, TOPICS) * VOCAB + rng.choice(VOCAB, size=NNZ, replace=False)
        for _ in range(n)
    ]
    return normalized_rows(columns, rng)


def far_rows(n: int, seed: int) -> scipy.sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    columns = [OOD_LOW + rng.choice(DIM - OOD_LOW, size=NNZ, replace=False) for _ in range(n)]
    return normalized_rows(columns, rng)


@pytest.fixture(scope="module")
def data():
    train, calib, fresh = topic_rows(1500, 0), topic_rows(600, 1), topic_rows(600, 2)
    return {
        "train": train,
        "calib": calib,
        "fresh": fresh,
        "far": far_rows(200, 3),
        "params": ood.fit_ood(train, calib),
    }


def test_constants():
    assert (ood.K, ood.TOP, ood.Q) == (32, 512, 0.99)


def test_ood_separates_far_cluster(data):
    params = data["params"]
    near = ood.ood_score(params, data["fresh"])
    far = ood.ood_score(params, data["far"])
    assert far.min() == pytest.approx(1.0)  # no feature in common with any centroid
    assert near.max() < far.min()
    assert (near > params["threshold"]).mean() <= 0.05
    assert (far > params["threshold"]).all()


def test_zero_vector_scores_one(data):
    zeros = scipy.sparse.csr_matrix((3, DIM))
    np.testing.assert_array_equal(ood.ood_score(data["params"], zeros), [1.0, 1.0, 1.0])
    mixed = scipy.sparse.vstack([zeros[:1], data["fresh"][:2]], format="csr")
    scores = ood.ood_score(data["params"], mixed)
    assert scores[0] == 1.0
    assert (scores[1:] < 1.0).all()


def test_threshold_is_calib_quantile(data):
    params = data["params"]
    calib_scores = ood.ood_score(params, data["calib"])
    assert params["threshold"] == pytest.approx(float(np.quantile(calib_scores, 0.99)), abs=1e-12)
    # The threshold follows the calibration rows, not the train rows: far calibration
    # rows push it to 1.
    shifted = ood.fit_ood(data["train"], data["far"])
    assert shifted["threshold"] == pytest.approx(1.0)
    for a, b in zip(shifted["centroids"], params["centroids"], strict=True):
        np.testing.assert_array_equal(a["idx"], b["idx"])
        np.testing.assert_array_equal(a["val"], b["val"])


def test_centroids_truncated_and_unit(data):
    params = data["params"]
    assert params["k"] == 32
    assert len(params["centroids"]) == 32
    sizes = []
    for centroid in params["centroids"]:
        idx, val = centroid["idx"], centroid["val"]
        assert idx.dtype == np.int32 and val.dtype == np.float32
        assert idx.shape == val.shape
        assert (np.diff(idx) > 0).all()  # ascending and unique
        assert idx.min() >= 0 and idx.max() < DIM
        assert np.linalg.norm(val.astype(np.float64)) == pytest.approx(1.0, abs=1e-6)
        sizes.append(idx.size)
    assert max(sizes) == ood.TOP  # the topics are wider than 512 features, so they are cut
    assert all(size <= ood.TOP for size in sizes)


def test_truncate_keeps_largest_magnitudes_with_ties_to_lower_index():
    dense = np.array([0.1, -0.9, 0.5, 0.5, 0.5, 0.0, 0.2, 0.0, 0.0, 0.5])
    kept = ood._truncate(dense, top=3)
    np.testing.assert_array_equal(kept["idx"], [1, 2, 3])
    want = np.array([-0.9, 0.5, 0.5]) / np.linalg.norm([-0.9, 0.5, 0.5])
    np.testing.assert_allclose(kept["val"], want, atol=1e-7)
    assert kept["idx"].dtype == np.int32 and kept["val"].dtype == np.float32


def test_truncate_drops_zero_entries_and_handles_a_zero_centroid():
    kept = ood._truncate(np.array([0.0, 3.0, 0.0, 4.0]), top=512)
    np.testing.assert_array_equal(kept["idx"], [1, 3])
    np.testing.assert_allclose(kept["val"], [0.6, 0.8], atol=1e-7)
    empty = ood._truncate(np.zeros(5), top=512)
    assert empty["idx"].size == 0 and empty["val"].size == 0


def test_score_is_one_minus_best_cosine_to_the_stored_centroids():
    params = {
        "k": 2,
        "threshold": 0.5,
        "centroids": [
            {"idx": np.array([0, 1], dtype=np.int32), "val": np.array([0.6, 0.8], np.float32)},
            {"idx": np.array([2], dtype=np.int32), "val": np.array([1.0], np.float32)},
        ],
    }
    rows = np.zeros((5, 4))
    rows[0, :2] = [0.6, 0.8]  # equals centroid 0
    rows[1, 0] = 1.0  # cos 0.6 with centroid 0
    rows[2, 2] = 2.0  # not unit length: cosine ignores the scale
    rows[3, :2] = [-0.6, -0.8]  # opposite of centroid 0, orthogonal to centroid 1
    rows[4, 3] = 1.0  # shares nothing with either centroid
    scores = ood.ood_score(params, scipy.sparse.csr_matrix(rows))
    np.testing.assert_allclose(scores, [0.0, 0.4, 0.0, 1.0, 1.0], atol=1e-6)


def test_fit_is_deterministic(data):
    again = ood.fit_ood(data["train"], data["calib"])
    params = data["params"]
    assert again["threshold"] == params["threshold"]
    for a, b in zip(again["centroids"], params["centroids"], strict=True):
        np.testing.assert_array_equal(a["idx"], b["idx"])
        np.testing.assert_array_equal(a["val"], b["val"])


def test_fit_ood_rejects_bad_input(data):
    with pytest.raises(ValueError, match="at least 32"):
        ood.fit_ood(data["train"][:31], data["calib"])
    with pytest.raises(ValueError, match="empty"):
        ood.fit_ood(data["train"], scipy.sparse.csr_matrix((0, DIM)))
    with pytest.raises(ValueError, match="columns"):
        ood.fit_ood(data["train"], data["calib"][:, : DIM - 1])
