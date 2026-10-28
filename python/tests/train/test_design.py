"""Tests for the feature-set helpers of the design matrix (the ablation's restricted fits)."""

import numpy as np
import pytest
import scipy.sparse

from routepilot_train import features
from routepilot_train.train import design

DIM = 5
MEAN = np.arange(12, dtype=np.float64)
STD = np.linspace(1.0, 2.0, 12)


def inputs() -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(0)
    hashed = scipy.sparse.csr_matrix(rng.random((3, DIM)) * (rng.random((3, DIM)) < 0.6))
    dense = rng.random((3, features.NUM_DENSE)) * 10
    return hashed, dense


def test_feature_sets_are_both_hashed_and_dense():
    assert design.FEATURE_SETS == ("both", "hashed", "dense")


def test_select_features_both_is_the_full_design_matrix():
    hashed, dense = inputs()

    got = design.select_features(hashed, dense, MEAN, STD, "both")

    expected = design.stack(hashed, dense, MEAN, STD)
    assert got.shape == (3, DIM + features.NUM_DENSE)
    assert (got != expected).nnz == 0


def test_select_features_hashed_keeps_only_the_hashed_block():
    hashed, dense = inputs()

    got = design.select_features(hashed, dense, MEAN, STD, "hashed")

    assert scipy.sparse.issparse(got) and got.shape == (3, DIM)
    assert (got != hashed).nnz == 0


def test_select_features_dense_keeps_only_the_standardized_dense_block():
    hashed, dense = inputs()

    got = design.select_features(hashed, dense, MEAN, STD, "dense")

    assert scipy.sparse.issparse(got) and got.shape == (3, features.NUM_DENSE)
    np.testing.assert_array_equal(got.toarray(), (dense - MEAN) / STD)


def test_expand_weights_puts_the_fitted_block_where_the_full_vector_reads_it():
    coef_hashed = np.arange(1.0, DIM + 1)
    coef_dense = -np.arange(1.0, features.NUM_DENSE + 1)
    coef_both = np.concatenate([coef_hashed, coef_dense])

    np.testing.assert_array_equal(design.expand_weights(coef_both, DIM, "both"), coef_both)
    only_hashed = design.expand_weights(coef_hashed, DIM, "hashed")
    np.testing.assert_array_equal(only_hashed[:DIM], coef_hashed)
    assert only_hashed.shape == (DIM + features.NUM_DENSE,) and not only_hashed[DIM:].any()
    only_dense = design.expand_weights(coef_dense, DIM, "dense")
    np.testing.assert_array_equal(only_dense[DIM:], coef_dense)
    assert only_dense.shape == (DIM + features.NUM_DENSE,) and not only_dense[:DIM].any()


def test_expanded_weights_score_like_the_restricted_model():
    hashed, dense = inputs()
    coef = np.linspace(-1, 1, features.NUM_DENSE)

    restricted = design.select_features(hashed, dense, MEAN, STD, "dense") @ coef
    full = design.stack(hashed, dense, MEAN, STD) @ design.expand_weights(coef, DIM, "dense")

    np.testing.assert_allclose(full, restricted, rtol=0, atol=1e-12)


def test_unknown_feature_sets_and_wrong_lengths_are_refused():
    hashed, dense = inputs()
    with pytest.raises(ValueError, match="feature set"):
        design.select_features(hashed, dense, MEAN, STD, "sparse")
    with pytest.raises(ValueError, match="feature set"):
        design.expand_weights(np.zeros(DIM), DIM, "sparse")
    with pytest.raises(ValueError, match="expects"):
        design.expand_weights(np.zeros(DIM + 1), DIM, "hashed")
    with pytest.raises(ValueError, match="expects"):
        design.expand_weights(np.zeros(DIM), DIM, "dense")
    with pytest.raises(ValueError, match="expects"):
        design.expand_weights(np.zeros(DIM), DIM, "both")
