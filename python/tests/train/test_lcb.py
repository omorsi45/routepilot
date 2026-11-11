"""Tests for the Clopper-Pearson lower confidence bound tables."""

import json

import numpy as np
import pytest
from scipy.stats import binom

from routepilot_train.train import lcb


def test_constants():
    assert (lcb.BINS, lcb.MIN_BIN, lcb.ALPHA) == (15, 30, 0.05)


def test_clopper_pearson_known():
    assert lcb.clopper_pearson_lower(30, 30) == pytest.approx(0.05 ** (1 / 30), abs=1e-9)
    assert lcb.clopper_pearson_lower(0, 30) == 0
    assert lcb.clopper_pearson_lower(0, 0) == 0


@pytest.mark.parametrize(("k", "n"), [(1, 10), (5, 10), (9, 10), (37, 120), (400, 1000)])
def test_clopper_pearson_is_the_exact_one_sided_bound(k: int, n: int):
    # The bound L solves P(Binomial(n, L) >= k) = alpha.
    lower = lcb.clopper_pearson_lower(k, n)
    assert 0.0 < lower < k / n
    assert binom.sf(k - 1, n, lower) == pytest.approx(0.05, abs=1e-9)


def test_clopper_pearson_rejects_impossible_counts():
    with pytest.raises(ValueError, match="0 <= k <= n"):
        lcb.clopper_pearson_lower(11, 10)
    with pytest.raises(ValueError, match="0 <= k <= n"):
        lcb.clopper_pearson_lower(-1, 10)


def test_lcb_monotone_and_merged():
    # 300 samples split into 15 bins of 20: every bin is under 30. The leftmost small
    # bin merges right, pair by pair, and the last one merges left.
    rng = np.random.default_rng(0)
    p = np.sort(rng.random(300))
    y = (rng.random(300) < 0.5).astype(np.int64)  # independent of p: raw bounds fluctuate
    table = lcb.fit_lcb(p, y)
    assert table["n"] == [40, 40, 40, 40, 40, 40, 60]
    assert min(table["n"]) >= lcb.MIN_BIN
    assert sum(table["n"]) == 300
    assert len(table["upper"]) == len(table["lower"]) == len(table["n"])
    assert all(b >= a for a, b in zip(table["lower"], table["lower"][1:], strict=False))
    assert table["upper"][:-1] == [float(p[i]) for i in (39, 79, 119, 159, 199, 239)]
    assert table["upper"][-1] == 1.0


def test_lcb_running_maximum():
    # Bins of 30 with the success rate falling from 1 to 0 as p rises: the raw bounds
    # fall, and the running maximum keeps the table non-decreasing.
    p = np.linspace(0.0, 1.0, 450)
    y = (np.arange(450) < 150).astype(np.int64)
    table = lcb.fit_lcb(p, y)
    assert table["n"] == [30] * 15
    top = lcb.clopper_pearson_lower(30, 30)
    np.testing.assert_allclose(table["lower"], [top] * 15)


def test_lcb_bins_are_equal_mass_when_none_is_small():
    p = np.linspace(0.0, 1.0, 1500)
    y = (np.arange(1500) % 2).astype(np.int64)
    table = lcb.fit_lcb(p, y)
    assert table["n"] == [100] * 15


def test_lcb_orders_by_p_not_by_input_order():
    rng = np.random.default_rng(1)
    p = rng.random(600)
    y = (rng.random(600) < p).astype(np.int64)
    shuffled = rng.permutation(600)
    assert lcb.fit_lcb(p, y) == lcb.fit_lcb(p[shuffled], y[shuffled])


def test_lcb_under_thirty_samples_is_one_bin():
    p = np.linspace(0.1, 0.9, 20)
    y = np.array([0, 1] * 10)
    table = lcb.fit_lcb(p, y)
    assert table["n"] == [20]
    assert table["upper"] == [1.0]
    assert table["lower"] == [lcb.clopper_pearson_lower(10, 20)]


def test_lcb_pooled_fields_and_json():
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(np.int64)
    table = lcb.fit_lcb(p, y)
    assert table["pooled_rate"] == pytest.approx(float(y.mean()))
    assert table["pooled_lcb"] == lcb.clopper_pearson_lower(int(y.sum()), 500)
    assert set(table) == {"upper", "lower", "n", "pooled_rate", "pooled_lcb"}
    assert json.loads(json.dumps(table)) == table


def test_lookup_edges():
    table = {"upper": [0.2, 0.5, 1.0], "lower": [0.1, 0.3, 0.6], "n": [40, 40, 40]}
    assert lcb.lookup_lcb(table, 0.0) == 0.1
    assert lcb.lookup_lcb(table, -1.0) == 0.1
    assert lcb.lookup_lcb(table, 0.2) == 0.1  # p <= upper picks the first bin
    assert lcb.lookup_lcb(table, 0.2000001) == 0.3
    assert lcb.lookup_lcb(table, 0.5) == 0.3
    assert lcb.lookup_lcb(table, 1.0) == 0.6
    assert lcb.lookup_lcb(table, 1.5) == 0.6  # beyond the last bound: the last bin


def test_lookup_finds_every_fitted_p():
    rng = np.random.default_rng(3)
    p = rng.random(400)
    y = (rng.random(400) < p).astype(np.int64)
    table = lcb.fit_lcb(p, y)
    looked_up = [lcb.lookup_lcb(table, float(v)) for v in np.sort(p)]
    assert all(b >= a for a, b in zip(looked_up, looked_up[1:], strict=False))
    assert set(looked_up) <= set(table["lower"])


def test_fit_lcb_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        lcb.fit_lcb(np.zeros(0), np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError, match="same length"):
        lcb.fit_lcb(np.zeros(5), np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="0 or 1"):
        lcb.fit_lcb(np.full(5, 0.5), np.full(5, 3))
    with pytest.raises(ValueError, match="between 0 and 1"):
        lcb.fit_lcb(np.array([0.5, 1.5]), np.array([0, 1]))
