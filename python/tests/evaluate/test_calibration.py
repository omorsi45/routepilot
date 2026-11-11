import math

import numpy as np
import pytest

from routepilot_train.evaluate import calibration


def test_ece_perfect_and_bad():
    # 100 items at p=0.2 with 20 positives, 100 at p=0.8 with 80: every 20-item bin is exact
    p = np.array([0.2] * 100 + [0.8] * 100)
    y = np.array([i % 5 == 0 for i in range(100)] + [i % 5 != 0 for i in range(100)])
    assert calibration.ece(p, y, bins=10) == pytest.approx(0.0, abs=1e-12)
    assert calibration.ece(p, y, bins=10, scheme="width") == pytest.approx(0.0, abs=1e-12)

    # a constant 0.9 forecast on data that is right half the time is off by 0.4
    p = np.full(300, 0.9)
    y = np.arange(300) % 2 == 0
    assert calibration.ece(p, y, bins=15) == pytest.approx(0.4)
    assert calibration.ece(p, y, bins=15, scheme="width") == pytest.approx(0.4)


def test_ece_width_scheme_uses_equal_width_bins():
    p = np.array([0.05, 0.05, 0.95, 0.95])
    y = np.array([0, 0, 1, 1])
    assert calibration.ece(p, y, bins=10, scheme="width") == pytest.approx(0.05)
    # p == 1.0 belongs to the last bin, not a bin of its own
    assert calibration.ece(np.array([1.0, 1.0]), np.array([1, 1]), bins=10, scheme="width") == 0.0


def test_brier_and_log_loss_known_values():
    p = np.array([0.9, 0.1, 0.5, 0.5])
    y = np.array([1, 0, 1, 0])
    assert calibration.brier(p, y) == pytest.approx((0.01 + 0.01 + 0.25 + 0.25) / 4)
    want = -(2 * math.log(0.9) + 2 * math.log(0.5)) / 4
    assert calibration.log_loss(p, y) == pytest.approx(want)
    # certain and wrong is clipped at eps instead of infinite
    assert calibration.log_loss(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(
        -math.log(1e-6)
    )
    assert calibration.log_loss(np.array([0.0]), np.array([1]), eps=1e-3) == pytest.approx(
        -math.log(1e-3)
    )


def test_reliability_table_mass_bins_hold_equal_counts():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.uniform(0, 1, 1000) < p
    table = calibration.reliability_table(p, y, bins=15)
    assert list(table.columns) == ["bin", "n", "mean_p", "mean_y"]
    assert len(table) == 15 and table["n"].sum() == 1000
    assert table["n"].max() - table["n"].min() <= 1
    assert table["mean_p"].is_monotonic_increasing
    # a calibrated forecast: each bin's rate is close to its mean forecast
    assert (table["mean_y"] - table["mean_p"]).abs().max() < 0.15


def test_reliability_table_width_bins_drop_empty_bins():
    p = np.array([0.05, 0.06, 0.95])
    y = np.array([0, 1, 1])
    table = calibration.reliability_table(p, y, bins=10, scheme="width")
    assert list(table["bin"]) == [0, 9]
    assert list(table["n"]) == [2, 1]
    assert table["mean_y"].tolist() == [0.5, 1.0]


def test_ece_is_the_count_weighted_gap_of_the_table():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 500)
    y = rng.uniform(0, 1, 500) < 0.5
    table = calibration.reliability_table(p, y, bins=15)
    want = float((table["n"] * (table["mean_y"] - table["mean_p"]).abs()).sum() / table["n"].sum())
    assert calibration.ece(p, y, bins=15) == pytest.approx(want)


@pytest.mark.parametrize(
    ("p", "y", "message"),
    [
        (np.array([0.5, 1.2]), np.array([1, 0]), "between 0 and 1"),
        (np.array([-0.1, 0.5]), np.array([1, 0]), "between 0 and 1"),
        (np.array([np.nan, 0.5]), np.array([1, 0]), "between 0 and 1"),
        (np.array([0.5, 0.5]), np.array([1, 0, 1]), "same length"),
        (np.array([0.5, 0.5]), np.array([1, 2]), "0 or 1"),
        (np.array([]), np.array([]), "empty"),
    ],
)
def test_calibration_rejects_bad_input(p, y, message):
    for fn in (calibration.ece, calibration.brier, calibration.log_loss):
        with pytest.raises(ValueError, match=message):
            fn(p, y)


def test_ece_rejects_bad_scheme_and_bins():
    p, y = np.array([0.5, 0.5]), np.array([1, 0])
    with pytest.raises(ValueError, match="scheme"):
        calibration.ece(p, y, scheme="quantile")
    with pytest.raises(ValueError, match="bins"):
        calibration.ece(p, y, bins=0)
