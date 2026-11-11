"""Tests for the score calibrators (isotonic above 1000 samples, Platt below)."""

import json
import math

import numpy as np
import pytest

from routepilot_train.train import calibrate
from routepilot_train.train.quality import sigmoid


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def synthetic_scores(n: int, true_slope: float = 0.5, seed: int = 0):
    """Overconfident scores: the true success logit is ``true_slope * s``."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 3.0, size=n)
    y = (rng.random(n) < sigmoid(true_slope * s)).astype(np.int64)
    return s, y


def test_isotonic_vs_platt_switch():
    s, y = synthetic_scores(calibrate.ISOTONIC_MIN)
    assert calibrate.ISOTONIC_MIN == 1000
    below = calibrate.fit_calibrator(s[:999], y[:999])
    at = calibrate.fit_calibrator(s, y)
    assert below["type"] == "platt"
    assert set(below) == {"type", "a", "b"}
    assert at["type"] == "isotonic"
    assert set(at) == {"type", "x", "y"}


def test_platt_shrinks_an_overconfident_score():
    s, y = synthetic_scores(999, true_slope=0.5)
    cal = calibrate.fit_calibrator(s, y)
    assert 0.3 < cal["a"] < 0.7
    assert abs(cal["b"]) < 0.3
    p = calibrate.apply_calibrator(cal, s)
    np.testing.assert_allclose(p, sigmoid(cal["a"] * s + cal["b"]))


def test_platt_single_class_predicts_the_clipped_rate():
    s = np.linspace(-2, 2, 50)
    cal = calibrate.fit_calibrator(s, np.ones(50, dtype=np.int64))
    assert cal["a"] == 0.0
    np.testing.assert_allclose(calibrate.apply_calibrator(cal, s), 0.999)
    cal = calibrate.fit_calibrator(s, np.zeros(50, dtype=np.int64))
    np.testing.assert_allclose(calibrate.apply_calibrator(cal, s), 0.001)


def test_isotonic_learns_a_step_and_is_json_serializable():
    s = np.linspace(-6, 6, 1200)
    y = (s > 0).astype(np.int64)
    cal = calibrate.fit_calibrator(s, y)
    assert cal["type"] == "isotonic"
    assert len(cal["x"]) == len(cal["y"]) >= 2
    assert all(isinstance(v, float) for v in cal["x"] + cal["y"])
    assert cal["x"] == sorted(cal["x"])
    assert json.loads(json.dumps(cal)) == cal
    p = calibrate.apply_calibrator(cal, np.array([-5.0, 5.0]))
    np.testing.assert_allclose(p, [0.0, 1.0])


def test_isotonic_output_is_monotone_and_bounded():
    s, y = synthetic_scores(3000)
    cal = calibrate.fit_calibrator(s, y)
    grid = np.linspace(-15.0, 15.0, 400)
    p = calibrate.apply_calibrator(cal, grid)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert (np.diff(p) >= -1e-12).all()


def test_apply_isotonic_interpolates_and_clamps():
    cal = {"type": "isotonic", "x": [0.2, 0.6, 0.8], "y": [0.1, 0.5, 0.9]}
    s = np.array([logit(0.05), logit(0.2), logit(0.4), logit(0.7), logit(0.8), logit(0.95)])
    p = calibrate.apply_calibrator(cal, s)
    np.testing.assert_allclose(p, [0.1, 0.1, 0.3, 0.7, 0.9, 0.9])


def test_apply_platt_is_a_sigmoid_of_the_affine_score():
    cal = {"type": "platt", "a": 2.0, "b": -1.0}
    s = np.array([-1.0, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(calibrate.apply_calibrator(cal, s), sigmoid(2.0 * s - 1.0))


def test_apply_calibrator_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown calibrator type"):
        calibrate.apply_calibrator({"type": "spline"}, np.zeros(3))


def test_fit_calibrator_rejects_bad_input():
    s = np.zeros(10)
    with pytest.raises(ValueError, match="0 or 1"):
        calibrate.fit_calibrator(s, np.full(10, 2))
    with pytest.raises(ValueError, match="same length"):
        calibrate.fit_calibrator(s, np.zeros(9, dtype=np.int64))
    with pytest.raises(ValueError, match="empty"):
        calibrate.fit_calibrator(np.zeros(0), np.zeros(0, dtype=np.int64))


# --- the calibration study's methods: platt, isotonic, histogram, switch ----------------------


def test_fit_calibrator_methods_force_the_type_and_switch_is_the_default():
    s, y = synthetic_scores(300)
    assert calibrate.fit_calibrator(s, y)["type"] == "platt"
    assert calibrate.fit_calibrator(s, y, method="switch")["type"] == "platt"
    assert calibrate.fit_calibrator(s, y, method="platt")["type"] == "platt"
    assert calibrate.fit_calibrator(s, y, method="isotonic")["type"] == "isotonic"  # below 1000
    assert calibrate.fit_calibrator(s, y, method="histogram")["type"] == "histogram"
    s, y = synthetic_scores(calibrate.ISOTONIC_MIN)
    assert calibrate.fit_calibrator(s, y, method="switch")["type"] == "isotonic"
    assert calibrate.fit_calibrator(s, y, method="platt")["type"] == "platt"  # above 1000
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate.fit_calibrator(s, y, method="spline")


def test_methods_are_the_documented_four():
    assert calibrate.METHODS == ("isotonic", "platt", "histogram", "switch")


def test_histogram_bins_have_equal_mass_and_each_holds_its_empirical_rate():
    s = np.arange(150, dtype=np.float64)  # 150 distinct scores: 15 bins of 10
    y = (s % 3 == 0).astype(np.int64)
    cal = calibrate.fit_calibrator(s, y, method="histogram")

    assert cal["type"] == "histogram" and set(cal) == {"type", "edges", "rate"}
    assert len(cal["rate"]) == calibrate.HISTOGRAM_BINS == 15
    assert len(cal["edges"]) == 14 and cal["edges"] == sorted(cal["edges"])
    groups = np.array_split(np.arange(150), 15)
    np.testing.assert_allclose(cal["rate"], [y[g].mean() for g in groups])
    # a score is mapped to the rate of its bin, a bin starts at its first score
    p = calibrate.apply_calibrator(cal, np.array([0.0, 9.0, 10.0, 149.0]))
    np.testing.assert_allclose(p, [y[:10].mean(), y[:10].mean(), y[10:20].mean(), y[140:].mean()])
    assert json.loads(json.dumps(cal)) == cal


def test_histogram_maps_scores_outside_the_fitted_range_to_the_end_bins():
    s = np.linspace(-3.0, 3.0, 90)
    y = (s > 0).astype(np.int64)
    cal = calibrate.fit_calibrator(s, y, method="histogram")

    low, high = calibrate.apply_calibrator(cal, np.array([-1e9, 1e9]))
    assert low == cal["rate"][0] == 0.0
    assert high == cal["rate"][-1] == 1.0
    assert np.isfinite(calibrate.apply_calibrator(cal, np.array([-np.inf, np.inf]))).all()


def test_histogram_with_ties_and_few_samples_keeps_every_bin_populated():
    # all scores equal: one bin holding the pooled rate
    cal = calibrate.fit_calibrator(np.zeros(40), np.array([1] * 10 + [0] * 30), method="histogram")
    assert cal["edges"] == [] and cal["rate"] == [0.25]
    # fewer samples than bins: at most one bin per sample, none empty
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cal = calibrate.fit_calibrator(s, np.array([0, 1, 0, 1, 1]), method="histogram")
    assert len(cal["rate"]) == len(cal["edges"]) + 1 <= 5
    np.testing.assert_allclose(calibrate.apply_calibrator(cal, s), [0, 1, 0, 1, 1])
    # a block of tied scores straddling a bin boundary lands in one bin
    s = np.repeat([0.0, 1.0], [100, 50])
    cal = calibrate.fit_calibrator(s, (np.arange(150) % 2).astype(np.int64), method="histogram")
    assert len(cal["rate"]) == len(cal["edges"]) + 1 == 2
    assert all(0.0 <= r <= 1.0 for r in cal["rate"])


def test_histogram_is_not_a_calibrator_the_artifact_can_hold():
    from routepilot_train.export import artifact

    cal = {"type": "histogram", "edges": [0.0], "rate": [0.2, 0.8]}
    with pytest.raises(ValueError, match="unknown calibrator type"):
        artifact._check_calibrator("m", cal)
