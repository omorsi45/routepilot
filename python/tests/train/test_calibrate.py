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
