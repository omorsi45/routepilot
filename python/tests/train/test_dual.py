"""Tests for the lambda tuning of the dual mode."""

import pandas as pd
import pytest

from routepilot_train.train import dual


def test_tune_lambda_bisection():
    calls: list[float] = []

    def step(lam: float) -> float:
        calls.append(lam)
        return 0.8 if lam <= 37.5 else 0.5  # monotone non-increasing, one jump

    best = dual.tune_lambda(step, 0.7)
    n_calls = len(calls)

    assert best <= 37.5
    assert best == pytest.approx(37.5, abs=1e-4)  # 1e6 / 2**40 is about 9e-7
    assert step(best) >= 0.7  # the returned lambda meets the target
    assert calls[0] == 0.0 and calls[1] == 1e6  # the ends are probed first
    assert n_calls == 2 + 40  # and then exactly `iters` bisection steps

    smooth = dual.tune_lambda(lambda lam: 1.0 / (1.0 + lam / 50.0), 0.5)  # 0.5 at lambda = 50
    assert smooth == pytest.approx(50.0, abs=1e-4) and smooth <= 50.0

    coarse = dual.tune_lambda(lambda lam: 1.0 if lam < 10 else 0.0, 0.5, lo=0.0, hi=100.0, iters=3)
    assert coarse == 0.0  # three halvings of [0, 100] never reach below 12.5: only lo is feasible
    finer = dual.tune_lambda(lambda lam: 1.0 if lam < 10 else 0.0, 0.5, lo=0.0, hi=100.0, iters=4)
    assert finer == 6.25


def test_tune_lambda_returns_hi_when_every_lambda_meets_the_target():
    seen: list[float] = []

    def always(lam: float) -> float:
        seen.append(lam)
        return 0.9

    assert dual.tune_lambda(always, 0.5, hi=123.0) == 123.0
    assert seen == [0.0, 123.0]  # no bisection is needed


def test_tune_lambda_a_target_met_exactly_counts():
    assert dual.tune_lambda(lambda lam: 0.6 if lam <= 4 else 0.0, 0.6, hi=8.0, iters=20) == 4.0


def test_tune_lambda_unreachable_raises():
    with pytest.raises(ValueError, match="lambda 0"):
        dual.tune_lambda(lambda lam: 0.4, 0.5)
    with pytest.raises(ValueError, match="0.4"):
        dual.tune_lambda(lambda lam: 0.4, 0.5, lo=2.0)  # the message gives the success it saw


def test_tune_lambda_rejects_bad_arguments_and_values():
    with pytest.raises(ValueError, match="lo"):
        dual.tune_lambda(lambda lam: 1.0, 0.5, lo=5.0, hi=5.0)
    with pytest.raises(ValueError, match="iters"):
        dual.tune_lambda(lambda lam: 1.0, 0.5, iters=0)
    with pytest.raises(ValueError, match="finite"):
        dual.tune_lambda(lambda lam: float("nan"), 0.5)
    with pytest.raises(ValueError, match="finite"):
        dual.tune_lambda(lambda lam: 1.0, float("inf"))


def frame(splits: list[str], success: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": [f"x/{i}" for i in range(len(splits))], "split": splits, "success": success}
    )


def test_realized_success_is_the_calibration_success_rate():
    rows = frame(["calib"] * 4, [True, False, True, True])
    assert dual.realized_success(rows) == 0.75


@pytest.mark.parametrize("bad", ["test", "train", "heldout"])
def test_realized_success_refuses_rows_outside_the_calibration_split(bad: str):
    rows = frame(["calib", "calib", bad], [True, False, True])
    with pytest.raises(ValueError, match="splits not allowed"):
        dual.realized_success(rows)
    with pytest.raises(ValueError, match="split"):
        dual.realized_success(rows.drop(columns="split"))
    with pytest.raises(ValueError, match="no rows"):
        dual.realized_success(rows.iloc[:0])


def test_calibration_target_is_best_single_minus_the_margin():
    rows = frame(["calib"] * 10, [True] * 5 + [False] * 5)
    assert dual.calibration_target(rows) == pytest.approx(0.49)
    assert dual.calibration_target(rows, margin=0.1) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="splits not allowed"):
        dual.calibration_target(frame(["calib", "test"], [True, True]))
