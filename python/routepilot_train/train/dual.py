"""Tuning the dual mode's lambda on the calibration split.

The dual mode picks the safe model with the highest ``P - lambda * expected cost``, so a larger
lambda trades quality for cost. ``tune_lambda`` finds the largest lambda whose realized success
on the calibration split still reaches a target, by bisection on a callable that replays the
policy. The guards here take replay frames and assert their ``split`` column
(``require_splits``), so a tuning step cannot read a test row by accident.
"""

import math
from collections.abc import Callable

import pandas as pd

from routepilot_train.data.splits import require_splits

# The dual mode may lose this much success against Best Single on the calibration split.
QUALITY_MARGIN = 0.01
LAMBDA_MAX = 1e6
BISECTION_STEPS = 40
TUNING_SPLITS = {"calib"}


def tune_lambda(
    run_calib: Callable[[float], float],
    target: float,
    lo: float = 0.0,
    hi: float = LAMBDA_MAX,
    iters: int = BISECTION_STEPS,
) -> float:
    """The largest lambda in ``[lo, hi]`` whose realized calibration success reaches ``target``.

    ``run_calib(lam)`` returns the success rate of the policy replayed on the calibration split
    at ``lam``; it is assumed non-increasing in ``lam``. ``lo`` and ``hi`` are probed first; if
    ``lo`` already misses the target no lambda can meet it and ``ValueError`` says so; if
    ``hi`` meets it, ``hi`` is returned. Otherwise ``iters`` halvings keep ``lo`` on the side
    that meets the target and ``hi`` on the side that misses it, and ``lo`` is returned, so the
    result always meets the target (to within ``(hi - lo) / 2**iters`` of the largest
    lambda that does, if the function is monotone).
    """
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"lo must be finite and below hi, got lo={lo} hi={hi}")
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    if not math.isfinite(target):
        raise ValueError(f"target must be finite, got {target}")

    def success_at(lam: float) -> float:
        value = float(run_calib(lam))
        if not math.isfinite(value):
            raise ValueError(f"run_calib({lam}) returned {value}, want a finite success rate")
        return value

    at_lo = success_at(lo)
    if at_lo < target:
        raise ValueError(
            f"the success {at_lo:.6g} at lambda {lo} is below the target {target:.6g}: "
            "no lambda in range meets it"
        )
    if success_at(hi) >= target:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if success_at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def realized_success(frame: pd.DataFrame) -> float:
    """The success rate of a replay frame, which must hold calibration rows only.

    Raises ``ValueError`` when a row's ``split`` is not ``calib`` (or the column is missing),
    or when the frame is empty.
    """
    require_splits(frame, TUNING_SPLITS)
    if len(frame) == 0:
        raise ValueError("no rows to measure the success of")
    return float(frame["success"].astype(bool).mean())


def calibration_target(best_single: pd.DataFrame, margin: float = QUALITY_MARGIN) -> float:
    """``Q_bar``: Best Single's calibration success minus ``margin``.

    ``best_single`` is the replay of the always-Best-Single policy on the calibration split.
    """
    return realized_success(best_single) - margin
