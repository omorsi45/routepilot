"""Metrics over replay output frames, with percentile bootstrap intervals.

A replay frame has one row per item with at least ``item_id``, ``success`` (bool),
``cost_picousd`` (int) and ``error`` (empty when routing succeeded). Money is held as
integer picodollars until the last step and reported as USD floats.
"""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd

PICO = 1e12
# Largest resampling index matrix drawn at once, in cells, to bound memory.
_MAX_CELLS = 4_000_000
DECILES = 10


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Counts and rates of one replay.

    A routing error is a failure at zero cost and counts in ``errors``.
    ``cost_per_success_usd`` is total cost over successes, or None without a success.
    """
    n = len(df)
    if n == 0:
        raise ValueError("cannot summarize an empty replay frame")
    successes = int(df["success"].astype(bool).sum())
    total = int(df["cost_picousd"].sum())
    errors = int(df["error"].fillna("").ne("").sum()) if "error" in df.columns else 0
    return {
        "n": n,
        "errors": errors,
        "success_rate": successes / n,
        "mean_cost_usd": total / n / PICO,
        "cost_per_success_usd": total / successes / PICO if successes else None,
    }


def _index_chunks(rng: np.random.Generator, size: int, n: int) -> Iterator[np.ndarray]:
    """Resampling indices, ``n`` rows of ``size`` draws, in chunks of bounded size."""
    step = max(1, _MAX_CELLS // size)
    for start in range(0, n, step):
        yield rng.integers(0, size, size=(min(step, n - start), size))


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[..., Any] = np.mean,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """The statistic of ``x`` and its percentile bootstrap interval, as (point, lo, hi).

    ``stat`` must take an ``axis`` argument, like ``np.mean`` and ``np.median``. The
    interval covers ``1 - alpha`` of ``n`` resamples drawn with a fixed seed.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    boots = np.concatenate([stat(x[idx], axis=1) for idx in _index_chunks(rng, x.size, n)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def _paired_arrays(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, ...]:
    for name, frame in (("a", a), ("b", b)):
        if frame["item_id"].duplicated().any():
            raise ValueError(f"duplicate item_id in {name}")
    if set(a["item_id"]) != set(b["item_id"]):
        raise ValueError("a and b must hold the same items")
    merged = a[["item_id", "success", "cost_picousd"]].merge(
        b[["item_id", "success", "cost_picousd"]],
        on="item_id",
        suffixes=("_a", "_b"),
        validate="1:1",
    )
    return (
        merged["success_a"].to_numpy(dtype=float),
        merged["success_b"].to_numpy(dtype=float),
        merged["cost_picousd_a"].to_numpy(dtype=float),
        merged["cost_picousd_b"].to_numpy(dtype=float),
    )


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Success delta and cost reduction of ``a`` against the reference ``b``.

    Items are joined on ``item_id`` and must be the same in both frames.
    ``success_delta`` is ``success_a - success_b`` and ``cost_reduction`` is
    ``1 - mean_cost_a / mean_cost_b`` (positive when ``a`` is cheaper). Each has a 95%
    percentile interval from resampling items jointly, so the pairing is kept.
    ``cost_reduction`` is None when ``b`` costs nothing.
    """
    sa, sb, ca, cb = _paired_arrays(a, b)
    size = len(sa)
    if size == 0:
        raise ValueError("cannot compare empty frames")
    rng = np.random.default_rng(seed)
    success_boot: list[np.ndarray] = []
    cost_boot: list[np.ndarray] = []
    for idx in _index_chunks(rng, size, n):
        success_boot.append(sa[idx].mean(axis=1) - sb[idx].mean(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            cost_boot.append(1.0 - ca[idx].mean(axis=1) / cb[idx].mean(axis=1))

    def interval(values: list[np.ndarray]) -> list[float] | None:
        joined = np.concatenate(values)
        joined = joined[np.isfinite(joined)]
        if joined.size == 0:
            return None
        lo, hi = np.quantile(joined, [0.025, 0.975])
        return [float(lo), float(hi)]

    reducible = cb.mean() > 0
    return {
        "n": size,
        "success_delta": float(sa.mean() - sb.mean()),
        "success_delta_ci": interval(success_boot),
        "cost_reduction": float(1.0 - ca.mean() / cb.mean()) if reducible else None,
        "cost_reduction_ci": interval(cost_boot) if reducible else None,
    }


def _percentile(df: pd.DataFrame, column: str, q: float) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(np.percentile(values, q)) if len(values) else None


def latency_summary(df: pd.DataFrame) -> dict[str, float | None]:
    """p50 and p95 of the simulated latencies and the SLO violation rate.

    Every key starts with ``simulated_`` because no latency here is measured. A value
    is None when the frame has no such data.
    """
    violated = (
        df["slo_violated"].dropna() if "slo_violated" in df.columns else pd.Series(dtype=bool)
    )
    return {
        "simulated_e2e_ms_p50": _percentile(df, "e2e_ms_sim", 50),
        "simulated_e2e_ms_p95": _percentile(df, "e2e_ms_sim", 95),
        "simulated_ttft_ms_p50": _percentile(df, "ttft_ms_sim", 50),
        "simulated_ttft_ms_p95": _percentile(df, "ttft_ms_sim", 95),
        "simulated_slo_violation_rate": float(violated.astype(bool).mean())
        if len(violated)
        else None,
    }


def _is_relaxed(value: Any) -> bool:
    return isinstance(value, list | tuple | np.ndarray) and len(value) > 0


def floor_compliance(df: pd.DataFrame) -> dict[str, Any]:
    """Whether the realized success rate meets the quality floor.

    Only rows with a ``floor`` and an empty ``relaxed`` list count: a request whose floor
    the router had to relax made no promise. Per floor value: the rows, the realized
    success rate and ``met = rate >= floor``. ``reliability`` is the realized success per
    decile of the predicted lower bound ``l`` over the same rows (one group per row when
    there are fewer than ten).
    """
    floor = pd.to_numeric(df["floor"], errors="coerce")
    has_floor = floor.notna()
    relaxed = df["relaxed"].map(_is_relaxed)
    eligible = df.loc[has_floor & ~relaxed].assign(floor=floor[has_floor & ~relaxed])
    result: dict[str, Any] = {
        "n": len(eligible),
        "n_relaxed_excluded": int((has_floor & relaxed).sum()),
        "realized_success": float(eligible["success"].astype(bool).mean())
        if len(eligible)
        else None,
        "floors": [],
        "met": None,
        "reliability": [],
    }
    for value, rows in eligible.groupby("floor"):
        rate = float(rows["success"].astype(bool).mean())
        result["floors"].append(
            {
                "floor": float(value),
                "n": len(rows),
                "realized_success": rate,
                "met": bool(rate >= value),
            }
        )
    if result["floors"]:
        result["met"] = all(f["met"] for f in result["floors"])

    scored = eligible.loc[pd.to_numeric(eligible["l"], errors="coerce").notna()]
    if len(scored):
        l_values = pd.to_numeric(scored["l"])
        groups = min(DECILES, len(scored))  # fewer rows than deciles: one group per row
        decile = pd.qcut(l_values.rank(method="first"), groups, labels=False) + 1
        table = (
            pd.DataFrame(
                {"decile": decile, "l": l_values, "success": scored["success"].astype(bool)}
            )
            .groupby("decile")
            .agg(n=("l", "size"), mean_l=("l", "mean"), realized_success=("success", "mean"))
            .reset_index()
        )
        result["reliability"] = table.to_dict("records")
    return result
