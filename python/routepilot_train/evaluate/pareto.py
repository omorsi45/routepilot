"""Pareto fronts over cost and quality."""

import numpy as np
import pandas as pd


def pareto_front(
    points: pd.DataFrame, cost: str = "mean_cost_usd", quality: str = "success_rate"
) -> pd.DataFrame:
    """The non-dominated rows, sorted by cost ascending.

    A row is dominated when another row costs no more and has no lower quality, and is
    strictly better on at least one of the two. Identical points do not dominate each
    other, so both stay. Other columns are kept; the index is reset.
    """
    missing = [column for column in (cost, quality) if column not in points.columns]
    if missing:
        raise ValueError(f"points has no column {missing[0]!r}")
    if points.empty:
        return points.reset_index(drop=True)
    c = points[cost].to_numpy(dtype=float)
    q = points[quality].to_numpy(dtype=float)
    dominated = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        no_worse = (c <= c[i]) & (q >= q[i])
        strictly_better = (c < c[i]) | (q > q[i])
        dominated[i] = bool((no_worse & strictly_better).any())
    front = points.loc[~dominated].sort_values(
        [cost, quality], ascending=[True, False], kind="stable"
    )
    return front.reset_index(drop=True)
