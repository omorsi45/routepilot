import pandas as pd
import pytest

from routepilot_train.evaluate import pareto


def points(rows):
    return pd.DataFrame(rows, columns=["name", "mean_cost_usd", "success_rate"])


def test_pareto_front_drops_dominated():
    df = points(
        [
            ("cheap-weak", 1.0, 0.50),
            ("mid", 2.0, 0.70),
            ("dominated-by-mid", 3.0, 0.60),  # dearer and worse than mid
            ("same-quality-dearer", 2.5, 0.70),  # dearer, same quality
            ("same-cost-worse", 2.0, 0.65),  # same cost, lower quality
            ("top", 4.0, 0.90),
        ]
    )
    front = pareto.pareto_front(df)
    assert list(front["name"]) == ["cheap-weak", "mid", "top"]
    assert list(front["mean_cost_usd"]) == [1.0, 2.0, 4.0]  # sorted by cost
    assert list(front.columns) == list(df.columns)  # other columns survive
    assert list(front.index) == [0, 1, 2]


def test_pareto_front_keeps_identical_points_and_a_single_point():
    df = points([("a", 1.0, 0.5), ("b", 1.0, 0.5), ("c", 0.5, 0.4)])
    assert sorted(pareto.pareto_front(df)["name"]) == ["a", "b", "c"]
    assert list(pareto.pareto_front(points([("only", 1.0, 0.1)]))["name"]) == ["only"]


def test_pareto_front_custom_columns_and_empty_input():
    df = pd.DataFrame({"usd": [3.0, 1.0, 2.0], "acc": [0.9, 0.5, 0.4]})
    front = pareto.pareto_front(df, cost="usd", quality="acc")
    assert list(front["usd"]) == [1.0, 3.0]
    empty = pareto.pareto_front(points([]))
    assert empty.empty and list(empty.columns) == ["name", "mean_cost_usd", "success_rate"]


def test_pareto_front_needs_its_columns():
    with pytest.raises(ValueError, match="mean_cost_usd"):
        pareto.pareto_front(pd.DataFrame({"success_rate": [0.5]}))


def test_pareto_front_does_not_modify_its_input():
    df = points([("a", 2.0, 0.5), ("b", 1.0, 0.9)])
    before = df.copy()
    pareto.pareto_front(df)
    pd.testing.assert_frame_equal(df, before)
