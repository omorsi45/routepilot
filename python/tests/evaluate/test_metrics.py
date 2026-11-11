import numpy as np
import pandas as pd
import pytest

from routepilot_train.evaluate import metrics

PICO = 1e12


def replay_frame(success, cost_pico, errors=None, dataset="d") -> pd.DataFrame:
    n = len(success)
    return pd.DataFrame(
        {
            "item_id": [f"{dataset}/{i}" for i in range(n)],
            "dataset": dataset,
            "success": [bool(s) for s in success],
            "cost_picousd": np.asarray(cost_pico, dtype="int64"),
            "error": errors if errors is not None else [""] * n,
        }
    )


def test_bootstrap_ci_covers_mean():
    x = np.random.default_rng(1).normal(5.0, 2.0, 400)
    point, lo, hi = metrics.bootstrap_ci(x, seed=3)
    assert point == pytest.approx(x.mean())
    assert lo < 5.0 < hi
    assert lo < point < hi
    # a 95% interval of a mean is about 2 * 1.96 * sd / sqrt(n) wide
    assert hi - lo == pytest.approx(2 * 1.96 * x.std(ddof=1) / np.sqrt(len(x)), rel=0.15)
    # deterministic for a seed, different for another
    assert metrics.bootstrap_ci(x, seed=3) == (point, lo, hi)
    assert metrics.bootstrap_ci(x, seed=4)[1:] != (lo, hi)


def test_bootstrap_ci_takes_other_statistics_and_alpha():
    x = np.random.default_rng(2).exponential(3.0, 300)
    point, lo, hi = metrics.bootstrap_ci(x, stat=np.median, seed=0)
    assert point == pytest.approx(np.median(x)) and lo < point < hi
    _, lo90, hi90 = metrics.bootstrap_ci(x, seed=0, alpha=0.10)
    _, lo95, hi95 = metrics.bootstrap_ci(x, seed=0, alpha=0.05)
    assert lo95 < lo90 < hi90 < hi95


def test_bootstrap_ci_degenerate_inputs():
    assert metrics.bootstrap_ci(np.full(20, 0.25)) == (0.25, 0.25, 0.25)
    with pytest.raises(ValueError, match="empty"):
        metrics.bootstrap_ci(np.array([]))


def test_summarize_counts_errors_as_failures_at_zero_cost():
    df = replay_frame(
        [True, True, False, False, False],
        [200, 400, 100, 0, 0],
        errors=["", "", "", "model_not_found", "model_not_found"],
    )
    got = metrics.summarize(df)
    assert got["n"] == 5 and got["errors"] == 2
    assert got["success_rate"] == pytest.approx(0.4)
    assert got["mean_cost_usd"] == pytest.approx(700 / 5 / PICO)
    assert got["cost_per_success_usd"] == pytest.approx(700 / 2 / PICO)


def test_summarize_without_a_success_has_no_cost_per_success():
    got = metrics.summarize(replay_frame([False, False], [10, 20]))
    assert got["success_rate"] == 0.0
    assert got["cost_per_success_usd"] is None


def test_paired_delta_identical_is_zero():
    df = replay_frame([1, 0, 1, 1, 0, 1], [300, 200, 100, 500, 400, 600])
    got = metrics.paired_delta(df, df.copy(), n=200, seed=0)
    assert got["n"] == 6
    assert got["success_delta"] == 0.0 and got["cost_reduction"] == 0.0
    assert got["success_delta_ci"] == [0.0, 0.0]
    assert got["cost_reduction_ci"] == [0.0, 0.0]


def test_paired_delta_known_reduction():
    cost_b = np.arange(1, 101) * 1000
    b = replay_frame([1] * 100, cost_b)
    a = replay_frame([1] * 100, cost_b // 2)
    got = metrics.paired_delta(a, b, n=200, seed=0)
    assert got["cost_reduction"] == pytest.approx(0.5)
    assert got["cost_reduction_ci"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert got["success_delta"] == 0.0


def test_paired_delta_success_delta_and_pairing_by_item_id():
    b = replay_frame([1] * 50 + [0] * 50, [100] * 100)
    a = replay_frame([1] * 80 + [0] * 20, [100] * 100)
    shuffled = a.sample(frac=1.0, random_state=0).reset_index(drop=True)
    got = metrics.paired_delta(shuffled, b, n=500, seed=1)
    assert got["success_delta"] == pytest.approx(0.3)
    lo, hi = got["success_delta_ci"]
    assert 0.0 < lo < 0.3 < hi < 0.6
    assert got["cost_reduction"] == 0.0


def test_paired_delta_needs_the_same_items():
    a = replay_frame([1, 0, 1], [1, 2, 3])
    b = replay_frame([1, 0, 1, 1], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="same items"):
        metrics.paired_delta(a, b)
    dup = pd.concat([a, a.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        metrics.paired_delta(dup, dup)


def test_latency_summary_is_prefixed_simulated():
    df = pd.DataFrame(
        {
            "e2e_ms_sim": [100.0, 200.0, 300.0, 400.0, 500.0],
            "ttft_ms_sim": [10.0, 20.0, 30.0, 40.0, 50.0],
            "slo_violated": pd.array([True, False, False, False, None], dtype="boolean"),
        }
    )
    got = metrics.latency_summary(df)
    assert got["simulated_e2e_ms_p50"] == pytest.approx(300.0)
    assert got["simulated_e2e_ms_p95"] == pytest.approx(480.0)
    assert got["simulated_ttft_ms_p50"] == pytest.approx(30.0)
    assert got["simulated_ttft_ms_p95"] == pytest.approx(48.0)
    assert got["simulated_slo_violation_rate"] == pytest.approx(0.25)  # 1 of 4 with an SLO
    assert all(key.startswith("simulated_") for key in got)


def test_latency_summary_without_latency_is_null():
    df = pd.DataFrame(
        {"e2e_ms_sim": [None] * 3, "ttft_ms_sim": [None] * 3, "slo_violated": [None] * 3}
    )
    got = metrics.latency_summary(df)
    assert set(got.values()) == {None}
    assert all(key.startswith("simulated_") for key in got)


def test_floor_compliance_excludes_relaxed():
    n = 40
    relaxed = [["quality"] if i < 10 else [] for i in range(n)]
    success = [False if i < 10 else (i % 10) < 8 for i in range(n)]  # 80% when not relaxed
    df = pd.DataFrame(
        {
            "floor": [0.7] * n,
            "relaxed": relaxed,
            "l": np.linspace(0.5, 0.95, n),
            "success": success,
        }
    )
    got = metrics.floor_compliance(df)

    assert got["n"] == 30 and got["n_relaxed_excluded"] == 10
    assert got["realized_success"] == pytest.approx(0.8)
    assert got["realized_success"] != pytest.approx(24 / 40)  # relaxed rows would pull it down
    assert got["floors"] == [{"floor": 0.7, "n": 30, "realized_success": 0.8, "met": True}]
    assert got["met"] is True
    table = pd.DataFrame(got["reliability"])
    assert list(table.columns) == ["decile", "n", "mean_l", "realized_success"]
    assert list(table["decile"]) == list(range(1, 11)) and table["n"].sum() == 30
    assert table["mean_l"].is_monotonic_increasing


def test_floor_compliance_reports_a_missed_floor_per_floor():
    df = pd.DataFrame(
        {
            "floor": [0.9] * 10 + [0.5] * 10 + [np.nan] * 5,
            "relaxed": [[]] * 25,
            "l": [0.6] * 25,
            "success": [True] * 7 + [False] * 3 + [True] * 6 + [False] * 4 + [True] * 5,
        }
    )
    got = metrics.floor_compliance(df)
    assert got["n"] == 20  # rows without a floor are not counted
    by_floor = {f["floor"]: f for f in got["floors"]}
    assert by_floor[0.9]["met"] is False and by_floor[0.9]["realized_success"] == pytest.approx(0.7)
    assert by_floor[0.5]["met"] is True and by_floor[0.5]["realized_success"] == pytest.approx(0.6)
    assert got["met"] is False


def test_floor_compliance_without_floors_is_empty():
    df = pd.DataFrame(
        {"floor": [None] * 3, "relaxed": [[]] * 3, "l": [None] * 3, "success": [True] * 3}
    )
    got = metrics.floor_compliance(df)
    assert got["n"] == 0 and got["met"] is None and got["floors"] == [] and got["reliability"] == []


def test_floor_compliance_reliability_with_fewer_rows_than_deciles():
    """One group per row when there are fewer than ten rows; one row used to raise."""
    for n in (1, 2, 4, 9):
        df = pd.DataFrame(
            {
                "floor": [0.5] * n,
                "relaxed": [[]] * n,
                "l": np.linspace(0.5, 0.9, n) if n > 1 else [0.7],
                "success": [True, False] * (n // 2) + [True] * (n % 2),
            }
        )

        got = metrics.floor_compliance(df)

        table = pd.DataFrame(got["reliability"])
        assert list(table["decile"]) == list(range(1, n + 1)), n
        assert (table["n"] == 1).all() and table["n"].sum() == got["n"] == n
        assert got["realized_success"] == pytest.approx(df["success"].mean())
