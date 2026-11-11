"""Tests of bench/scripts/failure_analysis.py: pairing, the breakdowns, the ranked losses and
the report. The replay frames are invented; nothing here runs the Go replay command."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import failure_analysis as fa  # noqa: E402

EM_DASH = "\u2014"
PICO = 10**12


def best_frame(rows: list[tuple[str, str, bool, float]]) -> pd.DataFrame:
    """(item_id, dataset, success, cost in USD) of the always-Best-Single replay."""
    return pd.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "dataset": [r[1] for r in rows],
            "success": [r[2] for r in rows],
            "cost_picousd": [round(r[3] * PICO) for r in rows],
        }
    )


def routed_frame(
    rows: list[tuple[str, str | None, bool, float, bool, float | None]],
) -> pd.DataFrame:
    """(item_id, chosen model, success, cost in USD, ood, p) of the learned replay."""
    return pd.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "model": [r[1] for r in rows],
            "success": [r[2] for r in rows],
            "cost_picousd": [round(r[3] * PICO) for r in rows],
            "ood": [r[4] for r in rows],
            "p": [r[5] for r in rows],
        }
    )


def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    best = best_frame(
        [
            ("a/0", "a", True, 1.0),
            ("a/1", "a", True, 1.0),
            ("a/2", "a", False, 1.0),
            ("b/0", "b", True, 2.0),
            ("b/1", "b", True, 2.0),
            ("b/2", "b", False, 2.0),
        ]
    )
    routed = routed_frame(
        [
            ("a/0", "cheap", False, 0.1, False, 0.9),  # loss
            ("a/1", "cheap", True, 0.1, False, 0.5),  # both right
            ("a/2", "cheap", True, 0.1, False, 0.4),  # gain
            ("b/0", "mid", False, 0.5, True, 0.95),  # loss, out of distribution
            ("b/1", "best", True, 2.0, False, 0.8),  # both right, same model
            ("b/2", None, False, 0.0, False, None),  # a routing error: a failure at zero cost
        ]
    )
    return best, routed


def test_pair_puts_the_two_outcomes_side_by_side() -> None:
    best, routed = sample()
    paired = fa.pair(best, routed)
    assert paired["item_id"].tolist() == ["a/0", "a/1", "a/2", "b/0", "b/1", "b/2"]
    assert paired["loss"].tolist() == [True, False, False, True, False, False]
    assert paired["gain"].tolist() == [False, False, True, False, False, False]
    assert paired.loc[paired["item_id"] == "b/2", "chosen"].iloc[0] == fa.NO_MODEL
    assert paired.loc[paired["item_id"] == "a/0", "routed_cost_usd"].iloc[0] == pytest.approx(0.1)


def test_pair_refuses_replays_of_different_items() -> None:
    best, routed = sample()
    with pytest.raises(ValueError, match="same items"):
        fa.pair(best.iloc[:-1], routed)
    with pytest.raises(ValueError, match="duplicate"):
        fa.pair(pd.concat([best, best.iloc[:1]]), routed)


def test_summarize_counts_losses_gains_and_the_cost_reduction() -> None:
    best, routed = sample()
    s = fa.summarize(fa.pair(best, routed))
    assert s["n"] == 6
    assert s["best_success"] == pytest.approx(4 / 6)
    assert s["routed_success"] == pytest.approx(3 / 6)
    assert s["success_delta"] == pytest.approx(-1 / 6)
    assert (s["losses"], s["gains"], s["net"]) == (2, 1, -1)
    assert (s["both_right"], s["both_wrong"]) == (2, 1)
    assert s["best_cost_usd"] == pytest.approx(9.0 / 6)
    assert s["routed_cost_usd"] == pytest.approx(2.8 / 6)
    assert s["cost_reduction"] == pytest.approx(1 - 2.8 / 9.0)
    # the four outcomes partition the items
    assert s["both_right"] + s["both_wrong"] + s["losses"] + s["gains"] == s["n"]


def test_breakdown_by_dataset_puts_the_worst_delta_first() -> None:
    best, routed = sample()
    paired = fa.pair(best, routed)
    rows = fa.breakdown(paired, "dataset")
    assert [r["dataset"] for r in rows] == ["b", "a"]  # b: -2/3, a: 0
    assert rows[0]["n"] == 3 and rows[0]["losses"] == 1 and rows[0]["net"] == -1
    assert rows[1]["losses"] == 1 and rows[1]["gains"] == 1 and rows[1]["net"] == 0


def test_the_breakdowns_sum_to_the_whole() -> None:
    best, routed = sample()
    analysis = fa.analyze(best, routed, bootstrap=20)
    for key in ("by_dataset", "by_chosen_model", "by_ood"):
        rows = analysis[key]
        assert sum(r["n"] for r in rows) == analysis["overall"]["n"]
        assert sum(r["losses"] for r in rows) == analysis["overall"]["losses"]
        assert sum(r["gains"] for r in rows) == analysis["overall"]["gains"]
    chosen = {r["chosen"]: r for r in analysis["by_chosen_model"]}
    assert set(chosen) == {"cheap", "mid", "best", fa.NO_MODEL}
    assert chosen["best"]["success_delta"] == 0 and chosen["best"]["cost_reduction"] == 0
    ood = {r["ood_label"]: r for r in analysis["by_ood"]}
    assert ood["out of distribution"]["n"] == 1 and ood["out of distribution"]["losses"] == 1
    assert ood["in distribution"]["n"] == 5


def test_top_losses_rank_the_most_confident_failures_first() -> None:
    n = 30
    best = best_frame([(f"d/{i}", "d", True, 1.0) for i in range(n)])
    # every item is lost; p decreases with i, except that two items tie and one has no p
    p = [0.99 - 0.01 * i for i in range(n)]
    p[5] = p[4]
    p[7] = None
    routed = routed_frame([(f"d/{i}", "m", False, 0.1, False, p[i]) for i in range(n)])
    top = fa.top_losses(fa.pair(best, routed))
    assert len(top) == fa.TOP_LOSSES == 20
    assert [t["rank"] for t in top] == list(range(1, 21))
    assert [t["item_id"] for t in top[:6]] == ["d/0", "d/1", "d/2", "d/3", "d/4", "d/5"]
    assert set(top[0]) == {"rank", "item_id", "dataset", "chosen"}  # nothing else is listed
    assert "d/7" not in {t["item_id"] for t in top}  # an item with no p ranks last
    assert len(fa.top_losses(fa.pair(best, routed), limit=3)) == 3


def test_top_losses_break_a_tie_by_item_id() -> None:
    best = best_frame([("z/1", "z", True, 1.0), ("a/9", "a", True, 1.0), ("m/5", "m", True, 1.0)])
    routed = routed_frame([(i, "m", False, 0.1, False, 0.5) for i in ("z/1", "a/9", "m/5")])
    assert [t["item_id"] for t in fa.top_losses(fa.pair(best, routed))] == ["a/9", "m/5", "z/1"]


def test_no_loss_means_an_empty_list() -> None:
    best = best_frame([("a/0", "a", True, 1.0)])
    routed = routed_frame([("a/0", "m", True, 0.1, False, 0.9)])
    assert fa.top_losses(fa.pair(best, routed)) == []


def test_the_overall_comparison_uses_the_paired_bootstrap() -> None:
    best, routed = sample()
    analysis = fa.analyze(best, routed, bootstrap=200)
    v = analysis["overall_vs_best_single"]
    assert v["success_delta"] == pytest.approx(-1 / 6)
    assert v["cost_reduction"] == pytest.approx(1 - 2.8 / 9.0)
    lo, hi = v["success_delta_ci"]
    assert lo <= v["success_delta"] <= hi
    assert v["bootstrap"]["n"] == 200


def test_the_report_lists_ids_datasets_and_models_but_no_text() -> None:
    best, routed = sample()
    analysis = fa.analyze(best, routed, bootstrap=20)
    results = {
        "best_single": "best",
        "lambda": 1.5,
        "artifact_version": "v1",
        **analysis,
    }
    tables = fa._tables(results)
    assert len(tables) == 5
    losses = tables[f"The {fa.TOP_LOSSES} largest losses: Best Single right, routed model wrong"]
    assert list(losses.columns) == ["rank", "item_id", "dataset", "chosen"]
    assert losses["item_id"].tolist() == ["b/0", "a/0"]  # p 0.95 before 0.9
    text = json.dumps(results, default=str) + "\n".join(t.to_csv() for t in tables.values())
    text += "\n".join(fa.LABELS)
    assert EM_DASH not in text
    assert np.isfinite(results["overall"]["best_cost_usd"])
