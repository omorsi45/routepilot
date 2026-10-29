"""Tests of bench/scripts/live_demo.py: the SLO values, the item subset, the nine rounds, the
learned-policy config and the per-round summary. Everything here is invented; nothing starts the
gateway or talks to Ollama."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import live_collect as lc  # noqa: E402
import live_demo as ld  # noqa: E402

EM_DASH = "\u2014"
SENTINEL = "PROMPT-SENTINEL-c2d1"


def splits_for(ids: list[str], kinds: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"item_id": ids, "dataset": "x", "split": kinds})


# --- the SLO values -----------------------------------------------------------------------------


def test_slo_values_are_the_fastest_p50_and_the_slowest_p95_of_the_tuning_requests() -> None:
    ids = [f"i{k}" for k in range(11)]
    kinds = ["train"] * 5 + ["calib"] * 5 + ["test"]
    splits = splits_for(ids, kinds)
    rows = []
    for k, item in enumerate(ids[:10]):  # 10 tuning requests per model
        rows.append({"item_id": item, "model": "fast", "e2e_ms": 1000.0 + k})
        rows.append({"item_id": item, "model": "slow", "e2e_ms": 5000.0 + 10 * k})
    rows.append({"item_id": "i10", "model": "slow", "e2e_ms": 1e9})  # a test request: never read
    rows.append({"item_id": "i10", "model": "fast", "e2e_ms": 1.0})
    slos = ld.slo_values(pd.DataFrame(rows), splits, ["fast", "slow"])
    assert slos["tight_ms"] == int(np.ceil(np.percentile(1000.0 + np.arange(10), 50)))  # 1005
    assert slos["loose_ms"] == int(np.ceil(np.percentile(5000.0 + 10 * np.arange(10), 95)))
    assert slos["tight_from"] == "p50 of fast" and slos["loose_from"] == "p95 of slow"
    assert slos["tight_ms"] == 1005 and slos["loose_ms"] == 5086


def test_slo_values_need_timings_for_every_model() -> None:
    splits = splits_for(["a"], ["train"])
    measured = pd.DataFrame({"item_id": ["a"], "model": ["x"], "e2e_ms": [10.0]})
    with pytest.raises(ValueError, match="no train or calibration timings for y"):
        ld.slo_values(measured, splits, ["x", "y"])


# --- items and rounds ---------------------------------------------------------------------------


def test_demo_items_are_a_seeded_subset_of_the_complete_test_items() -> None:
    ids = [f"gpqa/{k}" for k in range(200)]
    kinds = ["test" if k % 2 == 0 else "train" for k in range(200)]
    splits = splits_for(ids, kinds)
    complete = [i for k, i in enumerate(ids) if k % 10 != 0]  # every tenth item is incomplete
    picked = ld.demo_items(splits, complete, max_items=30)
    assert len(picked) == len(set(picked)) == 30
    test_complete = {i for k, i in enumerate(ids) if k % 2 == 0 and k % 10 != 0}
    assert set(picked) <= test_complete
    assert ld.demo_items(splits, complete, max_items=30) == picked
    assert ld.demo_items(splits, complete, max_items=30, seed=1) != picked
    assert len(ld.demo_items(splits, complete, max_items=500)) == len(test_complete)
    with pytest.raises(ValueError, match="no live test items"):
        ld.demo_items(splits, [], max_items=5)


def test_plan_items_per_round_fits_the_budget() -> None:
    # 25 minutes over 9 rounds at 3.7 s a request: 45 items
    assert ld.plan_items_per_round(3.7, available=200, budget_s=1500, rounds=9) == 45
    assert ld.plan_items_per_round(3.7, available=200, budget_s=1500, rounds=9, cap=30) == 30
    assert ld.plan_items_per_round(3.7, available=40, budget_s=1500, rounds=9) == 40
    # never below the minimum unless fewer exist
    assert ld.plan_items_per_round(30.0, available=200, budget_s=600, rounds=9) == ld.MIN_ITEMS
    assert ld.plan_items_per_round(30.0, available=7, budget_s=600, rounds=9) == 7
    with pytest.raises(ValueError, match="positive"):
        ld.plan_items_per_round(0.0, available=5, budget_s=10, rounds=9)


def test_the_nine_rounds_run_floors_outside_and_slos_inside() -> None:
    settings = ld.round_settings({"tight_ms": 1800, "loose_ms": 6000})
    assert [s["round"] for s in settings] == [
        "q0-none", "q0-tight", "q0-loose",
        "q0.4-none", "q0.4-tight", "q0.4-loose",
        "q0.6-none", "q0.6-tight", "q0.6-loose",
    ]  # fmt: skip
    assert [s["slo_e2e_ms"] for s in settings[:3]] == [None, 1800, 6000]
    assert {s["min_quality"] for s in settings} == {0.0, 0.4, 0.6}


def test_request_options_send_the_floor_always_and_the_slo_when_there_is_one() -> None:
    assert ld.routepilot_options({"min_quality": 0.0, "slo_e2e_ms": None}) == {"min_quality": 0.0}
    assert ld.routepilot_options({"min_quality": 0.6, "slo_e2e_ms": 1800}) == {
        "min_quality": 0.6,
        "slo_e2e_ms": 1800,
    }


def test_demo_config_serves_auto_on_the_learned_policy(tmp_path: Path) -> None:
    base = yaml.safe_load(lc.CONFIG.read_text(encoding="utf-8"))
    before = json.dumps(base, sort_keys=True)
    cfg = ld.demo_config(
        base,
        policies_root=tmp_path / "policies",
        version="abc123def456",
        policy_path=tmp_path / "policy.yaml",
        profiles_path=tmp_path / "profiles.yaml",
        overheads={"qwen2.5-0.5b": 19, "qwen2.5-3b": 21},
    )
    assert json.dumps(base, sort_keys=True) == before
    assert cfg["routing"] == {"policy": "learned"}
    stable = cfg["policies"]["stable"]
    assert (stable["name"], stable["version"]) == ("live-ollama", "abc123def456")
    for path in (cfg["policies"]["root"], stable["config"], cfg["server"]["latency_profiles"]):
        assert Path(path).is_absolute()
    overhead = {m["id"]: m["prompt_overhead_tokens"] for m in cfg["models"]}
    assert overhead == {"qwen2.5-0.5b": 19, "qwen2.5-1.5b": 0, "qwen2.5-3b": 21}


# --- the round summary --------------------------------------------------------------------------


def requests(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_summarize_round_counts_the_mix_relaxations_success_and_slo() -> None:
    rows = requests(
        ok=[True, True, True, True, False],
        served=["a", "a", "b", "b", ""],
        relaxed=[[], ["latency"], ["latency", "quality"], [], []],
        score=[1.0, 0.0, 1.0, 1.0, np.nan],
        e2e_ms=[900.0, 1500.0, 2500.0, 1000.0, np.nan],
        ttft_ms=[30.0, 40.0, 50.0, 60.0, np.nan],
        completion_tokens=[100, 200, 300, 400, 0],
    )
    setting = {"round": "q0.4-tight", "min_quality": 0.4, "slo_kind": "tight", "slo_e2e_ms": 1600}
    s = ld.summarize_round(rows, setting)
    assert (s["n"], s["ok"], s["failed"]) == (5, 4, 1)
    assert s["mix"] == {"a": 2, "b": 2}
    assert s["relaxed"] == {"any": 2, "latency": 2, "quality": 1}
    assert s["success"]["n"] == 4 and s["success"]["hits"] == 3 and s["success"]["rate"] == 0.75
    lo, hi = s["success"]["wilson_95"]
    assert 0 < lo < 0.75 < hi < 1
    assert s["success_not_relaxed"] == {"n": 2, "rate": 1.0}  # the two requests nothing relaxed
    assert s["floor_met_not_relaxed"] is True
    assert s["e2e_ms"]["p50"] == pytest.approx(np.percentile([900, 1500, 2500, 1000], 50))
    assert s["ttft_ms"]["p95"] == pytest.approx(np.percentile([30, 40, 50, 60], 95))
    assert s["completion_tokens_mean"] == 250
    assert s["slo"]["ms"] == 1600
    assert s["slo"]["met_share"] == 0.75  # 900, 1500 and 1000 are within 1600
    assert s["slo"]["p95_within"] is False
    assert s["slo"]["not_relaxed_met_share"] == 1.0


def test_summarize_round_without_an_slo_or_any_success() -> None:
    rows = requests(
        ok=[True],
        served=["a"],
        relaxed=[["quality"]],
        score=[0.0],
        e2e_ms=[700.0],
        ttft_ms=[20.0],
        completion_tokens=[50],
    )
    setting = {"round": "q0.6-none", "min_quality": 0.6, "slo_kind": "none", "slo_e2e_ms": None}
    s = ld.summarize_round(rows, setting)
    assert s["slo"] is None
    assert s["success"]["rate"] == 0.0
    assert s["success_not_relaxed"] == {"n": 0, "rate": None}
    assert s["floor_met_not_relaxed"] is None


def test_summarize_round_with_every_request_failed() -> None:
    rows = requests(
        ok=[False, False],
        served=["", ""],
        relaxed=[[], []],
        score=[np.nan, np.nan],
        e2e_ms=[np.nan, np.nan],
        ttft_ms=[np.nan, np.nan],
        completion_tokens=[0, 0],
    )
    setting = {"round": "q0-loose", "min_quality": 0.0, "slo_kind": "loose", "slo_e2e_ms": 5000}
    s = ld.summarize_round(rows, setting)
    assert (s["ok"], s["failed"], s["mix"]) == (0, 2, {})
    assert s["success"]["rate"] is None and s["e2e_ms"] is None
    assert s["slo"]["met_share"] is None


def test_estimator_bounds_take_the_median_bound_per_round_and_model() -> None:
    def cands(a: float | None, b: float) -> list[dict]:
        return [
            {"model": "a", "q_ms": a, "feasible": True},
            {"model": "b", "q_ms": b, "feasible": True},
        ]

    decisions = pd.DataFrame(
        {
            "decision_id": ["d1", "d2", "d3", "d4"],
            "candidates": [cands(100.0, 200.0), cands(300.0, 400.0), cands(None, 500.0), None],
        }
    )
    got = ld.estimator_bounds(decisions, {"d1": "r1", "d2": "r1", "d3": "r2", "d4": "r2"})
    assert got == {"r1": {"a": 200.0, "b": 300.0}, "r2": {"b": 500.0}}
    assert ld.estimator_bounds(decisions, {}) == {}


def test_the_report_tables_hold_no_prompt_text() -> None:
    rows = requests(
        ok=[True], served=["a"], relaxed=[[]], score=[1.0], e2e_ms=[900.0], ttft_ms=[30.0],
        completion_tokens=[10],
    )  # fmt: skip
    setting = {"round": "q0-none", "min_quality": 0.0, "slo_kind": "none", "slo_e2e_ms": None}
    results = {
        "models": ["a"],
        "artifact_version": "v1",
        "policy_config": {"mode": "min_cost"},
        "slo": {"tight_ms": 1, "loose_ms": 2, "tight_from": "p50 of a", "loose_from": "p95 of a"},
        "items_per_round": 1,
        "test_items_available": 1,
        "requests": 1,
        "served_differs_from_chosen": 0,
        "decisions_missing": 0,
        "wall_time_s": 1.0,
        "rounds": [ld.summarize_round(rows, setting)],
        "estimator_bound_ms_by_round": {"q0-none": {"a": 12.5}},
    }
    tables = ld._tables(results)
    text = "\n".join(t.to_csv() for t in tables.values()) + "\n".join(ld.LABELS)
    assert SENTINEL not in text and EM_DASH not in text
    assert "Median end-to-end bound the policy used (q_ms), per model and round" in tables
