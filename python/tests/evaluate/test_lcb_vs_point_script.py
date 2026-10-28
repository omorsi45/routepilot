"""Tests for the floor-on-LCB versus floor-on-P experiment script."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import pandas as pd
import pytest

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "lcb_vs_point.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lcb_vs_point", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure helpers ---------------------------------------------------------------------------------


def test_the_floors_are_fixed_in_advance_from_five_hundredths_to_sixty():
    script = load_script()
    assert script.FLOORS == (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
    assert script.ARMS == ("lcb", "point") and script.SPLITS == ("test", "calib")


def test_arm_flags_add_floor_on_point_only_to_the_point_arm():
    script = load_script()
    assert script.arm_flags(0.25, "lcb") == ["--mode", "min_cost", "--floor", "0.25"]
    assert script.arm_flags(0.25, "point") == [
        "--mode",
        "min_cost",
        "--floor",
        "0.25",
        "--floor-on",
        "point",
    ]
    with pytest.raises(ValueError, match="arm"):
        script.arm_flags(0.25, "median")


def test_wilson_interval_matches_known_values_and_handles_the_edges():
    script = load_script()
    lo, hi = script.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = script.wilson_interval(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-3)
    lo, hi = script.wilson_interval(20, 20)
    assert lo == pytest.approx(0.8389, abs=1e-3) and hi == 1.0
    assert script.wilson_interval(0, 0) is None


def replay_frame(floor: float, rows: list[tuple[bool, list[str], float, float]]) -> pd.DataFrame:
    """(success, relaxed, cost picodollars, l) rows of a replay at ``floor``."""
    return pd.DataFrame(
        {
            "item_id": [f"i{k}" for k in range(len(rows))],
            "split": "test",
            "model": "m1",
            "success": [r[0] for r in rows],
            "relaxed": [r[1] for r in rows],
            "cost_picousd": [int(r[2]) for r in rows],
            "l": [r[3] for r in rows],
            "floor": floor,
            "error": "",
        }
    )


def test_compliance_row_counts_only_decisions_that_were_not_relaxed():
    script = load_script()
    frame = replay_frame(
        0.5,
        [
            (True, [], 1000, 0.6),
            (True, [], 1000, 0.6),
            (False, [], 1000, 0.6),
            (False, [], 1000, 0.6),
            (False, ["quality"], 5000, 0.1),  # relaxed: no promise, not in the success rate
            (True, ["quality"], 5000, 0.1),
        ],
    )

    row = script.compliance_row(frame, 0.5, "test")

    assert row["n"] == 6 and row["n_not_relaxed"] == 4
    assert row["not_relaxed_share"] == pytest.approx(4 / 6)
    assert row["relaxed_share"] == pytest.approx(2 / 6)
    assert row["realized_success"] == 0.5 and row["met"] is True  # 0.5 >= 0.5
    lo, hi = row["realized_ci"]
    assert lo < 0.5 < hi
    assert row["mean_cost_usd"] == pytest.approx((4 * 1000 + 2 * 5000) / 6 / 1e12)
    assert row["success_rate"] == pytest.approx(3 / 6)

    miss = script.compliance_row(frame, 0.75, "test")
    assert miss["met"] is False  # the realized 0.5 is below 0.75


def test_compliance_row_with_every_decision_relaxed_has_nothing_to_judge():
    script = load_script()
    frame = replay_frame(0.9, [(True, ["quality"], 1000, 0.1), (False, ["quality"], 1000, 0.1)])
    row = script.compliance_row(frame, 0.9, "test")
    assert row["n_not_relaxed"] == 0 and row["not_relaxed_share"] == 0.0
    assert row["realized_success"] is None and row["realized_ci"] is None and row["met"] is None


def test_compliance_row_refuses_rows_of_another_split():
    script = load_script()
    frame = replay_frame(0.5, [(True, [], 1000, 0.6)])
    with pytest.raises(ValueError, match="splits not allowed"):
        script.compliance_row(frame, 0.5, "calib")


def test_summarize_arm_counts_floors_met_and_names_the_missed_ones():
    script = load_script()
    rows = [
        {"floor": 0.1, "met": True},
        {"floor": 0.2, "met": False},
        {"floor": 0.3, "met": None},
        {"floor": 0.4, "met": False},
    ]
    assert script.summarize_arm(rows) == {
        "floors": 4,
        "with_decisions": 3,
        "met": 1,
        "missed_floors": [0.2, 0.4],
    }


# --- the whole script on the synthetic pool -------------------------------------------------------


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Every replay the script makes: (split, policy, flags)."""
    calls: list[tuple] = []
    real = replaytool.replay

    def spy(binary, config, input, split, policy, out, extra=()):  # noqa: A002
        calls.append((split, policy, tuple(extra)))
        return real(binary, config, input, split, policy, out, extra)

    monkeypatch.setattr(replaytool, "replay", spy)
    return calls


FLOORS = (0.1, 0.4, 0.7)


def run(derived: Path, binary: Path, out: Path, floors=FLOORS) -> Path:
    script = load_script()
    return script.run_lcb_vs_point(
        derived,
        "tiny",
        TINY_ARTIFACT,
        LEARNED / "policy.yaml",
        out,
        ["python", "bench/scripts/lcb_vs_point.py", "--pool", "tiny"],
        binary,
        floors=floors,
        workers=2,
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_outputs_and_one_replay_per_arm_floor_and_split(
    tmp_path: Path, binary: Path, recorded: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out")

    assert sorted(p.name for p in out.iterdir()) == [
        "compliance.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    assert len(recorded) == len(set(recorded)) == 2 * 2 * len(FLOORS)  # splits x arms x floors
    point_calls = [c for c in recorded if "--floor-on" in c[2]]
    assert len(point_calls) == len(FLOORS) * 2  # the point arm, on both splits
    assert all(c[2][c[2].index("--floor-on") + 1] == "point" for c in point_calls)
    assert all(c[1] == "learned" and "min_cost" in c[2] for c in recorded)

    results = results_of(out)
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "lcb_vs_point"
    assert results["floors"] == list(FLOORS) and results["arms"] == ["lcb", "point"]
    assert results["n_items"] == {"test": 80, "calib": 80}
    assert results["policy_versions"]["point"].endswith("+floor-point")
    assert not results["policy_versions"]["lcb"].endswith("+floor-point")
    for split in ("test", "calib"):
        for arm in ("lcb", "point"):
            arm_result = results["results"][split][arm]
            assert [r["floor"] for r in arm_result["floors"]] == list(FLOORS)
            assert arm_result["summary"]["floors"] == 3
            for r in arm_result["floors"]:
                assert r["n"] == 80
                assert r["n_not_relaxed"] == round(r["not_relaxed_share"] * 80)
                assert r["not_relaxed_share"] + r["relaxed_share"] == pytest.approx(1.0)
                assert r["mean_cost_usd"] > 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "the floors are fixed in advance" in report
    assert "Test split: floor compliance and cost per floor" in report
    assert "lcb: realized success [95% CI]" in report and "point: >= floor" in report
    assert "no latency is measured or simulated" in report


def test_the_arms_differ_only_in_what_the_floor_is_compared_with(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    results = results_of(run(derived, binary, tmp_path / "out", floors=(0.0, 0.5, 0.9)))

    # at floor 0 nothing is relaxed whichever number is read, so the arms agree exactly
    for split in ("test", "calib"):
        lcb0, point0 = (results["results"][split][arm]["floors"][0] for arm in ("lcb", "point"))
        assert lcb0["relaxed_share"] == point0["relaxed_share"] == 0.0
        assert lcb0["mean_cost_usd"] == point0["mean_cost_usd"]
        assert lcb0["success_rate"] == point0["success_rate"]
        assert lcb0["model_counts"] == point0["model_counts"]
    # at a high floor the lower bounds are out of reach for more decisions than the P values
    lcb_hi, point_hi = (results["results"]["test"][arm]["floors"][2] for arm in ("lcb", "point"))
    assert lcb_hi["relaxed_share"] >= point_hi["relaxed_share"]


def test_no_prompt_text_in_the_outputs(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out")
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_main_uses_the_pool_artifact_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = load_script()
    seen = {}

    def fake(derived, pool, artifact_dir, policy_config, out_root, command, **kwargs):
        seen.update(pool=pool, artifact=artifact_dir, policy=policy_config, **kwargs)
        return tmp_path

    monkeypatch.setattr(script, "run_lcb_vs_point", fake)
    monkeypatch.setattr(script.sweeps, "find_artifact", lambda root, pool: tmp_path / "v1")

    assert script.main(["--pool", "flagship", "--workers", "3"]) == 0
    assert seen["pool"] == "flagship" and seen["artifact"] == tmp_path / "v1"
    assert seen["policy"].name == "flagship.yaml" and seen["workers"] == 3
