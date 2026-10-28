"""Tests for the distribution-shift and OOD experiment script."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from routepilot_train import replaytool
from routepilot_train.export import artifact as artifact_io
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "shift_ood.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT
LAMBDA = 100000.0


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("shift_ood", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure helpers ---------------------------------------------------------------------------------


def test_the_arms_and_the_floor_are_fixed_in_advance():
    script = load_script()
    assert script.FLOOR == 0.3
    assert script.OOD_ARMS == ("shipped", "off")
    assert script.POLICY_ARMS == ("dual", "min_cost")
    assert (script.TEST, script.HELDOUT) == ("test", "heldout")


def test_arm_flags_use_the_spec_lambda_and_add_ood_off_only_to_the_off_arm():
    script = load_script()
    assert script.arm_flags("dual", "shipped", 6.5, 0.3) == ["--mode", "dual", "--lambda", "6.5"]
    assert script.arm_flags("dual", "off", 6.5, 0.3) == [
        "--mode",
        "dual",
        "--lambda",
        "6.5",
        "--ood",
        "off",
    ]
    assert script.arm_flags("min_cost", "shipped", 6.5, 0.3) == [
        "--mode",
        "min_cost",
        "--floor",
        "0.3",
    ]
    assert script.arm_flags("min_cost", "off", 6.5, 0.3)[-2:] == ["--ood", "off"]
    with pytest.raises(ValueError, match="policy arm"):
        script.arm_flags("median", "shipped", 6.5, 0.3)
    with pytest.raises(ValueError, match="ood arm"):
        script.arm_flags("dual", "maybe", 6.5, 0.3)


def test_auroc_known_values_and_ties():
    script = load_script()
    assert script.auroc([3.0, 4.0], [1.0, 2.0]) == 1.0  # every positive above every negative
    assert script.auroc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert script.auroc([1.0, 1.0], [1.0, 1.0]) == 0.5  # nothing separates them
    # pairs: 1 vs 1 is a tie (half), 2 vs 1 is a win: (0.5 + 1) / 2
    assert script.auroc([1.0, 2.0], [1.0]) == pytest.approx(0.75)


def test_auroc_matches_scikit_learn_on_random_scores_with_ties():
    script = load_script()
    rng = np.random.default_rng(3)
    pos = np.round(rng.normal(0.4, 1.0, 300), 1)  # rounding makes many ties
    neg = np.round(rng.normal(0.0, 1.0, 500), 1)
    expected = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    assert script.auroc(pos, neg) == pytest.approx(expected, abs=1e-12)


def test_auroc_needs_both_groups_and_finite_scores():
    script = load_script()
    with pytest.raises(ValueError, match="positive"):
        script.auroc([], [1.0])
    with pytest.raises(ValueError, match="negative"):
        script.auroc([1.0], [])
    with pytest.raises(ValueError, match="finite"):
        script.auroc([1.0, float("nan")], [0.0])


def test_auroc_interval_is_seeded_and_brackets_the_point_estimate():
    script = load_script()
    rng = np.random.default_rng(5)
    pos, neg = rng.normal(0.5, 1.0, 200), rng.normal(0.0, 1.0, 300)
    point, lo, hi = script.auroc_ci(pos, neg, n_boot=200, seed=1)
    assert point == script.auroc(pos, neg)
    assert lo < point < hi and 0.0 <= lo and hi <= 1.0
    assert script.auroc_ci(pos, neg, n_boot=200, seed=1) == (point, lo, hi)
    assert script.auroc_ci(pos, neg, n_boot=200, seed=2) != (point, lo, hi)
    # a clearly separated pair of groups has a narrow interval near 1
    _, lo2, hi2 = script.auroc_ci(pos + 10.0, neg, n_boot=200, seed=1)
    assert lo2 == hi2 == 1.0


def frame(scores, flags, datasets, split) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"{split}-{i}" for i in range(len(scores))],
            "dataset": datasets,
            "split": split,
            "ood_score": scores,
            "ood": flags,
        }
    )


def test_ood_detection_scores_heldout_as_positives_and_reports_flag_rates():
    script = load_script()
    test = frame([0.1, 0.2, 0.3, 0.9], [False, False, False, True], ["a"] * 4, "test")
    held = frame(
        [0.5, 0.95, 0.8, 0.2],
        [False, True, False, False],
        ["h1", "h1", "h2", "h2"],
        "heldout",
    )

    result = script.ood_detection(test, held, threshold=0.85, n_boot=50, seed=0)

    expected = roc_auc_score(
        np.r_[np.ones(4), np.zeros(4)], np.r_[[0.5, 0.95, 0.8, 0.2], [0.1, 0.2, 0.3, 0.9]]
    )
    assert result["auroc"] == pytest.approx(expected)
    assert result["n_positive"] == 4 and result["n_negative"] == 4
    assert result["threshold"] == 0.85
    assert result["flag_rate_test"] == 0.25 and result["flag_rate_heldout"] == 0.25
    per = {r["dataset"]: r for r in result["per_dataset"]}
    assert set(per) == {"h1", "h2"}
    assert per["h1"]["n"] == 2 and per["h1"]["flag_rate"] == 0.5
    assert per["h1"]["auroc"] == pytest.approx(
        roc_auc_score([1, 1, 0, 0, 0, 0], [0.5, 0.95, 0.1, 0.2, 0.3, 0.9])
    )
    assert per["h2"]["flag_rate"] == 0.0


def test_action_effect_is_shipped_minus_off_on_the_items_both_arms_saw():
    script = load_script()
    ids = ["a", "b", "c", "d"]
    shipped = pd.DataFrame(
        {
            "item_id": ids,
            "model": ["m1", "m1", "m2", "m2"],
            "success": [True, True, False, True],
            "cost_picousd": [100, 100, 300, 300],
        }
    )
    off = pd.DataFrame(
        {
            "item_id": ids,
            "model": ["m1", "m2", "m2", "m2"],
            "success": [True, False, False, True],
            "cost_picousd": [100, 500, 300, 300],
        }
    )

    effect = script._action_effect(shipped, off, bootstrap=100)

    assert effect["n"] == 4 and effect["items_changed"] == 1  # only item b chose differently
    assert effect["success_delta"] == pytest.approx(0.25)  # 3 of 4 against 2 of 4
    assert effect["cost_reduction"] == pytest.approx(1 - 200 / 300)  # shipped is cheaper
    lo, hi = effect["success_delta_ci"]
    assert lo <= 0.25 <= hi
    reverse = script._action_effect(off, shipped, bootstrap=100)
    assert reverse["success_delta"] == pytest.approx(-0.25)


def test_arms_that_disagree_on_the_ood_verdict_or_the_items_are_refused():
    script = load_script()
    base = frame([0.1, 0.9], [False, True], ["h", "h"], "heldout")
    other_score = frame([0.1, 0.8], [False, True], ["h", "h"], "heldout")
    other_flag = frame([0.1, 0.9], [False, False], ["h", "h"], "heldout")
    other_items = base.assign(item_id=["x", "y"])

    script._same_verdict({"a": base, "b": base.copy()})  # the same verdict passes
    for other in (other_score, other_flag, other_items):
        with pytest.raises(ValueError, match="disagree"):
            script._same_verdict({"a": base, "b": other})


def test_ood_detection_refuses_frames_of_the_wrong_split():
    script = load_script()
    test = frame([0.1, 0.2], [False, False], ["a", "a"], "test")
    held = frame([0.5, 0.6], [False, False], ["h", "h"], "heldout")
    with pytest.raises(ValueError, match="splits not allowed"):
        script.ood_detection(held, held, threshold=0.5, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="splits not allowed"):
        script.ood_detection(test, test, threshold=0.5, n_boot=10, seed=0)


# --- the whole script on the synthetic pool -------------------------------------------------------


def add_heldout(derived: Path, per_dataset: int = 15, seed: int = 11) -> None:
    """Append held-out items of two invented datasets (digit strings, unlike the other splits)."""
    rng = np.random.default_rng(seed)
    path = derived / "replay" / "tiny.jsonl"
    lines = []
    for dataset in ("syn-h1", "syn-h2"):
        for i in range(per_dataset):
            prompt = " ".join(
                str(int(rng.integers(0, 10**6))) for _ in range(int(rng.integers(20, 90)))
            )
            outcomes = {
                model: {
                    "success": bool(rng.random() < (0.35, 0.6)[m]),
                    "cost_picousd": int(rng.integers(1, 4)) * (10**6, 5 * 10**6)[m],
                    "prompt_tokens": 10,
                    "completion_tokens": int(rng.integers(5, 80)),
                }
                for m, model in enumerate(fixture_pool.MODELS)
            }
            lines.append(
                json.dumps(
                    {
                        "item_id": f"{dataset}/{i}",
                        "dataset": dataset,
                        "split": "heldout",
                        "prompt_sha256": f"{900 + len(lines):064x}",
                        "messages": [{"role": "user", "content": prompt}],
                        "outcomes": outcomes,
                    }
                )
            )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


@pytest.fixture
def policy_config(tmp_path: Path) -> Path:
    """The fixture policy config as a dual policy with a lambda that trades quality for cost."""
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: min_cost", "mode: dual").replace("lambda: 0", f"lambda: {LAMBDA}")
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


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


def run(derived: Path, binary: Path, out: Path, policy_config: Path, bootstrap: int = 50) -> Path:
    script = load_script()
    return script.run_shift_ood(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy_config,
        out,
        ["python", "bench/scripts/shift_ood.py", "--pool", "tiny"],
        binary,
        bootstrap=bootstrap,
        workers=2,
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    root = fixture_pool.make_pool(tmp_path / "derived")
    add_heldout(root)
    return root


def test_outputs_and_the_splits_that_are_replayed(
    derived: Path, binary: Path, recorded: list[tuple], policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)

    assert sorted(p.name for p in out.iterdir()) == [
        "ood_scores.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    assert len(recorded) == len(set(recorded))  # no replay twice
    by_split: dict[str, list[tuple]] = {}
    for call in recorded:
        by_split.setdefault(call[0], []).append(call)
    assert set(by_split) == {"heldout", "test"}  # train and calibration are never replayed
    assert len(by_split["test"]) == 1  # the test split is read once, for the OOD scores only
    assert by_split["test"][0][1] == "learned" and "--ood" not in by_split["test"][0][2]
    heldout_learned = [c for c in by_split["heldout"] if c[1] == "learned"]
    assert len(heldout_learned) == 4  # two policy arms x two OOD arms
    assert sum("--ood" in c[2] for c in heldout_learned) == 2
    fixed = [c for c in by_split["heldout"] if c[1] == "fixed"]
    assert len(fixed) == 1 and fixed[0][2] == ("--model", "tiny-strong")

    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "shift_ood"
    assert run_json["policy_version"] == TINY_ARTIFACT.name


def test_every_arm_is_scored_on_the_same_heldout_items_against_always_strongest(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    assert results["n_items"] == {"heldout": 30, "test": 80}
    assert results["heldout_datasets"] == ["syn-h1", "syn-h2"]
    strongest = results["reference"]
    # always_strongest, independently from the replay input
    items = [
        json.loads(line)
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    held = [i for i in items if i["split"] == "heldout"]
    wins = sum(i["outcomes"]["tiny-strong"]["success"] for i in held)
    cost = sum(i["outcomes"]["tiny-strong"]["cost_picousd"] for i in held)
    assert strongest["success_rate"] == pytest.approx(wins / 30)
    assert strongest["mean_cost_usd"] == pytest.approx(cost / 30 / 1e12)

    assert set(results["arms"]) == {"dual/shipped", "dual/off", "min_cost/shipped", "min_cost/off"}
    for name, arm in results["arms"].items():
        assert arm["n"] == 30, name
        assert 0.0 <= arm["success_rate"] <= 1.0 and arm["mean_cost_usd"] > 0
        assert arm["success_delta_vs_strongest"] == pytest.approx(
            arm["success_rate"] - strongest["success_rate"]
        )
        lo, hi = arm["success_delta_ci"]
        assert lo <= arm["success_delta_vs_strongest"] <= hi
        assert 0.0 <= arm["ood_flag_rate"] <= 1.0
        assert ("off" in name) == arm["policy_version"].endswith("+ood-off")
    # the verdict is computed the same way whatever the arm does with it
    rates = {arm["ood_flag_rate"] for arm in results["arms"].values()}
    assert len(rates) == 1


def test_min_cost_arms_report_floor_compliance_at_the_fixed_floor(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    assert results["floor"] == 0.3
    for ood_arm in ("shipped", "off"):
        comp = results["arms"][f"min_cost/{ood_arm}"]["compliance"]
        assert comp["floor"] == 0.3 and comp["n"] == 30
        assert comp["n_not_relaxed"] <= 30
        if comp["n_not_relaxed"]:
            assert comp["met"] == (comp["realized_success"] >= 0.3)
    assert "compliance" not in results["arms"]["dual/shipped"]  # dual has no floor to meet


def test_the_action_effect_is_the_paired_difference_between_the_two_ood_arms(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    for policy_arm in ("dual", "min_cost"):
        effect = results["action_effect"][policy_arm]
        shipped, off = (results["arms"][f"{policy_arm}/{a}"] for a in ("shipped", "off"))
        assert effect["success_delta"] == pytest.approx(
            shipped["success_rate"] - off["success_rate"]
        )
        assert effect["n"] == 30
        assert effect["cost_reduction"] == pytest.approx(
            1.0 - shipped["mean_cost_usd"] / off["mean_cost_usd"]
        )


def test_ood_detection_uses_request_level_scores_of_test_and_heldout(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    det = results["ood_detection"]
    assert det["n_positive"] == 30 and det["n_negative"] == 80
    assert 0.0 <= det["auroc"] <= 1.0
    assert det["auroc_ci"][0] <= det["auroc"] <= det["auroc_ci"][1]
    assert det["threshold"] == artifact_io.load_artifact(TINY_ARTIFACT)[0]["ood"]["threshold"]
    assert [r["dataset"] for r in det["per_dataset"]] == ["syn-h1", "syn-h2"]
    assert all(r["n"] == 15 for r in det["per_dataset"])
    assert 0.0 <= det["flag_rate_test"] <= 1.0 and 0.0 <= det["flag_rate_heldout"] <= 1.0


def test_per_dataset_table_covers_each_heldout_dataset_and_arm(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    rows = results["per_dataset"]
    assert {r["dataset"] for r in rows} == {"syn-h1", "syn-h2"}
    for dataset in ("syn-h1", "syn-h2"):
        names = {r["arm"] for r in rows if r["dataset"] == dataset}
        assert names == {"always_strongest", *results["arms"]}
        for r in rows:
            if r["dataset"] == dataset:
                assert r["n"] == 15


def test_the_four_arms_are_checked_for_one_ood_verdict(
    derived: Path,
    binary: Path,
    policy_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script = load_script()
    seen: list[list[str]] = []
    real = script._same_verdict

    def spy(frames):
        seen.append(sorted(frames))
        real(frames)

    monkeypatch.setattr(script, "_same_verdict", spy)
    script.run_shift_ood(
        derived, "tiny", TINY_ARTIFACT, policy_config, tmp_path / "out", None, binary, 20, 2
    )
    assert seen == [["dual/off", "dual/shipped", "min_cost/off", "min_cost/shipped"]]


def test_report_states_what_was_fixed_in_advance_and_that_nothing_is_timed(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)
    report = (out / "report.md").read_text(encoding="utf-8")

    assert "held-out datasets were fixed before training" in report
    assert "OOD threshold comes from calibration" in report
    assert "no latency is measured or simulated" in report
    assert "AUROC" in report and "OOD action effect" in report
    assert "Held-out datasets" in report


def test_no_prompt_text_in_the_outputs(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_a_pool_without_heldout_items_is_an_error(
    binary: Path, policy_config: Path, tmp_path: Path
):
    plain = fixture_pool.make_pool(tmp_path / "plain")  # no held-out split
    with pytest.raises(RuntimeError, match="no items in split"):
        run(plain, binary, tmp_path / "out", policy_config)


def test_main_uses_the_pool_artifact_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = load_script()
    seen = {}

    def fake(derived, pool, artifact_dir, policy_config, out_root, command, **kwargs):
        seen.update(pool=pool, artifact=artifact_dir, policy=policy_config, **kwargs)
        return tmp_path

    monkeypatch.setattr(script, "run_shift_ood", fake)
    monkeypatch.setattr(script.sweeps, "find_artifact", lambda root, pool: tmp_path / "v1")

    assert script.main(["--pool", "flagship", "--workers", "3"]) == 0
    assert seen["pool"] == "flagship" and seen["artifact"] == tmp_path / "v1"
    assert seen["policy"].name == "flagship.yaml" and seen["workers"] == 3
