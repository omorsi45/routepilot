"""Tests for the headline script: targets, calibration quality, split discipline and outputs."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.figure import Figure

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root
from routepilot_train.train import crossfit

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "headline.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("headline", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- targets ------------------------------------------------------------------------------------


def row(cost_reduction, success_delta) -> dict:
    return {
        "cost_reduction_vs_strongest": cost_reduction,
        "success_delta_vs_strongest": success_delta,
    }


def test_target_check_is_25_percent_cost_and_success_within_one_point():
    h = load_script()
    assert (h.COST_TARGET, h.SUCCESS_MARGIN) == (0.25, 0.01)

    met = h.target_check(row(0.30, -0.005))
    assert met["cost_target_met"] and met["success_target_met"] and met["both_met"]

    edge = h.target_check(row(0.25, -0.01))  # both exactly on the line: met
    assert edge["both_met"] and edge["success_abs_within_margin"]

    cheap_but_worse = h.target_check(row(0.93, -0.0455))
    assert cheap_but_worse["cost_target_met"] and not cheap_but_worse["success_target_met"]
    assert not cheap_but_worse["both_met"]

    better_but_dear = h.target_check(row(0.10, 0.03))
    assert not better_but_dear["cost_target_met"] and better_but_dear["success_target_met"]
    assert not better_but_dear["success_abs_within_margin"]  # +3 points is not within 1 point
    assert not better_but_dear["both_met"]

    assert not h.target_check(row(None, 0.0))["cost_target_met"]  # nothing to reduce


def test_target_statement_says_plainly_whether_the_targets_were_met():
    h = load_script()
    missed = h.target_statement("learned dual", h.target_check(row(0.935, -0.0455)))
    assert "learned dual" in missed and "93.5%" in missed and "-4.55 points" in missed
    assert "meets the 25% target" in missed and "outside the 1 point target" in missed
    assert missed.endswith("targets NOT met.")
    hit = h.target_statement("x", h.target_check(row(0.4, 0.002)))
    assert "within the 1 point target" in hit and hit.endswith("targets MET.")
    assert "n/a" in h.target_statement("x", h.target_check(row(None, 0.0)))


# --- calibration quality --------------------------------------------------------------------------


def test_calibration_quality_per_model_and_pooled():
    h = load_script()
    y = {"a": np.array([1] * 15 + [0] * 15), "b": np.array([1, 0] * 15)}
    p = {"a": y["a"].astype(np.float64), "b": np.full(30, 0.5)}

    q = h.calibration_quality(p, y)

    assert set(q["per_model"]) == {"a", "b"}
    a, b = q["per_model"]["a"], q["per_model"]["b"]
    assert a["ece_mass"] == a["ece_width"] == a["brier"] == 0.0 and a["n"] == 30
    assert b["ece_mass"] == 0.0 and b["ece_width"] == 0.0  # every bin holds equal 0s and 1s
    assert b["brier"] == pytest.approx(0.25) and b["base_rate"] == 0.5
    assert b["log_loss"] == pytest.approx(np.log(2.0))
    pooled = q["pooled"]
    assert pooled["n"] == 60 and pooled["base_rate"] == 0.5
    assert pooled["brier"] == pytest.approx(0.125)  # the mean of 0 and 0.25
    for key in ("ece_mass", "ece_width", "brier", "log_loss", "base_rate", "mean_p", "n"):
        assert key in a


def test_calibration_quality_sees_overconfidence():
    h = load_script()
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 4000)
    y = (rng.random(4000) < 0.5 + 0.2 * (p - 0.5)).astype(np.int64)  # forecasts too extreme
    q = h.calibration_quality({"m": p}, {"m": y})
    assert q["per_model"]["m"]["ece_mass"] > 0.05
    assert q["pooled"]["ece_mass"] == q["per_model"]["m"]["ece_mass"]  # one model: same data


def test_reliability_figure_has_one_visible_panel_per_model():
    h = load_script()
    rng = np.random.default_rng(1)
    names = [f"m{i}" for i in range(12)]
    p = {m: rng.uniform(0, 1, 200) for m in names}
    y = {m: (rng.random(200) < p[m]).astype(np.int64) for m in names}

    fig = h.reliability_figure(p, y, h.calibration_quality(p, y), "title")

    assert isinstance(fig, Figure)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 12
    assert {ax.get_title().splitlines()[0] for ax in visible} == set(names)
    assert all("ECE" in ax.get_title() for ax in visible)


def test_go_parity_compares_the_chosen_model_and_skips_ood_and_errors():
    h = load_script()
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "model": ["m1", "m2", "m1", None],
            "p": [0.50, 0.70, 0.30, None],
            "ood": [False, False, True, False],
        }
    )
    ids = ["i1", "i2", "i3", "i4"]
    py = {"m1": np.array([0.5, 0.0, 0.9, 0.0]), "m2": np.array([0.0, 0.7 + 3e-9, 0.0, 0.0])}

    got = h.go_parity(frame, ids, py)

    assert got["items_compared"] == 2 and got["items_skipped"] == 2
    assert got["max_abs_diff"] == pytest.approx(3e-9, abs=1e-12)
    empty = h.go_parity(frame.iloc[2:], ids, py)
    assert empty["items_compared"] == 0 and empty["max_abs_diff"] is None


def test_per_dataset_table_compares_each_policy_with_the_reference():
    h = load_script()

    def frame(success, cost):
        return pd.DataFrame(
            {
                "item_id": [f"i{k}" for k in range(len(success))],
                "dataset": ["a", "a", "b", "b"],
                "success": success,
                "cost_picousd": cost,
                "error": "",
            }
        )

    ref = frame([True, True, True, False], [1000, 1000, 2000, 2000])
    cheap = frame([True, False, True, False], [100, 100, 1000, 1000])
    table = h.per_dataset_table({"always_strongest": ref, "learned_dual": cheap})

    assert list(table["dataset"]) == ["a", "b"] and list(table["n"]) == [2, 2]
    a = table.set_index("dataset").loc["a"]
    assert a["always_strongest success"] == 1.0 and a["learned_dual success"] == 0.5
    assert a["learned_dual success delta"] == pytest.approx(-0.5)
    assert a["learned_dual cost reduction"] == pytest.approx(0.9)
    b = table.set_index("dataset").loc["b"]
    assert b["learned_dual cost reduction"] == pytest.approx(0.5)


# --- main wiring ----------------------------------------------------------------------------------


def test_main_defaults_follow_the_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    h = load_script()
    seen: list[dict] = []

    def fake(derived, pool, artifact, policy_config, mf, out_root, command, **kwargs):
        seen.append({"pool": pool, "mf": mf, **kwargs})
        return tmp_path

    monkeypatch.setattr(h, "run_headline", fake)
    monkeypatch.setattr(h.sweeps, "find_artifact", lambda root, pool: tmp_path / pool)

    assert h.main(["--pool", "flagship"]) == 0
    assert h.main(["--pool", "mixed"]) == 0
    assert h.main(["--pool", "mixed", "--lambda-source", "config", "--crossfit"]) == 0
    assert h.main(["--pool", "flagship", "--no-crossfit"]) == 0

    flagship, mixed, forced, off = seen
    assert flagship["lambda_source"] == "config" and flagship["do_crossfit"] is True
    assert mixed["lambda_source"] == "tune" and mixed["do_crossfit"] is False
    assert forced["lambda_source"] == "config" and forced["do_crossfit"] is True
    assert off["do_crossfit"] is False
    assert mixed["mf"].name == "mixed.json"


# --- the whole script on the synthetic pool -------------------------------------------------------


def policy_file(tmp_path: Path, lam: float = 0.0) -> Path:
    path = tmp_path / "policy.yaml"
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: min_cost", "mode: dual").replace("lambda: 0", f"lambda: {lam}")
    path.write_text(text, encoding="utf-8")
    return path


def run(derived: Path, binary: Path, out: Path, policy: Path, **kwargs) -> Path:
    h = load_script()
    options = {
        "lambda_source": "tune",
        "bootstrap": 40,
        "n_thresholds": 5,
        "floors": [0.0, 0.3, 0.6, 0.9],
        "iters": 10,
        "workers": 2,
    }
    return h.run_headline(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy,
        derived / "mf" / "tiny.json",
        out,
        ["python", "bench/scripts/headline.py", "--pool", "tiny"],
        binary,
        **{**options, **kwargs},
    )


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


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_outputs_operating_points_and_the_crossfit_row(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path))

    assert sorted(p.name for p in out.iterdir()) == [
        "pareto_test.png",
        "reliability_test.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    results = results_of(out)
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "headline" and run_json["policy_version"] == "0782d801a5a3"
    assert results["n_items"] == {"calib": 80, "test": 80}

    target = results["target"]
    assert target["best_single"] == "tiny-strong"
    assert target["target"] == pytest.approx(target["best_single_calib_success"] - 0.01)
    assert results["lambda"]["source"] == "tune"
    assert results["lambda"]["calib_success"] >= target["target"]

    ops = results["operating_points"]
    assert set(ops) == {
        "dual",
        "min_cost",
        "routellm_mf",
        "dual_lambda0_reference",
        "crossfit_dual",
    }
    assert ops["dual"]["param"] == results["lambda"]["value"]
    assert ops["dual_lambda0_reference"]["param"] == 0.0
    assert ops["min_cost"]["param"] in (0.0, 0.3, 0.6, 0.9)
    for op in ops.values():
        r = op["test_row"]
        assert r["n"] == 80 and len(r["success_ci"]) == 2 and len(r["success_delta_ci"]) == 2
        assert set(op["targets"]) >= {"cost_target_met", "success_target_met", "both_met"}
    assert set(results["baselines"]) == {
        "always_strongest",
        "always_cheapest",
        "random",
        "static_rules",
        "oracle",
        "static_strongest_rank",
        "static_cheapest_worstcase",
    }
    assert results["baselines"]["always_strongest"]["success_delta_vs_strongest"] == 0.0

    xf = results["crossfit"]
    assert xf["status"] == "tuned" and xf["n_a"] + xf["n_b"] == 80
    assert xf["salt"] == "routepilot-crossfit-v1"
    assert xf["artifact_version"] != results["artifact_version"]
    assert xf["success_b_at_lambda"] >= xf["target_b"]
    assert xf["calibrator_types"] == {"platt": 2}  # half of 80 items: the spec rule is Platt
    assert ops["crossfit_dual"]["param"] == xf["lambda"]

    quality = results["calibration_quality"]
    assert set(quality["per_model"]) == {"tiny-weak", "tiny-strong"}
    assert quality["pooled"]["n"] == 160
    assert results["python_go_parity"]["items_compared"] > 0
    assert results["python_go_parity"]["max_abs_diff"] < 1e-9

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "targets NOT met" in report or "targets MET" in report
    assert "Cross-fitted sensitivity" in report and "lambda = 0 dual row is a reference" in report
    assert "no latency is measured or simulated" in report
    assert "ECE equal-mass" in report and "ECE equal-width" in report

    # no prompt text in any output file
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_the_test_split_is_replayed_once_per_point_and_after_every_choice(
    tmp_path: Path, binary: Path, recorded: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    run(derived, binary, tmp_path / "out", policy_file(tmp_path))

    test_calls = [c for c in recorded if c[0] == "test"]
    assert len(test_calls) == len(set(test_calls)), "a test point was replayed twice"
    # 7 baselines, dual, min_cost, MF, lambda 0, cross-fitted dual
    assert len(test_calls) == 12, test_calls
    first_test = next(i for i, c in enumerate(recorded) if c[0] == "test")
    assert all(c[0] == "calib" for c in recorded[:first_test])
    assert all(c[0] == "test" for c in recorded[first_test:])


def test_operating_points_ignore_the_test_split(tmp_path: Path, binary: Path):
    first = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "a" / "derived"),
            binary,
            tmp_path / "a" / "out",
            policy_file(tmp_path / "a"),
        )
    )
    flipped = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "b" / "derived", flip_test=True),
            binary,
            tmp_path / "b" / "out",
            policy_file(tmp_path / "b"),
        )
    )

    assert first["lambda"]["value"] == flipped["lambda"]["value"]
    assert first["crossfit"]["lambda"] == flipped["crossfit"]["lambda"]
    assert first["crossfit"]["artifact_version"] == flipped["crossfit"]["artifact_version"]
    for key in ("dual", "min_cost", "routellm_mf", "crossfit_dual"):
        assert first["operating_points"][key]["param"] == flipped["operating_points"][key]["param"]
    # while the numbers read from the test split moved: every test outcome succeeds
    for key in ("dual", "min_cost", "routellm_mf", "crossfit_dual"):
        after = flipped["operating_points"][key]["test_row"]
        before = first["operating_points"][key]["test_row"]
        assert after["success_rate"] == 1.0
        assert after["mean_cost_usd"] == pytest.approx(before["mean_cost_usd"] + 1000 / 1e12)


def test_the_crossfit_artifact_never_sees_the_outcomes_of_half_b(tmp_path: Path, binary: Path):
    base = fixture_pool.make_pool(tmp_path / "a" / "derived")
    other = fixture_pool.make_pool(tmp_path / "b" / "derived")
    path = other / "replay" / "tiny.jsonl"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["split"] == "calib" and crossfit.half_of(item["item_id"]) == 1:
            for outcome in item["outcomes"].values():
                outcome["success"] = not outcome["success"]
        lines.append(json.dumps(item))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    a = results_of(run(base, binary, tmp_path / "a" / "out", policy_file(tmp_path / "a")))
    b = results_of(run(other, binary, tmp_path / "b" / "out", policy_file(tmp_path / "b")))

    assert a["crossfit"]["artifact_version"] == b["crossfit"]["artifact_version"]
    assert a["crossfit"]["best_single_b_success"] != b["crossfit"]["best_single_b_success"]


def test_without_crossfit_there_is_no_cross_fitted_row(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path), do_crossfit=False)
    results = results_of(out)
    assert results["crossfit"] is None and "crossfit_dual" not in results["operating_points"]
    assert "Cross-fitted sensitivity" not in (out / "report.md").read_text(encoding="utf-8")


def test_an_unreachable_crossfit_target_is_reported_not_hidden(
    tmp_path: Path, binary: Path, monkeypatch: pytest.MonkeyPatch
):
    h = load_script()

    def unreachable(*args, **kwargs):
        raise ValueError("the success 0.1 at lambda 0.0 is below the target 0.5")

    monkeypatch.setattr(h.dual, "tune_lambda", unreachable)
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path), lambda_source="config")

    results = results_of(out)
    assert results["crossfit"]["status"] == "unreachable"
    assert "below the target" in results["crossfit"]["message"]
    assert "crossfit_dual" not in results["operating_points"]
    assert "unreachable" in (out / "report.md").read_text(encoding="utf-8")


def test_a_config_lambda_that_misses_the_target_stops_before_any_test_replay(
    tmp_path: Path, binary: Path, recorded: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    policy = policy_file(tmp_path, lam=1e9)  # a price of cost that sends everything to tiny-weak

    with pytest.raises(ValueError, match="below the target"):
        run(derived, binary, tmp_path / "out", policy, lambda_source="config")

    assert not any(c[0] == "test" for c in recorded)
    assert not (tmp_path / "out").exists()


def test_the_config_lambda_is_used_as_it_is(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(
        derived, binary, tmp_path / "out", policy_file(tmp_path, 0.25), lambda_source="config"
    )
    results = results_of(out)
    assert results["lambda"] == {**results["lambda"], "source": "config", "value": 0.25}
    assert results["operating_points"]["dual"]["param"] == 0.25
    # the policy config on disk is untouched
    assert yaml.safe_load((tmp_path / "policy.yaml").read_text(encoding="utf-8"))["lambda"] == 0.25


def test_an_mf_file_for_other_models_is_refused(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    mf_path = derived / "mf" / "tiny.json"
    document = json.loads(mf_path.read_text(encoding="utf-8"))
    document["strong"], document["weak"] = document["weak"], document["strong"]
    mf_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Best Single"):
        run(derived, binary, tmp_path / "out", policy_file(tmp_path))


def test_an_unknown_lambda_source_is_refused(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    with pytest.raises(ValueError, match="lambda_source"):
        run(derived, binary, tmp_path / "out", policy_file(tmp_path), lambda_source="guess")
