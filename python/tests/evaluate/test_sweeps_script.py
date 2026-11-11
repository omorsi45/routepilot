"""Tests for the Pareto sweeps script: grids, operating points, split discipline, outputs."""

import base64
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from routepilot_train import features, replaytool
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "sweeps.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = ROOT / "testdata" / "policy" / "tiny" / "0782d801a5a3"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sweeps", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure helpers ----------------------------------------------------------------------------


def test_lambda_grid_is_zero_then_forty_log_spaced_values_from_a_hundredth_to_a_hundred_thousand():
    grid = load_script().lambda_grid()
    assert len(grid) == 41 and grid[0] == 0.0
    assert grid[1] == pytest.approx(1e-2) and grid[-1] == pytest.approx(1e5)
    ratios = np.array(grid[2:]) / np.array(grid[1:-1])
    assert np.allclose(ratios, ratios[0])  # log spaced
    assert len(load_script().lambda_grid(3)) == 4


def test_floor_grid_is_zero_to_ninety_five_hundredths_in_steps_of_five():
    floors = load_script().floor_grid()
    assert floors == [round(0.05 * i, 2) for i in range(20)]
    assert floors[0] == 0.0 and floors[-1] == 0.95


def test_quantile_thresholds_sit_at_evenly_spaced_quantiles_of_the_scores():
    scores = np.linspace(0.1, 0.9, 101)
    thresholds = load_script().quantile_thresholds(scores, 41)
    assert len(thresholds) == 41
    assert thresholds[0] == pytest.approx(0.1) and thresholds[-1] == pytest.approx(0.9)
    assert thresholds[20] == pytest.approx(0.5)
    assert thresholds == sorted(thresholds)
    assert all(0.0 <= t <= 1.0 for t in thresholds)
    with pytest.raises(ValueError, match="no scores"):
        load_script().quantile_thresholds(np.array([]), 41)
    with pytest.raises(ValueError, match="finite"):
        load_script().quantile_thresholds(np.array([0.2, float("nan")]), 41)


def point(param: float, success: float, cost: float) -> dict:
    return {"param": param, "success_rate": success, "mean_cost_usd": cost}


def test_pick_floor_takes_the_calibration_success_closest_to_the_target():
    sweeps = load_script()
    points = [point(0.0, 0.60, 5.0), point(0.5, 0.52, 3.0), point(0.9, 0.40, 1.0)]
    assert sweeps.pick_floor(points, 0.50)["param"] == 0.5
    assert sweeps.pick_floor(points, 0.58)["param"] == 0.0
    # a tie in distance goes to the cheaper point, then to the lower floor
    assert sweeps.pick_floor([point(0.1, 0.75, 4.0), point(0.2, 0.25, 2.0)], 0.5)["param"] == 0.2
    assert sweeps.pick_floor([point(0.3, 0.75, 2.0), point(0.1, 0.25, 2.0)], 0.5)["param"] == 0.1
    with pytest.raises(ValueError, match="no points"):
        sweeps.pick_floor([], 0.5)


def test_pick_threshold_takes_the_cheapest_point_that_meets_the_target():
    sweeps = load_script()
    points = [
        point(0.2, 0.60, 5.0),
        point(0.5, 0.52, 3.0),
        point(0.7, 0.49, 2.0),  # cheaper but below the target
        point(0.9, 0.52, 3.0),  # same cost as 0.5: the higher threshold wins the tie
    ]
    assert sweeps.pick_threshold(points, 0.50)["param"] == 0.9
    assert sweeps.pick_threshold(points, 0.60)["param"] == 0.2
    assert sweeps.pick_threshold(points, 0.99) is None


def test_set_lambda_rewrites_only_the_lambda_line(tmp_path: Path):
    sweeps = load_script()
    path = tmp_path / "policy.yaml"
    path.write_text(
        "# lambda stays in this comment\nmode: dual\nlambda: 0\ndefault_floor: 0\n"
        "ood_action: conservative\n",
        encoding="utf-8",
    )

    sweeps.set_lambda(path, 1234.5678901234567)

    text = path.read_text(encoding="utf-8")
    assert "lambda: 1234.5678901234567\n" in text
    assert text.startswith("# lambda stays in this comment\nmode: dual\n")
    assert "default_floor: 0\n" in text and "\r" not in text
    # an exponent without a dot is not a float to YAML 1.1 readers; the file keeps a dot
    sweeps.set_lambda(path, 1e-05)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["lambda"] == 1e-05
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8"))["lambda"], float)
    with pytest.raises(ValueError, match="finite"):
        sweeps.set_lambda(path, float("inf"))
    path.write_text("mode: dual\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lambda"):
        sweeps.set_lambda(path, 1.0)


def test_find_artifact_wants_exactly_one_version_directory(tmp_path: Path):
    sweeps = load_script()
    root = tmp_path / "policies"
    with pytest.raises(ValueError, match="no artifact"):
        sweeps.find_artifact(root, "flagship")
    (root / "flagship" / "aaa111").mkdir(parents=True)
    assert sweeps.find_artifact(root, "flagship").name == "aaa111"
    (root / "flagship" / "bbb222").mkdir()
    with pytest.raises(ValueError, match="more than one"):
        sweeps.find_artifact(root, "flagship")


# --- the whole script on the learned-policy fixture -----------------------------------------

DIM = 8


def write_mf(path: Path) -> None:
    """strong tiny-strong, weak tiny-weak; the score grows with the text length."""
    u = np.zeros(DIM + features.NUM_DENSE, dtype="<f4")
    u[DIM] = 1.0
    mean = [0.0] * features.NUM_DENSE
    mean[0] = math.log1p(30)
    std = [1.0] * features.NUM_DENSE
    std[0] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "strong": "tiny-strong",
        "weak": "tiny-weak",
        "dim": DIM,
        "dense_mean": mean,
        "dense_std": std,
        "u": base64.b64encode(u.tobytes()).decode("ascii"),
    }
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


def make_derived(root: Path, flip_test: bool = False) -> Path:
    """A derived directory around the learned fixture: tiny-strong is Best Single.

    With ``flip_test`` every test outcome succeeds and costs 1000 picodollars more; the
    calibration and train rows are untouched.
    """
    (root / "pools").mkdir(parents=True)
    (root / "replay").mkdir()
    config = (LEARNED / "config.yaml").read_text(encoding="utf-8")
    config += (
        "\nrouting:\n  policy: rules\n  default_model: tiny-strong\n  rules:\n"
        '    - name: code\n      model: tiny-weak\n      when:\n        contains_any: ["def "]\n'
    )
    (root / "pools" / "tiny.config.yaml").write_text(config, encoding="utf-8")
    pool = {
        "pool": "tiny",
        "models": ["tiny-weak", "tiny-strong"],
        "best_single": "tiny-strong",
        "cheapest": "tiny-weak",
    }
    (root / "pools" / "tiny.json").write_text(json.dumps(pool), encoding="utf-8")
    lines = []
    for line in (LEARNED / "items.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if flip_test and item["split"] == "test":
            for outcome in item["outcomes"].values():
                outcome["success"] = True
                outcome["cost_picousd"] += 1000
        lines.append(json.dumps(item))
    (root / "replay" / "tiny.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_mf(root / "mf" / "tiny.json")
    return root


def run(derived: Path, binary: Path, out: Path, calls: list | None = None) -> Path:
    sweeps = load_script()
    policy = out.parent / "policy.yaml"
    # a lambda no tuning would pick, so that the write-back of the tuned value shows
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    policy.write_text(text.replace("lambda: 0", "lambda: 12345.5"), encoding="utf-8")
    return sweeps.run_sweeps(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy,
        derived / "mf" / "tiny.json",
        out,
        ["python", "bench/scripts/sweeps.py", "--pool", "tiny"],
        binary,
        bootstrap=50,
        n_lambda=3,
        n_thresholds=5,
        floors=[0.0, 0.5, 0.9],
        iters=6,
        workers=2,
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


def test_outputs_and_the_tuned_lambda(tmp_path: Path, binary: Path, recorded: list[tuple]):
    derived = make_derived(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out")

    assert sorted(p.name for p in out.iterdir()) == [
        "pareto_calib.png",
        "pareto_test.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "sweeps" and run_json["policy_version"] == "0782d801a5a3"

    # Best Single tiny-strong succeeded on one of the two calibration items
    target = results["target"]
    assert target["best_single"] == "tiny-strong"
    assert target["best_single_calib_success"] == 0.5
    assert target["target"] == pytest.approx(0.49)

    tuning = results["lambda_tuning"]
    assert tuning["status"] == "tuned" and 0.0 <= tuning["lambda"] <= 1e6
    assert tuning["calib_success"] >= target["target"]
    written = yaml.safe_load((tmp_path / "policy.yaml").read_text(encoding="utf-8"))
    assert written["lambda"] == tuning["lambda"] != 12345.5  # the tuned value, full precision
    assert written["mode"] == "min_cost"  # the rest of the file is as it was

    sweeps = results["sweeps"]
    assert [p["param"] for p in sweeps["lambda"]] == [0.0, *np.logspace(-2, 5, 3).tolist()]
    assert [p["param"] for p in sweeps["floor"]] == [0.0, 0.5, 0.9]
    assert len(sweeps["threshold"]) == 5
    for kind in sweeps.values():
        for p in kind:
            assert p["calib"]["n"] == 2 and p["test"]["n"] == 4
            assert 0.0 <= p["calib"]["relaxed_share"] <= 1.0

    operating = results["operating_points"]
    assert set(operating) == {"always_strongest", "dual", "min_cost", "routellm_mf"}
    assert operating["dual"]["param"] == tuning["lambda"]
    assert operating["min_cost"]["param"] in (0.0, 0.5, 0.9)
    for name in ("dual", "min_cost", "routellm_mf"):
        row = operating[name]["test_row"]
        assert row["n"] == 4 and len(row["success_ci"]) == 2
        assert len(row["success_delta_ci"]) == 2

    baselines = results["baselines"]
    assert "always_strongest" in baselines["calib"] and "oracle" in baselines["test"]

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "test, for reference; operating points chosen on calibration" in report
    assert "no latency is measured or simulated" in report

    # no prompt text in any output file
    items = [
        json.loads(line)
        for line in (LEARNED / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    texts = [m["content"] for item in items for m in item["messages"]]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts), name


def test_sweep_points_equal_direct_replays(tmp_path: Path, binary: Path):
    derived = make_derived(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    config = derived / "pools" / "tiny.config.yaml"
    replay_input = derived / "replay" / "tiny.jsonl"
    policy = LEARNED / "policy.yaml"

    def direct(split: str, mode_flags: list[str]) -> pd.DataFrame:
        flags = ["--artifact", str(TINY_ARTIFACT), "--policy-config", str(policy), *mode_flags]
        return replaytool.replay(
            binary, config, replay_input, split, "learned", tmp_path / "direct.jsonl", flags
        )

    zero = results["sweeps"]["lambda"][0]
    for split in ("calib", "test"):
        frame = direct(split, ["--mode", "dual", "--lambda", "0"])
        assert zero[split]["success_rate"] == pytest.approx(frame["success"].mean())
        assert zero[split]["mean_cost_usd"] == pytest.approx(frame["cost_picousd"].mean() / 1e12)
    half = results["sweeps"]["floor"][1]
    frame = direct("test", ["--mode", "min_cost", "--floor", "0.5"])
    assert half["test"]["success_rate"] == pytest.approx(frame["success"].mean())
    relaxed = frame["relaxed"].map(lambda v: len(v) > 0).mean()
    assert half["test"]["relaxed_share"] == pytest.approx(relaxed)


def test_operating_points_ignore_the_test_split(tmp_path: Path, binary: Path):
    first = json.loads(
        (
            run(make_derived(tmp_path / "a" / "derived"), binary, tmp_path / "a" / "out")
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    flipped = json.loads(
        (
            run(
                make_derived(tmp_path / "b" / "derived", flip_test=True),
                binary,
                tmp_path / "b" / "out",
            )
            / "results.json"
        ).read_text(encoding="utf-8")
    )

    # Same calibration data, different test outcomes: the chosen points are identical ...
    assert first["lambda_tuning"]["lambda"] == flipped["lambda_tuning"]["lambda"]
    for name in ("dual", "min_cost", "routellm_mf"):
        assert (
            first["operating_points"][name]["param"] == flipped["operating_points"][name]["param"]
        )
        assert (
            first["operating_points"][name]["calib"] == flipped["operating_points"][name]["calib"]
        )
    # ... while the numbers read from the test split moved: every test outcome succeeds and
    # costs 1000 picodollars more, whatever model a policy picks.
    for name in ("dual", "min_cost", "routellm_mf"):
        before = first["operating_points"][name]["test_row"]
        after = flipped["operating_points"][name]["test_row"]
        assert after["success_rate"] == 1.0
        assert after["mean_cost_usd"] == pytest.approx(before["mean_cost_usd"] + 1000 / 1e12)


def test_each_test_point_is_replayed_once(tmp_path: Path, binary: Path, recorded: list[tuple]):
    out = run(make_derived(tmp_path / "derived"), binary, tmp_path / "out")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))

    test_calls = [c for c in recorded if c[0] == "test"]
    assert len(test_calls) == len(set(test_calls)), "a test point was replayed twice"
    # 4 lambdas + the tuned one + 3 floors + 5 thresholds + 7 baselines (the first is
    # always_strongest, the reference); a repeated parameter would be replayed once
    assert 4 + 1 + 3 + 5 + 7 - 3 <= len(test_calls) <= 4 + 1 + 3 + 5 + 7, test_calls
    # nothing but calibration replays happened before the operating points were fixed
    first_test = next(i for i, c in enumerate(recorded) if c[0] == "test")
    assert all(c[0] == "calib" for c in recorded[:first_test])
    assert results["operating_points"]["dual"]["test_row"]["n"] == 4


def test_an_unreachable_target_raises_before_any_test_replay(
    tmp_path: Path, binary: Path, recorded: list[tuple], monkeypatch: pytest.MonkeyPatch
):
    sweeps = load_script()

    def unreachable(*args, **kwargs):
        raise ValueError("the success 0.1 at lambda 0.0 is below the target 0.5")

    monkeypatch.setattr(sweeps.dual, "tune_lambda", unreachable)
    derived = make_derived(tmp_path / "derived")
    policy = tmp_path / "policy.yaml"
    shutil.copy(LEARNED / "policy.yaml", policy)
    before = policy.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="below the target"):
        sweeps.run_sweeps(
            derived,
            "tiny",
            TINY_ARTIFACT,
            policy,
            derived / "mf" / "tiny.json",
            tmp_path / "out",
            ["x"],
            binary,
            bootstrap=20,
            n_lambda=2,
            n_thresholds=3,
            floors=[0.0],
            iters=2,
            workers=1,
        )

    assert not any(c[0] == "test" for c in recorded)
    assert policy.read_text(encoding="utf-8") == before
    assert not (tmp_path / "out").exists()


def test_a_mf_file_for_other_models_is_refused(tmp_path: Path, binary: Path):
    derived = make_derived(tmp_path / "derived")
    mf_path = derived / "mf" / "tiny.json"
    document = json.loads(mf_path.read_text(encoding="utf-8"))
    document["strong"] = "tiny-weak"
    document["weak"] = "tiny-strong"
    mf_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Best Single"):
        run(derived, binary, tmp_path / "out")
