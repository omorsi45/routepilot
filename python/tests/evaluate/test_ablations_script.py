"""Tests for the ablations script: variants, retuned operating points, split discipline, outputs."""

import importlib.util
import itertools
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pytest

from routepilot_train import replaytool
from routepilot_train.evaluate import calibration, rescore
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root
from routepilot_train.train import calibrate

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "ablations.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT
EXPECTED = [
    # name, features, dim, calibrator, shipped
    ("both-14", "both", 2**14, "switch", False),
    ("both-16", "both", 2**16, "switch", True),
    ("both-18", "both", 2**18, "switch", False),
    ("hashed-16", "hashed", 2**16, "switch", False),
    ("dense-16", "dense", 2**16, "switch", False),
    ("platt-16", "both", 2**16, "platt", False),
    ("isotonic-16", "both", 2**16, "isotonic", False),
]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ablations_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_file(tmp_path: Path, lam: float = 0.0) -> Path:
    path = tmp_path / "policy.yaml"
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: min_cost", "mode: dual").replace("lambda: 0", f"lambda: {lam}")
    path.write_text(text, encoding="utf-8")
    return path


class FakeTrainer:
    """Stands in for ``pipeline.run``: every variant is the tiny artifact with its own version."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, variant, derived, pool_name, scratch, progress=None) -> dict:
        index = len(self.calls)
        self.calls.append((variant.name, variant.features, variant.dim, variant.calibrator))
        manifest, blob = artifact.load_artifact(TINY_ARTIFACT)
        bundle, weights = artifact.bundle_from_artifact(manifest, blob)
        bundle["training"]["created_at"] = f"2026-10-28T00:{index:02d}:00Z"
        out = artifact.export_artifact(
            Path(scratch) / variant.name / "policies", pool_name, bundle, weights
        )
        return {
            "artifact_dir": out,
            "version": out.name,
            "wall_time_s": 1.5 + index,
            "trained": True,
            "wall_time_source": "this run",
        }


def run(derived: Path, binary: Path, out: Path, policy: Path, tmp_path: Path, **kwargs) -> Path:
    a = load_script()
    options = {
        "bootstrap": 40,
        "iters": 8,
        "workers": 2,
        "train": FakeTrainer(),
        "results_root": tmp_path / "no-results",
    }
    return a.run_ablations(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy,
        tmp_path / "scratch",
        out,
        ["python", "bench/scripts/ablations.py", "--pool", "tiny"],
        binary,
        **{**options, **kwargs},
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


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


# --- the variants -----------------------------------------------------------------------------


def test_the_variants_are_the_ones_of_the_brief():
    a = load_script()

    got = [(v.name, v.features, v.dim, v.calibrator, v.shipped) for v in a.VARIANTS]

    assert got == EXPECTED
    assert [v.name for v in a.VARIANTS if v.shipped] == ["both-16"]
    assert len({v.name for v in a.VARIANTS}) == len(a.VARIANTS)
    assert {v.group for v in a.VARIANTS} == {"features", "dimension", "calibrator"}


def test_a_real_training_goes_through_the_pipeline_into_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    a = load_script()
    seen: list[dict] = []

    def fake_run(pool_name, derived, out, dim=65536, results_root=None, **kwargs):
        seen.append(
            {"pool": pool_name, "out": out, "dim": dim, "results_root": results_root, **kwargs}
        )
        made = Path(out) / pool_name / "abcdef012345"
        made.mkdir(parents=True)
        return {"version": "abcdef012345", "artifact_dir": made, "wall_time_s": 12.5}

    monkeypatch.setattr(a.pipeline, "run", fake_run)
    variant = next(v for v in a.VARIANTS if v.name == "dense-16")
    stale = tmp_path / "scratch" / "dense-16" / "policies" / "tiny" / "oldversion000"
    stale.mkdir(parents=True)

    got = a.train_variant(variant, tmp_path / "derived", "tiny", tmp_path / "scratch")

    (call,) = seen
    assert call["pool"] == "tiny" and call["dim"] == 2**16
    assert call["feature_set"] == "dense" and call["calibrator"] == "switch"
    scratch = tmp_path / "scratch" / "dense-16"
    assert Path(call["out"]) == scratch / "policies"
    assert Path(call["results_root"]) == scratch / "results"  # never bench/results
    assert not stale.exists()  # an older run of this variant is wiped, so there is one version
    assert got["version"] == "abcdef012345" and got["wall_time_s"] == 12.5
    assert got["artifact_dir"] == scratch / "policies" / "tiny" / "abcdef012345"
    assert got["trained"] is True


def test_the_shipped_wall_time_comes_from_the_committed_train_run(tmp_path: Path):
    a = load_script()
    for run_id, pool, version, wall in (
        ("20260101T000000Z-aaaaaaa", "flagship", "111111111111", 99.0),
        ("20260102T000000Z-bbbbbbb", "flagship", "dea4486d7575", 134.1),
        ("20260103T000000Z-ccccccc", "mixed", "dea4486d7575", 75.0),
    ):
        d = tmp_path / "train" / run_id
        d.mkdir(parents=True)
        (d / "results.json").write_text(
            json.dumps({"pool": pool, "artifact_version": version, "wall_time_s": wall}),
            encoding="utf-8",
        )

    assert a.shipped_wall_time(tmp_path, "flagship", "dea4486d7575") == (
        134.1,
        "20260102T000000Z-bbbbbbb",
    )
    assert a.shipped_wall_time(tmp_path, "flagship", "ffffffffffff") == (None, None)
    assert a.shipped_wall_time(tmp_path / "missing", "flagship", "dea4486d7575") == (None, None)


# --- the whole script on the synthetic pool ---------------------------------------------------


def test_outputs_and_the_shape_of_the_results(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    trainer = FakeTrainer()
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path), tmp_path, train=trainer)

    assert sorted(p.name for p in out.iterdir()) == ["report.md", "results.json", "run.json"]
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "ablations" and run_json["policy_version"] == "0782d801a5a3"
    r = results_of(out)
    assert r["n_items"] == {"calib": 80, "test": 80}
    target = r["target"]
    assert target["best_single"] == "tiny-strong"
    assert target["target"] == pytest.approx(target["best_single_calib_success"] - 0.01)
    assert [v["name"] for v in r["variants"]] == [e[0] for e in EXPECTED]

    # the shipped configuration is reused, the other six are trained (in order, with the right
    # arguments)
    assert [c[:2] for c in trainer.calls] == [(e[0], e[1]) for e in EXPECTED if not e[4]]
    assert [c[2:] for c in trainer.calls] == [(e[2], e[3]) for e in EXPECTED if not e[4]]
    by_name = {v["name"]: v for v in r["variants"]}
    shipped = by_name["both-16"]
    assert shipped["trained"] is False and shipped["artifact_version"] == "0782d801a5a3"
    for name, v in by_name.items():
        assert "artifact_dir" not in v  # local paths stay out of the committed results
        if name != "both-16":
            assert v["trained"] is True and v["training_wall_time_s"] >= 1.5
            assert v["wall_time_source"] == "this run"
        assert v["weights_bin_bytes"] == (TINY_ARTIFACT / "weights.bin").stat().st_size
        assert v["calibrator_types"] == {"isotonic": 1, "platt": 1}
        assert v["lambda"]["status"] == "tuned"
        assert v["lambda"]["calib_success"] >= target["target"]
        row = v["test"]["row"]
        assert row["n"] == 80 and len(row["success_ci"]) == 2 and len(row["success_delta_ci"]) == 2
        assert v["test"]["vs_shipped"]["n"] == 80
        quality = v["test"]["calibration_quality"]
        assert quality["pooled"]["n"] == 160 and set(quality["per_model"]) == set(
            fixture_pool.MODELS
        )
        assert v["test"]["python_go_parity"]["max_abs_diff"] < 1e-9
    assert shipped["test"]["vs_shipped"]["success_delta"] == 0.0
    assert r["always_strongest"]["success_delta_vs_strongest"] == 0.0
    assert "PyTorch" in r["not_run"] and "Go-servable" in r["not_run"]

    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in (
        "Ablations: test split",
        "Artifacts",
        "Lambda retuned on calibration",
        "Against the shipped configuration",
        "Not run",
    ):
        assert heading in report, heading
    assert "no latency is measured or simulated" in report
    assert "training wall time" in report and "weights.bin" in report
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_the_test_split_is_replayed_once_per_artifact_after_every_calibration_choice(
    tmp_path: Path, binary: Path, recorded: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    run(derived, binary, tmp_path / "out", policy_file(tmp_path), tmp_path)

    test_calls = [c for c in recorded if c[0] == "test"]
    assert len(test_calls) == len(set(test_calls)), "a test point was replayed twice"
    assert len(test_calls) == 7 + 1  # one per artifact, plus always_strongest
    assert sorted(c[1] for c in test_calls) == ["fixed"] + ["learned"] * 7
    first_test = next(i for i, c in enumerate(recorded) if c[0] == "test")
    assert all(c[0] == "calib" for c in recorded[:first_test])
    assert all(c[0] == "test" for c in recorded[first_test:])
    # every artifact was tuned on its own: seven distinct artifact paths among the calibration runs
    paths = {c[2][c[2].index("--artifact") + 1] for c in recorded if c[1] == "learned"}
    assert len(paths) == 7


def test_lambda_is_tuned_per_artifact_on_calibration_only(tmp_path: Path, binary: Path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "a" / "derived"),
            binary,
            tmp_path / "a" / "out",
            policy_file(tmp_path / "a"),
            tmp_path / "a",
        )
    )
    flipped = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "b" / "derived", flip_test=True),
            binary,
            tmp_path / "b" / "out",
            policy_file(tmp_path / "b"),
            tmp_path / "b",
        )
    )

    assert first["target"] == flipped["target"]
    for a, b in zip(first["variants"], flipped["variants"], strict=True):
        assert a["lambda"] == b["lambda"], a["name"]
        # while the numbers read from the test split moved: every test outcome succeeds
        assert b["test"]["row"]["success_rate"] == 1.0
        assert a["test"]["row"]["success_rate"] < 1.0


def test_the_pooled_test_ece_is_that_of_the_artifacts_calibrated_probabilities(
    tmp_path: Path, binary: Path
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    r = results_of(run(derived, binary, tmp_path / "out", policy_file(tmp_path), tmp_path))
    models = list(fixture_pool.MODELS)
    rows = rescore.read_split(derived / "replay" / "tiny.jsonl", "test", models)
    manifest, blob = artifact.load_artifact(TINY_ARTIFACT)
    raw = rescore.quality_scores(manifest, blob, rows.items["prompt"].tolist())
    p = np.concatenate(
        [calibrate.apply_calibrator(m["calibrator"], raw[m["id"]]) for m in manifest["models"]]
    )
    y = np.concatenate([rows.success[m["id"]] for m in manifest["models"]])

    expected = calibration.ece(p, y, 15, "mass")

    for v in r["variants"]:
        assert v["test"]["calibration_quality"]["pooled"]["ece_mass"] == pytest.approx(expected)


def test_an_unreachable_operating_point_is_reported_and_not_replayed_on_test(
    tmp_path: Path, binary: Path, recorded: list[tuple], monkeypatch: pytest.MonkeyPatch
):
    a = load_script()
    real = a.dual.tune_lambda
    counter = itertools.count(1)

    def sometimes(run_calib, target, lo=0.0, hi=1e6, iters=40):
        if next(counter) == 3:  # atomic: the variants are tuned on several threads
            raise ValueError("the success 0.1 at lambda 0.0 is below the target 0.5")
        return real(run_calib, target, lo, hi, iters)

    monkeypatch.setattr(a.dual, "tune_lambda", sometimes)
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path), tmp_path, workers=1)

    r = results_of(out)
    missed = [v for v in r["variants"] if v["lambda"]["status"] == "unreachable"]
    assert len(missed) == 1 and "below the target" in missed[0]["lambda"]["message"]
    assert missed[0]["lambda"]["value"] is None and missed[0]["test"]["row"] is None
    assert missed[0]["test"]["calibration_quality"]["pooled"]["n"] == 160  # still a property
    assert len([c for c in recorded if c[0] == "test" and c[1] == "learned"]) == 6
    assert "unreachable" in (out / "report.md").read_text(encoding="utf-8")


def test_the_comparison_with_the_shipped_configuration_says_plainly_what_changed(
    tmp_path: Path, binary: Path
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    r = results_of(run(derived, binary, tmp_path / "out", policy_file(tmp_path), tmp_path))

    statements = r["statements"]

    assert len(statements) == 6  # every variant but the shipped one
    assert [s.split(":")[0] for s in statements] == [
        v["label"] for v in r["variants"] if v["name"] != "both-16"
    ]
    assert all("pooled test ECE" in s and "shipped" in s for s in statements)
    assert all(any(w in s for w in ("better", "worse", "no clear difference")) for s in statements)


def test_a_statement_names_the_direction_of_every_change():
    a = load_script()
    variant = {"label": "dense features only, dim 2^16"}
    vs = {
        "success_delta": -0.0412,
        "success_delta_ci": [-0.06, -0.02],
        "cost_reduction": 0.123,
        "cost_reduction_ci": [0.05, 0.2],
    }

    text = a.statement(variant, vs, 0.031, 0.016)

    assert text == (
        "dense features only, dim 2^16: test success -4.12 points against the shipped one "
        "(worse), mean cost 12.3% cheaper (better); pooled test ECE 0.0310 against 0.0160 for "
        "the shipped one (higher)."
    )
    dearer = a.statement(
        variant, {**vs, "cost_reduction": -0.05, "cost_reduction_ci": [-0.1, 0.01]}, 0.01, 0.016
    )
    assert "mean cost 5.0% dearer (no clear difference)" in dearer and dearer.endswith("(lower).")
    unknown = a.statement(variant, {**vs, "cost_reduction": None, "cost_reduction_ci": None}, 1, 1)
    assert "mean cost n/a;" in unknown and unknown.endswith("(equal).")
    absent = a.statement(variant, None, 0.03, 0.03)
    assert absent.startswith("dense features only, dim 2^16: no test row") and "(equal)" in absent


def test_the_judgement_words_follow_the_paired_interval():
    a = load_script()
    assert a.judge([0.01, 0.05]) == "better"
    assert a.judge([-0.05, -0.01]) == "worse"
    assert a.judge([-0.01, 0.02]) == "no clear difference"
    assert a.judge([0.0, 0.02]) == "no clear difference"  # an interval touching 0 is not clear
    assert a.judge(None) == "no clear difference"


def test_main_defaults_follow_the_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    a = load_script()
    seen: list[tuple] = []

    def fake(derived, pool, shipped, policy_config, scratch, out_root, command, **kwargs):
        seen.append((pool, shipped, policy_config, scratch, out_root))
        return tmp_path

    monkeypatch.setattr(a, "run_ablations", fake)
    monkeypatch.setattr(a.sweeps, "find_artifact", lambda root, pool: tmp_path / pool)

    assert a.main(["--pool", "flagship"]) == 0

    ((pool, shipped, policy_config, scratch, out_root),) = seen
    assert pool == "flagship" and shipped == tmp_path / "flagship"
    assert policy_config == ROOT / "bench" / "policies" / "flagship.yaml"
    assert scratch == ROOT / "bench" / "scratch" / "ablations"
    assert out_root == ROOT / "bench" / "results" / "ablations"
