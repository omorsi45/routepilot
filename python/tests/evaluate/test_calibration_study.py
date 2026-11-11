"""Tests for the calibration study script: subsampling, cells, aggregation and outputs."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pytest

from routepilot_train.evaluate import calibration, rescore
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root
from routepilot_train.train import calibrate

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "calibration_study.py"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT
MODELS = list(fixture_pool.MODELS)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibration_study", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- subsampling ----------------------------------------------------------------------------------


def test_the_study_covers_the_specified_sizes_methods_and_seeds():
    study = load_script()
    assert study.SIZES == (50, 100, 200, 500, 1000, 2000, "all")
    assert study.METHODS == ("platt", "isotonic", "histogram", "switch")
    assert study.N_SEEDS == 20 and study.ECE_BINS == 15


def test_draw_indices_are_sorted_distinct_and_reproducible():
    study = load_script()
    a = study.draw_indices(1773, 200, 0)

    assert a.shape == (200,) and len(set(a.tolist())) == 200 and (np.diff(a) > 0).all()
    assert 0 <= a.min() and a.max() < 1773
    assert a.tolist() == study.draw_indices(1773, 200, 0).tolist()
    assert a.tolist() != study.draw_indices(1773, 200, 1).tolist()
    assert a.tolist() != study.draw_indices(1773, 500, 0)[:200].tolist()
    assert study.draw_indices(1773, "all", 7).tolist() == list(range(1773))
    assert study.draw_indices(1773, "all", 8).tolist() == list(range(1773))


def test_draw_indices_refuses_a_size_the_split_cannot_supply():
    study = load_script()
    with pytest.raises(ValueError, match="size"):
        study.draw_indices(1773, 2000, 0)
    with pytest.raises(ValueError, match="size"):
        study.draw_indices(1773, 1773, 0)  # the whole split is "all"
    with pytest.raises(ValueError, match="size"):
        study.draw_indices(100, 0, 0)
    with pytest.raises(ValueError, match="size"):
        study.draw_indices(100, "most", 0)


# --- cells and aggregation ------------------------------------------------------------------------


def synthetic(n_cal: int = 300, n_test: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for n in (n_cal, n_test):
        raw = {m: rng.normal(0.0, 2.0, n) for m in ("a", "b")}
        y = {m: (rng.random(n) < 1 / (1 + np.exp(-0.5 * raw[m]))).astype(np.int64) for m in raw}
        out += [raw, y]
    return out


def test_run_cell_fits_each_model_on_the_drawn_rows_and_measures_the_test_ece():
    study = load_script()
    raw_cal, y_cal, raw_test, y_test = synthetic()
    idx = np.arange(0, 300, 3)

    cell = study.run_cell(raw_cal, y_cal, raw_test, y_test, idx, "platt")

    assert set(cell["ece"]) == {"a", "b"} and cell["types"] == {"a": "platt", "b": "platt"}
    forecasts = {}
    for m in ("a", "b"):
        cal = calibrate.fit_calibrator(raw_cal[m][idx], y_cal[m][idx], method="platt")
        forecasts[m] = calibrate.apply_calibrator(cal, raw_test[m])
        assert cell["ece"][m] == pytest.approx(calibration.ece(forecasts[m], y_test[m], 15, "mass"))
    both_p = np.concatenate([forecasts["a"], forecasts["b"]])
    both_y = np.concatenate([y_test["a"], y_test["b"]])
    assert cell["pooled"] == pytest.approx(calibration.ece(both_p, both_y, 15, "mass"))
    assert cell["pooled"] != cell["ece"]["a"] and cell["convergence_warnings"] >= 0


def test_run_cell_records_the_type_each_method_fitted():
    study = load_script()
    raw_cal, y_cal, raw_test, y_test = synthetic()
    idx = np.arange(300)
    types = {
        method: set(study.run_cell(raw_cal, y_cal, raw_test, y_test, idx, method)["types"].values())
        for method in study.METHODS
    }
    assert types == {
        "platt": {"platt"},
        "isotonic": {"isotonic"},
        "histogram": {"histogram"},
        "switch": {"platt"},  # 300 samples: below the 1000 of the fixed calibration rule
    }


def test_summarize_cells_reports_mean_and_sample_standard_deviation():
    study = load_script()
    cells = [
        {
            "ece": {"a": 0.10, "b": 0.20},
            "pooled": 0.15,
            "types": {"a": "platt", "b": "platt"},
            "convergence_warnings": 1,
        },
        {
            "ece": {"a": 0.20, "b": 0.20},
            "pooled": 0.25,
            "types": {"a": "platt", "b": "isotonic"},
            "convergence_warnings": 2,
        },
    ]
    s = study.summarize_cells(cells)

    assert s["n_seeds"] == 2
    assert s["per_model_mean"] == pytest.approx({"a": 0.15, "b": 0.20})
    assert s["per_model_std"]["a"] == pytest.approx(np.std([0.1, 0.2], ddof=1))
    assert s["per_model_std"]["b"] == 0.0
    assert s["pooled_mean"] == pytest.approx(0.20) and s["pooled_values"] == [0.15, 0.25]
    assert s["calibrator_types"] == {"platt": 3, "isotonic": 1}
    assert s["convergence_warnings"] == 3
    assert study.summarize_cells(cells[:1])["pooled_std"] == 0.0


# --- the whole script on the synthetic pool -------------------------------------------------------


def run(derived: Path, out: Path, **kwargs) -> Path:
    study = load_script()
    options = {"sizes": (10, 40, 2000, "all"), "n_seeds": 3}
    return study.run_study(
        derived, "tiny", TINY_ARTIFACT, out, ["python", "x"], **{**options, **kwargs}
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_outputs_sizes_skips_and_the_run_record(tmp_path: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, tmp_path / "out")

    assert sorted(p.name for p in out.iterdir()) == [
        "ece_vs_n.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    results = results_of(out)
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "calibration" and run_json["policy_version"] == "0782d801a5a3"
    assert results["sizes"] == ["10", "40", "all"] and results["seeds"] == 3
    assert set(results["skipped_sizes"]) == {"2000"}
    assert results["n_items"] == {"calib": 80, "test": 80}
    assert results["methods"] == ["platt", "isotonic", "histogram", "switch"]
    for size in results["sizes"]:
        assert set(results["cells"][size]) == set(results["methods"])
        for cell in results["cells"][size].values():
            assert cell["n_seeds"] == 3 and set(cell["per_model_mean"]) == set(MODELS)
            assert all(0.0 <= v <= 1.0 for v in cell["pooled_values"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "skipped size 2000" in report and "Pooled test ECE" in report
    assert "histogram binning is an experiment only" in report
    assert "no latency" in report


def test_all_items_gives_the_same_result_for_every_seed_and_subsets_do_not(tmp_path: Path):
    results = results_of(run(fixture_pool.make_pool(tmp_path / "derived"), tmp_path / "out"))
    for method in results["methods"]:
        whole = results["cells"]["all"][method]
        assert whole["pooled_std"] == 0.0 and len(set(whole["pooled_values"])) == 1
        assert all(v == 0.0 for v in whole["per_model_std"].values())
    assert results["cells"]["10"]["platt"]["pooled_std"] > 0.0


def test_a_cell_is_the_independent_fit_on_its_draw(tmp_path: Path):
    study = load_script()
    derived = fixture_pool.make_pool(tmp_path / "derived")
    results = results_of(run(derived, tmp_path / "out"))

    manifest, blob = artifact.load_artifact(TINY_ARTIFACT)
    replay = derived / "replay" / "tiny.jsonl"
    calib = rescore.read_split(replay, "calib", MODELS)
    test = rescore.read_split(replay, "test", MODELS)
    raw_cal = rescore.quality_scores(manifest, blob, calib.items["prompt"].tolist())
    raw_test = rescore.quality_scores(manifest, blob, test.items["prompt"].tolist())
    for method in ("platt", "histogram"):
        idx = study.draw_indices(80, 40, 1)
        forecasts = []
        for m in MODELS:
            cal = calibrate.fit_calibrator(raw_cal[m][idx], calib.success[m][idx], method=method)
            forecasts.append(calibrate.apply_calibrator(cal, raw_test[m]))
        want = calibration.ece(
            np.concatenate(forecasts),
            np.concatenate([test.success[m] for m in MODELS]),
            15,
            "mass",
        )
        assert results["cells"]["40"][method]["pooled_values"][1] == pytest.approx(want)


def test_switch_follows_the_spec_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    plain = results_of(run(derived, tmp_path / "a"))
    for size in plain["sizes"]:  # every size here is below 1000 samples: switch is Platt
        assert (
            plain["cells"][size]["switch"]["pooled_values"]
            == (plain["cells"][size]["platt"]["pooled_values"])
        )
        assert plain["cells"][size]["switch"]["calibrator_types"] == {"platt": 6}  # 2 models x 3

    monkeypatch.setattr(calibrate, "ISOTONIC_MIN", 30)
    low = results_of(run(derived, tmp_path / "b"))
    assert (
        low["cells"]["10"]["switch"]["pooled_values"]
        == low["cells"]["10"]["platt"]["pooled_values"]
    )
    for size in ("40", "all"):  # 30 or more samples: isotonic
        assert (
            low["cells"][size]["switch"]["pooled_values"]
            == (low["cells"][size]["isotonic"]["pooled_values"])
        )
        assert set(low["cells"][size]["switch"]["calibrator_types"]) == {"isotonic"}


def test_the_shipped_reference_is_the_artifacts_own_calibrators(tmp_path: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    results = results_of(run(derived, tmp_path / "out"))

    manifest, blob = artifact.load_artifact(TINY_ARTIFACT)
    test = rescore.read_split(derived / "replay" / "tiny.jsonl", "test", MODELS)
    raw = rescore.quality_scores(manifest, blob, test.items["prompt"].tolist())
    shipped = results["shipped"]
    for m in manifest["models"]:
        p = calibrate.apply_calibrator(m["calibrator"], raw[m["id"]])
        assert shipped["per_model"][m["id"]] == pytest.approx(
            calibration.ece(p, test.success[m["id"]], 15, "mass")
        )
    assert shipped["types"] == {"tiny-weak": "isotonic", "tiny-strong": "platt"}


def test_calibration_and_test_are_each_read_once_and_the_outputs_hold_no_prompt_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    real = rescore.read_split

    def spy(path, split, models):
        calls.append(split)
        return real(path, split, models)

    monkeypatch.setattr(rescore, "read_split", spy)
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, tmp_path / "out")

    assert calls == ["calib", "test"]
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_main_runs_with_the_pool_artifact_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    study = load_script()
    seen = {}

    def fake(derived, pool, artifact_dir, out_root, command, **kwargs):
        seen.update(pool=pool, artifact=artifact_dir, **kwargs)
        return tmp_path

    monkeypatch.setattr(study, "run_study", fake)
    monkeypatch.setattr(study.sweeps, "find_artifact", lambda root, pool: tmp_path / "v1")

    assert study.main(["--pool", "flagship", "--seeds", "5"]) == 0
    assert seen == {"pool": "flagship", "artifact": tmp_path / "v1", "n_seeds": 5}
