"""Tests for the training pipeline: split discipline, export and the recorded run."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from routepilot_train import features
from routepilot_train.data import selection, splits
from routepilot_train.export import artifact
from routepilot_train.train import pipeline, quality

DIM = 1024
SENTINEL = "sentinelzzword"
CREATED_AT = "2026-10-28T00:00:00Z"
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
COUNTS = {"train": 150, "calib": 90, "test": 30, "heldout": 20}
POOL = {
    "pool": "toy",
    "models": ["m-a", "m-b"],
    "prompt_overhead_tokens": {"m-a": 0, "m-b": 12},
}


def prompt_for(split: str, i: int, rng: np.random.Generator) -> str:
    words = [str(rng.choice(WORDS)) for _ in range(int(rng.integers(3, 20)))]
    if split in ("test", "heldout"):
        words.append(SENTINEL)  # a leak of these rows into any fit shows up in the texts
    if i % 3 == 0:
        words.append("?")
    return " ".join(words)


def write_derived(root: Path) -> Path:
    """A toy derived directory: 2 pool models, 1 model outside the pool, 1 incomplete item."""
    rng = np.random.default_rng(0)
    split_rows, prompt_rows, outcome_rows = [], [], []
    for split, count in COUNTS.items():
        dataset = "toy-held" if split == "heldout" else "toy"
        for i in range(count):
            item_id = f"{dataset}/{split}-{i}"
            prompt = prompt_for(split, i, rng)
            data = prompt.encode("utf-8")
            split_rows.append({"item_id": item_id, "dataset": dataset, "split": split})
            prompt_rows.append(
                {
                    "dataset": dataset,
                    "item_id": item_id,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(data).hexdigest(),
                    "prompt_bytes": len(data),
                }
            )
            for model in ("m-a", "m-b", "m-outside"):
                if model == "m-b" and split == "train" and i == 0:
                    continue  # an incomplete item: no record for m-b
                success = bool(rng.random() < (0.7 if "alpha" in prompt else 0.4))
                outcome_rows.append(
                    {
                        "dataset": dataset,
                        "item_id": item_id,
                        "model": model,
                        "score": float(success),
                        "success": success,
                        "prompt_tokens": max(1, len(data) // 4),
                        "completion_tokens": int(rng.integers(5, 400)),
                        "cost_usd": 0.001,
                        "prompt_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    derived = root / "derived"
    (derived / "pools").mkdir(parents=True)
    pd.DataFrame(split_rows).to_parquet(derived / "splits.parquet", index=False)
    pd.DataFrame(prompt_rows).to_parquet(derived / "prompts.parquet", index=False)
    pd.DataFrame(outcome_rows).to_parquet(derived / "outcomes.parquet", index=False)
    (derived / "pools" / "toy.json").write_text(json.dumps(POOL), encoding="utf-8")
    return derived


@pytest.fixture(scope="module")
def derived(tmp_path_factory) -> Path:
    return write_derived(tmp_path_factory.mktemp("toy"))


@pytest.fixture(scope="module")
def frames(derived):
    return pipeline.load_fit_data(derived, POOL)


def test_load_fit_data_returns_only_train_and_calib_rows_of_complete_items(frames):
    train, calib = frames

    assert set(train.items["split"]) == {"train"}
    assert set(calib.items["split"]) == {"calib"}
    # the first train item has no record for m-b, so it is dropped from the pool
    assert len(train.items) == COUNTS["train"] - 1
    assert len(calib.items) == COUNTS["calib"]
    assert "toy/train-0" not in set(train.items["item_id"])
    for data in frames:
        assert SENTINEL not in " ".join(data.items["prompt"])
        assert sorted(data.success) == ["m-a", "m-b"]  # m-outside is not in the pool
        for model in POOL["models"]:
            for arrays in (data.success, data.completion_tokens, data.prompt_tokens):
                assert arrays[model].shape == (len(data.items),)
    assert train.items["item_id"].is_monotonic_increasing
    assert set(np.unique(train.success["m-a"])) <= {0, 1}


def test_fit_pool_rejects_a_test_row_in_the_train_frame(frames, monkeypatch):
    train, calib = frames
    poisoned = train.items.copy()
    poisoned.loc[poisoned.index[3], "split"] = "test"
    monkeypatch.setattr(quality, "fit_quality", pytest.fail)  # no fit may run first

    with pytest.raises(ValueError, match="splits not allowed"):
        pipeline.fit_pool(replace(train, items=poisoned), calib, POOL, DIM)


def test_fit_pool_rejects_a_train_row_in_the_calib_frame_and_heldout_anywhere(frames, monkeypatch):
    train, calib = frames
    monkeypatch.setattr(quality, "fit_quality", pytest.fail)
    for bad in ("train", "heldout", "test"):
        poisoned = calib.items.copy()
        poisoned.loc[poisoned.index[0], "split"] = bad
        with pytest.raises(ValueError, match="splits not allowed"):
            pipeline.fit_pool(train, replace(calib, items=poisoned), POOL, DIM)
    unlabeled = train.items.drop(columns="split")
    with pytest.raises(ValueError, match="split"):
        pipeline.fit_pool(replace(train, items=unlabeled), calib, POOL, DIM)


def run_toy(derived: Path, tmp_path: Path, **overrides):
    args = {
        "pool_name": "toy",
        "derived": derived,
        "out": tmp_path / "policies",
        "dim": DIM,
        "results_root": tmp_path / "results",
        "created_at": CREATED_AT,
    }
    args.update(overrides)
    return pipeline.run(**args)


def test_run_exports_an_artifact_and_records_the_run(derived, tmp_path, monkeypatch):
    seen: list[str] = []
    real = features.extract_texts

    def spy(texts, dim=features.DEFAULT_DIM):
        seen.extend(texts)
        return real(texts, dim)

    monkeypatch.setattr(features, "extract_texts", spy)

    summary = run_toy(derived, tmp_path)

    # Only train and calibration prompts were ever featurized.
    assert len(seen) == (COUNTS["train"] - 1) + COUNTS["calib"]
    assert not any(SENTINEL in text for text in seen)

    out = summary["artifact_dir"]
    assert out == tmp_path / "policies" / "toy" / summary["version"]
    manifest, blob = artifact.load_artifact(out)
    raw = (out / "manifest.json").read_bytes()
    needle = f'"version": "{manifest["version"]}"'.encode()
    assert artifact.content_version(raw.replace(needle, b'"version": ""'), blob) == out.name
    assert [m["id"] for m in manifest["models"]] == POOL["models"]
    assert manifest["dim"] == DIM
    training = manifest["training"]
    assert training["pool"] == "toy"
    assert training["dataset_sha256"] == selection.DATASET_SHA256
    assert training["split_salt"] == splits.SALT
    assert training["trained_on"] == ["train"] and training["calibrated_on"] == ["calib"]
    assert training["created_at"] == CREATED_AT
    assert sorted(training["cv"]) == POOL["models"]
    # stored std has zeros already replaced by 1 so Go divides directly
    assert all(v > 0 for v in manifest["dense_std"])

    results = summary["results_dir"]
    assert results.parent == tmp_path / "results" / "train"
    run_json = json.loads((results / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "train" and run_json["policy_version"] == out.name
    assert run_json["run_id"] == results.name
    payload = json.loads((results / "results.json").read_text(encoding="utf-8"))
    assert payload["artifact_version"] == out.name
    assert payload["n_train"] == COUNTS["train"] - 1 and payload["n_calib"] == COUNTS["calib"]
    assert payload["wall_time_s"] >= 0
    assert payload["ood"]["threshold"] == manifest["ood"]["threshold"]
    assert sorted(payload["models"]) == POOL["models"]
    for model in payload["models"].values():
        assert {"C", "cv_log_loss", "alpha", "calibrator", "calib_ece", "calib_brier"} <= set(model)
        assert model["lcb_bins"] >= 1 and model["calibrator"] in {"isotonic", "platt"}
    report = (results / "report.md").read_text(encoding="utf-8")
    assert "in-sample for the calibrator" in report
    assert "calib_ece" in report and out.name in report
    assert SENTINEL not in report and "alpha bravo" not in report


def test_run_is_deterministic_for_a_fixed_created_at(derived, tmp_path):
    first = run_toy(derived, tmp_path / "a")
    second = run_toy(derived, tmp_path / "b")

    assert first["version"] == second["version"]
    assert (first["artifact_dir"] / "weights.bin").read_bytes() == (
        second["artifact_dir"] / "weights.bin"
    ).read_bytes()


def test_run_uses_the_current_time_when_created_at_is_absent(derived, tmp_path):
    before = datetime.now(UTC).replace(microsecond=0)

    summary = run_toy(derived, tmp_path, created_at=None)

    manifest, _ = artifact.load_artifact(summary["artifact_dir"])
    stamped = datetime.strptime(manifest["training"]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert stamped.replace(tzinfo=UTC) >= before


def test_run_rejects_an_unknown_pool(derived, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_toy(derived, tmp_path, pool_name="nonexistent")


def test_main_prints_the_version_and_the_artifact_path(derived, tmp_path, capsys):
    code = pipeline.main(
        [
            "--pool",
            "toy",
            "--derived",
            str(derived),
            "--out",
            str(tmp_path / "policies"),
            "--dim",
            str(DIM),
            "--results-root",
            str(tmp_path / "results"),
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    versions = [p.name for p in (tmp_path / "policies" / "toy").iterdir()]
    assert len(versions) == 1
    assert f"artifact version: {versions[0]}" in printed
    assert "results:" in printed
    assert "alpha bravo" not in printed
