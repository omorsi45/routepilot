"""Tests for the committed tiny policy artifact and the script that generates it."""

import json
from pathlib import Path

import make_tiny
import numpy as np
import pytest

from routepilot_train import features
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root


def committed_dir() -> Path:
    versions = [p for p in make_tiny.tiny_dir(repo_root()).iterdir() if p.is_dir()]
    assert len(versions) == 1, "testdata/policy/tiny must hold exactly one version directory"
    return versions[0]


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory) -> Path:
    bundle, weights = make_tiny.build_tiny()
    return artifact.export_artifact(tmp_path_factory.mktemp("tiny"), "tiny", bundle, weights)


def without_timestamps(manifest: dict) -> dict:
    stripped = json.loads(json.dumps(manifest))
    stripped["training"].pop("created_at")
    return stripped


def test_rebuilding_reproduces_the_committed_weights(rebuilt):
    committed = committed_dir()

    assert (rebuilt / "weights.bin").read_bytes() == (committed / "weights.bin").read_bytes()
    fresh, _ = artifact.load_artifact(rebuilt)
    kept, _ = artifact.load_artifact(committed)
    assert without_timestamps(fresh) == without_timestamps(kept)
    assert rebuilt.name == committed.name


def test_committed_artifact_is_self_consistent():
    committed = committed_dir()
    manifest, blob = artifact.load_artifact(committed)
    raw = (committed / "manifest.json").read_bytes()

    version = manifest["version"]
    needle = f'"version": "{version}"'.encode()
    assert committed.name == version and raw.count(needle) == 1
    assert artifact.content_version(raw.replace(needle, b'"version": ""'), blob) == version
    assert raw == artifact.manifest_bytes(manifest)
    assert b"\r" not in raw
    assert manifest["dim"] == make_tiny.DIM
    assert [m["id"] for m in manifest["models"]] == list(make_tiny.MODELS)
    assert [m["calibrator"]["type"] for m in manifest["models"]] == ["isotonic", "platt"]
    assert len(manifest["golden"]) == 8
    for entry in manifest["golden"]:
        got = artifact.predict_bundle(manifest, blob, [tuple(m) for m in entry["messages"]])
        assert got["ood_score"] == pytest.approx(entry["ood_score"], abs=1e-12)
        for model_id, expected in entry["models"].items():
            for key in ("p", "l", "e_out", "q99"):
                assert got["models"][model_id][key] == pytest.approx(expected[key], abs=1e-12)


def test_the_tiny_pool_covers_both_calibrators_and_a_usable_ood_threshold(rebuilt):
    manifest, _ = artifact.load_artifact(rebuilt)

    kinds = {m["id"]: m["calibrator"]["type"] for m in manifest["models"]}
    assert kinds == {"tiny-weak": "isotonic", "tiny-strong": "platt"}
    assert 0.0 < manifest["ood"]["threshold"] < 1.0
    assert manifest["ood"]["k"] == 32
    for centroid in manifest["ood"]["centroids"]:
        assert centroid["idx"]["len"] == centroid["val"]["len"] > 0
    assert manifest["feature_spec_version"] == features.SPEC_VERSION
    weights = np.fromfile(rebuilt / "weights.bin", dtype="<f4")
    assert np.isfinite(weights).all()
