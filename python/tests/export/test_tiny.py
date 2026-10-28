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


def parity_path() -> Path:
    return repo_root() / "testdata" / "policy" / "parity" / "cases.json"


def parity_document() -> dict:
    return json.loads(parity_path().read_text(encoding="utf-8"))


def case_messages(case: dict) -> list[tuple[str, str]]:
    return make_tiny.expand_messages(case["messages"])


def test_parity_cases_belong_to_the_committed_artifact():
    manifest, _ = artifact.load_artifact(committed_dir())
    document = parity_document()

    assert document["artifact_version"] == manifest["version"]
    assert len(document["cases"]) == 30
    names = [case["name"] for case in document["cases"]]
    assert len(set(names)) == 30


def test_parity_cases_match_the_reference_predictor():
    manifest, blob = artifact.load_artifact(committed_dir())

    for case in parity_document()["cases"]:
        messages = case_messages(case)
        got = artifact.predict_bundle(manifest, blob, messages)
        assert got["ood_score"] == pytest.approx(case["ood_score"], abs=1e-12), case["name"]
        assert got["ood"] == case["ood"], case["name"]
        assert sorted(got["models"]) == sorted(case["models"]), case["name"]
        for model_id, expected in case["models"].items():
            for key, value in expected.items():
                assert got["models"][model_id][key] == pytest.approx(value, abs=1e-12), (
                    case["name"],
                    model_id,
                    key,
                )
        in_upper = sum(len(text.encode("utf-8")) for _, text in messages) + 8 * len(messages) + 16
        assert case["in_upper"] == in_upper, case["name"]
        if len(messages) == 1 and messages[0][0] == "user":
            assert case["in_upper"] == len(messages[0][1].encode("utf-8")) + 24, case["name"]


def test_parity_cases_cover_the_edge_cases_and_every_dense_marker():
    cases = parity_document()["cases"]
    texts = [[text for _, text in case_messages(case)] for case in cases]
    roles = {role for case in cases for role, _ in case_messages(case)}
    dense = np.array([features.extract(case_messages(case), make_tiny.DIM)[2] for case in cases])

    assert any(all(text == "" for text in group) for group in texts)  # empty text
    long_cut = [
        "\n".join(group).encode("utf-8")
        for group in texts
        if len("\n".join(group).encode("utf-8")) >= 40 * 1024
    ]
    assert any(full[features.PREFIX_BYTES] & 0xC0 == 0x80 for full in long_cut), (
        "a 40 KiB text must be cut inside a UTF-8 sequence"
    )
    assert any(any(ord(ch) > 127 for ch in "".join(group)) for group in texts)  # non-ASCII
    assert {"system", "user", "assistant"} <= roles
    # fence, LaTeX marker, choice line, system role and question mark each occur, and the
    # share features (digits, uppercase, non-ASCII, punctuation) are non-zero somewhere
    for column in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        assert (dense[:, column] > 0).any(), f"no case exercises dense feature {column}"
    assert (dense[:, 1] > np.log1p(1)).any()  # more than one message


def test_parity_cases_include_both_ood_verdicts():
    verdicts = {case["ood"] for case in parity_document()["cases"]}
    assert verdicts == {True, False}


def test_parity_file_is_canonical_lf_and_reproducible(rebuilt):
    raw = parity_path().read_bytes()
    assert b"\r" not in raw
    document = json.loads(raw)
    canonical = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True) + "\n"
    assert raw == canonical.encode("ascii")

    manifest, blob = artifact.load_artifact(rebuilt)
    again = make_tiny.parity_document(manifest, blob)
    assert again["artifact_version"] == document["artifact_version"]
    assert without_noise(again) == without_noise(document)


def without_noise(document: dict) -> dict:
    """The document as plain JSON data, ready for an equality check."""
    return json.loads(json.dumps(document))
