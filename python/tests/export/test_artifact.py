"""Tests for the content-addressed policy artifact writer and its reference predictor."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from routepilot_train import features
from routepilot_train.export import artifact

DIM = 16
CREATED_AT = "2026-10-28T00:00:00Z"
ISOTONIC = {"type": "isotonic", "x": [0.1, 0.4, 0.7, 0.9], "y": [0.05, 0.3, 0.6, 0.95]}
PLATT = {"type": "platt", "a": 0.9, "b": -0.1}
LCB = {
    "upper": [0.4, 0.8, 1.0],
    "lower": [0.1, 0.3, 0.6],
    "n": [40, 40, 40],
    "pooled_rate": 0.5,
    "pooled_lcb": 0.4,
}
MANIFEST_KEYS = [
    "dense_mean",
    "dense_std",
    "dim",
    "feature_spec_version",
    "format",
    "format_version",
    "golden",
    "models",
    "ood",
    "training",
    "version",
]


def fit_shaped_model(rng: np.random.Generator, model_id: str, calibrator: dict, scale: float):
    """A dict shaped like the train modules' outputs, with random weights."""
    return {
        "id": model_id,
        "quality": {
            "coef": rng.normal(size=DIM + 12) * scale,
            "intercept": 0.25,
            "C": 1.0,
            "cv_log_loss": 0.5,
        },
        "calibrator": calibrator,
        "lcb": LCB,
        "length": {
            "coef": rng.normal(size=DIM + 12) * scale * 0.01,
            "intercept": 4.0,
            "alpha": 10.0,
            "smear": 1.1,
            "residual_quantiles": {"0.5": 0.0, "0.9": 0.5, "0.95": 0.7, "0.99": 1.2},
            "in_ratio": 0.3,
        },
    }


def fit_shaped_ood() -> dict:
    return {
        "k": 3,
        "threshold": 0.6,
        "centroids": [
            {
                "idx": np.array([1, 3, 5], dtype=np.int32),
                "val": np.array([0.5, 0.5, 0.70710678], dtype=np.float32),
            },
            {
                "idx": np.array([0, 2], dtype=np.int32),
                "val": np.array([0.6, 0.8], dtype=np.float32),
            },
            {"idx": np.array([9], dtype=np.int32), "val": np.array([1.0], dtype=np.float32)},
        ],
    }


def build(scale: float = 1.0, seed: int = 0, **overrides):
    rng = np.random.default_rng(seed)
    args = {
        "dim": DIM,
        "dense_mean": np.linspace(0.0, 1.0, 12),
        "dense_std": np.full(12, 2.0),
        "models": [
            fit_shaped_model(rng, "alpha-1", ISOTONIC, scale),
            fit_shaped_model(rng, "beta-2", PLATT, scale),
        ],
        "ood": fit_shaped_ood(),
        "training": {
            "pool": "unit",
            "dataset_sha256": "0" * 64,
            "split_salt": "salt",
            "git_sha": "abc",
            "created_at": CREATED_AT,
        },
    }
    args.update(overrides)
    return artifact.assemble_bundle(**args)


def read(out: Path) -> tuple[bytes, dict, bytes]:
    raw = (out / "manifest.json").read_bytes()
    return raw, json.loads(raw), (out / "weights.bin").read_bytes()


def test_version_is_content_hash(tmp_path):
    bundle, weights = build()

    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)

    raw, manifest, blob = read(out)
    version = manifest["version"]
    assert out == tmp_path / "pol" / version
    assert len(version) == 12 and set(version) <= set("0123456789abcdef")
    needle = f'"version": "{version}"'.encode()
    assert raw.count(needle) == 1
    unversioned = raw.replace(needle, b'"version": ""')
    assert hashlib.sha256(unversioned + blob).hexdigest()[:12] == version


def test_version_follows_content_and_is_deterministic(tmp_path):
    bundle, weights = build()
    first = artifact.export_artifact(tmp_path / "a", "pol", bundle, weights)
    again = artifact.export_artifact(tmp_path / "b", "pol", bundle, weights)
    assert (first / "manifest.json").read_bytes() == (again / "manifest.json").read_bytes()
    assert (first / "weights.bin").read_bytes() == (again / "weights.bin").read_bytes()

    nudged = {name: array.copy() for name, array in weights.items()}
    nudged[bundle["models"][0]["quality"]["weights"]][3] += 0.5
    other = artifact.export_artifact(tmp_path / "c", "pol", bundle, nudged)
    assert other.name != first.name

    changed_meta, same_weights = build(training={**build()[0]["training"], "pool": "other"})
    third = artifact.export_artifact(tmp_path / "d", "pol", changed_meta, same_weights)
    assert third.name != first.name


def test_manifest_canonical_bytes(tmp_path):
    bundle, weights = build()

    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)

    raw, manifest, _ = read(out)
    canonical = json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True) + "\n"
    assert raw == canonical.encode("ascii")
    assert b"\r" not in raw
    assert sorted(manifest) == MANIFEST_KEYS
    assert manifest["format"] == "routepilot-policy-artifact"
    assert manifest["format_version"] == 1
    assert manifest["feature_spec_version"] == features.SPEC_VERSION
    assert manifest["dim"] == DIM
    assert len(manifest["dense_mean"]) == len(manifest["dense_std"]) == 12
    assert [m["id"] for m in manifest["models"]] == ["alpha-1", "beta-2"]
    assert sorted(manifest["training"]) == [
        "calibrated_on",
        "created_at",
        "cv",
        "dataset_sha256",
        "git_sha",
        "pool",
        "split_salt",
        "trained_on",
    ]
    assert manifest["training"]["trained_on"] == ["train"]
    assert manifest["training"]["calibrated_on"] == ["calib"]
    assert manifest["training"]["cv"]["alpha-1"] == {"C": 1.0, "alpha": 10.0}


def test_weights_layout_is_packed_little_endian_arrays(tmp_path):
    bundle, weights = build()

    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)

    _, manifest, blob = read(out)
    refs = [
        *[r for m in manifest["models"] for r in (m["quality"]["weights"], m["length"]["weights"])],
        *[r for c in manifest["ood"]["centroids"] for r in (c["idx"], c["val"])],
    ]
    # Arrays follow each other in manifest order with no gaps: models first, then centroids.
    cursor = 0
    for ref in refs:
        assert set(ref) == {"offset", "len", "dtype"}
        assert ref["offset"] == cursor
        cursor += 4 * ref["len"]
    assert cursor == len(blob)
    assert [r["dtype"] for r in refs] == ["f32"] * 4 + ["i32", "f32"] * 3
    assert [r["len"] for r in refs[:4]] == [DIM + 12] * 4

    quality = manifest["models"][0]["quality"]["weights"]
    source = weights[bundle["models"][0]["quality"]["weights"]]
    np.testing.assert_array_equal(
        np.frombuffer(blob, dtype="<f4", count=quality["len"], offset=quality["offset"]),
        source.astype(np.float32),
    )
    first_idx = manifest["ood"]["centroids"][0]["idx"]
    np.testing.assert_array_equal(
        np.frombuffer(blob, dtype="<i4", count=first_idx["len"], offset=first_idx["offset"]),
        [1, 3, 5],
    )


def expected_scores(messages, bundle, weights, model_index: int, *, as_float32: bool):
    """Independent re-computation of one model's raw length score and quality score."""
    idx, val, dense = features.extract(messages, DIM)
    z = (dense - bundle["dense_mean"]) / bundle["dense_std"]
    model = bundle["models"][model_index]

    def linear(name: str, bias: float) -> float:
        w = weights[name]
        if as_float32:
            w = w.astype(np.float32)
        acc = 0.0
        for i, v in zip(idx.tolist(), val.tolist(), strict=True):
            acc += float(w[i]) * v
        for d in range(12):
            acc += float(w[DIM + d]) * float(z[d])
        return acc + bias

    return (
        linear(model["quality"]["weights"], model["quality"]["bias"]),
        linear(model["length"]["weights"], model["length"]["bias"]),
    )


def test_golden_uses_float32_weights(tmp_path):
    bundle, weights = build(scale=100.0)
    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)
    _, manifest, _ = read(out)
    entry = manifest["golden"][0]
    messages = [(role, text) for role, text in entry["messages"]]
    length = bundle["models"][0]["length"]

    _, s32 = expected_scores(messages, bundle, weights, 0, as_float32=True)
    _, s64 = expected_scores(messages, bundle, weights, 0, as_float32=False)

    e_out32 = math.exp(s32) * length["smear"] - 1.0
    e_out64 = math.exp(s64) * length["smear"] - 1.0
    got = entry["models"]["alpha-1"]["e_out"]
    assert got == pytest.approx(e_out32, rel=1e-12)
    assert abs(got - e_out64) > 1e-8 * abs(e_out64)  # float64 weights would give another value
    q99 = math.exp(s32 + length["residual_quantiles"]["0.99"]) - 1.0
    assert entry["models"]["alpha-1"]["q99"] == pytest.approx(q99, rel=1e-12)


def test_golden_entries_cover_every_model_and_prompt(tmp_path):
    bundle, weights = build()

    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)

    _, manifest, _ = read(out)
    assert len(manifest["golden"]) == 8
    for entry, prompt in zip(manifest["golden"], artifact.GOLDEN_PROMPTS, strict=True):
        assert sorted(entry) == ["messages", "models", "ood_score"]
        assert [tuple(m) for m in entry["messages"]] == [tuple(m) for m in prompt]
        assert sorted(entry["models"]) == ["alpha-1", "beta-2"]
        for scores in entry["models"].values():
            assert sorted(scores) == ["e_out", "l", "p", "q99"]
            assert 0.0 <= scores["p"] <= 1.0 and 0.0 <= scores["l"] <= 1.0
        assert math.isfinite(entry["ood_score"])


def test_golden_prompts_are_the_specified_invented_set():
    prompts = artifact.GOLDEN_PROMPTS
    assert len(prompts) == 8
    vectors = [features.extract(list(p), features.DEFAULT_DIM) for p in prompts]
    dense = [d for _, _, d in vectors]
    assert any(d[7] > 0 for d in dense)  # a code fence
    assert any(d[8] == 1 for d in dense)  # a LaTeX marker
    assert any(d[9] == 1 for d in dense)  # a multiple-choice line
    assert any(d[10] == 1 for d in dense)  # a system message
    assert any(d[5] > 0 for d in dense)  # non-ASCII bytes
    assert any(len(p) == 2 and p[0][0] == "system" for p in prompts)
    assert any(len(text.encode("utf-8")) >= 3000 for p in prompts for _, text in p)
    assert any(len(p) == 1 and len(p[0][1].split()) == 1 for p in prompts)  # a single word
    assert all(role in {"system", "user"} for p in prompts for role, _ in p)


def test_predict_bundle_agrees_with_the_training_functions(tmp_path):
    from routepilot_train.train import calibrate, lcb

    bundle, weights = build()
    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)
    _, manifest, blob = read(out)
    messages = [("user", "What is the sum of the first ten primes?")]

    got = artifact.predict_bundle(manifest, blob, messages)

    for position, model in enumerate(bundle["models"]):
        s_quality, s_length = expected_scores(messages, bundle, weights, position, as_float32=True)
        p = float(calibrate.apply_calibrator(model["calibrator"], s_quality))
        scores = got["models"][model["id"]]
        assert scores["p"] == pytest.approx(p, abs=1e-12)
        assert scores["l"] == lcb.lookup_lcb(model["lcb"], p)
        assert scores["e_out"] == pytest.approx(math.exp(s_length) * 1.1 - 1.0, rel=1e-12)
        assert scores["out_q"][3] == pytest.approx(math.exp(s_length + 1.2) - 1.0, rel=1e-12)
        assert scores["in_ratio"] == 0.3
        assert scores["pooled_rate"] == 0.5 and scores["pooled_lcb"] == 0.4
    assert 0.0 <= got["ood_score"] <= 1.0
    assert got["ood"] == (got["ood_score"] > 0.6)


def test_predict_bundle_empty_text_scores_one_and_is_ood(tmp_path):
    bundle, weights = build()
    out = artifact.export_artifact(tmp_path, "pol", bundle, weights)
    _, manifest, blob = read(out)

    got = artifact.predict_bundle(manifest, blob, [("user", "")])

    assert got["ood_score"] == 1.0
    assert got["ood"] is True


def test_export_rejects_a_weight_array_of_the_wrong_length(tmp_path):
    bundle, weights = build()
    name = bundle["models"][0]["quality"]["weights"]
    weights[name] = weights[name][:-1]
    with pytest.raises(ValueError, match="length"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_unreferenced_and_missing_weights(tmp_path):
    bundle, weights = build()
    with pytest.raises(ValueError, match="unreferenced"):
        artifact.export_artifact(tmp_path, "pol", bundle, {**weights, "stray": np.zeros(3)})
    name = bundle["models"][1]["length"]["weights"]
    with pytest.raises(ValueError, match="missing"):
        artifact.export_artifact(
            tmp_path, "pol", bundle, {k: v for k, v in weights.items() if k != name}
        )


def test_export_rejects_non_finite_weights(tmp_path):
    bundle, weights = build()
    weights[bundle["models"][0]["quality"]["weights"]][2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_float32_overflow(tmp_path):
    bundle, weights = build()
    weights[bundle["models"][0]["quality"]["weights"]][2] = 1e300
    with pytest.raises(ValueError, match="finite"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_bad_isotonic_knots(tmp_path):
    flat = {"type": "isotonic", "x": [0.1, 0.1, 0.5], "y": [0.1, 0.2, 0.3]}
    rng = np.random.default_rng(0)
    bundle, weights = build(models=[fit_shaped_model(rng, "alpha-1", flat, 1.0)])
    with pytest.raises(ValueError, match="strictly increasing"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_unsorted_centroid_indices(tmp_path):
    ood = fit_shaped_ood()
    ood["centroids"][0]["idx"] = np.array([3, 1, 5], dtype=np.int32)
    bundle, weights = build(ood=ood)
    with pytest.raises(ValueError, match="ascending"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_non_finite_scalars(tmp_path):
    bundle, weights = build()
    bundle["models"][0]["length"]["smear"] = float("nan")
    with pytest.raises(ValueError, match="NaN|finite|Out of range"):
        artifact.export_artifact(tmp_path, "pol", bundle, weights)


def test_export_rejects_an_unsafe_name(tmp_path):
    bundle, weights = build()
    with pytest.raises(ValueError, match="name"):
        artifact.export_artifact(tmp_path, "../escape", bundle, weights)


def test_assemble_requires_unique_model_ids():
    rng = np.random.default_rng(0)
    twin = [
        fit_shaped_model(rng, "same", ISOTONIC, 1.0),
        fit_shaped_model(rng, "same", PLATT, 1.0),
    ]
    with pytest.raises(ValueError, match="unique"):
        build(models=twin)
