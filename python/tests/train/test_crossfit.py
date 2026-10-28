"""Tests for the cross-fitting halves and the artifact refit on half of the calibration split."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from routepilot_train import replaytool
from routepilot_train.evaluate import rescore
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root
from routepilot_train.train import calibrate, crossfit, lcb

TINY = repo_root() / "testdata" / "policy" / "tiny" / "0782d801a5a3"
WORDS = ["ba", "ko", "zin", "pelu", "rav", "tuo", "mik", "sed"]
N = 240


@pytest.fixture(scope="module")
def binary() -> Path:
    try:
        return replaytool.build_go(repo_root())
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))


def invented_texts(n: int) -> list[str]:
    rng = np.random.default_rng(3)
    return [
        " ".join(str(rng.choice(WORDS)) for _ in range(int(rng.integers(2, 40)))) + " ?" * (i % 2)
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def calib():
    manifest, blob = artifact.load_artifact(TINY)
    text = invented_texts(N)
    raw = rescore.quality_scores(manifest, blob, text)
    rng = np.random.default_rng(5)
    y = {m: (rng.random(N) < 1 / (1 + np.exp(-0.7 * s))).astype(np.int64) for m, s in raw.items()}
    items = pd.DataFrame(
        {"item_id": [f"toy/{i}" for i in range(N)], "dataset": "toy", "split": "calib"}
    )
    return manifest, blob, text, raw, y, items


# --- the halves -------------------------------------------------------------------------------


def test_half_is_the_parity_of_the_hash_of_the_item_id_and_the_salt():
    assert crossfit.SALT == "routepilot-crossfit-v1"
    for item_id in ("aime/0", "gpqa/17", "mmlupro/2999", "simpleqa/4000", "é/ü"):
        digest = hashlib.sha256((item_id + "routepilot-crossfit-v1").encode()).hexdigest()
        assert crossfit.half_of(item_id) == int(digest, 16) % 2


def test_halves_partition_the_items_and_are_roughly_balanced():
    ids = [f"gpqa/{i}" for i in range(2000)]
    a, b = crossfit.split_halves(ids)

    assert a.dtype == bool and b.dtype == bool and a.shape == b.shape == (2000,)
    assert (a ^ b).all()  # every item is in exactly one half
    assert 0.45 < a.mean() < 0.55
    assert crossfit.split_halves(ids)[0].tolist() == a.tolist()  # no randomness
    assert crossfit.split_halves([])[0].shape == (0,)


# --- the refit --------------------------------------------------------------------------------


def refit(calib, tmp_path: Path, mask=None, y=None):
    manifest, blob, _, raw, ys, items = calib
    mask = crossfit.split_halves(items["item_id"])[0] if mask is None else mask
    return crossfit.refit_artifact(
        manifest, blob, items, raw, ys if y is None else y, mask, tmp_path / "crossfit", "tiny"
    )


def test_refit_keeps_every_weight_and_replaces_calibrators_and_tables(calib, tmp_path: Path):
    manifest, blob, _, raw, y, items = calib
    mask = crossfit.split_halves(items["item_id"])[0]

    out = refit(calib, tmp_path)

    new, new_blob = artifact.load_artifact(out)
    assert out.parent == tmp_path / "crossfit" / "tiny" and out.name == new["version"]
    assert new["version"] != manifest["version"]
    assert new_blob == blob  # quality, length and OOD arrays are untouched
    for key in ("dim", "dense_mean", "dense_std", "ood", "feature_spec_version", "training"):
        assert new[key] == manifest[key], key
    for old_model, model in zip(manifest["models"], new["models"], strict=True):
        assert model["quality"] == old_model["quality"] and model["length"] == old_model["length"]
        # A holds fewer than 1000 items, so the spec rule picks Platt for every model
        expected = calibrate.fit_calibrator(raw[model["id"]][mask], y[model["id"]][mask])
        assert expected["type"] == "platt" and model["calibrator"] == expected
        table = lcb.fit_lcb(
            calibrate.apply_calibrator(expected, raw[model["id"]][mask]), y[model["id"]][mask]
        )
        assert model["lcb"] == table and sum(model["lcb"]["n"]) == int(mask.sum())


def test_refit_ignores_the_outcomes_of_the_other_half(calib, tmp_path: Path):
    manifest, blob, _, raw, y, items = calib
    mask = crossfit.split_halves(items["item_id"])[0]
    flipped = {m: np.where(mask, v, 1 - v) for m, v in y.items()}  # B outcomes inverted

    same = refit(calib, tmp_path / "a")
    also_same = refit(calib, tmp_path / "b", y=flipped)
    other = refit(calib, tmp_path / "c", mask=~mask)

    assert same.name == also_same.name  # nothing from half B reached the artifact
    assert other.name != same.name


def test_refit_predicts_what_its_calibrators_say(calib, tmp_path: Path):
    manifest, blob, text, raw, y, items = calib
    out = refit(calib, tmp_path)
    new, new_blob = artifact.load_artifact(out)

    scores = rescore.quality_scores(new, new_blob, text[:12])
    for model in new["models"]:
        p = calibrate.apply_calibrator(model["calibrator"], scores[model["id"]])
        for i in range(12):
            got = artifact.predict_bundle(new, new_blob, [("user", text[i])])
            assert got["models"][model["id"]]["p"] == pytest.approx(p[i], abs=1e-9)
            assert got["models"][model["id"]]["l"] == lcb.lookup_lcb(model["lcb"], p[i])


def test_the_go_loader_accepts_the_refit_artifact(calib, tmp_path: Path, binary: Path):
    out = refit(calib, tmp_path)
    proc = subprocess.run(
        [str(binary), "policy", "verify", str(out)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == out.name


def test_refit_guards_the_split_and_the_inputs(calib, tmp_path: Path):
    manifest, blob, _, raw, y, items = calib
    mask = crossfit.split_halves(items["item_id"])[0]
    poisoned = items.copy()
    poisoned.loc[3, "split"] = "test"

    with pytest.raises(ValueError, match="splits not allowed"):
        crossfit.refit_artifact(manifest, blob, poisoned, raw, y, mask, tmp_path, "tiny")
    with pytest.raises(ValueError, match="mask"):
        crossfit.refit_artifact(manifest, blob, items, raw, y, mask[:-1], tmp_path, "tiny")
    with pytest.raises(ValueError, match="no items"):
        crossfit.refit_artifact(manifest, blob, items, raw, y, ~mask & mask, tmp_path, "tiny")
    with pytest.raises(ValueError, match="missing"):
        crossfit.refit_artifact(manifest, blob, items, {}, y, mask, tmp_path, "tiny")
    assert not any(tmp_path.iterdir())  # nothing was written for a refused call


def test_bundle_round_trip_reproduces_the_artifact(calib, tmp_path: Path):
    """Rebuilding the bundle of an artifact and exporting it again gives the same version."""
    manifest, blob, *_ = calib
    bundle, weights = artifact.bundle_from_artifact(manifest, blob)

    out = artifact.export_artifact(tmp_path, "tiny", bundle, weights)

    assert out.name == manifest["version"]
    assert (out / "weights.bin").read_bytes() == blob
    assert (out / "manifest.json").read_bytes() == (TINY / "manifest.json").read_bytes()
