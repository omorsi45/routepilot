"""Tests for recomputing an exported artifact's raw quality scores and reading split rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from routepilot_train import features
from routepilot_train.evaluate import rescore
from routepilot_train.export import artifact
from routepilot_train.runinfo import repo_root
from routepilot_train.train.calibrate import apply_calibrator

TINY = repo_root() / "testdata" / "policy" / "tiny" / "0782d801a5a3"
WORDS = ["ba", "ko", "zin", "pelu", "rav", "tuo", "mik", "sed"]


def texts(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        words = [str(rng.choice(WORDS)) for _ in range(int(rng.integers(2, 40)))]
        out.append(
            " ".join(words) + (" ?" if i % 3 == 0 else "") + (" 12 + 7" if i % 5 == 0 else "")
        )
    return out


def test_scores_calibrate_to_the_probabilities_of_the_reference_predictor():
    manifest, blob = artifact.load_artifact(TINY)
    single = [g for g in manifest["golden"] if len(g["messages"]) == 1]
    assert len(single) >= 6  # the golden prompts that are one user message

    scores = rescore.quality_scores(manifest, blob, [g["messages"][0][1] for g in single])

    assert sorted(scores) == sorted(m["id"] for m in manifest["models"])
    for model in manifest["models"]:
        s = scores[model["id"]]
        assert s.shape == (len(single),) and s.dtype == np.float64
        p = apply_calibrator(model["calibrator"], s)
        expected = [g["models"][model["id"]]["p"] for g in single]
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-9)


def test_scores_do_not_depend_on_which_other_texts_are_scored():
    manifest, blob = artifact.load_artifact(TINY)
    many = texts(30)

    together = rescore.quality_scores(manifest, blob, many)
    alone = rescore.quality_scores(manifest, blob, many[7:8])

    for model_id, s in together.items():
        np.testing.assert_allclose(alone[model_id], s[7:8], rtol=0, atol=1e-12)


def test_scores_of_no_texts_are_empty_arrays():
    manifest, blob = artifact.load_artifact(TINY)
    scores = rescore.quality_scores(manifest, blob, [])
    assert {k: v.shape for k, v in scores.items()} == {"tiny-weak": (0,), "tiny-strong": (0,)}


def test_scores_use_the_float32_weights_the_artifact_holds():
    manifest, blob = artifact.load_artifact(TINY)
    model = manifest["models"][0]
    w = artifact.read_array(blob, model["quality"]["weights"])
    assert w.dtype == np.dtype("<f4") and w.shape[0] == manifest["dim"] + features.NUM_DENSE

    text = texts(1)[0]
    idx, val, dense = features.extract([("user", text)], manifest["dim"])
    z = (dense - np.asarray(manifest["dense_mean"])) / np.asarray(manifest["dense_std"])
    by_hand = (
        float(np.dot(w[idx].astype(np.float64), val))
        + float(np.dot(w[manifest["dim"] :].astype(np.float64), z))
        + model["quality"]["bias"]
    )
    got = rescore.quality_scores(manifest, blob, [text])[model["id"]][0]
    assert got == pytest.approx(by_hand, abs=1e-12)


# --- reading split rows from a replay input ---------------------------------------------------


def write_replay(path: Path, rows: list[tuple[str, str, str, dict[str, bool]]]) -> None:
    lines = []
    for item_id, split, prompt, success in rows:
        lines.append(
            json.dumps(
                {
                    "item_id": item_id,
                    "dataset": item_id.split("/")[0],
                    "split": split,
                    "prompt_sha256": "0" * 64,
                    "messages": [{"role": "user", "content": prompt}],
                    "outcomes": {
                        m: {
                            "success": ok,
                            "cost_picousd": 1,
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                        }
                        for m, ok in success.items()
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ROWS = [
    ("a/2", "calib", "second calib", {"m1": True, "m2": False}),
    ("a/1", "calib", "first calib", {"m1": False, "m2": True}),
    ("a/3", "test", "a test row", {"m1": True, "m2": True}),
    ("b/1", "train", "a train row", {"m1": False, "m2": False}),
]


def test_read_split_keeps_only_the_wanted_split_sorted_by_item_id(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    write_replay(path, ROWS)

    got = rescore.read_split(path, "calib", ["m1", "m2"])

    assert list(got.items["item_id"]) == ["a/1", "a/2"]
    assert list(got.items["prompt"]) == ["first calib", "second calib"]
    assert set(got.items["split"]) == {"calib"} and set(got.items["dataset"]) == {"a"}
    assert got.success["m1"].tolist() == [0, 1] and got.success["m2"].tolist() == [1, 0]
    assert got.success["m1"].dtype == np.int64
    test = rescore.read_split(path, "test", ["m2"])
    assert list(test.items["item_id"]) == ["a/3"] and test.success["m2"].tolist() == [1]


def test_read_split_does_not_keep_the_text_of_other_splits(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    write_replay(path, ROWS)
    got = rescore.read_split(path, "calib", ["m1"])
    assert "a test row" not in " ".join(got.items["prompt"]) and len(got.items) == 2


def test_read_split_refuses_odd_rows_and_empty_results(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    write_replay(path, ROWS)
    with pytest.raises(ValueError, match="no items in split"):
        rescore.read_split(path, "heldout", ["m1"])
    with pytest.raises(ValueError, match="no recorded outcome for model"):
        rescore.read_split(path, "calib", ["m1", "missing"])
    with pytest.raises(ValueError, match="splits"):
        rescore.read_split(path, "everything", ["m1"])

    two = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    two["messages"].append({"role": "assistant", "content": "x"})
    path.write_text(json.dumps(two) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one user message"):
        rescore.read_split(path, "calib", ["m1"])
