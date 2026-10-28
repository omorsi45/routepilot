"""Recompute an exported artifact's raw quality scores in Python, and read split rows.

The calibration study and the cross-fitted refit need the raw linear scores ``s`` of each
model, which the Go replay does not report (it reports calibrated numbers). ``quality_scores``
recomputes them from the exported weights: the design matrix of the training pipeline with
the float32 weights widened to float64, the same arithmetic as the reference predictor up to
the order of float64 additions (about 1e-15), so that a calibrator applied to them agrees
with ``artifact.predict_bundle`` to far below any reported digit.

``read_split`` reads one split of a replay input (``data/derived/replay/<pool>.jsonl``): the
prompts and every pool model's 0/1 success. Only the rows of the wanted split are kept in
memory, so a caller that asks for ``calib`` never holds a test prompt. Prompt text stays in
memory; callers write item IDs and numbers only.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routepilot_train import features
from routepilot_train.export import artifact
from routepilot_train.train import design

SPLITS = ("train", "calib", "test", "heldout")


def quality_scores(
    manifest: Mapping[str, Any], blob: bytes, texts: Sequence[str]
) -> dict[str, np.ndarray]:
    """The raw quality score ``x . w + b`` of every model for each text (one user message).

    ``manifest`` and ``blob`` are an exported artifact (``artifact.load_artifact``). The
    result maps a model id to a float64 vector with one score per text.
    """
    hashed, dense = features.extract_texts(list(texts), manifest["dim"])
    x = design.stack(
        hashed, dense, np.asarray(manifest["dense_mean"]), np.asarray(manifest["dense_std"])
    )
    scores: dict[str, np.ndarray] = {}
    for model in manifest["models"]:
        w = artifact.read_array(blob, model["quality"]["weights"]).astype(np.float64)
        scores[model["id"]] = np.asarray(x @ w, dtype=np.float64).ravel() + float(
            model["quality"]["bias"]
        )
    return scores


@dataclass(frozen=True)
class SplitRows:
    """One split of a replay input: ``items`` (``item_id``, ``dataset``, ``split``, ``prompt``,
    sorted by ``item_id``) and each model's 0/1 success aligned with its rows."""

    items: pd.DataFrame
    success: dict[str, np.ndarray]


def read_split(path: Path, split: str, models: Sequence[str]) -> SplitRows:
    """The rows of ``split`` in the replay input at ``path`` with the success of ``models``.

    Raises ``ValueError`` for an unknown split, a split with no items, an item without
    exactly one user message, or a missing outcome of a model.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, the splits are {', '.join(SPLITS)}")
    rows: list[tuple[str, str, str]] = []
    success: dict[str, list[int]] = {m: [] for m in models}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] != split:
                continue
            messages = item["messages"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError(f"item {item['item_id']}: want one user message")
            for model in models:
                if model not in item["outcomes"]:
                    raise ValueError(
                        f"item {item['item_id']}: no recorded outcome for model {model}"
                    )
                success[model].append(int(bool(item["outcomes"][model]["success"])))
            rows.append((item["item_id"], item["dataset"], messages[0]["content"]))
    if not rows:
        raise ValueError(f"no items in split {split!r}")
    frame = pd.DataFrame(rows, columns=["item_id", "dataset", "prompt"])
    frame.insert(2, "split", split)
    order = np.argsort(frame["item_id"].to_numpy(), kind="stable")
    return SplitRows(
        items=frame.iloc[order].reset_index(drop=True),
        success={m: np.asarray(v, dtype=np.int64)[order] for m, v in success.items()},
    )
