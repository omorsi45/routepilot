"""Cross-fitting the calibrators: halves of the calibration split and the artifact refit on one.

The shipped calibrators and lower-bound tables are fitted on the whole calibration split and
the policy's lambda is then tuned on the same items, which makes the tuned policy's
calibration success optimistic. The cross-fitted sensitivity check splits the calibration
items into two halves by a hash of the item ID, refits the calibrators and tables on half A
only (the quality and length weights, the length block and the OOD detector are untouched)
and lets the caller tune lambda on half B, which the refit never saw.

``refit_artifact`` writes a content-addressed artifact like the pipeline does, with its own
version, to a directory the caller owns. Such an artifact is an experiment's input and is
never committed.
"""

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routepilot_train.data.splits import require_splits
from routepilot_train.export import artifact
from routepilot_train.train import calibrate, lcb

SALT = "routepilot-crossfit-v1"


def half_of(item_id: str) -> int:
    """0 for half A and 1 for half B: the parity of ``sha256(item_id + SALT)`` as an integer."""
    digest = hashlib.sha256((item_id + SALT).encode()).hexdigest()
    return int(digest, 16) % 2


def split_halves(item_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks ``(half A, half B)`` over ``item_ids``; every item is in exactly one."""
    halves = np.fromiter((half_of(i) for i in item_ids), dtype=np.int64)
    return halves == 0, halves == 1


def refit_artifact(
    manifest: Mapping[str, Any],
    blob: bytes,
    calib_items: pd.DataFrame,
    raw: Mapping[str, np.ndarray],
    success: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    out_root: Path,
    name: str,
) -> Path:
    """Export a copy of the artifact whose calibrators and lower-bound tables are refit.

    ``calib_items`` has one row per calibration item (its ``split`` column must hold
    ``calib`` only), ``raw[model]`` the model's raw quality scores and ``success[model]`` its
    0/1 outcomes on those items, and ``fit_mask`` marks the rows to fit on. Each model gets
    ``calibrate.fit_calibrator`` (the spec rule: isotonic from 1000 samples, else Platt) on
    its masked scores and ``lcb.fit_lcb`` on the resulting probabilities. Everything else of
    the artifact is kept, so ``weights.bin`` is byte for byte the original. Returns the
    directory ``<out_root>/<name>/<version>``.

    Raises ``ValueError`` before writing anything when a row of ``calib_items`` is not a
    calibration row, the mask has the wrong length or selects nothing, or a model has no
    scores or outcomes.
    """
    require_splits(calib_items, {"calib"})
    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != (len(calib_items),):
        raise ValueError(
            f"the mask has shape {mask.shape} but there are {len(calib_items)} calibration items"
        )
    if not mask.any():
        raise ValueError("the mask selects no items to fit on")
    bundle, weights = artifact.bundle_from_artifact(manifest, blob)
    for model in bundle["models"]:
        model_id = model["id"]
        if model_id not in raw or model_id not in success:
            raise ValueError(f"missing raw scores or outcomes for model {model_id}")
        s, y = np.asarray(raw[model_id])[mask], np.asarray(success[model_id])[mask]
        if s.shape != y.shape or len(s) != int(mask.sum()):
            raise ValueError(f"model {model_id}: scores and outcomes do not match the items")
        calibrator = calibrate.fit_calibrator(s, y)
        model["calibrator"] = calibrator
        model["lcb"] = lcb.fit_lcb(calibrate.apply_calibrator(calibrator, s), y)
    return artifact.export_artifact(Path(out_root), name, bundle, weights)
