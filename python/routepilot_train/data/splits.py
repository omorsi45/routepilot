"""Deterministic train, calibration, test and held-out splits.

A split depends only on the item id (and the dataset's held-out status), never
on the other rows, so adding or removing items never moves an existing one.
The rule is fixed before any model is trained.
"""

import hashlib
from collections.abc import Collection, Mapping

import pandas as pd

from routepilot_train.data.selection import DOMAINS

SALT = "routepilot-split-v1"
HELDOUT_COUNT = 3
TRAIN_BELOW = 0.6
CALIB_BELOW = 0.8


def _digest(text: str) -> str:
    return hashlib.sha256((text + SALT).encode()).hexdigest()


def heldout_datasets(datasets: list[str], domains: Mapping[str, str] | None = None) -> list[str]:
    """Pick up to three whole datasets to hold out, at most one per domain.

    Datasets are taken in order of ``sha256(name + SALT)``; one whose domain is
    already taken is skipped. ``domains`` defaults to ``selection.DOMAINS``.
    """
    domain_of = DOMAINS if domains is None else domains
    unknown = sorted(name for name in datasets if name not in domain_of)
    if unknown:
        raise ValueError(f"datasets without a domain: {unknown}")
    chosen: list[str] = []
    taken: set[str] = set()
    for name in sorted(datasets, key=_digest):
        if domain_of[name] in taken:
            continue
        chosen.append(name)
        taken.add(domain_of[name])
        if len(chosen) == HELDOUT_COUNT:
            break
    return chosen


def assign_splits(items: pd.DataFrame, domains: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Add a ``split`` column (train, calib, test or heldout) to ``items``.

    ``items`` needs ``item_id`` and ``dataset``. Items of a held-out dataset get
    ``heldout``; every other item is placed by ``sha256(item_id + SALT)``:
    the first 16 hex digits as a fraction of 2**64 below 0.6 is train, below 0.8
    is calib, otherwise test.
    """
    held = set(heldout_datasets(sorted(items["dataset"].unique()), domains))
    out = items.copy()
    out["split"] = [
        "heldout" if dataset in held else _place(item_id)
        for item_id, dataset in zip(items["item_id"], items["dataset"], strict=True)
    ]
    return out


def _place(item_id: str) -> str:
    u = int(_digest(item_id)[:16], 16) / 2**64
    if u < TRAIN_BELOW:
        return "train"
    if u < CALIB_BELOW:
        return "calib"
    return "test"


def place(item_id: str) -> str:
    """The split (``train``, ``calib`` or ``test``) one item id gets from the salt rule.

    This is the rule ``assign_splits`` applies to the items of a dataset that is not held out;
    an experiment with no held-out dataset (the live one) uses it directly.
    """
    return _place(item_id)


def require_splits(df: pd.DataFrame, allowed: Collection[str]) -> None:
    """Raise ValueError unless every row of ``df`` is in one of the ``allowed`` splits.

    Every fitting entry point calls this with its frame so a tuning step cannot
    read test or held-out rows by accident.
    """
    if "split" not in df.columns:
        raise ValueError("frame has no 'split' column")
    offenders = sorted(
        {
            "<missing>" if pd.isna(s) else str(s)
            for s in df["split"].unique()
            if pd.isna(s) or s not in allowed
        }
    )
    if offenders:
        raise ValueError(f"splits not allowed here: {offenders}; allowed: {sorted(allowed)}")
