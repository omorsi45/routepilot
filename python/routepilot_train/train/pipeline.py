"""Train a pool's routing policy and export it as a content-addressed artifact.

    python -m routepilot_train.train.pipeline --pool flagship --derived data/derived --out policies

The pipeline reads ``outcomes.parquet``, ``prompts.parquet``, ``splits.parquet`` and
``pools/<pool>.json`` from the derived directory and uses the pool's items that every pool
model has a record for. Only train and calibration rows are ever read: the splits table is
filtered first and the other two files are read with that item list as a filter. Every fit
entry point is then guarded by ``require_splits`` (train rows for the quality, length and
OOD fits and for the dense standardizer, calibration rows for the calibrators, lower
bounds, length residuals and the OOD threshold).

Per model: ``fit_quality`` on train, ``fit_calibrator`` and ``fit_lcb`` on the calibration
scores, ``fit_length`` with ``in_upper_model = prompt_bytes + 24 + prompt_overhead_tokens``.
Then ``fit_ood`` once on the hashed block. The artifact lands in ``<out>/<pool>/<version>/``
and the run in ``bench/results/train/<run-id>/``.

``--calibrator`` picks the calibrator: ``switch`` (the default and the spec rule: isotonic from
1000 calibration samples, else Platt), ``isotonic`` or ``platt``. ``histogram`` exists for the
calibration study only and is refused here, since an artifact cannot hold it.

``--features`` picks the columns the quality and length models are fitted on (the ablation):
``both`` (the default), ``hashed`` or ``dense``. A restricted model is fitted on the restricted
design matrix and exported at full length with exact zeros in the dropped block, so the Go
router serves it unchanged; the OOD detector is always fitted on the hashed part at ``--dim``.
The feature set is recorded in the run (``results.json``, ``run.json``) and not in the manifest,
whose fields Go checks strictly.

``--prompts-file`` names the prompts table in ``--derived`` (default ``prompts.parquet``). When
that table has a ``messages`` column (JSON text of the messages a model is sent), the fits
featurize those messages, roles included, instead of the ``prompt`` column, so that training
sees exactly what the gateway will see.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from routepilot_train import features, runinfo
from routepilot_train.data import selection, splits
from routepilot_train.data.pools import PROMPT_FRAMING_TOKENS, pool_outcomes
from routepilot_train.data.splits import require_splits
from routepilot_train.evaluate import calibration, report
from routepilot_train.export import artifact
from routepilot_train.train import calibrate, design, lcb, length, ood, quality

EXPERIMENT = "train"
PROMPTS_FILE = "prompts.parquet"
MESSAGES_COLUMN = "messages"
FIT_SPLITS = ("train", "calib")
EXPORTABLE_CALIBRATORS = ("switch", "isotonic", "platt")
CALIBRATION_BINS = 15
OUTCOME_COLUMNS = ["item_id", "model", "score", "success", "prompt_tokens", "completion_tokens"]
LABELS = (
    "calib_ece and calib_brier are measured on the calibration split, which is in-sample for "
    "the calibrator and the lower-bound tables",
    "no test or held-out row was read",
    "prices are not used here; nothing in this run is simulated or assumed",
)
FIT_TABLE = (
    "Per-model fit (calib_ece and calib_brier: calibration split, in-sample for the calibrator)"
)


@dataclass(frozen=True)
class SplitData:
    """One split's items and every pool model's outcome arrays, aligned by row.

    ``items`` has ``item_id``, ``dataset``, ``split``, ``prompt`` and ``prompt_bytes``,
    sorted by ``item_id``. The dicts map a model id to one value per item.
    """

    items: pd.DataFrame
    success: dict[str, np.ndarray]
    completion_tokens: dict[str, np.ndarray]
    prompt_tokens: dict[str, np.ndarray]


@dataclass(frozen=True)
class PoolFit:
    """Everything fitted for a pool, ready for ``artifact.assemble_bundle``."""

    dim: int
    dense_mean: np.ndarray
    dense_std: np.ndarray
    models: list[dict[str, Any]]
    ood: dict[str, Any]
    report: list[dict[str, Any]]
    n_train: int
    n_calib: int


def _split_data(
    items: pd.DataFrame, complete: pd.DataFrame, models: Sequence[str], split: str
) -> SplitData:
    part = items.loc[items["split"] == split].sort_values("item_id").reset_index(drop=True)
    if part.empty:
        raise ValueError(f"no {split} items have a record for every pool model")
    rows = complete.loc[complete["item_id"].isin(set(part["item_id"]))]

    def wide(column: str, dtype: type) -> dict[str, np.ndarray]:
        table = rows.pivot(index="item_id", columns="model", values=column).loc[part["item_id"]]
        return {model: table[model].to_numpy(dtype=dtype) for model in models}

    return SplitData(
        items=part,
        success=wide("success", np.int64),
        completion_tokens=wide("completion_tokens", np.int64),
        prompt_tokens=wide("prompt_tokens", np.int64),
    )


def load_fit_data(
    derived: Path, pool: Mapping[str, Any], prompts_file: str = PROMPTS_FILE
) -> tuple[SplitData, SplitData]:
    """Train and calibration data of ``pool``: complete items only, nothing else in memory.

    The splits table is filtered to train and calibration items before ``outcomes.parquet``
    and the prompts file are read with that item list (and the pool's models) as a
    filter, so test and held-out rows are never loaded.

    ``prompts_file`` (default ``prompts.parquet``) names the table of ``item_id``, ``prompt``
    and ``prompt_bytes`` in ``derived``. When it also has a ``messages`` column (JSON text of
    a list of ``{"role", "content"}`` objects, the messages a model is sent), that column is
    kept in ``SplitData.items`` and the fit featurizes it instead of ``prompt``.
    """
    derived = Path(derived)
    models = list(pool["models"])
    table = pd.read_parquet(derived / "splits.parquet", columns=["item_id", "dataset", "split"])
    table = table.loc[table["split"].isin(FIT_SPLITS)]
    ids = sorted(table["item_id"])
    outcomes = pd.read_parquet(
        derived / "outcomes.parquet",
        columns=OUTCOME_COLUMNS,
        filters=[("item_id", "in", ids), ("model", "in", models)],
    )
    prompts_path = derived / prompts_file
    prompt_columns = ["item_id", "prompt", "prompt_bytes"]
    if MESSAGES_COLUMN in pq.read_schema(prompts_path).names:
        prompt_columns.append(MESSAGES_COLUMN)
    prompts = pd.read_parquet(
        prompts_path,
        columns=prompt_columns,
        filters=[("item_id", "in", ids)],
    )
    complete = pool_outcomes(outcomes, models)
    items = prompts.merge(table, on="item_id", validate="1:1")
    lost = set(complete["item_id"]) - set(items["item_id"])
    if lost:
        raise ValueError(f"{len(lost)} pool items have outcomes but no prompt or split")
    items = items.loc[items["item_id"].isin(set(complete["item_id"]))]
    return (
        _split_data(items, complete, models, "train"),
        _split_data(items, complete, models, "calib"),
    )


def _featurize(items: pd.DataFrame, dim: int) -> tuple[Any, np.ndarray]:
    """Hashed and dense features of ``items``: its ``messages`` when it has them, else ``prompt``.

    A ``messages`` cell is JSON text of the messages a model is sent, so the fit sees what
    the gateway will see (system message and message count included). Without the column
    each ``prompt`` is one user message.
    """
    if MESSAGES_COLUMN in items.columns:
        rows = [
            [(str(m["role"]), str(m["content"])) for m in json.loads(cell)]
            for cell in items[MESSAGES_COLUMN]
        ]
        return features.extract_rows(rows, dim)
    return features.extract_texts(items["prompt"].tolist(), dim)


def fit_pool(
    train: SplitData,
    calib: SplitData,
    pool: Mapping[str, Any],
    dim: int = features.DEFAULT_DIM,
    progress: Callable[[str], None] | None = None,
    calibrator: str = "switch",
    feature_set: str = "both",
) -> PoolFit:
    """Fit every model of ``pool`` plus the OOD detector.

    Raises ``ValueError`` before any fit when ``train.items`` holds a row whose split is not
    ``train`` or ``calib.items`` a row whose split is not ``calib``. ``progress`` is called
    with one short line per fitted model. ``calibrator`` is ``switch`` (the spec rule, the
    default), ``isotonic`` or ``platt``; ``histogram`` is refused (see ``check_calibrator``).
    ``feature_set`` is ``both`` (the default), ``hashed`` or ``dense``: the quality and length
    models are fitted on those columns only and their weights are expanded to full length with
    zeros in the dropped block; the OOD detector always sees the hashed part.
    """
    check_calibrator(calibrator)
    design.check_feature_set(feature_set)
    require_splits(train.items, {"train"})
    require_splits(calib.items, {"calib"})
    models = list(pool["models"])
    overheads = pool.get("prompt_overhead_tokens", {})

    h_train, d_train = _featurize(train.items, dim)
    h_calib, d_calib = _featurize(calib.items, dim)
    mean, std = design.standardizer(d_train)
    x_train = design.select_features(h_train, d_train, mean, std, feature_set)
    x_calib = design.select_features(h_calib, d_calib, mean, std, feature_set)
    prompt_bytes = train.items["prompt_bytes"].to_numpy(dtype=np.int64)

    fitted, rows = [], []
    for model in models:
        y_calib = calib.success[model]
        q = quality.fit_quality(x_train, train.success[model])
        s_calib = quality.raw_score(q, x_calib)
        calibrator_fit = calibrate.fit_calibrator(s_calib, y_calib, method=calibrator)
        p_calib = calibrate.apply_calibrator(calibrator_fit, s_calib)
        table = lcb.fit_lcb(p_calib, y_calib)
        in_upper = prompt_bytes + PROMPT_FRAMING_TOKENS + int(overheads.get(model, 0))
        fitted_length = length.fit_length(
            x_train,
            train.completion_tokens[model],
            x_calib,
            calib.completion_tokens[model],
            train.prompt_tokens[model],
            in_upper,
        )
        fitted.append(
            {
                "id": model,
                "quality": {**q, "coef": design.expand_weights(q["coef"], dim, feature_set)},
                "calibrator": calibrator_fit,
                "lcb": table,
                "length": {
                    **fitted_length,
                    "coef": design.expand_weights(fitted_length["coef"], dim, feature_set),
                },
            }
        )
        row = {
            "model": model,
            "C": q["C"],
            "cv_log_loss": q["cv_log_loss"],
            "alpha": fitted_length["alpha"],
            "calibrator": calibrator_fit["type"],
            "calib_ece": calibration.ece(p_calib, y_calib, CALIBRATION_BINS, "mass"),
            "calib_brier": calibration.brier(p_calib, y_calib),
            "calib_success_rate": float(np.mean(y_calib)),
            "lcb_bins": len(table["upper"]),
            "length_smear": fitted_length["smear"],
            "in_ratio": fitted_length["in_ratio"],
        }
        rows.append(row)
        if progress is not None:
            progress(
                f"{model}: C={row['C']} alpha={row['alpha']} "
                f"calibrator={row['calibrator']} calib_ece={row['calib_ece']:.4f}"
            )
    detector = ood.fit_ood(h_train, h_calib)
    return PoolFit(
        dim=dim,
        dense_mean=mean,
        dense_std=std,
        models=fitted,
        ood=detector,
        report=rows,
        n_train=len(train.items),
        n_calib=len(calib.items),
    )


def _results(
    pool_name: str,
    version: str,
    fit: PoolFit,
    wall_time: float,
    calibrator: str,
    feature_set: str = "both",
) -> dict[str, Any]:
    return {
        "pool": pool_name,
        "artifact_version": version,
        "calibrator": calibrator,
        "features": feature_set,
        "dim": fit.dim,
        "n_train": fit.n_train,
        "n_calib": fit.n_calib,
        "ood": {"k": fit.ood["k"], "threshold": fit.ood["threshold"]},
        "wall_time_s": wall_time,
        "models": {
            row["model"]: {k: v for k, v in row.items() if k != "model"} for row in fit.report
        },
        "labels": list(LABELS),
    }


def _tables(
    pool_name: str, version: str, fit: PoolFit, wall_time: float, feature_set: str = "both"
) -> dict[str, Any]:
    summary = [
        ("pool", pool_name),
        ("artifact version", version),
        ("dim", fit.dim),
        ("features", feature_set),
        ("train items", fit.n_train),
        ("calibration items", fit.n_calib),
        ("ood k", fit.ood["k"]),
        ("ood threshold (0.99 quantile of calibration scores)", fit.ood["threshold"]),
        ("wall time (s)", round(wall_time, 1)),
    ]
    return {
        FIT_TABLE: pd.DataFrame(fit.report),
        "Artifact": pd.DataFrame(summary, columns=["key", "value"]),
    }


def check_calibrator(method: str) -> None:
    """Raise ``ValueError`` unless ``method`` is a calibrator an artifact can hold.

    ``switch`` (the spec rule), ``isotonic`` and ``platt`` can be exported. ``histogram`` is a
    calibration-study experiment and the artifact format has no place for it.
    """
    if method == "histogram":
        raise ValueError(
            "the histogram calibrator is an experiment only and is never exported; "
            f"choose one of {', '.join(EXPORTABLE_CALIBRATORS)}"
        )
    if method not in EXPORTABLE_CALIBRATORS:
        raise ValueError(
            f"unknown calibrator {method!r}, want one of {', '.join(EXPORTABLE_CALIBRATORS)}"
        )


def run(
    pool_name: str,
    derived: Path,
    out: Path,
    dim: int = features.DEFAULT_DIM,
    results_root: Path | None = None,
    created_at: str | None = None,
    progress: Callable[[str], None] | None = None,
    calibrator: str = "switch",
    feature_set: str = "both",
    prompts_file: str = PROMPTS_FILE,
) -> dict[str, Any]:
    """Train ``pool_name``, export its artifact and write the run under ``results_root``.

    ``results_root`` defaults to ``bench/results`` of this checkout; the run goes to
    ``<results_root>/train/<run-id>/``. ``created_at`` (UTC ISO text) is recorded in the
    manifest and is the only input the artifact depends on besides the data; it defaults to
    now. ``calibrator`` is ``switch`` (the spec rule: isotonic from 1000 calibration samples,
    else Platt), ``isotonic`` or ``platt``. ``feature_set`` is ``both``, ``hashed`` or ``dense``
    (see ``fit_pool``). ``prompts_file`` names the prompts table in ``derived`` (see
    ``load_fit_data``). Returns the version, the artifact and results directories and the wall
    time.
    """
    check_calibrator(calibrator)
    design.check_feature_set(feature_set)
    started = datetime.now(UTC)
    clock = time.perf_counter()
    derived = Path(derived)
    pool_path = derived / "pools" / f"{pool_name}.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))

    train, calib = load_fit_data(derived, pool, prompts_file)
    fit = fit_pool(train, calib, pool, dim, progress, calibrator, feature_set)
    git_sha, _ = runinfo.git_state()
    bundle, weights = artifact.assemble_bundle(
        dim=fit.dim,
        dense_mean=fit.dense_mean,
        dense_std=fit.dense_std,
        models=fit.models,
        ood=fit.ood,
        training={
            "pool": pool_name,
            "dataset_sha256": selection.DATASET_SHA256,
            "split_salt": splits.SALT,
            "git_sha": git_sha,
            "created_at": created_at,
        },
    )
    artifact_dir = artifact.export_artifact(Path(out), pool_name, bundle, weights)
    wall_time = time.perf_counter() - clock

    root = Path(results_root) if results_root is not None else runinfo.repo_root() / "bench/results"
    results_dir = root / EXPERIMENT / runinfo.new_run_id(started)
    command = [
        "python",
        "-m",
        "routepilot_train.train.pipeline",
        "--pool",
        pool_name,
        "--derived",
        derived.as_posix(),
        "--out",
        Path(out).as_posix(),
        "--dim",
        str(dim),
        "--calibrator",
        calibrator,
        "--features",
        feature_set,
    ]
    if prompts_file != PROMPTS_FILE:
        command += ["--prompts-file", prompts_file]
    version = artifact_dir.name
    report.write_results(
        results_dir,
        EXPERIMENT,
        _results(pool_name, version, fit, wall_time, calibrator, feature_set),
        _tables(pool_name, version, fit, wall_time, feature_set),
        {},
        {"split_salt": splits.SALT, "kfold": 0, "kmeans": 0},
        command,
        version,
        started,
    )
    return {
        "version": version,
        "artifact_dir": artifact_dir,
        "results_dir": results_dir,
        "wall_time_s": wall_time,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", required=True, help="pool name, e.g. flagship")
    parser.add_argument("--derived", type=Path, default=Path("data/derived"))
    parser.add_argument("--out", type=Path, default=Path("policies"))
    parser.add_argument("--dim", type=int, default=features.DEFAULT_DIM)
    parser.add_argument(
        "--results-root", type=Path, default=None, help="default: bench/results of this checkout"
    )
    parser.add_argument(
        "--calibrator",
        choices=calibrate.METHODS,
        default="switch",
        help="switch (default) is the spec rule: isotonic from 1000 calibration samples, else "
        "Platt; isotonic and platt force one; histogram is a study-only experiment and is "
        "refused here",
    )
    parser.add_argument(
        "--features",
        choices=design.FEATURE_SETS,
        default="both",
        help="the columns the quality and length models are fitted on (the ablation): both "
        "(default), hashed or dense; the artifact keeps full-length weights with zeros in the "
        "dropped block",
    )
    parser.add_argument(
        "--prompts-file",
        default=PROMPTS_FILE,
        help="the prompts table in --derived (default prompts.parquet); a messages column in it "
        "is featurized instead of the prompt column",
    )
    args = parser.parse_args(argv)

    def progress(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    summary = run(
        args.pool,
        args.derived,
        args.out,
        args.dim,
        args.results_root,
        progress=progress,
        calibrator=args.calibrator,
        feature_set=args.features,
        prompts_file=args.prompts_file,
    )
    print(f"artifact version: {summary['version']}")
    print(f"artifact: {summary['artifact_dir'].as_posix()}")
    print(f"results: {summary['results_dir'].as_posix()}")
    print(f"wall time: {summary['wall_time_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
