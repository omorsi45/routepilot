"""Sparse-data calibration study: how many calibration items does each calibrator need?

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/calibration_study.py --pool flagship

The shipped artifact's quality weights are fixed. For the calibration and the test items the
raw quality score of every model is recomputed in Python from the exported weights
(``rescore.quality_scores``). Then, for each calibration size n in {50, 100, 200, 500, 1000,
2000, all} and each seed 0 to 19, n calibration items are drawn without replacement (the same
items for every model and method of that draw; ``all`` is the whole calibration split, so its
seeds are identical), and each of four calibrators is fitted per model on them:

- ``platt``: a two-parameter logistic map of the raw score;
- ``isotonic``: isotonic regression on the sigmoid of the raw score;
- ``histogram``: 15 equal-mass bins of the raw score, each holding its empirical success rate
  (experiment only, never exported);
- ``switch``: the fixed calibration rule, isotonic from 1000 samples and Platt below.

The test ECE (15 equal-mass bins) is computed per model and pooled over all models' test
forecasts, and reported as a mean and standard deviation over the seeds. A size that is not
smaller than the calibration split (2000 against 1773 items for the flagship pool) is skipped
and said so. Nothing is chosen from the test numbers: every configuration is reported.

Writes bench/results/calibration/<run-id>/{results.json,report.md,ece_vs_n.png,run.json}.
Output holds numbers, model names and item IDs only, never prompt text.
"""

import argparse
import json
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning

from routepilot_train import runinfo
from routepilot_train.evaluate import calibration, report, rescore
from routepilot_train.export import artifact as artifact_io
from routepilot_train.train import calibrate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeps  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "calibration"
SIZES: tuple[int | str, ...] = (50, 100, 200, 500, 1000, 2000, "all")
METHODS = ("platt", "isotonic", "histogram", "switch")
N_SEEDS = 20
ECE_BINS = 15


def draw_indices(n_total: int, size: int | str, seed: int) -> np.ndarray:
    """Sorted indices of ``size`` of ``n_total`` items drawn without replacement.

    ``"all"`` is every item whatever the seed. A numeric size that is not below ``n_total``
    has no draw to make and raises ``ValueError`` (the caller skips such sizes). The generator
    is seeded by ``(seed, size)``, so a draw does not depend on the other sizes in the run.
    """
    if size == "all":
        return np.arange(n_total)
    if not isinstance(size, int) or size < 1 or size >= n_total:
        raise ValueError(f"a size must be an integer in [1, {n_total - 1}] or 'all', got {size!r}")
    rng = np.random.default_rng([seed, size])
    return np.sort(rng.choice(n_total, size=size, replace=False))


def run_cell(
    raw_cal: Mapping[str, np.ndarray],
    y_cal: Mapping[str, np.ndarray],
    raw_test: Mapping[str, np.ndarray],
    y_test: Mapping[str, np.ndarray],
    idx: np.ndarray,
    method: str,
) -> dict[str, Any]:
    """Fit ``method`` per model on the calibration rows ``idx`` and measure the test ECE.

    Returns ``ece`` per model, ``pooled`` (all models' test forecasts together), the
    calibrator ``types`` used per model and the number of ``convergence_warnings`` raised.
    """
    forecasts: dict[str, np.ndarray] = {}
    types: dict[str, str] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for model in raw_cal:
            cal = calibrate.fit_calibrator(raw_cal[model][idx], y_cal[model][idx], method=method)
            forecasts[model] = calibrate.apply_calibrator(cal, raw_test[model])
            types[model] = cal["type"]
    per_model = {m: calibration.ece(forecasts[m], y_test[m], ECE_BINS, "mass") for m in forecasts}
    pooled = calibration.ece(
        np.concatenate([forecasts[m] for m in forecasts]),
        np.concatenate([np.asarray(y_test[m]) for m in forecasts]),
        ECE_BINS,
        "mass",
    )
    return {
        "ece": per_model,
        "pooled": pooled,
        "types": types,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
    }


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation (n - 1 divisor, 0 for one seed) of the cells' test ECE."""

    def spread(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < 2 or bool(np.all(arr == arr[0])):  # identical draws: no spread, exactly
            return float(arr[0]), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    models = list(cells[0]["ece"])
    per_model = {m: spread([c["ece"][m] for c in cells]) for m in models}
    pooled = spread([c["pooled"] for c in cells])
    type_counts: dict[str, int] = {}
    for c in cells:
        for t in c["types"].values():
            type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "n_seeds": len(cells),
        "per_model_mean": {m: v[0] for m, v in per_model.items()},
        "per_model_std": {m: v[1] for m, v in per_model.items()},
        "pooled_mean": pooled[0],
        "pooled_std": pooled[1],
        "pooled_values": [c["pooled"] for c in cells],
        "calibrator_types": type_counts,
        "convergence_warnings": int(sum(c["convergence_warnings"] for c in cells)),
    }


def shipped_reference(
    manifest: Mapping[str, Any], raw_test: Mapping[str, np.ndarray], y_test: Mapping[str, Any]
) -> dict[str, Any]:
    """Test ECE of the artifact's own calibrators (fitted on the whole calibration split)."""
    forecasts = {
        m["id"]: calibrate.apply_calibrator(m["calibrator"], raw_test[m["id"]])
        for m in manifest["models"]
    }
    return {
        "per_model": {
            m: calibration.ece(forecasts[m], y_test[m], ECE_BINS, "mass") for m in forecasts
        },
        "pooled": calibration.ece(
            np.concatenate(list(forecasts.values())),
            np.concatenate([np.asarray(y_test[m]) for m in forecasts]),
            ECE_BINS,
            "mass",
        ),
        "types": {m["id"]: m["calibrator"]["type"] for m in manifest["models"]},
    }


def run_study(
    derived: Path,
    pool_name: str,
    artifact: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    sizes: Sequence[int | str] = SIZES,
    methods: Sequence[str] = METHODS,
    n_seeds: int = N_SEEDS,
) -> Path:
    """Run the study for ``pool_name`` and write the results under ``out_root/<run-id>``."""
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    derived, artifact = Path(derived), Path(artifact)
    manifest, blob = artifact_io.load_artifact(artifact)
    models = [m["id"] for m in manifest["models"]]
    replay_input = derived / "replay" / f"{pool_name}.jsonl"

    calib = rescore.read_split(replay_input, "calib", models)
    raw_cal = rescore.quality_scores(manifest, blob, calib.items["prompt"].tolist())
    n_cal = len(calib.items)
    test = rescore.read_split(replay_input, "test", models)  # the test split is read here, once
    raw_test = rescore.quality_scores(manifest, blob, test.items["prompt"].tolist())
    raw_time = time.perf_counter() - clock

    used = [s for s in sizes if s == "all" or (isinstance(s, int) and s < n_cal)]
    skipped = {
        str(s): f"{s} is not below the {n_cal} calibration items; 'all' is the full split"
        for s in sizes
        if s not in used
    }
    cells: dict[str, dict[str, Any]] = {}
    for size in used:
        cells[str(size)] = {}
        for method in methods:
            runs = [
                run_cell(
                    raw_cal,
                    calib.success,
                    raw_test,
                    test.success,
                    draw_indices(n_cal, size, seed),
                    method,
                )
                for seed in range(n_seeds)
            ]
            cells[str(size)][method] = summarize_cells(runs)
        print(f"size {size} done", file=sys.stderr, flush=True)
    shipped = shipped_reference(manifest, raw_test, test.success)

    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "models": models,
        "n_items": {"calib": n_cal, "test": len(test.items)},
        "sizes": [str(s) for s in used],
        "skipped_sizes": skipped,
        "methods": list(methods),
        "seeds": n_seeds,
        "ece_bins": ECE_BINS,
        "isotonic_min": calibrate.ISOTONIC_MIN,
        "cells": cells,
        "shipped": shipped,
        "wall_time_s": time.perf_counter() - clock,
        "rescoring_time_s": raw_time,
        "labels": _labels(),
    }
    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    shown = (
        list(command)
        if command
        else ["python", "bench/scripts/calibration_study.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {"ece_vs_n": _figure(results)},
        {"subsamples": list(range(n_seeds))},
        shown,
        artifact.name,
        started,
        git=start_git,
    )
    return out_dir


# --- report --------------------------------------------------------------------------------------


def _labels() -> list[str]:
    return [
        "no latency, price or cost is involved in this experiment",
        "the quality weights are those of the shipped artifact; raw scores are recomputed in "
        "Python from its float32 weights",
        "test ECE uses 15 equal-mass bins; every configuration is reported and none is chosen "
        "from the test numbers",
        "histogram binning is an experiment only and is never exported",
        "calibration items are subsampled without replacement; the 'all' rows use the whole "
        "calibration split for every seed, so their spread is zero",
    ]


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    items = results["n_items"]
    rows: list[list[Any]] = [
        ["pool", results["pool"]],
        ["artifact version", results["artifact_version"]],
        ["models", len(results["models"])],
        ["calibration / test items", f"{items['calib']} / {items['test']}"],
        ["sizes run", ", ".join(results["sizes"])],
        ["methods", ", ".join(results["methods"])],
        ["seeds per size", results["seeds"]],
        ["ECE bins", f"{results['ece_bins']} equal-mass"],
        ["switch rule", f"isotonic from {results['isotonic_min']} samples, else Platt"],
        ["shipped calibrators", json.dumps(_count(results["shipped"]["types"].values()))],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    rows += [[f"skipped size {size}", why] for size, why in results["skipped_sizes"].items()]
    warns = sum(
        c["convergence_warnings"]
        for by_method in results["cells"].values()
        for c in by_method.values()
    )
    rows.append(["optimizer convergence warnings (all fits)", warns])
    return pd.DataFrame(rows, columns=["setting", "value"])


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _pooled_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for size in results["sizes"]:
        label = f"all ({results['n_items']['calib']})" if size == "all" else size
        row: dict[str, Any] = {"calibration items": label}
        for method in results["methods"]:
            cell = results["cells"][size][method]
            row[method] = _fmt(cell["pooled_mean"], cell["pooled_std"])
        rows.append(row)
    shipped = {"calibration items": "shipped artifact (whole split)"}
    shipped |= {m: "" for m in results["methods"]}
    shipped["switch"] = f"{results['shipped']['pooled']:.4f}"
    rows.append(shipped)
    return pd.DataFrame(rows)


def _per_model_table(results: Mapping[str, Any], size: str) -> pd.DataFrame:
    rows = []
    for model in [*results["models"], "pooled (all models)"]:
        row: dict[str, Any] = {"model": model}
        for method in results["methods"]:
            cell = results["cells"][size][method]
            if model.startswith("pooled"):
                row[method] = _fmt(cell["pooled_mean"], cell["pooled_std"])
            else:
                row[method] = _fmt(cell["per_model_mean"][model], cell["per_model_std"][model])
        shipped = results["shipped"]
        row["shipped artifact"] = (
            f"{shipped['pooled']:.4f}"
            if model.startswith("pooled")
            else f"{shipped['per_model'][model]:.4f}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _types_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for size in results["sizes"]:
        for method in results["methods"]:
            cell = results["cells"][size][method]
            rows.append(
                {
                    "calibration items": size,
                    "method": method,
                    "fits by calibrator type (models x seeds)": json.dumps(
                        cell["calibrator_types"], sort_keys=True
                    ),
                    "convergence warnings": cell["convergence_warnings"],
                }
            )
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "Setup": _setup_table(results),
        "Pooled test ECE (mean +/- std over seeds)": _pooled_table(results),
    }
    for size in results["sizes"]:
        label = f"all {results['n_items']['calib']}" if size == "all" else size
        tables[f"Test ECE per model, {label} calibration items (mean +/- std over seeds)"] = (
            _per_model_table(results, size)
        )
    tables["Calibrator types and optimizer warnings"] = _types_table(results)
    return tables


def _figure(results: Mapping[str, Any]) -> Figure:
    fig = Figure(figsize=(7.5, 4.8))
    ax = fig.subplots()
    n_cal = results["n_items"]["calib"]
    xs = [n_cal if s == "all" else int(s) for s in results["sizes"]]
    for method in results["methods"]:
        mean = np.array([results["cells"][s][method]["pooled_mean"] for s in results["sizes"]])
        std = np.array([results["cells"][s][method]["pooled_std"] for s in results["sizes"]])
        ax.plot(xs, mean, marker="o", markersize=4, label=method)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
    ax.axhline(
        results["shipped"]["pooled"],
        linestyle="--",
        color="0.5",
        linewidth=0.9,
        label="shipped artifact (whole calibration split)",
    )
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"all\n({n_cal})" if s == "all" else s for s in results["sizes"]])
    ax.set_xlabel("calibration items used")
    ax.set_ylabel("pooled test ECE (15 equal-mass bins)")
    ax.set_title(
        f"Calibration study, {results['pool']} pool: "
        f"mean and std over {results['seeds']} subsamples",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sparse-data calibration study.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "calibration")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_study(
        args.derived,
        args.pool,
        args.artifact or sweeps.find_artifact(ROOT / "policies", args.pool),
        args.out_root,
        command,
        n_seeds=args.seeds,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
