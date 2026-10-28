"""How tight is ``in_upper``? Recorded prompt tokens against the router's upper bound.

Run from the repository root after ``make data``:

    python bench/scripts/inupper.py

The gateway admits and prices a request with ``in_upper``, an upper bound on its prompt
tokens derived from the bytes it forwards (a token is at least one byte of text) plus fixed
framing. This replays the ``strongest`` policy on every item of the flagship and mixed replay
files, joins each recorded (item, model) ``prompt_tokens`` against that item's ``in_upper``
and reports the ratio ``prompt_tokens / in_upper``: its maximum, the 50th, 99th and 99.9th
percentiles, and how many records exceed 1, overall, per dataset and per model. A ratio
above 1 means the bound was too low for that record.

Ruling V1-R24 lets a model declare the hidden prompt tokens its provider adds, so the bound
the gateway holds a model to is ``in_upper + prompt_overhead_tokens(model)``. The overheads
are read from ``pools/<pool>.json`` (fitted on train and calibration records only). The
adjusted ratio ``prompt_tokens / (in_upper + overhead)`` is reported per split; the test and
held-out rows are the out-of-sample check. Records far above what the other models recorded
for the same item are anomalies: counted and listed per split, and also shown with them left
out. A sensitivity row refits the overheads with no record excluded.

Records with ``prompt_tokens <= 0`` (the dataset has a few, kept as recorded) cannot exceed
the bound; they are counted and left out of the ratio statistics. Writes
bench/results/inupper/<run-id>/{results.json,report.md,run.json}. The output holds numbers,
model names, dataset names and item IDs only, never prompt text.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routepilot_train import replaytool, runinfo
from routepilot_train.data import pools as pool_fit
from routepilot_train.evaluate import report

ROOT = Path(__file__).resolve().parents[2]
POOLS = ("flagship", "mixed")
PERCENTILES = (50.0, 99.0, 99.9)
WORST = 20
SPLITS = ("train", "calib", "test", "heldout")
FIT_SPLITS = ("train", "calib")
MAX_LISTED_ANOMALIES = 20


def recorded_prompt_tokens(replay_input: Path) -> pd.DataFrame:
    """One row per (item, model): ``item_id``, ``dataset``, ``model``, ``prompt_tokens``.

    Reads the replay export, which holds prompt text; only the ids, names and token
    counts are kept.
    """
    rows: list[tuple[str, str, str, int]] = []
    with replay_input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            for model, outcome in item["outcomes"].items():
                rows.append(
                    (item["item_id"], item["dataset"], model, int(outcome["prompt_tokens"]))
                )
    return pd.DataFrame(rows, columns=["item_id", "dataset", "model", "prompt_tokens"])


def ratio_stats(ratio: np.ndarray) -> dict[str, Any]:
    """Maximum, percentiles and the count above 1 of positive-token ratios."""
    if ratio.size == 0:
        return {"n": 0, "max": None, "p50": None, "p99": None, "p99_9": None, "count_gt_1": 0}
    p50, p99, p999 = np.percentile(ratio, PERCENTILES)
    return {
        "n": int(ratio.size),
        "max": float(ratio.max()),
        "p50": float(p50),
        "p99": float(p99),
        "p99_9": float(p999),
        "count_gt_1": int((ratio > 1.0).sum()),
    }


def coverage_stats(ratio: np.ndarray) -> dict[str, Any]:
    """``ratio_stats`` plus ``coverage``: the share of records at or below 1 (None if none)."""
    stats = ratio_stats(ratio)
    stats["coverage"] = None if stats["n"] == 0 else (stats["n"] - stats["count_gt_1"]) / stats["n"]
    return stats


def _grouped(frame: pd.DataFrame, key: str) -> dict[str, Any]:
    return {
        str(name): ratio_stats(group["ratio"].to_numpy())
        for name, group in frame.groupby(key, sort=True)
    }


def pool_section(
    tokens: pd.DataFrame,
    in_upper: pd.DataFrame,
    overheads: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Statistics of one pool from its recorded tokens and the replay's ``in_upper``.

    ``in_upper`` is the replay output (``item_id``, ``split``, ``in_upper``). With
    ``overheads`` (model to hidden prompt tokens) the section also holds the ``adjusted``
    statistics and the ``sensitivity_no_exclusion`` refit.
    """
    joined = tokens.merge(
        in_upper[["item_id", "split", "in_upper"]], on="item_id", how="left", validate="m:1"
    )
    if joined["in_upper"].isna().any():
        raise ValueError("some recorded items have no in_upper from the replay")
    positive = joined.loc[joined["prompt_tokens"] > 0].copy()
    positive["ratio"] = positive["prompt_tokens"] / positive["in_upper"]
    worst = positive.sort_values(["ratio", "item_id", "model"], ascending=[False, True, True])
    section: dict[str, Any] = {
        "items": int(in_upper["item_id"].nunique()),
        "records": len(joined),
        "records_nonpositive_prompt_tokens": int((joined["prompt_tokens"] <= 0).sum()),
        "overall": ratio_stats(positive["ratio"].to_numpy()),
        "per_dataset": _grouped(positive, "dataset"),
        "per_model": _grouped(positive, "model"),
        "worst": [
            {
                "item_id": r.item_id,
                "dataset": r.dataset,
                "model": r.model,
                "prompt_tokens": int(r.prompt_tokens),
                "in_upper": int(r.in_upper),
                "ratio": float(r.ratio),
            }
            for r in worst.head(WORST).itertuples(index=False)
            if r.ratio > 1.0
        ],
    }
    if overheads is not None:
        section["adjusted"] = adjusted_section(joined, overheads)
        section["sensitivity_no_exclusion"] = sensitivity_section(joined)
    return section


def _bound(joined: pd.DataFrame, overheads: Mapping[str, int]) -> pd.Series:
    """The bound each record is held to: ``in_upper`` plus its model's overhead."""
    missing = sorted(set(joined["model"]) - set(overheads))
    if missing:
        raise ValueError(f"no prompt overhead for: {', '.join(missing)}")
    return joined["in_upper"] + joined["model"].map(dict(overheads)).astype("int64")


def _check_splits(joined: pd.DataFrame) -> None:
    unknown = sorted(set(joined["split"]) - set(SPLITS))
    if unknown:
        raise ValueError(f"splits not in {list(SPLITS)}: {unknown}")


def _anomaly_row(row: Any) -> dict[str, Any]:
    return {
        "item_id": str(row.item_id),
        "dataset": str(row.dataset),
        "model": str(row.model),
        "prompt_tokens": int(row.prompt_tokens),
        "others_median": float(row.others_median),
    }


def adjusted_section(joined: pd.DataFrame, overheads: Mapping[str, int]) -> dict[str, Any]:
    """Ratios against ``in_upper + overhead``, per split, with the anomalies of each split."""
    _check_splits(joined)
    joined = joined.reset_index(drop=True)
    flagged = pool_fit.token_anomalies(joined)
    joined["anomaly"] = joined.index.isin(flagged.index)
    joined["bound"] = _bound(joined, overheads)
    positive = joined.loc[joined["prompt_tokens"] > 0].copy()
    positive["ratio"] = positive["prompt_tokens"] / positive["bound"]

    per_split: dict[str, Any] = {}
    for split in SPLITS:
        rows = positive.loc[positive["split"] == split]
        found = flagged.loc[flagged["split"] == split].copy()
        found["size"] = found["prompt_tokens"] / found["others_median"]
        found = found.sort_values(["size", "item_id", "model"], ascending=[False, True, True])
        by_model = found["model"].value_counts().sort_index()
        per_split[split] = {
            **coverage_stats(rows["ratio"].to_numpy()),
            "n_anomalies": len(found),
            "anomalies_by_model": {str(model): int(count) for model, count in by_model.items()},
            "anomalies": [
                _anomaly_row(r) for r in found.head(MAX_LISTED_ANOMALIES).itertuples(index=False)
            ],
            "without_anomalies": coverage_stats(rows.loc[~rows["anomaly"], "ratio"].to_numpy()),
        }

    above = positive.loc[positive["ratio"] > 1.0]
    above = above.sort_values(["ratio", "item_id", "model"], ascending=[False, True, True])
    return {
        "bound": "in_upper + prompt_overhead_tokens(model)",
        "prompt_overhead_tokens": {str(m): int(v) for m, v in sorted(overheads.items())},
        "per_split": per_split,
        "worst": [
            {
                "item_id": r.item_id,
                "dataset": r.dataset,
                "split": r.split,
                "model": r.model,
                "prompt_tokens": int(r.prompt_tokens),
                "in_upper": int(r.in_upper),
                "prompt_overhead": int(r.bound - r.in_upper),
                "bound": int(r.bound),
                "ratio": float(r.ratio),
                "anomaly": bool(r.anomaly),
            }
            for r in above.head(WORST).itertuples(index=False)
        ],
    }


def sensitivity_section(joined: pd.DataFrame) -> dict[str, Any]:
    """Refit each overhead as the largest excess over ``in_upper`` with nothing excluded.

    Fitted on the train and calibration records only, then applied to every split, so the
    test and held-out rows stay out-of-sample.
    """
    _check_splits(joined)
    fit = joined.loc[joined["split"].isin(FIT_SPLITS) & (joined["prompt_tokens"] > 0)]
    excess = (fit["prompt_tokens"] - fit["in_upper"]).clip(lower=0)
    largest = excess.groupby(fit["model"]).max()
    refit = {str(m): int(largest.get(m, 0)) for m in sorted(joined["model"].unique())}
    positive = joined.loc[joined["prompt_tokens"] > 0].copy()
    positive["ratio"] = positive["prompt_tokens"] / _bound(positive, refit)
    return {
        "prompt_overhead_tokens": refit,
        "per_split": {
            split: coverage_stats(positive.loc[positive["split"] == split, "ratio"].to_numpy())
            for split in SPLITS
        },
    }


def _fitted_overheads(derived: Path, name: str) -> dict[str, int]:
    path = derived / "pools" / f"{name}.json"
    overheads = json.loads(path.read_text(encoding="utf-8")).get("prompt_overhead_tokens")
    if overheads is None:
        raise ValueError(
            f"{path.name} has no prompt_overhead_tokens; "
            "rebuild with python -m routepilot_train.data.build --derived data/derived"
        )
    return {str(model): int(tokens) for model, tokens in overheads.items()}


def run_inupper(
    derived: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    pools: Sequence[str] = POOLS,
) -> Path:
    """Compute the ratios for ``pools`` and write the results under ``out_root/<run-id>``."""
    started = datetime.now(UTC)
    overheads = {name: _fitted_overheads(derived, name) for name in pools}
    binary = binary or replaytool.build_go(ROOT)
    sections: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in pools:
            replay_input = derived / "replay" / f"{name}.jsonl"
            replayed = replaytool.replay(
                binary,
                derived / "pools" / f"{name}.config.yaml",
                replay_input,
                "all",
                "strongest",
                Path(scratch) / f"{name}.jsonl",
            )
            sections[name] = pool_section(
                recorded_prompt_tokens(replay_input), replayed, overheads[name]
            )

    results: dict[str, Any] = {
        "ratio": "recorded prompt_tokens / in_upper, over records with prompt_tokens > 0",
        "adjusted_ratio": (
            "recorded prompt_tokens / (in_upper + prompt_overhead_tokens(model)), "
            "over records with prompt_tokens > 0"
        ),
        "anomaly_rule": (
            f"prompt_tokens above {pool_fit.ANOMALY_FACTOR} times the median prompt_tokens "
            "of the other models on the same item"
        ),
        "percentiles": list(PERCENTILES),
        "pools": sections,
        "max_ratio": max(s["overall"]["max"] for s in sections.values() if s["overall"]["n"]),
        "any_above_1": any(s["overall"]["count_gt_1"] > 0 for s in sections.values()),
        "labels": [
            "in_upper is the router's upper bound on prompt tokens, computed from the prompt bytes",
            "prompt_tokens are the counts recorded in the dataset; nothing here is simulated",
            "prompt_overhead_tokens are fitted on train and calib records only, so the test "
            "and heldout rows are an out-of-sample check",
        ],
    }
    out_dir = out_root / runinfo.new_run_id(started)
    report.write_results(
        out_dir,
        "inupper",
        results,
        _tables(sections),
        {},
        {},
        command or ["python", "bench/scripts/inupper.py"],
        None,
        started,
    )
    return out_dir


def _row(label: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        **label,
        "records": stats["n"],
        "max_ratio": stats["max"],
        "p50": stats["p50"],
        "p99": stats["p99"],
        "p99.9": stats["p99_9"],
        "count_above_1": stats["count_gt_1"],
    }


def _tables(sections: dict[str, Any]) -> dict[str, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                **_row({"pool": name}, s["overall"]),
                "nonpositive_prompt_tokens": s["records_nonpositive_prompt_tokens"],
            }
            for name, s in sections.items()
        ]
    )
    by_dataset = pd.DataFrame(
        [
            _row({"pool": name, "dataset": dataset}, stats)
            for name, s in sections.items()
            for dataset, stats in s["per_dataset"].items()
        ]
    )
    by_model = pd.DataFrame(
        [
            _row({"pool": name, "model": model}, stats)
            for name, s in sections.items()
            for model, stats in s["per_model"].items()
        ]
    )
    tables = {
        "Ratio of recorded prompt tokens to in_upper, per pool": overall,
        "Per dataset": by_dataset,
        "Per model": by_model,
    }
    worst = pd.DataFrame([{"pool": name, **w} for name, s in sections.items() for w in s["worst"]])
    if not worst.empty:
        tables["Records above 1 (worst first)"] = worst
    tables.update(_adjusted_tables(sections))
    return tables


def _adjusted_tables(sections: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    overhead_rows = []
    split_rows = []
    sensitivity_rows = []
    anomaly_counts = []
    anomaly_listed = []
    above = []
    for name, s in sections.items():
        adjusted, sensitivity = s["adjusted"], s["sensitivity_no_exclusion"]
        for model, tokens in adjusted["prompt_overhead_tokens"].items():
            overhead_rows.append(
                {
                    "pool": name,
                    "model": model,
                    "prompt_overhead_tokens": tokens,
                    "refit_without_exclusion": sensitivity["prompt_overhead_tokens"][model],
                }
            )
        for split, stats in adjusted["per_split"].items():
            clean = stats["without_anomalies"]
            split_rows.append(
                {
                    **_row({"pool": name, "split": split}, stats),
                    "coverage": stats["coverage"],
                    "anomalies": stats["n_anomalies"],
                    "count_above_1_without_anomalies": clean["count_gt_1"],
                    "coverage_without_anomalies": clean["coverage"],
                }
            )
            for model, count in stats["anomalies_by_model"].items():
                anomaly_counts.append(
                    {"pool": name, "split": split, "model": model, "anomalies": count}
                )
            anomaly_listed += [{"pool": name, "split": split, **a} for a in stats["anomalies"]]
        for split, stats in sensitivity["per_split"].items():
            sensitivity_rows.append(
                {
                    **_row({"pool": name, "split": split}, stats),
                    "coverage": stats["coverage"],
                }
            )
        above += [{"pool": name, **w} for w in adjusted["worst"]]
    tables["Prompt overhead per model"] = pd.DataFrame(overhead_rows)
    tables["Adjusted ratio per split"] = pd.DataFrame(split_rows)
    tables["Sensitivity: overhead fitted without excluding any record"] = pd.DataFrame(
        sensitivity_rows
    )
    if above:
        tables["Records above 1 after the overhead (worst first)"] = pd.DataFrame(above)
    if anomaly_counts:
        tables["Anomalous records"] = pd.DataFrame(anomaly_counts)
        tables["Largest anomalous records"] = pd.DataFrame(anomaly_listed)
    return tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare recorded prompt tokens with in_upper.")
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "inupper")
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_inupper(args.derived, args.out_root, command)
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
