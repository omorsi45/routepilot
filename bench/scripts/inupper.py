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

Records with ``prompt_tokens <= 0`` (the dataset has a few, kept as recorded) cannot exceed
the bound; they are counted and left out of the ratio statistics. Writes
bench/results/inupper/<run-id>/{results.json,report.md,run.json}. The output holds numbers,
model names, dataset names and item IDs only, never prompt text.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routepilot_train import replaytool, runinfo
from routepilot_train.evaluate import report

ROOT = Path(__file__).resolve().parents[2]
POOLS = ("flagship", "mixed")
PERCENTILES = (50.0, 99.0, 99.9)
WORST = 20


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


def _grouped(frame: pd.DataFrame, key: str) -> dict[str, Any]:
    return {
        str(name): ratio_stats(group["ratio"].to_numpy())
        for name, group in frame.groupby(key, sort=True)
    }


def pool_section(tokens: pd.DataFrame, in_upper: pd.DataFrame) -> dict[str, Any]:
    """Statistics of one pool from its recorded tokens and the replay's ``in_upper``."""
    joined = tokens.merge(
        in_upper[["item_id", "in_upper"]], on="item_id", how="left", validate="m:1"
    )
    if joined["in_upper"].isna().any():
        raise ValueError("some recorded items have no in_upper from the replay")
    positive = joined.loc[joined["prompt_tokens"] > 0].copy()
    positive["ratio"] = positive["prompt_tokens"] / positive["in_upper"]
    worst = positive.sort_values(["ratio", "item_id", "model"], ascending=[False, True, True])
    return {
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


def run_inupper(
    derived: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    pools: Sequence[str] = POOLS,
) -> Path:
    """Compute the ratios for ``pools`` and write the results under ``out_root/<run-id>``."""
    started = datetime.now(UTC)
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
            sections[name] = pool_section(recorded_prompt_tokens(replay_input), replayed)

    results: dict[str, Any] = {
        "ratio": "recorded prompt_tokens / in_upper, over records with prompt_tokens > 0",
        "percentiles": list(PERCENTILES),
        "pools": sections,
        "max_ratio": max(s["overall"]["max"] for s in sections.values() if s["overall"]["n"]),
        "any_above_1": any(s["overall"]["count_gt_1"] > 0 for s in sections.values()),
        "labels": [
            "in_upper is the router's upper bound on prompt tokens, computed from the prompt bytes",
            "prompt_tokens are the counts recorded in the dataset; nothing here is simulated",
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
