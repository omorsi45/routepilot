"""Build the splits, model pools, replay configs and replay exports from the ingest.

Reads ``outcomes.parquet`` and ``prompts.parquet`` from the derived directory and
writes, next to them:

- ``splits.parquet``: ``item_id``, ``dataset``, ``split`` for every item.
- ``pools/<pool>.json`` and ``pools/<pool>.config.yaml`` for ``flagship`` and ``mixed``. The
  config's ``rules`` routing policy carries the static rules mapped on train and calibration.
  Both carry each model's ``prompt_overhead_tokens`` (fitted on train and
  calibration); the JSON also lists the ``anomalies`` left out of that fit.
- ``replay/<pool>.jsonl``: the only output that holds prompt text.

Pool selection and fitting finish before the first file is written.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from routepilot_train.baselines.static_rules import RULES, map_rules
from routepilot_train.data import selection
from routepilot_train.data.export import export_replay
from routepilot_train.data.pools import (
    build_pool,
    pick_seven_b,
    pool_outcomes,
    prompt_overheads,
    write_pool_config,
)
from routepilot_train.data.splits import assign_splits

SEVEN_B_COUNT = 4


def build(derived: Path) -> dict[str, Any]:
    """Build every derived table under ``derived`` and return counts only."""
    outcomes = pd.read_parquet(derived / "outcomes.parquet")
    prompts = pd.read_parquet(derived / "prompts.parquet")

    items = outcomes[["item_id", "dataset"]].drop_duplicates().sort_values("item_id")
    splits = assign_splits(items.reset_index(drop=True))

    flagship_rows = pool_outcomes(outcomes, selection.FLAGSHIP_MODELS)
    flagship = build_pool("flagship", flagship_rows, splits)

    seven_b = pick_seven_b(outcomes, splits, n=SEVEN_B_COUNT)
    if len(seven_b) < SEVEN_B_COUNT:
        raise ValueError(
            f"need {SEVEN_B_COUNT} small models with recorded cost for the mixed pool, "
            f"found {len(seven_b)}"
        )
    mixed_rows = pool_outcomes(outcomes, [*selection.FLAGSHIP_MODELS, *seven_b])
    mixed = build_pool("mixed", mixed_rows, splits, fitted_cost_models=seven_b)

    pool_rows = ((flagship, flagship_rows, ()), (mixed, mixed_rows, seven_b))
    rules = {pool["pool"]: _pool_rules(pool, rows, prompts, splits) for pool, rows, _ in pool_rows}
    for pool, rows, _ in pool_rows:
        pool["prompt_overhead_tokens"], pool["anomalies"] = prompt_overheads(
            rows, prompts, splits, pool["models"]
        )

    # everything is computed; from here on only writing
    splits.to_parquet(derived / "splits.parquet", index=False, compression="zstd")
    summary: dict[str, Any] = {
        "heldout_datasets": sorted(splits.loc[splits["split"] == "heldout", "dataset"].unique()),
        "seven_b_models": sorted(seven_b),
        "pools": {},
    }
    for pool, rows, fitted in pool_rows:
        name = pool["pool"]
        _write_json(derived / "pools" / f"{name}.json", pool)
        write_pool_config(
            pool,
            derived / "pools" / f"{name}.config.yaml",
            rules=rules[name],
            default_model=pool["best_single"],
        )
        lines = export_replay(
            pool,
            rows,
            prompts,
            splits,
            derived / "replay" / f"{name}.jsonl",
            fitted_cost_models=fitted,
        )
        summary["pools"][name] = {
            "n_items": pool["n_items"],
            "n_models": len(pool["models"]),
            "replay_lines": lines,
        }
    return summary


def _pool_rules(
    pool: dict[str, Any], rows: pd.DataFrame, prompts: pd.DataFrame, splits: pd.DataFrame
) -> list[dict[str, Any]]:
    """The static rules mapped on the pool's train and calibration items only."""
    items = prompts[["item_id", "prompt", "prompt_bytes"]].merge(
        splits[["item_id", "split"]], on="item_id", validate="1:1"
    )
    in_pool = items["item_id"].isin(set(rows["item_id"]))
    tuning = items.loc[in_pool & items["split"].isin(["train", "calib"])]
    return map_rules(tuning, rows, RULES, best_single=pool["best_single"])


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--derived", type=Path, default=Path("data/derived"))
    args = parser.parse_args(argv)
    summary = build(args.derived)
    print(f"held-out datasets: {', '.join(summary['heldout_datasets'])}")
    print(
        f"mixed pool small models (costs fitted from recorded costs): "
        f"{', '.join(summary['seven_b_models'])}"
    )
    for name, counts in summary["pools"].items():
        print(
            f"{name}: {counts['n_items']} items, {counts['n_models']} models, "
            f"{counts['replay_lines']} replay lines"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
