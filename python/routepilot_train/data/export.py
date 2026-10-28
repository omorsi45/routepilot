"""Replay export: one JSON line per item with every pool model's recorded outcome.

The file holds prompt text, so it stays in the gitignored data/derived. The Go
replay command reads it and never writes the prompt back out.
"""

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from routepilot_train.data.pools import pool_outcomes

PICO = 1e12


def export_replay(
    pool: Mapping[str, Any],
    outcomes: pd.DataFrame,
    prompts: pd.DataFrame,
    splits: pd.DataFrame,
    out: Path,
    *,
    fitted_cost_models: Collection[str] = (),
) -> int:
    """Write ``out`` and return the number of lines (items) written.

    Items are sorted by ``item_id`` and held-out items are included with
    ``split: "heldout"``. ``cost_picousd`` is ``round(cost_usd * 1e12)``, except
    for ``fitted_cost_models``, whose cost is recomputed from the pool's fitted
    prices and the recorded token counts (their recorded cost is zero on some
    datasets).
    """
    models = list(pool["models"])
    fitted = set(fitted_cost_models)
    if not fitted <= set(models):
        raise ValueError(f"fitted_cost_models not in the pool: {sorted(fitted - set(models))}")
    rows = pool_outcomes(outcomes, models)

    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    prompt_sha_of = dict(zip(prompts["item_id"], prompts["prompt_sha256"], strict=True))
    split_of = dict(zip(splits["item_id"], splits["split"], strict=True))
    items = rows["item_id"].unique()
    if missing := [i for i in items if i not in prompt_of]:
        raise ValueError(f"{len(missing)} exported items have no prompt, first: {missing[0]}")
    if missing := [i for i in items if i not in split_of]:
        raise ValueError(f"{len(missing)} exported items have no split, first: {missing[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for item_id, group in rows.groupby("item_id", sort=True):
            per_model: dict[str, dict[str, Any]] = {}
            for row in group.itertuples(index=False):
                if row.model in fitted:
                    price = pool["prices"][row.model]
                    cost = round(
                        (
                            price["in_usd_per_token"] * row.prompt_tokens
                            + price["out_usd_per_token"] * row.completion_tokens
                        )
                        * PICO
                    )
                else:
                    cost = round(row.cost_usd * PICO)
                per_model[str(row.model)] = {
                    "success": bool(row.success),
                    "cost_picousd": int(cost),
                    "prompt_tokens": int(row.prompt_tokens),
                    "completion_tokens": int(row.completion_tokens),
                }
            first = group.iloc[0]
            line = {
                "item_id": str(item_id),
                "dataset": str(first["dataset"]),
                "split": str(split_of[item_id]),
                "prompt_sha256": str(prompt_sha_of[item_id]),
                "messages": [{"role": "user", "content": prompt_of[item_id]}],
                "outcomes": per_model,
            }
            handle.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
