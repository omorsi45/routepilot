"""Synthetic outcome, prompt and split frames for the pool, export and build tests.

Every prompt is a made-up string, never text from the dataset.
"""

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from routepilot_train.data import selection

SENTINEL = "SENTINEL-PROMPT-TEXT"

# A synthetic copy of the release: the 11 kept datasets with the 12 flagship models, plus
# small models on the 8 datasets the real small models ran.
FLAGSHIP_PERCENT = {m: 35 + 5 * i for i, m in enumerate(selection.FLAGSHIP_MODELS)}
# small model -> success percent; s-e records no cost at all, so it is not eligible
SMALL = {"s-a": 60, "s-b": 50, "s-c": 40, "s-d": 30, "s-f": 20, "s-e": 70}
MIXED_DATASETS = [
    "aime",
    "arenahard_coding",
    "arenahard_creative_writing",
    "arenahard_math",
    "gpqa",
    "livecodebench",
    "livemathbench",
    "mmlupro",
]
PER_DATASET = 30


def prompt_for(item_id: str) -> str:
    return f"{SENTINEL} for {item_id}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def outcome_frame(rows: Iterable[tuple[str, str, bool, int, int, float]]) -> pd.DataFrame:
    """Rows are (item_id, model, success, prompt_tokens, completion_tokens, cost_usd)."""
    records = [
        {
            "dataset": item_id.split("/")[0],
            "item_id": item_id,
            "model": model,
            "score": 1.0 if success else 0.0,
            "success": success,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "prompt_sha256": sha(prompt_for(item_id)),
        }
        for item_id, model, success, prompt_tokens, completion_tokens, cost in rows
    ]
    frame = pd.DataFrame.from_records(records)
    return frame.astype({"prompt_tokens": "int64", "completion_tokens": "int64"})


def prompts_frame(
    item_ids: Iterable[str], overrides: Mapping[str, str] | None = None
) -> pd.DataFrame:
    texts = {item_id: prompt_for(item_id) for item_id in item_ids}
    texts.update(overrides or {})
    return pd.DataFrame(
        {
            "dataset": [item_id.split("/")[0] for item_id in texts],
            "item_id": list(texts),
            "prompt": list(texts.values()),
            "prompt_sha256": [sha(text) for text in texts.values()],
            "prompt_bytes": [len(text.encode("utf-8")) for text in texts.values()],
        }
    )


def splits_frame(assignment: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": list(assignment),
            "dataset": [item_id.split("/")[0] for item_id in assignment],
            "split": list(assignment.values()),
        }
    )


def bucket(*parts: str) -> int:
    return int(sha("/".join(parts))[:8], 16)


def price_of(model: str) -> tuple[float, float]:
    h = bucket(model)
    return (1 + h % 7) * 1e-7, (3 + h % 11) * 1e-7


def synthetic_outcomes() -> pd.DataFrame:
    rows = []
    for dataset in selection.KEPT:
        for i in range(PER_DATASET):
            item_id = f"{dataset}/{i}"
            models = dict(FLAGSHIP_PERCENT)
            if dataset in MIXED_DATASETS:
                models |= SMALL | {"gpt-4.1": 95}
            for model, pct in models.items():
                if item_id == "hle/3" and model == "gpt-5":
                    continue  # the one flagship record missing in the real data
                prompt_tokens = 50 + bucket(item_id, model, "p") % 400
                completion_tokens = 20 + bucket(item_id, model, "c") % 300
                p_in, p_out = price_of(model)
                cost = p_in * prompt_tokens + p_out * completion_tokens
                if model == "s-e" or (model in SMALL and dataset.startswith("arenahard")):
                    cost = 0.0  # small models record no cost on some datasets
                ok = bucket(item_id, model, "s") % 100 < pct
                rows.append(
                    {
                        "dataset": dataset,
                        "item_id": item_id,
                        "model": model,
                        "score": 1.0 if ok else 0.0,
                        "success": ok,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": cost,
                        "prompt_sha256": sha(prompt_for(item_id)),
                    }
                )
    return pd.DataFrame(rows).astype({"prompt_tokens": "int64", "completion_tokens": "int64"})


def synthetic_prompts(outcomes: pd.DataFrame) -> pd.DataFrame:
    items = outcomes[["dataset", "item_id"]].drop_duplicates().reset_index(drop=True)
    items["prompt"] = [prompt_for(i) for i in items["item_id"]]
    items["prompt_sha256"] = [sha(p) for p in items["prompt"]]
    items["prompt_bytes"] = [len(p.encode()) for p in items["prompt"]]
    return items


def write_derived(directory: Path, outcomes: pd.DataFrame | None = None) -> Path:
    """Write outcomes.parquet and prompts.parquet like the ingest does."""
    frame = synthetic_outcomes() if outcomes is None else outcomes
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "outcomes.parquet", index=False)
    synthetic_prompts(frame).to_parquet(directory / "prompts.parquet", index=False)
    return directory
