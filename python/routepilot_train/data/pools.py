"""Model pools: item sets, fitted per-token prices, Best Single, and the replay config.

A pool is a set of models plus the items every one of them has a record for.
Prices are not published list prices: they are fitted from the cost recorded in
the dataset, so every table that shows them says "fitted from recorded costs".
"""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

from routepilot_train.data import selection
from routepilot_train.data.splits import require_splits

PRICE_SOURCE = "fitted_from_recorded_cost"

# Non-flagship models in the release that are large, not 7B class. They ran only some
# datasets, so they are never candidates for the mixed pool's four small models.
NOT_SEVEN_B: tuple[str, ...] = (
    "deepseek-v3.1-terminus",
    "gpt-4.1",
    "intern-s1-new",
    "qwen3-235b-a22b-no-thinking",
    "qwen3-235b-a22b-thinking",
)

# Replay config constants (shared data contract).
REPLAY_KEY = "replay-key-offline"
REPLAY_BASE_URL = "http://127.0.0.1:1/v1"
REPLAY_CONTEXT_WINDOW = 1048576
REPLAY_MAX_OUTPUT_TOKENS = 65536
REPLAY_CAPABILITIES = ("streaming", "tools", "json_object", "json_schema", "vision")

# The router bounds the prompt tokens of a one-message request by its text bytes plus this
# framing (8 per message and 16 per request, catalog.perMessageOverhead and fixedOverhead).
PROMPT_FRAMING_TOKENS = 24
# A recorded prompt count is an anomaly (Ruling V1-R24a), not a hidden per-model overhead,
# when it is above this many times the median of the other models' counts for the same item
# and also more than ANOMALY_MIN_EXCESS tokens above that median. The second condition keeps
# a constant hidden system prompt (about 150 tokens on intern-s1), which is many times the
# count of a short prompt, in the fit it exists for.
ANOMALY_FACTOR = 3
ANOMALY_MIN_EXCESS = 1000

_MICRO = Decimal("0.000001")


def fit_prices(df: pd.DataFrame) -> dict[str, Any]:
    """Fit USD per input and output token to the recorded cost of one model.

    Uses only the rows with ``cost_usd > 0`` and non-negative least squares on
    ``[prompt_tokens, completion_tokens]``, so a price is never negative. The
    columns and the target are scaled to about one before solving, because raw
    token counts (thousands) and prices (millionths) are badly conditioned.
    Returns ``in_usd_per_token``, ``out_usd_per_token``, ``fit_rmse_usd``,
    ``fit_r2`` and ``n_fit``.
    """
    paid = df.loc[df["cost_usd"] > 0]
    if paid.empty:
        raise ValueError("no rows with positive recorded cost to fit prices on")
    tokens = paid[["prompt_tokens", "completion_tokens"]].to_numpy(dtype=float)
    cost = paid["cost_usd"].to_numpy(dtype=float)
    peak = tokens.max(axis=0)
    column_scale = np.where(peak > 0, peak, 1.0)
    cost_scale = float(cost.max())
    coef, _ = nnls(tokens / column_scale, cost / cost_scale)
    prices = coef * cost_scale / column_scale
    residual = cost - tokens @ prices
    sse = float(residual @ residual)
    sst = float(((cost - cost.mean()) ** 2).sum())
    if sst > 0:
        r2 = 1.0 - sse / sst
    else:
        r2 = 1.0 if sse == 0 else 0.0
    return {
        "in_usd_per_token": float(prices[0]),
        "out_usd_per_token": float(prices[1]),
        "fit_rmse_usd": float(np.sqrt(sse / len(cost))),
        "fit_r2": float(r2),
        "n_fit": int(len(cost)),
    }


def pool_outcomes(outcomes: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Rows of ``models`` on the items where every one of them has a record."""
    wanted = set(models)
    rows = outcomes.loc[outcomes["model"].isin(wanted)]
    if rows.duplicated(["item_id", "model"]).any():
        raise ValueError("duplicate (item_id, model) rows in outcomes")
    seen = rows.groupby("item_id")["model"].nunique()
    complete = seen.index[seen == len(wanted)]
    kept = rows.loc[rows["item_id"].isin(complete)]
    return kept.sort_values(["item_id", "model"], kind="stable").reset_index(drop=True)


def pick_seven_b(
    outcomes: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    flagship: Sequence[str] = selection.FLAGSHIP_MODELS,
    not_seven_b: Collection[str] = NOT_SEVEN_B,
    n: int = 4,
) -> list[str]:
    """The ``n`` small models with the highest train-split success, best first.

    A candidate is a model that is not flagship, not a router and not in
    ``not_seven_b``, with a non-zero recorded cost somewhere. Its success is the
    mean over train items where every flagship model and the candidate have a
    record. Ties go to the lower name. Only train rows are read.
    """
    skip = set(flagship) | set(selection.NON_MODELS) | set(not_seven_b)
    flagship_items = set(pool_outcomes(outcomes, flagship)["item_id"])
    candidates = outcomes.loc[~outcomes["model"].isin(skip)]
    paid = set(candidates.loc[candidates["cost_usd"] > 0, "model"])
    split_of = splits[["item_id", "split"]]
    ranked: list[tuple[float, str]] = []
    for model, rows in candidates.loc[candidates["model"].isin(paid)].groupby("model"):
        rows = rows.loc[rows["item_id"].isin(flagship_items)]
        rows = rows.merge(split_of, on="item_id", how="left")
        if rows["split"].isna().any():
            raise ValueError(f"items of {model} have no split")
        train = rows.loc[rows["split"] == "train"]
        require_splits(train, {"train"})
        if train.empty:
            continue
        ranked.append((-float(train["success"].mean()), str(model)))
    ranked.sort()
    return [model for _, model in ranked[:n]]


def build_pool(
    name: str,
    outcomes: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    fitted_cost_models: Collection[str] = (),
) -> dict[str, Any]:
    """Describe the pool made of exactly the models in ``outcomes``.

    Items without a record from every model are dropped. ``rank`` is 1 plus the
    position in ascending train+calib success, so Best Single has the highest
    rank (the same order that picks ``best_single``). ``rank`` is an addition to
    the documented pool JSON keys; the replay config needs it.

    ``cheapest`` is the model with the lowest mean cost per item on the train and
    calibration items, ties by name (Ruling V1-R23): the cost the replay charges,
    which is the recorded cost, except for ``fitted_cost_models``, whose cost is
    their fitted prices times the recorded token counts (as in ``export_replay``).
    It is neither the lowest fitted price nor the gateway's worst-case ordering.
    """
    models = sorted(outcomes["model"].unique())
    if not models:
        raise ValueError("outcomes hold no models")
    fitted = set(fitted_cost_models)
    if not fitted <= set(models):
        raise ValueError(f"fitted_cost_models not in the pool: {sorted(fitted - set(models))}")
    rows = pool_outcomes(outcomes, models)
    if rows.empty:
        raise ValueError("no item has a record from every model")
    joined = rows.merge(splits[["item_id", "split"]], on="item_id", how="left", validate="m:1")
    if joined["split"].isna().any():
        raise ValueError("some items have no split")

    prices: dict[str, dict[str, Any]] = {}
    for model, model_rows in rows.groupby("model"):
        fit = fit_prices(model_rows)
        prices[str(model)] = {
            "in_usd_per_token": fit["in_usd_per_token"],
            "out_usd_per_token": fit["out_usd_per_token"],
            "fit_rmse_usd": fit["fit_rmse_usd"],
            "fit_r2": fit["fit_r2"],
            "price_source": PRICE_SOURCE,
        }

    tuning = joined.loc[joined["split"].isin(["train", "calib"])]
    require_splits(tuning, {"train", "calib"})
    if tuning.empty:
        raise ValueError("no train or calib rows to choose the best single model on")
    stats = tuning.groupby("model").agg(wins=("success", "sum"), cost=("cost_usd", "mean"))
    strongest_first = sorted(
        models, key=lambda m: (-int(stats.loc[m, "wins"]), float(stats.loc[m, "cost"]), m)
    )
    replay_cost = tuning["cost_usd"].copy()
    for model in fitted:
        mask = tuning["model"] == model
        price = prices[model]
        replay_cost[mask] = (
            price["in_usd_per_token"] * tuning.loc[mask, "prompt_tokens"]
            + price["out_usd_per_token"] * tuning.loc[mask, "completion_tokens"]
        )
    mean_replay_cost = replay_cost.groupby(tuning["model"]).mean()
    cheapest = min(models, key=lambda m: (float(mean_replay_cost[m]), m))
    return {
        "pool": name,
        "models": models,
        "datasets": sorted(rows["dataset"].unique()),
        "n_items": int(rows["item_id"].nunique()),
        "prices": prices,
        "best_single": strongest_first[0],
        "cheapest": cheapest,
        "rank": {model: len(models) - i for i, model in enumerate(strongest_first)},
    }


def token_anomalies(records: pd.DataFrame) -> pd.DataFrame:
    """The records whose prompt count is far above what the other models recorded.

    ``records`` has ``item_id``, ``model`` and ``prompt_tokens`` and a unique index. A
    record is an anomaly when ``prompt_tokens`` is above ``ANOMALY_FACTOR`` times the
    median ``prompt_tokens`` of the other models' records on the same item and more than
    ``ANOMALY_MIN_EXCESS`` tokens above it. Counts of zero
    or less are gaps in the dataset: they are never anomalies and never enter a median, and
    an item with no other positive count has nothing to compare with. Returns the
    anomalous rows (same index) with an ``others_median`` column.
    """
    positive = records.loc[records["prompt_tokens"] > 0]
    labels: list[Any] = []
    medians: list[float] = []
    for _, group in positive.groupby("item_id", sort=False):
        tokens = group["prompt_tokens"].to_numpy(dtype=float)
        if tokens.size < 2:
            continue
        for position, label in enumerate(group.index):
            others_median = float(np.median(np.delete(tokens, position)))
            count = tokens[position]
            far = count - others_median > ANOMALY_MIN_EXCESS
            if far and count > ANOMALY_FACTOR * others_median:
                labels.append(label)
                medians.append(others_median)
    found = records.loc[labels].copy()
    found["others_median"] = medians
    return found


def prompt_overheads(
    outcomes: pd.DataFrame,
    prompts: pd.DataFrame,
    splits: pd.DataFrame,
    models: Sequence[str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Hidden prompt tokens per model (Ruling V1-R24) and the records excluded from the fit.

    Some providers add prompt tokens the router cannot see. For each model this is the
    largest ``max(0, prompt_tokens - (prompt_bytes + PROMPT_FRAMING_TOKENS))`` over its
    train and calibration records, so the gateway's bound ``in_upper + overhead`` covers
    every record it was fitted on. Test and held-out records are never read. Records that
    ``token_anomalies`` flags are left out and returned as ``item_id``, ``model``,
    ``prompt_tokens`` and ``others_median``, sorted by item and model.
    """
    names = sorted(set(models))
    if not names:
        raise ValueError("no models to fit prompt overheads for")
    rows = outcomes.loc[outcomes["model"].isin(names), ["item_id", "model", "prompt_tokens"]]
    joined = rows.merge(
        prompts[["item_id", "prompt_bytes"]], on="item_id", how="left", validate="m:1"
    )
    if joined["prompt_bytes"].isna().any():
        raise ValueError("some items have no prompt")
    joined = joined.merge(splits[["item_id", "split"]], on="item_id", how="left", validate="m:1")
    if joined["split"].isna().any():
        raise ValueError("some items have no split")
    tuning = joined.loc[joined["split"].isin(["train", "calib"])].reset_index(drop=True)
    require_splits(tuning, {"train", "calib"})

    flagged = token_anomalies(tuning)
    kept = tuning.drop(index=flagged.index)
    excess = (kept["prompt_tokens"] - (kept["prompt_bytes"] + PROMPT_FRAMING_TOKENS)).clip(lower=0)
    largest = excess.groupby(kept["model"]).max()
    overheads = {model: int(largest.get(model, 0)) for model in names}
    anomalies = [
        {
            "item_id": str(row.item_id),
            "model": str(row.model),
            "prompt_tokens": int(row.prompt_tokens),
            "others_median": float(row.others_median),
        }
        for row in flagged.sort_values(["item_id", "model"]).itertuples(index=False)
    ]
    return overheads, anomalies


def usd_per_mtok(usd_per_token: float) -> str:
    """USD per million tokens as a decimal with 6 fractional digits, rounded half up."""
    exact = Decimal(repr(float(usd_per_token))) * 1_000_000
    return format(exact.quantize(_MICRO, rounding=ROUND_HALF_UP), "f")


def write_pool_config(
    pool: Mapping[str, Any],
    out: Path,
    rules: list[dict[str, Any]] | None = None,
    default_model: str | None = None,
) -> None:
    """Write a RoutePilot config that replays ``pool`` without calling any provider.

    The routing policy is ``strongest`` when ``rules`` is None, else ``rules``
    with ``default_model`` as the fallback. Each model carries
    ``prompt_overhead_tokens`` from ``pool["prompt_overhead_tokens"]`` (0 when the
    pool has none).
    """
    if rules is not None and not default_model:
        raise ValueError("default_model is required when rules are given")
    overheads = pool.get("prompt_overhead_tokens", {})
    models = []
    for model in pool["models"]:
        price = pool["prices"][model]
        models.append(
            {
                "id": model,
                "rank": pool["rank"][model],
                "context_window": REPLAY_CONTEXT_WINDOW,
                "max_output_tokens": REPLAY_MAX_OUTPUT_TOKENS,
                "prompt_overhead_tokens": int(overheads.get(model, 0)),
                "capabilities": list(REPLAY_CAPABILITIES),
                "input_price_per_mtok": usd_per_mtok(price["in_usd_per_token"]),
                "output_price_per_mtok": usd_per_mtok(price["out_usd_per_token"]),
                "deployments": [
                    {"name": f"{model}-replay", "provider": "replay", "upstream_model": model}
                ],
            }
        )
    if rules is None:
        routing: dict[str, Any] = {"policy": "strongest"}
    else:
        routing = {"policy": "rules", "default_model": default_model, "rules": rules}
    tenants = [
        {
            "id": "replay",
            "api_key_sha256": hashlib.sha256(REPLAY_KEY.encode()).hexdigest(),
            "allowed_models": ["*"],
            "default_max_tokens": REPLAY_MAX_OUTPUT_TOKENS,
        }
    ]
    providers = [{"name": "replay", "type": "openaicompat", "base_url": REPLAY_BASE_URL}]

    parts = [
        f"# RoutePilot replay config for the {pool['pool']} pool, generated by\n"
        "# routepilot_train.data.build. The replay provider is never called.\n",
        _dump({"providers": providers}),
        "\n# Prices are USD per million tokens, fitted from recorded costs and rounded half up\n"
        "# to 6 fractional digits. rank is 1 + position in ascending train+calib success.\n"
        "# context_window and max_output_tokens are placeholders:\n"
        "# replay does not test context feasibility.\n"
        "# prompt_overhead_tokens (Ruling V1-R24) is the largest excess of a recorded prompt\n"
        "# count over prompt bytes + 24 on the train and calib items, outliers left out.\n",
        _dump({"models": models}),
        "\n",
        _dump({"routing": routing}),
        "\n",
        _dump({"tenants": tenants}),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes("".join(parts).encode("utf-8"))


def _dump(section: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(section), sort_keys=False, default_flow_style=False, width=1000)
