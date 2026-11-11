"""Dataset summary: items per dataset and split, pools, fitted prices, zero-cost records.

Run from the repository root after ``make data``:

    python bench/scripts/data_summary.py

Reads data/derived and writes bench/results/data/<run-id>/{results.json,
report.md, run.json}. The output holds numbers, model names and item IDs only,
never prompt text. Prices are fitted from recorded costs, not list prices, and
the report says so wherever it shows them.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from routepilot_train import runinfo
from routepilot_train.data import pools as pool_code
from routepilot_train.data import selection, splits

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "calib", "test", "heldout")
LOW_R2 = 0.9
FITTED = "fitted from recorded costs"


def summarize(derived: Path, out_root: Path, command: Sequence[str] | None = None) -> Path:
    """Write the summary under ``out_root/<run-id>`` and return that directory."""
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    outcomes = pd.read_parquet(derived / "outcomes.parquet")
    split_table = pd.read_parquet(derived / "splits.parquet")
    pool_json = {
        name: json.loads((derived / "pools" / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("flagship", "mixed")
    }

    heldout = sorted(split_table.loc[split_table["split"] == "heldout", "dataset"].unique())
    results: dict[str, Any] = {
        "experiment": "data",
        "dataset_sha256": selection.DATASET_SHA256,
        "heldout_datasets": heldout,
        "heldout_domains": {d: selection.DOMAINS[d] for d in heldout},
        "split_salt": splits.SALT,
        "split_totals": {s: int((split_table["split"] == s).sum()) for s in SPLITS},
        "items_per_dataset_split": _items_per_dataset_split(split_table),
        "pools": {},
    }
    pool_rows: dict[str, pd.DataFrame] = {}
    for name, pool in pool_json.items():
        rows = pool_code.pool_outcomes(outcomes, pool["models"]).merge(
            split_table[["item_id", "split"]], on="item_id", how="left", validate="m:1"
        )
        pool_rows[name] = rows
        results["pools"][name] = _pool_section(pool, rows)
    results["pools"]["mixed"]["seven_b_models"] = sorted(
        set(pool_json["mixed"]["models"]) - set(selection.FLAGSHIP_MODELS)
    )
    results["pools"]["mixed"]["seven_b_cost_basis"] = FITTED
    results["r2_range"] = {
        name: [
            min(p["fit_r2"] for p in section["prices"].values()),
            max(p["fit_r2"] for p in section["prices"].values()),
        ]
        for name, section in results["pools"].items()
    }
    results["zero_cost_flagship"] = _zero_cost(pool_rows["flagship"])
    results["negative_token_rows"] = _negative_tokens(outcomes)

    run_dir = out_root / runinfo.new_run_id(started, start_git)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_text(run_dir / "results.json", json.dumps(results, indent=2) + "\n")
    _write_text(run_dir / "report.md", _report(results, run_dir.name))
    runinfo.write_run_json(
        run_dir,
        "data",
        {"split_salt": splits.SALT},
        command or ["python", "bench/scripts/data_summary.py"],
        None,
        started,
        git=start_git,
    )
    return run_dir


def _items_per_dataset_split(split_table: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = split_table.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(SPLITS), fill_value=0)
    return {str(d): {s: int(row[s]) for s in SPLITS} for d, row in counts.iterrows()}


def _pool_section(pool: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    items = rows.drop_duplicates("item_id")
    success: dict[str, dict[str, float | None]] = {}
    for model, group in rows.groupby("model"):
        success[str(model)] = {
            s: float(group.loc[group["split"] == s, "success"].mean())
            if (group["split"] == s).any()
            else None
            for s in SPLITS
        }
    tuning = rows.loc[rows["split"].isin(["train", "calib"])]
    prices: dict[str, dict[str, Any]] = {}
    for model in pool["models"]:
        price = pool["prices"][model]
        n_fit = int(((tuning["model"] == model) & (tuning["cost_usd"] > 0)).sum())
        prices[model] = {
            "in_usd_per_mtok": price["in_usd_per_token"] * 1e6,
            "out_usd_per_mtok": price["out_usd_per_token"] * 1e6,
            "fit_rmse_usd": price["fit_rmse_usd"],
            "fit_r2": price["fit_r2"],
            "n_fit": n_fit,
            "price_source": price["price_source"],
        }
    return {
        "n_items": pool["n_items"],
        "models": pool["models"],
        "datasets": pool["datasets"],
        "items_per_split": {s: int((items["split"] == s).sum()) for s in SPLITS},
        "items_per_dataset_split": _items_per_dataset_split(items),
        "best_single": pool["best_single"],
        "cheapest": pool["cheapest"],
        "rank": pool["rank"],
        "success_by_split": success,
        "prices": prices,
        "low_r2_models": sorted(m for m, p in prices.items() if p["fit_r2"] < LOW_R2),
    }


def _zero_cost(rows: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for (model, dataset), group in rows.groupby(["model", "dataset"]):
        zero = group.loc[group["cost_usd"] == 0]
        out.setdefault(str(model), {})[str(dataset)] = {
            "records": int(len(group)),
            "zero_cost": int(len(zero)),
            "zero_cost_zero_completion": int((zero["completion_tokens"] == 0).sum()),
        }
    return out


def _negative_tokens(outcomes: pd.DataFrame) -> list[dict[str, Any]]:
    bad = outcomes.loc[(outcomes["prompt_tokens"] < 0) | (outcomes["completion_tokens"] < 0)]
    bad = bad.sort_values(["item_id", "model"])
    return [
        {
            "item_id": str(r.item_id),
            "model": str(r.model),
            "prompt_tokens": int(r.prompt_tokens),
            "completion_tokens": int(r.completion_tokens),
            "cost_usd": float(r.cost_usd),
        }
        for r in bad.itertuples()
    ]


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _report(results: dict[str, Any], run_id: str) -> str:
    out = [f"# Dataset summary\n\nRun `{run_id}`. Dataset SHA-256 `{results['dataset_sha256']}`.\n"]
    out.append(
        "Counts, rates and item IDs only: no prompt text. Prices are not list prices: they are "
        f"{FITTED} (non-negative least squares on prompt and completion tokens, rows with a "
        "non-zero recorded cost of the train and calibration splits only).\n"
    )

    out.append("## Held-out datasets\n")
    out.append(
        "Chosen before training by `sha256(name + salt)` order, at most one per domain, "
        f"salt `{results['split_salt']}`.\n"
    )
    out.append(
        _table(
            ["dataset", "domain"],
            [[d, results["heldout_domains"][d]] for d in results["heldout_datasets"]],
        )
    )

    out.append("## Items per dataset and split\n")
    per = results["items_per_dataset_split"]
    rows = [[d, *(c[s] for s in SPLITS), sum(c.values())] for d, c in sorted(per.items())]
    totals = results["split_totals"]
    rows.append(["total", *(totals[s] for s in SPLITS), sum(totals.values())])
    out.append(_table(["dataset", *SPLITS, "total"], rows))

    for name, section in results["pools"].items():
        out.append(f"## Pool `{name}`\n")
        out.append(
            f"{section['n_items']} items with a record from every one of {len(section['models'])} "
            f"models, over {len(section['datasets'])} datasets: {', '.join(section['datasets'])}. "
            f"Items per split: "
            + ", ".join(f"{s} {section['items_per_split'][s]}" for s in SPLITS)
            + ".\n"
        )
        bullets = [
            f"- Best Single (highest train+calib success): `{section['best_single']}`",
            f"- Cheapest (lowest mean cost per item on train+calib, as the replay charges it): "
            f"`{section['cheapest']}`",
        ]
        fitted_cost = set(section.get("seven_b_models", []))
        if name == "mixed":
            bullets.append(
                "- Small models added to the flagship pool: "
                f"{', '.join('`' + m + '`' for m in section['seven_b_models'])}. Their cost in "
                "the replay export is recomputed from their prices (fitted from recorded costs) "
                "and their recorded token counts, because their recorded cost is zero on some "
                "datasets. Flagship models keep their recorded cost."
            )
        out.append("\n".join(bullets) + "\n")
        out.append(f"### Success per split (pool `{name}`)\n")
        ranked = sorted(section["models"], key=lambda m: -section["rank"][m])
        out.append(
            _table(
                ["model", "rank", *SPLITS],
                [
                    [
                        m,
                        section["rank"][m],
                        *(_rate(section["success_by_split"][m][s]) for s in SPLITS),
                    ]
                    for m in ranked
                ],
            )
        )
        out.append(f"### Prices {FITTED} (pool `{name}`)\n")
        out.append(
            _table(
                [
                    "model",
                    "in USD/Mtok",
                    "out USD/Mtok",
                    "RMSE USD",
                    "R^2",
                    "rows fitted",
                    "replay cost",
                    "flag",
                ],
                [
                    [
                        m,
                        f"{p['in_usd_per_mtok']:.4f}",
                        f"{p['out_usd_per_mtok']:.4f}",
                        f"{p['fit_rmse_usd']:.6f}",
                        f"{p['fit_r2']:.5f}",
                        p["n_fit"],
                        "fitted" if m in fitted_cost else "recorded",
                        f"R^2 below {LOW_R2}" if p["fit_r2"] < LOW_R2 else "",
                    ]
                    for m, p in sorted(section["prices"].items())
                ],
            )
        )
        low = section["low_r2_models"]
        lo, hi = results["r2_range"][name]
        verdict = (
            f"Fits with R^2 below {LOW_R2} (still used as fitted): {', '.join(low)}."
            if low
            else f"No fit has R^2 below {LOW_R2}."
        )
        out.append(f"R^2 range: {lo:.5f} to {hi:.5f}. {verdict}\n")

    out.append("## Zero recorded cost (flagship pool)\n")
    out.append(
        "Records with `cost_usd == 0`, per model and dataset, with the count that also have zero "
        "completion tokens in brackets. They are kept as recorded.\n"
    )
    zero = results["zero_cost_flagship"]
    datasets = sorted({d for by_dataset in zero.values() for d in by_dataset})
    rows = []
    for model in sorted(zero):
        cells = []
        for d in datasets:
            c = zero[model].get(d)
            cells.append(
                "-" if c is None else f"{c['zero_cost']} ({c['zero_cost_zero_completion']})"
            )
        rows.append([model, *cells])
    out.append(_table(["model", *datasets], rows))

    out.append("## Negative token counts\n")
    negative = results["negative_token_rows"]
    if negative:
        out.append(
            f"{len(negative)} records have a negative token count. They are kept as recorded.\n"
        )
        out.append(
            _table(
                ["item_id", "model", "prompt_tokens", "completion_tokens", "cost_usd"],
                [
                    [
                        r["item_id"],
                        r["model"],
                        r["prompt_tokens"],
                        r["completion_tokens"],
                        r["cost_usd"],
                    ]
                    for r in negative
                ],
            )
        )
    else:
        out.append("None.\n")
    return "\n".join(out)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize data/derived into bench/results/data.")
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "data")
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    run_dir = summarize(args.derived, args.out_root, command)
    print(run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
