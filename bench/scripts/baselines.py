"""Static baselines on the test split, with bootstrap intervals.

Run from the repository root after ``make data``:

    python bench/scripts/baselines.py --pool flagship
    python bench/scripts/baselines.py --pool mixed

Replays seven policies with the Go ``routepilot replay`` command and writes
bench/results/baselines/<run-id>/{results.json,report.md,pareto.png,run.json}:

- ``always_strongest``: the pool's Best Single (train+calib success), always.
- ``always_cheapest``: the model with the lowest mean cost per item on train+calib, always.
- ``random``: a seeded random pick (seed 0 in the tables; seeds 0 to 4 for the spread).
- ``static_rules``: the keyword and length rules of the pool config, mapped on train+calib.
- ``oracle``: the cheapest successful model per item. An upper bound, not a policy.
- ``static_strongest_rank`` and ``static_cheapest_worstcase``: the gateway's own ``strongest``
  and ``cheapest`` policies (config rank, and worst-case cost with the replay output budget),
  shown so their difference from the two always_* rows is visible.

Every row is on the test split, read once per policy. Deltas and cost reductions are against
always_strongest, paired by item. The output holds numbers, model names and item IDs only,
never prompt text.
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
import yaml
from matplotlib.figure import Figure

from routepilot_train import replaytool, runinfo
from routepilot_train.baselines.static_rules import match_rule
from routepilot_train.evaluate import metrics, pareto, report

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "test"
RANDOM_SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
REFERENCE = "always_strongest"
PICO = 1e12


def policy_specs(pool: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """(row name, replay policy, extra flags) in table order."""
    return [
        (REFERENCE, "fixed", ["--model", pool["best_single"]]),
        ("always_cheapest", "fixed", ["--model", pool["cheapest"]]),
        ("random", "random", ["--seed", "0"]),
        ("static_rules", "rules", []),
        ("oracle", "oracle", []),
        ("static_strongest_rank", "strongest", []),
        ("static_cheapest_worstcase", "cheapest", []),
    ]


def run_baselines(
    derived: Path,
    pool_name: str,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    bootstrap: int = BOOTSTRAP,
) -> Path:
    """Replay the baselines on ``pool_name`` and write the results under ``out_root/<run-id>``."""
    started = datetime.now(UTC)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    binary = binary or replaytool.build_go(ROOT)

    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)

        def run(split: str, policy: str, extra: Sequence[str], name: str) -> pd.DataFrame:
            return replaytool.replay(binary, config, replay_input, split, policy, tmp / name, extra)

        frames = {
            name: run(SPLIT, policy, extra, f"{name}.jsonl")
            for name, policy, extra in policy_specs(pool)
        }
        random_frames = {
            seed: frames["random"]
            if seed == 0
            else run(SPLIT, "random", ["--seed", str(seed)], f"random-{seed}.jsonl")
            for seed in RANDOM_SEEDS
        }
        everything = run("all", "rules", [], "rules-all.jsonl")
    parity = rules_parity(config, derived / "prompts.parquet", everything)

    reference = frames[REFERENCE]
    rows = {name: _policy_row(name, df, reference, bootstrap) for name, df in frames.items()}
    seeds = _random_spread(random_frames)
    per_dataset = _per_dataset(frames)
    points = pd.DataFrame(
        {
            "policy": list(rows),
            "mean_cost_usd": [r["mean_cost_usd"] for r in rows.values()],
            "success_rate": [r["success_rate"] for r in rows.values()],
        }
    )
    front = pareto.pareto_front(points)

    results: dict[str, Any] = {
        "pool": pool_name,
        "split": SPLIT,
        "n_items": len(reference),
        "best_single": pool["best_single"],
        "cheapest": pool["cheapest"],
        "bootstrap": {
            "n": bootstrap,
            "seed": BOOTSTRAP_SEED,
            "alpha": 0.05,
            "method": "percentile",
        },
        "policies": rows,
        "random_seeds": seeds,
        "per_dataset": per_dataset,
        "pareto_front": list(front["policy"]),
        "rules_mapping": _rules_mapping(config),
        "rules_parity": parity,
        "labels": _labels(pool_name),
    }
    tables = _tables(results, rows, seeds, per_dataset, frames)
    figures = {"pareto": _pareto_figure(points, front, pool_name)}

    out_dir = out_root / runinfo.new_run_id(started)
    seed_info = {
        "random": list(RANDOM_SEEDS),
        "bootstrap": BOOTSTRAP_SEED,
    }
    report.write_results(
        out_dir,
        "baselines",
        results,
        tables,
        figures,
        seed_info,
        command or ["python", "bench/scripts/baselines.py", "--pool", pool_name],
        None,
        started,
    )
    return out_dir


def _policy_row(
    name: str, df: pd.DataFrame, reference: pd.DataFrame, bootstrap: int
) -> dict[str, Any]:
    summary = metrics.summarize(df)
    success = df["success"].astype(float).to_numpy()
    cost_usd = df["cost_picousd"].to_numpy(dtype=float) / PICO
    _, s_lo, s_hi = metrics.bootstrap_ci(success, n=bootstrap, seed=BOOTSTRAP_SEED)
    _, c_lo, c_hi = metrics.bootstrap_ci(cost_usd, n=bootstrap, seed=BOOTSTRAP_SEED)
    delta = metrics.paired_delta(df, reference, n=bootstrap, seed=BOOTSTRAP_SEED)
    fixed = df["policy"].iloc[0] == "fixed"
    usage = df["model"].fillna("(routing error)").value_counts().sort_index()
    return {
        "policy": name,
        "replay_policy": str(df["policy"].iloc[0]),
        "fixed_model": str(df["model"].iloc[0]) if fixed else None,
        **summary,
        "success_ci": [s_lo, s_hi],
        "mean_cost_ci": [c_lo, c_hi],
        "success_delta_vs_strongest": delta["success_delta"],
        "success_delta_ci": delta["success_delta_ci"],
        "cost_reduction_vs_strongest": delta["cost_reduction"],
        "cost_reduction_ci": delta["cost_reduction_ci"],
        "model_counts": {str(m): int(c) for m, c in usage.items()},
    }


def _random_spread(frames: dict[int, pd.DataFrame]) -> dict[str, Any]:
    summaries = {str(seed): metrics.summarize(df) for seed, df in frames.items()}
    per_seed = {
        seed: {"success_rate": s["success_rate"], "mean_cost_usd": s["mean_cost_usd"]}
        for seed, s in summaries.items()
    }
    out: dict[str, Any] = {"seeds": per_seed}
    for key in ("success_rate", "mean_cost_usd"):
        values = np.array([v[key] for v in per_seed.values()])
        out[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def _per_dataset(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    reference = frames[REFERENCE]
    for dataset in sorted(reference["dataset"].unique()):
        entry: dict[str, Any] = {"n": int((reference["dataset"] == dataset).sum())}
        for name, df in frames.items():
            sub = metrics.summarize(df.loc[df["dataset"] == dataset])
            entry[name] = {
                "success_rate": sub["success_rate"],
                "mean_cost_usd": sub["mean_cost_usd"],
            }
        out[str(dataset)] = entry
    return out


def rules_parity(config: Path, prompts: Path, replayed: pd.DataFrame) -> dict[str, Any]:
    """Compare the model the Go ``rules`` policy picked with the Python matcher's prediction.

    ``replayed`` is a replay of the ``rules`` policy; the prediction applies
    ``match_rule`` to each item's prompt text with the rules of the pool config. Only
    counts and item IDs leave this function.
    """
    routing = yaml.safe_load(config.read_text(encoding="utf-8"))["routing"]
    rules = routing["rules"]
    model_of = {rule["name"]: rule["model"] for rule in rules}
    text = pd.read_parquet(prompts, columns=["item_id", "prompt", "prompt_bytes"])
    text = text.loc[text["item_id"].isin(set(replayed["item_id"]))]
    expected = {
        item_id: model_of.get(match_rule(str(prompt), int(size), rules), routing["default_model"])
        for item_id, prompt, size in zip(
            text["item_id"], text["prompt"], text["prompt_bytes"], strict=True
        )
    }
    got = dict(zip(replayed["item_id"], replayed["model"], strict=True))
    missing = sorted(set(got) - set(expected))
    wrong = sorted(i for i in expected if got.get(i) != expected[i])
    return {
        "split": "all",
        "items": len(got),
        "mismatches": len(wrong) + len(missing),
        "mismatched_item_ids": (wrong + missing)[:20],
    }


def _rules_mapping(config: Path) -> dict[str, Any]:
    routing = yaml.safe_load(config.read_text(encoding="utf-8"))["routing"]
    return {
        "default_model": routing["default_model"],
        "rules": {rule["name"]: rule["model"] for rule in routing["rules"]},
    }


def _labels(pool_name: str) -> list[str]:
    labels = [
        "no latency is measured or simulated in this experiment",
        "Best Single, the cheapest model and the rules mapping are chosen on train+calib; "
        "this table is the test split, read once per policy",
        "oracle looks at the recorded outcomes: an upper bound, not a policy",
    ]
    if pool_name == "mixed":
        labels.insert(
            0,
            "costs of the four small models are fitted from recorded costs and applied to their "
            "recorded token counts; flagship costs are recorded",
        )
    else:
        labels.insert(0, "costs are the recorded costs of each model on each item")
    return labels


def _ci_text(point: float, ci: Sequence[float] | None, digits: str = ".4f") -> str:
    if point is None or ci is None:
        return "n/a"
    return f"{point:{digits}} [{ci[0]:{digits}}, {ci[1]:{digits}}]"


def _tables(
    results: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    seeds: dict[str, Any],
    per_dataset: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    setup = pd.DataFrame(
        [
            ["pool", results["pool"]],
            ["split", results["split"]],
            ["items", results["n_items"]],
            ["Best Single (always_strongest)", results["best_single"]],
            ["cheapest (always_cheapest)", results["cheapest"]],
            [
                "bootstrap",
                f"{results['bootstrap']['n']} resamples, percentile 95% intervals, "
                f"seed {results['bootstrap']['seed']}",
            ],
            ["rules default model", results["rules_mapping"]["default_model"]],
            *([f"rule {name}", model] for name, model in results["rules_mapping"]["rules"].items()),
            [
                "static rules Go/Python parity (all items)",
                f"{results['rules_parity']['mismatches']} mismatches of "
                f"{results['rules_parity']['items']}",
            ],
        ],
        columns=["setting", "value"],
    )
    quality = pd.DataFrame(
        [
            {
                "policy": name,
                "n": r["n"],
                "errors": r["errors"],
                "success_rate [95% CI]": _ci_text(r["success_rate"], r["success_ci"]),
                "mean_cost_usd [95% CI]": _ci_text(r["mean_cost_usd"], r["mean_cost_ci"], ".6g"),
                "cost_per_success_usd": r["cost_per_success_usd"],
            }
            for name, r in rows.items()
        ]
    )
    against = pd.DataFrame(
        [
            {
                "policy": name,
                "success_delta [95% CI]": _ci_text(
                    r["success_delta_vs_strongest"], r["success_delta_ci"]
                ),
                "cost_reduction [95% CI]": _ci_text(
                    r["cost_reduction_vs_strongest"], r["cost_reduction_ci"]
                ),
            }
            for name, r in rows.items()
        ]
    )
    spread = pd.DataFrame(
        [
            {"seed": seed, "success_rate": v["success_rate"], "mean_cost_usd": v["mean_cost_usd"]}
            for seed, v in seeds["seeds"].items()
        ]
        + [
            {
                "seed": stat,
                "success_rate": seeds["success_rate"][stat],
                "mean_cost_usd": seeds["mean_cost_usd"][stat],
            }
            for stat in ("mean", "min", "max")
        ]
    )
    names = list(frames)
    datasets = list(per_dataset)
    by_dataset_success = pd.DataFrame(
        [
            [d, per_dataset[d]["n"], *[per_dataset[d][n]["success_rate"] for n in names]]
            for d in datasets
        ],
        columns=["dataset", "n", *names],
    )
    by_dataset_cost = pd.DataFrame(
        [
            [d, per_dataset[d]["n"], *[per_dataset[d][n]["mean_cost_usd"] for n in names]]
            for d in datasets
        ],
        columns=["dataset", "n", *names],
    )
    return {
        "Setup": setup,
        "Quality and cost on the test split": quality,
        "Against always_strongest (paired by item)": against,
        "Random policy over seeds 0 to 4": spread,
        "Success rate by dataset": by_dataset_success,
        "Mean cost per item (USD) by dataset": by_dataset_cost,
    }


def _pareto_figure(points: pd.DataFrame, front: pd.DataFrame, pool_name: str) -> Figure:
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.scatter(points["mean_cost_usd"], points["success_rate"], color="tab:blue", zorder=3)
    for row in points.itertuples(index=False):
        ax.annotate(
            row.policy,
            (row.mean_cost_usd, row.success_rate),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.step(front["mean_cost_usd"], front["success_rate"], where="post", color="tab:red", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per item (USD)")
    ax.set_ylabel("success rate")
    ax.set_title(f"Baselines on the test split, {pool_name} pool (red: Pareto front)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the static baselines on the test split.")
    parser.add_argument("--pool", choices=("flagship", "mixed"), required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "baselines")
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_baselines(args.derived, args.pool, args.out_root, command)
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
