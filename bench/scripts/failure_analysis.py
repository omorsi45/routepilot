"""Failure analysis of the flagship learned policy: where routing loses to Best Single.

    python bench/scripts/failure_analysis.py

Replays the flagship test split twice with the Go ``routepilot replay`` command (each once):
always Best Single, and the learned ``dual`` operating point (the lambda of
``bench/policies/flagship.yaml``, tuned on calibration by ``sweeps.py``, the point the headline
reports). Nothing is tuned here and no choice looks at the test split.

Per item the two outcomes are paired. A *loss* is an item Best Single got right and the routed
model got wrong; a *gain* is the reverse. The report breaks the success and the cost down by
dataset, by the model the policy chose, and by whether the request was out of distribution, and
lists the 20 items with the largest loss. Every loss costs one success, so the 20 are ranked by
how sure the policy was: the predicted success probability of the model it chose (highest
first), then by item ID. An item is listed by its ID, dataset and chosen model only.

Costs are the recorded costs of each model on each item, in USD. The output holds numbers,
model names, dataset names and item IDs only, never prompt text.

Writes ``bench/results/failure_analysis/<run-id>/{results.json,report.md,run.json}``.
"""

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from routepilot_train import replaytool, runinfo
from routepilot_train.evaluate import metrics, report

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (sibling scripts, not packages)
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "failure_analysis"
POOL = "flagship"
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"
RESULTS_ROOT = ROOT / "bench" / "results" / "failure_analysis"
TOP_LOSSES = 20
PICO = 1e12
NO_MODEL = "(routing error)"

LABELS = (
    "costs are the recorded costs of each model on each item",
    "latency is not used anywhere in this analysis",
    "the learned point is the one the headline reports: lambda tuned on the calibration split",
    "a loss is an item Best Single got right and the routed model got wrong; a gain is the reverse",
    "the test split is read once per replay; nothing is chosen on it",
    "no prompt text is in this report",
)


# --- pairing and tables -------------------------------------------------------------------------


def pair(best: pd.DataFrame, routed: pd.DataFrame) -> pd.DataFrame:
    """One row per item: Best Single's and the routed policy's outcome side by side.

    Both frames are replays of the same items. Columns: ``item_id``, ``dataset``,
    ``best_success``, ``best_cost_usd``, ``chosen`` (the routed model, ``NO_MODEL`` after a
    routing error), ``routed_success``, ``routed_cost_usd``, ``ood``, ``p``, ``loss`` and
    ``gain``.
    """
    if best["item_id"].duplicated().any() or routed["item_id"].duplicated().any():
        raise ValueError("duplicate item_id in a replay")
    if set(best["item_id"]) != set(routed["item_id"]):
        raise ValueError("the two replays must hold the same items")
    left = best[["item_id", "dataset", "success", "cost_picousd"]].rename(
        columns={"success": "best_success", "cost_picousd": "best_cost"}
    )
    right = routed[["item_id", "model", "success", "cost_picousd", "ood", "p"]].rename(
        columns={"model": "chosen", "success": "routed_success", "cost_picousd": "routed_cost"}
    )
    out = left.merge(right, on="item_id", validate="1:1")
    out["best_success"] = out["best_success"].astype(bool)
    out["routed_success"] = out["routed_success"].astype(bool)
    out["ood"] = out["ood"].astype(bool)
    out["chosen"] = out["chosen"].fillna(NO_MODEL)
    out["best_cost_usd"] = out["best_cost"].astype("float64") / PICO
    out["routed_cost_usd"] = out["routed_cost"].astype("float64") / PICO
    out["loss"] = out["best_success"] & ~out["routed_success"]
    out["gain"] = ~out["best_success"] & out["routed_success"]
    return (
        out.drop(columns=["best_cost", "routed_cost"]).sort_values("item_id").reset_index(drop=True)
    )


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    """Counts, success, cost and the paired loss and gain counts of a set of items."""
    n = len(rows)
    best_cost, routed_cost = (
        float(rows["best_cost_usd"].mean()),
        float(rows["routed_cost_usd"].mean()),
    )
    best, routed = float(rows["best_success"].mean()), float(rows["routed_success"].mean())
    return {
        "n": n,
        "best_success": best,
        "routed_success": routed,
        "success_delta": routed - best,
        "best_cost_usd": best_cost,
        "routed_cost_usd": routed_cost,
        "cost_reduction": 1.0 - routed_cost / best_cost if best_cost > 0 else None,
        "both_right": int((rows["best_success"] & rows["routed_success"]).sum()),
        "both_wrong": int((~rows["best_success"] & ~rows["routed_success"]).sum()),
        "losses": int(rows["loss"].sum()),
        "gains": int(rows["gain"].sum()),
        "net": int(rows["gain"].sum() - rows["loss"].sum()),
    }


def breakdown(paired: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    """``summarize`` for each value of column ``key``, worst success delta first."""
    rows = [
        {key: str(value), **summarize(group)} for value, group in paired.groupby(key, sort=True)
    ]
    return sorted(rows, key=lambda r: (r["success_delta"], r[key]))


def top_losses(paired: pd.DataFrame, limit: int = TOP_LOSSES) -> list[dict[str, Any]]:
    """The ``limit`` items Best Single got right and the routed model got wrong, most sure first.

    Ranked by the predicted success probability of the chosen model (highest first, missing
    last), then by item ID. Each entry has the item ID, dataset and chosen model only.
    """
    lost = paired.loc[paired["loss"]].copy()
    lost["_p"] = lost["p"].fillna(-1.0)
    lost = lost.sort_values(["_p", "item_id"], ascending=[False, True]).head(limit)
    return [
        {"rank": i + 1, "item_id": r.item_id, "dataset": r.dataset, "chosen": r.chosen}
        for i, r in enumerate(lost.itertuples(index=False))
    ]


def analyze(best: pd.DataFrame, routed: pd.DataFrame, bootstrap: int = baselines.BOOTSTRAP) -> dict:
    """Everything the report shows, from the two replay frames."""
    paired = pair(best, routed)
    paired["ood_label"] = paired["ood"].map({True: "out of distribution", False: "in distribution"})
    delta = metrics.paired_delta(routed, best, n=bootstrap, seed=baselines.BOOTSTRAP_SEED)
    return {
        "overall": summarize(paired),
        "overall_vs_best_single": {
            "success_delta": delta["success_delta"],
            "success_delta_ci": delta["success_delta_ci"],
            "cost_reduction": delta["cost_reduction"],
            "cost_reduction_ci": delta["cost_reduction_ci"],
            "bootstrap": {"n": bootstrap, "seed": baselines.BOOTSTRAP_SEED, "alpha": 0.05},
        },
        "by_dataset": breakdown(paired, "dataset"),
        "by_chosen_model": breakdown(paired, "chosen"),
        "by_ood": breakdown(paired, "ood_label"),
        "top_losses": top_losses(paired),
        "losses_total": int(paired["loss"].sum()),
    }


# --- the run ------------------------------------------------------------------------------------


def run_failure_analysis(
    derived: Path,
    out_root: Path,
    *,
    pool_name: str = POOL,
    artifact: Path | None = None,
    policy_config: Path = POLICY_CONFIG,
    bootstrap: int = baselines.BOOTSTRAP,
    binary: Path | None = None,
) -> Path:
    """Replay Best Single and the learned dual point on the test split and write the report."""
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    derived, policy_config = Path(derived), Path(policy_config)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    artifact = Path(artifact) if artifact else sweeps.find_artifact(ROOT / "policies", pool_name)
    base = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    if base.get("mode") != "dual" or "lambda" not in base:
        raise ValueError(f"{policy_config} must configure the dual mode with a lambda")
    lam = float(base["lambda"])
    binary = binary or replaytool.build_go(ROOT)
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    with tempfile.TemporaryDirectory() as scratch:
        rp = sweeps.Replayer(binary, config, replay_input, Path(scratch), workers=2)
        try:
            best, routed = rp.many(
                [
                    (sweeps.TEST, "fixed", ["--model", pool["best_single"]]),
                    (sweeps.TEST, "learned", learned + sweeps.dual_flags(lam)),
                ]
            )
        finally:
            rp.close()
    analysis = analyze(best, routed, bootstrap)
    results: dict[str, Any] = {
        "pool": pool_name,
        "split": sweeps.TEST,
        "best_single": pool["best_single"],
        "artifact_version": artifact.name,
        "lambda": lam,
        "policy_config": {k: v for k, v in base.items() if k != "lambda"},
        **analysis,
        "wall_time_s": time.perf_counter() - clock,
        "labels": list(LABELS),
    }
    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    command = ["python", "bench/scripts/failure_analysis.py"]
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {},
        {"bootstrap": baselines.BOOTSTRAP_SEED},
        command,
        artifact.name,
        started,
        git=start_git,
    )
    return out_dir


def _group_table(rows: Sequence[Mapping[str, Any]], key: str, title: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                title: r[key],
                "n": r["n"],
                "Best Single success": r["best_success"],
                "routed success": r["routed_success"],
                "success delta": r["success_delta"],
                "Best Single cost USD": r["best_cost_usd"],
                "routed cost USD": r["routed_cost_usd"],
                "cost reduction": r["cost_reduction"],
                "losses": r["losses"],
                "gains": r["gains"],
                "net": r["net"],
            }
            for r in rows
        ]
    )


def _ci(value: float | None, ci: Sequence[float] | None) -> str:
    return baselines._ci_text(value, ci)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    o, v = results["overall"], results["overall_vs_best_single"]
    overall = pd.DataFrame(
        [
            ["Best Single", results["best_single"]],
            ["learned point", f"dual, lambda {results['lambda']!r}"],
            ["artifact version", results["artifact_version"]],
            ["items (test split)", o["n"]],
            [
                "Best Single success / cost USD",
                f"{o['best_success']:.4f} / {o['best_cost_usd']:.6g}",
            ],
            [
                "routed success / cost USD",
                f"{o['routed_success']:.4f} / {o['routed_cost_usd']:.6g}",
            ],
            ["success delta [95% CI]", _ci(v["success_delta"], v["success_delta_ci"])],
            ["cost reduction [95% CI]", _ci(v["cost_reduction"], v["cost_reduction_ci"])],
            ["both right / both wrong", f"{o['both_right']} / {o['both_wrong']}"],
            ["losses / gains / net", f"{o['losses']} / {o['gains']} / {o['net']}"],
        ],
        columns=["quantity", "value"],
    )
    top = pd.DataFrame(results["top_losses"])
    return {
        "Overall, test split": overall,
        "By dataset (worst success delta first)": _group_table(
            results["by_dataset"], "dataset", "dataset"
        ),
        "By the model the policy chose (worst success delta first)": _group_table(
            results["by_chosen_model"], "chosen", "chosen model"
        ),
        "Out of distribution or not": _group_table(results["by_ood"], "ood_label", "request"),
        f"The {TOP_LOSSES} largest losses: Best Single right, routed model wrong": top,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--artifact", type=Path, default=None, help="default: the one found")
    parser.add_argument("--policy-config", type=Path, default=POLICY_CONFIG)
    args = parser.parse_args(argv)
    out = run_failure_analysis(
        args.derived,
        args.out_root,
        artifact=args.artifact,
        policy_config=args.policy_config,
    )
    print(out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
