"""Why the quality floor is compared with the lower confidence bound and not with P.

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/lcb_vs_point.py --pool flagship

The learned ``min_cost`` mode picks the cheapest model that clears a quality floor. By default
"clears" means the model's lower confidence bound L is at least the floor (``--floor-on lcb``);
the experiment arm compares P, the calibrated success probability, with the floor instead
(``--floor-on point``) and changes nothing else. Both arms are replayed at the floors 0.05,
0.10, ..., 0.60, which are fixed in advance (nothing is chosen from the results), on the test
split and, for reference, on the calibration split.

Per floor and arm the script reports the share of decisions that were not relaxed (a decision
whose floor the router had to give up made no promise), the realized success among the decisions
that were not relaxed with a Wilson 95% interval, whether it is at least the floor (floor
compliance, Ruling V1-R5), the mean cost over all decisions and the share relaxed. A reliability
table of the chosen model's L against the realized success is pooled over the floors. The
calibration rows are in-sample for the calibrators and tables, so only the test rows are
evidence of compliance.

Writes bench/results/lcb_vs_point/<run-id>/{results.json,report.md,compliance.png,run.json}.
Output holds numbers, model names and item IDs only, never prompt text.
"""

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from matplotlib.figure import Figure

from routepilot_train import replaytool, runinfo
from routepilot_train.evaluate import metrics, report

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeps  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "lcb_vs_point"
ARMS = ("lcb", "point")
SPLITS = ("test", "calib")
FLOORS = tuple(round(0.05 * i, 2) for i in range(1, 13))  # 0.05, 0.10, ..., 0.60
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"


def arm_flags(floor: float, arm: str) -> list[str]:
    """Replay flags of ``min_cost`` at ``floor``: the point arm adds ``--floor-on point``."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    flags = sweeps.floor_flags(floor)
    return [*flags, "--floor-on", "point"] if arm == "point" else flags


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """The Wilson score interval of a proportion, or None without trials."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compliance_row(frame: pd.DataFrame, floor: float, split: str) -> dict[str, Any]:
    """Floor compliance of one ``min_cost`` replay at ``floor``.

    ``realized_success`` is over the decisions that were not relaxed; ``met`` is
    ``realized_success >= floor`` (None when every decision was relaxed).
    """
    point = sweeps.summarize_point(frame, split)
    comp = metrics.floor_compliance(frame)
    kept = comp["n"]
    realized = comp["realized_success"]
    hits = round(realized * kept) if realized is not None else 0
    interval = wilson_interval(hits, kept)
    return {
        "floor": floor,
        "n": point["n"],
        "n_not_relaxed": kept,
        "not_relaxed_share": kept / point["n"],
        "relaxed_share": point["relaxed_share"],
        "relaxed_quality_share": point["relaxed_quality_share"],
        "realized_success": realized,
        "realized_ci": list(interval) if interval else None,
        "met": None if realized is None else bool(realized >= floor),
        "mean_cost_usd": point["mean_cost_usd"],
        "success_rate": point["success_rate"],
        "model_counts": point["model_counts"],
    }


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many floors the arm met: floors with decisions, floors met and the lowest missed."""
    decided = [r for r in rows if r["met"] is not None]
    missed = [r["floor"] for r in decided if not r["met"]]
    return {
        "floors": len(rows),
        "with_decisions": len(decided),
        "met": sum(1 for r in decided if r["met"]),
        "missed_floors": missed,
    }


def run_lcb_vs_point(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    floors: Sequence[float] = FLOORS,
    workers: int = sweeps.WORKERS,
) -> Path:
    """Replay both arms at every floor on both splits and write ``out_root/<run-id>``."""
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    binary = binary or replaytool.build_go(ROOT)
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    jobs = [
        (split, "learned", learned + arm_flags(q, arm))
        for split in SPLITS
        for arm in ARMS
        for q in floors
    ]
    with tempfile.TemporaryDirectory() as scratch:
        rp = sweeps.Replayer(binary, config, replay_input, Path(scratch), workers)
        try:
            frames = rp.many(jobs)
            replay_counts = rp.counts()
        finally:
            rp.close()

    by_key = dict(
        zip(((s, a, q) for s in SPLITS for a in ARMS for q in floors), frames, strict=True)
    )
    arms: dict[str, dict[str, Any]] = {split: {} for split in SPLITS}
    versions: dict[str, str] = {}
    for split in SPLITS:
        for arm in ARMS:
            rows = [compliance_row(by_key[(split, arm, q)], q, split) for q in floors]
            pooled = pd.concat([by_key[(split, arm, q)] for q in floors])
            arms[split][arm] = {
                "floors": rows,
                "summary": summarize_arm(rows),
                "reliability": metrics.floor_compliance(pooled)["reliability"],
            }
            versions[arm] = str(by_key[(split, arm, floors[0])]["policy_version"].iloc[0])

    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "floors": list(floors),
        "arms": list(ARMS),
        "n_items": {s: arms[s]["lcb"]["floors"][0]["n"] for s in SPLITS},
        "policy_config": {k: v for k, v in base_config.items() if k != "lambda"},
        "policy_versions": versions,
        "results": arms,
        "replays": replay_counts,
        "wall_time_s": time.perf_counter() - clock,
        "labels": _labels(),
    }
    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    shown = (
        list(command)
        if command
        else ["python", "bench/scripts/lcb_vs_point.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {"compliance": _figure(results)},
        {},
        shown,
        artifact.name,
        started,
        git=start_git,
    )
    return out_dir


# --- report --------------------------------------------------------------------------------------


def _labels() -> list[str]:
    return [
        "no latency is measured or simulated in this experiment",
        "costs are the recorded costs of each model on each item",
        "the floors are fixed in advance and nothing is chosen from these results",
        "a decision whose floor the router relaxed made no promise and is left out of the "
        "realized success",
        "the calibration rows are in-sample for the calibrators and lower-bound tables, so only "
        "the test rows are evidence of compliance",
        "the point arm is an experiment switch (--floor-on point); the gateway always compares "
        "the lower confidence bound",
    ]


def _ci(row: Mapping[str, Any]) -> str:
    if row["realized_success"] is None:
        return "n/a"
    lo, hi = row["realized_ci"]
    return f"{row['realized_success']:.3f} [{lo:.3f}, {hi:.3f}]"


def _met(row: Mapping[str, Any]) -> str:
    return "n/a" if row["met"] is None else ("yes" if row["met"] else "NO")


def _split_table(results: Mapping[str, Any], split: str) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(results["floors"]):
        row: dict[str, Any] = {"floor": q}
        for arm in results["arms"]:
            r = results["results"][split][arm]["floors"][i]
            row |= {
                f"{arm}: not relaxed": r["not_relaxed_share"],
                f"{arm}: realized success [95% CI]": _ci(r),
                f"{arm}: >= floor": _met(r),
                f"{arm}: mean cost USD": r["mean_cost_usd"],
                f"{arm}: relaxed": r["relaxed_share"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for arm in results["arms"]:
            s = results["results"][split][arm]["summary"]
            rows.append(
                {
                    "split": split,
                    "arm": arm,
                    "floors": s["floors"],
                    "floors with decisions": s["with_decisions"],
                    "floors met": s["met"],
                    "floors missed": ", ".join(str(q) for q in s["missed_floors"]) or "none",
                }
            )
    return pd.DataFrame(rows)


def _reliability_table(results: Mapping[str, Any], split: str) -> pd.DataFrame:
    rows = []
    for arm in results["arms"]:
        for r in results["results"][split][arm]["reliability"]:
            rows.append(
                {
                    "arm": arm,
                    "decile of L": int(r["decile"]),
                    "decisions": int(r["n"]),
                    "mean L": r["mean_l"],
                    "realized success": r["realized_success"],
                }
            )
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    items = results["n_items"]
    return {
        "Setup": pd.DataFrame(
            [
                ["pool", results["pool"]],
                ["artifact version", results["artifact_version"]],
                ["items (test / calibration)", f"{items['test']} / {items['calib']}"],
                ["floors", ", ".join(str(q) for q in results["floors"])],
                ["policy versions", json.dumps(results["policy_versions"], sort_keys=True)],
                ["policy config (other than lambda)", json.dumps(results["policy_config"])],
                ["wall time (s)", round(results["wall_time_s"], 1)],
            ],
            columns=["setting", "value"],
        ),
        "Floors met (realized success among non-relaxed decisions at least the floor)": (
            _summary_table(results)
        ),
        "Test split: floor compliance and cost per floor": _split_table(results, "test"),
        "Calibration split, for reference (in-sample): floor compliance and cost per floor": (
            _split_table(results, "calib")
        ),
        "Test split: realized success per decile of L, pooled over the floors": (
            _reliability_table(results, "test")
        ),
    }


def _figure(results: Mapping[str, Any]) -> Figure:
    fig = Figure(figsize=(10.0, 4.4))
    axes = fig.subplots(1, 2)
    colors = {"lcb": "tab:blue", "point": "tab:red"}
    for ax, split in zip(axes, SPLITS, strict=True):
        ax.plot([0, 0.65], [0, 0.65], color="0.5", linestyle="--", linewidth=0.9, label="floor")
        for arm in results["arms"]:
            rows = [r for r in results["results"][split][arm]["floors"] if r["met"] is not None]
            ax.errorbar(
                [r["floor"] for r in rows],
                [r["realized_success"] for r in rows],
                yerr=[
                    [r["realized_success"] - r["realized_ci"][0] for r in rows],
                    [r["realized_ci"][1] - r["realized_success"] for r in rows],
                ],
                marker="o",
                markersize=3,
                capsize=2,
                linewidth=1,
                color=colors[arm],
                label=f"floor on {arm}",
            )
        title = "test" if split == "test" else "calibration (in-sample, for reference)"
        ax.set_title(f"{results['pool']} pool, {title}", fontsize=9)
        ax.set_xlabel("quality floor")
        ax.set_ylabel("realized success, non-relaxed (Wilson 95%)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Floor on the lower bound versus on P.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument("--policy-config", type=Path, default=POLICY_CONFIG)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "bench" / "results" / "lcb_vs_point"
    )
    parser.add_argument("--workers", type=int, default=sweeps.WORKERS)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_lcb_vs_point(
        args.derived,
        args.pool,
        args.artifact or sweeps.find_artifact(ROOT / "policies", args.pool),
        args.policy_config,
        args.out_root,
        command,
        workers=args.workers,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
