"""Distribution shift and out-of-distribution (OOD) detection on the held-out datasets.

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/shift_ood.py --pool flagship

The held-out datasets were fixed before training (they are the datasets of a domain the
training, calibration and test splits never contain) and appear only in this experiment. The
OOD threshold of the artifact comes from the calibration split. Everything goes through the Go
``routepilot replay`` command.

Policy. The learned policy at the spec operating point (``--policy-config``, default
``bench/policies/flagship.yaml``): ``dual`` at that file's lambda, and ``min_cost`` at the
fixed floor 0.3. Each is replayed on the held-out split twice: with the shipped OOD action
(``ood_action`` of the policy config) and with ``--ood off``, which makes the optimizer ignore
the OOD verdict (no pooled replacement, no strongest action; the record still carries ``ood``
and ``ood_score``). ``always_strongest`` (Best Single) is the reference, replayed on the same
held-out items.

Per arm: success and mean cost, paired deltas against always_strongest, the share of items the
OOD detector flagged, the share of decisions whose floor was relaxed, and, for the ``min_cost``
arms, floor compliance (Ruling V1-R5: the realized success among the decisions that were not
relaxed, at least the floor). The paired difference between the two OOD arms says what the OOD
action bought. A per held-out dataset table repeats this by dataset.

OOD detection quality: the AUROC of ``ood_score`` with the held-out items as positives and the
test items as negatives (request-level scores, from replay records), with a bootstrap interval,
overall and per held-out dataset, and the flag rate on each side. The test split is replayed
once, for these scores only.

Writes bench/results/shift_ood/<run-id>/{results.json,report.md,ood_scores.png,run.json}.
Output holds numbers, model names, dataset names and item IDs only, never prompt text.
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

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from scipy.stats import rankdata

from routepilot_train import replaytool, runinfo
from routepilot_train.data.splits import require_splits
from routepilot_train.evaluate import metrics, report
from routepilot_train.export import artifact as artifact_io

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (sibling scripts, not packages)
import lcb_vs_point  # noqa: E402
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "shift_ood"
TEST, HELDOUT = "test", "heldout"
FLOOR = 0.3  # the floor of the min_cost arms
POLICY_ARMS = ("dual", "min_cost")
OOD_ARMS = ("shipped", "off")
REFERENCE = baselines.REFERENCE
BOOTSTRAP = 1000
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"


# --- arms ----------------------------------------------------------------------------------------


def arm_flags(policy_arm: str, ood_arm: str, lam: float, floor: float) -> list[str]:
    """Replay flags of a policy arm (``dual`` at ``lam``, ``min_cost`` at ``floor``).

    The ``off`` OOD arm adds ``--ood off``; ``shipped`` leaves the policy config's action alone.
    """
    if policy_arm == "dual":
        flags = sweeps.dual_flags(lam)
    elif policy_arm == "min_cost":
        flags = sweeps.floor_flags(floor)
    else:
        raise ValueError(f"policy arm must be one of {POLICY_ARMS}, got {policy_arm!r}")
    if ood_arm == "off":
        return [*flags, "--ood", "off"]
    if ood_arm != "shipped":
        raise ValueError(f"ood arm must be one of {OOD_ARMS}, got {ood_arm!r}")
    return flags


# --- OOD detection quality ------------------------------------------------------------------------


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(positive > negative) with a tie counting a half."""
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def _scores(values: Sequence[float] | np.ndarray, what: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.size == 0:
        raise ValueError(f"auroc needs at least one {what} score")
    return scores


def auroc(
    positives: Sequence[float] | np.ndarray, negatives: Sequence[float] | np.ndarray
) -> float:
    """The probability that a random positive scores above a random negative (ties count half).

    Raises ValueError without a positive or a negative score or with a non-finite one.
    """
    pos, neg = _scores(positives, "positive"), _scores(negatives, "negative")
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("auroc needs finite scores")
    return _auroc(pos, neg)


def auroc_ci(
    positives: Sequence[float] | np.ndarray,
    negatives: Sequence[float] | np.ndarray,
    n_boot: int = BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """The AUROC and its percentile bootstrap interval, as (point, lo, hi).

    The positives and the negatives are resampled separately, ``n_boot`` times with a fixed
    ``seed``, so the same scores always give the same interval.
    """
    point = auroc(positives, negatives)
    pos, neg = np.asarray(positives, dtype=float), np.asarray(negatives, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        values[b] = _auroc(
            pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)]
        )
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def _spread(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p95": float(np.quantile(scores, 0.95)),
    }


def ood_detection(
    test_frame: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    threshold: float,
    n_boot: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """How well ``ood_score`` separates the held-out items (positives) from the test items.

    Both frames are learned-policy replay records (``ood_score``, ``ood``, ``dataset``).
    Returns the AUROC with its interval, the two flag rates, the score spreads and the same
    AUROC and flag rate for each held-out dataset against all of the test items. Raises
    ValueError when a frame holds rows of another split.
    """
    require_splits(test_frame, {TEST})
    require_splits(heldout_frame, {HELDOUT})
    negatives = test_frame["ood_score"].to_numpy(dtype=float)
    positives = heldout_frame["ood_score"].to_numpy(dtype=float)
    point, lo, hi = auroc_ci(positives, negatives, n_boot, seed)
    per_dataset = []
    for dataset in sorted(heldout_frame["dataset"].unique()):
        part = heldout_frame.loc[heldout_frame["dataset"] == dataset]
        point_d, lo_d, hi_d = auroc_ci(
            part["ood_score"].to_numpy(dtype=float), negatives, n_boot, seed
        )
        per_dataset.append(
            {
                "dataset": str(dataset),
                "n": len(part),
                "auroc": point_d,
                "auroc_ci": [lo_d, hi_d],
                "flag_rate": float(part["ood"].astype(bool).mean()),
                "scores": _spread(part["ood_score"].to_numpy(dtype=float)),
            }
        )
    test_per_dataset = [
        {
            "dataset": str(dataset),
            "n": len(part),
            "flag_rate": float(part["ood"].astype(bool).mean()),
        }
        for dataset, part in sorted(test_frame.groupby("dataset"), key=lambda kv: str(kv[0]))
    ]
    return {
        "auroc": point,
        "auroc_ci": [lo, hi],
        "n_positive": int(positives.size),
        "n_negative": int(negatives.size),
        "threshold": float(threshold),
        "flag_rate_test": float(test_frame["ood"].astype(bool).mean()),
        "flag_rate_heldout": float(heldout_frame["ood"].astype(bool).mean()),
        "scores_test": _spread(negatives),
        "scores_heldout": _spread(positives),
        "per_dataset": per_dataset,
        "test_per_dataset": test_per_dataset,
        "bootstrap": {"n": n_boot, "seed": seed, "alpha": 0.05, "method": "percentile"},
    }


# --- the experiment -----------------------------------------------------------------------------


def _same_verdict(frames: Mapping[str, pd.DataFrame]) -> None:
    """Raise ValueError unless every arm saw the same items with the same OOD verdict and score.

    The detector is part of the artifact and the arms only differ in what the optimizer does
    with its verdict, so a difference would mean the replays are not comparable.
    """
    names = list(frames)
    first = frames[names[0]]
    for name in names[1:]:
        other = frames[name]
        same_items = list(other["item_id"]) == list(first["item_id"])
        if not same_items or not other[["ood", "ood_score"]].equals(first[["ood", "ood_score"]]):
            raise ValueError(f"arms {names[0]} and {name} disagree on the OOD verdict or the items")


def _arm_row(
    name: str,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    bootstrap: int,
    floor: float | None,
) -> dict[str, Any]:
    point = sweeps.summarize_point(frame, HELDOUT)  # asserts the split
    row = baselines._policy_row(name, frame, reference, bootstrap)
    row |= {
        "policy_version": str(frame["policy_version"].iloc[0]),
        "ood_flag_rate": float(frame["ood"].astype(bool).mean()),
        "relaxed_share": point["relaxed_share"],
        "relaxed_quality_share": point["relaxed_quality_share"],
    }
    if floor is not None:
        row["compliance"] = lcb_vs_point.compliance_row(frame, floor, HELDOUT)
    return row


def _action_effect(shipped: pd.DataFrame, off: pd.DataFrame, bootstrap: int) -> dict[str, Any]:
    """The shipped OOD action against ignoring the verdict, paired on the same items.

    ``success_delta`` is shipped minus off; ``cost_reduction`` is positive when the action made
    the policy cheaper. ``items_changed`` counts the items where the two arms chose a model
    differently.
    """
    delta = metrics.paired_delta(shipped, off, n=bootstrap, seed=baselines.BOOTSTRAP_SEED)
    joined = shipped[["item_id", "model"]].merge(
        off[["item_id", "model"]], on="item_id", suffixes=("_shipped", "_off"), validate="1:1"
    )
    changed = int((joined["model_shipped"].fillna("") != joined["model_off"].fillna("")).sum())
    return {**delta, "items_changed": changed}


def _dataset_rows(
    reference: pd.DataFrame, arms: Mapping[str, pd.DataFrame], bootstrap: int
) -> list[dict[str, Any]]:
    """Success, cost, OOD flag rate and change against always_strongest, per held-out dataset."""
    rows = []
    for dataset in sorted(reference["dataset"].unique()):
        ref = reference.loc[reference["dataset"] == dataset]
        for name, frame in {REFERENCE: reference, **arms}.items():
            part = frame.loc[frame["dataset"] == dataset]
            summary = metrics.summarize(part)
            row: dict[str, Any] = {
                "dataset": str(dataset),
                "arm": name,
                "n": summary["n"],
                "success_rate": summary["success_rate"],
                "mean_cost_usd": summary["mean_cost_usd"],
                "ood_flag_rate": None,
                "success_delta_vs_strongest": None,
                "success_delta_ci": None,
                "cost_reduction_vs_strongest": None,
            }
            if name != REFERENCE:
                delta = metrics.paired_delta(part, ref, n=bootstrap, seed=baselines.BOOTSTRAP_SEED)
                row["ood_flag_rate"] = float(part["ood"].astype(bool).mean())
                row["success_delta_vs_strongest"] = delta["success_delta"]
                row["success_delta_ci"] = delta["success_delta_ci"]
                row["cost_reduction_vs_strongest"] = delta["cost_reduction"]
            rows.append(row)
    return rows


def run_shift_ood(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    bootstrap: int = BOOTSTRAP,
    workers: int = sweeps.WORKERS,
    floor: float = FLOOR,
) -> Path:
    """Replay the arms on the held-out split and write the results under ``out_root/<run-id>``.

    The test split is replayed once, with the shipped dual arm, for its OOD scores only.
    Raises ValueError when the policy config has no lambda or the arms disagree on the OOD
    verdict, and RuntimeError when the replay input has no held-out (or test) items.
    """
    started = datetime.now(UTC)
    clock = time.perf_counter()
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    strong = pool["best_single"]
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    if "lambda" not in base_config:
        raise ValueError(f"{policy_config} has no lambda")
    lam = float(base_config["lambda"])
    manifest, _ = artifact_io.load_artifact(artifact)
    threshold = float(manifest["ood"]["threshold"])
    binary = binary or replaytool.build_go(ROOT)
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    arm_keys = [(p, o) for p in POLICY_ARMS for o in OOD_ARMS]
    arm_names = [f"{p}/{o}" for p, o in arm_keys]
    jobs: list[tuple[str, str, Sequence[str]]] = [(HELDOUT, "fixed", ["--model", strong])]
    jobs += [(HELDOUT, "learned", learned + arm_flags(p, o, lam, floor)) for p, o in arm_keys]
    jobs.append((TEST, "learned", learned + arm_flags("dual", "shipped", lam, floor)))

    tick = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        rp = sweeps.Replayer(binary, config, replay_input, Path(scratch), workers)
        try:
            frames = rp.many(jobs)
            replay_counts = rp.counts()
        finally:
            rp.close()
    replay_seconds = time.perf_counter() - tick
    reference, test_frame = frames[0], frames[-1]
    arm_frames = dict(zip(arm_names, frames[1:-1], strict=True))
    _same_verdict(arm_frames)
    if list(arm_frames[arm_names[0]]["item_id"]) != list(reference["item_id"]):
        raise ValueError("the arms and always_strongest were not replayed on the same items")

    tick = time.perf_counter()
    arms = {
        name: _arm_row(
            name, frame, reference, bootstrap, floor if name.startswith("min_cost") else None
        )
        for name, frame in arm_frames.items()
    }
    effect = {
        p: _action_effect(arm_frames[f"{p}/shipped"], arm_frames[f"{p}/off"], bootstrap)
        for p in POLICY_ARMS
    }
    detection = ood_detection(test_frame, arm_frames[arm_names[0]], threshold, bootstrap)
    per_dataset = _dataset_rows(reference, arm_frames, bootstrap)
    stats_seconds = time.perf_counter() - tick

    sizes = reference["dataset"].value_counts().sort_index()
    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "n_items": {HELDOUT: len(reference), TEST: len(test_frame)},
        "heldout_datasets": [str(d) for d in sizes.index],
        "heldout_sizes": {str(d): int(n) for d, n in sizes.items()},
        "best_single": strong,
        "floor": floor,
        "lambda": lam,
        "ood_action": base_config.get("ood_action"),
        "policy_config": {k: v for k, v in base_config.items() if k != "lambda"},
        "bootstrap": {
            "n": bootstrap,
            "seed": baselines.BOOTSTRAP_SEED,
            "alpha": 0.05,
            "method": "percentile",
        },
        "reference": baselines._policy_row(REFERENCE, reference, reference, bootstrap),
        "arms": arms,
        "action_effect": effect,
        "ood_detection": detection,
        "per_dataset": per_dataset,
        "replays": replay_counts,
        "wall_time_s": time.perf_counter() - clock,
        "phase_time_s": {"replays": replay_seconds, "statistics": stats_seconds},
        "labels": _labels(base_config.get("ood_action")),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started)
    shown = (
        list(command) if command else ["python", "bench/scripts/shift_ood.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {"ood_scores": _figure(test_frame, arm_frames[arm_names[0]], threshold, pool_name)},
        {"bootstrap": baselines.BOOTSTRAP_SEED},
        shown,
        artifact.name,
        started,
    )
    return out_dir


# --- report --------------------------------------------------------------------------------------


def _labels(ood_action: str | None) -> list[str]:
    return [
        "held-out datasets were fixed before training (Task 3) and appear only in the shift and "
        "OOD experiments; the OOD threshold comes from calibration",
        "no latency is measured or simulated in this experiment",
        "costs are the recorded costs of each model on each item",
        f"shipped is the policy config's OOD action ({ood_action}); off makes the optimizer "
        "ignore the OOD verdict, an experiment switch (--ood off) the gateway never uses",
        "the OOD verdict and score are request-level and computed by the artifact, so they are "
        "the same in every arm; the off arm computes them and does not act on them",
        "the test split is replayed once, only for its OOD scores (the negatives of the AUROC)",
        "floor compliance counts only the decisions whose floor was not relaxed (Ruling V1-R5)",
    ]


def _ci(point: float | None, ci: Sequence[float] | None, digits: str = ".4f") -> str:
    return baselines._ci_text(point, ci, digits)


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    boot, items, replays = results["bootstrap"], results["n_items"], results["replays"]
    rows: list[list[Any]] = [
        ["pool", results["pool"]],
        ["artifact version", results["artifact_version"]],
        [
            "items (held-out / test, test for the OOD scores only)",
            f"{items[HELDOUT]} / {items[TEST]}",
        ],
        ["Best Single (always_strongest, the reference)", results["best_single"]],
        ["dual lambda (USD^-1, from the policy config)", repr(results["lambda"])],
        ["min_cost floor (fixed in advance)", results["floor"]],
        ["policy config (other than lambda)", json.dumps(results["policy_config"])],
        ["OOD threshold (from calibration)", results["ood_detection"]["threshold"]],
        ["bootstrap", f"{boot['n']} resamples, percentile 95% intervals, seed {boot['seed']}"],
        [
            "replays run (held-out / test)",
            f"{replays.get(HELDOUT, 0)} / {replays.get(TEST, 0)}",
        ],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _datasets_table(results: Mapping[str, Any]) -> pd.DataFrame:
    total = sum(results["heldout_sizes"].values())
    return pd.DataFrame(
        [
            {"dataset": d, "items": n, "share of the held-out split": n / total}
            for d, n in results["heldout_sizes"].items()
        ]
    )


def _arms_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, r in {REFERENCE: results["reference"], **results["arms"]}.items():
        reference = name == REFERENCE
        rows.append(
            {
                "arm": name,
                "success [95% CI]": _ci(r["success_rate"], r["success_ci"]),
                "mean cost USD [95% CI]": _ci(r["mean_cost_usd"], r["mean_cost_ci"], ".6g"),
                "success delta vs always_strongest [95% CI]": "reference"
                if reference
                else _ci(r["success_delta_vs_strongest"], r["success_delta_ci"]),
                "cost reduction vs always_strongest [95% CI]": "reference"
                if reference
                else _ci(r["cost_reduction_vs_strongest"], r["cost_reduction_ci"]),
                "OOD flagged": None if reference else r["ood_flag_rate"],
                "floor relaxed": None if reference else r["relaxed_share"],
            }
        )
    return pd.DataFrame(rows)


def _effect_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for policy_arm, e in results["action_effect"].items():
        rows.append(
            {
                "policy": policy_arm,
                "items": e["n"],
                "items where the arms chose differently": e["items_changed"],
                "success delta, shipped minus off [95% CI]": _ci(
                    e["success_delta"], e["success_delta_ci"]
                ),
                "cost reduction by the action [95% CI]": _ci(
                    e["cost_reduction"], e["cost_reduction_ci"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _compliance_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, r in results["arms"].items():
        c = r.get("compliance")
        if c is None:
            continue
        met = "n/a" if c["met"] is None else ("yes" if c["met"] else "NO")
        rows.append(
            {
                "arm": name,
                "floor": c["floor"],
                "decisions": c["n"],
                "not relaxed": c["not_relaxed_share"],
                "realized success among them [95% Wilson CI]": _ci(
                    c["realized_success"], c["realized_ci"], ".3f"
                ),
                "realized >= floor": met,
                "mean cost USD": c["mean_cost_usd"],
            }
        )
    return pd.DataFrame(rows)


def _detection_table(results: Mapping[str, Any]) -> pd.DataFrame:
    det = results["ood_detection"]
    rows: list[dict[str, Any]] = [
        {
            "held-out set": "all held-out items",
            "items": det["n_positive"],
            "AUROC against test [95% CI]": _ci(det["auroc"], det["auroc_ci"]),
            "OOD flag rate": det["flag_rate_heldout"],
            "median ood_score": det["scores_heldout"]["median"],
        }
    ]
    for r in det["per_dataset"]:
        rows.append(
            {
                "held-out set": r["dataset"],
                "items": r["n"],
                "AUROC against test [95% CI]": _ci(r["auroc"], r["auroc_ci"]),
                "OOD flag rate": r["flag_rate"],
                "median ood_score": r["scores"]["median"],
            }
        )
    rows.append(
        {
            "held-out set": "test split (the negatives)",
            "items": det["n_negative"],
            "AUROC against test [95% CI]": "n/a",
            "OOD flag rate": det["flag_rate_test"],
            "median ood_score": det["scores_test"]["median"],
        }
    )
    return pd.DataFrame(rows)


def _per_dataset_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for r in results["per_dataset"]:
        reference = r["arm"] == REFERENCE
        rows.append(
            {
                "dataset": r["dataset"],
                "arm": r["arm"],
                "items": r["n"],
                "success": r["success_rate"],
                "mean cost USD": r["mean_cost_usd"],
                "success delta vs always_strongest [95% CI]": "reference"
                if reference
                else _ci(r["success_delta_vs_strongest"], r["success_delta_ci"]),
                "cost reduction vs always_strongest": r["cost_reduction_vs_strongest"],
                "OOD flagged": r["ood_flag_rate"],
            }
        )
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    floor = results["floor"]
    return {
        "Setup": _setup_table(results),
        "Held-out datasets": _datasets_table(results),
        "Held-out split: each arm against always_strongest (paired on the same items)": (
            _arms_table(results)
        ),
        "OOD action effect (shipped against off, paired on the same held-out items)": (
            _effect_table(results)
        ),
        f"Floor compliance on the held-out split (min_cost, floor {floor}, Ruling V1-R5)": (
            _compliance_table(results)
        ),
        "OOD detection: AUROC of ood_score, held-out items as positives, test items as negatives": (
            _detection_table(results)
        ),
        "Per held-out dataset: each arm against always_strongest": _per_dataset_table(results),
    }


def _figure(
    test_frame: pd.DataFrame, heldout_frame: pd.DataFrame, threshold: float, pool_name: str
) -> Figure:
    test_scores = test_frame["ood_score"].to_numpy(dtype=float)
    held_scores = heldout_frame["ood_score"].to_numpy(dtype=float)
    low = float(min(test_scores.min(), held_scores.min()))
    high = float(max(test_scores.max(), held_scores.max()))
    bins = np.linspace(low, high if high > low else low + 1.0, 41)
    fig = Figure(figsize=(11.0, 4.4))
    left, right = fig.subplots(1, 2)
    left.hist(
        test_scores, bins=bins, density=True, alpha=0.55, label=f"test (n={len(test_scores)})"
    )
    left.hist(
        held_scores, bins=bins, density=True, alpha=0.55, label=f"held-out (n={len(held_scores)})"
    )
    left.set_title(f"OOD score, {pool_name} pool", fontsize=9)
    right.hist(
        test_scores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        color="0.2",
        label="test",
    )
    for dataset in sorted(heldout_frame["dataset"].unique()):
        part = heldout_frame.loc[heldout_frame["dataset"] == dataset, "ood_score"]
        right.hist(
            part.to_numpy(dtype=float),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            label=f"{dataset} (n={len(part)})",
        )
    right.set_title("OOD score by held-out dataset", fontsize=9)
    for ax in (left, right):
        ax.axvline(
            threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label="OOD threshold (calibration)",
        )
        ax.set_xlabel("ood_score (request-level)")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distribution shift and OOD detection.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument("--policy-config", type=Path, default=POLICY_CONFIG)
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / EXPERIMENT)
    parser.add_argument("--workers", type=int, default=sweeps.WORKERS)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_shift_ood(
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
