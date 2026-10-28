"""Validate the off-policy estimators (IPS, SNIPS, DR) against true replay values.

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/ope_validation.py --pool flagship

The logging policy is the learned ``dual`` policy at the spec lambda (``--policy-config``,
default ``bench/policies/flagship.yaml``) with epsilon-greedy exploration (epsilon 0.1), replayed
on the test split once per seed, for 50 seeds, with ``--emit-candidates`` so every record carries
the policy's own predictions for every model. Four target policies are evaluated from each log
(Ruling V1-R36):

* ``always_strongest``: ``fixed`` on the pool's Best Single.
* ``always_cheapest``: ``fixed`` on the pool's cheapest model.
* ``static_rules``: the pool config's keyword and length rules.
* ``learned_min_cost``: the learned ``min_cost`` policy at floor 0.3, no exploration. (The plan
  said 0.8, written before the lower bounds were known: at 0.8 nearly every flagship decision
  relaxes to quality, which makes that target the highest-L policy, not a floor policy.)

The true value of a target is its own deterministic full-information replay on the same test
items: the success rate and the mean recorded cost. For each seed and target the script computes
the IPS, SNIPS and DR estimates of the success rate and of the mean cost (USD per item) with 95%
percentile bootstrap intervals (500 resamples per estimate, seed = the logging seed), using the
estimators of ``routepilot_train.ope``. Across the 50 seeds it reports, per estimator, the bias
(mean estimate minus the true value), the RMSE, the coverage of the 95% intervals (the share of
seeds whose interval holds the true value) and the mean interval width; per target the effective
sample size of the IPS weights and the support coverage: the share of items where the target's
action had positive logging propensity (epsilon exploration only covers the safe set, and a
decision that relaxed a constraint does not explore).

DR uses one outcome model, the logging policy's own prediction (Ruling V1-R32): ``p`` for
success and ``cost_exp_picousd`` for cost.

Writes bench/results/ope_validation/<run-id>/{results.json,report.md,estimator_error.png,
run.json}. Output holds numbers, model names and item IDs only, never prompt text.
"""

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from routepilot_train import ope, replaytool, runinfo
from routepilot_train.evaluate import metrics, report

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeps  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "ope_validation"
TEST = "test"
SEEDS = range(50)
EPSILON = 0.1
FLOOR = 0.3  # Ruling V1-R36
BOOTSTRAP = 500
ESTIMATORS = ("ips", "snips", "dr")
METRICS = ("success", "cost_usd")
PICO = 1e12
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"
# Largest resampling index matrix drawn at once, in cells (the same bound as metrics.bootstrap_ci,
# so equal weights give the same resamples as ope.estimate).
_MAX_CELLS = 4_000_000


# --- design --------------------------------------------------------------------------------------


def target_specs(
    pool: Mapping[str, Any], learned: Sequence[str], floor: float
) -> list[tuple[str, str, list[str]]]:
    """(name, replay policy, flags) of the four target policies of Ruling V1-R36.

    The learned target is deterministic: epsilon 0 is passed whatever the policy config says.
    """
    return [
        ("always_strongest", "fixed", ["--model", pool["best_single"]]),
        ("always_cheapest", "fixed", ["--model", pool["cheapest"]]),
        ("static_rules", "rules", []),
        (
            "learned_min_cost",
            "learned",
            [*learned, "--mode", "min_cost", "--floor", repr(float(floor)), "--epsilon", "0"],
        ),
    ]


def logging_flags(lam: float, epsilon: float, seed: int) -> list[str]:
    """Flags of the logging policy: learned dual at ``lam``, exploring at ``epsilon``, ``seed``."""
    return [
        *sweeps.dual_flags(lam),
        "--epsilon",
        repr(float(epsilon)),
        "--seed",
        str(seed),
        "--emit-candidates",
    ]


# --- estimators -----------------------------------------------------------------------------------


def snips_estimate(
    r: Sequence[float] | np.ndarray,
    a_logged: Sequence[Any] | np.ndarray,
    propensity: Sequence[float] | np.ndarray,
    a_target: Sequence[Any] | np.ndarray,
    n_boot: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """The SNIPS estimate with a 95% percentile bootstrap interval, as ``ope.estimate`` does.

    The point is ``ope.snips``. The interval resamples the items (the same resampling stream as
    ``ope.estimate`` for the same ``seed``) and recomputes the weighted mean on each resample, so
    with equal weights it is the interval of the plain mean. A resample with no agreeing item has
    no estimate and is left out. The estimate and the interval are NaN when the two policies
    never agree.
    """
    point = ope.snips(r, a_logged, propensity, a_target)  # validates the inputs
    rewards = np.asarray(r, dtype=float)
    agree = np.asarray(a_logged, dtype=object) == np.asarray(a_target, dtype=object)
    weights = agree.astype(float) / np.asarray(propensity, dtype=float)
    weighted = weights * rewards
    n = rewards.size
    rng = np.random.default_rng(seed)
    step = max(1, _MAX_CELLS // n)
    ratios: list[np.ndarray] = []
    for start in range(0, n_boot, step):
        idx = rng.integers(0, n, size=(min(step, n_boot - start), n))
        den, num = weights[idx].sum(axis=1), weighted[idx].sum(axis=1)
        ratios.append(num[den > 0] / den[den > 0])
    values = np.concatenate(ratios)
    if math.isnan(point) or values.size == 0:
        return {"mean": point, "lo": math.nan, "hi": math.nan, "n": int(n)}
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n": int(n)}


def aggregate_estimates(estimates: Sequence[Mapping[str, float]], truth: float) -> dict[str, Any]:
    """Bias, RMSE, spread, interval coverage and width of per-seed estimates against ``truth``.

    ``estimates`` are ``{"mean", "lo", "hi"}`` per seed; a seed whose estimate is NaN (the target
    never agreed with the logged action) is left out and counted in ``n_missing``. Coverage is
    the share of the remaining seeds whose interval holds ``truth`` (ends included), ``sd`` is the
    sample standard deviation of the estimates across seeds and ``bias_rel`` and ``rmse_rel`` are
    relative to ``truth``. Everything is None when no seed has an estimate.
    """
    valid = [e for e in estimates if math.isfinite(e["mean"])]
    out: dict[str, Any] = {
        "n": len(valid),
        "n_missing": len(estimates) - len(valid),
        "mean_estimate": None,
        "bias": None,
        "bias_rel": None,
        "rmse": None,
        "rmse_rel": None,
        "sd": None,
        "coverage": None,
        "n_ci": 0,
        "ci_width": None,
    }
    if not valid:
        return out
    means = np.array([e["mean"] for e in valid], dtype=float)
    errors = means - truth
    out["mean_estimate"] = float(means.mean())
    out["bias"] = float(errors.mean())
    out["rmse"] = float(math.sqrt(np.mean(errors**2)))
    out["sd"] = float(means.std(ddof=1)) if means.size > 1 else None
    if truth != 0:
        out["bias_rel"], out["rmse_rel"] = out["bias"] / truth, out["rmse"] / truth
    with_ci = [e for e in valid if math.isfinite(e["lo"]) and math.isfinite(e["hi"])]
    out["n_ci"] = len(with_ci)
    if with_ci:
        out["coverage"] = float(np.mean([e["lo"] <= truth <= e["hi"] for e in with_ci]))
        out["ci_width"] = float(np.mean([e["hi"] - e["lo"] for e in with_ci]))
    return out


# --- support --------------------------------------------------------------------------------------


def read_logging_support(path: str | Path, ood_action: str) -> pd.DataFrame:
    """What the logging policy could have done on each item, from its replay output.

    One row per item with the logged ``model``, ``logged_error``, ``safe`` (the models the
    policy scored as feasible: its exploration set) and ``eligible``: whether the decision could
    explore. A decision explores only when it relaxed nothing and was not sent to the strongest
    model by an out-of-distribution verdict (``ood_action == "strongest"``); the logging epsilon
    is positive. An eligible decision gives every model of ``safe`` a positive probability, any
    other decision only the model it logged.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            model = rec.get("model")
            logged_error = model is None or bool(rec.get("error"))
            ranked = bool(rec.get("ood")) and ood_action == "strongest"
            safe = tuple(c["model"] for c in (rec.get("candidates") or []) if c.get("feasible"))
            rows.append(
                {
                    "item_id": rec["item_id"],
                    "model": model,
                    "logged_error": logged_error,
                    "eligible": not logged_error and not rec.get("relaxed") and not ranked,
                    "safe": safe,
                }
            )
    return pd.DataFrame(rows, columns=["item_id", "model", "logged_error", "eligible", "safe"])


def support_coverage(support: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    """The share of items where the target's action had positive logging propensity.

    ``support`` is ``read_logging_support`` and ``target`` the target policy's records
    (``item_id``, ``model``, ``error``). The items counted (``n``) are those with a logged action
    and a target action; ``logged_errors`` and ``target_errors`` count the items left out for a
    routing error. ``logged_outside_safe`` counts decisions that could explore whose logged model
    is not in the safe set, which would mean the support rule does not describe the logging
    policy (0 is expected).
    """
    tgt = target[["item_id", "model", "error"]].rename(
        columns={"model": "target_model", "error": "target_error"}
    )
    rows = support.merge(tgt, on="item_id", how="inner", validate="1:1")
    logged_errors = int(rows["logged_error"].sum())
    logged = rows.loc[~rows["logged_error"]]
    failed = logged["target_model"].isna() | logged["target_error"].fillna("").ne("")
    usable = logged.loc[~failed]
    in_safe = [t in s for t, s in zip(usable["target_model"], usable["safe"], strict=True)]
    positive = (usable["target_model"] == usable["model"]) | (
        usable["eligible"].to_numpy(dtype=bool) & np.array(in_safe, dtype=bool)
    )
    outside = [
        bool(e) and m not in s
        for e, m, s in zip(logged["eligible"], logged["model"], logged["safe"], strict=True)
    ]
    n, covered = len(usable), int(positive.sum())
    return {
        "n": n,
        "covered": covered,
        "share": covered / n if n else None,
        "logged_errors": logged_errors,
        "target_errors": int(failed.sum()),
        "logged_outside_safe": int(sum(outside)),
    }


# --- the experiment -------------------------------------------------------------------------------


def _cell(estimate: Mapping[str, float]) -> dict[str, float]:
    return {"mean": estimate["mean"], "lo": estimate["lo"], "hi": estimate["hi"]}


def _seed_row(
    seed: int,
    logged: pd.DataFrame,
    target: pd.DataFrame,
    bootstrap: int,
) -> dict[str, Any]:
    """IPS, SNIPS and DR of one target from one seed's log (ope.py's estimators)."""
    prep = ope._prepare(logged, target)
    summary = ope._summarize(prep, bootstrap, seed)
    cost_usd = prep.cost / PICO
    return {
        "seed": seed,
        "success": {
            "ips": _cell(summary["success"]["ips"]),
            "snips": _cell(
                snips_estimate(
                    prep.success, prep.a_logged, prep.props, prep.a_target, bootstrap, seed
                )
            ),
            "dr": _cell(summary["success"]["dr"]),
        },
        "cost_usd": {
            "ips": _cell(summary["cost_usd"]["ips"]),
            "snips": _cell(
                snips_estimate(cost_usd, prep.a_logged, prep.props, prep.a_target, bootstrap, seed)
            ),
            "dr": _cell(summary["cost_usd"]["dr"]),
        },
        "ess": summary["effective_sample_size"],
        "matches": summary["matches"],
        "n_used": summary["n_used"],
        "n_logged": summary["n_logged"],
        "excluded": sum(summary["excluded"].values()),
    }


def _target_block(
    description: str,
    truth: Mapping[str, float],
    target_errors: int,
    rows: Sequence[Mapping[str, Any]],
    supports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shares = [s["share"] for s in supports if s["share"] is not None]
    ess = [r["ess"] for r in rows]
    matches = [r["matches"] for r in rows]
    return {
        "description": description,
        "truth": dict(truth),
        "target_routing_errors": target_errors,
        "n_used_min": min(r["n_used"] for r in rows),
        "n_logged": rows[0]["n_logged"],
        "excluded_total": sum(r["excluded"] for r in rows),
        "support": {
            **supports[0],
            "share_min": min(shares) if shares else None,
            "share_max": max(shares) if shares else None,
        },
        "ess": {"mean": float(np.mean(ess)), "min": float(np.min(ess)), "max": float(np.max(ess))},
        "matches_mean": float(np.mean(matches)),
        "matches_min": int(np.min(matches)),
        "matches_max": int(np.max(matches)),
        "estimators": {
            metric: {
                est: aggregate_estimates([r[metric][est] for r in rows], truth[metric])
                for est in ESTIMATORS
            }
            for metric in METRICS
        },
        "per_seed": list(rows),
    }


def run_ope_validation(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    seeds: Sequence[int] = SEEDS,
    epsilon: float = EPSILON,
    floor: float = FLOOR,
    bootstrap: int = BOOTSTRAP,
    workers: int = sweeps.WORKERS,
) -> Path:
    """Run the validation on the test split and write the results under ``out_root/<run-id>``.

    One replay per target (the truth) and one logging replay per seed, all on the test split.
    Raises ValueError when the policy config has no lambda or no ``ood_action``.
    """
    started = datetime.now(UTC)
    clock = time.perf_counter()
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    seeds = list(seeds)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    for key in ("lambda", "ood_action"):
        if key not in base_config:
            raise ValueError(f"{policy_config} has no {key}")
    lam, ood_action = float(base_config["lambda"]), str(base_config["ood_action"])
    binary = binary or replaytool.build_go(ROOT)
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    specs = target_specs(pool, learned, floor)
    descriptions = {
        "always_strongest": f"fixed {pool['best_single']} (Best Single)",
        "always_cheapest": f"fixed {pool['cheapest']}",
        "static_rules": "the pool config's keyword and length rules, mapped on train+calib",
        "learned_min_cost": f"learned min_cost at floor {floor}, no exploration (Ruling V1-R36)",
    }

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    phase: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        target_files = {name: scratch / f"target-{name}.jsonl" for name, _, _ in specs}
        log_files = {seed: scratch / f"logged-{seed}.jsonl" for seed in seeds}
        jobs: list[tuple[Path, str, list[str]]] = [
            (target_files[name], policy, flags) for name, policy, flags in specs
        ]
        jobs += [
            (log_files[seed], "learned", [*learned, *logging_flags(lam, epsilon, seed)])
            for seed in seeds
        ]

        def replay(job: tuple[Path, str, list[str]]) -> pd.DataFrame:
            out, policy, flags = job
            return replaytool.replay(binary, config, replay_input, TEST, policy, out, flags)

        tick = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_executor:
            frames = list(pool_executor.map(replay, jobs))
        phase["replays"] = time.perf_counter() - tick
        say(f"{len(jobs)} replays done in {phase['replays']:.1f} s")

        target_frames = dict(zip(target_files, frames[: len(specs)], strict=True))
        truth = {}
        for name, frame in target_frames.items():
            summary = metrics.summarize(frame)
            truth[name] = {
                "success": summary["success_rate"],
                "cost_usd": summary["mean_cost_usd"],
                "errors": summary["errors"],
            }
        n_items = len(next(iter(target_frames.values())))
        policy_version = str(frames[len(specs)]["policy_version"].iloc[0])
        targets_loaded = {name: ope.load_records(path) for name, path in target_files.items()}

        tick = time.perf_counter()
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in target_files}
        supports: dict[str, list[dict[str, Any]]] = {name: [] for name in target_files}
        for k, seed in enumerate(seeds):
            logged = ope.load_records(log_files[seed])
            support = read_logging_support(log_files[seed], ood_action)
            for name, target in targets_loaded.items():
                rows[name].append(_seed_row(seed, logged, target, bootstrap))
                supports[name].append(support_coverage(support, target))
            log_files[seed].unlink()
            if (k + 1) % 10 == 0:
                say(f"estimated {k + 1} of {len(seeds)} seeds")
        phase["estimation"] = time.perf_counter() - tick

    targets = {
        name: _target_block(
            descriptions[name],
            {"success": truth[name]["success"], "cost_usd": truth[name]["cost_usd"]},
            truth[name]["errors"],
            rows[name],
            supports[name],
        )
        for name in target_files
    }
    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "logging_policy_version": policy_version,
        "split": TEST,
        "n_items": n_items,
        "logging": {
            "mode": "dual",
            "lambda": lam,
            "epsilon": epsilon,
            "ood_action": ood_action,
            "seeds": seeds,
        },
        "floor": floor,
        "bootstrap": {
            "n": bootstrap,
            "seed": "the logging seed",
            "alpha": 0.05,
            "method": "percentile",
        },
        "targets": targets,
        "wall_time_s": time.perf_counter() - clock,
        "phase_time_s": phase,
        "labels": _labels(bootstrap, floor),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started)
    shown = (
        list(command)
        if command
        else ["python", "bench/scripts/ope_validation.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {"estimator_error": _figure(results)},
        {"logging": seeds, "bootstrap": "the logging seed"},
        shown,
        artifact.name,
        started,
    )
    return out_dir


# --- report --------------------------------------------------------------------------------------


def _labels(bootstrap: int, floor: float) -> list[str]:
    return [
        "no latency is measured or simulated in this experiment",
        "costs are the recorded costs of each model on each item",
        "the true value of a target policy is its own deterministic full-information replay on "
        "the same test items the estimators see (success rate and mean recorded cost)",
        f"each interval is a 95% percentile bootstrap with {bootstrap} bootstrap resamples of the "
        "items, seeded by the logging seed; the SNIPS interval resamples the weighted ratio",
        "the logging seed changes only the exploration draws: the test items are fixed, so the "
        "spread across seeds is exploration noise while each interval also resamples the items "
        "and is wider than that spread; coverage is measured against the in-sample true value",
        f"Ruling V1-R36: the learned target is min_cost at floor {floor}; the plan's 0.8 relaxes "
        "nearly every flagship decision to quality",
        "support coverage is the share of items where the target's action had positive logging "
        "propensity; epsilon exploration covers only the safe set and a decision that relaxed a "
        "constraint does not explore, so IPS cannot estimate an action outside it",
        "DR uses one outcome model, the logging policy's own predictions (Ruling V1-R32), so its "
        "bias includes that model's calibration error",
    ]


def _estimate_table(results: Mapping[str, Any], metric: str, digits: str) -> pd.DataFrame:
    rows = []
    for name, target in results["targets"].items():
        for est in ESTIMATORS:
            a = target["estimators"][metric][est]
            rows.append(
                {
                    "target": name,
                    "estimator": est.upper(),
                    "true value": target["truth"][metric],
                    "mean estimate": a["mean_estimate"],
                    "bias": a["bias"],
                    "bias / true value": a["bias_rel"],
                    "RMSE": a["rmse"],
                    "RMSE / true value": a["rmse_rel"],
                    "SD across seeds": a["sd"],
                    "CI coverage": None if a["coverage"] is None else format(a["coverage"], digits),
                    "mean CI width": a["ci_width"],
                    "seeds with an estimate": a["n"],
                }
            )
    return pd.DataFrame(rows)


def _targets_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, t in results["targets"].items():
        sup = t["support"]
        rows.append(
            {
                "target": name,
                "policy": t["description"],
                "true success": t["truth"]["success"],
                "true cost USD per item": t["truth"]["cost_usd"],
                "support coverage": sup["share"],
                "logged action = target action (mean items)": t["matches_mean"],
                "ESS of the IPS weights, mean [min, max]": (
                    f"{t['ess']['mean']:.1f} [{t['ess']['min']:.1f}, {t['ess']['max']:.1f}]"
                ),
                "items used (min over seeds)": t["n_used_min"],
                "items left out (all seeds)": t["excluded_total"],
                "target routing errors": t["target_routing_errors"],
            }
        )
    return pd.DataFrame(rows)


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    log, boot = results["logging"], results["bootstrap"]
    seeds = log["seeds"]
    rows: list[list[Any]] = [
        ["pool", results["pool"]],
        ["artifact version", results["artifact_version"]],
        ["logging policy version (last seed's replay)", results["logging_policy_version"]],
        ["test items", results["n_items"]],
        [
            "logging policy",
            f"learned {log['mode']}, lambda {log['lambda']!r}, epsilon {log['epsilon']}",
        ],
        ["OOD action of the logging policy", log["ood_action"]],
        ["logging seeds", f"{len(seeds)} ({seeds[0]} to {seeds[-1]})"],
        ["learned target floor (Ruling V1-R36)", results["floor"]],
        ["bootstrap", f"{boot['n']} bootstrap resamples per estimate, percentile 95% intervals"],
        ["wall time (s)", round(results["wall_time_s"], 1)],
        [
            "replays (s) / estimation (s)",
            f"{results['phase_time_s']['replays']:.1f} / "
            f"{results['phase_time_s']['estimation']:.1f}",
        ],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    n_seeds = len(results["logging"]["seeds"])
    return {
        "Setup": _setup_table(results),
        "Targets: true value, support coverage and importance weights": _targets_table(results),
        f"Success: bias, RMSE and CI coverage over {n_seeds} logging seeds": _estimate_table(
            results, "success", ".3f"
        ),
        f"Cost (USD per item): bias, RMSE and CI coverage over {n_seeds} logging seeds": (
            _estimate_table(results, "cost_usd", ".3f")
        ),
    }


def _figure(results: Mapping[str, Any]) -> Figure:
    targets = list(results["targets"])
    colors = {"ips": "tab:blue", "snips": "tab:orange", "dr": "tab:green"}
    panels = (
        ("success", "bias", "bias of the success rate (points)", 100.0),
        ("success", "rmse", "RMSE of the success rate (points)", 100.0),
        ("cost_usd", "bias_rel", "bias of the mean cost (% of the true value)", 100.0),
        ("cost_usd", "rmse_rel", "RMSE of the mean cost (% of the true value)", 100.0),
    )
    fig = Figure(figsize=(11.0, 7.0))
    axes = fig.subplots(2, 2)
    width = 0.26
    for ax, (metric, key, title, scale) in zip(axes.ravel(), panels, strict=True):
        for j, est in enumerate(ESTIMATORS):
            values = [
                (results["targets"][t]["estimators"][metric][est][key] or 0.0) * scale
                for t in targets
            ]
            ax.bar(
                np.arange(len(targets)) + (j - 1) * width,
                values,
                width,
                color=colors[est],
                label=est.upper(),
            )
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_xticks(np.arange(len(targets)), targets, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    n_seeds = len(results["logging"]["seeds"])
    fig.suptitle(
        f"Off-policy estimators against the true replay value, {n_seeds} logging seeds, "
        f"{results['pool']} pool, test split",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the off-policy estimators.")
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
    out_dir = run_ope_validation(
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
