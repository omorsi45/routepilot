"""Contextual bandits (LinUCB, linear Thompson sampling) against the supervised learned policy.

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/bandits.py --pool flagship

Arms are the pool models. The context of an item is the same for every arm: a 128-component
``TruncatedSVD`` (``random_state=0``, fitted on the train rows) of the hashed features, the 12
dense features standardized with the train mean and standard deviation, and a constant 1 (141
numbers, ``routepilot_train.bandits``). The reward of an arm on an item is
``success - lambda * cost_usd`` with the recorded cost of that model on that item and ``lambda``
from the policy config (``bench/policies/flagship.yaml``, the supervised policy's own lambda).

Fitting phase (train and calibration rows only; the test split is not read):

1. The context map is fitted on the train rows.
2. LinUCB's ``alpha`` (0.1, 0.5, 1.0, 2.0) and Thompson sampling's ``v`` (0.05, 0.1, 0.5, 1.0)
   are chosen by mean reward: warm up on the train rows (seeded random order, bandit feedback),
   then evaluate on the calibration rows (random order, seed 0, still learning).

Test phase (seeds 0 to 9): each learner, with its chosen value, warms up on the train and
calibration rows in a seeded random order (bandit feedback, not evaluated) and then plays the
test rows in two orders, from copies of the same warm state: ``random`` (a seeded shuffle) and
``shift`` (sorted by domain, math, code, knowledge, reasoning, writing, then by ``item_id``; the
learner keeps updating during the stream). They are compared, on the same items and orders, with
the supervised learned policy at the spec operating point (a Go ``routepilot replay`` of
``--policy learned`` with the policy config's ``dual`` lambda), always_strongest (Best Single, a
Go replay), the oracle by reward and, as an extra reference, the best single arm by mean
reward on train and calibration. Per learner and order: mean reward per item, success rate,
mean cost, cumulative regret against the per-item best arm by reward, mean and range across
seeds, and cumulative reward curves. The report states plainly whether either bandit beats the
supervised policy on the test split.

Writes bench/results/bandits/<run-id>/{results.json,report.md,curves_random.png,
curves_shift.png,run.json}. Output holds numbers, model names and item IDs only.
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

from routepilot_train import bandits, replaytool, runinfo
from routepilot_train.data import selection
from routepilot_train.data.splits import require_splits
from routepilot_train.evaluate import metrics, report
from routepilot_train.export import artifact as artifact_io

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeps  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "bandits"
TEST = "test"
BOOTSTRAP = 1000
ORDERS = ("random", "shift")
LEARNERS = {
    "linucb": ("LinUCB", "alpha", bandits.ALPHAS, bandits.linucb_factory),
    "linear_ts": ("LinearTS", "v", bandits.TS_SCALES, bandits.linear_ts_factory),
}
METRICS = ("mean_reward", "success_rate", "mean_cost_usd", "cumulative_regret")
COMPARATORS = (
    ("supervised", "supervised learned policy (dual, spec operating point)"),
    ("always_strongest", "always_strongest (Best Single)"),
    ("best_constant", "best single arm by reward on train+calib (extra reference)"),
    ("oracle", "oracle by reward (upper bound)"),
)
COLORS = {
    "linucb": "tab:blue",
    "linear_ts": "tab:green",
    "supervised": "tab:red",
    "always_strongest": "0.4",
    "best_constant": "tab:purple",
    "oracle": "tab:orange",
}


# --- comparators ------------------------------------------------------------------------------


def policy_arms(frame: pd.DataFrame, block: bandits.Block, models: Sequence[str]) -> np.ndarray:
    """The arm a replayed policy chose for each row of ``block`` (-1 for a routing error).

    ``frame`` is a Go replay of the same items. A chosen model's recorded success and cost in the
    replay must equal the block's tables; otherwise the tables and the Go replay disagree and
    ``ValueError`` says so.
    """
    require_splits(frame, {TEST})
    index = {model: i for i, model in enumerate(models)}
    by_id = frame.set_index("item_id")
    if set(by_id.index) != set(block.items["item_id"]) or len(by_id) != len(block.items):
        raise ValueError("the replay and the bandit rows hold different items")
    arms = np.full(len(block.items), -1, dtype=np.int64)
    for row, item_id in enumerate(block.items["item_id"]):
        record = by_id.loc[item_id]
        model = record["model"]
        if not isinstance(model, str) or record["error"] != "":
            continue
        arm = index[model]
        if bool(record["success"]) != bool(block.success[row, arm]) or int(
            record["cost_picousd"]
        ) != round(block.cost_usd[row, arm] * bandits.PICO):
            raise ValueError(
                f"item {item_id}: the replay's outcome for {model} differs from the recorded one"
            )
        arms[row] = arm
    return arms


def item_rewards(block: bandits.Block, arms: np.ndarray) -> np.ndarray:
    """The reward each row earned when ``arms[row]`` was played (0 for a routing error)."""
    played = arms >= 0
    return np.where(played, block.reward[np.arange(len(arms)), np.where(played, arms, 0)], 0.0)


def comparator_arms(
    block: bandits.Block, warm: bandits.Block, supervised: np.ndarray, strongest: np.ndarray
) -> dict[str, np.ndarray]:
    """The arm each comparator plays on every test row."""
    constant = bandits.best_constant_arm(warm.reward)
    return {
        "supervised": supervised,
        "always_strongest": strongest,
        "best_constant": np.full(len(block.items), constant, dtype=np.int64),
        "oracle": bandits.oracle_arms(block.reward),
    }


def standardizer_check(context: bandits.Context, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """How far the context's dense mean and std are from the artifact's (same train rows)."""
    mean = float(np.max(np.abs(context.dense_mean - np.asarray(manifest["dense_mean"]))))
    std = float(np.max(np.abs(context.dense_std - np.asarray(manifest["dense_std"]))))
    return {
        "max_abs_diff_mean": mean,
        "max_abs_diff_std": std,
        "matches_artifact": mean + std < 1e-9,
    }


# --- statements -------------------------------------------------------------------------------


def verdict(above: int, n_seeds: int) -> str:
    """Plain words for how many seeds a bandit's mean reward is above the reference."""
    if above == n_seeds:
        return "beats"
    if above == 0:
        return "does NOT beat"
    return "does NOT consistently beat"


def statement(
    label: str,
    order: str,
    reward: Mapping[str, float],
    reference: float,
    above: int,
    n_seeds: int,
    reference_name: str,
    paired: tuple[float, float, float],
) -> str:
    """One sentence on a learner's mean reward per item against a reference.

    ``paired`` is the seed-averaged paired difference per test item (bandit minus reference)
    with its 95% bootstrap interval, as ``(point, lo, hi)``.
    """
    point, lo, hi = paired
    return (
        f"{label}, {order} order: {verdict(above, n_seeds)} {reference_name} "
        f"(mean reward per item {reward['mean']:.4f}, seeds {reward['min']:.4f} to "
        f"{reward['max']:.4f}, against {reference:.4f}; above it on {above} of {n_seeds} seeds; "
        f"paired difference {point:+.4f} [{lo:+.4f}, {hi:+.4f}])."
    )


# --- the experiment ---------------------------------------------------------------------------


def run_bandits(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    seeds: Sequence[int] = bandits.SEEDS,
    bootstrap: int = BOOTSTRAP,
    dim: int | None = None,
    n_components: int = bandits.SVD_COMPONENTS,
    domains: Mapping[str, str] | None = None,
) -> Path:
    """Run both phases for ``pool_name`` and write the results under ``out_root/<run-id>``.

    ``dim`` is the hashed feature dimension (default: the artifact's). The test rows are read
    after every hyperparameter is chosen, and the Go replays of the comparators are the only
    other reads of the test split.
    """
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    models, strong = list(pool["models"]), pool["best_single"]
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    policy = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    lam = float(policy["lambda"])
    manifest, _ = artifact_io.load_artifact(artifact)
    dim = int(dim if dim is not None else manifest["dim"])
    binary = binary or replaytool.build_go(ROOT)
    n_arms = len(models)
    seeds = list(seeds)
    phase: dict[str, float] = {}

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    # ---- fitting phase: train and calibration rows only
    tick = time.perf_counter()
    fit_arms = bandits.read_arms(replay_input, {"train", "calib"}, models)
    train_arms, calib_arms = fit_arms.of_split("train"), fit_arms.of_split("calib")
    context = bandits.fit_context(train_arms.items, dim, n_components)
    train = bandits.make_block(train_arms, context, lam)
    calib = bandits.make_block(calib_arms, context, lam)
    warm = bandits.concat_blocks(train, calib)
    check = standardizer_check(context, manifest)
    say(
        f"context {context.size} numbers; standardizer matches the artifact: "
        f"{check['matches_artifact']}"
    )
    chosen: dict[str, float] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for key, (label, param, grid, make) in LEARNERS.items():
        chosen[key], tables[key] = bandits.select_hyperparameter(
            make(n_arms, context.size), grid, train, calib, seed=0
        )
        say(f"{label}: {param} = {chosen[key]} chosen on calibration")
    phase["fitting"] = time.perf_counter() - tick

    # ---- test phase: the test rows are read here, after every choice
    tick = time.perf_counter()
    test_arms = bandits.read_arms(replay_input, {TEST}, models)
    test = bandits.make_block(test_arms, context, lam)
    n_test = len(test.items)
    shift = bandits.shift_order(test.items, domains)
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        k: {o: [] for o in ORDERS} for k in LEARNERS
    }
    for key, (label, _, _, make) in LEARNERS.items():
        for seed in seeds:
            orders = {"random": bandits.random_order(n_test, seed), "shift": shift}
            result = bandits.run_seed(
                make(n_arms, context.size), chosen[key], warm, test, orders, seed
            )
            for order in ORDERS:
                runs[key][order].append({**result[order], "order": orders[order]})
        say(f"{label}: {len(seeds)} seeds done")
    phase["bandit streams"] = time.perf_counter() - tick

    tick = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        learned_flags = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
        learned_flags += sweeps.dual_flags(lam)
        sup_frame = replaytool.replay(
            binary, config, replay_input, TEST, "learned", scratch / "learned.jsonl", learned_flags
        )
        strong_frame = replaytool.replay(
            binary,
            config,
            replay_input,
            TEST,
            "fixed",
            scratch / "fixed.jsonl",
            ["--model", strong],
        )
    arms_of = comparator_arms(
        test,
        warm,
        policy_arms(sup_frame, test, models),
        policy_arms(strong_frame, test, models),
    )
    phase["comparator replays"] = time.perf_counter() - tick

    # ---- summaries, per order
    comparator_item_reward = {k: item_rewards(test, a) for k, a in arms_of.items()}
    comparators: dict[str, dict[str, Any]] = {k: {} for k in arms_of}
    curves: dict[str, dict[str, Any]] = {o: {} for o in ORDERS}
    for order in ORDERS:
        for name, arms in arms_of.items():
            per_order = [
                bandits.summarize_choices(
                    test.reward, test.success, test.cost_usd, run["order"], arms[run["order"]]
                )
                for run in runs["linucb"][order]
            ]
            comparators[name][order] = {
                **{m: per_order[0][m] for m in METRICS},
                "arm_share": per_order[0]["arm_share"].tolist(),
                "n": per_order[0]["n"],
            }
            curves[order][name] = np.mean([s["reward_curve"] for s in per_order], axis=0)
        for key in LEARNERS:
            curves[order][key] = {
                "mean": np.mean([r["reward_curve"] for r in runs[key][order]], axis=0),
                "min": np.min([r["reward_curve"] for r in runs[key][order]], axis=0),
                "max": np.max([r["reward_curve"] for r in runs[key][order]], axis=0),
            }

    learners: dict[str, Any] = {}
    statements: list[str] = []
    reference_statements: list[str] = []
    for key, (label, param, _, _) in LEARNERS.items():
        learners[key] = {"label": label, "param_name": param, "param": chosen[key], "orders": {}}
        for order in ORDERS:
            per_seed = runs[key][order]
            entry: dict[str, Any] = {m: bandits.spread([r[m] for r in per_seed]) for m in METRICS}
            entry["per_seed"] = {m: [r[m] for r in per_seed] for m in METRICS}
            entry["arm_share"] = np.mean([r["arm_share"] for r in per_seed], axis=0).tolist()
            # Paired by item: the seed-averaged reward of each test item against the reference
            # policy's reward on it.
            seed_mean = np.mean(
                [item_rewards(test, _item_arms(r["order"], r["arms"])) for r in per_seed], axis=0
            )
            title = f"{label} ({param} = {chosen[key]})"
            for field, name, noun, bucket in (
                ("vs_supervised", "supervised", "the supervised learned policy", statements),
                (
                    "vs_best_constant",
                    "best_constant",
                    "the best single arm by reward (extra reference)",
                    reference_statements,
                ),
            ):
                reference = comparators[name][order]["mean_reward"]
                above = sum(r["mean_reward"] > reference for r in per_seed)
                paired = metrics.bootstrap_ci(
                    seed_mean - comparator_item_reward[name], n=bootstrap, seed=0
                )
                entry[field] = {
                    "seeds_above": int(above),
                    "seeds": len(per_seed),
                    "mean_reward_difference": paired[0],
                    "mean_reward_difference_ci": [paired[1], paired[2]],
                }
                bucket.append(
                    statement(
                        title,
                        order,
                        entry["mean_reward"],
                        reference,
                        above,
                        len(per_seed),
                        noun,
                        paired,
                    )
                )
            learners[key]["orders"][order] = entry
    domain_of = selection.DOMAINS if domains is None else domains
    per_domain = test.items["dataset"].map(domain_of).value_counts()
    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "models": models,
        "best_single": strong,
        "lambda": lam,
        "n_items": {
            "train": len(train.items),
            "calib": len(calib.items),
            "test": n_test,
        },
        "seeds": seeds,
        "context": {
            "size": context.size,
            "svd_components": context.svd.n_components,
            "svd_explained_variance": float(context.svd.explained_variance_ratio_.sum()),
            "dense_features": 12,
            "hash_dim": dim,
            "standardizer": check,
        },
        "shift": {
            "domain_order": list(bandits.DOMAIN_ORDER),
            "items_per_domain": {str(k): int(v) for k, v in per_domain.items()},
        },
        "hyperparameters": {
            key: {
                "param_name": LEARNERS[key][1],
                "grid": list(LEARNERS[key][2]),
                "chosen": chosen[key],
                "calibration_table": tables[key],
            }
            for key in LEARNERS
        },
        "learners": learners,
        "comparators": comparators,
        "statements": statements,
        "reference_statements": reference_statements,
        "bootstrap": {"n": bootstrap, "seed": 0, "alpha": 0.05, "method": "percentile"},
        "wall_time_s": time.perf_counter() - clock,
        "phase_time_s": phase,
        "labels": _labels(lam),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    shown = (
        list(command) if command else ["python", "bench/scripts/bandits.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {f"curves_{order}": _figure(order, results, curves[order]) for order in ORDERS},
        {"bandit_seeds": seeds, "hyperparameter_seed": 0, "svd": 0, "bootstrap": 0},
        shown,
        artifact.name,
        started,
        git=start_git,
    )
    return out_dir


def _item_arms(order: np.ndarray, arms_in_order: np.ndarray) -> np.ndarray:
    """Per-row arms from the arms played in ``order``."""
    by_row = np.empty(len(order), dtype=np.int64)
    by_row[order] = arms_in_order
    return by_row


# --- report -----------------------------------------------------------------------------------


def _labels(lam: float) -> list[str]:
    return [
        "costs are the recorded costs of each model on each item; no latency is measured or "
        "simulated in this experiment",
        f"reward = success - lambda * cost_usd with lambda = {lam!r} per USD, the supervised "
        "policy's own lambda from the policy config",
        "the bandits see only the reward of the arm they play, while the supervised policy was "
        "fitted on the train split with the outcomes of all models on every item",
        "alpha and v are chosen on the calibration split (warm-up on train, random order, "
        "seed 0); the test split is read after, once per learner, order and seed",
        "the supervised policy's calibrators were fitted on the calibration split, and its "
        "lambda was tuned there, before the test split was replayed",
        "oracle by reward looks at the recorded outcomes: an upper bound, not a policy; the "
        "best single arm is an extra reference chosen on train and calibration rewards",
    ]


def _range_text(s: Mapping[str, float], digits: str = ".4f") -> str:
    return f"{s['mean']:{digits}} [{s['min']:{digits}}, {s['max']:{digits}}]"


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    ctx, items = results["context"], results["n_items"]
    check = ctx["standardizer"]
    rows = [
        ["pool", results["pool"]],
        ["supervised artifact", results["artifact_version"]],
        ["arms (models)", len(results["models"])],
        [
            "items (train / calibration / test)",
            f"{items['train']} / {items['calib']} / {items['test']}",
        ],
        ["reward lambda (USD^-1)", repr(results["lambda"])],
        [
            "context",
            f"{ctx['size']} = {ctx['svd_components']} SVD components of the hashed features "
            f"(dim {ctx['hash_dim']}) + {ctx['dense_features']} standardized dense + 1",
        ],
        ["variance of the hashed features kept by the SVD", ctx["svd_explained_variance"]],
        [
            "dense standardizer equals the artifact's",
            f"{check['matches_artifact']} (largest differences {check['max_abs_diff_mean']:.3g}, "
            f"{check['max_abs_diff_std']:.3g})",
        ],
        ["seeds", f"{results['seeds'][0]} to {results['seeds'][-1]} ({len(results['seeds'])})"],
        ["items per domain in the test split", json.dumps(results["shift"]["items_per_domain"])],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _hyperparameter_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for key, h in results["hyperparameters"].items():
        for row in h["calibration_table"]:
            rows.append(
                {
                    "learner": results["learners"][key]["label"],
                    "parameter": h["param_name"],
                    "value": row["value"],
                    "calib mean reward": row["mean_reward"],
                    "calib success": row["success_rate"],
                    "calib mean cost USD": row["mean_cost_usd"],
                    "chosen": "yes" if row["value"] == h["chosen"] else "",
                }
            )
    return pd.DataFrame(rows)


def _order_table(results: Mapping[str, Any], order: str) -> pd.DataFrame:
    rows = []
    for entry in results["learners"].values():
        o = entry["orders"][order]
        rows.append(
            {
                "policy": f"{entry['label']} ({entry['param_name']} = {entry['param']})",
                "mean reward per item [min, max over seeds]": _range_text(o["mean_reward"]),
                "success rate": _range_text(o["success_rate"]),
                "mean cost USD": _range_text(o["mean_cost_usd"], ".5f"),
                "cumulative regret (total over the stream)": _range_text(
                    o["cumulative_regret"], ".1f"
                ),
                "seeds above the supervised policy": (
                    f"{o['vs_supervised']['seeds_above']} of {o['vs_supervised']['seeds']}"
                ),
            }
        )
    for name, description in COMPARATORS:
        c = results["comparators"][name][order]
        rows.append(
            {
                "policy": description,
                "mean reward per item [min, max over seeds]": f"{c['mean_reward']:.4f}",
                "success rate": f"{c['success_rate']:.4f}",
                "mean cost USD": f"{c['mean_cost_usd']:.5f}",
                "cumulative regret (total over the stream)": f"{c['cumulative_regret']:.1f}",
                "seeds above the supervised policy": "",
            }
        )
    return pd.DataFrame(rows)


def _paired_table(results: Mapping[str, Any]) -> pd.DataFrame:
    def cell(v: Mapping[str, Any]) -> str:
        lo, hi = v["mean_reward_difference_ci"]
        return f"{v['mean_reward_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    rows = []
    for entry in results["learners"].values():
        for order in ORDERS:
            o = entry["orders"][order]
            sup, best = o["vs_supervised"], o["vs_best_constant"]
            rows.append(
                {
                    "learner": entry["label"],
                    "order": order,
                    "minus the supervised policy [95% CI]": cell(sup),
                    "seeds above the supervised policy": f"{sup['seeds_above']} of {sup['seeds']}",
                    "minus the best single arm [95% CI]": cell(best),
                    "seeds above the best single arm": f"{best['seeds_above']} of {best['seeds']}",
                }
            )
    return pd.DataFrame(rows)


def _usage_table(results: Mapping[str, Any], order: str) -> pd.DataFrame:
    models = results["models"]
    rows = []
    for entry in results["learners"].values():
        share = entry["orders"][order]["arm_share"]
        rows.append({"policy": entry["label"], **dict(zip(models, share, strict=True))})
    for name, description in COMPARATORS:
        share = results["comparators"][name][order]["arm_share"]
        rows.append({"policy": description, **dict(zip(models, share, strict=True))})
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "Do the bandits beat the supervised policy? (test split, mean reward per item)": (
            pd.DataFrame({"statement": results["statements"]})
        ),
        "Against the best single arm by reward, an extra reference (test split)": (
            pd.DataFrame({"statement": results["reference_statements"]})
        ),
        "Setup": _setup_table(results),
        "Hyperparameter choice (warm-up on train, evaluation on calibration, random order, "
        "seed 0)": _hyperparameter_table(results),
        "Test split, random order (10 seeds: mean [min, max])": _order_table(results, "random"),
        "Test split, shift order (domains in sequence; 10 seeds: mean [min, max])": _order_table(
            results, "shift"
        ),
        "Paired against the supervised policy (seed-averaged reward of each test item)": (
            _paired_table(results)
        ),
        "Share of steps each model was played, random order (mean over seeds)": _usage_table(
            results, "random"
        ),
        "Share of steps each model was played, shift order (mean over seeds)": _usage_table(
            results, "shift"
        ),
    }


def _figure(order: str, results: Mapping[str, Any], curves: Mapping[str, Any]) -> Figure:
    fig = Figure(figsize=(8.0, 5.2))
    ax = fig.subplots()
    n = len(curves["oracle"])
    steps = np.arange(1, n + 1)
    for key, entry in results["learners"].items():
        c = curves[key]
        ax.plot(
            steps,
            c["mean"],
            color=COLORS[key],
            linewidth=1.4,
            label=f"{entry['label']} ({entry['param_name']} = {entry['param']}), mean of seeds",
        )
        ax.fill_between(steps, c["min"], c["max"], color=COLORS[key], alpha=0.2, linewidth=0)
    for name, description in COMPARATORS:
        ax.plot(
            steps,
            curves[name],
            color=COLORS[name],
            linewidth=1.2,
            linestyle="--" if name in ("oracle", "best_constant") else "-",
            label=description,
        )
    ax.set_xlabel("test items played")
    ax.set_ylabel("cumulative reward (success - lambda x recorded cost)")
    ax.set_title(
        f"Cumulative reward, test split, {order} order ({len(results['seeds'])} seeds, band = "
        "range)\nhyperparameters chosen on calibration",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contextual bandits against the learned policy.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument(
        "--policy-config", type=Path, default=None, help="default: bench/policies/<pool>.yaml"
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "bandits")
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_bandits(
        args.derived,
        args.pool,
        args.artifact or sweeps.find_artifact(ROOT / "policies", args.pool),
        args.policy_config or ROOT / "bench" / "policies" / f"{args.pool}.yaml",
        args.out_root,
        command,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
