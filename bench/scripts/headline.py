"""The headline offline evaluation: the learned policy against every baseline, with paired CIs.

Run from the repository root after ``make data``, ``make train``, the MF export and
``bench/scripts/sweeps.py`` (which tunes lambda into ``bench/policies/flagship.yaml``):

    python bench/scripts/headline.py --pool flagship
    python bench/scripts/headline.py --pool mixed      # sensitivity pool, see below

The mixed pool needs its own artifact and RouteLLM scorer first:

    python -m routepilot_train.train.pipeline --pool mixed --derived data/derived --out policies
    python -m routepilot_train.baselines.routellm_mf --pool mixed --derived data/derived

Everything goes through the Go ``routepilot replay`` command and has two phases.

Calibration phase (the test split is not replayed or read):

1. Best Single's calibration success gives the target ``Q_bar = success - 0.01``.
2. lambda for the learned ``dual`` mode: read from the policy config (``--lambda-source
   config``, the flagship value tuned by ``sweeps.py``; its calibration success is checked
   against ``Q_bar``) or tuned here by bisection (``tune``, used for the mixed pool).
3. ``min_cost`` floor: the one of 0, 0.05, ..., 0.95 whose calibration success is closest to
   ``Q_bar`` (the ``sweeps.py`` rule). RouteLLM MF threshold: the cheapest of 41 calibration
   score quantiles whose calibration success reaches ``Q_bar``.
4. Cross-fitted sensitivity row (flagship): the calibration items are split by the parity of
   ``sha256(item_id + "routepilot-crossfit-v1")`` into halves A and B. The calibrators and
   lower-bound tables are refit on A only (quality and length weights, the length block and
   the OOD detector are the shipped ones; the artifact is exported to a temporary directory
   and never committed), and lambda is tuned on B with the same ``Q_bar`` rule (Best Single's
   success on B minus 0.01).

Test phase: every operating point and every baseline is replayed on the test split once, with
paired bootstrap intervals against always_strongest, a per-dataset table, and the test
calibration quality (ECE with equal-mass and equal-width bins, Brier, log-loss) of the
shipped artifact, recomputed in Python from its exported weights and cross-checked against
the Go predictor on the replayed decisions. The lambda = 0 dual point is a labeled reference,
not an operating point. The report says plainly whether the targets (cost reduction of at
least 25% and success within 1 point of always_strongest) were met; nothing is tuned to meet
them.

Writes bench/results/headline/<run-id>/{results.json,report.md,pareto_test.png,
reliability_test.png,run.json}. Output holds numbers, model names and item IDs only.
"""

import argparse
import hashlib
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

from routepilot_train import replaytool, runinfo
from routepilot_train.baselines import routellm_mf
from routepilot_train.evaluate import calibration, metrics, pareto, report, rescore
from routepilot_train.export import artifact as artifact_io
from routepilot_train.train import calibrate, crossfit, dual

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (sibling scripts, not packages)
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CALIB, TEST = "calib", "test"
EXPERIMENT = "headline"
COST_TARGET = 0.25  # the spec: at least this cost reduction against always_strongest ...
SUCCESS_MARGIN = 0.01  # ... with success no more than this far below always_strongest
BINS = 15
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"
LAMBDA_SOURCES = ("config", "tune")


# --- targets -----------------------------------------------------------------------------------


def target_check(row: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a policy row meets the spec's targets against always_strongest.

    The cost target is a reduction of at least 25%. The success target is "within 1 point":
    the success rate is not more than 0.01 below always_strongest's (a higher rate is within
    it); ``success_abs_within_margin`` also says whether the difference is at most 0.01 either
    way. Both use the point estimates, the intervals are reported next to them.
    """
    cost, delta = row["cost_reduction_vs_strongest"], row["success_delta_vs_strongest"]
    cost_ok = cost is not None and cost >= COST_TARGET
    success_ok = delta >= -SUCCESS_MARGIN
    return {
        "cost_reduction": cost,
        "cost_target": COST_TARGET,
        "cost_target_met": bool(cost_ok),
        "success_delta": delta,
        "success_margin": SUCCESS_MARGIN,
        "success_target_met": bool(success_ok),
        "success_abs_within_margin": bool(abs(delta) <= SUCCESS_MARGIN),
        "both_met": bool(cost_ok and success_ok),
    }


def target_statement(label: str, check: Mapping[str, Any]) -> str:
    """One plain sentence on whether the targets were met."""
    cost = check["cost_reduction"]
    cost_text = "n/a" if cost is None else f"{cost * 100:.1f}%"
    verdict = "targets MET" if check["both_met"] else "targets NOT met"
    return (
        f"{label}: cost reduction {cost_text} "
        f"({'meets' if check['cost_target_met'] else 'misses'} the {COST_TARGET:.0%} target), "
        f"success {check['success_delta'] * 100:+.2f} points against always_strongest "
        f"({'within' if check['success_target_met'] else 'outside'} the 1 point target): "
        f"{verdict}."
    )


# --- calibration quality -------------------------------------------------------------------------


def quality_row(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Count, base rate, ECE (equal-mass and equal-width, 15 bins), Brier and log-loss."""
    return {
        "n": int(len(p)),
        "base_rate": float(np.mean(y)),
        "mean_p": float(np.mean(p)),
        "ece_mass": calibration.ece(p, y, BINS, "mass"),
        "ece_width": calibration.ece(p, y, BINS, "width"),
        "brier": calibration.brier(p, y),
        "log_loss": calibration.log_loss(p, y),
    }


def calibration_quality(p: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Quality of the forecasts ``p[model]`` against outcomes ``y[model]``, per model and pooled.

    Pooled puts every model's forecasts and outcomes together.
    """
    per_model = {m: quality_row(np.asarray(p[m]), np.asarray(y[m])) for m in p}
    pooled = quality_row(
        np.concatenate([np.asarray(p[m]) for m in p]), np.concatenate([np.asarray(y[m]) for m in p])
    )
    return {"per_model": per_model, "pooled": pooled}


def reliability_figure(
    p: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray], quality: Mapping[str, Any], title: str
) -> Figure:
    """One small reliability diagram (15 equal-mass bins) per model, four panels per row."""
    models = list(p)
    cols = 4
    rows = max(1, -(-len(models) // cols))
    fig = Figure(figsize=(3.0 * cols, 2.8 * rows + 0.6))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, model in zip(axes.ravel(), models, strict=False):
        ax.set_visible(True)
        table = calibration.reliability_table(p[model], y[model], BINS, "mass")
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.8)
        ax.plot(table["mean_p"], table["mean_y"], marker="o", markersize=3, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model}\nECE {quality['per_model'][model]['ece_mass']:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.supxlabel("mean predicted success (15 equal-mass bins)", fontsize=8)
    fig.supylabel("observed success rate", fontsize=8)
    fig.tight_layout()
    return fig


def go_parity(
    frame: pd.DataFrame, item_ids: Sequence[str], p: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Largest gap between the Go predictor's ``p`` for the chosen model and the Python one.

    ``frame`` is a learned-policy replay; ``p[model]`` holds the Python forecasts for
    ``item_ids``. Out-of-distribution items are skipped (Go scores them with the pooled rate,
    not the model's forecast) and so are routing errors.
    """
    index = {item_id: i for i, item_id in enumerate(item_ids)}
    usable = frame.loc[~frame["ood"] & frame["model"].notna() & frame["p"].notna()]
    gaps = [
        abs(float(row.p) - float(p[row.model][index[row.item_id]]))
        for row in usable.itertuples(index=False)
    ]
    return {
        "items_compared": len(gaps),
        "items_skipped": int(len(frame) - len(gaps)),
        "max_abs_diff": max(gaps) if gaps else None,
    }


# --- tables -------------------------------------------------------------------------------------


def per_dataset_table(
    frames: Mapping[str, pd.DataFrame], reference: str = baselines.REFERENCE
) -> pd.DataFrame:
    """Success and mean cost per dataset for each policy, with the change against ``reference``.

    ``frames`` maps a policy label to its replay frame (the same items in each).
    """
    ref = frames[reference]
    rows = []
    for dataset in sorted(ref["dataset"].unique()):
        ref_part = ref.loc[ref["dataset"] == dataset]
        ref_sum = metrics.summarize(ref_part)
        row: dict[str, Any] = {
            "dataset": dataset,
            "n": ref_sum["n"],
            f"{reference} success": ref_sum["success_rate"],
            f"{reference} cost USD": ref_sum["mean_cost_usd"],
        }
        for label, frame in frames.items():
            if label == reference:
                continue
            part = metrics.summarize(frame.loc[frame["dataset"] == dataset])
            row[f"{label} success"] = part["success_rate"]
            row[f"{label} cost USD"] = part["mean_cost_usd"]
            row[f"{label} success delta"] = part["success_rate"] - ref_sum["success_rate"]
            ref_cost = ref_sum["mean_cost_usd"]
            row[f"{label} cost reduction"] = (
                1.0 - part["mean_cost_usd"] / ref_cost if ref_cost > 0 else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --- the experiment ------------------------------------------------------------------------------


def _op_entry(
    param_name: str | None,
    param: float | None,
    calib_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    row: dict[str, Any],
    **more: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        CALIB: sweeps.summarize_point(calib_frame, CALIB),
        TEST: sweeps.summarize_point(test_frame, TEST),
        "test_row": row,
        "targets": target_check(row),
        **more,
    }
    if param_name is not None:
        entry = {"param_name": param_name, "param": param, **entry}
    return entry


def _crossfit_phase(
    rp: "sweeps.Replayer",
    scratch: Path,
    manifest: Mapping[str, Any],
    blob: bytes,
    replay_input: Path,
    models: Sequence[str],
    pool_name: str,
    policy_config: Path,
    best_calib: pd.DataFrame,
    iters: int,
) -> dict[str, Any]:
    """Refit the calibrators on half A, tune lambda on half B (calibration items only)."""
    rows = rescore.read_split(replay_input, CALIB, models)
    if set(rows.items["item_id"]) != set(best_calib["item_id"]):
        raise ValueError("the replay input's calibration items differ from the replayed ones")
    raw = rescore.quality_scores(manifest, blob, rows.items["prompt"].tolist())
    in_a, in_b = crossfit.split_halves(rows.items["item_id"])
    xf_dir = crossfit.refit_artifact(
        manifest, blob, rows.items, raw, rows.success, in_a, scratch / "crossfit", pool_name
    )
    xf_manifest, _ = artifact_io.load_artifact(xf_dir)
    b_ids = set(rows.items.loc[in_b, "item_id"])
    best_b = best_calib.loc[best_calib["item_id"].isin(b_ids)]
    target_b = dual.calibration_target(best_b)
    flags = ["--artifact", str(xf_dir), "--policy-config", str(policy_config)]

    def run_b(lam: float) -> float:
        frame = rp.one(CALIB, "learned", flags + sweeps.dual_flags(lam))
        return dual.realized_success(frame.loc[frame["item_id"].isin(b_ids)])

    info: dict[str, Any] = {
        "artifact_version": xf_dir.name,
        "artifact_dir": xf_dir,
        "flags": flags,
        "n_a": int(in_a.sum()),
        "n_b": int(in_b.sum()),
        "salt": crossfit.SALT,
        "calibrator_types": _count(m["calibrator"]["type"] for m in xf_manifest["models"]),
        "best_single_b_success": dual.realized_success(best_b),
        "target_b": target_b,
        "success_b_at_zero": run_b(0.0),
    }
    try:
        lam = dual.tune_lambda(run_b, target_b, iters=iters)
    except ValueError as err:
        info |= {"status": "unreachable", "message": str(err), "lambda": None}
        return info
    info |= {"status": "tuned", "lambda": lam, "success_b_at_lambda": run_b(lam)}
    return info


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def run_headline(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    mf_path: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    bootstrap: int = baselines.BOOTSTRAP,
    n_thresholds: int = sweeps.N_THRESHOLDS,
    floors: Sequence[float] | None = None,
    iters: int = dual.BISECTION_STEPS,
    workers: int = sweeps.WORKERS,
    lambda_source: str = "config",
    do_crossfit: bool = True,
) -> Path:
    """Run both phases for ``pool_name`` and write the results under ``out_root/<run-id>``.

    Raises ``ValueError`` before any test replay when the policy config's lambda does not
    reach the calibration target (``lambda_source="config"``) or no lambda does (``tune``).
    """
    if lambda_source not in LAMBDA_SOURCES:
        raise ValueError(f"lambda_source must be one of {LAMBDA_SOURCES}, got {lambda_source!r}")
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    strong, cheapest, models = pool["best_single"], pool["cheapest"], list(pool["models"])
    mf = routellm_mf.read_mf(mf_path)
    if (mf["strong"], mf["weak"]) != (strong, cheapest):
        raise ValueError(
            f"the MF file routes {mf['strong']} against {mf['weak']}, not the pool's Best Single "
            f"{strong} against its cheapest {cheapest}; retrain it"
        )
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    binary = binary or replaytool.build_go(ROOT)
    manifest, blob = artifact_io.load_artifact(artifact)
    floor_values = list(floors) if floors is not None else sweeps.floor_grid()
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    specs = baselines.policy_specs(pool)
    spec_names = [name for name, _, _ in specs]
    phase_times: dict[str, float] = {}

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        rp = sweeps.Replayer(binary, config, replay_input, scratch, workers)
        try:
            # ---- calibration phase: nothing here replays or reads the test split
            tick = time.perf_counter()
            best_calib = rp.one(CALIB, "fixed", ["--model", strong])
            best_success = dual.realized_success(best_calib)
            direct = sweeps.best_single_calib_success(replay_input, strong)
            if abs(direct - best_success) > 1e-12:
                raise ValueError(
                    f"Best Single's calibration success is {best_success} by replay but {direct} "
                    "by the replay input"
                )
            target = dual.calibration_target(best_calib)

            def run_calib(lam: float) -> float:
                return dual.realized_success(
                    rp.one(CALIB, "learned", learned + sweeps.dual_flags(lam))
                )

            success_at_zero = run_calib(0.0)
            if lambda_source == "tune":
                lam = dual.tune_lambda(run_calib, target, iters=iters)
            else:
                if "lambda" not in base_config:
                    raise ValueError(f"{policy_config} has no lambda")
                lam = float(base_config["lambda"])
                if run_calib(lam) < target:
                    raise ValueError(
                        f"the policy config's lambda {lam!r} reaches {run_calib(lam):.6g} on the "
                        f"calibration split, below the target {target:.6g}; run sweeps.py again"
                    )
            say(f"lambda {lam!r} ({lambda_source}): calibration success {run_calib(lam):.5f}")

            floor_frames = rp.many(
                [(CALIB, "learned", learned + sweeps.floor_flags(q)) for q in floor_values]
            )
            floor_pick = sweeps.pick_floor(sweeps._calib_view(floor_values, floor_frames), target)
            scores = rp.one(CALIB, "routellm_mf", sweeps.mf_flags(mf_path, 0.5))["p"]
            thresholds = sweeps.quantile_thresholds(scores.to_numpy(dtype=np.float64), n_thresholds)
            mf_calib = rp.many(
                [(CALIB, "routellm_mf", sweeps.mf_flags(mf_path, t)) for t in thresholds]
            )
            mf_pick = sweeps.pick_threshold(sweeps._calib_view(thresholds, mf_calib), target)
            if mf_pick is None:
                raise ValueError("no RouteLLM MF threshold reaches the calibration target")
            floor_at = floor_values.index(floor_pick["param"])
            mf_at = thresholds.index(mf_pick["param"])
            say(f"floor {floor_pick['param']}, MF threshold {mf_pick['param']!r}")
            phase_times["calibration"] = time.perf_counter() - tick

            tick = time.perf_counter()
            xf = None
            if do_crossfit:
                xf = _crossfit_phase(
                    rp,
                    scratch,
                    manifest,
                    blob,
                    replay_input,
                    models,
                    pool_name,
                    policy_config,
                    best_calib,
                    iters,
                )
                say(f"cross-fit: A {xf['n_a']} B {xf['n_b']} lambda {xf['lambda']!r}")
            phase_times["crossfit"] = time.perf_counter() - tick

            # The calibration frames of the reported points (all cached: the replays ran above).
            calib_lambda_frame = rp.one(CALIB, "learned", learned + sweeps.dual_flags(lam))
            calib_zero_frame = rp.one(CALIB, "learned", learned + sweeps.dual_flags(0.0))
            xf_ok = xf is not None and xf["status"] == "tuned"
            calib_xf_frame = (
                rp.one(CALIB, "learned", xf["flags"] + sweeps.dual_flags(xf["lambda"]))
                if xf_ok
                else None
            )

            # ---- test phase: each point is replayed once
            tick = time.perf_counter()
            jobs: list[tuple[str, str, Sequence[str]]] = [
                (TEST, policy, extra) for _, policy, extra in specs
            ]
            jobs += [
                (TEST, "learned", learned + sweeps.dual_flags(lam)),
                (TEST, "learned", learned + sweeps.floor_flags(floor_pick["param"])),
                (TEST, "routellm_mf", sweeps.mf_flags(mf_path, mf_pick["param"])),
                (TEST, "learned", learned + sweeps.dual_flags(0.0)),
            ]
            if xf_ok:
                jobs.append((TEST, "learned", xf["flags"] + sweeps.dual_flags(xf["lambda"])))
            test_frames = rp.many(jobs)
            replay_counts = rp.counts()
            phase_times["test replays"] = time.perf_counter() - tick
        finally:
            rp.close()

        # ---- the shipped artifact's test calibration quality (the test split is read here)
        tick = time.perf_counter()
        test_rows = rescore.read_split(replay_input, TEST, models)
        raw_test = rescore.quality_scores(manifest, blob, test_rows.items["prompt"].tolist())
        p_test = {
            m["id"]: calibrate.apply_calibrator(m["calibrator"], raw_test[m["id"]])
            for m in manifest["models"]
        }
        quality = calibration_quality(p_test, test_rows.success)
        figure_reliability = reliability_figure(
            p_test,
            test_rows.success,
            quality,
            f"Test reliability of the shipped artifact {artifact.name}, {pool_name} pool",
        )
        phase_times["calibration quality"] = time.perf_counter() - tick
        if xf is not None:
            xf = {k: v for k, v in xf.items() if k not in ("artifact_dir", "flags")}

    frames = dict(zip(spec_names, test_frames[: len(specs)], strict=True))
    extra_frames = test_frames[len(specs) :]
    dual_test, floor_test, mf_test, zero_test = extra_frames[:4]
    xf_test = extra_frames[4] if xf_ok else None
    reference = frames[baselines.REFERENCE]

    def row_of(label: str, frame: pd.DataFrame) -> dict[str, Any]:
        return baselines._policy_row(label, frame, reference, bootstrap)

    baseline_rows = {name: row_of(name, frames[name]) for name in spec_names}
    operating: dict[str, Any] = {
        "dual": _op_entry(
            "lambda", lam, calib_lambda_frame, dual_test, row_of("learned_dual", dual_test)
        ),
        "min_cost": _op_entry(
            "floor",
            floor_pick["param"],
            floor_frames[floor_at],
            floor_test,
            row_of("learned_min_cost", floor_test),
        ),
        "routellm_mf": _op_entry(
            "threshold",
            mf_pick["param"],
            mf_calib[mf_at],
            mf_test,
            row_of("routellm_mf", mf_test),
            quantile=float(np.linspace(0.0, 1.0, n_thresholds)[mf_at]),
        ),
        "dual_lambda0_reference": _op_entry(
            "lambda",
            0.0,
            calib_zero_frame,
            zero_test,
            row_of("learned_dual_lambda0 (reference)", zero_test),
        ),
    }
    if xf_ok:
        assert xf_test is not None and calib_xf_frame is not None
        operating["crossfit_dual"] = _op_entry(
            "lambda",
            xf["lambda"],
            calib_xf_frame,
            xf_test,
            row_of("learned_dual_crossfit", xf_test),
        )
    parity = go_parity(dual_test, list(test_rows.items["item_id"]), p_test)

    dataset_frames = {
        baselines.REFERENCE: reference,
        "learned_dual": dual_test,
        **({"learned_dual_crossfit": xf_test} if xf_ok else {}),
    }
    per_dataset = per_dataset_table(dataset_frames)
    statements = [
        target_statement("learned dual (spec operating point)", operating["dual"]["targets"]),
        target_statement("learned min_cost", operating["min_cost"]["targets"]),
        target_statement("RouteLLM MF", operating["routellm_mf"]["targets"]),
        *(
            [target_statement("learned dual, cross-fitted", operating["crossfit_dual"]["targets"])]
            if xf_ok
            else []
        ),
    ]
    points = pd.DataFrame(
        {
            "policy": [*baseline_rows, *(op["test_row"]["policy"] for op in operating.values())],
            "mean_cost_usd": [
                *(r["mean_cost_usd"] for r in baseline_rows.values()),
                *(op["test_row"]["mean_cost_usd"] for op in operating.values()),
            ],
            "success_rate": [
                *(r["success_rate"] for r in baseline_rows.values()),
                *(op["test_row"]["success_rate"] for op in operating.values()),
            ],
        }
    )

    results: dict[str, Any] = {
        "pool": pool_name,
        "artifact_version": artifact.name,
        "n_items": {CALIB: len(best_calib), TEST: len(reference)},
        "best_single": strong,
        "cheapest": cheapest,
        "bootstrap": {
            "n": bootstrap,
            "seed": baselines.BOOTSTRAP_SEED,
            "alpha": 0.05,
            "method": "percentile",
        },
        "target": {
            "best_single": strong,
            "best_single_calib_success": best_success,
            "margin": dual.QUALITY_MARGIN,
            "target": target,
        },
        "lambda": {
            "source": lambda_source,
            "value": lam,
            "calib_success": dual.realized_success(calib_lambda_frame),
            "calib_success_at_zero": success_at_zero,
        },
        "policy_config": {k: v for k, v in base_config.items() if k != "lambda"},
        "calibrators": _count(m["calibrator"]["type"] for m in manifest["models"]),
        "mf_sha256": hashlib.sha256(Path(mf_path).read_bytes()).hexdigest(),
        "operating_points": operating,
        "baselines": baseline_rows,
        "per_dataset": per_dataset.to_dict("records"),
        "crossfit": xf,
        "calibration_quality": quality,
        "python_go_parity": parity,
        "target_statements": statements,
        "targets": {"cost_reduction": COST_TARGET, "success_margin": SUCCESS_MARGIN},
        "replays": replay_counts,
        "wall_time_s": time.perf_counter() - clock,
        "phase_time_s": phase_times,
        "labels": _labels(pool_name, xf is not None),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    shown = (
        list(command) if command else ["python", "bench/scripts/headline.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results, per_dataset, quality),
        {
            "pareto_test": _pareto_figure(points, pool_name),
            "reliability_test": figure_reliability,
        },
        {"bootstrap": baselines.BOOTSTRAP_SEED, "random": 0, "crossfit_salt": crossfit.SALT},
        shown,
        artifact.name,
        started,
        git=start_git,
    )
    return out_dir


# --- report --------------------------------------------------------------------------------------


def _labels(pool_name: str, crossfit_on: bool) -> list[str]:
    labels = [
        "no latency is measured or simulated in this experiment",
        "operating points (lambda, the floor, the RouteLLM threshold) are chosen on the "
        "calibration split only; the test split is read once per reported point",
        "the calibrators and lower-bound tables of the learned policy were fitted on the "
        "calibration split, so the learned policy's calibration success is in-sample and "
        "optimistic",
        "oracle looks at the recorded outcomes: an upper bound, not a policy",
        "the lambda = 0 dual row is a reference, not an operating point",
    ]
    if pool_name == "flagship":
        labels.insert(0, "costs are the recorded costs of each model on each item")
    else:
        labels.insert(
            0,
            "sensitivity pool; the four small models' costs are fitted from their recorded "
            "costs and applied to their recorded token counts (7B costs fitted from recorded "
            "costs); flagship costs are recorded",
        )
    if crossfit_on:
        labels.append(
            "the cross-fitted row refits calibrators and lower-bound tables on half A of the "
            "calibration items and tunes lambda on half B; its artifact is temporary and not "
            "committed"
        )
    return labels


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    target, lam = results["target"], results["lambda"]
    boot, items, replays = results["bootstrap"], results["n_items"], results["replays"]
    rows: list[list[Any]] = [
        ["pool", results["pool"]],
        ["artifact version", results["artifact_version"]],
        ["calibrators in the artifact", json.dumps(results["calibrators"])],
        ["items (calibration / test)", f"{items[CALIB]} / {items[TEST]}"],
        ["Best Single (always_strongest)", results["best_single"]],
        ["cheapest (RouteLLM weak model)", results["cheapest"]],
        ["Best Single calibration success", target["best_single_calib_success"]],
        ["target Q_bar (that minus the margin)", f"{target['target']:.6g} ({target['margin']})"],
        ["lambda (USD^-1)", f"{lam['value']!r} ({lam['source']})"],
        [
            "calibration success at that lambda / at lambda 0",
            f"{lam['calib_success']:.6g} / {lam['calib_success_at_zero']:.6g}",
        ],
        ["policy config (other than lambda)", json.dumps(results["policy_config"])],
        ["RouteLLM MF weights SHA-256", results["mf_sha256"]],
        ["bootstrap", f"{boot['n']} resamples, percentile 95% intervals, seed {boot['seed']}"],
        [
            "replays run (calibration / test)",
            f"{replays.get(CALIB, 0)} / {replays.get(TEST, 0)}",
        ],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _headline_table(results: Mapping[str, Any]) -> pd.DataFrame:
    ci = baselines._ci_text
    rows = []

    def add(kind: str, param: str, calib: Any, r: Mapping[str, Any]) -> None:
        rows.append(
            {
                "policy": r["policy"],
                "kind": kind,
                "parameter (chosen on calibration)": param,
                "calib success": calib,
                "test success [95% CI]": ci(r["success_rate"], r["success_ci"]),
                "test cost USD [95% CI]": ci(r["mean_cost_usd"], r["mean_cost_ci"], ".6g"),
                "success delta [95% CI]": ci(
                    r["success_delta_vs_strongest"], r["success_delta_ci"]
                ),
                "cost reduction [95% CI]": ci(
                    r["cost_reduction_vs_strongest"], r["cost_reduction_ci"]
                ),
            }
        )

    for r in results["baselines"].values():
        add("baseline", "", None, r)
    kinds = {
        "dual": "operating point (spec)",
        "min_cost": "operating point",
        "routellm_mf": "operating point (baseline)",
        "dual_lambda0_reference": "reference, not an operating point",
        "crossfit_dual": "sensitivity (cross-fitted)",
    }
    for key, op in results["operating_points"].items():
        add(
            kinds[key],
            f"{op['param_name']} = {op['param']!r}",
            op[CALIB]["success_rate"],
            op["test_row"],
        )
    return pd.DataFrame(rows)


def _targets_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for op in results["operating_points"].values():
        t = op["targets"]
        rows.append(
            {
                "policy": op["test_row"]["policy"],
                "cost reduction": t["cost_reduction"],
                f"cost target (>= {COST_TARGET:.0%})": "met" if t["cost_target_met"] else "NOT met",
                "success delta": t["success_delta"],
                f"success target (>= -{SUCCESS_MARGIN})": (
                    "met" if t["success_target_met"] else "NOT met"
                ),
                "both": "met" if t["both_met"] else "NOT met",
            }
        )
    return pd.DataFrame(rows)


def _crossfit_table(results: Mapping[str, Any]) -> pd.DataFrame | None:
    xf = results["crossfit"]
    if xf is None:
        return None
    rows: list[list[Any]] = [
        ["cross-fit artifact version", xf["artifact_version"]],
        ["salt", xf["salt"]],
        ["items in half A (refit) / half B (tuning)", f"{xf['n_a']} / {xf['n_b']}"],
        ["calibrators fitted on A (the spec rule)", json.dumps(xf["calibrator_types"])],
        ["Best Single success on B", xf["best_single_b_success"]],
        ["target Q_bar on B", xf["target_b"]],
        ["success on B at lambda 0", xf["success_b_at_zero"]],
        ["status", xf["status"]],
    ]
    if xf["status"] == "tuned":
        rows += [
            ["lambda tuned on B (USD^-1)", repr(xf["lambda"])],
            ["success on B at that lambda", xf["success_b_at_lambda"]],
        ]
    else:
        rows.append(["message", xf["message"]])
    return pd.DataFrame(rows, columns=["setting", "value"])


def _quality_table(quality: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, q in [*quality["per_model"].items(), ("pooled (all models)", quality["pooled"])]:
        rows.append(
            {
                "model": name,
                "n": q["n"],
                "base rate": q["base_rate"],
                "mean p": q["mean_p"],
                "ECE equal-mass": q["ece_mass"],
                "ECE equal-width": q["ece_width"],
                "Brier": q["brier"],
                "log-loss": q["log_loss"],
            }
        )
    return pd.DataFrame(rows)


def _tables(
    results: Mapping[str, Any], per_dataset: pd.DataFrame, quality: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    parity = results["python_go_parity"]
    tables: dict[str, pd.DataFrame] = {
        "Were the targets met?": pd.DataFrame(
            {"statement (test split, against always_strongest)": results["target_statements"]}
        ),
        "Setup": _setup_table(results),
        "Headline: test split, paired against always_strongest": _headline_table(results),
        "Targets (cost reduction >= 25%, success within 1 point of always_strongest)": (
            _targets_table(results)
        ),
    }
    xf = _crossfit_table(results)
    if xf is not None:
        tables["Cross-fitted sensitivity (calibration halves)"] = xf
    tables["Per dataset on the test split (operating point against always_strongest)"] = per_dataset
    tables["Test calibration quality of the shipped artifact (15 bins)"] = _quality_table(quality)
    tables["Python recomputation against the Go predictor"] = pd.DataFrame(
        [
            [
                "decisions compared (out-of-distribution and errors skipped)",
                parity["items_compared"],
            ],
            ["skipped", parity["items_skipped"]],
            ["largest absolute difference in p", parity["max_abs_diff"]],
        ],
        columns=["check", "value"],
    )
    return tables


def _pareto_figure(points: pd.DataFrame, pool_name: str) -> Figure:
    front = pareto.pareto_front(points)
    fig = Figure(figsize=(8.0, 5.0))
    ax = fig.subplots()
    ax.scatter(points["mean_cost_usd"], points["success_rate"], color="tab:blue", zorder=3)
    middle = float(np.sqrt(points["mean_cost_usd"].min() * points["mean_cost_usd"].max()))
    # Labels in cost order take turns above and below their points so that a cluster of
    # nearby points (the strong baselines) stays readable.
    for k, (cost, success, label) in enumerate(sorted(baselines.merged_labels(points))):
        right = cost > middle
        ax.annotate(
            label,
            (cost, success),
            fontsize=6,
            xytext=(-4 if right else 4, (5, -11, 15, -21)[k % 4]),
            ha="right" if right else "left",
            textcoords="offset points",
        )
    ax.step(
        front["mean_cost_usd"],
        front["success_rate"],
        where="post",
        color="tab:red",
        zorder=2,
        label="Pareto front (a point right of or below it is dominated)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per item (USD, recorded costs)")
    ax.set_ylabel("success rate")
    ax.set_title(f"Headline, test split, {pool_name} pool (operating points chosen on calibration)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headline offline evaluation.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=POLICY_CONFIG,
        help="default: bench/policies/flagship.yaml (its lambda is read for the flagship pool)",
    )
    parser.add_argument("--mf", type=Path, default=None, help="default: <derived>/mf/<pool>.json")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "headline")
    parser.add_argument(
        "--lambda-source",
        choices=LAMBDA_SOURCES,
        default=None,
        help="config: the policy config's lambda; tune: bisection on calibration "
        "(default: config for the flagship pool, tune otherwise)",
    )
    parser.add_argument(
        "--crossfit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="the cross-fitted sensitivity row (default: on for the flagship pool only)",
    )
    parser.add_argument("--workers", type=int, default=sweeps.WORKERS)
    args = parser.parse_args(argv)
    flagship = args.pool == "flagship"
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_headline(
        args.derived,
        args.pool,
        args.artifact or sweeps.find_artifact(ROOT / "policies", args.pool),
        args.policy_config,
        args.mf or args.derived / "mf" / f"{args.pool}.json",
        args.out_root,
        command,
        workers=args.workers,
        lambda_source=args.lambda_source or ("config" if flagship else "tune"),
        do_crossfit=flagship if args.crossfit is None else args.crossfit,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
