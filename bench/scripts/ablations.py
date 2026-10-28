"""Predictor ablations: feature set, hashed dimension and calibrator, each replayed on test.

Run from the repository root after ``make data`` and ``make train``:

    python bench/scripts/ablations.py --pool flagship

Seven artifacts, all on the same train and calibration data (ablation artifacts live under
``bench/scratch/ablations/`` and are never committed):

- feature set (dim 2^16, spec calibrators): ``both`` (the shipped artifact, not retrained),
  ``hashed`` only and ``dense`` only. A restricted model is fitted on the restricted design
  matrix and exported at full length with exact zeros in the dropped block, so Go serves it
  unchanged; the OOD detector is always fitted on the hashed part.
- hashed dimension (both features): 2^14, 2^16 (the shipped artifact) and 2^18.
- calibrator (both features, dim 2^16): every model ``platt`` or every model ``isotonic``. The
  spec rule picks isotonic from 1000 calibration items, so on this pool the shipped artifact is
  already all isotonic and the ``isotonic`` artifact is its retrain.

For each artifact, in two phases:

Calibration phase (the test split is not replayed or read): Best Single's calibration success
gives the target ``Q_bar = success - 0.01`` and lambda of the ``dual`` mode is tuned by
bisection on the calibration split, the same rule as the headline (``dual.tune_lambda``). An
artifact for which no lambda reaches ``Q_bar`` is reported as unreachable and has no test row.

Test phase: each tuned artifact is replayed on the test split once, with paired bootstrap
intervals against always_strongest and against the shipped configuration; the pooled test ECE
(15 equal-mass bins over all models' forecasts), Brier score and log-loss of the artifact's
calibrated ``p`` are recomputed in Python from the exported weights and cross-checked against
the Go predictor. Training wall time and ``weights.bin`` size are reported per artifact. Every
ablation is reported as it came out, whether or not it helps.

A PyTorch MLP on local Ollama embeddings is not run (Ruling V1-R19).

Writes bench/results/ablations/<run-id>/{results.json,report.md,run.json}. Output holds numbers,
model names and item IDs only, never prompt text.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from routepilot_train import replaytool, runinfo
from routepilot_train.evaluate import metrics, report, rescore
from routepilot_train.export import artifact as artifact_io
from routepilot_train.train import calibrate, dual, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (sibling scripts, not packages)
import headline  # noqa: E402
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "ablations"
CALIB, TEST = "calib", "test"
BOOTSTRAP = 1000
NOT_RUN = (
    "A PyTorch MLP on local Ollama embeddings is not run in this plan (Ruling V1-R19: no PyTorch "
    "dependency, and the predictor must stay Go-servable without an embedding model on the "
    "request path); the ADR of Task 36 records it as a deferred ablation."
)


@dataclass(frozen=True)
class Variant:
    """One artifact of the ablation: which columns, which hashed dimension, which calibrator."""

    name: str
    label: str
    group: str
    features: str
    dim: int
    calibrator: str
    shipped: bool = False


VARIANTS = (
    Variant("both-14", "both features, dim 2^14", "dimension", "both", 2**14, "switch"),
    Variant(
        "both-16", "both features, dim 2^16 (shipped)", "features", "both", 2**16, "switch", True
    ),
    Variant("both-18", "both features, dim 2^18", "dimension", "both", 2**18, "switch"),
    Variant("hashed-16", "hashed features only, dim 2^16", "features", "hashed", 2**16, "switch"),
    Variant("dense-16", "dense features only, dim 2^16", "features", "dense", 2**16, "switch"),
    Variant("platt-16", "every model Platt, dim 2^16", "calibrator", "both", 2**16, "platt"),
    Variant(
        "isotonic-16", "every model isotonic, dim 2^16", "calibrator", "both", 2**16, "isotonic"
    ),
)


# --- artifacts --------------------------------------------------------------------------------


def train_variant(
    variant: Variant,
    derived: Path,
    pool_name: str,
    scratch: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train ``variant`` into ``scratch/<name>/`` and return its artifact and wall time.

    Anything an earlier run left in that directory is removed first, so it holds one version.
    The training run's own result files go to ``scratch/<name>/results`` and not to
    ``bench/results``.
    """
    root = Path(scratch) / variant.name
    if root.exists():
        shutil.rmtree(root)
    summary = pipeline.run(
        pool_name,
        derived,
        root / "policies",
        variant.dim,
        root / "results",
        progress=progress,
        calibrator=variant.calibrator,
        feature_set=variant.features,
    )
    return {
        "artifact_dir": summary["artifact_dir"],
        "version": summary["version"],
        "wall_time_s": summary["wall_time_s"],
        "trained": True,
        "wall_time_source": "this run",
    }


def shipped_wall_time(
    results_root: Path, pool_name: str, version: str
) -> tuple[float | None, str | None]:
    """Wall time and run ID of the committed training run that produced artifact ``version``.

    Looks in ``<results_root>/train/*/results.json``; ``(None, None)`` when there is none.
    """
    base = Path(results_root) / "train"
    if not base.is_dir():
        return None, None
    for run_dir in sorted(base.iterdir()):
        path = run_dir / "results.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("pool") == pool_name and data.get("artifact_version") == version:
            return float(data["wall_time_s"]), run_dir.name
    return None, None


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# --- judgement --------------------------------------------------------------------------------


def judge(interval: Sequence[float] | None) -> str:
    """``better`` or ``worse`` when a paired 95% interval lies wholly above or below zero."""
    if interval is None:
        return "no clear difference"
    if interval[0] > 0:
        return "better"
    if interval[1] < 0:
        return "worse"
    return "no clear difference"


def statement(
    variant: Mapping[str, Any], vs: Mapping[str, Any] | None, ece: float, ece_shipped: float
) -> str:
    """One plain sentence on how an ablation compares with the shipped configuration."""
    word = "lower" if ece < ece_shipped else "higher" if ece > ece_shipped else "equal"
    tail = f"pooled test ECE {ece:.4f} against {ece_shipped:.4f} for the shipped one ({word})"
    if vs is None:
        return f"{variant['label']}: no test row to compare with the shipped one; {tail}."
    cost = vs["cost_reduction"]
    cost_text = (
        "n/a" if cost is None else f"{cost * 100:+.1f}% cheaper ({judge(vs['cost_reduction_ci'])})"
    )
    return (
        f"{variant['label']}: test success {vs['success_delta'] * 100:+.2f} points against the "
        f"shipped one ({judge(vs['success_delta_ci'])}), mean cost {cost_text}; {tail}."
    )


# --- the experiment ---------------------------------------------------------------------------


def run_ablations(
    derived: Path,
    pool_name: str,
    shipped: Path,
    policy_config: Path,
    scratch: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    bootstrap: int = BOOTSTRAP,
    iters: int = dual.BISECTION_STEPS,
    workers: int = sweeps.WORKERS,
    train: Callable[..., dict[str, Any]] = train_variant,
    variants: Sequence[Variant] = VARIANTS,
    results_root: Path | None = None,
) -> Path:
    """Train, tune, replay and write the results under ``out_root/<run-id>``.

    ``shipped`` is the shipped artifact (not retrained). ``train(variant, derived, pool_name,
    scratch, progress)`` builds the others. ``results_root`` (default ``bench/results``) is where
    the shipped artifact's committed training run is looked up for its wall time.
    """
    started = datetime.now(UTC)
    clock = time.perf_counter()
    derived, shipped, policy_config = Path(derived), Path(shipped), Path(policy_config)
    scratch = Path(scratch)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    strong, models = pool["best_single"], list(pool["models"])
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    binary = binary or replaytool.build_go(ROOT)
    results_root = Path(results_root) if results_root is not None else ROOT / "bench" / "results"
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    phase: dict[str, float] = {}

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    # ---- training: train and calibration rows only
    tick = time.perf_counter()
    built: dict[str, dict[str, Any]] = {}
    for v in variants:
        if v.shipped:
            wall, source = shipped_wall_time(results_root, pool_name, shipped.name)
            built[v.name] = {
                "artifact_dir": shipped,
                "wall_time_s": wall,
                "trained": False,
                "wall_time_source": f"bench/results/train/{source}" if source else None,
            }
        else:
            say(f"training {v.name} ({v.features}, dim {v.dim}, {v.calibrator})")
            built[v.name] = train(v, derived, pool_name, scratch, say)
            say(f"  {v.name}: {built[v.name]['wall_time_s']:.1f} s")
    loaded = {v.name: artifact_io.load_artifact(built[v.name]["artifact_dir"]) for v in variants}
    phase["training"] = time.perf_counter() - tick

    def flags(v: Variant) -> list[str]:
        return [
            "--artifact",
            str(built[v.name]["artifact_dir"]),
            "--policy-config",
            str(policy_config),
        ]

    with tempfile.TemporaryDirectory() as scratch_name:
        rp = sweeps.Replayer(binary, config, replay_input, Path(scratch_name), workers)
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

            def tune(v: Variant) -> dict[str, Any]:
                def run_calib(lam: float) -> float:
                    frame = rp.one(CALIB, "learned", flags(v) + sweeps.dual_flags(lam))
                    return dual.realized_success(frame)

                at_zero = run_calib(0.0)
                try:
                    lam = dual.tune_lambda(run_calib, target, iters=iters)
                except ValueError as err:
                    return {
                        "status": "unreachable",
                        "message": str(err),
                        "value": None,
                        "calib_success_at_zero": at_zero,
                    }
                frame = rp.one(CALIB, "learned", flags(v) + sweeps.dual_flags(lam))
                summary = sweeps.summarize_point(frame, CALIB)
                return {
                    "status": "tuned",
                    "value": lam,
                    "calib_success": summary["success_rate"],
                    "calib_mean_cost_usd": summary["mean_cost_usd"],
                    "calib_success_at_zero": at_zero,
                }

            with ThreadPoolExecutor(max_workers=len(variants)) as outer:
                tuned = list(outer.map(tune, variants))
            phase["calibration"] = time.perf_counter() - tick
            say(
                "calibration phase done: "
                + ", ".join(
                    f"{v.name}={t['value']!r}" for v, t in zip(variants, tuned, strict=True)
                )
            )

            # ---- test phase: each tuned artifact and always_strongest are replayed once
            tick = time.perf_counter()
            jobs: list[tuple[str, str, Sequence[str]]] = [(TEST, "fixed", ["--model", strong])]
            jobs += [
                (TEST, "learned", flags(v) + sweeps.dual_flags(t["value"]))
                for v, t in zip(variants, tuned, strict=True)
                if t["status"] == "tuned"
            ]
            test_frames = rp.many(jobs)
            replay_counts = rp.counts()
            phase["test replays"] = time.perf_counter() - tick
        finally:
            rp.close()

    reference = test_frames[0]
    frame_of = dict(
        zip(
            [v.name for v, t in zip(variants, tuned, strict=True) if t["status"] == "tuned"],
            test_frames[1:],
            strict=True,
        )
    )

    # ---- test calibration quality of every artifact (the test split is read here)
    tick = time.perf_counter()
    test_rows = rescore.read_split(replay_input, TEST, models)
    texts = test_rows.items["prompt"].tolist()
    item_ids = list(test_rows.items["item_id"])
    shipped_name = next((v.name for v in variants if v.shipped), None)
    shipped_frame = frame_of.get(shipped_name) if shipped_name else None
    entries: list[dict[str, Any]] = []
    for v, t in zip(variants, tuned, strict=True):
        manifest, blob = loaded[v.name]
        raw = rescore.quality_scores(manifest, blob, texts)
        p_test = {
            m["id"]: calibrate.apply_calibrator(m["calibrator"], raw[m["id"]])
            for m in manifest["models"]
        }
        frame = frame_of.get(v.name)
        row = vs_shipped = parity = None
        if frame is not None:
            row = baselines._policy_row(v.label, frame, reference, bootstrap)
            if shipped_frame is not None:
                vs_shipped = metrics.paired_delta(frame, shipped_frame, n=bootstrap, seed=0)
            parity = headline.go_parity(frame, item_ids, p_test)
        info = built[v.name]
        entries.append(
            {
                "name": v.name,
                "label": v.label,
                "group": v.group,
                "features": v.features,
                "dim": v.dim,
                "calibrator": v.calibrator,
                "calibrator_types": _count(m["calibrator"]["type"] for m in manifest["models"]),
                "artifact_version": built[v.name]["artifact_dir"].name,
                "trained": info["trained"],
                "training_wall_time_s": info["wall_time_s"],
                "wall_time_source": info["wall_time_source"],
                "weights_bin_bytes": (built[v.name]["artifact_dir"] / "weights.bin").stat().st_size,
                "ood_threshold": manifest["ood"]["threshold"],
                "lambda": t,
                "test": {
                    "row": row,
                    "vs_shipped": vs_shipped,
                    "calibration_quality": headline.calibration_quality(p_test, test_rows.success),
                    "python_go_parity": parity,
                },
            }
        )
    phase["test calibration quality"] = time.perf_counter() - tick

    always = baselines._policy_row(baselines.REFERENCE, reference, reference, bootstrap)
    ece_shipped = next(
        (
            e["test"]["calibration_quality"]["pooled"]["ece_mass"]
            for e in entries
            if e["name"] == shipped_name
        ),
        None,
    )
    statements = [
        statement(
            e,
            e["test"]["vs_shipped"],
            e["test"]["calibration_quality"]["pooled"]["ece_mass"],
            ece_shipped,
        )
        for e in entries
        if e["name"] != shipped_name and ece_shipped is not None
    ]
    results: dict[str, Any] = {
        "pool": pool_name,
        "shipped_artifact": shipped.name,
        "n_items": {CALIB: len(best_calib), TEST: len(reference)},
        "best_single": strong,
        "bootstrap": {"n": bootstrap, "seed": 0, "alpha": 0.05, "method": "percentile"},
        "target": {
            "best_single": strong,
            "best_single_calib_success": best_success,
            "margin": dual.QUALITY_MARGIN,
            "target": target,
        },
        "policy_config": {k: v for k, v in base_config.items() if k != "lambda"},
        "always_strongest": always,
        "variants": entries,
        "statements": statements,
        "not_run": NOT_RUN,
        "replays": replay_counts,
        "wall_time_s": time.perf_counter() - clock,
        "phase_time_s": phase,
        "labels": _labels(),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started)
    shown = (
        list(command) if command else ["python", "bench/scripts/ablations.py", "--pool", pool_name]
    )
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {},
        {"bootstrap": 0, "kfold": 0, "kmeans": 0},
        shown,
        shipped.name,
        started,
    )
    return out_dir


# --- report -----------------------------------------------------------------------------------


def _labels() -> list[str]:
    return [
        "costs are the recorded costs of each model on each item; no latency is measured or "
        "simulated in this experiment",
        "lambda of the dual mode is retuned on the calibration split for each artifact with the "
        "same rule as the headline (the largest value whose calibration success reaches Best "
        "Single's minus 0.01); the test split is replayed once per artifact",
        "calibration success is in-sample for the calibrators and lower-bound tables of every "
        "artifact (fitted on the same split lambda is tuned on)",
        "because every operating point is retuned to the same calibration target, test success "
        "differs little by construction; the cost, the paired intervals and the pooled test ECE "
        "carry the comparison",
        "better or worse means the paired 95% bootstrap interval against the shipped "
        "configuration lies wholly above or below zero; ECE is compared by its point estimate",
        "pooled test ECE: 15 equal-mass bins over all models' test forecasts, recomputed in "
        "Python from the exported weights",
        "training wall time is one run per artifact on this machine, measured around the whole "
        "fit and export; the shipped artifact was not retrained, its time is that of the "
        "committed train run",
        "ablation artifacts are written under bench/scratch/ablations and are not committed",
    ]


def _ci(point: float | None, ci: Sequence[float] | None, digits: str = ".4f") -> str:
    return baselines._ci_text(point, ci, digits)


def _lambda_text(t: Mapping[str, Any]) -> str:
    return "n/a" if t["value"] is None else repr(t["value"])


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    target, items, boot = results["target"], results["n_items"], results["bootstrap"]
    a = results["always_strongest"]
    rows = [
        ["pool", results["pool"]],
        ["shipped artifact", results["shipped_artifact"]],
        ["items (calibration / test)", f"{items[CALIB]} / {items[TEST]}"],
        ["Best Single (always_strongest)", results["best_single"]],
        ["Best Single calibration success", target["best_single_calib_success"]],
        ["target Q_bar (that minus the margin)", f"{target['target']:.6g} ({target['margin']})"],
        ["always_strongest test success [95% CI]", _ci(a["success_rate"], a["success_ci"])],
        [
            "always_strongest test cost USD [95% CI]",
            _ci(a["mean_cost_usd"], a["mean_cost_ci"], ".6g"),
        ],
        ["policy config (other than lambda)", json.dumps(results["policy_config"])],
        ["bootstrap", f"{boot['n']} resamples, percentile 95% intervals, seed {boot['seed']}"],
        [
            "replays run (calibration / test)",
            f"{results['replays'].get(CALIB, 0)} / {results['replays'].get(TEST, 0)}",
        ],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _results_table(results: Mapping[str, Any]) -> pd.DataFrame:
    a = results["always_strongest"]
    rows = [
        {
            "variant": "always_strongest (reference)",
            "lambda": "",
            "calib success": "",
            "test success [95% CI]": _ci(a["success_rate"], a["success_ci"]),
            "test cost USD [95% CI]": _ci(a["mean_cost_usd"], a["mean_cost_ci"], ".6g"),
            "success delta vs always_strongest [95% CI]": "",
            "cost reduction vs always_strongest [95% CI]": "",
        }
    ]
    for e in results["variants"]:
        r, t = e["test"]["row"], e["lambda"]
        rows.append(
            {
                "variant": e["label"],
                "lambda": _lambda_text(t),
                "calib success": t.get("calib_success", "n/a"),
                "test success [95% CI]": "n/a"
                if r is None
                else _ci(r["success_rate"], r["success_ci"]),
                "test cost USD [95% CI]": (
                    "n/a" if r is None else _ci(r["mean_cost_usd"], r["mean_cost_ci"], ".6g")
                ),
                "success delta vs always_strongest [95% CI]": (
                    "n/a"
                    if r is None
                    else _ci(r["success_delta_vs_strongest"], r["success_delta_ci"])
                ),
                "cost reduction vs always_strongest [95% CI]": (
                    "n/a"
                    if r is None
                    else _ci(r["cost_reduction_vs_strongest"], r["cost_reduction_ci"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _against_shipped_table(results: Mapping[str, Any]) -> pd.DataFrame:
    shipped = next((e for e in results["variants"] if e["name"] == _shipped_name(results)), None)
    ece_shipped = shipped["test"]["calibration_quality"]["pooled"]["ece_mass"] if shipped else None
    rows = []
    for e in results["variants"]:
        vs = e["test"]["vs_shipped"]
        ece = e["test"]["calibration_quality"]["pooled"]["ece_mass"]
        rows.append(
            {
                "variant": e["label"],
                "success delta [95% CI]": (
                    "n/a" if vs is None else _ci(vs["success_delta"], vs["success_delta_ci"])
                ),
                "cost reduction [95% CI]": (
                    "n/a"
                    if vs is None or vs["cost_reduction"] is None
                    else _ci(vs["cost_reduction"], vs["cost_reduction_ci"])
                ),
                "pooled test ECE": ece,
                "ECE minus the shipped one": "n/a" if ece_shipped is None else ece - ece_shipped,
            }
        )
    return pd.DataFrame(rows)


def _shipped_name(results: Mapping[str, Any]) -> str | None:
    return next(
        (
            e["name"]
            for e in results["variants"]
            if e["artifact_version"] == results["shipped_artifact"]
        ),
        None,
    )


def _artifact_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for e in results["variants"]:
        q = e["test"]["calibration_quality"]["pooled"]
        parity = e["test"]["python_go_parity"]
        rows.append(
            {
                "variant": e["label"],
                "artifact version": e["artifact_version"],
                "calibrators": json.dumps(e["calibrator_types"]),
                "pooled test ECE (15 equal-mass bins)": q["ece_mass"],
                "pooled test ECE (equal-width)": q["ece_width"],
                "Brier": q["brier"],
                "log-loss": q["log_loss"],
                "OOD threshold": e["ood_threshold"],
                "training wall time (s)": (
                    "n/a"
                    if e["training_wall_time_s"] is None
                    else round(e["training_wall_time_s"], 1)
                ),
                "wall time from": e["wall_time_source"] or "not available",
                "weights.bin size (bytes)": e["weights_bin_bytes"],
                "largest |p Go - p Python|": "n/a" if parity is None else parity["max_abs_diff"],
            }
        )
    return pd.DataFrame(rows)


def _lambda_table(results: Mapping[str, Any]) -> pd.DataFrame:
    target = results["target"]["target"]
    rows = []
    for e in results["variants"]:
        t = e["lambda"]
        rows.append(
            {
                "variant": e["label"],
                "status": t["status"],
                "lambda (USD^-1)": _lambda_text(t),
                "calib success at lambda": t.get("calib_success", "n/a"),
                "target Q_bar": target,
                "calib success at lambda 0": t["calib_success_at_zero"],
                "calib mean cost USD": t.get("calib_mean_cost_usd", "n/a"),
            }
        )
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "Does each ablation help? (against the shipped configuration, test split)": (
            pd.DataFrame({"statement": results["statements"]})
        ),
        "Setup": _setup_table(results),
        "Ablations: test split, dual operating point retuned on calibration "
        "(paired against always_strongest)": _results_table(results),
        "Against the shipped configuration (paired by test item)": _against_shipped_table(results),
        "Artifacts: calibration quality, training wall time and weights.bin size": (
            _artifact_table(results)
        ),
        "Lambda retuned on calibration (target Q_bar = Best Single's minus 0.01)": _lambda_table(
            results
        ),
        "Not run": pd.DataFrame({"note": [results["not_run"]]}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predictor ablations on the test split.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="the shipped one; default: under policies/<pool>",
    )
    parser.add_argument(
        "--policy-config", type=Path, default=None, help="default: bench/policies/<pool>.yaml"
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=ROOT / "bench" / "scratch" / "ablations",
        help="where the ablation artifacts are written (not committed)",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "ablations")
    parser.add_argument("--workers", type=int, default=sweeps.WORKERS)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_ablations(
        args.derived,
        args.pool,
        args.artifact or sweeps.find_artifact(ROOT / "policies", args.pool),
        args.policy_config or ROOT / "bench" / "policies" / f"{args.pool}.yaml",
        args.scratch,
        args.out_root,
        command,
        workers=args.workers,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
