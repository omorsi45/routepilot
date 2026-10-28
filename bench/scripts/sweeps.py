"""Pareto sweeps of the learned policy against every baseline, on calibration and test.

Run from the repository root after ``make data``, ``make train`` and the MF export
(``python -m routepilot_train.baselines.routellm_mf --pool flagship --derived data/derived``):

    python bench/scripts/sweeps.py --pool flagship

Everything goes through the Go ``routepilot replay`` command and has two phases.

Calibration phase (nothing else is read):

1. Best Single's calibration success gives the target ``Q_bar = success - 0.01``.
2. ``dual.tune_lambda`` bisects lambda in [0, 1e6] (40 halvings) for the largest value whose
   learned ``dual`` replay still reaches ``Q_bar`` on the calibration split.
3. Three sweeps on the calibration split: the ``dual`` mode over lambda (0 and 40 log-spaced
   values from 1e-2 to 1e5 per USD), ``min_cost`` over the floor (0, 0.05, ..., 0.95) and the
   RouteLLM MF scorer over 41 thresholds at calibration-score quantiles; plus the static
   baselines of ``baselines.py``.
4. The operating points: the tuned lambda; the floor whose calibration success is closest to
   ``Q_bar``; the cheapest MF threshold whose calibration success reaches ``Q_bar``.

Test phase: every point of the sweeps and every baseline is replayed on the test split, once,
and the three operating points get paired bootstrap intervals against always_strongest. The
test curves are shown "for reference": no choice above reads them.

The tuned lambda is written to the policy config last, after the results were written, so the
recorded git state is that of the code that ran. Writes
bench/results/sweeps/<run-id>/{results.json,report.md,pareto_test.png,pareto_calib.png,run.json}.
Output holds numbers, model names and item IDs only, never prompt text.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from routepilot_train import replaytool, runinfo
from routepilot_train.baselines import routellm_mf
from routepilot_train.data.splits import require_splits
from routepilot_train.evaluate import metrics, report
from routepilot_train.train import dual

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
CALIB, TEST = "calib", "test"
N_LAMBDA = 40
N_THRESHOLDS = 41
N_FLOORS = 20
WORKERS = 8
EXPERIMENT = "sweeps"
NOTE = "test, for reference; operating points chosen on calibration"
# Operating point key: (row label, name of its parameter, plot color).
OPERATING = {
    "dual": ("learned_dual", "lambda", "tab:blue"),
    "min_cost": ("learned_min_cost", "floor", "tab:orange"),
    "routellm_mf": ("routellm_mf", "threshold", "tab:green"),
}
LAMBDA_LINE = re.compile(r"^lambda:[^\n]*$", re.MULTILINE)
Key = tuple[str, str, tuple[str, ...]]


# --- grids and choices -------------------------------------------------------------------------


def lambda_grid(n: int = N_LAMBDA) -> list[float]:
    """0 and ``n`` log-spaced values from 1e-2 to 1e5, in USD^-1."""
    return [0.0, *np.logspace(-2, 5, n).tolist()]


def floor_grid() -> list[float]:
    """The ``min_cost`` floors 0, 0.05, ..., 0.95."""
    return [round(0.05 * i, 2) for i in range(N_FLOORS)]


def quantile_thresholds(scores: Sequence[float] | np.ndarray, n: int = N_THRESHOLDS) -> list[float]:
    """``n`` thresholds at evenly spaced quantiles (0 to 1) of the calibration ``scores``."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no scores to take quantiles of")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return np.quantile(values, np.linspace(0.0, 1.0, n)).tolist()


def pick_floor(points: Sequence[Mapping[str, Any]], target: float) -> dict[str, Any]:
    """The calibration point whose success is closest to ``target``.

    Each point has ``param`` (the floor), ``success_rate`` and ``mean_cost_usd`` from the
    calibration split. A tie in distance goes to the cheaper point, then to the lower floor.
    """
    if not points:
        raise ValueError("no points to pick a floor from")
    best = min(
        points,
        key=lambda p: (round(abs(p["success_rate"] - target), 12), p["mean_cost_usd"], p["param"]),
    )
    return dict(best)


def pick_threshold(points: Sequence[Mapping[str, Any]], target: float) -> dict[str, Any] | None:
    """The cheapest calibration point whose success reaches ``target``, or None.

    A tie in cost goes to the higher threshold.
    """
    meeting = [p for p in points if p["success_rate"] >= target]
    if not meeting:
        return None
    return dict(min(meeting, key=lambda p: (p["mean_cost_usd"], -p["param"])))


def set_lambda(path: Path, value: float) -> None:
    """Set the ``lambda:`` line of a policy config to ``value`` at full float precision.

    Every other byte of the file stays as it was. The number is written so that YAML 1.1
    readers also see a float (``1.0e-05``, never ``1e-05``).
    """
    if not math.isfinite(value):
        raise ValueError(f"lambda must be finite, got {value}")
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not LAMBDA_LINE.search(text):
        raise ValueError(f"{path} has no lambda line")
    shown = repr(float(value))
    if "e" in shown and "." not in shown:
        shown = shown.replace("e", ".0e")
    path.write_text(
        LAMBDA_LINE.sub(lambda _: f"lambda: {shown}", text, count=1), encoding="utf-8", newline="\n"
    )


def find_artifact(root: Path, pool: str) -> Path:
    """The only version directory under ``root/pool``."""
    parent = Path(root) / pool
    versions = sorted(p for p in parent.glob("*") if p.is_dir()) if parent.is_dir() else []
    if not versions:
        raise ValueError(f"no artifact under {parent.as_posix()}; run make train")
    if len(versions) > 1:
        names = ", ".join(p.name for p in versions)
        raise ValueError(f"more than one artifact under {parent.as_posix()} ({names})")
    return versions[0]


def best_single_calib_success(replay_input: Path, model: str) -> float:
    """Best Single's success rate over the calibration items, read from the replay input.

    Only the ``split`` of each line decides whether it is read further; this is the figure
    ``Q_bar`` is derived from, computed without the Go replay as a cross-check.
    """
    hits, total = 0, 0
    with Path(replay_input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] == CALIB:
                total += 1
                hits += bool(item["outcomes"][model]["success"])
    if total == 0:
        raise ValueError("the replay input has no calibration items")
    return hits / total


# --- replays -----------------------------------------------------------------------------------


class Replayer:
    """Runs ``routepilot replay`` jobs in parallel and caches them by split, policy and flags.

    A job that was already run is not run again, so no point is replayed twice.
    """

    def __init__(
        self, binary: Path, config: Path, replay_input: Path, scratch: Path, workers: int
    ) -> None:
        self.binary, self.config, self.input, self.scratch = binary, config, replay_input, scratch
        self._cache: dict[Key, pd.DataFrame] = {}
        self._lock = threading.Lock()
        self._count = 0
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def _run(self, key: Key) -> pd.DataFrame:
        split, policy, extra = key
        with self._lock:
            self._count += 1
            name = f"replay-{self._count}.jsonl"
        return replaytool.replay(
            self.binary, self.config, self.input, split, policy, self.scratch / name, extra
        )

    def many(self, jobs: Sequence[tuple[str, str, Sequence[str]]]) -> list[pd.DataFrame]:
        """The replay frame of every ``(split, policy, flags)`` job, in order."""
        keys: list[Key] = [(split, policy, tuple(extra)) for split, policy, extra in jobs]
        todo = [key for key in dict.fromkeys(keys) if key not in self._cache]
        for key, frame in zip(todo, self._pool.map(self._run, todo), strict=True):
            self._cache[key] = frame
        return [self._cache[key] for key in keys]

    def one(self, split: str, policy: str, extra: Sequence[str] = ()) -> pd.DataFrame:
        return self.many([(split, policy, extra)])[0]

    def counts(self) -> dict[str, int]:
        """How many distinct replays ran per split."""
        out: dict[str, int] = {}
        for split, _, _ in self._cache:
            out[split] = out.get(split, 0) + 1
        return out

    def close(self) -> None:
        self._pool.shutdown()


def dual_flags(lam: float) -> list[str]:
    return ["--mode", "dual", "--lambda", repr(float(lam))]


def floor_flags(floor: float) -> list[str]:
    return ["--mode", "min_cost", "--floor", repr(float(floor))]


def mf_flags(mf_path: Path, threshold: float) -> list[str]:
    return ["--mf", str(mf_path), "--threshold", repr(float(threshold))]


def summarize_point(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    """Success, cost, how often the floor or latency was relaxed, and the models used.

    Raises ``ValueError`` when a row of ``frame`` is not from ``split``.
    """
    require_splits(frame, {split})
    base = metrics.summarize(frame)
    relaxed = frame["relaxed"].map(
        lambda v: list(v) if isinstance(v, list | tuple | np.ndarray) else []
    )
    counts = frame["model"].fillna("(routing error)").value_counts().sort_index()
    return {
        **base,
        "relaxed_share": float((relaxed.map(len) > 0).mean()),
        "relaxed_quality_share": float(relaxed.map(lambda v: "quality" in v).mean()),
        "model_counts": {str(m): int(c) for m, c in counts.items()},
    }


def _entry(param: float, calib: pd.DataFrame, test: pd.DataFrame, **more: Any) -> dict[str, Any]:
    return {
        "param": param,
        **more,
        CALIB: summarize_point(calib, CALIB),
        TEST: summarize_point(test, TEST),
    }


def _calib_view(params: Sequence[float], frames: Sequence[pd.DataFrame]) -> list[dict[str, Any]]:
    """The parameter and the calibration numbers of each point: all a choice may look at."""
    views = []
    for param, frame in zip(params, frames, strict=True):
        summary = summarize_point(frame, CALIB)
        views.append(
            {
                "param": param,
                "success_rate": summary["success_rate"],
                "mean_cost_usd": summary["mean_cost_usd"],
            }
        )
    return views


# --- the experiment ----------------------------------------------------------------------------


def run_sweeps(
    derived: Path,
    pool_name: str,
    artifact: Path,
    policy_config: Path,
    mf_path: Path,
    out_root: Path,
    command: Sequence[str] | None = None,
    binary: Path | None = None,
    bootstrap: int = baselines.BOOTSTRAP,
    n_lambda: int = N_LAMBDA,
    n_thresholds: int = N_THRESHOLDS,
    floors: Sequence[float] | None = None,
    iters: int = dual.BISECTION_STEPS,
    workers: int = WORKERS,
) -> Path:
    """Run both phases for ``pool_name`` and write the results under ``out_root/<run-id>``.

    ``policy_config`` is read for every replay and, once the results are written, gets the
    tuned lambda. Raises ``ValueError`` (before any test replay and with the policy config
    untouched) when no lambda reaches the target on the calibration split.
    """
    started = datetime.now(UTC)
    derived, artifact, policy_config = Path(derived), Path(artifact), Path(policy_config)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    strong, cheapest = pool["best_single"], pool["cheapest"]
    mf = routellm_mf.read_mf(mf_path)
    if (mf["strong"], mf["weak"]) != (strong, cheapest):
        raise ValueError(
            f"the MF file routes {mf['strong']} against {mf['weak']}, not the pool's Best Single "
            f"{strong} against its cheapest {cheapest}; retrain it"
        )
    mf_sha256 = hashlib.sha256(Path(mf_path).read_bytes()).hexdigest()
    config = derived / "pools" / f"{pool_name}.config.yaml"
    replay_input = derived / "replay" / f"{pool_name}.jsonl"
    binary = binary or replaytool.build_go(ROOT)
    floor_values = list(floors) if floors is not None else floor_grid()
    learned = ["--artifact", str(artifact), "--policy-config", str(policy_config)]
    base_config = yaml.safe_load(policy_config.read_text(encoding="utf-8"))
    lambdas = lambda_grid(n_lambda)
    specs = baselines.policy_specs(pool)
    names = [name for name, _, _ in specs]

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory() as scratch:
        rp = Replayer(binary, config, replay_input, Path(scratch), workers)
        try:
            # ---- calibration phase: nothing here replays the test split
            best_calib = rp.one(CALIB, "fixed", ["--model", strong])
            best_success = dual.realized_success(best_calib)
            direct = best_single_calib_success(replay_input, strong)
            if abs(direct - best_success) > 1e-12:
                raise ValueError(
                    f"Best Single's calibration success is {best_success} by replay but {direct} "
                    "by the replay input"
                )
            target = dual.calibration_target(best_calib)
            say(
                f"best single {strong}: calibration success {best_success:.5f}, target {target:.5f}"
            )

            def run_calib(lam: float) -> float:
                return dual.realized_success(rp.one(CALIB, "learned", learned + dual_flags(lam)))

            success_at_zero = run_calib(0.0)
            tuned = dual.tune_lambda(run_calib, target, iters=iters)
            say(f"tuned lambda {tuned!r}: calibration success {run_calib(tuned):.5f}")

            scores_job = (CALIB, "routellm_mf", mf_flags(mf_path, 0.5))
            frames = rp.many(
                [(CALIB, "learned", learned + dual_flags(lam)) for lam in lambdas]
                + [(CALIB, "learned", learned + floor_flags(q)) for q in floor_values]
                + [(CALIB, policy, extra) for _, policy, extra in specs]
                + [scores_job]
            )
            n_l, n_f = len(lambdas), len(floor_values)
            lambda_calib, floor_calib = frames[:n_l], frames[n_l : n_l + n_f]
            baseline_calib = frames[n_l + n_f : -1]
            scores = frames[-1]["p"].to_numpy(dtype=np.float64)
            thresholds = quantile_thresholds(scores, n_thresholds)
            quantiles = np.linspace(0.0, 1.0, n_thresholds).tolist()
            mf_calib = rp.many([(CALIB, "routellm_mf", mf_flags(mf_path, t)) for t in thresholds])
            say("calibration sweeps done")

            # The operating points use the calibration numbers and nothing else.
            floor_pick = pick_floor(_calib_view(floor_values, floor_calib), target)
            mf_pick = pick_threshold(_calib_view(thresholds, mf_calib), target)
            if mf_pick is None:
                raise ValueError("no RouteLLM MF threshold reaches the calibration target")
            tuned_calib = rp.one(CALIB, "learned", learned + dual_flags(tuned))
            say(
                f"operating points: lambda {tuned!r}, floor {floor_pick['param']}, "
                f"threshold {mf_pick['param']!r}"
            )

            # ---- test phase: each point is replayed once
            n_t = len(thresholds)
            test_frames = rp.many(
                [(TEST, "learned", learned + dual_flags(lam)) for lam in lambdas]
                + [(TEST, "learned", learned + dual_flags(tuned))]
                + [(TEST, "learned", learned + floor_flags(q)) for q in floor_values]
                + [(TEST, "routellm_mf", mf_flags(mf_path, t)) for t in thresholds]
                + [(TEST, policy, extra) for _, policy, extra in specs]
            )
            lambda_test, tuned_test = test_frames[:n_l], test_frames[n_l]
            floor_test = test_frames[n_l + 1 : n_l + 1 + n_f]
            mf_test = test_frames[n_l + 1 + n_f : n_l + 1 + n_f + n_t]
            baseline_test = test_frames[n_l + 1 + n_f + n_t :]
            replay_counts = rp.counts()
            say("test replays done")
        finally:
            rp.close()

    sweeps = {
        "lambda": [
            _entry(lam, c, t) for lam, c, t in zip(lambdas, lambda_calib, lambda_test, strict=True)
        ],
        "floor": [
            _entry(q, c, t) for q, c, t in zip(floor_values, floor_calib, floor_test, strict=True)
        ],
        "threshold": [
            _entry(thr, c, t, quantile=q)
            for thr, q, c, t in zip(thresholds, quantiles, mf_calib, mf_test, strict=True)
        ],
    }
    baseline_summaries = {
        CALIB: {n: summarize_point(f, CALIB) for n, f in zip(names, baseline_calib, strict=True)},
        TEST: {n: summarize_point(f, TEST) for n, f in zip(names, baseline_test, strict=True)},
    }

    at = names.index(baselines.REFERENCE)
    reference = baseline_test[at]
    floor_at = floor_values.index(floor_pick["param"])
    mf_at = thresholds.index(mf_pick["param"])
    chosen = {
        "always_strongest": (None, baselines.REFERENCE, baseline_calib[at], reference),
        "dual": (tuned, OPERATING["dual"][0], tuned_calib, tuned_test),
        "min_cost": (
            floor_pick["param"],
            OPERATING["min_cost"][0],
            floor_calib[floor_at],
            floor_test[floor_at],
        ),
        "routellm_mf": (
            mf_pick["param"],
            OPERATING["routellm_mf"][0],
            mf_calib[mf_at],
            mf_test[mf_at],
        ),
    }
    operating: dict[str, Any] = {}
    for key, (param, label, calib_frame, test_frame) in chosen.items():
        entry: dict[str, Any] = {
            CALIB: summarize_point(calib_frame, CALIB),
            TEST: summarize_point(test_frame, TEST),
            "test_row": baselines._policy_row(label, test_frame, reference, bootstrap),
        }
        if param is not None:
            entry = {"param_name": OPERATING[key][1], "param": param, **entry}
        operating[key] = entry
    operating["routellm_mf"]["quantile"] = quantiles[mf_at]

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
        "lambda_tuning": {
            "status": "tuned",
            "lambda": tuned,
            "lo": 0.0,
            "hi": dual.LAMBDA_MAX,
            "iters": iters,
            "calib_success": dual.realized_success(tuned_calib),
            "calib_success_at_zero": success_at_zero,
        },
        "policy_config": {k: v for k, v in base_config.items() if k != "lambda"},
        "mf": {
            "sha256": mf_sha256,
            "strong": mf["strong"],
            "weak": mf["weak"],
            "dim": mf["dim"],
        },
        "operating_points": operating,
        "sweeps": sweeps,
        "baselines": baseline_summaries,
        "replays": replay_counts,
        "labels": _labels(),
    }

    out_dir = Path(out_root) / runinfo.new_run_id(started)
    shown = list(command) if command else ["python", "bench/scripts/sweeps.py", "--pool", pool_name]
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {"pareto_test": _figure(TEST, results), "pareto_calib": _figure(CALIB, results)},
        {"bootstrap": baselines.BOOTSTRAP_SEED, "random": 0, "mf": 0},
        shown,
        artifact.name,
        started,
    )
    set_lambda(policy_config, tuned)
    return out_dir


# --- report ------------------------------------------------------------------------------------


def _labels() -> list[str]:
    return [
        "costs are the recorded costs of each model on each item",
        "no latency is measured or simulated in this experiment",
        "operating points (the tuned lambda, the floor, the RouteLLM threshold) are chosen on "
        f"the calibration split only; test curves are labeled: {NOTE}",
        "the calibrators and lower-bound tables of the learned policy were fitted on the "
        "calibration split, so the learned policy's calibration success is in-sample and "
        "optimistic",
        "the RouteLLM MF scorer is trained on the train split; its threshold is chosen on "
        "calibration",
        "oracle looks at the recorded outcomes: an upper bound, not a policy",
    ]


def _sweep_table(entries: Sequence[Mapping[str, Any]], param: str, extra: str = "") -> pd.DataFrame:
    rows = []
    for e in entries:
        row: dict[str, Any] = {param: e["param"]}
        if extra:
            row[extra] = e[extra]
        for split in (CALIB, TEST):
            s = e[split]
            row |= {
                f"{split} success": s["success_rate"],
                f"{split} cost USD": s["mean_cost_usd"],
                f"{split} relaxed": s["relaxed_share"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _setup_table(results: Mapping[str, Any]) -> pd.DataFrame:
    target, tuning = results["target"], results["lambda_tuning"]
    boot, items, replays = results["bootstrap"], results["n_items"], results["replays"]
    rows = [
        ["pool", results["pool"]],
        ["artifact version", results["artifact_version"]],
        ["items (calibration / test)", f"{items[CALIB]} / {items[TEST]}"],
        ["Best Single (always_strongest)", results["best_single"]],
        ["cheapest (RouteLLM weak model)", results["cheapest"]],
        ["Best Single calibration success", target["best_single_calib_success"]],
        ["target Q_bar (that minus the margin)", f"{target['target']:.6g} ({target['margin']})"],
        ["tuned lambda (USD^-1, bisection on calibration)", repr(tuning["lambda"])],
        [
            "calibration success at the tuned lambda / at lambda 0",
            f"{tuning['calib_success']:.6g} / {tuning['calib_success_at_zero']:.6g}",
        ],
        ["policy config (other than lambda)", json.dumps(results["policy_config"])],
        ["RouteLLM MF weights SHA-256", results["mf"]["sha256"]],
        [
            "bootstrap",
            f"{boot['n']} resamples, percentile 95% intervals, seed {boot['seed']}",
        ],
        [
            "replays run (calibration / test)",
            f"{replays.get(CALIB, 0)} / {replays.get(TEST, 0)}",
        ],
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def _operating_table(results: Mapping[str, Any]) -> pd.DataFrame:
    ci = baselines._ci_text
    rows = []
    for op in results["operating_points"].values():
        r = op["test_row"]
        rows.append(
            {
                "policy": r["policy"],
                "parameter": f"{op['param_name']} = {op['param']!r}" if "param_name" in op else "",
                "calib success": op[CALIB]["success_rate"],
                "calib cost USD": op[CALIB]["mean_cost_usd"],
                "test success [95% CI]": ci(r["success_rate"], r["success_ci"]),
                "test cost USD [95% CI]": ci(r["mean_cost_usd"], r["mean_cost_ci"], ".6g"),
                "success delta [95% CI]": ci(
                    r["success_delta_vs_strongest"], r["success_delta_ci"]
                ),
                "cost reduction [95% CI]": ci(
                    r["cost_reduction_vs_strongest"], r["cost_reduction_ci"]
                ),
                "test relaxed": op[TEST]["relaxed_share"],
            }
        )
    return pd.DataFrame(rows)


def _baseline_table(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, calib in results["baselines"][CALIB].items():
        test = results["baselines"][TEST][name]
        rows.append(
            {
                "policy": name,
                "calib success": calib["success_rate"],
                "calib cost USD": calib["mean_cost_usd"],
                "test success": test["success_rate"],
                "test cost USD": test["mean_cost_usd"],
            }
        )
    return pd.DataFrame(rows)


def _tables(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    sweeps = results["sweeps"]
    return {
        "Setup": _setup_table(results),
        "Operating points (chosen on calibration; paired against always_strongest on test)": (
            _operating_table(results)
        ),
        "Static baselines": _baseline_table(results),
        f"Lambda sweep, learned dual ({NOTE})": _sweep_table(sweeps["lambda"], "lambda"),
        f"Floor sweep, learned min_cost ({NOTE}); relaxed is the share of items whose floor "
        "was relaxed": _sweep_table(sweeps["floor"], "floor"),
        f"RouteLLM MF threshold sweep ({NOTE})": _sweep_table(
            sweeps["threshold"], "threshold", "quantile"
        ),
    }


def _figure(split: str, results: Mapping[str, Any]) -> Figure:
    fig = Figure(figsize=(8.5, 5.5))
    ax = fig.subplots()
    curves = (
        ("learned dual (lambda sweep)", results["sweeps"]["lambda"], "tab:blue", "o"),
        ("learned min_cost (floor sweep)", results["sweeps"]["floor"], "tab:orange", "s"),
        ("RouteLLM MF (threshold sweep)", results["sweeps"]["threshold"], "tab:green", "^"),
    )
    for label, entries, color, marker in curves:
        ax.plot(
            [e[split]["mean_cost_usd"] for e in entries],
            [e[split]["success_rate"] for e in entries],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1,
            label=label,
        )
    base = results["baselines"][split]
    points = pd.DataFrame(
        {
            "policy": list(base),
            "mean_cost_usd": [s["mean_cost_usd"] for s in base.values()],
            "success_rate": [s["success_rate"] for s in base.values()],
        }
    )
    ax.scatter(
        points["mean_cost_usd"],
        points["success_rate"],
        color="0.35",
        zorder=3,
        label="static baselines",
    )
    for cost, success, label in baselines.merged_labels(points):
        ax.annotate(
            label,
            (cost, success),
            fontsize=6,
            color="0.25",
            xytext=(4, -9),
            textcoords="offset points",
        )
    for key, (_, param_name, color) in OPERATING.items():
        op = results["operating_points"][key]
        ax.scatter(
            op[split]["mean_cost_usd"],
            op[split]["success_rate"],
            marker="*",
            s=170,
            color=color,
            edgecolor="black",
            zorder=5,
            label=f"operating point: {param_name} = {op['param']:.4g}",
        )
    ax.axhline(
        results["target"]["target"],
        linestyle="--",
        color="0.6",
        linewidth=0.8,
        label="target Q_bar (Best Single on calibration minus 0.01)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per item (USD, recorded costs)")
    ax.set_ylabel("success rate")
    title = f"{split} split, {results['pool']} pool"
    if split == TEST:
        title += f"\n{NOTE}"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pareto sweeps of the learned policy.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--derived", type=Path, default=ROOT / "data" / "derived")
    parser.add_argument(
        "--artifact", type=Path, default=None, help="default: the one under policies/<pool>"
    )
    parser.add_argument(
        "--policy-config", type=Path, default=None, help="default: bench/policies/<pool>.yaml"
    )
    parser.add_argument("--mf", type=Path, default=None, help="default: <derived>/mf/<pool>.json")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench" / "results" / "sweeps")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0]).resolve()
    shown = script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name
    command = ["python", shown, *(argv if argv is not None else sys.argv[1:])]
    out_dir = run_sweeps(
        args.derived,
        args.pool,
        args.artifact or find_artifact(ROOT / "policies", args.pool),
        args.policy_config or ROOT / "bench" / "policies" / f"{args.pool}.yaml",
        args.mf or args.derived / "mf" / f"{args.pool}.json",
        args.out_root,
        command,
        workers=args.workers,
    )
    print(out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
