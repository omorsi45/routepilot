"""The live local experiment, part 2: split the live items, train the policy, profile the models.

    python bench/scripts/live_train.py

Reads what ``live_collect.py collect`` wrote under ``--live-dir`` (default ``data/derived/live``;
the collection's protocol is recorded in the report) and

1. splits the live items with the same salt rule as the main experiment (60/20/20 over the item
   IDs, no held-out dataset);
2. builds the pool ``live-ollama`` (the three models, Best Single and cheapest from the train and
   calibration items only, the ASSUMPTION prices of Ruling V1-R42), with its replay config and
   replay input;
3. trains the policy with the usual pipeline at dimension 16384 and exports the artifact to
   ``policies/live-ollama/<version>/`` (it must stay under 2 MB);
4. writes the measured latency profiles ``bench/latency/live-ollama.yaml`` (medians and spreads of
   the time to first token and the time per output token of the train and calibration requests,
   as the gateway measured them) and the policy config ``bench/policies/live-ollama.yaml``;
5. replays the learned policy and the baselines on the live test split, offline and on the
   recorded outcomes, with the measured end-to-end time of each chosen model on each item (from
   the decision log, not simulated). The learned rows use the floors 0, 0.4 and 0.6 that the live
   demo uses, fixed in advance: nothing is chosen on the test split.

Writes ``bench/results/live_train/<run-id>/{results.json,report.md,run.json}``. The output holds
counts, rates, latencies and model names only, never prompt text.
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

from routepilot_train import replaytool, runinfo
from routepilot_train.baselines.static_rules import RULES, map_rules
from routepilot_train.data import splits as splits_mod
from routepilot_train.data.export import export_replay
from routepilot_train.data.pools import (
    build_pool,
    pool_outcomes,
    prompt_overheads,
    write_pool_config,
)
from routepilot_train.evaluate import report
from routepilot_train.train import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402  (sibling scripts, not packages)
import live_collect  # noqa: E402
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "live_train"
DIM = 16384
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
FLOORS = (0.0, 0.4, 0.6)  # the live demo's min_quality values, fixed before anything is trained
POLICY_PATH = ROOT / "bench" / "policies" / "live-ollama.yaml"
PROFILES_PATH = ROOT / "bench" / "latency" / "live-ollama.yaml"
RESULTS_ROOT = ROOT / "bench" / "results" / "live_train"
GPU = "AMD Radeon RX 9060 XT"
PRICE_SOURCE = (
    "assumed (ASSUMPTION, Ruling V1-R42): 0.02 / 0.06 USD per Mtok per billion parameters"
)

POLICY_YAML = """\
# Learned routing policy for the live local experiment (plan Task 33).
#
# min_cost: pick the cheapest model whose quality lower bound clears the floor. The floor
# comes from the request (routepilot.min_quality) or the tenant; with none the default floor 0
# lets the cheapest model answer. A request whose floor no model clears is relaxed
# (best_effort) and answered by the model with the best lower bound.
mode: min_cost
default_floor: 0
epsilon: 0
ood_action: conservative
on_infeasible: best_effort
"""

LABELS = (
    "prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per "
    "million input tokens and 0.06 per million output tokens per billion parameters",
    "latency is measured through RoutePilot on this machine (the decision log), never simulated "
    "or assumed",
    "the learned rows' floors (0, 0.4, 0.6) are fixed in advance; the test split is replayed "
    "once per row and nothing is chosen on it",
    "the test split is small, so the intervals are wide",
    "no prompt text and no model output is in this report",
)


# --- splits, pool, tables ---------------------------------------------------------------------


def live_splits(items: pd.DataFrame) -> pd.DataFrame:
    """``item_id``, ``dataset`` and ``split`` (train, calib or test) for every live item."""
    unique = items[["item_id", "dataset"]].drop_duplicates().sort_values("item_id")
    out = unique.reset_index(drop=True)
    out["split"] = [splits_mod.place(i) for i in out["item_id"]]
    return out


def build_live_pool(
    outcomes: pd.DataFrame,
    prompts: pd.DataFrame,
    splits: pd.DataFrame,
    catalog: Sequence[live_collect.CatalogModel],
) -> dict[str, Any]:
    """The pool of the catalog models over the items every one of them answered.

    Best Single, cheapest and the ranks use the train and calibration items only. The prices
    are the ASSUMPTION prices of the catalog, not a fit: the recorded costs were computed from
    them.
    """
    models = [c.id for c in catalog]
    rows = pool_outcomes(outcomes, models)
    if rows.empty:
        raise ValueError("no item has an outcome from every model")
    pool = build_pool(live_collect.POOL, rows, splits)
    if set(pool["models"]) != set(models):
        raise ValueError(f"the pool holds {pool['models']}, the catalog {sorted(models)}")
    pool["prices"] = {
        c.id: {
            "in_usd_per_token": float(c.in_usd_per_mtok / 1_000_000),
            "out_usd_per_token": float(c.out_usd_per_mtok / 1_000_000),
            "fit_rmse_usd": 0.0,
            "fit_r2": 1.0,
            "price_source": PRICE_SOURCE,
        }
        for c in catalog
    }
    pool["prompt_overhead_tokens"], pool["anomalies"] = prompt_overheads(
        rows, prompts, splits, models
    )
    return pool


def write_live_tables(
    live_dir: Path,
    pool: Mapping[str, Any],
    outcomes: pd.DataFrame,
    prompts: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[str, int]:
    """Write ``splits.parquet``, the pool JSON and replay config, and the replay input.

    The static rules are mapped on the train and calibration items only. Returns the number
    of replay lines.
    """
    live_dir = Path(live_dir)
    rows = pool_outcomes(outcomes, pool["models"])
    items = prompts[["item_id", "prompt", "prompt_bytes"]].merge(
        splits[["item_id", "split"]], on="item_id", validate="1:1"
    )
    in_pool = items["item_id"].isin(set(rows["item_id"]))
    tuning = items.loc[in_pool & items["split"].isin(["train", "calib"])]
    rules = map_rules(tuning, rows, RULES, best_single=pool["best_single"])

    splits.to_parquet(live_dir / "splits.parquet", index=False, compression="zstd")
    (live_dir / "pools").mkdir(parents=True, exist_ok=True)
    (live_dir / "pools" / f"{pool['pool']}.json").write_text(
        json.dumps(pool, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_pool_config(
        pool,
        live_dir / "pools" / f"{pool['pool']}.config.yaml",
        rules=rules,
        default_model=pool["best_single"],
    )
    lines = export_replay(
        pool, rows, prompts, splits, live_dir / "replay" / f"{pool['pool']}.jsonl"
    )
    return {"replay_lines": lines}


# --- latency profiles -------------------------------------------------------------------------


def _log_stats(values: np.ndarray) -> tuple[float, float]:
    """The median of ``values`` and the standard deviation of their logarithms."""
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        raise ValueError("need at least two positive latency values per model")
    return float(np.median(values)), float(np.std(np.log(values), ddof=1))


def latency_profiles(measured: pd.DataFrame, splits: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per model, the medians and log spreads of TTFT and TPOT over train and calibration.

    TTFT is the decision log's ``ttft_ms``. TPOT is ``(e2e_ms - ttft_ms) / completion_tokens``,
    the quantity the gateway's own estimator records. Test items are never read.
    """
    split_of = dict(zip(splits["item_id"], splits["split"], strict=True))
    tuning = measured.loc[measured["item_id"].map(split_of).isin(["train", "calib"])]
    out: dict[str, dict[str, Any]] = {}
    for model, rows in tuning.groupby("model"):
        tpot = (rows["e2e_ms"] - rows["ttft_ms"]) / rows["completion_tokens"].where(
            rows["completion_tokens"] > 0
        )
        ttft_median, ttft_sigma = _log_stats(rows["ttft_ms"].to_numpy(dtype=float))
        tpot_median, tpot_sigma = _log_stats(tpot.to_numpy(dtype=float))
        out[str(model)] = {
            "n": len(rows),
            "ttft_median_ms": ttft_median,
            "ttft_sigma": ttft_sigma,
            "tpot_median_ms": tpot_median,
            "tpot_sigma": tpot_sigma,
        }
    return out


def write_profiles(
    path: Path, profiles: Mapping[str, Mapping[str, Any]], machine: str, date: str
) -> None:
    """Write the profile file the gateway's ``server.latency_profiles`` reads (source measured)."""
    entries = [
        {
            "model": model,
            "ttft_median_ms": round(float(p["ttft_median_ms"]), 4),
            "ttft_sigma": round(float(p["ttft_sigma"]), 6),
            "tpot_median_ms": round(float(p["tpot_median_ms"]), 4),
            "tpot_sigma": round(float(p["tpot_sigma"]), 6),
            "source": "measured",
            "retrieved": date,
            "note": f"measured through RoutePilot on {machine}, {date}",
        }
        for model, p in sorted(profiles.items())
    ]
    header = (
        "# Latency profiles of the live local experiment, measured through RoutePilot.\n"
        "# Medians and spreads (standard deviation of the logarithm) of the time to first token\n"
        "# and of the time per output token, from the train and calibration requests of\n"
        "# live_collect.py, as the gateway's decision log recorded them.\n"
        "# Not simulated and not assumed.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"profiles": entries}, sort_keys=False, width=1000)
    path.write_text(header + body, encoding="utf-8", newline="\n")


def machine_name(gpu: str) -> str:
    """The CPU and GPU the measurements ran on."""
    cpu = runinfo._cpu_name()
    return f"{cpu} and {gpu}"


# --- the replay table -------------------------------------------------------------------------


def measured_latency(frame: pd.DataFrame, measured: pd.DataFrame) -> dict[str, float | None]:
    """p50 and p95 of the measured end-to-end time of each row's chosen model on its item."""
    picked = frame.loc[frame["model"].notna(), ["item_id", "model"]]
    joined = picked.merge(measured[["item_id", "model", "e2e_ms"]], on=["item_id", "model"])
    values = joined["e2e_ms"].dropna().to_numpy(dtype=float)
    if not len(values):
        return {"e2e_ms_p50": None, "e2e_ms_p95": None, "n_timed": 0}
    return {
        "e2e_ms_p50": float(np.percentile(values, 50)),
        "e2e_ms_p95": float(np.percentile(values, 95)),
        "n_timed": int(len(values)),
    }


def replay_rows(
    named_frames: Mapping[str, pd.DataFrame], measured: pd.DataFrame, bootstrap: int
) -> list[dict[str, Any]]:
    """One row per policy: success, assumed cost, measured e2e, with paired bootstrap intervals."""
    reference = named_frames[baselines.REFERENCE]
    rows = []
    for name, frame in named_frames.items():
        row = baselines._policy_row(name, frame, reference, bootstrap)
        row.update(measured_latency(frame, measured))
        row["relaxed_share"] = float(
            frame["relaxed"].map(lambda v: len(v) > 0 if hasattr(v, "__len__") else False).mean()
        )
        row["ood_share"] = float(frame["ood"].astype(bool).mean())
        rows.append(row)
    return rows


def _replay_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    ci = baselines._ci_text
    return pd.DataFrame(
        [
            {
                "policy": r["policy"],
                "n": r["n"],
                "success [95% CI]": ci(r["success_rate"], r["success_ci"]),
                "mean cost USD (assumed) [95% CI]": ci(
                    r["mean_cost_usd"], r["mean_cost_ci"], ".6g"
                ),
                "e2e ms p50 (measured)": r["e2e_ms_p50"],
                "e2e ms p95 (measured)": r["e2e_ms_p95"],
                "relaxed share": r["relaxed_share"],
                "OOD share": r["ood_share"],
                "model mix": json.dumps(r["model_counts"]),
            }
            for r in rows
        ]
    )


# --- the run ----------------------------------------------------------------------------------


def artifact_size(directory: Path) -> int:
    """Total bytes of the files in an artifact directory."""
    return sum(f.stat().st_size for f in Path(directory).iterdir() if f.is_file())


def run_live_train(
    live_dir: Path,
    policies_out: Path,
    out_root: Path,
    *,
    config: Path = live_collect.CONFIG,
    policy_path: Path = POLICY_PATH,
    profiles_path: Path = PROFILES_PATH,
    gpu: str = GPU,
    dim: int = DIM,
    bootstrap: int = baselines.BOOTSTRAP,
    workers: int = sweeps.WORKERS,
    binary: Path | None = None,
) -> Path:
    """Run the whole step and return the results directory."""
    started = datetime.now(UTC)
    clock = time.perf_counter()
    live_dir = Path(live_dir).resolve()
    catalog = live_collect.load_catalog(config)
    protocol = live_collect.load_protocol(live_dir)
    outcomes = pd.read_parquet(live_dir / "outcomes.parquet")
    prompts = pd.read_parquet(live_dir / "prompts.parquet")
    measured_all = pd.read_parquet(live_dir / "measured.parquet")

    splits = live_splits(outcomes)
    pool = build_live_pool(outcomes, prompts, splits, catalog)
    tables = write_live_tables(live_dir, pool, outcomes, prompts, splits)
    complete = pool_outcomes(outcomes, pool["models"])["item_id"].unique()
    per_split = splits.loc[splits["item_id"].isin(complete), "split"].value_counts().to_dict()

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(POLICY_YAML, encoding="utf-8", newline="\n")

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        summary = pipeline.run(
            live_collect.POOL, live_dir, Path(policies_out), dim, results_root=scratch / "train"
        )
        artifact = Path(summary["artifact_dir"])
        size = artifact_size(artifact)
        if size >= MAX_ARTIFACT_BYTES:
            raise ValueError(f"the artifact is {size} bytes, the limit is {MAX_ARTIFACT_BYTES}")
        fit = json.loads((Path(summary["results_dir"]) / "results.json").read_text("utf-8"))

        profiles = latency_profiles(measured_all, splits)
        date = started.strftime("%Y-%m-%d")
        write_profiles(profiles_path, profiles, machine_name(gpu), date)

        binary = binary or replaytool.build_go(ROOT)
        config_path = live_dir / "pools" / f"{live_collect.POOL}.config.yaml"
        replay_input = live_dir / "replay" / f"{live_collect.POOL}.jsonl"
        learned = ["--artifact", str(artifact), "--policy-config", str(policy_path)]
        rp = sweeps.Replayer(binary, config_path, replay_input, scratch, workers)
        try:
            specs = baselines.policy_specs(pool)
            jobs: list[tuple[str, str, Sequence[str]]] = [
                (sweeps.TEST, policy, extra) for _, policy, extra in specs
            ]
            jobs += [(sweeps.TEST, "learned", learned + sweeps.floor_flags(f)) for f in FLOORS]
            frames = rp.many(jobs)
        finally:
            rp.close()

    names = [name for name, _, _ in specs] + [f"learned_min_cost_floor_{f:g}" for f in FLOORS]
    named = dict(zip(names, frames, strict=True))
    test_ids = set(next(iter(named.values()))["item_id"])
    measured_test = measured_all.loc[measured_all["item_id"].isin(test_ids)]
    rows = replay_rows(named, measured_test, bootstrap)
    learned_summaries = {
        name: sweeps.summarize_point(named[name], sweeps.TEST)
        for name in names
        if name.startswith("learned_")
    }

    results: dict[str, Any] = {
        "protocol": protocol.as_dict(),
        "pool": {
            k: pool[k]
            for k in ("pool", "models", "datasets", "n_items", "best_single", "cheapest", "rank")
        }
        | {
            "prices": {
                m: {
                    "in_usd_per_mtok": p["in_usd_per_token"] * 1e6,
                    "out_usd_per_mtok": p["out_usd_per_token"] * 1e6,
                    "price_source": p["price_source"],
                }
                for m, p in pool["prices"].items()
            },
            "prompt_overhead_tokens": pool["prompt_overhead_tokens"],
        },
        "items_per_split": per_split,
        "static_rules_replay_lines": tables["replay_lines"],
        "artifact": {"version": summary["version"], "dim": dim, "size_bytes": size},
        "fit": {
            "calibrator": fit["calibrator"],
            "n_train": fit["n_train"],
            "n_calib": fit["n_calib"],
            "ood": fit["ood"],
            "models": fit["models"],
        },
        "profiles": profiles,
        "profiles_machine": machine_name(gpu),
        "policy_config": yaml.safe_load(POLICY_YAML),
        "floors": list(FLOORS),
        "replay": {
            "split": sweeps.TEST,
            "n_items": len(test_ids),
            "bootstrap": {"n": bootstrap, "seed": baselines.BOOTSTRAP_SEED},
            "rows": rows,
            "learned": learned_summaries,
        },
        "wall_time_s": time.perf_counter() - clock,
        "labels": list(LABELS),
    }
    out_dir = Path(out_root) / runinfo.new_run_id(started)
    command = ["python", "bench/scripts/live_train.py"]
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results, rows),
        {},
        {"bootstrap": baselines.BOOTSTRAP_SEED, "random": 0},
        command,
        summary["version"],
        started,
        measurement={"protocol": protocol.as_dict()},
    )
    return out_dir


def _tables(results: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pool, artifact, fit = results["pool"], results["artifact"], results["fit"]
    protocol = results["protocol"]
    setup = [
        ["models", ", ".join(pool["models"])],
        ["collection protocol", protocol["name"]],
        ["system message", protocol["system_message"] or "none"],
        ["max_tokens", protocol["max_tokens"]],
        ["Best Single (train+calib) / cheapest", f"{pool['best_single']} / {pool['cheapest']}"],
        ["items every model answered", pool["n_items"]],
        ["items per split", json.dumps(results["items_per_split"], sort_keys=True)],
        [
            "artifact version / dim / bytes",
            f"{artifact['version']} / {artifact['dim']} / {artifact['size_bytes']}",
        ],
        ["calibrator", fit["calibrator"]],
        ["OOD k / threshold", f"{fit['ood']['k']} / {fit['ood']['threshold']:.6g}"],
        ["policy config", json.dumps(results["policy_config"])],
        ["latency profiles measured on", results["profiles_machine"]],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    prices = pd.DataFrame(
        [
            {
                "model": m,
                "input USD per Mtok (ASSUMPTION)": p["in_usd_per_mtok"],
                "output USD per Mtok (ASSUMPTION)": p["out_usd_per_mtok"],
                "prompt overhead tokens": pool["prompt_overhead_tokens"][m],
            }
            for m, p in pool["prices"].items()
        ]
    )
    fit_table = pd.DataFrame(
        [{"model": m, **{k: v for k, v in row.items()}} for m, row in fit["models"].items()]
    )
    profile_table = pd.DataFrame(
        [
            {
                "model": m,
                "train+calib requests": p["n"],
                "TTFT median ms": p["ttft_median_ms"],
                "TTFT sigma": p["ttft_sigma"],
                "TPOT median ms": p["tpot_median_ms"],
                "TPOT sigma": p["tpot_sigma"],
            }
            for m, p in results["profiles"].items()
        ]
    )
    return {
        "Setup": pd.DataFrame(setup, columns=["setting", "value"]),
        "Assumed prices": prices,
        "Per-model fit (calibration columns are in-sample for the calibrator)": fit_table,
        "Measured latency profiles (train and calibration requests)": profile_table,
        "Live test split: offline replay on recorded outcomes, measured latency": _replay_table(
            rows
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live-dir", type=Path, default=live_collect.LIVE_DIR)
    parser.add_argument("--policies-out", type=Path, default=ROOT / "policies")
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--policy-path", type=Path, default=POLICY_PATH)
    parser.add_argument("--profiles-path", type=Path, default=PROFILES_PATH)
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--workers", type=int, default=sweeps.WORKERS)
    args = parser.parse_args(argv)
    out = run_live_train(
        args.live_dir,
        args.policies_out,
        args.out_root,
        policy_path=args.policy_path,
        profiles_path=args.profiles_path,
        gpu=args.gpu,
        workers=args.workers,
    )
    print(out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
