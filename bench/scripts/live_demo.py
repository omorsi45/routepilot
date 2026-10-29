"""The live local experiment, part 3: serve ``auto`` on the live artifact and send real requests.

    python bench/scripts/live_demo.py

Starts ``routepilot serve`` with the learned policy of ``policies/live-ollama`` (min_cost,
``bench/policies/live-ollama.yaml``), the measured latency profiles of
``bench/latency/live-ollama.yaml`` as the latency priors, and the gateway's own online latency
estimator, which learns from the traffic it serves. It sends live test items (a seeded subset of
at most 120, one request at a time, the same items in the same order every round) in nine rounds:
``min_quality`` in {0, 0.4, 0.6} times ``slo_e2e_ms`` in {none, tight, loose}. Every request is
sent under the protocol the training data was collected with (its system message and its
``max_tokens``, read from ``protocol.json`` in the live directory). The tight SLO is
the fastest model's measured p50 end-to-end time and the loose SLO the slowest model's measured
p95, both over the train and calibration requests of ``live_collect.py``.

Per round it reports the mix of models that served, the constraints the policy relaxed, the
realized success (graded against the ground truth), the measured time to first token and
end-to-end time from the decision log, and whether the SLO was met. A round's number of items
is chosen from the measured speeds so that the demo fits its time budget (25 minutes by default).

Outputs stay in ``data/derived/live`` (raw answers, decision log); what is written under
``bench/results/live_demo/<run-id>`` holds counts, rates, latencies and item IDs only. Prices are
an ASSUMPTION (Ruling V1-R42); latency is measured.
"""

import argparse
import contextlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from routepilot_train import replaytool, runinfo
from routepilot_train.evaluate import report
from routepilot_train.logs import load_decisions

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_collect  # noqa: E402  (sibling scripts, not packages)
import live_train  # noqa: E402
import sweeps  # noqa: E402
from lcb_vs_point import wilson_interval  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "live_demo"
RESULTS_ROOT = ROOT / "bench" / "results" / "live_demo"
FLOORS = live_train.FLOORS
SLO_KINDS = ("none", "tight", "loose")
MAX_ITEMS = 120
BUDGET_MINUTES = 25.0
SEED = 0
MIN_ITEMS = 20

LABELS = (
    "latency is measured through RoutePilot on this machine (the decision log), never simulated",
    "prices are an ASSUMPTION (Ruling V1-R42); the policy chooses on its own predictions",
    "the tight SLO is the fastest model's measured p50 end-to-end time and the loose SLO the "
    "slowest model's measured p95, both over the train and calibration requests",
    "the gateway's online latency estimator starts from the measured profiles and learns from "
    "the demo's own traffic, so later rounds are not independent of earlier ones",
    "realized success is graded against the release's ground truth; a round has a few dozen "
    "items, so its intervals are wide",
    "no prompt text and no model output is in this report",
)


# --- settings ---------------------------------------------------------------------------------


def slo_values(
    measured: pd.DataFrame, splits: pd.DataFrame, models: Sequence[str]
) -> dict[str, Any]:
    """The tight and loose end-to-end SLOs in milliseconds, and the models they come from.

    Tight: the p50 of the fastest model's end-to-end time (the model with the lowest p50).
    Loose: the p95 of the slowest model's (the one with the highest p50). Only the train and
    calibration requests are read. Both are rounded up to whole milliseconds.
    """
    split_of = dict(zip(splits["item_id"], splits["split"], strict=True))
    tuning = measured.loc[measured["item_id"].map(split_of).isin(["train", "calib"])]
    stats = {}
    for model in models:
        values = tuning.loc[tuning["model"] == model, "e2e_ms"].dropna().to_numpy(dtype=float)
        if not len(values):
            raise ValueError(f"no train or calibration timings for {model}")
        stats[model] = (float(np.percentile(values, 50)), float(np.percentile(values, 95)))
    fastest = min(models, key=lambda m: (stats[m][0], m))
    slowest = max(models, key=lambda m: (stats[m][0], m))
    return {
        "tight_ms": int(math.ceil(stats[fastest][0])),
        "loose_ms": int(math.ceil(stats[slowest][1])),
        "tight_from": f"p50 of {fastest}",
        "loose_from": f"p95 of {slowest}",
        "p50_p95_ms": {m: list(v) for m, v in stats.items()},
    }


def demo_items(
    splits: pd.DataFrame,
    complete_items: Sequence[str],
    *,
    max_items: int = MAX_ITEMS,
    seed: int = SEED,
) -> list[str]:
    """A seeded subset (at most ``max_items``) of the live test items, in the order to send.

    Only items of the test split that every model answered are eligible, so the realized
    outcomes of the same items are on record for the offline replay.
    """
    test = set(splits.loc[splits["split"] == "test", "item_id"]) & set(complete_items)
    ids = sorted(test)
    if not ids:
        raise ValueError("no live test items")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))[: min(max_items, len(ids))]
    return [ids[i] for i in order]


def plan_items_per_round(
    mean_request_s: float, *, available: int, budget_s: float, rounds: int, cap: int = MAX_ITEMS
) -> int:
    """How many items each round sends so that all rounds fit the budget.

    ``mean_request_s`` is the typical request time; the result is at most ``cap`` and
    ``available``, and at least ``MIN_ITEMS`` unless fewer items exist.
    """
    if not mean_request_s > 0:
        raise ValueError("mean_request_s must be positive")
    fit = math.floor(budget_s / (rounds * mean_request_s))
    return min(available, cap, max(MIN_ITEMS, fit))


def round_settings(slos: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The nine rounds in the order they run: floors outside, SLO kinds inside."""
    ms = {"none": None, "tight": slos["tight_ms"], "loose": slos["loose_ms"]}
    return [
        {
            "round": f"q{floor:g}-{kind}",
            "min_quality": floor,
            "slo_kind": kind,
            "slo_e2e_ms": ms[kind],
        }
        for floor in FLOORS
        for kind in SLO_KINDS
    ]


def routepilot_options(setting: Mapping[str, Any]) -> dict[str, Any]:
    """The ``routepilot`` object of a request body for one round."""
    options: dict[str, Any] = {"min_quality": setting["min_quality"]}
    if setting["slo_e2e_ms"] is not None:
        options["slo_e2e_ms"] = int(setting["slo_e2e_ms"])
    return options


def demo_config(
    base: Mapping[str, Any],
    *,
    policies_root: Path,
    version: str,
    policy_path: Path,
    profiles_path: Path,
    overheads: Mapping[str, int],
) -> dict[str, Any]:
    """The live config with the learned policy, the measured profiles and the prompt overheads.

    Each model gets the ``prompt_overhead_tokens`` the pool was built with, so that the
    gateway bounds the input tokens as the artifact was trained to expect.
    """
    cfg = json.loads(json.dumps(base))
    cfg["routing"] = {"policy": "learned"}
    cfg["policies"] = {
        "root": Path(policies_root).resolve().as_posix(),
        "stable": {
            "name": live_collect.POOL,
            "version": version,
            "config": Path(policy_path).resolve().as_posix(),
        },
    }
    cfg["server"]["latency_profiles"] = Path(profiles_path).resolve().as_posix()
    for model in cfg["models"]:
        model["prompt_overhead_tokens"] = int(overheads.get(model["id"], 0))
    return cfg


# --- summaries --------------------------------------------------------------------------------


def _percentiles(values: pd.Series) -> dict[str, float] | None:
    arr = values.dropna().to_numpy(dtype=float)
    if not len(arr):
        return None
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def summarize_round(rows: pd.DataFrame, setting: Mapping[str, Any]) -> dict[str, Any]:
    """One round: the mix, the relaxations, the realized success, the latency, the SLO check.

    ``rows`` has one row per request with ``ok``, ``served`` (the model that answered),
    ``relaxed`` (a list), ``score`` (1.0 or 0.0, missing when the request failed),
    ``e2e_ms``, ``ttft_ms`` and ``completion_tokens``.
    """
    n = len(rows)
    ok = rows.loc[rows["ok"]]
    relaxed = ok["relaxed"].map(lambda v: list(v) if isinstance(v, list) else [])
    any_relaxed = relaxed.map(len) > 0
    scored = ok["score"].dropna()
    hits = int(scored.sum())
    interval = wilson_interval(hits, len(scored))
    kept = ok.loc[~any_relaxed.to_numpy()]
    kept_scored = kept["score"].dropna()
    slo = setting["slo_e2e_ms"]
    out: dict[str, Any] = {
        **{k: setting[k] for k in ("round", "min_quality", "slo_kind", "slo_e2e_ms")},
        "n": n,
        "ok": len(ok),
        "failed": n - len(ok),
        "mix": {str(m): int(c) for m, c in ok["served"].value_counts().sort_index().items()},
        "relaxed": {
            "any": int(any_relaxed.sum()),
            "latency": int(relaxed.map(lambda v: "latency" in v).sum()),
            "quality": int(relaxed.map(lambda v: "quality" in v).sum()),
        },
        "success": {
            "n": len(scored),
            "hits": hits,
            "rate": hits / len(scored) if len(scored) else None,
            "wilson_95": list(interval) if interval else None,
        },
        "success_not_relaxed": {
            "n": len(kept_scored),
            "rate": float(kept_scored.mean()) if len(kept_scored) else None,
        },
        "floor_met_not_relaxed": (
            bool(kept_scored.mean() >= setting["min_quality"]) if len(kept_scored) else None
        ),
        "e2e_ms": _percentiles(ok["e2e_ms"]),
        "ttft_ms": _percentiles(ok["ttft_ms"]),
        "completion_tokens_mean": float(ok["completion_tokens"].mean()) if len(ok) else None,
    }
    if slo is None:
        out["slo"] = None
    else:
        timed = ok["e2e_ms"].dropna()
        kept_timed = kept["e2e_ms"].dropna()
        out["slo"] = {
            "ms": int(slo),
            "met_share": float((timed <= slo).mean()) if len(timed) else None,
            "p95_within": bool(np.percentile(timed, 95) <= slo) if len(timed) else None,
            "not_relaxed_met_share": (
                float((kept_timed <= slo).mean()) if len(kept_timed) else None
            ),
        }
    return out


def estimator_bounds(decisions: pd.DataFrame, setting_of: Mapping[str, str]) -> dict[str, Any]:
    """Per round and model, the median end-to-end bound (``q_ms``) the policy used.

    ``setting_of`` maps a decision ID to its round name. A model with no bound (the policy
    computed none) is left out. Shows how the online estimator's view moves as it learns.
    """
    rows = []
    for decision_id, candidates in zip(
        decisions["decision_id"], decisions["candidates"], strict=True
    ):
        round_name = setting_of.get(decision_id)
        if round_name is None or not isinstance(candidates, list):
            continue
        for c in candidates:
            if c.get("q_ms") is not None:
                rows.append({"round": round_name, "model": c["model"], "q_ms": float(c["q_ms"])})
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for (round_name, model), g in frame.groupby(["round", "model"], sort=False):
        out.setdefault(str(round_name), {})[str(model)] = float(g["q_ms"].median())
    return out


# --- the run ----------------------------------------------------------------------------------


def run_demo(
    live_dir: Path,
    out_root: Path,
    *,
    config: Path = live_collect.CONFIG,
    policies_root: Path = ROOT / "policies",
    policy_path: Path = live_train.POLICY_PATH,
    profiles_path: Path = live_train.PROFILES_PATH,
    budget_minutes: float = BUDGET_MINUTES,
    max_items: int = MAX_ITEMS,
    items_per_round: int | None = None,
    binary: Path | None = None,
) -> Path:
    """Run the nine rounds and write the report under ``out_root/<run-id>``."""
    started = datetime.now(UTC)
    clock = time.perf_counter()
    live_dir = Path(live_dir).resolve()
    base = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    catalog = live_collect.load_catalog(config)
    protocol = live_collect.load_protocol(live_dir)  # the one the training data was collected with
    models = [c.id for c in catalog]
    prompts = pd.read_parquet(live_dir / "prompts.parquet")
    outcomes = pd.read_parquet(live_dir / "outcomes.parquet")
    measured = pd.read_parquet(live_dir / "measured.parquet")
    splits = pd.read_parquet(live_dir / "splits.parquet")
    truth = pd.read_parquet(live_dir / "ground_truth.parquet")
    pool = json.loads((live_dir / "pools" / f"{live_collect.POOL}.json").read_text("utf-8"))
    artifact = sweeps.find_artifact(Path(policies_root), live_collect.POOL)

    seen = outcomes.groupby("item_id")["model"].nunique()
    complete = list(seen.index[seen == len(models)])
    slos = slo_values(measured, splits, models)
    pool_items = demo_items(splits, complete, max_items=max_items)
    mean_s = float(np.mean([v[0] for v in slos["p50_p95_ms"].values()])) / 1000.0
    settings = round_settings(slos)
    per_round = items_per_round or plan_items_per_round(
        mean_s,
        available=len(pool_items),
        budget_s=budget_minutes * 60.0,
        rounds=len(settings),
        cap=max_items,
    )
    items = pool_items[:per_round]
    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))

    cfg = demo_config(
        base,
        policies_root=policies_root,
        version=artifact.name,
        policy_path=policy_path,
        profiles_path=profiles_path,
        overheads=pool.get("prompt_overhead_tokens", {}),
    )
    binary = binary or replaytool.build_go(ROOT)
    raw_path = live_dir / "demo-raw.jsonl"
    records: list[dict[str, Any]] = []
    warmups = []
    decision_ids: list[str] = []
    with live_collect.gateway(binary, cfg, live_dir, "demo") as (port, decision_log):
        with raw_path.open("w", encoding="utf-8", newline="\n") as sink:
            for c in catalog:
                live_collect.ollama_load(c.tag)
                warm = live_collect.stream_chat(
                    port,
                    live_collect.chat_body(c.id, live_collect.WARMUP_PROMPT, protocol.max_tokens),
                    key=live_collect.KEY,
                    timeout_s=300,
                )
                warmups.append({"model": c.id, "ok": warm.ok, "wall_ms": warm.wall_ms})
                if warm.headers.get("x-routepilot-decision-id"):
                    decision_ids.append(warm.headers["x-routepilot-decision-id"])
            for setting in settings:
                options = routepilot_options(setting)
                for item_id in items:
                    chat = live_collect.stream_chat(
                        port,
                        protocol.body("auto", prompt_of[item_id], options),
                        key=live_collect.KEY,
                        timeout_s=300,
                    )
                    decision_id = chat.headers.get("x-routepilot-decision-id", "")
                    relaxed = chat.headers.get("x-routepilot-relaxed", "")
                    record = {
                        "round": setting["round"],
                        "item_id": item_id,
                        "decision_id": decision_id,
                        "served": chat.headers.get("x-routepilot-model", ""),
                        "relaxed": [r for r in relaxed.split(",") if r],
                        "text": chat.text,
                        "finish_reason": chat.finish_reason,
                        "prompt_tokens": chat.prompt_tokens,
                        "completion_tokens": chat.completion_tokens,
                        "ok": chat.ok,
                        "error": chat.error,
                        "client_wall_ms": chat.wall_ms,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    records.append(record)
                    if decision_id:
                        decision_ids.append(decision_id)
        missing = live_collect.wait_for_decisions(decision_log, decision_ids)
    for c in catalog:
        with contextlib.suppress(OSError):  # best-effort cleanup of video memory
            live_collect.ollama_unload(c.tag)

    decisions = load_decisions(live_dir / "decisions-demo*.jsonl")
    raw = pd.DataFrame(records)
    graded = live_collect.grade_requests(raw.loc[raw["ok"]], prompts, truth)
    raw = raw.merge(graded[["round", "item_id", "score"]], on=["round", "item_id"], how="left")
    logged = decisions[["decision_id", "status", "ttft_ms", "latency_ms", "relaxed", "chosen"]]
    joined = raw.merge(
        logged.rename(columns={"relaxed": "relaxed_logged", "latency_ms": "e2e_ms"}),
        on="decision_id",
        how="left",
        validate="m:1",
    )
    joined["relaxed"] = [
        seen_in_log if isinstance(seen_in_log, list) else header
        for seen_in_log, header in zip(joined["relaxed_logged"], joined["relaxed"], strict=True)
    ]
    joined["ok"] = joined["ok"] & joined["status"].eq(200).fillna(False).astype(bool)
    rounds = [summarize_round(joined.loc[joined["round"] == s["round"]], s) for s in settings]
    setting_of = dict(zip(joined["decision_id"], joined["round"], strict=True))
    diverged = int((joined["ok"] & (joined["served"] != joined["chosen"])).sum())

    results: dict[str, Any] = {
        "protocol": protocol.as_dict(),
        "models": models,
        "artifact_version": artifact.name,
        "policy_config": yaml.safe_load(Path(policy_path).read_text(encoding="utf-8")),
        "slo": {k: slos[k] for k in ("tight_ms", "loose_ms", "tight_from", "loose_from")},
        "e2e_ms_p50_p95_train_calib": slos["p50_p95_ms"],
        "items_per_round": len(items),
        "test_items_available": len(pool_items),
        "budget_minutes": budget_minutes,
        "rounds": rounds,
        "requests": len(raw),
        "served_differs_from_chosen": diverged,
        "decisions_missing": len(missing),
        "warmups": warmups,
        "estimator_bound_ms_by_round": estimator_bounds(decisions, setting_of),
        "wall_time_s": time.perf_counter() - clock,
        "labels": list(LABELS),
    }
    out_dir = Path(out_root) / runinfo.new_run_id(started)
    command = ["python", "bench/scripts/live_demo.py"]
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        _tables(results),
        {},
        {"items": SEED},
        command,
        artifact.name,
        started,
        measurement={"protocol": protocol.as_dict()},
    )
    return out_dir


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _tables(results: Mapping[str, Any]) -> dict[str, Any]:
    slo = results["slo"]
    protocol = results["protocol"]
    setup = [
        ["models", ", ".join(results["models"])],
        ["artifact version", results["artifact_version"]],
        ["protocol", protocol["name"]],
        ["system message", protocol["system_message"] or "none"],
        ["max_tokens", protocol["max_tokens"]],
        ["policy config", json.dumps(results["policy_config"])],
        ["tight SLO (ms)", f"{slo['tight_ms']} ({slo['tight_from']})"],
        ["loose SLO (ms)", f"{slo['loose_ms']} ({slo['loose_from']})"],
        [
            "items per round / test items available",
            f"{results['items_per_round']} / {results['test_items_available']}",
        ],
        ["requests", results["requests"]],
        [
            "served model differs from the policy's pick (failover)",
            results["served_differs_from_chosen"],
        ],
        ["decision records missing", results["decisions_missing"]],
        ["wall time (s)", round(results["wall_time_s"], 1)],
    ]
    main = []
    for r in results["rounds"]:
        s = r["success"]
        e2e = r["e2e_ms"] or {}
        ttft = r["ttft_ms"] or {}
        slo_r = r["slo"]
        main.append(
            {
                "round": r["round"],
                "min_quality": r["min_quality"],
                "slo_e2e_ms": r["slo_e2e_ms"] if r["slo_e2e_ms"] is not None else "none",
                "n (failed)": f"{r['n']} ({r['failed']})",
                "model mix": json.dumps(r["mix"]),
                "relaxed any / latency / quality": (
                    f"{r['relaxed']['any']} / {r['relaxed']['latency']} / {r['relaxed']['quality']}"
                ),
                "success [Wilson 95%]": (
                    "n/a"
                    if s["rate"] is None
                    else f"{s['rate']:.3f} [{s['wilson_95'][0]:.3f}, {s['wilson_95'][1]:.3f}]"
                ),
                "success when not relaxed (n)": (
                    "n/a"
                    if r["success_not_relaxed"]["rate"] is None
                    else f"{r['success_not_relaxed']['rate']:.3f} ({r['success_not_relaxed']['n']})"
                ),
                "TTFT ms p50": ttft.get("p50"),
                "e2e ms p50 / p95": ("n/a" if not e2e else f"{e2e['p50']:.0f} / {e2e['p95']:.0f}"),
                "SLO met (share of requests)": (
                    "no SLO" if slo_r is None else _pct(slo_r["met_share"])
                ),
                "p95 within SLO": "no SLO" if slo_r is None else slo_r["p95_within"],
            }
        )
    bounds = results["estimator_bound_ms_by_round"]
    bound_rows = [{"round": name, **per} for name, per in bounds.items()]
    tables: dict[str, Any] = {
        "Setup": pd.DataFrame(setup, columns=["setting", "value"]),
        "Rounds (min_quality x slo_e2e_ms), measured through RoutePilot": pd.DataFrame(main),
    }
    if bound_rows:
        tables["Median end-to-end bound the policy used (q_ms), per model and round"] = (
            pd.DataFrame(bound_rows)
        )
    return tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live-dir", type=Path, default=live_collect.LIVE_DIR)
    parser.add_argument("--policies-root", type=Path, default=ROOT / "policies")
    parser.add_argument("--policy-path", type=Path, default=live_train.POLICY_PATH)
    parser.add_argument("--profiles-path", type=Path, default=live_train.PROFILES_PATH)
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--budget-minutes", type=float, default=BUDGET_MINUTES)
    parser.add_argument("--max-items", type=int, default=MAX_ITEMS)
    parser.add_argument("--items-per-round", type=int, default=None, help="override the plan")
    args = parser.parse_args(argv)
    out = run_demo(
        args.live_dir,
        args.out_root,
        policies_root=args.policies_root,
        policy_path=args.policy_path,
        profiles_path=args.profiles_path,
        budget_minutes=args.budget_minutes,
        max_items=args.max_items,
        items_per_round=args.items_per_round,
    )
    print(out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
