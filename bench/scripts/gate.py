"""The replay regression gate: replay a policy on the test split and compare with a baseline.

Run from the repository root:

    python bench/scripts/gate.py --check  --pool flagship --policy rules \\
        --config data/derived/pools/flagship.config.yaml \\
        --input data/derived/replay/flagship.jsonl --baseline bench/gates/baseline.json
    python bench/scripts/gate.py --update ...   # same flags: write a new baseline

``make bench-gate`` is the first (it needs the dataset) and ``make bench-smoke`` runs the same
check on the synthetic fixture under testdata/bench/gate/ (no dataset). The gate builds the Go
binary, replays the policy on the ``test`` split and compares two numbers with the baseline:
the success rate (it may fall by at most 0.01, absolute) and the mean cost per item in
picodollars (it may rise by at most 5%, relative). An improvement never fails. The baseline
also records the pool, policy, split and item count; a baseline for other data is a failure,
because its numbers cannot be compared.

Exit status: 0 on a pass, 1 when the gate fails, 2 for a bad argument or an input error.
Nothing printed holds prompt text.
"""

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routepilot_train import replaytool, runinfo

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "test"
MAX_SUCCESS_DROP = 0.01
MAX_COST_RISE = 0.05
# Rates and costs come from float divisions: a change of exactly the limit must pass.
_REL_TOL = 1e-9
_IDENTITY_KEYS = ("pool", "split", "policy", "items")
_UPDATE_KEYS = (
    "pool",
    "split",
    "policy",
    "items",
    "success_rate",
    "mean_cost_picousd",
    "git_sha",
    "updated_at",
)


def compare(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    max_success_drop: float = MAX_SUCCESS_DROP,
    max_cost_rise: float = MAX_COST_RISE,
) -> list[str]:
    """The ways ``current`` is worse than ``baseline``; an empty list means the gate passes.

    ``success_rate`` may fall by at most ``max_success_drop`` (absolute, 0.01 is one point)
    and ``mean_cost_picousd`` may rise by at most ``max_cost_rise`` (relative, 0.05 is 5%).
    Each message names the metric, both values and the limit. A ``pool``, ``split``,
    ``policy`` or ``items`` that differs between the two (when both hold it) is reported
    too, as their numbers do not measure the same thing.
    """
    problems = [
        f"{key} differs: current {current[key]!r}, baseline {baseline[key]!r}; "
        "the baseline is for other data, run --update to replace it"
        for key in _IDENTITY_KEYS
        if key in current and key in baseline and current[key] != baseline[key]
    ]

    base_rate, rate = float(baseline["success_rate"]), float(current["success_rate"])
    drop = base_rate - rate
    if drop > max_success_drop and not math.isclose(drop, max_success_drop, rel_tol=_REL_TOL):
        problems.append(
            f"success_rate fell from {base_rate:.6g} to {rate:.6g}, a drop of {drop:.6g}; "
            f"the limit is a drop of {max_success_drop:.6g}"
        )

    base_cost, cost = float(baseline["mean_cost_picousd"]), float(current["mean_cost_picousd"])
    ceiling = base_cost * (1.0 + max_cost_rise)
    if cost > ceiling and not math.isclose(cost, ceiling, rel_tol=_REL_TOL):
        rise = f"a rise of {(cost - base_cost) / base_cost:.2%}" if base_cost > 0 else "a rise"
        problems.append(
            f"mean_cost_picousd rose from {base_cost:.6g} to {cost:.6g}, {rise}; "
            f"the limit is a rise of {max_cost_rise:.0%}"
        )
    return problems


def measure(
    binary: Path, config: Path, replay_input: Path, policy: str, pool: str
) -> dict[str, Any]:
    """Replay ``policy`` on the test split and return the gated numbers."""
    with tempfile.TemporaryDirectory() as scratch:
        frame = replaytool.replay(
            binary, config, replay_input, SPLIT, policy, Path(scratch) / "gate.jsonl"
        )
    items = len(frame)
    return {
        "pool": pool,
        "split": SPLIT,
        "policy": policy,
        "items": items,
        "success_rate": int(frame["success"].sum()) / items,
        "mean_cost_picousd": int(frame["cost_picousd"].sum()) / items,
    }


def _summary(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    return (
        f"pool={current['pool']} policy={current['policy']} split={current['split']} "
        f"items={current['items']} "
        f"success_rate={current['success_rate']:.6g} (baseline {baseline['success_rate']:.6g}) "
        f"mean_cost_picousd={current['mean_cost_picousd']:.6g} "
        f"(baseline {baseline['mean_cost_picousd']:.6g})"
    )


def _write_baseline(path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    sha, _ = runinfo.git_state()
    updated = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    baseline = {**current, "git_sha": sha, "updated_at": updated}
    ordered = {key: baseline[key] for key in _UPDATE_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8", newline="\n")
    return ordered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay regression gate on the test split.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare with the baseline")
    mode.add_argument("--update", action="store_true", help="write the baseline")
    parser.add_argument("--pool", required=True, help="pool name recorded in the baseline")
    parser.add_argument("--policy", default="rules", help="replay policy to gate (default rules)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None, binary: Path | None = None) -> int:
    """Run the gate. ``binary`` is a prebuilt routepilot; by default it is built from the tree."""
    args = _parser().parse_args(argv)
    try:
        if args.check:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        binary = binary or replaytool.build_go(ROOT)
        current = measure(binary, args.config, args.input, args.policy, args.pool)
    except (OSError, ValueError, RuntimeError) as err:
        print(f"gate: {err}", file=sys.stderr)
        return 2

    if args.update:
        written = _write_baseline(args.baseline, current)
        print(f"gate updated {args.baseline.as_posix()}: {_summary(current, written)}")
        return 0

    problems = compare(current, baseline)
    if problems:
        print(f"gate FAILED: {_summary(current, baseline)}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"gate ok: {_summary(current, baseline)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
