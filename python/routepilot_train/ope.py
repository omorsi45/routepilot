"""Off-policy evaluation and the offline promotion gate (spec 8).

A candidate routing policy is judged on what an exploring policy logged: for every item the
logging policy chose a model with a known probability (its propensity), and the recorded
outcome of that model is the reward. The estimators here answer "what would the target policy
have earned on these items?" without running it.

* ``ips``: inverse propensity scoring. The term of an item is ``r / propensity`` when the
  logged model is the one the target policy picks, else 0. Unbiased, high variance.
* ``snips``: the self-normalized version, which rescales the weights to sum to ``n``. Biased
  a little, much steadier.
* ``dr``: doubly robust. The target's predicted reward plus an IPS correction of the
  prediction error on the logged model. One outcome model, the logging policy's own
  predictions over all candidates (Ruling V1-R32): ``q(x, a)`` is the logged record's
  candidate ``p`` for model ``a`` (success) and ``cost_exp_picousd`` (cost).

Rewards are the success of the logged model (0 or 1) and its cost in picodollars (reported in
USD). The input is either a replay output (``routepilot replay --emit-candidates``) or the
gateway's decision log, both read by ``load_records``. Nothing here holds prompt text.

The promotion gate (``gate``) passes a candidate when its success is not below the incumbent's
by more than 0.005 at 95% confidence, both measured with DR on the same logged items, and its
DR cost is at most 1.02 times the incumbent's. The success comparison is made on the per-item
difference of the two DR terms (a paired bootstrap), not on each policy's own interval: one
policy's value is known to a few points on a calibration split, so an identical candidate
could never clear a 0.005 margin that way. ``decide`` reports the unpaired comparison as well.

Command line::

    python -m routepilot_train.ope gate --logged F --candidate F --incumbent F --out DIR

writes ``DIR/gate.json``, prints a one-line verdict and exits 0 when the candidate passes, 1
when it fails and 2 on an error.
"""

import argparse
import json
import math
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routepilot_train.evaluate.metrics import bootstrap_ci
from routepilot_train.logs import load_decisions

PICO = 1e12
# The promotion rule (spec 8): the candidate's DR lower bound on success must reach the
# incumbent's DR point estimate less SUCCESS_MARGIN, and its DR cost may be at most COST_RATIO
# times the incumbent's.
SUCCESS_MARGIN = 0.005
COST_RATIO = 1.02
# Rates and costs come from float arithmetic: a value exactly on a limit must pass.
_REL_TOL = 1e-12

# The reasons an item is left out of an estimate, in the order they are checked.
EXCLUSIONS = (
    "not_in_target",
    "target_error",
    "logged_error",
    "no_propensity",
    "no_reward",
    "no_q",
)
_RECORD_COLUMNS = [
    "item_id",
    "prompt_sha256_16",
    "model",
    "propensity",
    "success",
    "cost_picousd",
    "error",
    "candidates",
]
_LOGGED_COLUMNS = [
    "item_id",
    "model",
    "propensity",
    "success",
    "cost_picousd",
    "error",
    "candidates",
]
_TARGET_COLUMNS = ["item_id", "model", "error"]


def _arrays(
    r: Sequence[float] | np.ndarray,
    a_logged: Sequence[Any] | np.ndarray,
    propensity: Sequence[float] | np.ndarray,
    a_target: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(r, propensity, match)``; ``match`` is true where the two actions agree."""
    rewards = np.asarray(r, dtype=float)
    props = np.asarray(propensity, dtype=float)
    logged = np.asarray(a_logged, dtype=object)
    target = np.asarray(a_target, dtype=object)
    if rewards.ndim != 1 or not (rewards.shape == props.shape == logged.shape == target.shape):
        raise ValueError(
            "r, a_logged, propensity and a_target must be one-dimensional and the same length, "
            f"got {rewards.shape}, {logged.shape}, {props.shape}, {target.shape}"
        )
    if not np.all((props > 0) & (props <= 1)):  # NaN fails both comparisons
        raise ValueError("propensities must be in (0, 1]")
    return rewards, props, np.asarray(logged == target, dtype=bool)


def ips(
    r: Sequence[float] | np.ndarray,
    a_logged: Sequence[Any] | np.ndarray,
    propensity: Sequence[float] | np.ndarray,
    a_target: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """The inverse propensity terms: ``1[a_logged == a_target] / propensity * r`` per item.

    The estimate is the mean of the terms. ``propensity`` is the probability the logging
    policy had of the action it took.
    """
    rewards, props, match = _arrays(r, a_logged, propensity, a_target)
    return match / props * rewards


def snips(
    r: Sequence[float] | np.ndarray,
    a_logged: Sequence[Any] | np.ndarray,
    propensity: Sequence[float] | np.ndarray,
    a_target: Sequence[Any] | np.ndarray,
) -> float:
    """The self-normalized IPS estimate: the IPS weights rescaled to sum to ``n``.

    It is the weighted mean of the rewards of the items the target policy agrees on, so it
    always lies between the smallest and the largest of those rewards. NaN when the two
    policies never agree on an item (there is nothing to estimate from).
    """
    rewards, props, match = _arrays(r, a_logged, propensity, a_target)
    weights = match / props
    total = weights.sum()
    if not total > 0:
        return math.nan
    return float((weights * rewards).sum() / total)


def dr(
    r: Sequence[float] | np.ndarray,
    a_logged: Sequence[Any] | np.ndarray,
    propensity: Sequence[float] | np.ndarray,
    a_target: Sequence[Any] | np.ndarray,
    q_logged: Sequence[float] | np.ndarray,
    q_target: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """The doubly robust terms per item.

    The term is ``q_target + 1[a_logged == a_target] / propensity * (r - q_logged)``.
    ``q_logged`` is the outcome model's prediction for the logged action and ``q_target`` its
    prediction for the action the target policy takes. The estimate is the mean of the terms.
    It is exact when the outcome model is, and unbiased when the propensities are.
    """
    rewards, props, match = _arrays(r, a_logged, propensity, a_target)
    q_l = np.asarray(q_logged, dtype=float)
    q_t = np.asarray(q_target, dtype=float)
    if q_l.shape != rewards.shape or q_t.shape != rewards.shape:
        raise ValueError("q_logged and q_target must have the length of r")
    return q_t + match / props * (rewards - q_l)


def estimate(
    terms: Sequence[float] | np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """The mean of per-item terms with a 95% percentile bootstrap interval.

    Returns ``{"mean", "lo", "hi", "n"}``. The interval comes from ``n_boot`` resamples of the
    items with a fixed seed, so the same terms always give the same interval.
    """
    values = np.asarray(terms, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("estimate needs a non-empty one-dimensional array of terms")
    if not np.all(np.isfinite(values)):
        raise ValueError("estimate needs finite terms")
    mean, lo, hi = bootstrap_ci(values, np.mean, n=n_boot, seed=seed)
    return {"mean": mean, "lo": lo, "hi": hi, "n": int(values.size)}


def _is_replay(path: Path) -> bool:
    """True when the first record of ``path`` is a replay record (it has an ``item_id``)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                return isinstance(record, dict) and "item_id" in record
    return False


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=_RECORD_COLUMNS)
    for column in ("propensity", "success", "cost_picousd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["error"] = frame["error"].fillna("").astype(str)
    frame["candidates"] = frame["candidates"].astype(object)
    return frame


def _from_replay(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "item_id": rec["item_id"],
                    "prompt_sha256_16": rec.get("prompt_sha256_16"),
                    "model": rec.get("model"),
                    "propensity": rec.get("propensity"),
                    "success": rec.get("success"),
                    "cost_picousd": rec.get("cost_picousd"),
                    "error": _text(rec.get("error")),
                    "candidates": rec.get("candidates"),
                }
            )
    return _frame(rows)


def _decision_error(chosen: str, status: int | None) -> str:
    """The error of a logged decision: nothing chosen, or the request failed on the server."""
    if not chosen:
        return "no_choice"
    return f"http_{status}" if status is not None and status >= 500 else ""


def _from_decisions(path: str | Path) -> pd.DataFrame:
    rows = []
    for rec in load_decisions(path).to_dict("records"):
        chosen = _text(rec["chosen"])
        status = None if pd.isna(rec["status"]) else int(rec["status"])
        rows.append(
            {
                "item_id": rec["decision_id"],
                "prompt_sha256_16": rec["prompt_sha256_16"],
                "model": chosen or None,
                "propensity": rec["propensity"],
                # The reward of a served decision is the client's verdict, when it gave one.
                "success": None
                if pd.isna(rec["feedback_success"])
                else bool(rec["feedback_success"]),
                "cost_picousd": None
                if pd.isna(rec["cost_actual_picousd"])
                else int(rec["cost_actual_picousd"]),
                "error": _decision_error(chosen, status),
                "candidates": rec["candidates"] if isinstance(rec["candidates"], list) else None,
            }
        )
    return _frame(rows)


def load_records(path: str | Path) -> pd.DataFrame:
    """Read logged decisions or a replay output into one frame.

    The path is a replay JSONL file (``routepilot replay`` output; ``--emit-candidates`` adds
    the per-model scores the outcome model needs), or the gateway's decision log: a glob, a
    directory, or one log file. Either way the frame has one row per item with ``item_id``
    (the replay item, or the decision ID), ``prompt_sha256_16``, ``model`` (the action, None
    when nothing was chosen), ``propensity``, ``success`` and ``cost_picousd`` (floats, NaN
    when unknown), ``error`` (empty when routing succeeded) and ``candidates`` (the list of
    per-model dicts, or None).

    For a decision log the reward ``success`` is the client's feedback, so a decision nobody
    gave feedback on has no reward and is left out of an estimate.
    """
    target = Path(path)
    if target.is_file() and _is_replay(target):
        return _from_replay(target)
    return _from_decisions(path)


def _scored(candidates: Any) -> dict[str, tuple[float, float]]:
    """``model -> (p, cost_exp_picousd)`` for the candidates the policy scored."""
    found: dict[str, tuple[float, float]] = {}
    if not isinstance(candidates, list):
        return found
    for cand in candidates:
        p, cost = cand.get("p"), cand.get("cost_exp_picousd")
        if cand.get("model") is not None and p is not None and cost is not None:
            found[cand["model"]] = (float(p), float(cost))
    return found


def _require(frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {name} frame lacks the columns {missing}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"duplicate item_id in the {name} frame")


@dataclass(frozen=True)
class _Prepared:
    """The usable items of one target policy against the logged records, as arrays.

    ``q_success`` and ``q_cost`` are ``(logged, target)`` pairs: the outcome model's
    prediction for the model the logging policy took and for the one the target policy takes.
    """

    n_logged: int
    excluded: dict[str, int]
    item_ids: np.ndarray
    a_logged: np.ndarray
    a_target: np.ndarray
    props: np.ndarray
    success: np.ndarray
    cost: np.ndarray
    q_success: tuple[np.ndarray, np.ndarray]
    q_cost: tuple[np.ndarray, np.ndarray]

    @property
    def n_used(self) -> int:
        return int(self.item_ids.size)

    def success_dr(self) -> np.ndarray:
        return dr(self.success, self.a_logged, self.props, self.a_target, *self.q_success)

    def cost_dr(self) -> np.ndarray:
        return dr(self.cost, self.a_logged, self.props, self.a_target, *self.q_cost)


def _prepare(logged: pd.DataFrame, target: pd.DataFrame) -> _Prepared:
    """Join ``target`` onto ``logged`` and keep the items an estimate can use."""
    _require(logged, "logged", _LOGGED_COLUMNS)
    _require(target, "target", _TARGET_COLUMNS)
    tgt = target[_TARGET_COLUMNS].rename(columns={"model": "target_model", "error": "target_error"})
    rows = logged.merge(tgt, on="item_id", how="left", indicator=True).reset_index(drop=True)

    masks = {
        "not_in_target": rows["_merge"].eq("left_only"),
        "target_error": rows["target_model"].isna() | rows["target_error"].fillna("").ne(""),
        "logged_error": rows["model"].isna() | rows["error"].fillna("").ne(""),
        "no_propensity": ~(rows["propensity"] > 0) | ~(rows["propensity"] <= 1),
        "no_reward": rows["success"].isna() | rows["cost_picousd"].isna(),
    }
    reason = pd.Series("", index=rows.index, dtype=object)
    for name in EXCLUSIONS[:-1]:
        reason[(reason == "") & masks[name]] = name

    q = np.full((4, len(rows)), np.nan)  # success and cost: of the logged model, of the target's
    for i in np.flatnonzero(reason.to_numpy() == ""):
        scored = _scored(rows.at[i, "candidates"])
        own, wanted = scored.get(rows.at[i, "model"]), scored.get(rows.at[i, "target_model"])
        if own is None or wanted is None:
            reason.iat[i] = "no_q"
        else:
            q[:, i] = (own[0], own[1], wanted[0], wanted[1])

    excluded = {name: int((reason == name).sum()) for name in EXCLUSIONS}
    used = (reason == "").to_numpy()
    if not used.any():
        raise ValueError(f"no usable item among {len(rows)} logged ones: excluded {excluded}")
    return _Prepared(
        n_logged=len(rows),
        excluded=excluded,
        item_ids=rows.loc[used, "item_id"].to_numpy(dtype=object),
        a_logged=rows.loc[used, "model"].to_numpy(dtype=object),
        a_target=rows.loc[used, "target_model"].to_numpy(dtype=object),
        props=rows.loc[used, "propensity"].to_numpy(dtype=float),
        success=rows.loc[used, "success"].to_numpy(dtype=float),
        cost=rows.loc[used, "cost_picousd"].to_numpy(dtype=float),
        q_success=(q[0][used], q[2][used]),
        q_cost=(q[1][used], q[3][used]),
    )


def _summarize(p: _Prepared, n_boot: int, seed: int) -> dict[str, Any]:
    agree = p.a_logged == p.a_target
    weights = agree / p.props
    ess = float(weights.sum() ** 2 / (weights**2).sum()) if agree.any() else 0.0

    def usd(est: dict[str, Any]) -> dict[str, Any]:
        return {**est, "mean": est["mean"] / PICO, "lo": est["lo"] / PICO, "hi": est["hi"] / PICO}

    return {
        "n_logged": p.n_logged,
        "n_used": p.n_used,
        "excluded": p.excluded,
        "matches": int(agree.sum()),
        "effective_sample_size": ess,
        "logged_mean_success": float(p.success.mean()),
        "logged_mean_cost_usd": float(p.cost.mean() / PICO),
        "success": {
            "ips": estimate(ips(p.success, p.a_logged, p.props, p.a_target), n_boot, seed),
            "snips": snips(p.success, p.a_logged, p.props, p.a_target),
            "dr": estimate(p.success_dr(), n_boot, seed),
        },
        "cost_usd": {
            "ips": usd(estimate(ips(p.cost, p.a_logged, p.props, p.a_target), n_boot, seed)),
            "snips": snips(p.cost, p.a_logged, p.props, p.a_target) / PICO,
            "dr": usd(estimate(p.cost_dr(), n_boot, seed)),
        },
    }


def evaluate(
    logged: pd.DataFrame, target: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """What the target policy would have earned on the items ``logged`` was collected on.

    ``logged`` is the logging policy's records (with ``candidates``) and ``target`` the same
    items replayed with the policy under evaluation, joined on ``item_id``. An item is left
    out, and counted under ``excluded`` by the first reason that applies (see ``EXCLUSIONS``),
    when the target has no record for it or errored, the logging policy errored or has no
    propensity or no reward, or the outcome model has no prediction for the logged or the
    target model. Raises ValueError when no item is left.

    Returns the item counts, the effective sample size of the importance weights, and for
    success (a rate) and cost (USD per item) the IPS and DR estimates with intervals and the
    SNIPS estimate.
    """
    return _summarize(_prepare(logged, target), n_boot, seed)


def _paired_delta(
    candidate: _Prepared, incumbent: _Prepared, n_boot: int, seed: int
) -> dict[str, Any]:
    """The mean DR success of the candidate less the incumbent's, with a bootstrap interval.

    The two policies are scored on the same logged items, so the difference is taken item by
    item, over the items both could use, and the items are resampled together. What the two
    share (which items were easy, which model the logging policy took) cancels out of the
    difference instead of widening its interval.
    """
    cand = pd.Series(candidate.success_dr(), index=candidate.item_ids)
    inc = pd.Series(incumbent.success_dr(), index=incumbent.item_ids)
    common = cand.index.intersection(inc.index, sort=False)
    if common.empty:
        raise ValueError("the candidate and the incumbent have no usable item in common")
    return estimate(cand.loc[common].to_numpy() - inc.loc[common].to_numpy(), n_boot, seed)


def decide(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any], delta: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the promotion rule to two ``evaluate`` results and their paired difference.

    ``delta`` is the estimate (``mean``, ``lo``, ``hi``) of the candidate's DR success less the
    incumbent's over the same items. The candidate passes only when

    * success: the 95% lower bound of that difference is at least -0.005, which is the
      candidate's DR lower bound on success being at least the incumbent's DR point estimate
      less 0.005 with both measured on the same items, and
    * cost: the candidate's DR cost estimate is at most 1.02 times the incumbent's.

    A NaN fails. The same comparison made with each policy's own, unpaired, interval is
    reported next to it (``candidate_dr_lower`` against ``required``) but does not decide: an
    identical candidate fails it unless the logged sample is far larger than a calibration
    split, because each interval carries the whole sampling noise of one policy's value.
    """
    delta_lower = delta["lo"]
    success_ok = bool(delta_lower >= -SUCCESS_MARGIN - _REL_TOL)

    candidate_lower = candidate["success"]["dr"]["lo"]
    incumbent_point = incumbent["success"]["dr"]["mean"]
    required = incumbent_point - SUCCESS_MARGIN
    unpaired_ok = bool(candidate_lower >= required - _REL_TOL * max(1.0, abs(required)))

    candidate_cost = candidate["cost_usd"]["dr"]["mean"]
    incumbent_cost = incumbent["cost_usd"]["dr"]["mean"]
    limit = incumbent_cost * COST_RATIO
    cost_ok = bool(candidate_cost <= limit + _REL_TOL * abs(limit))
    return {
        "pass": success_ok and cost_ok,
        "rule": {"success_margin": SUCCESS_MARGIN, "cost_ratio": COST_RATIO, "interval": 0.95},
        "checks": {
            "success": {
                "paired_delta": dict(delta),
                "required_delta_lower": -SUCCESS_MARGIN,
                "candidate_dr_lower": candidate_lower,
                "incumbent_dr_point": incumbent_point,
                "required": required,
                "unpaired_pass": unpaired_ok,
                "pass": success_ok,
            },
            "cost": {
                "candidate_dr_usd": candidate_cost,
                "incumbent_dr_usd": incumbent_cost,
                "limit_usd": limit,
                "pass": cost_ok,
            },
        },
    }


def gate(
    logged: pd.DataFrame,
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The offline promotion gate (spec 8).

    ``logged`` holds the exploring logging policy's records with their candidate scores, and
    ``candidate`` and ``incumbent`` the deterministic replays of the two policies on the same
    items. Both are evaluated with the DR estimator on ``logged`` and the rule in ``decide``
    is applied. The result has the verdict (``pass``), the two checks and each policy's
    estimates (``candidate``, ``incumbent``).
    """
    cand, inc = _prepare(logged, candidate), _prepare(logged, incumbent)
    delta = _paired_delta(cand, inc, n_boot, seed)
    cand_est, inc_est = _summarize(cand, n_boot, seed), _summarize(inc, n_boot, seed)
    return {
        **decide(cand_est, inc_est, delta),
        "n_boot": n_boot,
        "seed": seed,
        "candidate": cand_est,
        "incumbent": inc_est,
    }


def _jsonable(value: Any) -> Any:
    """``value`` with numpy scalars as Python ones and non-finite floats as null."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verdict_line(result: Mapping[str, Any]) -> str:
    """One line for a terminal: the verdict and the numbers the rule compared."""
    success, cost = result["checks"]["success"], result["checks"]["cost"]
    delta = success["paired_delta"]
    cand, inc = result["candidate"], result["incumbent"]
    return (
        f"gate: {'PASS' if result['pass'] else 'FAIL'}"
        f" | success: paired DR delta {delta['mean']:+.4f}"
        f" (95% lower bound {delta['lo']:+.4f} vs {success['required_delta_lower']:+.3f},"
        f" {'ok' if success['pass'] else 'FAIL'});"
        f" candidate DR {cand['success']['dr']['mean']:.4f}"
        f" vs incumbent DR {success['incumbent_dr_point']:.4f}"
        f" | cost: candidate DR ${cost['candidate_dr_usd']:.8f}"
        f" vs {COST_RATIO} x incumbent ${cost['incumbent_dr_usd']:.8f}"
        f" = ${cost['limit_usd']:.8f} ({'ok' if cost['pass'] else 'FAIL'})"
        f" | items used: candidate {cand['n_used']}/{cand['n_logged']},"
        f" incumbent {inc['n_used']}/{inc['n_logged']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns 0 when the candidate passes, 1 when it fails, 2 on an error."""
    parser = argparse.ArgumentParser(prog="python -m routepilot_train.ope", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate_cmd = commands.add_parser(
        "gate", help="decide whether a candidate policy may replace the incumbent"
    )
    gate_cmd.add_argument("--logged", required=True, help="replay of the logging policy")
    gate_cmd.add_argument("--candidate", required=True, help="deterministic candidate replay")
    gate_cmd.add_argument("--incumbent", required=True, help="deterministic incumbent replay")
    gate_cmd.add_argument("--out", required=True, help="directory that receives gate.json")
    gate_cmd.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    gate_cmd.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args(argv)  # a usage error exits 2

    try:
        result = gate(
            load_records(args.logged),
            load_records(args.candidate),
            load_records(args.incumbent),
            n_boot=args.boot,
            seed=args.seed,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate.json").write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(verdict_line(result))
    except Exception as err:  # the exit status must tell an error (2) from a failed gate (1)
        traceback.print_exc()
        print(f"ope: error: {err}", file=sys.stderr)
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
