import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from routepilot_train import ope

N_ACTIONS = 4


def mu(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The true success probability of action ``a`` in context ``x`` (always in [0.6, 0.94])."""
    return 0.6 + 0.08 * a + 0.1 * x * (a % 2)


def world(n: int, seed: int) -> dict[str, np.ndarray]:
    """Contexts, a uniform logging policy over four actions, and a deterministic target."""
    g = np.random.default_rng(seed)
    x = g.uniform(size=n)
    a_logged = g.integers(0, N_ACTIONS, size=n)
    r = (g.uniform(size=n) < mu(x, a_logged)).astype(float)
    a_target = np.where(x < 0.5, 3, 1)
    return {
        "x": x,
        "r": r,
        "a_logged": a_logged,
        "a_target": a_target,
        "propensity": np.full(n, 1 / N_ACTIONS),
        "truth": np.full(n, mu(x, a_target).mean()),
    }


# ---- the estimators ----


def test_ips_unbiased_on_uniform_logging() -> None:
    # Exactly: log every action of every context once, uniformly. The mean IPS term is then
    # the mean reward of the target's action, with no sampling error at all.
    g = np.random.default_rng(1)
    x = g.uniform(size=50)
    target = np.where(x < 0.5, 3, 1)
    full_r = (g.uniform(size=(50, N_ACTIONS)) < mu(x[:, None], np.arange(N_ACTIONS))).astype(float)
    xs = np.repeat(x, N_ACTIONS)
    logged = np.tile(np.arange(N_ACTIONS), 50)
    terms = ope.ips(
        full_r.reshape(-1), logged, np.full(xs.size, 1 / N_ACTIONS), np.repeat(target, N_ACTIONS)
    )
    assert terms.mean() == pytest.approx(full_r[np.arange(50), target].mean(), abs=1e-12)

    # And on sampled data: within four standard errors of the true value of the target.
    w = world(200_000, seed=2)
    terms = ope.ips(w["r"], w["a_logged"], w["propensity"], w["a_target"])
    stderr = terms.std(ddof=1) / math.sqrt(terms.size)
    assert abs(terms.mean() - w["truth"][0]) < 4 * stderr
    # The per-item term is r / propensity where the actions agree, 0 elsewhere.
    agree = w["a_logged"] == w["a_target"]
    assert np.array_equal(terms[~agree], np.zeros((~agree).sum()))
    assert np.allclose(terms[agree], w["r"][agree] * N_ACTIONS)


def test_ips_hand_example() -> None:
    terms = ope.ips([1, 0, 1, 1], ["a", "b", "a", "a"], [0.5, 0.5, 0.25, 0.5], ["a", "a", "a", "b"])
    assert terms.tolist() == [2.0, 0.0, 4.0, 0.0]


def test_snips_normalizes() -> None:
    a_logged = ["a", "a", "a", "b", "a"]
    a_target = ["a", "a", "a", "a", "b"]
    props = [0.5, 0.25, 0.1, 0.5, 0.5]
    # A constant reward is estimated exactly by the self-normalized estimator, while the
    # plain IPS mean is off by the sum of the weights over n.
    r = [0.7] * 5
    assert ope.snips(r, a_logged, props, a_target) == pytest.approx(0.7)
    assert ope.ips(r, a_logged, props, a_target).mean() != pytest.approx(0.7)

    # The weights are scaled to sum to n, and the estimate is their mean against the reward.
    r = [1.0, 0.0, 1.0, 1.0, 0.0]
    weights = np.array([1 / 0.5, 1 / 0.25, 1 / 0.1, 0.0, 0.0])
    scaled = weights * len(r) / weights.sum()
    assert scaled.sum() == pytest.approx(len(r))
    assert ope.snips(r, a_logged, props, a_target) == pytest.approx((scaled * np.array(r)).mean())

    # Hand example: the weights of the agreeing items are 2 and 4, the rewards 1 and 0.
    assert ope.snips(
        [1, 0, 0, 1], ["a", "b", "a", "a"], [0.5, 0.5, 0.25, 0.5], ["a", "a", "a", "b"]
    ) == (pytest.approx(2 / 6))

    # It does not depend on the scale of the weights (IPS does).
    assert ope.snips(r, a_logged, [p / 2 for p in props], a_target) == pytest.approx(
        ope.snips(r, a_logged, props, a_target)
    )

    # It stays inside the rewards it averages, however large the weights.
    g = np.random.default_rng(3)
    rewards = g.uniform(0.2, 0.8, size=500)
    estimate = ope.snips(
        rewards, g.integers(0, 2, 500), g.uniform(0.01, 0.5, 500), g.integers(0, 2, 500)
    )
    assert 0.2 <= estimate <= 0.8


def test_snips_without_overlap_is_nan() -> None:
    assert math.isnan(ope.snips([1, 0], ["a", "a"], [0.5, 0.5], ["b", "b"]))


def test_dr_reduces_variance() -> None:
    ips_means, dr_means, truths = [], [], []
    for seed in range(300):
        w = world(400, seed)
        # An imperfect outcome model: right up to a constant offset of 0.03.
        q_logged = mu(w["x"], w["a_logged"]) + 0.03
        q_target = mu(w["x"], w["a_target"]) + 0.03
        ips_means.append(ope.ips(w["r"], w["a_logged"], w["propensity"], w["a_target"]).mean())
        dr_means.append(
            ope.dr(w["r"], w["a_logged"], w["propensity"], w["a_target"], q_logged, q_target).mean()
        )
        truths.append(w["truth"][0])
    ips_err, dr_err = np.array(ips_means) - truths, np.array(dr_means) - truths
    assert np.var(dr_err) < 0.5 * np.var(ips_err)
    # Both are unbiased: the mean error is a few standard errors from zero at most.
    for err in (ips_err, dr_err):
        assert abs(err.mean()) < 4 * err.std(ddof=1) / math.sqrt(err.size)


def test_dr_with_exact_predictions_is_the_target_prediction() -> None:
    g = np.random.default_rng(4)
    r = g.uniform(size=50)
    a_logged, a_target = g.integers(0, 3, 50), g.integers(0, 3, 50)
    q_target = g.uniform(size=50)
    terms = ope.dr(r, a_logged, g.uniform(0.1, 1, 50), a_target, q_logged=r, q_target=q_target)
    assert np.allclose(terms, q_target)


def test_dr_adds_the_weighted_residual_where_the_policies_agree() -> None:
    terms = ope.dr(
        r=[1.0, 0.0, 1.0],
        a_logged=["a", "b", "a"],
        propensity=[0.5, 0.5, 0.25],
        a_target=["a", "a", "b"],
        q_logged=[0.75, 0.25, 0.5],
        q_target=[0.5, 0.5, 0.125],
    )
    assert terms.tolist() == [0.5 + 2.0 * 0.25, 0.5, 0.125]


@pytest.mark.parametrize(
    "r, a_logged, propensity, a_target",
    [
        ([1, 0], ["a"], [0.5, 0.5], ["a", "a"]),  # lengths differ
        ([1, 0], ["a", "a"], [0.5, 0.0], ["a", "a"]),  # a zero propensity
        ([1, 0], ["a", "a"], [0.5, 1.5], ["a", "a"]),  # more than one
        ([1, 0], ["a", "a"], [0.5, math.nan], ["a", "a"]),  # not a number
        ([[1, 0]], [["a", "a"]], [[0.5, 0.5]], [["a", "a"]]),  # two-dimensional
    ],
)
def test_estimators_refuse_bad_input(r: Any, a_logged: Any, propensity: Any, a_target: Any) -> None:
    with pytest.raises(ValueError):
        ope.ips(r, a_logged, propensity, a_target)
    with pytest.raises(ValueError):
        ope.snips(r, a_logged, propensity, a_target)


def test_estimate_gives_the_mean_and_a_stable_interval() -> None:
    terms = np.random.default_rng(5).normal(0.8, 1.0, size=2000)
    est = ope.estimate(terms, n_boot=500, seed=7)
    assert est["n"] == 2000
    assert est["mean"] == pytest.approx(terms.mean())
    assert est["lo"] < est["mean"] < est["hi"]
    half = 1.96 * terms.std(ddof=1) / math.sqrt(terms.size)
    assert est["hi"] - est["lo"] == pytest.approx(2 * half, rel=0.15)
    assert ope.estimate(terms, n_boot=500, seed=7) == est  # the seed fixes the interval
    assert ope.estimate(terms, n_boot=500, seed=8) != est
    assert ope.estimate(terms[:200], n_boot=500, seed=7)["hi"] - ope.estimate(terms[:200], 500, 7)[
        "lo"
    ] > (est["hi"] - est["lo"])  # fewer items, a wider interval


def test_estimate_of_constant_terms_is_a_point() -> None:
    est = ope.estimate([0.5] * 10)
    assert est["lo"] == est["mean"] == est["hi"] == 0.5


@pytest.mark.parametrize("terms", [[], [[1.0, 2.0]], [1.0, math.nan], [1.0, math.inf]])
def test_estimate_refuses_bad_terms(terms: Any) -> None:
    with pytest.raises(ValueError):
        ope.estimate(terms)


# ---- reading logged records ----

CANDIDATES = [
    {
        "model": "good",
        "p": 1.0,
        "l": 0.9,
        "cost_exp_picousd": 100,
        "cost_max_picousd": 200,
        "feasible": True,
    },
    {
        "model": "bad",
        "p": 0.0,
        "l": 0.0,
        "cost_exp_picousd": 10,
        "cost_max_picousd": 20,
        "feasible": True,
    },
    {
        "model": "off",
        "p": None,
        "l": None,
        "cost_exp_picousd": None,
        "cost_max_picousd": None,
        "feasible": False,
    },
    {  # scored for success but with no cost estimate: the outcome model is incomplete
        "model": "partial",
        "p": 0.5,
        "l": 0.4,
        "cost_exp_picousd": None,
        "cost_max_picousd": None,
        "feasible": True,
    },
]


def replay_record(item: str, model: str | None, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "item_id": item,
        "dataset": "d",
        "split": "calib",
        "prompt_sha256_16": "0123456789abcdef",
        "policy": "learned",
        "model": model,
        "success": model == "good",
        "cost_picousd": {"good": 100, "bad": 10}.get(model or "", 0),
        "propensity": 0.5 if model else None,
        "error": "" if model else "no_feasible_model",
        "candidates": CANDIDATES,
    }
    record.update(extra)
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
    return path


def test_load_records_reads_a_replay(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "logged.jsonl",
        [
            replay_record("a/1", "good"),
            replay_record("a/2", None),
            replay_record("a/3", "bad", candidates=None),
        ],
    )
    frame = ope.load_records(path)
    assert frame["item_id"].tolist() == ["a/1", "a/2", "a/3"]
    assert frame["model"].tolist()[0] == "good" and frame["model"].isna().tolist() == [
        False,
        True,
        False,
    ]
    assert frame["error"].tolist() == ["", "no_feasible_model", ""]
    assert frame["propensity"].tolist()[0] == 0.5 and math.isnan(frame["propensity"].tolist()[1])
    assert frame["success"].tolist() == [1.0, 0.0, 0.0]
    assert frame["cost_picousd"].tolist() == [100.0, 0.0, 10.0]
    assert frame["candidates"].iloc[0] == CANDIDATES and frame["candidates"].iloc[2] is None
    assert not any("prompt" in c and c != "prompt_sha256_16" for c in frame.columns)


def decision_line(decision_id: str, chosen: str, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "decision",
        "decision_id": decision_id,
        "time": "2026-10-28T12:00:00Z",
        "tenant": "t",
        "policy_version": "v+1",
        "policy_role": "stable",
        "prompt_sha256_16": "fedcba9876543210",
        "prompt_bytes": 10,
        "candidates": CANDIDATES,
        "chosen": chosen,
        "propensity": 0.5,
        "status": 200,
        "cost_actual_picousd": 100,
    }
    record.update(extra)
    return record


def test_load_records_reads_a_decision_log(tmp_path: Path) -> None:
    log = write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            decision_line("dec_a", "good"),
            decision_line("dec_b", "bad"),
            decision_line("dec_c", "", propensity=None, status=422),
            decision_line("dec_d", "good", status=503),
            {
                "type": "feedback",
                "decision_id": "dec_a",
                "tenant": "t",
                "success": True,
                "time": "2026-10-28T12:01:00Z",
            },
            {
                "type": "feedback",
                "decision_id": "dec_b",
                "tenant": "t",
                "success": False,
                "time": "2026-10-28T12:01:00Z",
            },
        ],
    )
    for source in (log, tmp_path, str(tmp_path / "decisions*.jsonl")):
        frame = ope.load_records(source).set_index("item_id")
        assert sorted(frame.index) == ["dec_a", "dec_b", "dec_c", "dec_d"]
        assert frame.loc["dec_a", "model"] == "good" and frame.loc["dec_a", "propensity"] == 0.5
        # The reward is the client's verdict; no verdict, no reward.
        assert frame.loc["dec_a", "success"] == 1.0 and frame.loc["dec_b", "success"] == 0.0
        assert math.isnan(frame.loc["dec_d", "success"])
        assert frame.loc["dec_a", "cost_picousd"] == 100.0
        assert frame.loc["dec_a", "error"] == ""
        assert frame.loc["dec_c", "error"] != "" and pd.isna(frame.loc["dec_c", "model"])
        assert frame.loc["dec_d", "error"] == "http_503"
        assert frame.loc["dec_a", "candidates"] == CANDIDATES


# ---- evaluating a target policy on logged records ----


def write_world(
    tmp_path: Path, logged: list[dict[str, Any]], target: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        ope.load_records(write_jsonl(tmp_path / "logged.jsonl", logged)),
        ope.load_records(write_jsonl(tmp_path / "target.jsonl", target)),
    )


def target_record(item: str, model: str | None) -> dict[str, Any]:
    return replay_record(item, model, propensity=1.0 if model else None, candidates=None)


def test_evaluate_counts_what_it_leaves_out(tmp_path: Path) -> None:
    logged = [
        replay_record("i/0", "good"),  # used
        replay_record("i/1", "good"),  # the target has no record of it
        replay_record("i/2", "good"),  # the target errored
        replay_record("i/3", None),  # the logging policy errored
        replay_record("i/4", "good", propensity=None),  # no propensity
        replay_record("i/5", "good", success=None),  # no reward
        replay_record("i/6", "good"),  # the target picks a model the candidates do not score
        replay_record("i/7", "bad"),  # used
        replay_record(
            "i/8", None
        ),  # logging policy errored AND the target errored: first reason wins
        replay_record("i/9", "good"),  # the target picks a model scored for success only
    ]
    target = [
        target_record("i/0", "good"),
        target_record("i/2", None),
        target_record("i/3", "good"),
        target_record("i/4", "good"),
        target_record("i/5", "good"),
        target_record("i/6", "off"),
        target_record("i/7", "good"),
        target_record("i/8", None),
        target_record("i/9", "partial"),
    ]
    lg, tg = write_world(tmp_path, logged, target)
    result = ope.evaluate(lg, tg, n_boot=50)
    assert result["n_logged"] == 10
    assert result["n_used"] == 2
    assert result["excluded"] == {
        "not_in_target": 1,
        "target_error": 2,
        "logged_error": 1,
        "no_propensity": 1,
        "no_reward": 1,
        "no_q": 2,
    }
    assert sum(result["excluded"].values()) + result["n_used"] == result["n_logged"]
    assert result["matches"] == 1  # only i/0 logged what the target picks


def test_evaluate_with_exact_predictions_is_the_target_value(tmp_path: Path) -> None:
    # Perfect predictions: "good" always succeeds at 100 pico, "bad" never at 10. DR is then the
    # target's prediction, whatever the logged action was.
    logged = [replay_record(f"i/{i}", ["good", "bad"][i % 2]) for i in range(40)]
    target = [target_record(f"i/{i}", "good") for i in range(40)]
    lg, tg = write_world(tmp_path, logged, target)
    result = ope.evaluate(lg, tg, n_boot=100)
    assert result["success"]["dr"]["mean"] == pytest.approx(1.0)
    assert result["success"]["dr"]["lo"] == result["success"]["dr"]["hi"] == pytest.approx(1.0)
    assert result["cost_usd"]["dr"]["mean"] == pytest.approx(100 / 1e12)
    assert result["cost_usd"]["snips"] == pytest.approx(100 / 1e12)
    assert result["cost_usd"]["ips"]["mean"] == pytest.approx(100 / 1e12)
    assert result["success"]["snips"] == pytest.approx(1.0)  # the 20 items logged "good"
    assert result["success"]["ips"]["mean"] == pytest.approx(1.0)  # half the items, weight 2
    assert result["effective_sample_size"] == pytest.approx(20.0)
    assert result["logged_mean_success"] == pytest.approx(0.5)
    assert result["logged_mean_cost_usd"] == pytest.approx(55 / 1e12)


def test_evaluate_from_a_decision_log(tmp_path: Path) -> None:
    log = write_jsonl(
        tmp_path / "decisions.jsonl",
        [decision_line(f"dec_{i}", ["good", "bad"][i % 2]) for i in range(10)]
        + [
            {
                "type": "feedback",
                "decision_id": f"dec_{i}",
                "tenant": "t",
                "success": i % 2 == 0,
                "time": "2026-10-28T12:01:00Z",
            }
            for i in range(10)
        ],
    )
    target = write_jsonl(
        tmp_path / "t.jsonl", [target_record(f"dec_{i}", "good") for i in range(10)]
    )
    result = ope.evaluate(ope.load_records(log), ope.load_records(target), n_boot=20)
    assert result["n_used"] == 10
    assert result["success"]["dr"]["mean"] == pytest.approx(1.0)


def test_evaluate_refuses_nothing_to_estimate(tmp_path: Path) -> None:
    lg, tg = write_world(tmp_path, [replay_record("i/0", "good")], [target_record("i/0", None)])
    with pytest.raises(ValueError, match="no usable item"):
        ope.evaluate(lg, tg)


def test_evaluate_refuses_duplicate_items(tmp_path: Path) -> None:
    lg, tg = write_world(
        tmp_path,
        [replay_record("i/0", "good"), replay_record("i/0", "bad")],
        [target_record("i/0", "good")],
    )
    with pytest.raises(ValueError, match="duplicate item_id"):
        ope.evaluate(lg, tg)
    with pytest.raises(ValueError, match="duplicate item_id"):
        ope.evaluate(tg, lg)
    with pytest.raises(ValueError, match="lacks the columns"):
        ope.evaluate(lg.drop(columns=["propensity"]).drop_duplicates("item_id"), tg)


# ---- the promotion gate ----


def evaluation(lower: float, mean: float, cost: float) -> dict[str, Any]:
    """Just the numbers the promotion rule reads from an ``evaluate`` result."""
    return {
        "success": {"dr": {"lo": lower, "mean": mean}},
        "cost_usd": {"dr": {"mean": cost}},
    }


def paired(lower: float) -> dict[str, Any]:
    """A paired success difference whose 95% lower bound is ``lower``."""
    return {"mean": lower + 0.01, "lo": lower, "hi": lower + 0.02, "n": 100}


def test_gate_rules() -> None:
    incumbent = evaluation(0.80, 0.85, 0.0010)
    candidate = evaluation(0.84, 0.86, 0.0010)

    def check(candidate: dict[str, Any], delta_lower: float = 0.0) -> dict[str, Any]:
        return ope.decide(candidate, incumbent, paired(delta_lower))

    # The success rule: the lower bound of the candidate's DR success less the incumbent's, on
    # the same items, must be at least -0.005.
    assert check(candidate, 0.02)["pass"]
    assert check(candidate, 0.0)["pass"]  # an identical policy
    assert check(candidate, -0.004)["pass"]
    assert check(candidate, -0.005)["pass"]  # exactly on the limit
    assert not check(candidate, -0.0051)["pass"]
    low = check(candidate, -0.02)
    assert (
        not low["pass"] and not low["checks"]["success"]["pass"] and low["checks"]["cost"]["pass"]
    )
    assert not check(candidate, math.nan)["pass"]  # a missing number never passes

    # The unpaired view (each policy's own interval) is reported but does not decide.
    unpaired_fails = check(evaluation(0.80, 0.85, 0.0010), 0.0)
    assert unpaired_fails["pass"] and not unpaired_fails["checks"]["success"]["unpaired_pass"]
    unpaired_passes = check(evaluation(0.85, 0.90, 0.0010), -0.02)
    assert not unpaired_passes["pass"] and unpaired_passes["checks"]["success"]["unpaired_pass"]
    assert check(evaluation(0.845, 0.86, 0.0010))["checks"]["success"]["unpaired_pass"]  # on it
    assert not check(evaluation(0.8449, 0.95, 0.0010))["checks"]["success"]["unpaired_pass"]

    # The cost rule: the candidate's DR cost against 1.02 times the incumbent's.
    assert check(evaluation(0.90, 0.95, 0.0009))["pass"]  # cheaper
    assert check(evaluation(0.90, 0.95, 0.00102))["pass"]  # exactly 1.02 times
    pricey = check(evaluation(0.90, 0.95, 0.0010201))
    assert (
        not pricey["pass"]
        and pricey["checks"]["success"]["pass"]
        and not pricey["checks"]["cost"]["pass"]
    )
    assert not check(evaluation(0.9, 0.9, math.nan))["pass"]

    # Both can fail at once.
    both = check(evaluation(0.5, 0.6, 0.01), -0.3)
    assert (
        not both["pass"]
        and not both["checks"]["success"]["pass"]
        and not both["checks"]["cost"]["pass"]
    )

    # The verdict carries the numbers it compared.
    shown = check(evaluation(0.84, 0.9, 0.0011), -0.01)["checks"]
    assert shown["success"] == {
        "paired_delta": paired(-0.01),
        "required_delta_lower": -0.005,
        "candidate_dr_lower": 0.84,
        "incumbent_dr_point": 0.85,
        "required": pytest.approx(0.845),
        "unpaired_pass": False,
        "pass": False,
    }
    assert shown["cost"]["limit_usd"] == pytest.approx(0.00102)


MODELS = {  # model -> (success, cost in picodollars)
    "good": (True, 100),
    "cheap": (True, 50),
    "bad": (False, 10),
    "pricey": (True, 103),
    "edge": (True, 102),
}
# The logging policy scores this model like the others but never picks it, so a target that
# picks it has no item to learn from by importance weighting.
UNLOGGED = ("never", True, 100)


def scored_candidates() -> list[dict[str, Any]]:
    models = {**MODELS, UNLOGGED[0]: UNLOGGED[1:]}
    return [
        {
            "model": m,
            "p": float(ok),
            "l": float(ok),
            "cost_exp_picousd": cost,
            "cost_max_picousd": cost,
            "feasible": True,
        }
        for m, (ok, cost) in models.items()
    ]


def logged_rows(n: int) -> list[dict[str, Any]]:
    """A logging policy uniform over the five models, whose predictions get every outcome right."""
    names = list(MODELS)
    return [
        {
            "item_id": f"i/{i}",
            "model": names[i % 5],
            "propensity": 0.2,
            "success": MODELS[names[i % 5]][0],
            "cost_picousd": MODELS[names[i % 5]][1],
            "error": "",
            "candidates": scored_candidates(),
        }
        for i in range(n)
    ]


def policy_files(tmp_path: Path, n: int = 500) -> dict[str, pd.DataFrame]:
    files = {"logged": ope.load_records(write_jsonl(tmp_path / "logged.jsonl", logged_rows(n)))}
    for name in [*MODELS, UNLOGGED[0]]:
        rows = [target_record(f"i/{i}", name) for i in range(n)]
        files[name] = ope.load_records(write_jsonl(tmp_path / f"{name}.jsonl", rows))
    return files


@pytest.mark.parametrize(
    "candidate, passes, success_ok, cost_ok",
    [
        ("good", True, True, True),  # identical to the incumbent
        ("cheap", True, True, True),  # as good and cheaper
        ("bad", False, False, True),  # cheaper but never succeeds
        ("pricey", False, True, False),  # 103 against 1.02 x 100 = 102
        ("edge", True, True, True),  # 102, exactly on the limit
        ("never", True, True, True),  # no item to weight, but the predictions carry it (DR)
    ],
)
def test_gate_decides_from_logged_data(
    tmp_path: Path, candidate: str, passes: bool, success_ok: bool, cost_ok: bool
) -> None:
    files = policy_files(tmp_path)
    result = ope.gate(files["logged"], files[candidate], files["good"], n_boot=50)
    assert result["pass"] is passes
    assert result["checks"]["success"]["pass"] is success_ok
    assert result["checks"]["cost"]["pass"] is cost_ok
    assert result["candidate"]["n_used"] == result["incumbent"]["n_used"] == 500
    assert result["incumbent"]["success"]["dr"]["mean"] == pytest.approx(1.0)
    if candidate == "never":
        assert result["candidate"]["matches"] == 0
        assert result["candidate"]["effective_sample_size"] == 0.0
        assert math.isnan(result["candidate"]["success"]["snips"])


def noisy_world(
    tmp_path: Path, n: int = 3000, seed: int = 0
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Models with different success rates, drawn at random: every estimate is noisy.

    The logging policy is uniform over A, B and C (propensity 1/3). Its predictions are the true
    success rates, so DR is unbiased, but each outcome is a coin flip and a policy's value has
    a sampling error of a few points, far wider than the gate's 0.005 margin.
    """
    rates = {"A": 0.60, "B": 0.50, "C": 0.70}
    g = np.random.default_rng(seed)
    scored = [
        {
            "model": m,
            "p": p,
            "l": p,
            "cost_exp_picousd": 100,
            "cost_max_picousd": 100,
            "feasible": True,
        }
        for m, p in rates.items()
    ]
    names = list(rates)
    logged = []
    for i in range(n):
        model = names[g.integers(0, 3)]
        logged.append(
            {
                "item_id": f"i/{i}",
                "model": model,
                "propensity": 1 / 3,
                "success": bool(g.uniform() < rates[model]),
                "cost_picousd": 100,
                "error": "",
                "candidates": scored,
            }
        )
    frames = {
        name: ope.load_records(
            write_jsonl(
                tmp_path / f"{name}.jsonl", [target_record(f"i/{i}", name) for i in range(n)]
            )
        )
        for name in names
    }
    return ope.load_records(write_jsonl(tmp_path / "logged.jsonl", logged)), frames


def test_gate_compares_policies_on_the_same_items(tmp_path: Path) -> None:
    logged, policies = noisy_world(tmp_path)

    # An identical candidate passes, although its own interval is wide: the difference between
    # two policies scored on the same items is exactly zero.
    same = ope.gate(logged, policies["A"], policies["A"], n_boot=200)
    success = same["checks"]["success"]
    assert same["pass"] and success["pass"]
    assert success["paired_delta"]["mean"] == success["paired_delta"]["lo"] == 0.0
    # The unpaired comparison the spec words literally does not pass it: one policy's value
    # is only known to about +-0.04 here, against a margin of 0.005.
    assert not success["unpaired_pass"]
    assert success["candidate_dr_lower"] < success["required"] - 0.02

    # A candidate that is really 0.10 worse fails, one that is 0.10 better passes.
    worse = ope.gate(logged, policies["B"], policies["A"], n_boot=200)
    assert not worse["pass"] and not worse["checks"]["success"]["pass"]
    assert worse["checks"]["success"]["paired_delta"]["hi"] < -0.02
    better = ope.gate(logged, policies["C"], policies["A"], n_boot=200)
    assert better["pass"]
    assert better["checks"]["success"]["paired_delta"]["lo"] > 0.0


def test_gate_pairs_only_the_items_both_policies_could_use(tmp_path: Path) -> None:
    n = 100
    logged = ope.load_records(write_jsonl(tmp_path / "logged.jsonl", logged_rows(n)))
    incumbent = ope.load_records(
        write_jsonl(tmp_path / "inc.jsonl", [target_record(f"i/{i}", "good") for i in range(n)])
    )
    # The candidate is as good where it answers, but refuses the first ten items.
    answers = [target_record(f"i/{i}", "good" if i >= 10 else None) for i in range(n)]
    candidate = ope.load_records(write_jsonl(tmp_path / "cand.jsonl", answers))
    result = ope.gate(logged, candidate, incumbent, n_boot=50)
    assert result["candidate"]["n_used"] == 90 and result["incumbent"]["n_used"] == 100
    assert result["candidate"]["excluded"]["target_error"] == 10
    assert result["checks"]["success"]["paired_delta"]["n"] == 90
    assert result["pass"]

    # Nothing in common is an error, not a pass.
    disjoint = ope.load_records(
        write_jsonl(
            tmp_path / "disjoint.jsonl",
            [target_record(f"i/{i}", "good" if i < 50 else None) for i in range(n)],
        )
    )
    other = ope.load_records(
        write_jsonl(
            tmp_path / "other.jsonl",
            [target_record(f"i/{i}", None if i < 50 else "good") for i in range(n)],
        )
    )
    with pytest.raises(ValueError, match="in common"):
        ope.gate(logged, disjoint, other, n_boot=50)


# ---- the command line ----


def run_gate(tmp_path: Path, candidate: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    n = 100
    write_jsonl(tmp_path / "logged.jsonl", logged_rows(n))
    write_jsonl(
        tmp_path / "candidate.jsonl", [target_record(f"i/{i}", candidate) for i in range(n)]
    )
    write_jsonl(tmp_path / "incumbent.jsonl", [target_record(f"i/{i}", "good") for i in range(n)])
    code = ope.main(
        [
            "gate",
            "--logged", str(tmp_path / "logged.jsonl"),
            "--candidate", str(tmp_path / "candidate.jsonl"),
            "--incumbent", str(tmp_path / "incumbent.jsonl"),
            "--out", str(tmp_path / "out" / "nested"),
            "--boot", "50",
        ]
    )  # fmt: skip
    return code, capsys.readouterr().out


def test_cli_passes_and_writes_gate_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_gate(tmp_path, "cheap", capsys)
    assert code == 0
    lines = out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("gate: PASS")
    raw = (tmp_path / "out" / "nested" / "gate.json").read_text(encoding="utf-8")
    assert "\r" not in raw and "NaN" not in raw and "Infinity" not in raw
    doc = json.loads(raw)
    assert doc["pass"] is True
    assert set(doc) >= {"pass", "rule", "checks", "candidate", "incumbent", "n_boot", "seed"}
    assert doc["candidate"]["success"]["dr"].keys() >= {"mean", "lo", "hi", "n"}
    assert doc["candidate"]["excluded"]["target_error"] == 0


def test_cli_writes_null_for_an_estimate_with_nothing_to_weight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run_gate(tmp_path, "never", capsys)
    assert code == 0
    raw = (tmp_path / "out" / "nested" / "gate.json").read_text(encoding="utf-8")
    doc = json.loads(raw, parse_constant=lambda c: pytest.fail(f"{c} is not JSON"))
    assert doc["candidate"]["success"]["snips"] is None
    assert doc["candidate"]["cost_usd"]["snips"] is None


def test_cli_fails_the_gate_with_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_gate(tmp_path, "bad", capsys)
    assert code == 1
    assert out.startswith("gate: FAIL")
    assert (
        json.loads((tmp_path / "out" / "nested" / "gate.json").read_text(encoding="utf-8"))["pass"]
        is False
    )


def test_cli_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "nope.jsonl")
    code = ope.main(["gate", "--logged", missing, "--candidate", missing, "--incumbent", missing,
                     "--out", str(tmp_path / "o")])  # fmt: skip
    assert code == 2
    assert "ope: error" in capsys.readouterr().err
    assert not (tmp_path / "o" / "gate.json").exists()


def test_cli_nothing_usable_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_jsonl(tmp_path / "logged.jsonl", [replay_record("i/0", "good")])
    write_jsonl(tmp_path / "t.jsonl", [target_record("i/0", None)])
    code = ope.main(["gate", "--logged", str(tmp_path / "logged.jsonl"), "--candidate",
                     str(tmp_path / "t.jsonl"), "--incumbent", str(tmp_path / "t.jsonl"),
                     "--out", str(tmp_path / "o")])  # fmt: skip
    assert code == 2
    assert "no usable item" in capsys.readouterr().err


def test_cli_usage_error_exits_2() -> None:
    with pytest.raises(SystemExit) as raised:
        ope.main(["gate", "--logged", "only-one-flag"])
    assert raised.value.code == 2
