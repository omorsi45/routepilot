"""Tests for the contextual bandits: LinUCB, linear Thompson sampling and the stream protocol."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from routepilot_train import bandits, features
from routepilot_train.train import design

D = 4  # context size of the synthetic worlds: three random coordinates and a constant 1
K = 3


def world(n: int, seed: int = 0, noise: float = 0.05):
    """Contexts and arm rewards: arm 2 is best for every context (reward about 1, others 0.2)."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=(n, D - 1)), np.ones(n)])
    r = np.full((n, K), 0.2) + noise * rng.normal(size=(n, K))
    r[:, 2] += 0.8
    return x, r


def flip_world(n: int, seed: int = 0):
    """The best arm depends on the context: arm 0 when x0 > 0, arm 1 otherwise; arm 2 is poor."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=(n, D - 1)), np.ones(n)])
    r = np.column_stack([x[:, 0], -x[:, 0], np.zeros(n)]) + 0.02 * rng.normal(size=(n, K))
    return x, r


def play_rewards(learner, x, r) -> np.ndarray:
    """Run a bandit-feedback stream and return the arms chosen."""
    arms = np.empty(len(x), dtype=np.int64)
    for t in range(len(x)):
        arms[t] = learner.choose(x[t])
        learner.update(int(arms[t]), x[t], float(r[t, arms[t]]))
    return arms


# --- LinUCB -----------------------------------------------------------------------------------


def test_linucb_learns_best_arm():
    x, r = world(600)
    learner = bandits.LinUCB(K, D, alpha=0.5)

    arms = play_rewards(learner, x, r)

    assert (arms[-200:] == 2).mean() >= 0.95
    assert learner.counts[2] > learner.counts[0] and learner.counts[2] > learner.counts[1]


def test_linucb_learns_a_context_dependent_best_arm():
    x, r = flip_world(1500)
    learner = bandits.LinUCB(K, D, alpha=0.3)
    play_rewards(learner, x, r)

    x_new, r_new = flip_world(400, seed=1)
    chosen = np.array([learner.choose(row) for row in x_new])

    assert (chosen == r_new.argmax(axis=1)).mean() >= 0.9


def test_linucb_score_is_the_ridge_estimate_plus_alpha_times_the_width():
    x, r = world(40)
    alpha = 0.7
    learner = bandits.LinUCB(K, D, alpha=alpha)
    arms = play_rewards(learner, x, r)
    probe = np.array([0.3, -1.0, 0.5, 1.0])

    scores = learner.scores(probe)

    for a in range(K):
        xs = x[arms == a]
        rewards = r[arms == a, a]  # bandit feedback: only the chosen arm's reward was seen
        gram = np.eye(D) + xs.T @ xs
        theta = np.linalg.solve(gram, xs.T @ rewards)
        width = np.sqrt(probe @ np.linalg.solve(gram, probe))
        assert scores[a] == pytest.approx(theta @ probe + alpha * width, abs=1e-9)


def test_sherman_morrison_updates_match_the_direct_inverse():
    rng = np.random.default_rng(2)
    learner = bandits.LinUCB(2, D, alpha=1.0)
    xs, rs = [], []
    for _ in range(300):
        x = rng.normal(size=D)
        r = float(rng.normal())
        learner.update(1, x, r)
        xs.append(x)
        rs.append(r)
    gram = np.eye(D) + np.array(xs).T @ np.array(xs)

    np.testing.assert_allclose(learner.a_inv[1], np.linalg.inv(gram), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        learner.theta[1], np.linalg.solve(gram, np.array(xs).T @ np.array(rs)), atol=1e-10
    )
    np.testing.assert_array_equal(learner.a_inv[0], np.eye(D))  # the other arm is untouched
    assert not learner.theta[0].any() and learner.counts.tolist() == [0, 300]


def test_linucb_ties_go_to_the_lower_arm_index():
    x = np.array([0.5, -0.5, 2.0, 1.0])
    assert bandits.LinUCB(K, D, alpha=0.0).choose(x) == 0
    assert bandits.LinUCB(K, D, alpha=1.0).choose(x) == 0  # fresh arms: equal width, equal mean

    learner = bandits.LinUCB(K, D, alpha=0.0)
    learner.update(2, x, 1.0)
    assert learner.choose(x) == 2  # arm 2 now has the only positive estimate
    learner.update(0, x, 1.0)  # arms 0 and 2 now tie exactly at the top
    assert learner.choose(x) == 0


def test_the_uncertainty_width_shrinks_as_an_arm_is_played():
    learner = bandits.LinUCB(1, D, alpha=1.0)
    probe = np.array([1.0, 0.0, 0.0, 1.0])
    before = learner.scores(probe)[0]  # theta is 0, so this is the width alone
    for _ in range(50):
        learner.update(0, probe, 0.0)

    assert learner.scores(probe)[0] < before / 3


def test_learners_validate_their_inputs():
    with pytest.raises(ValueError, match="alpha"):
        bandits.LinUCB(K, D, alpha=-0.1)
    with pytest.raises(ValueError, match="v must"):
        bandits.LinearTS(K, D, v=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="arms"):
        bandits.LinUCB(0, D, alpha=1.0)
    learner = bandits.LinUCB(K, D, alpha=1.0)
    good = np.ones(D)
    with pytest.raises(ValueError, match="arm"):
        learner.update(K, good, 0.0)
    with pytest.raises(ValueError, match="context"):
        learner.update(0, np.ones(D + 1), 0.0)
    with pytest.raises(ValueError, match="context"):
        learner.choose(np.ones(D + 1))
    with pytest.raises(ValueError, match="finite"):
        learner.update(0, good, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        learner.update(0, np.array([np.inf, 0, 0, 1.0]), 0.0)
    assert learner.counts.sum() == 0  # nothing was applied


# --- linear Thompson sampling -----------------------------------------------------------------


def ts(seed: int, v: float = 0.5, n_arms: int = K) -> bandits.LinearTS:
    return bandits.LinearTS(n_arms, D, v, np.random.default_rng(seed))


def test_linear_ts_seeded():
    x, r = world(300)
    first = play_rewards(ts(3, v=1.0), x, r)
    again = play_rewards(ts(3, v=1.0), x, r)
    other = play_rewards(ts(4, v=1.0), x, r)

    np.testing.assert_array_equal(first, again)
    assert (first != other).any()


def test_linear_ts_learns_the_best_arm():
    x, r = world(800)

    arms = play_rewards(ts(0, v=0.1), x, r)

    assert (arms[-200:] == 2).mean() >= 0.9


def test_a_thompson_draw_comes_from_the_gaussian_posterior():
    rng = np.random.default_rng(5)
    learner = bandits.LinearTS(2, 3, 0.7, np.random.default_rng(6))
    for _ in range(12):
        learner.update(0, rng.normal(size=3), float(rng.normal()))

    draws = np.array([learner.draw()[0] for _ in range(20000)])

    np.testing.assert_allclose(draws.mean(axis=0), learner.theta[0], atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), 0.7**2 * learner.a_inv[0], atol=0.02)
    # the arm that was never played keeps the prior N(0, v^2 I)
    prior = np.array([learner.draw()[1] for _ in range(20000)])
    np.testing.assert_allclose(np.cov(prior.T), 0.7**2 * np.eye(3), atol=0.03)


def test_the_posterior_that_is_drawn_from_follows_the_updates():
    learner = bandits.LinearTS(1, D, 1.0, np.random.default_rng(8))
    probe = np.array([1.0, 0.0, 0.0, 1.0])
    before = np.std([learner.draw()[0] @ probe for _ in range(4000)])
    for _ in range(100):
        learner.update(0, probe, 0.0)

    after = np.std([learner.draw()[0] @ probe for _ in range(4000)])

    # the variance of theta . x is v^2 x'A^-1x: |x|^2 = 2 before, 2 / 201 after
    assert before == pytest.approx(np.sqrt(2.0), rel=0.05)
    assert after == pytest.approx(np.sqrt(2.0 / 201.0), rel=0.05)


def test_each_choice_takes_one_block_of_normals_from_the_generator():
    learner = ts(11)
    twin = np.random.default_rng(11)
    x = np.array([0.1, 0.2, 0.3, 1.0])

    learner.choose(x)
    twin.standard_normal((K, D))  # one draw per arm, all in one block

    assert learner.rng.standard_normal() == twin.standard_normal()


def test_ts_ties_are_not_broken_by_the_arm_index_alone():
    x = np.array([0.1, 0.2, 0.3, 1.0])
    picks = {ts(seed).choose(x) for seed in range(40)}
    assert picks == set(range(K))  # a fresh posterior is symmetric: every arm gets chosen


def test_a_copy_of_a_learner_continues_the_same_stream_and_is_independent():
    x, r = world(200)
    for make in (lambda: bandits.LinUCB(K, D, 0.5), lambda: ts(2)):
        learner = make()
        play_rewards(learner, x[:100], r[:100])
        twin = copy.deepcopy(learner)

        a = play_rewards(learner, x[100:], r[100:])
        b = play_rewards(twin, x[100:], r[100:])

        np.testing.assert_array_equal(a, b)
    snapshot = copy.deepcopy(learner)
    learner.update(0, x[0], 1.0)
    assert snapshot.counts.sum() == learner.counts.sum() - 1


# --- the context ------------------------------------------------------------------------------

TEXTS = [
    "alpha bravo charlie ? 12 + 7",
    "delta echo foxtrot golf",
    "hotel india juliet kilo lima mike",
    "```python\nprint(1)\n``` what does it print?",
    "november oscar papa 3.14 \\frac{1}{2}",
    "quebec romeo sierra tango",
    "uniform victor whiskey xray yankee zulu alpha",
    "bravo charlie delta echo ?",
    "foxtrot golf hotel india",
    "juliet kilo lima mike november",
    "oscar papa quebec romeo sierra tango uniform",
    "victor whiskey xray yankee zulu",
]


def train_items(n: int = len(TEXTS)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"toy/{i}" for i in range(n)],
            "dataset": "toy",
            "split": "train",
            "prompt": TEXTS[:n],
        }
    )


def test_the_default_context_has_141_dimensions():
    assert bandits.SVD_COMPONENTS == 128 and features.NUM_DENSE == 12
    assert bandits.context_size() == 141
    assert bandits.context_size(4) == 4 + 12 + 1


def test_a_context_is_the_svd_of_the_hashed_part_the_standardized_dense_part_and_a_one():
    context = bandits.fit_context(train_items(), dim=64, n_components=4)

    got = context.transform(TEXTS)

    assert got.shape == (len(TEXTS), 17) and context.size == 17
    hashed, dense = features.extract_texts(TEXTS, 64)
    np.testing.assert_allclose(got[:, :4], context.svd.transform(hashed), atol=1e-12)
    mean, std = design.standardizer(dense)
    np.testing.assert_allclose(got[:, 4:16], (dense - mean) / std, atol=1e-12)
    np.testing.assert_array_equal(got[:, 16], np.ones(len(TEXTS)))
    np.testing.assert_allclose(context.dense_mean, mean)
    np.testing.assert_allclose(context.dense_std, std)


def test_the_context_is_deterministic_and_each_row_stands_alone():
    a = bandits.fit_context(train_items(), dim=64, n_components=4)
    b = bandits.fit_context(train_items(), dim=64, n_components=4)

    np.testing.assert_array_equal(a.transform(TEXTS), b.transform(TEXTS))
    np.testing.assert_allclose(a.transform(TEXTS[3:4]), a.transform(TEXTS)[3:4], atol=1e-12)
    assert a.svd.random_state == 0 and a.svd.n_components == 4


def test_the_context_is_fitted_on_train_rows_only():
    for bad in ("calib", "test", "heldout"):
        items = train_items()
        items.loc[3, "split"] = bad
        with pytest.raises(ValueError, match="splits not allowed"):
            bandits.fit_context(items, dim=64, n_components=4)
    with pytest.raises(ValueError, match="split"):
        bandits.fit_context(train_items().drop(columns="split"), dim=64, n_components=4)


def test_a_context_needs_more_features_than_components_and_a_dim():
    with pytest.raises(ValueError, match="n_components"):
        bandits.fit_context(train_items(), dim=8, n_components=8)
    with pytest.raises(ValueError, match="n_components"):
        bandits.fit_context(train_items(), dim=64, n_components=0)


# --- reading the arms -------------------------------------------------------------------------


def write_replay(path: Path, rows: list[tuple]) -> None:
    lines = []
    for item_id, split, prompt, outcomes in rows:
        lines.append(
            json.dumps(
                {
                    "item_id": item_id,
                    "dataset": item_id.split("/")[0],
                    "split": split,
                    "prompt_sha256": "0" * 64,
                    "messages": [{"role": "user", "content": prompt}],
                    "outcomes": {
                        m: {
                            "success": ok,
                            "cost_picousd": cost,
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                        }
                        for m, (ok, cost) in outcomes.items()
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ROWS = [
    ("a/2", "calib", "second calib", {"m1": (True, 2_000_000_000_000), "m2": (False, 5)}),
    ("a/1", "train", "first train", {"m1": (False, 1_000_000_000_000), "m2": (True, 7)}),
    ("a/3", "test", "a test row", {"m1": (True, 3), "m2": (True, 4)}),
    ("b/1", "heldout", "a heldout row", {"m1": (False, 1), "m2": (False, 1)}),
]


def test_read_arms_keeps_the_wanted_splits_sorted_by_item_id(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    write_replay(path, ROWS)

    arms = bandits.read_arms(path, {"train", "calib"}, ["m1", "m2"])

    assert list(arms.items["item_id"]) == ["a/1", "a/2"]
    assert list(arms.items["split"]) == ["train", "calib"]
    assert list(arms.items["prompt"]) == ["first train", "second calib"]
    assert arms.models == ("m1", "m2")
    assert arms.success.tolist() == [[0, 1], [1, 0]] and arms.success.dtype == np.int64
    assert arms.cost_picousd.tolist() == [[1_000_000_000_000, 7], [2_000_000_000_000, 5]]
    assert arms.cost_picousd.dtype == np.int64
    np.testing.assert_allclose(arms.cost_usd, [[1.0, 7e-12], [2.0, 5e-12]])
    assert "a test row" not in " ".join(arms.items["prompt"])


def test_read_arms_refuses_odd_input(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    write_replay(path, ROWS)
    with pytest.raises(ValueError, match="unknown split"):
        bandits.read_arms(path, {"everything"}, ["m1"])
    with pytest.raises(ValueError, match="no items in"):
        bandits.read_arms(path, set(), ["m1"])
    only_train = tmp_path / "train.jsonl"
    write_replay(only_train, ROWS[1:2])
    with pytest.raises(ValueError, match="no items in"):
        bandits.read_arms(only_train, {"test"}, ["m1"])
    with pytest.raises(ValueError, match="no recorded outcome for model"):
        bandits.read_arms(path, {"train"}, ["m1", "missing"])
    two = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    two["messages"].append({"role": "assistant", "content": "x"})
    path.write_text(json.dumps(two) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one user message"):
        bandits.read_arms(path, {"calib"}, ["m1"])


def test_the_reward_is_success_minus_lambda_times_the_recorded_cost():
    success = np.array([[1, 0], [0, 1]])
    cost = np.array([[0.5, 0.1], [2.0, 0.0]])

    got = bandits.reward_matrix(success, cost, 3.0)

    np.testing.assert_allclose(got, [[1 - 1.5, 0 - 0.3], [0 - 6.0, 1.0]])
    with pytest.raises(ValueError, match="lambda"):
        bandits.reward_matrix(success, cost, float("nan"))
    with pytest.raises(ValueError, match="shape"):
        bandits.reward_matrix(success, cost[:1], 1.0)


# --- orders -----------------------------------------------------------------------------------


def test_orders_are_seeded_permutations_and_the_streams_are_independent():
    warm = bandits.warmup_order(50, seed=3)
    assert sorted(warm.tolist()) == list(range(50))
    np.testing.assert_array_equal(warm, bandits.warmup_order(50, seed=3))
    assert (warm != bandits.warmup_order(50, seed=4)).any()
    shuffled = bandits.random_order(50, seed=3)
    assert sorted(shuffled.tolist()) == list(range(50))
    np.testing.assert_array_equal(shuffled, bandits.random_order(50, seed=3))
    assert (shuffled != warm).any()  # the warm-up and the evaluation orders are different draws


def test_the_shift_order_runs_through_the_domains_then_item_ids():
    items = pd.DataFrame(
        {
            "item_id": ["w/1", "m/2", "m/10", "c/1", "k/1", "r/1", "m/1", "c/0"],
            "dataset": [
                "arenahard_creative_writing",
                "aime",
                "aime",
                "livecodebench",
                "gpqa",
                "arc-agi",
                "arenahard_math",
                "arenahard_coding",
            ],
            "split": "test",
        }
    )

    order = bandits.shift_order(items)

    got = [items["item_id"].iloc[i] for i in order]
    # math (aime, arenahard_math), code, knowledge, reasoning, writing; item_id inside a domain
    assert got == ["m/1", "m/10", "m/2", "c/0", "c/1", "k/1", "r/1", "w/1"]
    assert bandits.DOMAIN_ORDER == ("math", "code", "knowledge", "reasoning", "writing")


def test_the_shift_order_takes_a_domain_map_and_refuses_unknown_datasets():
    items = pd.DataFrame({"item_id": ["x/1", "y/1"], "dataset": ["x", "y"], "split": "test"})
    assert bandits.shift_order(items, {"x": "writing", "y": "math"}).tolist() == [1, 0]
    with pytest.raises(ValueError, match="domain"):
        bandits.shift_order(items)
    with pytest.raises(ValueError, match="domain"):
        bandits.shift_order(items, {"x": "writing", "y": "poetry"})


# --- the stream -------------------------------------------------------------------------------


class Spy:
    """A learner that chooses arm ``t % 2`` and records what it is told."""

    def __init__(self) -> None:
        self.t = 0
        self.updates: list[tuple[int, float]] = []

    def choose(self, x) -> int:
        self.t += 1
        return (self.t - 1) % 2

    def update(self, arm: int, x, reward: float) -> None:
        self.updates.append((arm, reward))


def test_play_gives_the_learner_the_reward_of_the_chosen_arm_only():
    x = np.zeros((4, 2))
    r = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
    spy = Spy()

    arms = bandits.play(spy, x, r, np.array([2, 0, 3, 1]))

    assert arms.tolist() == [0, 1, 0, 1]
    assert spy.updates == [(0, 30.0), (1, 11.0), (0, 40.0), (1, 21.0)]


def test_summarize_choices_reports_reward_success_cost_and_regret():
    reward = np.array([[1.0, 0.0], [0.2, 0.9], [0.5, 0.4]])
    success = np.array([[1, 0], [0, 1], [1, 1]])
    cost = np.array([[0.0, 0.1], [0.2, 0.1], [0.1, 0.1]])
    order = np.array([2, 0, 1])
    arms = np.array([1, 0, 1])  # the chosen arms, in play order

    s = bandits.summarize_choices(reward, success, cost, order, arms)

    # in play order the chosen rewards are 0.4, 1.0, 0.9; the best arm gives 0.5, 1.0, 0.9
    assert s["n"] == 3
    assert s["mean_reward"] == pytest.approx((0.4 + 1.0 + 0.9) / 3)
    assert s["cumulative_regret"] == pytest.approx(0.1)
    assert s["success_rate"] == pytest.approx(3 / 3)
    assert s["mean_cost_usd"] == pytest.approx((0.1 + 0.0 + 0.1) / 3)
    np.testing.assert_allclose(s["reward_curve"], [0.4, 1.4, 2.3])
    np.testing.assert_allclose(s["regret_curve"], [0.1, 0.1, 0.1])
    assert s["arm_share"] == pytest.approx([1 / 3, 2 / 3])


def test_a_routing_error_is_a_failure_at_zero_cost_with_reward_zero():
    reward = np.array([[0.5, -2.0], [0.3, 0.9]])
    success = np.array([[1, 0], [0, 1]])
    cost = np.array([[0.1, 0.2], [0.3, 0.1]])

    s = bandits.summarize_choices(reward, success, cost, np.array([0, 1]), np.array([-1, 1]))

    assert s["mean_reward"] == pytest.approx((0.0 + 0.9) / 2)
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["mean_cost_usd"] == pytest.approx(0.1 / 2)
    assert s["cumulative_regret"] == pytest.approx(0.5)  # item 0 earned 0 where 0.5 was possible
    assert s["arm_share"] == pytest.approx([0.0, 0.5])  # the error is not an arm


def test_the_comparator_choices_cover_the_fixed_arms_and_the_oracle():
    reward = np.array([[1.0, 0.0, 0.5], [0.2, 0.9, 0.1]])
    assert bandits.oracle_arms(reward).tolist() == [0, 1]
    assert bandits.best_constant_arm(reward) == 0  # mean rewards 0.6, 0.45, 0.3
    tie = np.array([[1.0, 1.0]])
    assert bandits.oracle_arms(tie).tolist() == [0] and bandits.best_constant_arm(tie) == 0


# --- blocks and the hyperparameter choice -----------------------------------------------------


def block(split: str, n: int, seed: int = 0) -> bandits.Block:
    x, r = world(n, seed=seed)
    items = pd.DataFrame(
        {"item_id": [f"{split}/{i}" for i in range(n)], "dataset": "toy", "split": split}
    )
    cost = np.zeros_like(r)
    return bandits.Block(
        items=items, x=x, success=(r > 0.5).astype(np.int64), cost_usd=cost, reward=r
    )


def linucb_factory(value: float, seed: int):
    return bandits.LinUCB(K, D, alpha=value)


def test_the_hyperparameter_is_chosen_by_warming_up_on_train_and_scoring_on_calib():
    train, calib = block("train", 300), block("calib", 200, seed=1)

    best, table = bandits.select_hyperparameter(
        linucb_factory, (0.1, 0.5, 1.0, 2.0), train, calib, seed=0
    )

    assert [row["value"] for row in table] == [0.1, 0.5, 1.0, 2.0]
    rewards = [row["mean_reward"] for row in table]
    assert best == table[int(np.argmax(rewards))]["value"]
    for row in table:
        assert set(row) >= {"value", "mean_reward", "success_rate", "mean_cost_usd"}
        # recompute one row by hand with the protocol's orders
        learner = linucb_factory(row["value"], 0)
        bandits.play(learner, train.x, train.reward, bandits.warmup_order(300, 0))
        order = bandits.random_order(200, 0)
        arms = bandits.play(learner, calib.x, calib.reward, order)
        assert row["mean_reward"] == pytest.approx(calib.reward[order, arms].mean())


def test_a_tie_in_the_hyperparameter_goes_to_the_smaller_value():
    train, calib = block("train", 20), block("calib", 20, seed=1)

    best, table = bandits.select_hyperparameter(
        lambda value, seed: bandits.LinUCB(K, D, alpha=0.0), (0.3, 0.1, 0.2), train, calib
    )

    assert len({row["mean_reward"] for row in table}) == 1  # every value plays the same way
    assert best == 0.1


def test_the_choice_refuses_anything_but_train_to_warm_up_and_calib_to_score():
    train, calib = block("train", 30), block("calib", 30, seed=1)
    test = block("test", 30, seed=2)
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.select_hyperparameter(linucb_factory, (0.5,), train, test)
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.select_hyperparameter(linucb_factory, (0.5,), test, calib)
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.select_hyperparameter(linucb_factory, (0.5,), calib, calib)
    mixed = bandits.concat_blocks(train, test)
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.select_hyperparameter(linucb_factory, (0.5,), mixed, calib)
    with pytest.raises(ValueError, match="at least one"):
        bandits.select_hyperparameter(linucb_factory, (), train, calib)


def test_concat_blocks_stacks_rows_in_order():
    a, b = block("train", 5), block("calib", 7, seed=1)

    both = bandits.concat_blocks(a, b)

    assert len(both.items) == 12 and both.x.shape == (12, D) and both.reward.shape == (12, K)
    assert list(both.items["split"]) == ["train"] * 5 + ["calib"] * 7
    np.testing.assert_array_equal(both.reward[5:], b.reward)
    assert both.items.index.tolist() == list(range(12))


# --- the protocol for one seed ----------------------------------------------------------------


def test_one_seed_warms_up_on_train_and_calib_then_evaluates_each_order_from_that_state():
    warm = bandits.concat_blocks(block("train", 200), block("calib", 100, seed=1))
    test = block("test", 120, seed=2)
    orders = {"random": bandits.random_order(120, 5), "same": bandits.random_order(120, 5)}

    result = bandits.run_seed(linucb_factory, 0.5, warm, test, orders, seed=5)

    assert set(result) == {"random", "same"}
    assert result["random"]["n"] == 120  # the warm-up stream is not evaluated
    # both orders start from the same warm state, so identical orders play identically
    np.testing.assert_array_equal(result["random"]["arms"], result["same"]["arms"])
    # replay by hand: warm up in the seeded order, then play the test order
    learner = linucb_factory(0.5, 5)
    bandits.play(learner, warm.x, warm.reward, bandits.warmup_order(300, 5))
    arms = bandits.play(learner, test.x, test.reward, orders["random"])
    np.testing.assert_array_equal(result["random"]["arms"], arms)
    assert result["random"]["mean_reward"] == pytest.approx(
        test.reward[orders["random"], arms].mean()
    )


def test_each_order_is_played_from_the_same_warm_state_not_from_the_one_before():
    warm = bandits.concat_blocks(block("train", 25), block("calib", 15, seed=1))
    test = block("test", 120, seed=2)
    orders = {"first": bandits.random_order(120, 1), "second": bandits.random_order(120, 2)}
    factories = (
        (bandits.linucb_factory(K, D), 2.0),  # strong exploration: the state matters
        (bandits.linear_ts_factory(K, D), 1.0),
    )
    for factory, value in factories:
        result = bandits.run_seed(factory, value, warm, test, orders, seed=3)

        # the second order, played by a learner that was warmed up and has seen nothing else
        learner = factory(value, 3)
        bandits.play(learner, warm.x, warm.reward, bandits.warmup_order(40, 3))
        arms = bandits.play(learner, test.x, test.reward, orders["second"])
        np.testing.assert_array_equal(result["second"]["arms"], arms)
        assert (result["first"]["arms"] != result["second"]["arms"]).any()


def test_one_seed_refuses_a_test_row_in_the_warm_up_and_anything_but_test_to_evaluate():
    train, calib, test = block("train", 30), block("calib", 30, seed=1), block("test", 30, seed=2)
    orders = {"random": bandits.random_order(30, 0)}
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.run_seed(linucb_factory, 0.5, bandits.concat_blocks(train, test), test, orders, 0)
    with pytest.raises(ValueError, match="splits not allowed"):
        bandits.run_seed(linucb_factory, 0.5, bandits.concat_blocks(train, calib), calib, orders, 0)


def test_one_seed_needs_an_order_that_is_a_permutation_of_the_evaluation_rows():
    warm = bandits.concat_blocks(block("train", 20), block("calib", 20, seed=1))
    test = block("test", 10, seed=2)
    for bad in (np.arange(9), np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]), np.arange(1, 11)):
        with pytest.raises(ValueError, match="permutation"):
            bandits.run_seed(linucb_factory, 0.5, warm, test, {"random": bad}, seed=0)


def test_the_factories_build_the_learners_of_the_protocol():
    ucb = bandits.linucb_factory(K, D)(0.5, 3)
    assert isinstance(ucb, bandits.LinUCB) and ucb.alpha == 0.5 and ucb.n_arms == K
    a = bandits.linear_ts_factory(K, D)(0.5, 3)
    b = bandits.linear_ts_factory(K, D)(0.5, 3)
    c = bandits.linear_ts_factory(K, D)(0.5, 4)
    assert isinstance(a, bandits.LinearTS) and a.v == 0.5
    np.testing.assert_array_equal(a.draw(), b.draw())  # the same seed: the same stream
    assert (a.draw() != c.draw()).any()
    # the draws and the orders come from different streams of one seed
    stream = np.random.default_rng([3, bandits.TS_STREAM]).permutation(50)
    assert (stream != bandits.random_order(50, 3)).any()
    assert (stream != bandits.warmup_order(50, 3)).any()


def test_the_grids_and_seeds_are_the_ones_of_the_brief():
    assert bandits.ALPHAS == (0.1, 0.5, 1.0, 2.0)
    assert bandits.TS_SCALES == (0.05, 0.1, 0.5, 1.0)
    assert bandits.SEEDS == tuple(range(10))


def test_spread_is_the_mean_and_the_range_across_seeds():
    got = bandits.spread([0.2, 0.5, 0.3])
    assert got == {"mean": pytest.approx(1.0 / 3), "min": 0.2, "max": 0.5}
    with pytest.raises(ValueError, match="no values"):
        bandits.spread([])
