"""Tests for the bandits script: protocol order, split discipline, comparators and outputs."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pandas as pd
import pytest

from routepilot_train import bandits, replaytool
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "bandits.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT
DOMAINS = {"syn-a": "math", "syn-b": "code"}
LAMBDA = 1e5  # the fixture's recorded costs are about 1e-6 to 1e-5 USD: this makes them matter
SEEDS = (0, 1, 2)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bandits_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_file(tmp_path: Path, lam: float = LAMBDA) -> Path:
    path = tmp_path / "policy.yaml"
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: min_cost", "mode: dual").replace("lambda: 0", f"lambda: {lam}")
    path.write_text(text, encoding="utf-8")
    return path


def run(derived: Path, binary: Path, out: Path, policy: Path, **kwargs) -> Path:
    b = load_script()
    options = {
        "seeds": SEEDS,
        "bootstrap": 40,
        "dim": 64,
        "n_components": 4,
        "domains": DOMAINS,
    }
    return b.run_bandits(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy,
        out,
        ["python", "bench/scripts/bandits.py", "--pool", "tiny"],
        binary,
        **{**options, **kwargs},
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Every read of rows, hyperparameter choice, seed run and replay, in call order."""
    log: list[tuple] = []
    real_read, real_select = bandits.read_arms, bandits.select_hyperparameter
    real_seed, real_replay = bandits.run_seed, replaytool.replay

    def read(path, splits, models):
        log.append(("read", frozenset(splits)))
        return real_read(path, splits, models)

    def select(factory, values, warm, evaluate, seed=0):
        log.append(("select", tuple(values)))
        return real_select(factory, values, warm, evaluate, seed)

    def seed_run(factory, value, warm, evaluate, orders, seed):
        log.append(("seed", seed, {k: np.asarray(v) for k, v in orders.items()}))
        return real_seed(factory, value, warm, evaluate, orders, seed)

    def replay(binary, config, input, split, policy, out, extra=()):  # noqa: A002
        log.append(("replay", split, policy))
        return real_replay(binary, config, input, split, policy, out, extra)

    monkeypatch.setattr(bandits, "read_arms", read)
    monkeypatch.setattr(bandits, "select_hyperparameter", select)
    monkeypatch.setattr(bandits, "run_seed", seed_run)
    monkeypatch.setattr(replaytool, "replay", replay)
    return log


# --- the whole script on the synthetic pool -------------------------------------------------


def test_outputs_and_the_shape_of_the_results(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    out = run(derived, binary, tmp_path / "out", policy_file(tmp_path))

    assert sorted(p.name for p in out.iterdir()) == [
        "curves_random.png",
        "curves_shift.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    assert all((out / n).stat().st_size > 1000 for n in ("curves_random.png", "curves_shift.png"))
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "bandits" and run_json["policy_version"] == "0782d801a5a3"
    assert run_json["seeds"]["bandit_seeds"] == list(SEEDS)

    r = results_of(out)
    assert r["n_items"] == {"train": 20, "calib": 80, "test": 80}
    assert r["lambda"] == LAMBDA and r["seeds"] == list(SEEDS)
    assert r["context"]["size"] == 4 + 12 + 1 and r["context"]["svd_components"] == 4
    assert r["shift"]["items_per_domain"] == {"math": 40, "code": 40}
    assert set(r["hyperparameters"]) == set(r["learners"]) == {"linucb", "linear_ts"}
    assert r["hyperparameters"]["linucb"]["grid"] == list(bandits.ALPHAS)
    assert r["hyperparameters"]["linear_ts"]["grid"] == list(bandits.TS_SCALES)
    for key, h in r["hyperparameters"].items():
        assert h["chosen"] in h["grid"] and r["learners"][key]["param"] == h["chosen"]
        assert [row["value"] for row in h["calibration_table"]] == h["grid"]
    for learner in r["learners"].values():
        for order in ("random", "shift"):
            o = learner["orders"][order]
            for metric in ("mean_reward", "success_rate", "mean_cost_usd", "cumulative_regret"):
                assert set(o[metric]) == {"mean", "min", "max"}
                assert len(o["per_seed"][metric]) == len(SEEDS)
                assert o[metric]["min"] <= o[metric]["mean"] <= o[metric]["max"]
            assert o["vs_supervised"]["seeds"] == len(SEEDS)
            assert len(o["vs_supervised"]["mean_reward_difference_ci"]) == 2
            # the item-averaged difference is the seed-averaged mean reward minus the policy's
            sup = r["comparators"]["supervised"][order]["mean_reward"]
            assert o["vs_supervised"]["mean_reward_difference"] == pytest.approx(
                o["mean_reward"]["mean"] - sup, abs=1e-12
            )
            lo, hi = o["vs_supervised"]["mean_reward_difference_ci"]
            assert lo <= o["vs_supervised"]["mean_reward_difference"] <= hi
            assert sum(o["arm_share"]) == pytest.approx(1.0)
    assert set(r["comparators"]) == {"supervised", "always_strongest", "best_constant", "oracle"}
    assert len(r["statements"]) == 4 and len(r["reference_statements"]) == 4
    assert all("the supervised learned policy" in s for s in r["statements"])
    assert all("the best single arm by reward" in s for s in r["reference_statements"])
    best = r["learners"]["linucb"]["orders"]["random"]["vs_best_constant"]
    assert best["seeds"] == len(SEEDS) and len(best["mean_reward_difference_ci"]) == 2
    check = r["context"]["standardizer"]
    assert isinstance(check["matches_artifact"], bool) and check["max_abs_diff_mean"] >= 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Do the bandits beat the supervised policy?" in report
    assert "Test split, random order" in report and "Test split, shift order" in report
    assert "Hyperparameter choice" in report and "no latency is measured or simulated" in report
    section = report.split("## Hyperparameter choice")[1].split("\n## ")[0]
    assert "n/a" not in section and "| LinUCB | alpha | 0.1 |" in section
    assert "| LinearTS | v | 0.05 |" in section
    assert any(w in report for w in ("beats", "does NOT beat", "does NOT consistently beat"))

    # no prompt text in any output file
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_the_test_rows_are_read_after_every_choice_and_replayed_once(
    tmp_path: Path, binary: Path, events: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    run(derived, binary, tmp_path / "out", policy_file(tmp_path))

    kinds = [e[0] for e in events]
    reads = [e for e in events if e[0] == "read"]
    assert [e[1] for e in reads] == [frozenset({"train", "calib"}), frozenset({"test"})]
    first_test_read = events.index(reads[1])
    # train and calibration rows are read, both hyperparameters are chosen, and only then the test
    assert events[:first_test_read] == [
        reads[0],
        ("select", bandits.ALPHAS),
        ("select", bandits.TS_SCALES),
    ]
    replays = [e for e in events if e[0] == "replay"]
    assert sorted(replays) == [("replay", "test", "fixed"), ("replay", "test", "learned")]
    assert all(events.index(e) > first_test_read for e in replays)
    assert kinds.count("seed") == 2 * len(SEEDS)  # two learners, one call per seed


def test_the_context_is_fitted_on_train_rows_only(
    tmp_path: Path, binary: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: list[pd.DataFrame] = []
    real = bandits.fit_context

    def spy(train_items, dim=65536, n_components=128):
        seen.append(train_items)
        return real(train_items, dim, n_components)

    monkeypatch.setattr(bandits, "fit_context", spy)
    derived = fixture_pool.make_pool(tmp_path / "derived")

    run(derived, binary, tmp_path / "out", policy_file(tmp_path))

    (items,) = seen
    assert set(items["split"]) == {"train"} and len(items) == 20


def test_the_choices_ignore_the_test_split(tmp_path: Path, binary: Path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "a" / "derived"),
            binary,
            tmp_path / "a" / "out",
            policy_file(tmp_path / "a"),
        )
    )
    flipped = results_of(
        run(
            fixture_pool.make_pool(tmp_path / "b" / "derived", flip_test=True),
            binary,
            tmp_path / "b" / "out",
            policy_file(tmp_path / "b"),
        )
    )

    for key in ("linucb", "linear_ts"):
        assert first["hyperparameters"][key] == flipped["hyperparameters"][key]
        assert first["learners"][key]["param"] == flipped["learners"][key]["param"]
    assert first["context"] == flipped["context"]
    # while the numbers read from the test split moved: every test outcome succeeds
    assert flipped["comparators"]["always_strongest"]["random"]["success_rate"] == 1.0
    assert first["comparators"]["always_strongest"]["random"]["success_rate"] < 1.0


def test_the_stream_orders_are_the_seeded_shuffle_and_the_domain_shift(
    tmp_path: Path, binary: Path, events: list[tuple]
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    run(derived, binary, tmp_path / "out", policy_file(tmp_path))
    items = bandits.read_arms(
        derived / "replay" / "tiny.jsonl", {"test"}, list(fixture_pool.MODELS)
    )
    seed_calls = [e for e in events if e[0] == "seed"]

    for _, seed, orders in seed_calls:
        np.testing.assert_array_equal(orders["random"], bandits.random_order(80, seed))
        shifted = [
            (
                bandits.DOMAIN_ORDER.index(DOMAINS[items.items["dataset"][i]]),
                items.items["item_id"][i],
            )
            for i in orders["shift"]
        ]
        assert shifted == sorted(shifted)
        assert {DOMAINS[items.items["dataset"][i]] for i in orders["shift"][:40]} == {"math"}


def test_the_comparators_are_measured_from_the_recorded_outcomes(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    policy = policy_file(tmp_path)
    r = results_of(run(derived, binary, tmp_path / "out", policy))
    models = list(fixture_pool.MODELS)
    arms = {
        s: bandits.read_arms(derived / "replay" / "tiny.jsonl", {s}, models)
        for s in ("train", "calib", "test")
    }
    reward = {s: bandits.reward_matrix(a.success, a.cost_usd, LAMBDA) for s, a in arms.items()}
    strong = models.index("tiny-strong")

    # the supervised policy: an independent replay of the same command
    frame = replaytool.replay(
        binary,
        derived / "pools" / "tiny.config.yaml",
        derived / "replay" / "tiny.jsonl",
        "test",
        "learned",
        tmp_path / "direct.jsonl",
        [
            "--artifact",
            str(TINY_ARTIFACT),
            "--policy-config",
            str(policy),
            "--mode",
            "dual",
            "--lambda",
            repr(LAMBDA),
        ],
    )
    by_id = frame.set_index("item_id")
    ids = arms["test"].items["item_id"]
    sup_reward = np.array(
        [by_id.loc[i, "success"] * 1.0 - LAMBDA * by_id.loc[i, "cost_picousd"] / 1e12 for i in ids]
    )
    sup = r["comparators"]["supervised"]
    for order in ("random", "shift"):  # a deterministic policy: the order does not matter
        assert sup[order]["mean_reward"] == pytest.approx(sup_reward.mean(), abs=1e-12)
        assert sup[order]["success_rate"] == pytest.approx(frame["success"].mean())
    assert sup["random"]["mean_cost_usd"] == pytest.approx(frame["cost_picousd"].mean() / 1e12)

    strongest = r["comparators"]["always_strongest"]["random"]
    assert strongest["mean_reward"] == pytest.approx(reward["test"][:, strong].mean())
    assert strongest["success_rate"] == pytest.approx(arms["test"].success[:, strong].mean())
    oracle = r["comparators"]["oracle"]["shift"]
    assert oracle["mean_reward"] == pytest.approx(reward["test"].max(axis=1).mean())
    assert oracle["cumulative_regret"] == pytest.approx(0.0, abs=1e-9)
    assert sup["random"]["cumulative_regret"] == pytest.approx(
        (reward["test"].max(axis=1) - sup_reward).sum()
    )
    warm_mean = np.vstack([reward["train"], reward["calib"]]).mean(axis=0)
    constant = r["comparators"]["best_constant"]["random"]
    assert constant["mean_reward"] == pytest.approx(
        reward["test"][:, int(warm_mean.argmax())].mean()
    )
    for name, c in r["comparators"].items():
        for order in ("random", "shift"):
            assert c[order]["cumulative_regret"] >= -1e-9, name
            assert sum(c[order]["arm_share"]) == pytest.approx(1.0), name


def test_the_best_single_arm_is_chosen_on_train_and_calibration_rewards_not_on_test(
    tmp_path: Path, binary: Path
):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    path = derived / "replay" / "tiny.jsonl"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["split"] == "test":  # the strong model fails every test item, the weak one wins
            item["outcomes"]["tiny-strong"]["success"] = False
            item["outcomes"]["tiny-weak"]["success"] = True
        lines.append(json.dumps(item))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    r = results_of(run(derived, binary, tmp_path / "out", policy_file(tmp_path, 0.0)))

    # lambda 0: the reward is the success, and on train+calib tiny-strong succeeds more often
    constant = r["comparators"]["best_constant"]["random"]
    assert constant["success_rate"] == 0.0  # tiny-strong, whatever the test split says
    assert r["comparators"]["oracle"]["random"]["success_rate"] == 1.0
    assert r["comparators"]["always_strongest"]["random"]["success_rate"] == 0.0


def test_the_numbers_are_reproducible_from_the_seeds(tmp_path: Path, binary: Path):
    derived = fixture_pool.make_pool(tmp_path / "derived")
    policy = policy_file(tmp_path)

    a = results_of(run(derived, binary, tmp_path / "a", policy))
    b = results_of(run(derived, binary, tmp_path / "b", policy))

    for key in ("learners", "hyperparameters", "comparators", "statements", "context"):
        assert a[key] == b[key], key


# --- helpers ----------------------------------------------------------------------------------


def tiny_block() -> bandits.Block:
    items = pd.DataFrame({"item_id": ["t/0", "t/1"], "dataset": "syn-a", "split": "test"})
    success = np.array([[1, 0], [0, 1]])
    cost = np.array([[0.5, 0.1], [0.2, 0.3]])
    return bandits.Block(
        items=items,
        x=np.zeros((2, 3)),
        success=success,
        cost_usd=cost,
        reward=bandits.reward_matrix(success, cost, 2.0),
    )


def replay_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["item_id", "split", "model", "success", "cost_picousd", "error"]
    )


def test_policy_arms_maps_models_to_arms_and_marks_routing_errors():
    b = load_script()
    frame = replay_frame(
        [
            ("t/1", "test", "m1", True, 300_000_000_000, ""),
            ("t/0", "test", None, False, 0, "no_route"),
        ]
    )

    arms = b.policy_arms(frame, tiny_block(), ["m0", "m1"])

    assert arms.tolist() == [-1, 1]  # rows follow the block, not the replay file
    rewards = b.item_rewards(tiny_block(), arms)
    np.testing.assert_allclose(rewards, [0.0, 1 - 2.0 * 0.3])


def test_a_replay_that_disagrees_with_the_recorded_outcomes_is_refused():
    b = load_script()
    wrong_success = replay_frame(
        [
            ("t/0", "test", "m0", False, 500_000_000_000, ""),
            ("t/1", "test", "m1", True, 300_000_000_000, ""),
        ]
    )
    wrong_cost = replay_frame(
        [
            ("t/0", "test", "m0", True, 500_000_000_001, ""),
            ("t/1", "test", "m1", True, 300_000_000_000, ""),
        ]
    )
    other_items = replay_frame([("t/0", "test", "m0", True, 500_000_000_000, "")])
    not_test = replay_frame(
        [
            ("t/0", "calib", "m0", True, 500_000_000_000, ""),
            ("t/1", "calib", "m1", True, 300_000_000_000, ""),
        ]
    )

    with pytest.raises(ValueError, match="differs from the recorded"):
        b.policy_arms(wrong_success, tiny_block(), ["m0", "m1"])
    with pytest.raises(ValueError, match="differs from the recorded"):
        b.policy_arms(wrong_cost, tiny_block(), ["m0", "m1"])
    with pytest.raises(ValueError, match="different items"):
        b.policy_arms(other_items, tiny_block(), ["m0", "m1"])
    with pytest.raises(ValueError, match="splits not allowed"):
        b.policy_arms(not_test, tiny_block(), ["m0", "m1"])


def test_the_verdict_says_plainly_whether_a_bandit_beats_the_reference():
    b = load_script()
    assert b.verdict(10, 10) == "beats"
    assert b.verdict(0, 10) == "does NOT beat"
    assert b.verdict(4, 10) == "does NOT consistently beat"  # above on some seeds only

    text = b.statement(
        "LinUCB (alpha = 0.5)",
        "shift",
        {"mean": 0.31, "min": 0.3, "max": 0.33},
        0.45,
        0,
        10,
        "the supervised learned policy",
        (-0.14, -0.16, -0.12),
    )
    assert text == (
        "LinUCB (alpha = 0.5), shift order: does NOT beat the supervised learned policy "
        "(mean reward per item 0.3100, seeds 0.3000 to 0.3300, against 0.4500; above it on 0 "
        "of 10 seeds; paired difference -0.1400 [-0.1600, -0.1200])."
    )
    mixed = b.statement(
        "LinearTS (v = 0.1)",
        "random",
        {"mean": 0.44, "min": 0.37, "max": 0.46},
        0.45,
        2,
        10,
        "the supervised learned policy",
        (-0.01, -0.02, 0.0004),
    )
    assert "random order: does NOT consistently beat the supervised learned policy (mean" in mixed
    assert "above it on 2 of 10 seeds; paired difference -0.0100 [-0.0200, +0.0004])." in mixed


def test_main_defaults_follow_the_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    b = load_script()
    seen: list[tuple] = []

    def fake(derived, pool, artifact, policy_config, out_root, command, **kwargs):
        seen.append((pool, artifact, policy_config, out_root))
        return tmp_path

    monkeypatch.setattr(b, "run_bandits", fake)
    monkeypatch.setattr(b.sweeps, "find_artifact", lambda root, pool: tmp_path / pool)

    assert b.main(["--pool", "flagship"]) == 0

    ((pool, artifact, policy_config, out_root),) = seen
    assert pool == "flagship" and artifact == tmp_path / "flagship"
    assert policy_config == ROOT / "bench" / "policies" / "flagship.yaml"
    assert out_root == ROOT / "bench" / "results" / "bandits"
