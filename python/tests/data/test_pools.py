import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from frames import outcome_frame, prompt_for, prompts_frame, sha, splits_frame

from routepilot_train.data import export, pools, selection


def priced_rows(
    item_ids: list[str],
    model: str,
    in_price: float,
    out_price: float,
    success_of,
) -> list[tuple[str, str, bool, int, int, float]]:
    """Rows whose recorded cost is exactly in_price * prompt + out_price * completion tokens."""
    rows = []
    for k, item_id in enumerate(item_ids):
        prompt_tokens = 100 + 7 * k
        completion_tokens = 40 + 13 * (k % 5) + k
        cost = in_price * prompt_tokens + out_price * completion_tokens
        rows.append(
            (item_id, model, success_of(k, item_id), prompt_tokens, completion_tokens, cost)
        )
    return rows


def three_model_pool() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    item_ids = [f"a/{i}" for i in range(12)] + [f"b/{i}" for i in range(6)]
    assignment = {item_id: ("train", "calib", "test")[k % 3] for k, item_id in enumerate(item_ids)}
    rows = (
        priced_rows(item_ids, "weak", 1e-7, 4e-7, lambda k, _: k % 5 == 0)
        + priced_rows(item_ids, "mid", 1e-6, 4e-6, lambda k, _: k % 2 == 0)
        + priced_rows(item_ids, "strong", 5e-6, 2.5e-5, lambda k, _: True)
    )
    return outcome_frame(rows), splits_frame(assignment), item_ids


def test_fit_prices_recovers_known_prices():
    rng = np.random.default_rng(7)
    n = 400
    prompt_tokens = rng.integers(50, 5000, n)
    completion_tokens = rng.integers(20, 3000, n)
    cost = 2e-6 * prompt_tokens + 8e-6 * completion_tokens + rng.normal(0, 1e-5, n)
    df = pd.DataFrame(
        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost_usd": cost}
    )
    zeros = pd.DataFrame(
        {"prompt_tokens": [9000] * 10, "completion_tokens": [9000] * 10, "cost_usd": [0.0] * 10}
    )
    fit = pools.fit_prices(pd.concat([df, zeros], ignore_index=True))
    assert fit["in_usd_per_token"] == pytest.approx(2e-6, rel=0.01)
    assert fit["out_usd_per_token"] == pytest.approx(8e-6, rel=0.01)
    assert fit["fit_r2"] > 0.999
    assert fit["fit_rmse_usd"] < 5e-5
    assert fit["n_fit"] == n


def test_fit_prices_is_never_negative_and_handles_a_zero_column():
    ct = np.arange(1, 60)
    only_out = pd.DataFrame(
        {"prompt_tokens": 100 + 3 * (ct % 7), "completion_tokens": ct, "cost_usd": 5e-6 * ct}
    )
    fit = pools.fit_prices(only_out)
    assert fit["in_usd_per_token"] >= 0 and fit["out_usd_per_token"] == pytest.approx(5e-6)
    no_output = pd.DataFrame(
        {"prompt_tokens": ct * 10, "completion_tokens": [0] * len(ct), "cost_usd": 3e-6 * ct * 10}
    )
    fit = pools.fit_prices(no_output)
    assert fit["in_usd_per_token"] == pytest.approx(3e-6)
    assert fit["out_usd_per_token"] == 0.0


def test_fit_prices_needs_rows_with_positive_cost():
    df = pd.DataFrame({"prompt_tokens": [10], "completion_tokens": [5], "cost_usd": [0.0]})
    with pytest.raises(ValueError, match="positive"):
        pools.fit_prices(df)


def test_build_pool_contract():
    outcomes, splits, _ = three_model_pool()
    pool = pools.build_pool("demo", outcomes, splits)
    assert json.loads(json.dumps(pool)) == pool  # plain JSON types only
    assert pool["pool"] == "demo"
    assert pool["models"] == ["mid", "strong", "weak"]
    assert pool["datasets"] == ["a", "b"]
    assert pool["n_items"] == 18
    for model, (want_in, want_out) in {
        "weak": (1e-7, 4e-7),
        "mid": (1e-6, 4e-6),
        "strong": (5e-6, 2.5e-5),
    }.items():
        price = pool["prices"][model]
        assert set(price) == {
            "in_usd_per_token",
            "out_usd_per_token",
            "fit_rmse_usd",
            "fit_r2",
            "price_source",
        }
        assert price["price_source"] == "fitted_from_recorded_cost"
        assert price["in_usd_per_token"] == pytest.approx(want_in, rel=1e-6)
        assert price["out_usd_per_token"] == pytest.approx(want_out, rel=1e-6)
    assert pool["best_single"] == "strong"
    assert pool["cheapest"] == "weak"
    # rank = 1 + position in ascending train+calib success; Best Single is highest
    assert pool["rank"] == {"weak": 1, "mid": 2, "strong": 3}


def test_build_pool_keeps_only_items_every_model_has():
    outcomes, splits, _ = three_model_pool()
    drop = (outcomes["item_id"] == "a/3") & (outcomes["model"] == "mid")
    pool = pools.build_pool("demo", outcomes.loc[~drop], splits)
    assert pool["n_items"] == 17


def test_build_pool_rejects_items_without_a_split_and_duplicate_rows():
    outcomes, splits, _ = three_model_pool()
    with pytest.raises(ValueError, match="split"):
        pools.build_pool("demo", outcomes, splits.loc[splits["item_id"] != "a/1"])
    with pytest.raises(ValueError, match="duplicate"):
        pools.build_pool("demo", pd.concat([outcomes, outcomes.iloc[:1]]), splits)


def test_best_single_uses_train_and_calib_only():
    assignment = {f"a/{i}": "train" for i in range(4)}
    assignment |= {f"a/{i}": "calib" for i in range(4, 6)}
    assignment |= {f"a/{i}": "test" for i in range(6, 16)}
    item_ids = list(assignment)
    # alpha wins every test row and loses every train and calib row; beta is the reverse
    rows = priced_rows(item_ids, "alpha", 1e-6, 2e-6, lambda _, i: assignment[i] == "test")
    rows += priced_rows(item_ids, "beta", 2e-6, 4e-6, lambda _, i: assignment[i] != "test")
    outcomes = outcome_frame(rows)
    # over all rows alpha has 10 successes and beta 6, so reading test rows would pick alpha
    assert outcomes.groupby("model")["success"].sum().to_dict() == {"alpha": 10, "beta": 6}
    pool = pools.build_pool("demo", outcomes, splits_frame(assignment))
    assert pool["best_single"] == "beta"
    assert pool["rank"]["beta"] > pool["rank"]["alpha"]


def test_best_single_ties_prefer_lower_cost_then_name():
    assignment = {f"a/{i}": "train" for i in range(6)}
    item_ids = list(assignment)
    always = lambda *_: True  # noqa: E731
    rows = priced_rows(item_ids, "m-b", 2e-6, 4e-6, always)
    rows += priced_rows(item_ids, "m-a", 2e-6, 4e-6, always)
    rows += priced_rows(item_ids, "m-c", 1e-6, 2e-6, always)
    pool = pools.build_pool("demo", outcome_frame(rows), splits_frame(assignment))
    assert pool["best_single"] == "m-c"  # same success, lowest mean recorded cost
    rows = priced_rows(item_ids, "m-b", 2e-6, 4e-6, always) + priced_rows(
        item_ids, "m-a", 2e-6, 4e-6, always
    )
    pool = pools.build_pool("demo", outcome_frame(rows), splits_frame(assignment))
    assert pool["best_single"] == "m-a"  # same success and cost: name decides


def per_item_rows(
    model: str,
    split_of: dict[str, str],
    base_prompt: int,
    base_completion: int,
    in_price: float,
    out_price: float,
    scale_of=lambda item_id, split: 1.0,  # noqa: ARG005
) -> list[tuple[str, str, bool, int, int, float]]:
    """One row per item; token counts vary a little so both prices can be fitted."""
    rows = []
    for k, (item_id, split) in enumerate(split_of.items()):
        prompt_tokens = base_prompt + 7 * k
        completion_tokens = base_completion + 3 * (k % 5)
        cost = scale_of(item_id, split) * (in_price * prompt_tokens + out_price * completion_tokens)
        rows.append((item_id, model, False, prompt_tokens, completion_tokens, cost))
    return rows


def test_cheapest_is_the_lowest_mean_recorded_cost_per_item_on_train_and_calib():
    """Ruling V1-R23: not the lowest fitted price, and test rows are not read."""
    split_of = {f"a/{i}": "train" if i < 2 else "calib" if i < 4 else "test" for i in range(14)}
    # lowest price, but a very long output: expensive per item
    rows = per_item_rows("low-price", split_of, 100, 100_000, 1e-7, 1e-6)
    # higher price, short output: cheapest per item
    rows += per_item_rows("short-output", split_of, 100, 10, 1e-6, 3e-6)
    # free on test rows, dear elsewhere: would win if test rows were read
    rows += per_item_rows(
        "free-on-test",
        split_of,
        100,
        10,
        2e-6,
        6e-6,
        lambda _, split: 0.0 if split == "test" else 1.0,
    )
    pool = pools.build_pool("demo", outcome_frame(rows), splits_frame(split_of))

    def summed_price(model: str) -> float:
        return (
            pool["prices"][model]["in_usd_per_token"] + pool["prices"][model]["out_usd_per_token"]
        )

    assert min(pool["models"], key=summed_price) == "low-price"  # the old definition
    assert pool["cheapest"] == "short-output"


def test_cheapest_ties_go_to_the_lower_name():
    split_of = {f"a/{i}": "train" for i in range(5)}
    rows = per_item_rows("m-b", split_of, 100, 10, 1e-6, 3e-6)
    rows += per_item_rows("m-a", split_of, 100, 10, 1e-6, 3e-6)
    pool = pools.build_pool("demo", outcome_frame(rows), splits_frame(split_of))
    assert pool["cheapest"] == "m-a"


def test_cheapest_uses_the_replay_cost_of_models_priced_from_fitted_prices():
    """A model whose recorded cost is zero on half its rows is not cheap in the replay."""
    split_of = {f"a/{i}": "train" for i in range(10)}
    rows = per_item_rows(
        "small",
        split_of,
        100,
        100,
        6e-7,
        6e-7,
        lambda item_id, _: 0.0 if int(item_id[2:]) % 2 else 1.0,
    )
    rows += per_item_rows("other", split_of, 100, 100, 5e-7, 5e-7)
    outcomes, splits = outcome_frame(rows), splits_frame(split_of)

    recorded = pools.build_pool("demo", outcomes, splits)
    replayed = pools.build_pool("demo", outcomes, splits, fitted_cost_models=["small"])

    assert recorded["cheapest"] == "small"  # mean recorded cost is about half of its real cost
    assert (
        replayed["cheapest"] == "other"
    )  # the replay charges small its fitted price on every item
    with pytest.raises(ValueError, match="not in the pool"):
        pools.build_pool("demo", outcomes, splits, fitted_cost_models=["ghost"])


def test_usd_per_mtok_rounds_half_up():
    assert pools.usd_per_mtok(2e-6) == "2.000000"
    assert pools.usd_per_mtok(np.float64(8e-6)) == "8.000000"
    assert pools.usd_per_mtok(1.2345665e-6) == "1.234567"  # half-even would give 1.234566
    assert pools.usd_per_mtok(0.0) == "0.000000"
    assert pools.usd_per_mtok(4e-13) == "0.000000"
    assert pools.usd_per_mtok(5e-13) == "0.000001"


def hand_pool() -> dict:
    def price(i: float, o: float) -> dict:
        return {
            "in_usd_per_token": i,
            "out_usd_per_token": o,
            "fit_rmse_usd": 0.0,
            "fit_r2": 1.0,
            "price_source": "fitted_from_recorded_cost",
        }

    return {
        "pool": "demo",
        "models": ["m-a", "m-b", "m-c"],
        "datasets": ["a"],
        "n_items": 3,
        "prices": {
            "m-a": price(2e-6, 8e-6),
            "m-b": price(1.2345665e-6, 0.0),
            "m-c": price(3e-7, 1.5e-6),
        },
        "best_single": "m-c",
        "cheapest": "m-b",
        "rank": {"m-a": 1, "m-b": 2, "m-c": 3},
    }


def test_pool_config_contract(tmp_path: Path):
    out = tmp_path / "demo.config.yaml"
    pools.write_pool_config(hand_pool(), out)
    text = out.read_bytes().decode("utf-8")
    assert "\r" not in text and "\u2014" not in text
    assert "replay does not test context feasibility" in text
    cfg = yaml.safe_load(text)

    assert cfg["providers"] == [
        {"name": "replay", "type": "openaicompat", "base_url": "http://127.0.0.1:1/v1"}
    ]
    assert [m["id"] for m in cfg["models"]] == ["m-a", "m-b", "m-c"]
    by_id = {m["id"]: m for m in cfg["models"]}
    for model_id, m in by_id.items():
        assert m["deployments"] == [
            {"name": f"{model_id}-replay", "provider": "replay", "upstream_model": model_id}
        ]
        assert m["context_window"] == 1048576
        assert m["max_output_tokens"] == 65536
        assert m["capabilities"] == ["streaming", "tools", "json_object", "json_schema", "vision"]
    assert by_id["m-a"]["input_price_per_mtok"] == "2.000000"
    assert by_id["m-a"]["output_price_per_mtok"] == "8.000000"
    assert by_id["m-b"]["input_price_per_mtok"] == "1.234567"
    assert by_id["m-b"]["output_price_per_mtok"] == "0.000000"
    assert [by_id[i]["rank"] for i in ("m-a", "m-b", "m-c")] == [1, 2, 3]

    assert cfg["routing"] == {"policy": "strongest"}
    key_hash = hashlib.sha256(b"replay-key-offline").hexdigest()
    assert cfg["tenants"] == [
        {
            "id": "replay",
            "api_key_sha256": key_hash,
            "allowed_models": ["*"],
            "default_max_tokens": 65536,
        }
    ]


def test_pool_config_with_rules(tmp_path: Path):
    rules = [{"name": "code", "model": "m-c", "when": {"contains_any": ["def "]}}]
    out = tmp_path / "rules.config.yaml"
    pools.write_pool_config(hand_pool(), out, rules=rules, default_model="m-a")
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert cfg["routing"] == {"policy": "rules", "default_model": "m-a", "rules": rules}
    with pytest.raises(ValueError, match="default_model"):
        pools.write_pool_config(hand_pool(), tmp_path / "bad.yaml", rules=rules)


def pick(outcomes: pd.DataFrame, splits: pd.DataFrame, **kw) -> list[str]:
    return pools.pick_seven_b(outcomes, splits, flagship=("f1", "f2"), not_seven_b=("big",), **kw)


def test_pick_seven_b_ranks_by_train_success_and_skips_the_ineligible():
    train = [f"d/{i}" for i in range(10)]
    test = [f"d/{i}" for i in range(10, 13)]
    assignment = {i: "train" for i in train} | {i: "test" for i in test}
    everything = train + test

    def rows(model: str, wins: int, test_wins: int, cost: float = 0.001):
        out = []
        for k, item_id in enumerate(everything):
            won = k < wins if item_id in train else (k - 10) < test_wins
            out.append((item_id, model, won, 100, 50, cost))
        return out

    all_rows = []
    for flagship in ("f1", "f2"):
        all_rows += rows(flagship, 5, 0)
    all_rows += rows("s-a", 5, 0)
    all_rows += rows("s-b", 8, 0)
    all_rows += rows("s-c", 8, 0)
    all_rows += rows("s-d", 3, 0)
    all_rows += rows("s-e", 9, 0, cost=0.0)  # no recorded cost at all: not eligible
    all_rows += rows("s-f", 0, 3)  # wins only on test rows: last by train success
    all_rows += rows("big", 10, 0)  # a large non-flagship model: not 7B
    all_rows += rows("openrouter", 10, 0)  # a router, not a model
    outcomes = outcome_frame(all_rows)
    splits = splits_frame(assignment)

    assert pick(outcomes, splits) == ["s-b", "s-c", "s-a", "s-d"]  # tie s-b/s-c by name
    assert pick(outcomes, splits, n=5) == ["s-b", "s-c", "s-a", "s-d", "s-f"]
    # a model that wins only on test rows does not outrank one that wins on train
    assert pick(outcomes, splits, n=1) == ["s-b"]


def test_pick_seven_b_uses_items_where_the_flagships_and_the_candidate_have_records():
    assignment = {f"d/{i}": "train" for i in range(4)}
    rows = []
    for item_id in assignment:
        rows += [(item_id, "f1", True, 10, 10, 0.1), (item_id, "f2", True, 10, 10, 0.1)]
    rows.append(("d/0", "s-x", True, 10, 10, 0.1))  # s-x covers one item and wins it
    rows += [(item_id, "s-y", item_id != "d/0", 10, 10, 0.1) for item_id in assignment]
    picked = pick(outcome_frame(rows), splits_frame(assignment))
    assert picked == ["s-x", "s-y"]  # 1/1 beats 3/4


def test_export_replay_contract(tmp_path: Path):
    ids = ["b/2", "a/10", "a/2", "a/3"]
    rows = [
        ("b/2", "m1", True, 120, 30, 0.0123456789),
        ("b/2", "m2", False, 121, 31, 0.5),
        ("a/10", "m1", False, 10, 20, 1.5e-7),
        ("a/10", "m2", True, 11, 21, 0.25),
        ("a/2", "m1", True, 50, 60, 0.0),
        ("a/2", "m2", True, 51, 61, 0.1),
        ("a/3", "m1", True, 5, 5, 0.1),  # m2 has no record: the item is not exported
    ]
    outcomes = outcome_frame(rows)
    cafe = "caf\u00e9 \u4e2d\u6587 \U0001f600 line one\nline two"
    prompts = prompts_frame(ids, overrides={"a/10": cafe})
    splits = splits_frame({"b/2": "test", "a/10": "heldout", "a/2": "train", "a/3": "train"})
    pool = {"models": ["m1", "m2"], "prices": {}}

    out = tmp_path / "replay" / "demo.jsonl"
    count = export.export_replay(pool, outcomes, prompts, splits, out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines.pop() == ""  # one trailing newline
    assert count == len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [p["item_id"] for p in parsed] == ["a/10", "a/2", "b/2"]  # sorted by item_id
    for p in parsed:
        assert list(p) == ["item_id", "dataset", "split", "prompt_sha256", "messages", "outcomes"]
        assert list(p["outcomes"]) == ["m1", "m2"]
        for o in p["outcomes"].values():
            assert list(o) == ["success", "cost_picousd", "prompt_tokens", "completion_tokens"]

    held = parsed[0]
    assert held["split"] == "heldout" and held["dataset"] == "a"
    assert held["messages"] == [{"role": "user", "content": cafe}]
    assert held["prompt_sha256"] == sha(cafe)
    assert held["outcomes"]["m1"] == {
        "success": False,
        "cost_picousd": round(1.5e-7 * 1e12),
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }
    assert parsed[1]["outcomes"]["m1"]["cost_picousd"] == 0
    assert parsed[2]["outcomes"]["m1"]["cost_picousd"] == round(0.0123456789 * 1e12)
    assert parsed[2]["outcomes"]["m2"]["success"] is False
    assert all(isinstance(o["cost_picousd"], int) for p in parsed for o in p["outcomes"].values())
    assert parsed[1]["messages"][0]["content"] == prompt_for("a/2")


def test_export_replay_prices_fitted_cost_models_from_their_tokens(tmp_path: Path):
    outcomes = outcome_frame(
        [
            ("a/1", "big", True, 100, 40, 0.5),
            ("a/1", "small", True, 1000, 500, 0.0),  # recorded cost is zero
            ("a/2", "big", False, 200, 80, 0.25),
            ("a/2", "small", False, 2000, 0, 0.0),
        ]
    )
    pool = {
        "models": ["big", "small"],
        "prices": {"small": {"in_usd_per_token": 1.5e-7, "out_usd_per_token": 6e-7}},
    }
    out = tmp_path / "mixed.jsonl"
    export.export_replay(
        pool,
        outcomes,
        prompts_frame(["a/1", "a/2"]),
        splits_frame({"a/1": "train", "a/2": "test"}),
        out,
        fitted_cost_models=["small"],
    )
    first, second = (json.loads(line) for line in out.read_text(encoding="utf-8").splitlines())
    assert first["outcomes"]["big"]["cost_picousd"] == round(0.5 * 1e12)  # recorded cost kept
    assert first["outcomes"]["small"]["cost_picousd"] == round((1.5e-7 * 1000 + 6e-7 * 500) * 1e12)
    assert second["outcomes"]["small"]["cost_picousd"] == round(1.5e-7 * 2000 * 1e12)


def test_export_replay_refuses_missing_prompts_and_splits(tmp_path: Path):
    outcomes = outcome_frame([("a/1", "m1", True, 1, 1, 0.1)])
    pool = {"models": ["m1"], "prices": {}}
    splits = splits_frame({"a/1": "train"})
    with pytest.raises(ValueError, match="prompt"):
        export.export_replay(pool, outcomes, prompts_frame([]), splits, tmp_path / "x.jsonl")
    with pytest.raises(ValueError, match="split"):
        export.export_replay(
            pool, outcomes, prompts_frame(["a/1"]), splits_frame({}), tmp_path / "y.jsonl"
        )


def test_selection_names_are_what_the_pool_code_expects():
    assert len(selection.FLAGSHIP_MODELS) == 12
    assert pools.NOT_SEVEN_B and not set(pools.NOT_SEVEN_B) & set(selection.FLAGSHIP_MODELS)
