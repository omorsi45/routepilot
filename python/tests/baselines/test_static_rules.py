import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from routepilot_train.baselines import static_rules
from routepilot_train.baselines.static_rules import MIN_RULE_ITEMS, RULES, map_rules, match_rule

CASES_PATH = Path(__file__).resolve().parents[3] / "testdata" / "rules" / "cases.json"


def case_text(case: dict) -> str:
    return case["text"] + "x" * case.get("pad", 0) + case.get("tail", "")


def build(specs: list[tuple[str, str, str, dict[str, tuple[bool, float]]]]):
    """Items and outcomes from (item_id, prompt, split, {model: (success, cost_usd)})."""
    items = pd.DataFrame(
        {
            "item_id": [s[0] for s in specs],
            "prompt": [s[1] for s in specs],
            "prompt_bytes": [len(s[1].encode("utf-8")) for s in specs],
            "split": [s[2] for s in specs],
        }
    )
    rows = [
        {"item_id": item_id, "model": model, "success": ok, "cost_usd": cost}
        for item_id, _, _, per_model in specs
        for model, (ok, cost) in per_model.items()
    ]
    return items, pd.DataFrame(rows)


def group(prefix: str, text: str, n: int, wins: dict[str, int], cost: float = 1.0):
    """n items with the same text where model m succeeds on the first wins[m] of them."""
    return [
        (
            f"{prefix}/{i}",
            text,
            "train" if i % 2 == 0 else "calib",
            {m: (i < k, cost) for m, k in wins.items()},
        )
        for i in range(n)
    ]


def test_rules_parity():
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert data["rules"] == RULES  # Go builds its policy from this copy
    assert len(data["cases"]) >= 15
    for case in data["cases"]:
        got = match_rule(case_text(case), case["prompt_bytes"], data["rules"])
        assert got == case["expect"], case["note"]


def test_rules_are_ruling_v1_r6_in_order():
    assert [r["name"] for r in RULES] == ["code", "math", "choices", "long"]
    assert RULES[3]["when"] == {"min_prompt_bytes": 4000}


def test_match_rule_first_match_wins_and_none_otherwise():
    assert match_rule("def f(): solve the equation", 10, RULES) == "code"
    assert match_rule("nothing here", 10, RULES) is None
    assert match_rule("nothing here", 4000, RULES) == "long"


def test_match_rule_max_prompt_bytes():
    rules = [{"name": "short", "when": {"max_prompt_bytes": 10}}]
    assert match_rule("hi", 10, rules) == "short"
    assert match_rule("hi", 11, rules) is None


def test_cut_text_is_a_rune_boundary():
    text = "x" * 32767 + "\u00e9" + "tail"  # the 2-byte rune straddles byte 32768
    cut = static_rules.cut_text(text)
    assert cut == "x" * 32767
    assert len(static_rules.cut_text("x" * 40000).encode()) == 32768
    assert static_rules.cut_text("short") == "short"


def test_lower_follows_go_for_the_dotted_capital_i():
    assert static_rules.lower("\u0130NTEGER") == "integer"
    assert static_rules.lower("ABC") == "abc"


def test_map_rules_picks_the_best_model_per_rule():
    specs = group("code", "def f(): pass", 40, {"a": 10, "b": 30, "c": 20})
    specs += group("plain", "just some words", 60, {"a": 50, "b": 5, "c": 20})
    items, outcomes = build(specs)

    mapped = map_rules(items, outcomes, RULES)

    assert [r["name"] for r in mapped] == ["code", "math", "choices", "long"]
    by_name = {r["name"]: r["model"] for r in mapped}
    # Best Single is a (60 of 100); code has 40 items and b wins them; the others match nothing
    assert by_name == {"code": "b", "math": "a", "choices": "a", "long": "a"}


def test_small_rule_maps_to_best_single():
    for n, expected in ((MIN_RULE_ITEMS - 1, "a"), (MIN_RULE_ITEMS, "b")):
        specs = group("code", "def f(): pass", n, {"a": 0, "b": n})
        specs += group("plain", "just some words", 50, {"a": 50, "b": 0})
        items, outcomes = build(specs)
        by_name = {r["name"]: r["model"] for r in map_rules(items, outcomes, RULES)}
        assert by_name["code"] == expected, n


def test_best_single_argument_overrides_the_computed_one():
    specs = group("plain", "just some words", 50, {"a": 50, "b": 0})
    items, outcomes = build(specs)
    assert {r["model"] for r in map_rules(items, outcomes, RULES)} == {"a"}
    assert {r["model"] for r in map_rules(items, outcomes, RULES, best_single="b")} == {"b"}


def test_only_the_first_matching_rule_counts_an_item():
    # 35 items match both the code and the math keywords: they belong to code alone
    specs = group("both", "def f(): solve the equation", 35, {"a": 0, "b": 35})
    specs += group("plain", "just some words", 50, {"a": 50, "b": 0})
    items, outcomes = build(specs)
    by_name = {r["name"]: r["model"] for r in map_rules(items, outcomes, RULES)}
    assert by_name["code"] == "b"
    assert by_name["math"] == "a"  # no item first-matches math, so Best Single


def test_ties_go_to_lower_cost_then_name():
    def mapped(costs: dict[str, float]) -> str:
        specs = [
            (f"c/{i}", "def f(): pass", "train", {m: (True, cost) for m, cost in costs.items()})
            for i in range(MIN_RULE_ITEMS)
        ]
        items, outcomes = build(specs)
        return {r["name"]: r["model"] for r in map_rules(items, outcomes, RULES)}["code"]

    assert mapped({"x": 2.0, "y": 1.0, "z": 1.0}) == "y"  # cheaper wins, then the lower name
    assert mapped({"x": 1.0, "y": 1.0}) == "x"


def test_map_rules_train_calib_only():
    specs = group("code", "def f(): pass", 40, {"a": 40, "b": 0})
    items, outcomes = build(specs)
    for bad in ("test", "heldout"):
        leaked = items.copy()
        leaked.loc[0, "split"] = bad
        with pytest.raises(ValueError, match=bad):
            map_rules(leaked, outcomes, RULES)
    with pytest.raises(ValueError, match="split"):
        map_rules(items.drop(columns="split"), outcomes, RULES)


def test_outcomes_of_items_outside_the_given_items_are_ignored():
    specs = group("code", "def f(): pass", 40, {"a": 40, "b": 0})
    items, outcomes = build(specs)
    # a test item where b wins hugely: its outcomes must not move the mapping
    extra = pd.DataFrame(
        [
            {"item_id": f"test/{i}", "model": m, "success": m == "b", "cost_usd": 0.0}
            for i in range(500)
            for m in ("a", "b")
        ]
    )
    mapped = map_rules(items, pd.concat([outcomes, extra], ignore_index=True), RULES)
    assert {r["name"]: r["model"] for r in mapped}["code"] == "a"


def test_mapped_rules_are_config_rules_and_do_not_alias_the_rules():
    specs = group("plain", "just some words", 50, {"a": 50, "b": 0})
    items, outcomes = build(specs)
    original = copy.deepcopy(RULES)
    mapped = map_rules(items, outcomes, RULES)
    for rule, source in zip(mapped, RULES, strict=True):
        assert set(rule) == {"name", "model", "when"}
        assert rule["when"] == source["when"]
    mapped[0]["when"]["contains_any"].append("mutated")
    assert RULES == original
    json.dumps(mapped)  # plain JSON and YAML types only


def test_map_rules_needs_outcomes():
    items, _ = build(group("code", "def f(): pass", 3, {"a": 1}))
    empty = pd.DataFrame(columns=["item_id", "model", "success", "cost_usd"])
    with pytest.raises(ValueError, match="no outcomes"):
        map_rules(items, empty, RULES)
