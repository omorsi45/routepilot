"""The static rules baseline: keyword and length rules mapped to models on training data.

The rules are fixed before any model is trained. They are the same
ones the Go ``rules`` policy runs from the replay config, so ``match_rule`` mirrors
``rules.Order`` in internal/router/policies.go: the first rule whose conditions all
hold wins, ``contains_any`` looks for a lowercased keyword in the joined message text
cut to 32768 bytes on a rune boundary, and the prompt-byte bounds compare against the
total bytes of the message text. ``testdata/rules/cases.json`` holds the cases both
sides check.

Only the model each rule maps to is learned: among the train and calibration items
whose first matching rule is that rule, the model with the highest success.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from routepilot_train.data.splits import require_splits

TEXT_LIMIT = 32768
MIN_RULE_ITEMS = 30

RULES: list[dict[str, Any]] = [
    {
        "name": "code",
        "when": {
            "contains_any": [
                "```",
                "def ",
                "class ",
                "#include",
                "function ",
                "import ",
                "public static",
            ]
        },
    },
    {
        "name": "math",
        "when": {"contains_any": ["\\frac", "$$", "\\(", "\\sqrt", "\\sum", "integer", "equation"]},
    },
    {"name": "choices", "when": {"contains_any": ["(a)", "a) ", "a. ", "options:"]}},
    {"name": "long", "when": {"min_prompt_bytes": 4000}},
]


def cut_text(text: str, limit: int = TEXT_LIMIT) -> str:
    """The longest prefix of ``text`` that is at most ``limit`` UTF-8 bytes.

    Mirrors ``truncateUTF8`` in internal/catalog: when byte ``limit`` is a
    continuation byte the cut moves back to the start of that rune.
    """
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    cut = limit
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    return raw[:cut].decode("utf-8")


def lower(text: str) -> str:
    """Lowercase like Go's ``strings.ToLower``.

    Python's full case mapping turns U+0130 into two code points where Go's simple
    mapping gives a plain ``i``; that is the only difference that can change a match
    of the ASCII keywords above.
    """
    return text.replace("\u0130", "i").lower()


def _matches(when: Mapping[str, Any], lowered: str, prompt_bytes: int) -> bool:
    if prompt_bytes < when.get("min_prompt_bytes", 0):
        return False
    max_bytes = when.get("max_prompt_bytes", 0)
    if max_bytes > 0 and prompt_bytes > max_bytes:
        return False
    keywords = when.get("contains_any", ())
    return not keywords or any(lower(k) in lowered for k in keywords)


def match_rule(text: str, prompt_bytes: int, rules: Sequence[Mapping[str, Any]]) -> str | None:
    """The name of the first rule that matches, or None."""
    lowered = lower(cut_text(text))
    for rule in rules:
        if _matches(rule["when"], lowered, prompt_bytes):
            return str(rule["name"])
    return None


def _best_single(rows: pd.DataFrame) -> str:
    """Highest success, then lower mean recorded cost, then name (as the pool does)."""
    stats = rows.groupby("model").agg(wins=("success", "sum"), cost=("cost_usd", "mean"))
    return str(
        min(stats.index, key=lambda m: (-int(stats.loc[m, "wins"]), stats.loc[m, "cost"], m))
    )


def map_rules(
    items: pd.DataFrame,
    outcomes: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    best_single: str | None = None,
) -> list[dict[str, Any]]:
    """Map each rule to a model using train and calibration items only.

    ``items`` needs ``item_id``, ``split``, ``prompt`` and ``prompt_bytes`` and may
    hold only ``train`` and ``calib`` rows. ``outcomes`` holds one row per
    (item, model); rows of items that are not in ``items`` are ignored. Among the
    items whose first matching rule is a rule, that rule maps to the model with the
    highest success, ties going to the lower mean recorded cost and then the name.
    A rule that first-matches fewer than ``MIN_RULE_ITEMS`` items maps to Best Single,
    which is ``best_single`` when given and else computed from the same rows.

    Returns config rules in order, each ``{"name", "model", "when"}``.
    """
    require_splits(items, {"train", "calib"})
    matched = pd.DataFrame(
        {
            "item_id": items["item_id"].to_numpy(),
            "rule": [
                match_rule(str(prompt), int(size), rules)
                for prompt, size in zip(items["prompt"], items["prompt_bytes"], strict=True)
            ],
        }
    )
    rows = outcomes.merge(matched, on="item_id", how="inner")
    if rows.empty:
        raise ValueError("no outcomes for the given items")
    fallback = best_single if best_single is not None else _best_single(rows)

    mapped: list[dict[str, Any]] = []
    for rule in rules:
        name = str(rule["name"])
        rule_rows = rows.loc[rows["rule"] == name]
        if rule_rows["item_id"].nunique() < MIN_RULE_ITEMS:
            model = fallback
        else:
            stats = rule_rows.groupby("model").agg(
                rate=("success", "mean"), cost=("cost_usd", "mean")
            )
            model = str(
                min(stats.index, key=lambda m: (-stats.loc[m, "rate"], stats.loc[m, "cost"], m))
            )
        mapped.append({"name": name, "model": model, "when": copy.deepcopy(dict(rule["when"]))})
    return mapped
