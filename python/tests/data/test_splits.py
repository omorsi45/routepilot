import hashlib

import pandas as pd
import pytest

from routepilot_train.data import selection, splits


def hash_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: hashlib.sha256((n + splits.SALT).encode()).hexdigest())


def synthetic_items(n_datasets: int, n_items: int) -> tuple[pd.DataFrame, dict[str, str]]:
    datasets = [f"d{i:02d}" for i in range(n_datasets)]
    domains = {name: f"domain{i:02d}" for i, name in enumerate(datasets)}
    owner = [datasets[i % n_datasets] for i in range(n_items)]
    items = pd.DataFrame({"item_id": [f"{o}/{i}" for i, o in enumerate(owner)], "dataset": owner})
    return items, domains


def test_assign_splits_deterministic_and_proportions():
    items, domains = synthetic_items(20, 20_000)
    first = splits.assign_splits(items, domains=domains)
    second = splits.assign_splits(items.copy(), domains=domains)
    assert first.equals(second)
    assert list(first.columns) == ["item_id", "dataset", "split"]
    assert first["item_id"].tolist() == items["item_id"].tolist()

    held = splits.heldout_datasets(sorted(set(items["dataset"])), domains=domains)
    assert len(held) == 3
    is_held = first["dataset"].isin(held)
    assert (first.loc[is_held, "split"] == "heldout").all()

    rest = first.loc[~is_held]
    assert set(rest["split"]) == {"train", "calib", "test"}
    shares = rest["split"].value_counts(normalize=True)
    for name, want in (("train", 0.6), ("calib", 0.2), ("test", 0.2)):
        assert abs(shares[name] - want) < 0.015, (name, shares[name])


def test_place_is_the_rule_assign_splits_applies_outside_the_heldout_datasets():
    items, domains = synthetic_items(20, 2_000)
    assigned = splits.assign_splits(items, domains=domains)
    kept = assigned.loc[assigned["split"] != "heldout"]
    assert [splits.place(i) for i in kept["item_id"]] == kept["split"].tolist()
    assert splits.place("gpqa/7") == splits.place("gpqa/7")
    assert {splits.place(f"x/{i}") for i in range(500)} == {"train", "calib", "test"}


def test_split_follows_the_documented_hash():
    items, domains = synthetic_items(6, 600)
    out = splits.assign_splits(items, domains=domains)
    held = set(splits.heldout_datasets(sorted(set(items["dataset"])), domains=domains))
    for row in out.itertuples():
        if row.dataset in held:
            assert row.split == "heldout"
            continue
        digest = hashlib.sha256((row.item_id + splits.SALT).encode()).hexdigest()
        u = int(digest[:16], 16) / 2**64
        assert row.split == ("train" if u < 0.6 else "calib" if u < 0.8 else "test")


def test_split_does_not_depend_on_the_other_rows():
    items, domains = synthetic_items(8, 800)
    full = splits.assign_splits(items, domains=domains).set_index("item_id")["split"]
    # drop a third of the rows; every dataset is still present, so the held-out set is the same
    kept = items.loc[items.index % 3 != 0]
    assert set(kept["dataset"]) == set(items["dataset"])
    part = splits.assign_splits(kept, domains=domains).set_index("item_id")["split"]
    assert len(part) < len(full)
    assert (full.loc[part.index] == part).all()


def test_heldout_rule_one_per_domain():
    names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    order = hash_order(names)
    # the two first-hashed datasets share a domain, so the second one is skipped
    domains = {
        order[0]: "x",
        order[1]: "x",
        order[2]: "y",
        order[3]: "z",
        order[4]: "w",
    }
    assert splits.heldout_datasets(names, domains=domains) == [order[0], order[2], order[3]]


def test_heldout_stops_at_three_and_needs_distinct_domains():
    names = ["a", "b", "c", "d"]
    one_domain = {n: "same" for n in names}
    assert splits.heldout_datasets(names, domains=one_domain) == [hash_order(names)[0]]
    own_domain = {n: n for n in names}
    assert splits.heldout_datasets(names, domains=own_domain) == hash_order(names)[:3]


def test_heldout_rejects_a_dataset_without_a_domain():
    with pytest.raises(ValueError, match="mystery"):
        splits.heldout_datasets(["gpqa", "mystery"])


def test_heldout_on_the_real_datasets_covers_three_domains():
    held = splits.heldout_datasets(sorted(selection.KEPT))
    assert len(held) == 3
    assert len({selection.DOMAINS[name] for name in held}) == 3
    assert held == splits.heldout_datasets(sorted(selection.KEPT, reverse=True))


def test_require_splits_rejects_test_rows():
    df = pd.DataFrame({"split": ["train", "calib", "test"]})
    with pytest.raises(ValueError, match="test"):
        splits.require_splits(df, {"train", "calib"})
    assert splits.require_splits(df, {"train", "calib", "test"}) is None
    assert splits.require_splits(df.iloc[:2], {"train", "calib"}) is None


def test_require_splits_names_every_offender_and_handles_missing_values():
    df = pd.DataFrame({"split": ["train", "test", "heldout", None]})
    with pytest.raises(ValueError) as err:
        splits.require_splits(df, {"train"})
    message = str(err.value)
    assert "test" in message and "heldout" in message and "<missing>" in message
    with pytest.raises(ValueError, match="split"):
        splits.require_splits(pd.DataFrame({"item_id": ["a/1"]}), {"train"})
