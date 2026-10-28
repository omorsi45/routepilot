import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from frames import (
    MIXED_DATASETS,
    PER_DATASET,
    SENTINEL,
    SMALL,
    prompt_for,
    sha,
    synthetic_outcomes,
    write_derived,
)

from routepilot_train.baselines.static_rules import RULES
from routepilot_train.data import build, pools, selection


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    return write_derived(tmp_path / "derived")


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pool_config_rules_are_mapped_on_train_and_calib_items(derived: Path):
    """Odd-numbered items get code text, and glm-4.6 solves all of them, test split included."""
    prompts = pd.read_parquet(derived / "prompts.parquet")
    odd = prompts["item_id"].map(lambda item_id: int(item_id.split("/")[1]) % 2 == 1)
    prompts.loc[odd, "prompt"] = "def f(): " + prompts.loc[odd, "prompt"]
    prompts["prompt_sha256"] = [sha(p) for p in prompts["prompt"]]
    prompts["prompt_bytes"] = [len(p.encode()) for p in prompts["prompt"]]
    prompts.to_parquet(derived / "prompts.parquet", index=False)
    outcomes = pd.read_parquet(derived / "outcomes.parquet")
    solved = outcomes["model"].eq("glm-4.6") & outcomes["item_id"].map(
        lambda item_id: int(item_id.split("/")[1]) % 2 == 1
    )
    outcomes.loc[solved, ["score", "success"]] = [1.0, True]
    outcomes.to_parquet(derived / "outcomes.parquet", index=False)

    build.build(derived)

    pool = json.loads((derived / "pools" / "flagship.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((derived / "pools" / "flagship.config.yaml").read_text("utf-8"))
    rules = {r["name"]: r["model"] for r in cfg["routing"]["rules"]}
    assert pool["best_single"] != "glm-4.6"
    assert rules == {
        "code": "glm-4.6",
        "math": pool["best_single"],
        "choices": pool["best_single"],
        "long": pool["best_single"],
    }
    assert cfg["routing"]["default_model"] == pool["best_single"]
    assert cfg["routing"]["rules"][0]["when"]["contains_any"][:2] == ["```", "def "]


def test_build_writes_splits_pools_configs_and_replay(derived: Path):
    summary = build.build(derived)

    splits = pd.read_parquet(derived / "splits.parquet")
    assert list(splits.columns) == ["item_id", "dataset", "split"]
    assert len(splits) == len(selection.KEPT) * PER_DATASET
    assert splits["item_id"].is_unique
    assert set(splits["split"]) == {"train", "calib", "test", "heldout"}
    heldout = sorted(splits.loc[splits["split"] == "heldout", "dataset"].unique())
    assert len(heldout) == 3
    assert len({selection.DOMAINS[d] for d in heldout}) == 3
    assert summary["heldout_datasets"] == heldout

    flagship = json.loads((derived / "pools" / "flagship.json").read_text(encoding="utf-8"))
    assert flagship["pool"] == "flagship"
    assert flagship["models"] == sorted(selection.FLAGSHIP_MODELS)
    assert flagship["n_items"] == len(selection.KEPT) * PER_DATASET - 1  # hle/3 lacks gpt-5
    assert sorted(flagship["datasets"]) == sorted(selection.KEPT)
    assert flagship["best_single"] in flagship["models"]
    assert flagship["rank"][flagship["best_single"]] == 12
    assert sorted(flagship["rank"].values()) == list(range(1, 13))

    mixed = json.loads((derived / "pools" / "mixed.json").read_text(encoding="utf-8"))
    assert mixed["pool"] == "mixed"
    assert mixed["datasets"] == MIXED_DATASETS
    assert mixed["n_items"] == len(MIXED_DATASETS) * PER_DATASET
    assert len(mixed["models"]) == 16
    assert set(selection.FLAGSHIP_MODELS) <= set(mixed["models"])
    assert "gpt-4.1" not in mixed["models"]  # large: never one of the four small models
    assert summary["seven_b_models"] == sorted(
        set(mixed["models"]) - set(selection.FLAGSHIP_MODELS)
    )

    for name in ("flagship", "mixed"):
        cfg = yaml.safe_load((derived / "pools" / f"{name}.config.yaml").read_text("utf-8"))
        pool = flagship if name == "flagship" else mixed
        assert [m["id"] for m in cfg["models"]] == pool["models"]
        assert {m["id"]: m["rank"] for m in cfg["models"]} == pool["rank"]
        routing = cfg["routing"]
        assert routing["policy"] == "rules"
        assert routing["default_model"] == pool["best_single"]
        assert [r["name"] for r in routing["rules"]] == [r["name"] for r in RULES]
        assert [r["when"] for r in routing["rules"]] == [r["when"] for r in RULES]
        assert {r["model"] for r in routing["rules"]} <= set(pool["models"])
        assert (derived / "replay" / f"{name}.jsonl").is_file()

    assert summary["pools"]["flagship"]["replay_lines"] == flagship["n_items"]
    assert summary["pools"]["mixed"]["replay_lines"] == mixed["n_items"]
    assert len(read_lines(derived / "replay" / "flagship.jsonl")) == flagship["n_items"]


def test_mixed_picks_are_the_top_train_success_small_models_with_recorded_cost(derived: Path):
    build.build(derived)
    outcomes = pd.read_parquet(derived / "outcomes.parquet")
    splits = pd.read_parquet(derived / "splits.parquet")
    train = outcomes.merge(splits[["item_id", "split"]], on="item_id")
    train = train.loc[(train["split"] == "train") & train["dataset"].isin(MIXED_DATASETS)]
    rate = train.groupby("model")["success"].mean()
    eligible = {m: rate[m] for m in SMALL if m != "s-e"}  # s-e has no recorded cost
    want = sorted(sorted(eligible), key=lambda m: -eligible[m])[:4]
    mixed = json.loads((derived / "pools" / "mixed.json").read_text(encoding="utf-8"))
    picked = sorted(set(mixed["models"]) - set(selection.FLAGSHIP_MODELS))
    assert picked == sorted(want)


def test_mixed_replay_prices_small_models_from_fitted_prices(derived: Path):
    build.build(derived)
    mixed = json.loads((derived / "pools" / "mixed.json").read_text(encoding="utf-8"))
    small = sorted(set(mixed["models"]) - set(selection.FLAGSHIP_MODELS))
    outcomes = pd.read_parquet(derived / "outcomes.parquet").set_index(["item_id", "model"])
    lines = read_lines(derived / "replay" / "mixed.jsonl")
    assert len(lines) == mixed["n_items"]
    saw_recorded_zero = False
    for line in lines:
        for model, got in line["outcomes"].items():
            rec = outcomes.loc[(line["item_id"], model)]
            if model in small:
                price = mixed["prices"][model]
                want = round(
                    (
                        price["in_usd_per_token"] * rec["prompt_tokens"]
                        + price["out_usd_per_token"] * rec["completion_tokens"]
                    )
                    * 1e12
                )
                saw_recorded_zero |= rec["cost_usd"] == 0 and got["cost_picousd"] > 0
            else:
                want = round(rec["cost_usd"] * 1e12)
            assert got["cost_picousd"] == want, (line["item_id"], model)
            assert got["success"] == bool(rec["success"])
    assert saw_recorded_zero  # a small model with zero recorded cost now has a fitted cost
    for model in small:
        assert mixed["prices"][model]["price_source"] == "fitted_from_recorded_cost"


def test_flagship_replay_keeps_recorded_costs_and_heldout_items(derived: Path):
    build.build(derived)
    outcomes = pd.read_parquet(derived / "outcomes.parquet").set_index(["item_id", "model"])
    lines = read_lines(derived / "replay" / "flagship.jsonl")
    assert [x["item_id"] for x in lines] == sorted(x["item_id"] for x in lines)
    assert "hle/3" not in {x["item_id"] for x in lines}
    assert "heldout" in {x["split"] for x in lines}
    for line in lines[:50]:
        for model, got in line["outcomes"].items():
            rec = outcomes.loc[(line["item_id"], model)]
            assert got["cost_picousd"] == round(rec["cost_usd"] * 1e12)
        assert line["messages"] == [{"role": "user", "content": prompt_for(line["item_id"])}]
        assert line["prompt_sha256"] == sha(prompt_for(line["item_id"]))


def test_only_the_replay_export_holds_prompt_text(derived: Path):
    build.build(derived)
    for path in [derived / "splits.parquet", *sorted((derived / "pools").iterdir())]:
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    assert SENTINEL in (derived / "replay" / "flagship.jsonl").read_text(encoding="utf-8")


def test_build_is_deterministic(derived: Path):
    build.build(derived)
    first = {
        p.relative_to(derived).as_posix(): p.read_bytes()
        for p in sorted(derived.rglob("*"))
        if p.is_file() and p.name not in ("outcomes.parquet", "prompts.parquet")
    }
    build.build(derived)
    second = {
        p.relative_to(derived).as_posix(): p.read_bytes()
        for p in sorted(derived.rglob("*"))
        if p.is_file() and p.name not in ("outcomes.parquet", "prompts.parquet")
    }
    assert first == second
    assert set(first) >= {
        "splits.parquet",
        "pools/flagship.json",
        "pools/mixed.json",
        "pools/flagship.config.yaml",
        "pools/mixed.config.yaml",
        "replay/flagship.jsonl",
        "replay/mixed.jsonl",
    }


def test_build_refuses_a_mixed_pool_with_too_few_small_models(tmp_path: Path):
    outcomes = synthetic_outcomes()
    outcomes = outcomes.loc[~outcomes["model"].isin(["s-a", "s-b", "s-c", "s-d", "s-f", "gpt-4.1"])]
    directory = write_derived(tmp_path / "derived", outcomes)
    with pytest.raises(ValueError, match="small models"):
        build.build(directory)
    assert not (directory / "pools").exists()  # nothing is written when the build fails
    assert not (directory / "splits.parquet").exists()


def test_main_builds_from_the_command_line(derived: Path, capsys: pytest.CaptureFixture[str]):
    assert build.main(["--derived", str(derived)]) == 0
    printed = capsys.readouterr().out
    assert "flagship" in printed and "mixed" in printed
    assert SENTINEL not in printed
    key_hash = hashlib.sha256(pools.REPLAY_KEY.encode()).hexdigest()
    assert key_hash in (derived / "pools" / "mixed.config.yaml").read_text(encoding="utf-8")
