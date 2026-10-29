"""Tests of bench/scripts/live_train.py: splits, the pool with assumed prices, the measured
latency profiles, the replay rows and one run on synthetic live data. Every prompt and number
here is invented; the run uses the real Go replay command, never Ollama."""

import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from routepilot_train import replaytool
from routepilot_train.export import artifact as artifact_io
from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import live_collect as lc  # noqa: E402
import live_train as lt  # noqa: E402

SENTINEL = "PROMPT-SENTINEL-55e0"
EM_DASH = "\u2014"
MODELS = {"small": 0.2, "mid": 0.45, "big": 0.7}  # id -> chance of a right answer
TPOT_MS = {"small": 6.0, "mid": 10.0, "big": 13.0}


@pytest.fixture(scope="session")
def binary() -> Path:
    try:
        return replaytool.build_go(repo_root())
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))


def catalog() -> list[lc.CatalogModel]:
    return [
        lc.CatalogModel("small", "m:0.5b", 0.5, Decimal("0.01"), Decimal("0.03")),
        lc.CatalogModel("mid", "m:1.5b", 1.5, Decimal("0.03"), Decimal("0.09")),
        lc.CatalogModel("big", "m:3b", 3.0, Decimal("0.06"), Decimal("0.18")),
    ]


def synthetic_live(n_gpqa: int = 50, n_mmlu: int = 110, seed: int = 3):
    """prompts, outcomes and measured tables of a small invented live collection."""
    rng = np.random.default_rng(seed)
    ids = [f"gpqa/{i}" for i in range(n_gpqa)] + [f"mmlupro/{i}" for i in range(n_mmlu)]
    prompts, outcomes, measured = [], [], []
    for k, item_id in enumerate(ids):
        text = f"{SENTINEL} {item_id} topic{k % 7} " + "filler " * (k % 11 + 3)
        prompts.append(
            {
                "dataset": item_id.split("/")[0],
                "item_id": item_id,
                "prompt": text,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_bytes": len(text.encode()),
            }
        )
        for c in catalog():
            ok = bool(rng.random() < MODELS[c.id])
            pt, ct = len(text) // 3 + 10, int(rng.integers(120, 385))
            outcomes.append(
                {
                    "dataset": item_id.split("/")[0],
                    "item_id": item_id,
                    "model": c.id,
                    "score": 1.0 if ok else 0.0,
                    "success": ok,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cost_usd": c.cost_usd(pt, ct),
                    "prompt_sha256": prompts[-1]["prompt_sha256"],
                }
            )
            ttft = float(rng.lognormal(np.log(40.0), 0.3))
            measured.append(
                {
                    "item_id": item_id,
                    "model": c.id,
                    "decision_id": f"dec_{k}_{c.id}",
                    "ttft_ms": ttft,
                    "e2e_ms": ttft + ct * TPOT_MS[c.id],
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "finish_reason": "length",
                    "how": "cue",
                    "client_wall_ms": ttft + ct * TPOT_MS[c.id],
                }
            )
    return pd.DataFrame(prompts), pd.DataFrame(outcomes), pd.DataFrame(measured)


# --- splits and pool -----------------------------------------------------------------------------


def test_live_splits_apply_the_salt_rule_with_no_heldout_dataset() -> None:
    prompts, outcomes, _ = synthetic_live(300, 700)
    splits = lt.live_splits(outcomes)
    assert list(splits.columns) == ["item_id", "dataset", "split"]
    assert splits["item_id"].is_unique and len(splits) == 1000
    assert set(splits["split"]) == {"train", "calib", "test"}
    shares = splits["split"].value_counts(normalize=True)
    for name, want in (("train", 0.6), ("calib", 0.2), ("test", 0.2)):
        assert abs(shares[name] - want) < 0.05
    again = lt.live_splits(outcomes.sample(frac=1.0, random_state=1))
    assert again.equals(splits)  # a split depends on the item ID alone


def test_the_pool_uses_assumed_prices_and_the_tuning_splits_only() -> None:
    prompts, outcomes, _ = synthetic_live()
    splits = lt.live_splits(outcomes)
    # "small" wins every test item and loses every train and calibration item
    test_ids = set(splits.loc[splits["split"] == "test", "item_id"])
    in_test = outcomes["item_id"].isin(test_ids)
    is_small = outcomes["model"] == "small"
    outcomes.loc[is_small & in_test, "score"] = 1.0
    outcomes.loc[is_small & in_test, "success"] = True
    outcomes.loc[is_small & ~in_test, "score"] = 0.0
    outcomes.loc[is_small & ~in_test, "success"] = False
    pool = lt.build_live_pool(outcomes, prompts, splits, catalog())
    assert pool["pool"] == "live-ollama"
    assert pool["models"] == ["big", "mid", "small"]
    assert pool["best_single"] == "big"  # not "small": its wins are all on test rows
    assert pool["cheapest"] == "small"
    assert pool["rank"]["big"] == 3 and pool["rank"]["small"] == 1
    assert pool["n_items"] == len(prompts)
    price = pool["prices"]["mid"]
    assert price["in_usd_per_token"] == pytest.approx(0.03e-6)
    assert price["out_usd_per_token"] == pytest.approx(0.09e-6)
    assert "assumed" in price["price_source"] and "ASSUMPTION" in price["price_source"]
    assert set(pool["prompt_overhead_tokens"]) == set(pool["models"])


def test_the_pool_needs_an_item_every_model_answered() -> None:
    prompts, outcomes, _ = synthetic_live(5, 5)
    outcomes = outcomes.loc[outcomes["model"] != "big"]
    with pytest.raises(ValueError, match="no item has"):
        lt.build_live_pool(outcomes, prompts, lt.live_splits(outcomes), catalog())


def test_only_the_replay_input_holds_prompt_text(tmp_path: Path) -> None:
    prompts, outcomes, _ = synthetic_live()
    splits = lt.live_splits(outcomes)
    pool = lt.build_live_pool(outcomes, prompts, splits, catalog())
    info = lt.write_live_tables(tmp_path, pool, outcomes, prompts, splits)
    assert info["replay_lines"] == len(prompts)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            holds = SENTINEL.encode() in path.read_bytes()
            assert holds == (path.name == "live-ollama.jsonl"), path.name
    cfg = yaml.safe_load((tmp_path / "pools" / "live-ollama.config.yaml").read_text("utf-8"))
    by_id = {m["id"]: m for m in cfg["models"]}
    assert by_id["small"]["input_price_per_mtok"] == "0.010000"
    assert by_id["big"]["output_price_per_mtok"] == "0.180000"
    assert by_id["big"]["rank"] == 3
    first = json.loads(
        (tmp_path / "replay" / "live-ollama.jsonl").read_text("utf-8").splitlines()[0]
    )
    assert set(first["outcomes"]) == {"small", "mid", "big"}
    assert pd.read_parquet(tmp_path / "splits.parquet").equals(splits)


# --- latency profiles -----------------------------------------------------------------------------


def test_profiles_are_medians_and_log_spreads_of_the_tuning_requests() -> None:
    splits = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "dataset": "x",
            "split": ["train", "calib", "train", "test"],
        }
    )
    measured = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "model": "m",
            "ttft_ms": [100.0, 200.0, 400.0, 99999.0],
            "e2e_ms": [1100.0, 2200.0, 4400.0, 9e9],
            "completion_tokens": [100, 200, 400, 1],
        }
    )
    p = lt.latency_profiles(measured, splits)["m"]
    assert p["n"] == 3  # the test request is never read
    assert p["ttft_median_ms"] == 200.0
    assert p["ttft_sigma"] == pytest.approx(float(np.std(np.log([100, 200, 400]), ddof=1)))
    # tpot = (e2e - ttft) / completion tokens = 10 ms for all three
    assert p["tpot_median_ms"] == pytest.approx(10.0)
    assert p["tpot_sigma"] == pytest.approx(0.0, abs=1e-12)


def test_profiles_need_two_requests_per_model() -> None:
    splits = pd.DataFrame({"item_id": ["a"], "dataset": "x", "split": ["train"]})
    measured = pd.DataFrame(
        {
            "item_id": ["a"],
            "model": "m",
            "ttft_ms": [5.0],
            "e2e_ms": [50.0],
            "completion_tokens": [9],
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        lt.latency_profiles(measured, splits)


def test_the_profile_file_says_measured_and_names_the_machine(tmp_path: Path) -> None:
    profiles = {"m": {"ttft_median_ms": 41.23456, "ttft_sigma": 0.31, "tpot_median_ms": 7.5,
                      "tpot_sigma": 0.05}}  # fmt: skip
    path = tmp_path / "bench" / "latency" / "live.yaml"
    lt.write_profiles(path, profiles, "CPU-X (8 threads) and GPU-Y", "2026-10-29")
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text and EM_DASH not in text
    entry = yaml.safe_load(text)["profiles"][0]
    assert entry["model"] == "m"
    assert entry["source"] == "measured"
    assert entry["retrieved"] == "2026-10-29"  # a string, as the Go loader wants
    assert entry["note"] == "measured through RoutePilot on CPU-X (8 threads) and GPU-Y, 2026-10-29"
    assert (entry["ttft_median_ms"], entry["tpot_median_ms"]) == (41.2346, 7.5)


# --- the replay table -----------------------------------------------------------------------------


def frame(models: list[str | None], success: list[bool], cost: list[int]) -> pd.DataFrame:
    n = len(models)
    return pd.DataFrame(
        {
            "item_id": [f"i{k}" for k in range(n)],
            "dataset": "x",
            "split": "test",
            "policy": "fixed",
            "model": models,
            "success": success,
            "cost_picousd": cost,
            "error": "",
            "relaxed": [[] for _ in range(n)],
            "ood": [False] * n,
        }
    )


def test_measured_latency_looks_up_the_chosen_model_on_each_item() -> None:
    measured = pd.DataFrame(
        {
            "item_id": ["i0", "i0", "i1", "i1"],
            "model": ["a", "b", "a", "b"],
            "e2e_ms": [100.0, 900.0, 300.0, 700.0],
        }
    )
    got = lt.measured_latency(frame(["a", "b"], [True, True], [1, 1]), measured)
    assert got["n_timed"] == 2
    assert got["e2e_ms_p50"] == pytest.approx(np.percentile([100.0, 700.0], 50))
    assert got["e2e_ms_p95"] == pytest.approx(np.percentile([100.0, 700.0], 95))
    none = lt.measured_latency(frame([None, None], [False, False], [0, 0]), measured)
    assert none == {"e2e_ms_p50": None, "e2e_ms_p95": None, "n_timed": 0}


def test_replay_rows_pair_each_policy_with_the_reference() -> None:
    reference = frame(["big"] * 4, [True, True, False, True], [100, 100, 100, 100])
    other = frame(["small"] * 4, [True, False, False, True], [10, 10, 10, 10])
    measured = pd.DataFrame(
        {
            "item_id": [f"i{k}" for k in range(4)] * 2,
            "model": ["big"] * 4 + ["small"] * 4,
            "e2e_ms": [4000.0] * 4 + [1000.0] * 4,
        }
    )
    rows = lt.replay_rows({"always_strongest": reference, "cheap": other}, measured, bootstrap=50)
    by = {r["policy"]: r for r in rows}
    assert by["always_strongest"]["success_rate"] == 0.75
    assert by["cheap"]["success_rate"] == 0.5
    assert by["cheap"]["cost_reduction_vs_strongest"] == pytest.approx(0.9)
    assert by["cheap"]["e2e_ms_p50"] == 1000.0
    assert by["always_strongest"]["e2e_ms_p95"] == 4000.0
    assert by["cheap"]["model_counts"] == {"small": 4}
    text = lt._replay_table(rows).to_csv()
    assert "always_strongest" in text and "cheap" in text


def test_the_policy_config_is_the_one_the_experiment_names() -> None:
    cfg = yaml.safe_load(lt.POLICY_YAML)
    assert cfg == {
        "mode": "min_cost",
        "default_floor": 0,
        "epsilon": 0,
        "ood_action": "conservative",
        "on_infeasible": "best_effort",
    }
    assert EM_DASH not in lt.POLICY_YAML
    assert not hasattr(lt, "FLOORS")  # the floors come from calibration, not from a constant


# --- what the models were sent --------------------------------------------------------------------


@pytest.mark.parametrize("name", lc.PROTOCOLS)
def test_the_sent_view_holds_the_messages_the_protocol_sends(name: str) -> None:
    prompts, _, _ = synthetic_live(3, 4)
    protocol = lc.protocol_for(name)
    before = prompts.copy()
    sent = lt.sent_view(prompts, protocol)
    assert prompts.equals(before)  # the collection's table is not touched
    assert sent["item_id"].tolist() == prompts["item_id"].tolist()
    for row, original in zip(sent.itertuples(), prompts.itertuples(), strict=True):
        messages = json.loads(row.messages)
        assert messages == protocol.body("m", original.prompt)["messages"]
        texts = [m["content"] for m in messages]
        assert row.prompt == "\n".join(texts)  # what the gateway's rules and features read
        assert row.prompt_bytes == sum(len(t.encode()) for t in texts)  # the gateway's PromptBytes
        assert row.prompt_sha256 == original.prompt_sha256  # still the dataset item's identity
    if name == "as-written":
        assert sent["prompt"].tolist() == prompts["prompt"].tolist()
    else:
        assert (sent["prompt_bytes"] > prompts["prompt_bytes"]).all()


def test_floor_values_are_the_median_and_the_ninetieth_percentile_rounded_down() -> None:
    best = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert lt.floor_values(best) == (0.55, 0.91)  # the 90th percentile is 0.91 exactly
    # rounded down, never to the nearest: 0.1239999 -> 0.123 and 0.9999 -> 0.999
    assert lt.floor_values([0.1239999, 0.1239999]) == (0.123, 0.123)
    assert lt.floor_values([0.9999, 0.9999, 0.9999]) == (0.999, 0.999)
    assert lt.floor_values([0.3, 0.3]) == (0.3, 0.3)  # no float noise on a clean value
    with pytest.raises(ValueError, match="no lower bounds"):
        lt.floor_values([])


def test_artifact_size_adds_up_the_files(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b"x" * 10)
    (tmp_path / "weights.bin").write_bytes(b"y" * 32)
    assert lt.artifact_size(tmp_path) == 42


# --- one run --------------------------------------------------------------------------------------


def test_a_run_trains_exports_profiles_and_replays_the_test_split(
    binary: Path, tmp_path: Path
) -> None:
    prompts, outcomes, measured = synthetic_live()
    live = tmp_path / "live"
    live.mkdir()
    prompts.to_parquet(live / "prompts.parquet", index=False)
    outcomes.to_parquet(live / "outcomes.parquet", index=False)
    measured.to_parquet(live / "measured.parquet", index=False)
    config = tmp_path / "config.yaml"
    base = yaml.safe_load(lc.CONFIG.read_text(encoding="utf-8"))
    base["models"] = [
        {**m, "id": c.id, "deployments": [{**m["deployments"][0], "upstream_model": c.tag}]}
        for m, c in zip(base["models"], catalog(), strict=True)
    ]
    for m, c in zip(base["models"], catalog(), strict=True):
        m["input_price_per_mtok"], m["output_price_per_mtok"] = (
            str(c.in_usd_per_mtok),
            str(c.out_usd_per_mtok),
        )
    config.write_text(yaml.safe_dump(base), encoding="utf-8")

    out = lt.run_live_train(
        live,
        tmp_path / "policies",
        tmp_path / "results",
        config=config,
        policy_path=tmp_path / "bench" / "policies" / "live.yaml",
        profiles_path=tmp_path / "bench" / "latency" / "live.yaml",
        dim=1024,
        bootstrap=50,
        workers=2,
        binary=binary,
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    version = results["artifact"]["version"]
    artifact = tmp_path / "policies" / "live-ollama" / version
    assert artifact.is_dir() and results["artifact"]["dim"] == 1024
    assert results["artifact"]["size_bytes"] < lt.MAX_ARTIFACT_BYTES
    # Go loads and verifies what Python exported
    verified = subprocess.run(
        [str(binary), "policy", "verify", str(artifact)], capture_output=True, text=True
    )
    assert verified.returncode == 0 and verified.stdout.strip() == version
    # the profile file is the one the gateway reads
    profiles = yaml.safe_load((tmp_path / "bench" / "latency" / "live.yaml").read_text("utf-8"))
    assert {p["model"] for p in profiles["profiles"]} == set(MODELS)
    assert all(p["source"] == "measured" for p in profiles["profiles"])
    assert (tmp_path / "bench" / "policies" / "live.yaml").read_text() == lt.POLICY_YAML
    # one replay row per baseline and per floor, all on the same test items
    rows = {r["policy"]: r for r in results["replay"]["rows"]}
    assert {"always_strongest", "always_cheapest", "random", "static_rules", "oracle"} <= set(rows)
    # the floors are 0, the median and the 90th percentile of the best lower bound over the
    # calibration items, recomputed here from the exported artifact and the messages sent
    floors = results["floors"]
    assert floors["source"] == "calibration items only"
    assert floors["values"][0] == 0.0
    assert floors["values"] == sorted(set(floors["values"]))
    manifest, blob = artifact_io.load_artifact(artifact)
    sent = lt.sent_view(prompts, lc.protocol_for("as-written")).set_index("item_id")
    calib_ids = sorted(
        pd.read_parquet(live / "splits.parquet").query("split == 'calib'")["item_id"]
    )
    best = []
    for item_id in calib_ids:
        messages = [(m["role"], m["content"]) for m in json.loads(sent.loc[item_id, "messages"])]
        predicted = artifact_io.predict_bundle(manifest, blob, messages)
        best.append(max(m["l"] for m in predicted["models"].values()))
    assert floors["n_calibration_items"] == len(calib_ids) == results["fit"]["n_calib"]
    assert floors["values"] == sorted({0.0, *lt.floor_values(best)})
    assert floors["best_lower_bound"]["median"] == pytest.approx(float(np.median(best)))
    assert floors["best_lower_bound"]["p90"] == pytest.approx(float(np.percentile(best, 90)))
    assert {f"learned_min_cost_floor_{f:g}" for f in floors["values"]} <= set(rows)
    n_test = results["replay"]["n_items"]
    assert n_test > 10 and all(r["n"] == n_test for r in rows.values())
    assert rows["always_strongest"]["fixed_model"] == results["pool"]["best_single"]
    assert all(r["e2e_ms_p50"] is not None for r in rows.values())
    # floor 0 picks the cheapest model for every request
    assert rows["learned_min_cost_floor_0"]["model_counts"] == {results["pool"]["cheapest"]: n_test}
    # the items in the test replay are exactly the test split
    assert (pd.read_parquet(live / "splits.parquet")["split"] == "test").sum() == n_test
    # nothing in the report holds prompt text, and every label is there
    for name in ("results.json", "report.md", "run.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert SENTINEL not in text and EM_DASH not in text
    assert "ASSUMPTION" in (out / "report.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["experiment"] == "live_train" and run["policy_version"] == version
