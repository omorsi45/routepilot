import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from frames import SENTINEL, synthetic_outcomes, write_derived

from routepilot_train.data import build, selection

SCRIPT = Path(__file__).resolve().parents[3] / "bench" / "scripts" / "data_summary.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("data_summary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    outcomes = synthetic_outcomes()
    # three zero-cost gemini-2.5-flash aime records, two of them with no completion tokens
    zero = (outcomes["model"] == "gemini-2.5-flash") & outcomes["item_id"].isin(
        ["aime/0", "aime/1", "aime/2"]
    )
    outcomes.loc[zero, "cost_usd"] = 0.0
    outcomes.loc[zero & outcomes["item_id"].isin(["aime/0", "aime/1"]), "completion_tokens"] = 0
    # one record with a negative completion token count, as gpt-5 has in the real data
    negative = (outcomes["model"] == "gpt-5") & (outcomes["item_id"] == "gpqa/4")
    outcomes.loc[negative, "completion_tokens"] = -7
    outcomes.loc[negative, "cost_usd"] = 0.0
    write_derived(tmp_path / "derived", outcomes)
    build.build(tmp_path / "derived")
    return tmp_path / "derived"


def test_summary_writes_results_report_and_run_json(derived: Path, tmp_path: Path):
    run_dir = load_script().summarize(derived, tmp_path / "results" / "data", ["python", "x.py"])
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{7,}|\d{8}T\d{6}Z-nogit", run_dir.name)
    assert sorted(p.name for p in run_dir.iterdir()) == ["report.md", "results.json", "run.json"]

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["experiment"] == "data" and run["run_id"] == run_dir.name
    assert run["command"] == ["python", "x.py"]

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    splits = pd.read_parquet(derived / "splits.parquet")
    assert results["heldout_datasets"] == sorted(
        splits.loc[splits["split"] == "heldout", "dataset"].unique()
    )
    assert set(results["heldout_domains"]) == set(results["heldout_datasets"])
    totals = results["split_totals"]
    assert sum(totals.values()) == len(splits)
    assert totals["heldout"] == int((splits["split"] == "heldout").sum())
    per_dataset = results["items_per_dataset_split"]
    assert set(per_dataset) == set(selection.KEPT)
    assert all(sum(counts.values()) == 30 for counts in per_dataset.values())

    flagship = results["pools"]["flagship"]
    assert flagship["n_items"] == len(selection.KEPT) * 30 - 1
    assert flagship["best_single"] in flagship["models"]
    assert sum(flagship["items_per_split"].values()) == flagship["n_items"]
    for model in flagship["models"]:
        assert set(flagship["success_by_split"][model]) == {"train", "calib", "test", "heldout"}
        price = flagship["prices"][model]
        assert price["n_fit"] > 0 and 0.0 <= price["fit_r2"] <= 1.0
        assert price["price_source"] == "fitted_from_recorded_cost"
    assert results["r2_range"]["flagship"][0] <= results["r2_range"]["flagship"][1]

    mixed = results["pools"]["mixed"]
    assert len(mixed["seven_b_models"]) == 4
    assert mixed["seven_b_cost_basis"] == "fitted from recorded costs"
    assert mixed["datasets"] == sorted(mixed["datasets"]) and "hle" not in mixed["datasets"]


def test_summary_counts_zero_cost_records_and_negative_tokens(derived: Path, tmp_path: Path):
    run_dir = load_script().summarize(derived, tmp_path / "out")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cell = results["zero_cost_flagship"]["gemini-2.5-flash"]["aime"]
    assert cell == {"records": 30, "zero_cost": 3, "zero_cost_zero_completion": 2}
    assert results["zero_cost_flagship"]["gemini-2.5-pro"]["aime"]["zero_cost"] == 0
    assert set(results["zero_cost_flagship"]) == set(selection.FLAGSHIP_MODELS)
    assert results["negative_token_rows"] == [
        {
            "item_id": "gpqa/4",
            "model": "gpt-5",
            "prompt_tokens": results["negative_token_rows"][0]["prompt_tokens"],
            "completion_tokens": -7,
            "cost_usd": 0.0,
        }
    ]


def test_summary_flags_fits_with_low_r2(derived: Path, tmp_path: Path):
    path = derived / "pools" / "flagship.json"
    pool = json.loads(path.read_text(encoding="utf-8"))
    assert all(p["fit_r2"] >= 0.9 for p in pool["prices"].values())
    pool["prices"]["gpt-5-chat"]["fit_r2"] = 0.89
    path.write_text(json.dumps(pool), encoding="utf-8", newline="\n")

    run_dir = load_script().summarize(derived, tmp_path / "out")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["pools"]["flagship"]["low_r2_models"] == ["gpt-5-chat"]
    assert results["pools"]["mixed"]["low_r2_models"] == []
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "below 0.9" in report and "gpt-5-chat" in report


def test_summary_report_labels_fitted_prices_and_holds_no_prompt_text(
    derived: Path, tmp_path: Path
):
    run_dir = load_script().summarize(derived, tmp_path / "out")
    report = (run_dir / "report.md").read_bytes()
    results = (run_dir / "results.json").read_bytes()
    for raw in (report, results, (run_dir / "run.json").read_bytes()):
        assert SENTINEL.encode() not in raw
        assert "\u2014".encode() not in raw
        assert b"\r" not in raw
    text = report.decode("utf-8")
    assert "fitted from recorded costs" in text
    assert "Best Single" in text and "Held-out datasets" in text
    assert "Zero recorded cost" in text and "Negative token counts" in text
    pool_json = json.loads((derived / "pools" / "mixed.json").read_text(encoding="utf-8"))
    for model in pool_json["models"]:
        assert model in text
    small = sorted(set(pool_json["models"]) - set(selection.FLAGSHIP_MODELS))
    for model in small:  # the replay export prices these from fitted prices
        assert re.search(rf"^\| {re.escape(model)} \|.*\| fitted \|", text, re.MULTILINE), model
    assert "| recorded |" in text
