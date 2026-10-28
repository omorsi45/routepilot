import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from routepilot_train.runinfo import repo_root

SCRIPT = repo_root() / "bench" / "scripts" / "baselines.py"
FIXTURE = repo_root() / "testdata" / "replay"
SENTINEL = "SENTINEL-PROMPT-7f3a"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("baselines", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    """A derived directory around the replay fixture: top-c is Best Single, cheap-a cheapest."""
    root = tmp_path / "derived"
    (root / "pools").mkdir(parents=True)
    (root / "replay").mkdir()
    shutil.copy(FIXTURE / "config.yaml", root / "pools" / "flagship.config.yaml")
    shutil.copy(FIXTURE / "items.jsonl", root / "replay" / "flagship.jsonl")
    pool = {
        "pool": "flagship",
        "models": ["cheap-a", "mid-b", "top-c"],
        "best_single": "top-c",
        "cheapest": "cheap-a",
    }
    (root / "pools" / "flagship.json").write_text(json.dumps(pool), encoding="utf-8")
    items = [
        json.loads(line)
        for line in (FIXTURE / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    texts = ["\n".join(m["content"] for m in item["messages"]) for item in items]
    pd.DataFrame(
        {
            "item_id": [item["item_id"] for item in items],
            "prompt": texts,
            "prompt_bytes": [sum(len(m["content"].encode()) for m in i["messages"]) for i in items],
        }
    ).to_parquet(root / "prompts.parquet", index=False)
    return root


def run(derived: Path, binary: Path, tmp_path: Path) -> Path:
    return load_script().run_baselines(
        derived, "flagship", tmp_path / "out", ["python", "x.py"], binary, bootstrap=100
    )


def test_baseline_rows_on_the_test_split(derived: Path, binary: Path, tmp_path: Path):
    out = run(derived, binary, tmp_path)

    assert sorted(p.name for p in out.iterdir()) == [
        "pareto.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["pool"] == "flagship" and results["split"] == "test"
    assert results["n_items"] == 2
    assert results["best_single"] == "top-c" and results["cheapest"] == "cheap-a"
    rows = results["policies"]
    assert list(rows) == [
        "always_strongest",
        "always_cheapest",
        "random",
        "static_rules",
        "oracle",
        "static_strongest_rank",
        "static_cheapest_worstcase",
    ]

    # test items: fx-b/4 and fx-b/5; top-c recorded ok 250 and fail 140, cheap-a ok 250 and ok 120
    strongest = rows["always_strongest"]
    assert strongest["fixed_model"] == "top-c" and strongest["replay_policy"] == "fixed"
    assert strongest["success_rate"] == 0.5
    assert strongest["mean_cost_usd"] == pytest.approx(195e-12)
    assert strongest["success_delta_vs_strongest"] == 0.0
    assert strongest["cost_reduction_vs_strongest"] == 0.0

    cheapest = rows["always_cheapest"]
    assert cheapest["fixed_model"] == "cheap-a"
    assert cheapest["success_rate"] == 1.0
    assert cheapest["mean_cost_usd"] == pytest.approx(185e-12)
    assert cheapest["success_delta_vs_strongest"] == pytest.approx(0.5)
    assert cheapest["cost_reduction_vs_strongest"] == pytest.approx(1 - 185 / 195)
    lo, hi = cheapest["success_ci"]
    assert lo <= 1.0 <= hi

    # fx-b/4 has no "def " (default mid-b, ok 250); fx-b/5 has it (top-c, fail 140)
    assert rows["static_rules"]["success_rate"] == 0.5
    assert rows["static_rules"]["mean_cost_usd"] == pytest.approx(195e-12)
    # the oracle takes cheap-a on both (tie on fx-b/4 goes to the lower name)
    assert rows["oracle"]["success_rate"] == 1.0
    assert rows["oracle"]["mean_cost_usd"] == pytest.approx(185e-12)
    # the gateway's own policies: strongest by rank is top-c, cheapest by worst case is cheap-a
    assert rows["static_strongest_rank"]["fixed_model"] is None
    assert rows["static_strongest_rank"]["model_counts"] == {"top-c": 2}
    assert rows["static_cheapest_worstcase"]["model_counts"] == {"cheap-a": 2}
    for row in rows.values():
        assert row["n"] == 2 and row["errors"] == 0

    spread = results["random_seeds"]
    assert list(spread["seeds"]) == ["0", "1", "2", "3", "4"]
    assert (
        spread["success_rate"]["min"]
        <= spread["success_rate"]["mean"]
        <= spread["success_rate"]["max"]
    )
    assert rows["random"]["success_rate"] == spread["seeds"]["0"]["success_rate"]

    assert set(results["per_dataset"]) == {"fx-b"} and results["per_dataset"]["fx-b"]["n"] == 2
    assert set(results["pareto_front"]) <= set(rows)
    assert "oracle" in results["pareto_front"]  # best quality at the lowest cost: nothing beats it


def test_static_rules_go_python_parity_is_checked_on_every_item(
    derived: Path, binary: Path, tmp_path: Path
):
    results = json.loads(
        (run(derived, binary, tmp_path) / "results.json").read_text(encoding="utf-8")
    )
    parity = results["rules_parity"]
    assert parity["split"] == "all" and parity["items"] == 5
    assert parity["mismatches"] == 0 and parity["mismatched_item_ids"] == []
    assert results["rules_mapping"] == {"default_model": "mid-b", "rules": {"code": "top-c"}}


def test_parity_check_reports_a_disagreement_by_item_id(
    derived: Path, binary: Path, tmp_path: Path
):
    # make the Python side see a keyword that the Go side was never given
    prompts = pd.read_parquet(derived / "prompts.parquet")
    prompts.loc[prompts["item_id"] == "fx-a/2", "prompt"] = "def forced match"
    prompts.to_parquet(derived / "prompts.parquet", index=False)

    results = json.loads(
        (run(derived, binary, tmp_path) / "results.json").read_text(encoding="utf-8")
    )
    assert results["rules_parity"]["mismatches"] == 1
    assert results["rules_parity"]["mismatched_item_ids"] == ["fx-a/2"]


def test_report_labels_bootstrap_and_holds_no_prompt_text(
    derived: Path, binary: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path)
    for path in out.iterdir():
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# baselines\n") and "**Labels:**" in md
    assert "recorded costs" in md and "oracle looks at the recorded outcomes" in md
    for heading in (
        "Setup",
        "Quality and cost on the test split",
        "Against always_strongest (paired by item)",
        "Random policy over seeds 0 to 4",
        "Success rate by dataset",
    ):
        assert f"## {heading}" in md
    assert "![pareto](pareto.png)" in md
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "baselines"
    assert run_json["seeds"] == {"random": [0, 1, 2, 3, 4], "bootstrap": 0}


def test_mixed_pool_labels_small_model_costs_as_fitted(derived: Path, binary: Path, tmp_path):
    for suffix in (".json", ".config.yaml"):
        shutil.copy(derived / "pools" / f"flagship{suffix}", derived / "pools" / f"mixed{suffix}")
    shutil.copy(derived / "replay" / "flagship.jsonl", derived / "replay" / "mixed.jsonl")
    out = load_script().run_baselines(
        derived, "mixed", tmp_path / "out", None, binary, bootstrap=50
    )
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "fitted from recorded costs" in md
