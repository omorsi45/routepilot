import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from routepilot_train.runinfo import repo_root

SCRIPT = repo_root() / "bench" / "scripts" / "inupper.py"
FIXTURE_CONFIG = repo_root() / "testdata" / "replay" / "config.yaml"

# The router bounds the prompt tokens of a one-message request by its text bytes + 8 + 16.
FRAMING = 24
SENTINEL = "SENTINEL-INUPPER-91c2"
PROMPTS = {
    "t/1": SENTINEL,
    "t/2": SENTINEL + " " + "x" * 100,
    "t/3": SENTINEL + " " + "y" * 1000,
}
DATASET = {"t/1": "ds-a", "t/2": "ds-a", "t/3": "ds-b"}


def upper(item_id: str) -> int:
    return len(PROMPTS[item_id].encode("utf-8")) + FRAMING


# recorded prompt_tokens per (item, model), built from each item's bound
TOKENS = {
    "t/1": {"cheap-a": upper("t/1"), "mid-b": 2 * upper("t/1"), "top-c": upper("t/1") // 2},
    "t/2": {"cheap-a": upper("t/2") // 2, "mid-b": upper("t/2"), "top-c": 0},
    "t/3": {"cheap-a": 300, "mid-b": upper("t/3") * 3 // 2, "top-c": -3},
}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inupper", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    root = tmp_path / "derived"
    (root / "pools").mkdir(parents=True)
    (root / "replay").mkdir()
    shutil.copy(FIXTURE_CONFIG, root / "pools" / "flagship.config.yaml")
    lines = []
    for item_id, prompt in PROMPTS.items():
        outcomes = {
            model: {
                "success": True,
                "cost_picousd": 10,
                "prompt_tokens": tokens,
                "completion_tokens": 5,
            }
            for model, tokens in TOKENS[item_id].items()
        }
        lines.append(
            json.dumps(
                {
                    "item_id": item_id,
                    "dataset": DATASET[item_id],
                    "split": "test",
                    "prompt_sha256": "ab" * 32,
                    "messages": [{"role": "user", "content": prompt}],
                    "outcomes": outcomes,
                }
            )
        )
    (root / "replay" / "flagship.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_ratios_overall_per_dataset_and_per_model(derived: Path, binary: Path, tmp_path: Path):
    out = load_script().run_inupper(
        derived, tmp_path / "out", ["python", "x.py"], binary, pools=["flagship"]
    )

    assert sorted(p.name for p in out.iterdir()) == ["report.md", "results.json", "run.json"]
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    pool = results["pools"]["flagship"]
    assert pool["items"] == 3 and pool["records"] == 9
    assert pool["records_nonpositive_prompt_tokens"] == 2  # a zero and a negative count

    ratios = [
        1.0,
        2.0,
        TOKENS["t/1"]["top-c"] / upper("t/1"),  # t/1
        TOKENS["t/2"]["cheap-a"] / upper("t/2"),
        1.0,  # t/2
        300 / upper("t/3"),
        TOKENS["t/3"]["mid-b"] / upper("t/3"),  # t/3
    ]
    overall = pool["overall"]
    assert overall["n"] == 7
    assert overall["max"] == pytest.approx(2.0)
    assert overall["p50"] == pytest.approx(sorted(ratios)[3])
    assert overall["p99"] == pytest.approx(pd.Series(ratios).quantile(0.99))
    assert overall["p99_9"] == pytest.approx(pd.Series(ratios).quantile(0.999))
    assert overall["count_gt_1"] == 2  # 2.0 and the t/3 mid-b record; exactly 1.0 does not count
    assert results["max_ratio"] == pytest.approx(2.0) and results["any_above_1"] is True

    assert pool["per_dataset"]["ds-a"]["n"] == 5 and pool["per_dataset"]["ds-a"]["count_gt_1"] == 1
    assert pool["per_dataset"]["ds-b"]["n"] == 2 and pool["per_dataset"]["ds-b"]["count_gt_1"] == 1
    assert pool["per_model"]["mid-b"]["max"] == pytest.approx(2.0)
    assert pool["per_model"]["mid-b"]["count_gt_1"] == 2
    assert pool["per_model"]["cheap-a"]["count_gt_1"] == 0
    assert pool["per_model"]["top-c"]["n"] == 1  # its zero and negative counts are excluded

    worst = pool["worst"]
    assert [(w["item_id"], w["model"]) for w in worst] == [("t/1", "mid-b"), ("t/3", "mid-b")]
    assert worst[0]["in_upper"] == upper("t/1") and worst[0]["prompt_tokens"] == 2 * upper("t/1")


def test_outputs_hold_no_prompt_text_and_report_the_ratio(derived: Path, binary: Path, tmp_path):
    out = load_script().run_inupper(derived, tmp_path / "out", None, binary, pools=["flagship"])
    for path in out.iterdir():
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# inupper\n") and "Per dataset" in md and "Per model" in md
    assert "Records above 1" in md and "ds-a" in md
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["experiment"] == "inupper"


def test_no_record_above_1_is_reported_as_such(derived: Path, binary: Path, tmp_path: Path):
    lines = (derived / "replay" / "flagship.jsonl").read_text(encoding="utf-8").splitlines()
    tight = []
    for line in lines:
        item = json.loads(line)
        bound = len(item["messages"][0]["content"].encode()) + FRAMING
        for outcome in item["outcomes"].values():
            outcome["prompt_tokens"] = bound // 2
        tight.append(json.dumps(item))
    (derived / "replay" / "flagship.jsonl").write_text("\n".join(tight) + "\n", encoding="utf-8")

    out = load_script().run_inupper(derived, tmp_path / "out", None, binary, pools=["flagship"])

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["any_above_1"] is False and results["max_ratio"] <= 0.5
    assert results["pools"]["flagship"]["worst"] == []
    assert "Records above 1" not in (out / "report.md").read_text(encoding="utf-8")
