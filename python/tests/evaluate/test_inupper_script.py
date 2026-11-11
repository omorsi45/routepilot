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


def write_derived(
    root: Path,
    prompts: dict[str, str],
    dataset_of: dict[str, str],
    split_of: dict[str, str],
    tokens_of: dict[str, dict[str, int]],
    overheads: dict[str, int] | None,
) -> Path:
    """A derived directory: the replay export, the fixture config and a pool JSON."""
    (root / "pools").mkdir(parents=True)
    (root / "replay").mkdir()
    shutil.copy(FIXTURE_CONFIG, root / "pools" / "flagship.config.yaml")
    pool: dict = {"pool": "flagship", "models": sorted(next(iter(tokens_of.values())))}
    if overheads is not None:
        pool["prompt_overhead_tokens"] = overheads
        pool["anomalies"] = []
    (root / "pools" / "flagship.json").write_text(json.dumps(pool), encoding="utf-8")
    lines = []
    for item_id, prompt in prompts.items():
        outcomes = {
            model: {
                "success": True,
                "cost_picousd": 10,
                "prompt_tokens": tokens,
                "completion_tokens": 5,
            }
            for model, tokens in tokens_of[item_id].items()
        }
        lines.append(
            json.dumps(
                {
                    "item_id": item_id,
                    "dataset": dataset_of[item_id],
                    "split": split_of[item_id],
                    "prompt_sha256": "ab" * 32,
                    "messages": [{"role": "user", "content": prompt}],
                    "outcomes": outcomes,
                }
            )
        )
    (root / "replay" / "flagship.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    zero = {"cheap-a": 0, "mid-b": 0, "top-c": 0}
    return write_derived(
        tmp_path / "derived", PROMPTS, DATASET, dict.fromkeys(PROMPTS, "test"), TOKENS, zero
    )


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


# --- the bound a model is held to is in_upper + its prompt overhead ---------

MODELS = ("cheap-a", "mid-b", "top-c")
OVERHEADS = {"cheap-a": 0, "mid-b": 100, "top-c": 0}
ADJ_PROMPTS = {
    item_id: SENTINEL + " " + "z" * length
    for item_id, length in {"p/1": 400, "p/2": 410, "p/3": 420, "p/4": 430, "p/5": 440}.items()
}
ADJ_SPLIT = {"p/1": "train", "p/2": "calib", "p/3": "test", "p/4": "heldout", "p/5": "train"}
ADJ_DATASET = {"p/1": "ds-a", "p/2": "ds-a", "p/3": "ds-b", "p/4": "ds-c", "p/5": "ds-a"}
SPLITS = ("train", "calib", "test", "heldout")


def adj_upper(item_id: str) -> int:
    return len(ADJ_PROMPTS[item_id].encode("utf-8")) + FRAMING


def _tokens(item_id: str, cheap: int | None = None, mid: int = 0, top: int = 0) -> dict[str, int]:
    half = adj_upper(item_id) // 2
    return {"cheap-a": half if cheap is None else cheap, "mid-b": mid or half, "top-c": top or half}


ADJ_TOKENS = {
    # train: mid-b sits exactly on its adjusted bound
    "p/1": _tokens("p/1", mid=adj_upper("p/1") + 100),
    # calib: mid-b is below its adjusted bound
    "p/2": _tokens("p/2", mid=adj_upper("p/2") + 60),
    # test: top-c has no overhead and goes 5 tokens over the request bound
    "p/3": _tokens("p/3", mid=adj_upper("p/3") + 100, top=adj_upper("p/3") + 5),
    # heldout: mid-b is one token over its adjusted bound, top-c is an outlier
    "p/4": _tokens("p/4", mid=adj_upper("p/4") + 101, top=5000),
    # train: an outlier that the fit leaves out
    "p/5": _tokens("p/5", top=5000),
}
OUTLIERS = {("p/4", "top-c"), ("p/5", "top-c")}


@pytest.fixture
def adjusted(tmp_path: Path) -> Path:
    return write_derived(
        tmp_path / "derived", ADJ_PROMPTS, ADJ_DATASET, ADJ_SPLIT, ADJ_TOKENS, OVERHEADS
    )


def run_adjusted(derived: Path, binary: Path, tmp_path: Path, module: ModuleType | None = None):
    module = module or load_script()
    out = module.run_inupper(derived, tmp_path / "out", None, binary, pools=["flagship"])
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    return out, results["pools"]["flagship"]


def brute_force(split: str, overheads: dict[str, int], skip: set = frozenset()):
    """(records, max ratio, count above 1) over the positive records of one split."""
    ratios = [
        tokens / (adj_upper(item_id) + overheads[model])
        for item_id, by_model in ADJ_TOKENS.items()
        if ADJ_SPLIT[item_id] == split
        for model, tokens in by_model.items()
        if tokens > 0 and (item_id, model) not in skip
    ]
    return len(ratios), max(ratios), sum(r > 1.0 for r in ratios)


def test_adjusted_ratio_per_split_uses_each_models_own_overhead(
    adjusted: Path, binary: Path, tmp_path: Path
):
    _, pool = run_adjusted(adjusted, binary, tmp_path)

    section = pool["adjusted"]
    assert section["prompt_overhead_tokens"] == OVERHEADS
    assert list(section["per_split"]) == list(SPLITS)
    for split in SPLITS:
        n, top, over = brute_force(split, OVERHEADS)
        got = section["per_split"][split]
        assert got["n"] == n and got["count_gt_1"] == over, split
        assert got["max"] == pytest.approx(top), split
        assert got["coverage"] == pytest.approx((n - over) / n), split
    # the pinned shapes behind those numbers
    assert section["per_split"]["train"]["count_gt_1"] == 1  # only the outlier
    assert section["per_split"]["calib"]["count_gt_1"] == 0
    assert section["per_split"]["test"]["count_gt_1"] == 1  # top-c, +5 over, no overhead
    assert section["per_split"]["heldout"]["count_gt_1"] == 2

    # the raw table is still the request-level bound: mid-b is above 1 without its overhead
    raw = pool["per_model"]["mid-b"]
    assert raw["count_gt_1"] == 4  # p/1 to p/4 exceed in_upper; p/5 does not
    assert pool["overall"]["max"] == pytest.approx(5000 / adj_upper("p/4"))


def test_adjusted_records_above_1_are_listed_with_their_split(
    adjusted: Path, binary: Path, tmp_path: Path
):
    _, pool = run_adjusted(adjusted, binary, tmp_path)

    worst = pool["adjusted"]["worst"]
    # p/4 is the shorter prompt, so the same 5000 tokens are a larger ratio there than on p/5
    assert [(w["item_id"], w["model"], w["split"]) for w in worst] == [
        ("p/4", "top-c", "heldout"),
        ("p/5", "top-c", "train"),
        ("p/3", "top-c", "test"),
        ("p/4", "mid-b", "heldout"),
    ]
    ratios = [w["ratio"] for w in worst]
    assert ratios == sorted(ratios, reverse=True)
    last = worst[-1]
    assert last["in_upper"] == adj_upper("p/4")  # the request-level bound
    assert last["prompt_overhead"] == OVERHEADS["mid-b"]
    assert last["bound"] == adj_upper("p/4") + OVERHEADS["mid-b"]  # the model's own bound
    assert last["prompt_tokens"] == adj_upper("p/4") + 101


def test_anomalies_are_listed_per_split_and_can_be_left_out(
    adjusted: Path, binary: Path, tmp_path: Path
):
    _, pool = run_adjusted(adjusted, binary, tmp_path)

    per_split = pool["adjusted"]["per_split"]
    assert [per_split[s]["n_anomalies"] for s in SPLITS] == [1, 0, 0, 1]
    assert per_split["train"]["anomalies_by_model"] == {"top-c": 1}
    assert per_split["calib"]["anomalies"] == []
    heldout = per_split["heldout"]
    (anomaly,) = heldout["anomalies"]
    others = [ADJ_TOKENS["p/4"]["cheap-a"], ADJ_TOKENS["p/4"]["mid-b"]]
    assert anomaly == {
        "item_id": "p/4",
        "dataset": "ds-c",
        "model": "top-c",
        "prompt_tokens": 5000,
        "others_median": sum(others) / 2,
    }

    for split in SPLITS:
        n, top, over = brute_force(split, OVERHEADS, skip=OUTLIERS)
        got = per_split[split]["without_anomalies"]
        assert got["n"] == n and got["count_gt_1"] == over, split
        assert got["max"] == pytest.approx(top), split
    assert per_split["heldout"]["without_anomalies"]["count_gt_1"] == 1  # mid-b, one token over
    assert per_split["train"]["without_anomalies"]["coverage"] == 1.0


def test_sensitivity_refits_the_overhead_without_excluding_any_record(
    adjusted: Path, binary: Path, tmp_path: Path
):
    _, pool = run_adjusted(adjusted, binary, tmp_path)

    sensitivity = pool["sensitivity_no_exclusion"]
    refit = {
        "cheap-a": 0,
        "mid-b": 100,
        "top-c": 5000 - adj_upper("p/5"),  # the train outlier now sets it
    }
    assert sensitivity["prompt_overhead_tokens"] == refit
    for split in SPLITS:
        n, top, over = brute_force(split, refit)
        got = sensitivity["per_split"][split]
        assert got["n"] == n and got["count_gt_1"] == over, split
        assert got["max"] == pytest.approx(top), split
    # the test row that was 5 over is covered by the larger top-c overhead
    assert sensitivity["per_split"]["test"]["count_gt_1"] == 0


def test_fitted_overheads_are_read_from_the_pool_json_not_refit(
    adjusted: Path, binary: Path, tmp_path: Path
):
    path = adjusted / "pools" / "flagship.json"
    pool_json = json.loads(path.read_text(encoding="utf-8"))
    pool_json["prompt_overhead_tokens"] = {"cheap-a": 0, "mid-b": 0, "top-c": 0}
    path.write_text(json.dumps(pool_json), encoding="utf-8")

    _, pool = run_adjusted(adjusted, binary, tmp_path)

    zero = dict.fromkeys(MODELS, 0)
    assert pool["adjusted"]["prompt_overhead_tokens"] == zero
    _, _, over = brute_force("train", zero)
    assert pool["adjusted"]["per_split"]["train"]["count_gt_1"] == over


def test_a_pool_json_without_overheads_or_a_model_is_refused(tmp_path: Path, binary: Path):
    stale = write_derived(tmp_path / "stale", ADJ_PROMPTS, ADJ_DATASET, ADJ_SPLIT, ADJ_TOKENS, None)
    with pytest.raises(ValueError, match="prompt_overhead_tokens"):
        load_script().run_inupper(stale, tmp_path / "o1", None, binary, pools=["flagship"])

    partial = write_derived(
        tmp_path / "partial", ADJ_PROMPTS, ADJ_DATASET, ADJ_SPLIT, ADJ_TOKENS, {"mid-b": 100}
    )
    with pytest.raises(ValueError, match="top-c"):
        load_script().run_inupper(partial, tmp_path / "o2", None, binary, pools=["flagship"])


def test_listed_anomalies_are_capped_but_still_counted(
    adjusted: Path, binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_script()
    monkeypatch.setattr(module, "MAX_LISTED_ANOMALIES", 0)

    _, pool = run_adjusted(adjusted, binary, tmp_path, module)

    train = pool["adjusted"]["per_split"]["train"]
    assert train["n_anomalies"] == 1 and train["anomalies"] == []
    assert train["anomalies_by_model"] == {"top-c": 1}


def test_adjusted_report_names_the_tables_and_holds_no_prompt_text(
    adjusted: Path, binary: Path, tmp_path: Path
):
    out, _ = run_adjusted(adjusted, binary, tmp_path)

    for path in out.iterdir():
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    md = (out / "report.md").read_text(encoding="utf-8")
    for heading in (
        "Prompt overhead per model",
        "Adjusted ratio per split",
        "Sensitivity: overhead fitted without excluding any record",
        "Records above 1 after the overhead (worst first)",
        "Anomalous records",
    ):
        assert f"## {heading}" in md, heading
    assert "\u2014" not in md and "\r" not in md
