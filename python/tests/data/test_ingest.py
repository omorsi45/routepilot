import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fixtures import (
    SyntheticFile,
    build_tarball,
    prompt_for,
    record,
    result_file,
    simple_records,
)

from routepilot_train.data import download, ingest

OUTCOME_COLUMNS = [
    "dataset",
    "item_id",
    "model",
    "score",
    "success",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "prompt_sha256",
]


def run_ingest(
    tmp_path: Path,
    files: list[SyntheticFile],
    extra_members: dict[str, bytes] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    tar = build_tarball(tmp_path / "bench-release.tar.gz", files, extra_members)
    out = tmp_path / "derived"
    report = ingest.ingest(tar, out)
    assert json.loads((out / "ingest_report.json").read_text(encoding="utf-8")) == report
    for splits in report["kept"].values():
        for group in splits.values():
            for stats in group["models"].values():
                dropped = sum(stats["dropped"].values())
                assert stats["records_kept"] + dropped == stats["records_read"]
    return (
        report,
        pd.read_parquet(out / "outcomes.parquet"),
        pd.read_parquet(out / "prompts.parquet"),
    )


def model_stats(report: dict[str, Any], dataset: str, split: str, model: str) -> dict[str, Any]:
    return report["kept"][dataset][split]["models"][model]


def test_reads_without_extracting(tmp_path, monkeypatch):
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ingest must not extract to disk")

    monkeypatch.setattr(tarfile.TarFile, "extract", refuse)
    monkeypatch.setattr(tarfile.TarFile, "extractall", refuse)
    files = [
        result_file("gpqa", "test", model, simple_records("gpqa", range(3)))
        for model in ("model-a", "model-b")
    ]
    _, outcomes, _ = run_ingest(tmp_path, files)
    assert len(outcomes) == 6
    assert sorted(p.name for p in (tmp_path / "derived").iterdir()) == [
        "ingest_report.json",
        "outcomes.parquet",
        "prompts.parquet",
    ]
    assert not (tmp_path / "bench-release").exists()


def test_string_values_are_parsed(tmp_path):
    records = [
        record(0, prompt_for("gpqa", 0), score="0.0", prompt_tokens="123", completion_tokens="7"),
        record(1, prompt_for("gpqa", 1), score="1", prompt_tokens="4.0", cost="4.5e-05"),
    ]
    _, outcomes, _ = run_ingest(tmp_path, [result_file("gpqa", "test", "model-a", records)])
    assert list(outcomes.columns) == OUTCOME_COLUMNS
    assert outcomes["score"].dtype == "float64"
    assert outcomes["success"].dtype == "bool"
    assert outcomes["prompt_tokens"].dtype == "int64"
    assert outcomes["completion_tokens"].dtype == "int64"
    assert outcomes["cost_usd"].dtype == "float64"
    for column in ("dataset", "item_id", "model", "prompt_sha256"):
        assert pd.api.types.is_string_dtype(outcomes[column])
    first, second = outcomes.iloc[0], outcomes.iloc[1]
    assert (first["score"], first["prompt_tokens"], first["completion_tokens"]) == (0.0, 123, 7)
    assert (second["score"], second["prompt_tokens"]) == (1.0, 4)
    assert second["cost_usd"] == pytest.approx(4.5e-05)
    assert first["item_id"] == "gpqa/0"
    assert first["prompt_sha256"] == hashlib.sha256(prompt_for("gpqa", 0).encode()).hexdigest()


def test_success_threshold(tmp_path):
    scores = ["0.5", "0.49", "0", "1", "0.75"]
    records = [record(i, prompt_for("gpqa", i), score=s) for i, s in enumerate(scores)]
    report, outcomes, _ = run_ingest(tmp_path, [result_file("gpqa", "test", "model-a", records)])
    assert outcomes["success"].tolist() == [True, False, False, True, True]
    assert report["score_histogram"]["gpqa"] == {"0": 1, "1": 1, "fractional": 3, "unparseable": 0}


@pytest.mark.parametrize("newer_first", [True, False])
def test_newest_timestamp_wins(tmp_path, newer_first):
    old = result_file(
        "gpqa",
        "test",
        "model-a",
        simple_records("gpqa", range(3), score="0"),
        timestamp="20251029_012339",
    )
    new = result_file(
        "gpqa",
        "test",
        "model-a",
        simple_records("gpqa", range(2), score="1"),
        timestamp="20251030_000000",
    )
    report, outcomes, _ = run_ingest(tmp_path, [new, old] if newer_first else [old, new])
    assert outcomes["success"].tolist() == [True, True]
    stats = model_stats(report, "gpqa", "test", "model-a")
    assert stats["files_seen"] == 2
    assert stats["file_chosen"] == {"member": new.member, "timestamp": "20251030_000000"}
    assert stats["alternatives"] == [{"member": old.member, "timestamp": "20251029_012339"}]
    assert stats["records_read"] == 2


def test_excluded_and_unknown_datasets_reported(tmp_path):
    files = [
        result_file("gpqa", "test", "model-a", simple_records("gpqa", range(2))),
        result_file("gpqa", "test", "openrouter", simple_records("gpqa", range(3))),
        result_file("swe-bench", "verified", "model-a", simple_records("swe-bench", range(2))),
        result_file("mystery", "test", "model-a", simple_records("mystery", range(2))),
        result_file(
            "gpqa",
            "extra_split",
            "model-a",
            simple_records("gpqa", range(2)),
            timestamp="20251101_000000",
        ),
    ]
    readme = {"bench-release/README.md": b"not a result file"}
    report, outcomes, prompts = run_ingest(tmp_path, files, readme)
    assert set(outcomes["dataset"]) == {"gpqa"}
    assert set(outcomes["model"]) == {"model-a"}
    assert set(prompts["dataset"]) == {"gpqa"}
    excluded = {(e["dataset"], e["split"]): e for e in report["excluded"]}
    assert set(excluded) == {
        ("swe-bench", "verified"),
        ("mystery", "test"),
        ("gpqa", "extra_split"),
    }
    assert "agentic" in excluded[("swe-bench", "verified")]["reason"]
    assert excluded[("mystery", "test")]["reason"] == "dataset is not in the selection"
    assert "not the selected split" in excluded[("gpqa", "extra_split")]["reason"]
    assert report["excluded_models"] == {"openrouter": {"files": 1, "records": 3}}
    assert [s["member"] for s in report["skipped_members"]] == ["bench-release/README.md"]
    assert "gpqa:test" not in report["missing_selected"]
    assert "aime:hybrid" in report["missing_selected"]


def test_prompt_mismatch_dropped(tmp_path):
    agree = simple_records("gpqa", range(3))
    odd = simple_records("gpqa", range(3))
    odd[1] = record(1, "a different prompt at index 1")
    files = [
        result_file("gpqa", "test", "model-a", agree),
        result_file("gpqa", "test", "model-b", agree),
        result_file("gpqa", "test", "model-c", odd),
    ]
    report, outcomes, _ = run_ingest(tmp_path, files)
    assert model_stats(report, "gpqa", "test", "model-c")["dropped"] == {"prompt_mismatch": 1}
    assert model_stats(report, "gpqa", "test", "model-c")["records_kept"] == 2
    assert len(outcomes[(outcomes["model"] == "model-c")]) == 2
    assert len(outcomes[outcomes["item_id"] == "gpqa/1"]) == 2


def test_prompt_vote_tie_goes_to_first_sorted_model(tmp_path):
    files = [
        result_file("gpqa", "test", "model-b", [record(0, "prompt from b")]),
        result_file("gpqa", "test", "model-a", [record(0, "prompt from a")]),
    ]
    report, outcomes, prompts = run_ingest(tmp_path, files)
    assert outcomes["model"].tolist() == ["model-a"]
    assert prompts["prompt"].tolist() == ["prompt from a"]
    assert model_stats(report, "gpqa", "test", "model-b")["dropped"] == {"prompt_mismatch": 1}


def test_mixed_extra_rekeyed_by_prompt(tmp_path):
    canonical = [
        result_file("mmlupro", "test_3000", m, simple_records("mmlupro", range(5), score="1"))
        for m in ("model-a", "model-b")
    ]
    # The 7B model numbers its items differently: its indexes 0..2 are canonical 4, 2, 0.
    small = [
        record(0, prompt_for("mmlupro", 4), score="0.75"),
        record(1, prompt_for("mmlupro", 2), score="0"),
        record(2, prompt_for("mmlupro", 0), score="1"),
        record(3, "a prompt that is not in test_3000", score="1"),
    ]
    extra = [
        result_file("mmlupro", "test_1000", "model-c", small),
        result_file(
            "mmlupro",
            "test_1000",
            "model-a",
            simple_records("mmlupro", range(2)),
            timestamp="20251101_000000",
        ),
    ]
    report, outcomes, prompts = run_ingest(tmp_path, canonical + extra)
    rows = outcomes[outcomes["model"] == "model-c"].set_index("item_id")
    assert sorted(rows.index) == ["mmlupro/0", "mmlupro/2", "mmlupro/4"]
    assert rows.loc["mmlupro/4", "score"] == 0.75
    assert not rows.loc["mmlupro/2", "success"]
    assert set(outcomes["dataset"]) == {"mmlupro"}
    assert len(prompts) == 5
    split = report["kept"]["mmlupro"]["test_1000"]
    assert split["role"] == "mixed_extra"
    assert split["models"]["model-c"]["dropped"] == {"no_canonical_item": 1}
    assert split["models"]["model-a"]["dropped"] == {"model_in_canonical_split": 2}
    assert len(outcomes[outcomes["model"] == "model-a"]) == 5


def test_duplicate_prompt_kept_once(tmp_path):
    shared = "the same prompt in two datasets"
    files = []
    for model in ("model-a", "model-b"):
        gpqa = simple_records("gpqa", [1, 2]) + [record(0, shared)]
        hle = simple_records("hle", [1]) + [record(5, shared)]
        files.append(result_file("gpqa", "test", model, gpqa))
        files.append(result_file("hle", "test", model, hle))
    report, outcomes, prompts = run_ingest(tmp_path, files)
    assert "hle/5" not in set(outcomes["item_id"])
    assert "gpqa/0" in set(outcomes["item_id"])
    assert (prompts["prompt"] == shared).sum() == 1
    sha16 = hashlib.sha256(shared.encode()).hexdigest()[:16]
    assert report["duplicate_prompts"]["global"] == [
        {"prompt_sha256_16": sha16, "kept_item_id": "gpqa/0", "dropped_item_ids": ["hle/5"]}
    ]
    assert model_stats(report, "hle", "test", "model-a")["dropped"] == {
        "duplicate_prompt_global": 1
    }


def test_duplicate_prompt_in_split_keeps_first_index(tmp_path):
    shared = "repeated inside one split"
    records = simple_records("gpqa", [0, 1, 3]) + [record(2, shared), record(10, shared)]
    report, outcomes, _ = run_ingest(tmp_path, [result_file("gpqa", "test", "model-a", records)])
    assert set(outcomes["item_id"]) == {"gpqa/0", "gpqa/1", "gpqa/2", "gpqa/3"}
    sha16 = hashlib.sha256(shared.encode()).hexdigest()[:16]
    assert report["duplicate_prompts"]["within_split"] == [
        {
            "dataset": "gpqa",
            "split": "test",
            "prompt_sha256_16": sha16,
            "kept_item_id": "gpqa/2",
            "dropped_item_ids": ["gpqa/10"],
        }
    ]
    assert model_stats(report, "gpqa", "test", "model-a")["dropped"] == {
        "duplicate_prompt_in_split": 1
    }


def test_unparseable_score_dropped(tmp_path):
    scores = ["n/a", None, "nan", "inf", "0.9"]
    records = [record(i, prompt_for("gpqa", i), score=s) for i, s in enumerate(scores)]
    report, outcomes, _ = run_ingest(tmp_path, [result_file("gpqa", "test", "model-a", records)])
    assert outcomes["item_id"].tolist() == ["gpqa/4"]
    assert model_stats(report, "gpqa", "test", "model-a")["dropped"] == {"unparseable_score": 4}
    assert report["score_histogram"]["gpqa"] == {"0": 0, "1": 0, "fractional": 1, "unparseable": 4}


def test_malformed_records_dropped_with_reasons(tmp_path):
    lone_surrogate = "\ud800 cannot be encoded as UTF-8"
    records = [
        record(0, prompt_for("gpqa", 0)),
        record("x", prompt_for("gpqa", 1)),
        record(2, None),
        record(3, lone_surrogate),
        record(4, prompt_for("gpqa", 4), prompt_tokens="many"),
        record(5, prompt_for("gpqa", 5), cost=None),
        record(0, prompt_for("gpqa", 0)),
        "not an object",
    ]
    report, outcomes, _ = run_ingest(tmp_path, [result_file("gpqa", "test", "model-a", records)])
    assert outcomes["item_id"].tolist() == ["gpqa/0"]
    assert model_stats(report, "gpqa", "test", "model-a")["dropped"] == {
        "duplicate_record": 1,
        "invalid_prompt": 2,
        "invalid_record": 1,
        "unparseable_index": 1,
        "unparseable_usage": 2,
    }


def test_rows_sorted_by_numeric_index(tmp_path):
    indexes = [10, 2, 0, 11, 1]
    files = [
        result_file("gpqa", "test", m, simple_records("gpqa", indexes))
        for m in ("model-b", "model-a")
    ]
    _, outcomes, prompts = run_ingest(tmp_path, files)
    expected = [f"gpqa/{i}" for i in (0, 1, 2, 10, 11) for _ in range(2)]
    assert outcomes["item_id"].tolist() == expected
    assert outcomes["model"].tolist()[:2] == ["model-a", "model-b"]
    assert prompts["item_id"].tolist() == ["gpqa/0", "gpqa/1", "gpqa/2", "gpqa/10", "gpqa/11"]


def test_prompts_table(tmp_path):
    text = "h\u00e9llo \u65e5\u672c\u8a9e"
    files = [result_file("gpqa", "test", "model-a", [record(0, text)])]
    _, _, prompts = run_ingest(tmp_path, files)
    assert list(prompts.columns) == [
        "dataset",
        "item_id",
        "prompt",
        "prompt_sha256",
        "prompt_bytes",
    ]
    row = prompts.iloc[0]
    assert row["prompt"] == text
    assert row["prompt_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert row["prompt_bytes"] == len(text.encode("utf-8"))
    assert prompts["prompt_bytes"].dtype == "int64"


def test_report_has_no_prompt_text(tmp_path):
    sentinel = "SENTINEL-PROMPT-TEXT-0451"
    files = [
        result_file("gpqa", "test", "model-a", [record(0, sentinel), record(1, sentinel + "x")]),
        result_file("gpqa", "test", "model-b", [record(0, sentinel + "y")]),
    ]
    run_ingest(tmp_path, files)
    assert sentinel not in (tmp_path / "derived" / "ingest_report.json").read_text(encoding="utf-8")


def test_split_directories_are_read(tmp_path):
    # Datasets with several splits nest a split directory under the dataset.
    files = [
        result_file("hle", "test", m, simple_records("hle", range(3)), split_dir=True)
        for m in ("model-a", "model-b")
    ]
    files.append(
        result_file("hle", "subset_500", "model-a", simple_records("hle", range(2)), split_dir=True)
    )
    report, outcomes, _ = run_ingest(tmp_path, files)
    assert len(outcomes) == 6
    assert report["json_files"] == 3
    assert report["skipped_members"] == []
    assert [e["split"] for e in report["excluded"]] == ["subset_500"]
    chosen = model_stats(report, "hle", "test", "model-a")["file_chosen"]["member"]
    assert chosen == "bench-release/hle/test/model-a/hle-test-model-a-20251029_012339.json"


def test_invalid_json_member_is_an_error(tmp_path):
    bad = {"bench-release/gpqa/model-a/gpqa-model-a-20251029_012339.json": b"{not json"}
    tar = build_tarball(tmp_path / "bench-release.tar.gz", [], bad)
    with pytest.raises(ValueError, match="gpqa-model-a-20251029_012339.json"):
        ingest.ingest(tar, tmp_path / "derived")


def test_verify_rejects_wrong_sha(tmp_path):
    path = tmp_path / "bench-release.tar.gz"
    payload = b"x" * 64
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="sha256"):
        download.verify(path, expected_bytes=len(payload))
    right = hashlib.sha256(payload).hexdigest()
    assert download.verify(path, expected_bytes=len(payload), expected_sha256=right) == right
