import sys

import pandas as pd
import pytest

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root

FIXTURE = repo_root() / "testdata" / "replay"


def test_build_go_writes_bin_routepilot(binary):
    expected = "routepilot.exe" if sys.platform == "win32" else "routepilot"
    assert binary == repo_root() / "bin" / expected
    assert binary.is_file()


def test_build_go_without_a_toolchain_is_a_toolchain_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(replaytool.GoToolchainError, match="not on PATH"):
        replaytool.build_go(repo_root())


def test_replay_returns_one_row_per_item_with_the_output_columns(binary, tmp_path):
    df = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "all",
        "cheapest",
        tmp_path / "out.jsonl",
    )
    assert list(df["item_id"]) == ["fx-a/1", "fx-a/2", "fx-b/3", "fx-b/4", "fx-b/5"]
    assert set(df["model"]) == {"cheap-a"} and set(df["policy"]) == {"cheapest"}
    assert df["success"].dtype == bool
    assert df["cost_picousd"].dtype == "int64"
    assert list(df["cost_picousd"]) == [300, 150, 100, 250, 120]
    assert df["p"].isna().all() and df["slo_violated"].isna().all()
    assert (df["error"] == "").all()
    assert "messages" not in df.columns


def test_replay_frame_carries_the_chosen_models_prompt_overhead(binary, tmp_path):
    df = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "all",
        "cheapest",
        tmp_path / "out.jsonl",
    )
    assert (df["prompt_overhead"] == 0).all()  # the fixture models declare no overhead
    ghost = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "all",
        "fixed",
        tmp_path / "ghost.jsonl",
        ["--model", "ghost"],
    )
    assert ghost["prompt_overhead"].isna().all()  # no chosen model, no value


def test_replay_passes_extra_flags_and_reports_routing_errors(binary, tmp_path):
    fixed = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "test",
        "fixed",
        tmp_path / "fixed.jsonl",
        ["--model", "top-c"],
    )
    assert list(fixed["model"]) == ["top-c", "top-c"]
    ghost = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "all",
        "fixed",
        tmp_path / "ghost.jsonl",
        ["--model", "ghost"],
    )
    assert (ghost["error"] == "model_not_found").all()
    assert ghost["model"].isna().all() and not ghost["success"].any()
    assert (ghost["cost_picousd"] == 0).all()


def test_replay_failure_raises_with_the_cli_message(binary, tmp_path):
    with pytest.raises(RuntimeError, match="no items in split"):
        replaytool.replay(
            binary,
            FIXTURE / "config.yaml",
            FIXTURE / "items.jsonl",
            "heldout",
            "cheapest",
            tmp_path / "out.jsonl",
        )


def test_replay_frame_from_empty_file_has_the_columns(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    df = replaytool.read_replay(path)
    assert isinstance(df, pd.DataFrame) and df.empty
    assert {"item_id", "model", "success", "cost_picousd", "error"} <= set(df.columns)
