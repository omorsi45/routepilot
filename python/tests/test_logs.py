import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from routepilot_train import logs


def decision(
    decision_id: str,
    tenant: str,
    time: str,
    *,
    role: str = "stable",
    chosen: str = "m-small",
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "decision",
        "decision_id": decision_id,
        "time": time,
        "tenant": tenant,
        "policy_version": "v1+abcd1234",
        "policy_role": role,
        "prompt_sha256_16": "0123456789abcdef",
        "prompt_bytes": 42,
        "required": [],
        "ood_score": 0.25,
        "ood": False,
        "candidates": [
            {
                "model": "m-small",
                "p": 0.6,
                "l": 0.5,
                "cost_exp_picousd": 100,
                "cost_max_picousd": 400,
                "q_ms": None,
                "feasible": True,
                "reasons": [],
            },
            {
                "model": "m-big",
                "p": 0.9,
                "l": 0.8,
                "cost_exp_picousd": 900,
                "cost_max_picousd": 2000,
                "q_ms": None,
                "feasible": True,
                "reasons": [],
            },
        ],
        "chosen": chosen,
        "propensity": 0.95,
        "epsilon": 0.05,
        "explored": False,
        "relaxed": [],
    }
    if role != "shadow":
        record.update(
            status=200,
            latency_ms=120,
            ttft_ms=None,
            prompt_tokens=5,
            completion_tokens=4,
            cost_actual_picousd=450,
            usage_estimated=False,
            attempts=[{"deployment": "m-small-a", "outcome": "ok"}],
        )
    record.update(extra)
    return record


def feedback(decision_id: str, tenant: str, success: bool, time: str) -> dict[str, Any]:
    return {
        "type": "feedback",
        "decision_id": decision_id,
        "tenant": tenant,
        "success": success,
        "time": time,
    }


def write_log(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_load_decisions_joins_feedback_by_tenant(tmp_path: Path):
    write_log(
        tmp_path / "decisions.jsonl",
        [
            decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00Z"),
            decision("dec_bbbbbbbb", "tenant-b", "2026-10-28T01:00:01.5Z"),
            decision("dec_cccccccc", "tenant-a", "2026-10-28T01:00:02.123456789Z"),
            decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00Z", role="shadow"),
            # tenant-b answers for tenant-a's decision: not tenant-matched, so it is ignored.
            feedback("dec_aaaaaaaa", "tenant-b", False, "2026-10-28T01:01:00Z"),
            feedback("dec_aaaaaaaa", "tenant-a", True, "2026-10-28T01:01:01Z"),
            # Two answers from the right tenant: the later one wins, whatever the line order.
            feedback("dec_bbbbbbbb", "tenant-b", False, "2026-10-28T01:03:00Z"),
            feedback("dec_bbbbbbbb", "tenant-b", True, "2026-10-28T01:02:00Z"),
            # A decision nobody answered, and an answer to a decision that is not in the log.
            feedback("dec_zzzzzzzz", "tenant-a", True, "2026-10-28T01:01:00Z"),
        ],
    )

    df = logs.load_decisions(str(tmp_path / "decisions.jsonl"))

    assert list(df["decision_id"]) == ["dec_aaaaaaaa", "dec_bbbbbbbb", "dec_cccccccc"]
    assert df["decision_id"].is_unique
    by_id = df.set_index("decision_id")
    assert bool(by_id.loc["dec_aaaaaaaa", "feedback_success"])
    assert not bool(by_id.loc["dec_bbbbbbbb", "feedback_success"])
    assert pd.isna(by_id.loc["dec_cccccccc", "feedback_success"])
    assert str(df["feedback_success"].dtype) == "boolean"
    assert by_id.loc["dec_aaaaaaaa", "feedback_time"] == pd.Timestamp("2026-10-28T01:01:01Z")
    assert pd.isna(by_id.loc["dec_cccccccc", "feedback_time"])


def test_load_decisions_columns_and_types(tmp_path: Path):
    write_log(
        tmp_path / "decisions.jsonl",
        [decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00.5Z")],
    )

    df = logs.load_decisions(tmp_path)

    assert list(df.columns) == logs.COLUMNS
    row = df.iloc[0]
    assert df["time"].dt.tz is not None
    assert row["time"] == pd.Timestamp("2026-10-28T01:00:00.5Z")
    assert row["policy_role"] == "stable"
    assert row["propensity"] == pytest.approx(0.95)
    assert row["chosen"] == "m-small"
    assert row["status"] == 200
    assert [c["model"] for c in row["candidates"]] == ["m-small", "m-big"]
    assert row["attempts"] == [{"deployment": "m-small-a", "outcome": "ok"}]
    assert pd.isna(row["ttft_ms"])


def test_load_decisions_reads_rotated_files_and_orders_by_time(tmp_path: Path):
    write_log(
        tmp_path / "decisions.20261028T010000Z.jsonl",
        [
            decision("dec_11111111", "tenant-a", "2026-10-28T00:59:00Z"),
            decision("dec_22222222", "tenant-a", "2026-10-28T00:59:30Z"),
        ],
    )
    write_log(
        tmp_path / "decisions.jsonl",
        [
            decision("dec_33333333", "tenant-a", "2026-10-28T01:00:30Z"),
            # Feedback in the active file answers a decision in a rotated one.
            feedback("dec_11111111", "tenant-a", True, "2026-10-28T01:00:40Z"),
        ],
    )
    write_log(
        tmp_path / "other.jsonl", [decision("dec_99999999", "tenant-a", "2026-10-28T00:00:00Z")]
    )

    for source in (str(tmp_path / "decisions*.jsonl"), tmp_path / "decisions*.jsonl"):
        df = logs.load_decisions(source)
        assert list(df["decision_id"]) == ["dec_11111111", "dec_22222222", "dec_33333333"]
        assert bool(df.set_index("decision_id").loc["dec_11111111", "feedback_success"])

    everything = logs.load_decisions(tmp_path)  # a directory reads every *.jsonl in it
    assert list(everything["decision_id"]) == [
        "dec_99999999",
        "dec_11111111",
        "dec_22222222",
        "dec_33333333",
    ]


def test_load_decisions_shadow_rows_are_opt_in_and_get_no_feedback(tmp_path: Path):
    write_log(
        tmp_path / "decisions.jsonl",
        [
            decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00Z"),
            decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00Z", role="shadow"),
            feedback("dec_aaaaaaaa", "tenant-a", True, "2026-10-28T01:01:00Z"),
        ],
    )

    default = logs.load_decisions(tmp_path)
    with_shadow = logs.load_decisions(tmp_path, include_shadow=True)

    assert list(default["policy_role"]) == ["stable"]
    assert list(with_shadow["policy_role"]) == ["stable", "shadow"]
    shadow = with_shadow[with_shadow["policy_role"] == "shadow"].iloc[0]
    assert pd.isna(shadow["feedback_success"])
    assert pd.isna(shadow["status"])  # a shadow decision has no outcome


def test_load_decisions_skips_torn_lines_and_unknown_types(tmp_path: Path):
    good = json.dumps(decision("dec_aaaaaaaa", "tenant-a", "2026-10-28T01:00:00Z"))
    (tmp_path / "decisions.jsonl").write_text(
        "\n".join(
            [
                good,
                '{"type":"decision","decision_id":"dec_torn',  # a crash cut this line
                "",
                json.dumps({"type": "something_new", "decision_id": "dec_aaaaaaaa"}),
                json.dumps(decision("dec_bbbbbbbb", "tenant-a", "2026-10-28T01:00:05Z")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="skipped 1 malformed line"):
        df = logs.load_decisions(tmp_path)

    assert list(df["decision_id"]) == ["dec_aaaaaaaa", "dec_bbbbbbbb"]


def test_load_decisions_without_files_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no decision log"):
        logs.load_decisions(str(tmp_path / "nope*.jsonl"))
    with pytest.raises(FileNotFoundError, match="no decision log"):
        logs.load_decisions(tmp_path)


def test_load_decisions_empty_log_has_the_columns(tmp_path: Path):
    (tmp_path / "decisions.jsonl").write_text("", encoding="utf-8")

    df = logs.load_decisions(tmp_path)

    assert df.empty
    assert list(df.columns) == logs.COLUMNS
