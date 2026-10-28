import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from routepilot_train.evaluate import report

RUN_ID = "20261028T010203Z-abc1234"
RUN_JSON_KEYS = [
    "run_id",
    "experiment",
    "git_sha",
    "git_dirty",
    "policy_version",
    "dataset_sha256",
    "seeds",
    "machine",
    "go_version",
    "python_version",
    "started_at",
    "ended_at",
    "command",
]


def figure() -> Figure:
    fig = Figure(figsize=(3, 2))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def write(tmp_path: Path, **overrides) -> Path:
    args = {
        "out_dir": tmp_path / RUN_ID,
        "experiment": "demo",
        "results": {"pool": "flagship", "count": np.int64(3), "rate": np.float64(0.25)},
        "tables": {"main": pd.DataFrame({"policy": ["a", "b"], "cost": [0.000185, 12.5]})},
        "figures": {"pareto": figure()},
        "seeds": {"bootstrap": 0},
        "command": ["python", "bench/scripts/demo.py"],
        "policy_version": None,
        "started_at": datetime(2026, 10, 28, 1, 2, 3, tzinfo=UTC),
    }
    args.update(overrides)
    return report.write_results(**args)


def test_write_results_layout(tmp_path):
    out = write(tmp_path)

    assert out == tmp_path / RUN_ID
    assert sorted(p.name for p in out.iterdir()) == [
        "pareto.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    assert json.loads((out / "results.json").read_text(encoding="utf-8")) == {
        "pool": "flagship",
        "count": 3,
        "rate": 0.25,
    }
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert list(run) == RUN_JSON_KEYS
    assert run["run_id"] == RUN_ID and run["experiment"] == "demo"
    assert run["seeds"] == {"bootstrap": 0} and run["policy_version"] is None
    assert run["command"] == ["python", "bench/scripts/demo.py"]
    assert run["started_at"] == "2026-10-28T01:02:03Z"
    assert (out / "pareto.png").read_bytes().startswith(b"\x89PNG")

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# demo\n")
    assert RUN_ID in md
    assert "Labels" in md and "simulated" in md and "fitted" in md
    assert "| policy | cost |" in md
    assert "| a | 0.000185 |" in md and "| b | 12.5 |" in md
    assert "![pareto](pareto.png)" in md


def test_written_files_use_lf_and_are_ascii_safe(tmp_path):
    out = write(tmp_path)
    for name in ("results.json", "report.md", "run.json"):
        raw = (out / name).read_bytes()
        assert b"\r" not in raw, name
        assert "\u2014".encode() not in raw, name


def test_nan_and_inf_become_null_in_results_json(tmp_path):
    out = write(
        tmp_path,
        results={"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": np.float64("nan")}},
    )
    text = (out / "results.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_custom_labels_replace_the_default_line(tmp_path):
    out = write(tmp_path, results={"labels": ["Latency is simulated.", "Prices are assumed."]})
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "Latency is simulated." in md and "Prices are assumed." in md
    assert "fitted from recorded costs" not in md


def test_missing_cells_render_as_n_a(tmp_path):
    table = pd.DataFrame({"policy": ["a", "b"], "ci": [None, 0.5], "n": [1, 2]})
    out = write(tmp_path, tables={"t": table})
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "| a | n/a | 1 |" in md and "| b | 0.5 | 2 |" in md


def test_pipes_in_cells_are_escaped(tmp_path):
    out = write(tmp_path, tables={"t": pd.DataFrame({"name": ["a|b"]})})
    assert "| a\\|b |" in (out / "report.md").read_text(encoding="utf-8")


def test_write_results_rejects_an_unsafe_figure_name(tmp_path):
    with pytest.raises(ValueError, match="figure name"):
        write(tmp_path, figures={"../escape": figure()})
