"""Tests for bench/scripts/clock_probe.py: the five probes, their summary and the written run.

The probe and the Go build are replaced by fakes; the real `rp-load --clock-probe` runs only when
someone runs the script.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import clock_probe as cp  # noqa: E402

EM_DASH = "\u2014"


def probes(steps: list[int], zeros: list[float] | None = None) -> list[dict]:
    zeros = zeros or [0.0] * len(steps)
    return [{"clock_step_ns": s, "zero_fraction": z} for s, z in zip(steps, zeros, strict=True)]


def test_five_probes_by_default() -> None:
    assert cp.PROBES == 5


def test_summary_gives_min_median_max_and_the_worst_zero_fraction() -> None:
    summary = cp.summarize(probes([30, 31, 30, 100, 30], [0.0, 0.001, 0.0, 0.02, 0.0]))

    assert summary["probes"] == 5
    assert summary["min_step_ns"] == 30
    assert summary["median_step_ns"] == 30
    assert summary["max_step_ns"] == 100
    assert summary["max_zero_fraction"] == pytest.approx(0.02)
    assert summary["threshold_ns"] == 1000
    assert summary["coarse"] is False


def test_summary_flags_a_clock_step_above_the_threshold() -> None:
    fine = cp.summarize(probes([1000] * 5))
    coarse = cp.summarize(probes([1000, 1000, 1000, 1000, 503_000]))

    assert fine["coarse"] is False
    assert coarse["coarse"] is True
    assert coarse["max_step_ns"] == 503_000


def test_summary_needs_at_least_one_probe() -> None:
    with pytest.raises(ValueError, match="no probes"):
        cp.summarize([])


def test_collect_calls_the_probe_once_per_run_and_keeps_every_result() -> None:
    answers = iter(probes([30, 31, 32, 33, 34]))
    calls: list[Path] = []

    def fake(binary: Path) -> dict:
        calls.append(binary)
        return next(answers)

    got = cp.collect(Path("rp-load"), 5, fake)

    assert [p["clock_step_ns"] for p in got] == [30, 31, 32, 33, 34]
    assert calls == [Path("rp-load")] * 5


def test_build_loadgen_builds_only_rp_load_into_the_bin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    binary = cp.build_loadgen(tmp_path / "bin")

    suffix = ".exe" if sys.platform == "win32" else ""
    assert binary == tmp_path / "bin" / f"rp-load{suffix}"
    assert seen == [["go", "build", "-o", str(binary), "./cmd/rp-load"]]


def test_build_loadgen_reports_a_failed_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="no go here")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no go here"):
        cp.build_loadgen(tmp_path / "bin")


def test_run_writes_results_report_and_run_json(tmp_path: Path) -> None:
    answers = iter(probes([30, 30, 31, 30, 30], [0.0, 0.0, 0.002, 0.0, 0.0]))

    out_dir = cp.run_clock_probe(
        tmp_path / "clock",
        ["python", "bench/scripts/clock_probe.py"],
        build=lambda bin_dir: Path("rp-load"),
        probe=lambda binary: next(answers),
    )

    assert out_dir.parent == tmp_path / "clock"
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert [p["clock_step_ns"] for p in results["probes"]] == [30, 30, 31, 30, 30]
    assert [p["zero_fraction"] for p in results["probes"]] == [0.0, 0.0, 0.002, 0.0, 0.0]
    assert results["os"] == platform.platform()
    assert results["summary"]["max_step_ns"] == 31
    assert results["summary"]["coarse"] is False

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# clock\n")
    assert "**Labels:**" in report
    assert "| clock_step_ns |" in report
    assert platform.platform() in report
    assert EM_DASH not in report

    info = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert info["experiment"] == "clock"
    assert info["run_id"] == out_dir.name
    assert info["command"] == ["python", "bench/scripts/clock_probe.py"]
    assert info["measurement"]["max_step_ns"] == 31
