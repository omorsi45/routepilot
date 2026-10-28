import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "litellm_compare.py"
EM_DASH = "\u2014"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("litellm_compare", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lc() -> ModuleType:
    return load_script()


def load_block(base: float, completed: int = 1000, stream: bool = False) -> dict[str, Any]:
    """A rp-load result whose latency percentiles are base plus a fixed ladder."""
    ladder = {"p50": 0.0, "p90": 1.0, "p95": 2.0, "p99": 3.0, "p99.9": 4.0, "max": 6.0}
    block: dict[str, Any] = {
        "scheduled": completed,
        "completed": completed,
        "errors": 0,
        "dropped": 0,
        "throughput_rps": 100.0,
        "latency_ms": {k: base + v for k, v in ladder.items()},
        "ttft_ms": {k: base / 2 + v for k, v in ladder.items()} if stream else None,
        "launch_lag_ms": {"p50": 0.1, "p99": 0.5, "max": 2.0},
    }
    return block


def run(path: str, rps: int, stream: bool, base: float) -> dict[str, Any]:
    mode = "stream" if stream else "nonstream"
    return {
        "name": f"{path}-{rps}rps-{mode}",
        "path": path,
        "rps": rps,
        "stream": stream,
        "model": "bench-model",
        "preflight": {"status": 200, "headers": {}},
        "load": load_block(base, stream=stream),
        "server": None,
    }


# --- the matrix -------------------------------------------------------------------------------


def test_matrix_runs_the_three_paths_together_in_each_cell_group(lc: ModuleType) -> None:
    specs = lc.matrix((100, 500))
    assert [s.name for s in specs] == [
        "direct-100rps-nonstream",
        "gateway-100rps-nonstream",
        "litellm-100rps-nonstream",
        "direct-100rps-stream",
        "gateway-100rps-stream",
        "litellm-100rps-stream",
        "direct-500rps-nonstream",
        "gateway-500rps-nonstream",
        "litellm-500rps-nonstream",
        "direct-500rps-stream",
        "gateway-500rps-stream",
        "litellm-500rps-stream",
    ]
    assert all(s.model == "bench-model" for s in specs)


def test_default_levels_are_the_briefed_rates(lc: ModuleType) -> None:
    assert lc.LEVELS == (100, 500, 1000, 2000)
    assert lc.WARMUP_S == 10 and lc.DURATION_S == 60
    assert len(lc.matrix(lc.LEVELS)) == 24


# --- comparisons ------------------------------------------------------------------------------


def test_compare_gives_both_gateways_against_direct(lc: ModuleType) -> None:
    runs = [
        run("direct", 100, False, 37.0),
        run("gateway", 100, False, 37.5),
        run("litellm", 100, False, 40.0),
    ]
    (got,) = lc.compare(runs)
    assert (got["rps"], got["stream"]) == (100, False)
    assert got["routepilot_vs_direct_ms"]["p95"] == pytest.approx(0.5)
    assert got["litellm_vs_direct_ms"]["p95"] == pytest.approx(3.0)
    assert got["litellm_vs_routepilot_ms"]["p95"] == pytest.approx(2.5)
    assert got["routepilot_vs_direct_ms"]["p99.9"] == pytest.approx(0.5)
    assert got["ttft_diff_ms"] is None  # not a streaming cell


def test_compare_streaming_cells_also_difference_the_time_to_first_token(lc: ModuleType) -> None:
    runs = [
        run("direct", 500, True, 40.0),
        run("gateway", 500, True, 41.0),
        run("litellm", 500, True, 44.0),
    ]
    (got,) = lc.compare(runs)
    assert got["ttft_diff_ms"]["routepilot_vs_direct"]["p50"] == pytest.approx(0.5)
    assert got["ttft_diff_ms"]["litellm_vs_direct"]["p50"] == pytest.approx(2.0)
    assert got["ttft_diff_ms"]["litellm_vs_routepilot"]["p50"] == pytest.approx(1.5)


def test_compare_needs_all_three_paths(lc: ModuleType) -> None:
    with pytest.raises(KeyError, match="litellm"):
        lc.compare([run("direct", 100, False, 37.0), run("gateway", 100, False, 37.5)])


# --- the LiteLLM side -------------------------------------------------------------------------


def test_litellm_config_routes_one_model_to_the_mock(lc: ModuleType) -> None:
    cfg = lc.litellm_config(19191)
    (model,) = cfg["model_list"]
    assert model["model_name"] == "bench-model"
    assert model["litellm_params"]["model"] == "openai/bench-model"
    assert model["litellm_params"]["api_base"] == "http://127.0.0.1:19191/v1"
    assert model["litellm_params"]["api_key"].startswith("test-key-")
    assert cfg["litellm_settings"] == {"telemetry": False, "drop_params": True}
    # No master key (Ruling V1-R37); 1.106.6 only starts that way when told it is intended.
    assert "master_key" not in cfg["general_settings"]
    assert cfg["general_settings"]["dangerously_permit_weak_or_unset_master_key"] is True


def test_the_committed_config_has_every_placeholder(lc: ModuleType) -> None:
    text = lc.LITELLM_TEMPLATE.read_text(encoding="utf-8")
    assert "{{MOCK_PORT}}" in text
    assert EM_DASH not in text
    assert yaml.safe_load(text.replace("{{MOCK_PORT}}", "1"))["model_list"]


def test_litellm_command_binds_loopback_with_eight_workers(lc: ModuleType) -> None:
    command = lc.litellm_command(Path("/venv/bin/litellm"), Path("/tmp/c.yaml"), 4000)
    assert command == [
        str(Path("/venv/bin/litellm")),
        "--config",
        str(Path("/tmp/c.yaml")),
        "--host",
        "127.0.0.1",
        "--port",
        "4000",
        "--num_workers",
        "8",
    ]
    assert lc.WORKERS == 8


def test_litellm_environment_does_not_fetch_the_cost_map(lc: ModuleType) -> None:
    env = lc.litellm_env({"PATH": "/bin"})
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert env["LITELLM_LOG"] == "ERROR"
    assert env["PATH"] == "/bin"


def test_version_comes_from_the_response_header(lc: ModuleType) -> None:
    assert lc.version_from_headers({"x-litellm-version": "1.106.6", "other": "x"}) == "1.106.6"
    assert lc.version_from_headers({"x-other": "1"}) is None


def test_pinned_version_is_read_from_the_requirements(lc: ModuleType) -> None:
    text = (
        "# header\n\nfastapi==0.162.0 \\\n    --hash=sha256:abc\n"
        "litellm==1.106.6 \\\n    --hash=sha256:def\n"
    )
    assert lc.pinned_litellm_version(text) == "1.106.6"
    with pytest.raises(ValueError, match="litellm"):
        lc.pinned_litellm_version("fastapi==0.162.0\n")


def test_the_committed_requirements_pin_litellm_and_every_dependency_exactly(
    lc: ModuleType,
) -> None:
    text = lc.REQUIREMENTS.read_text(encoding="utf-8")
    assert lc.pinned_litellm_version(text) == "1.106.6"
    requirements = [
        line
        for line in text.splitlines()
        if line and not line.startswith(("#", " ", "\t")) and not line.startswith("--")
    ]
    assert len(requirements) > 50
    for line in requirements:
        name_version = line.split(" ", 1)[0]
        assert "==" in name_version and ">" not in name_version and "~" not in name_version, line
    assert "--hash=sha256:" in text  # installed with --require-hashes
    assert EM_DASH not in text


# --- stopping the proxy -----------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are a Linux feature here")
def test_stop_group_ends_the_proxy_and_its_workers(lc: ModuleType) -> None:
    # A parent that starts two children, as the proxy does with its workers.
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 300 & sleep 300 & wait"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pgid = os.getpgid(proc.pid)
    time.sleep(0.3)
    lc.stop_group(proc)
    assert proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)  # no process of the group is left


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are a Linux feature here")
def test_stop_group_kills_what_ignores_the_first_signal(lc: ModuleType) -> None:
    proc = subprocess.Popen(
        ["bash", "-c", "trap '' TERM; sleep 300 & wait"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pgid = os.getpgid(proc.pid)
    time.sleep(0.3)
    lc.stop_group(proc, grace_s=0.5)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    assert signal.SIGKILL  # the fallback signal exists on this platform


# --- the report -------------------------------------------------------------------------------

MACHINE = {
    "cpu": "Test CPU (8 threads)",
    "ram_gb": 15.2,
    "os": "Linux-6.18-microsoft-standard-WSL2",
}


def full_results(lc: ModuleType) -> dict[str, Any]:
    runs = []
    for stream in (False, True):
        runs += [
            run("direct", 100, stream, 37.0),
            run("gateway", 100, stream, 37.4),
            run("litellm", 100, stream, 41.0),
        ]
    return {
        "experiment": "litellm",
        "settings": {
            "warmup_s": 10,
            "duration_s": 60,
            "levels": [100],
            "workers": 8,
            "litellm_version": "1.106.6",
            "litellm_python": "Python 3.14.9",
            "mock_args": ["--seed", "1"],
            "litellm_env": {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_LOG": "ERROR"},
        },
        "runs": runs,
        "comparisons": lc.compare(runs),
        "clock": {"clock_step_ns": 30, "zero_fraction": 0.0, "coarse": False},
        "wall_time_s": 123.4,
    }


def test_render_report(lc: ModuleType) -> None:
    text = lc.render_report(full_results(lc), MACHINE, "20261029T000000Z-abc1234", "go1.27.2")
    assert EM_DASH not in text
    assert "Labels:" in text and "nothing simulated" in text
    assert "difference of percentiles, not a percentile of differences" in text
    assert "1.106.6" in text
    assert "8 worker" in text
    assert "LiteLLM" in text and "RoutePilot" in text
    assert "WSL2" in text
    assert "30 ns" in text  # the clock step is recorded
    assert "master key" in text
    assert "20261029T000000Z-abc1234" in text
    # Each of the six cells appears in the absolute table.
    for name in ("litellm-100rps-nonstream", "gateway-100rps-stream", "direct-100rps-stream"):
        assert name in text
    # Vendor claims are not restated as measured here.
    assert "vendor" in text.lower()


def test_render_report_shows_errors_and_drops(lc: ModuleType) -> None:
    results = full_results(lc)
    results["runs"][2]["load"].update(errors=7, dropped=3)
    results["comparisons"] = lc.compare(results["runs"])
    text = lc.render_report(results, MACHINE, "20261029T000000Z-abc1234", "go1.27.2")
    row = next(line for line in text.splitlines() if line.startswith("| litellm-100rps-nonstream"))
    assert "| 7 |" in row and "| 3 |" in row


# --- main -------------------------------------------------------------------------------------


def fake_main_setup(
    lc: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step_ns: int
) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(lc, "COOLDOWN_S", 0.0)
    monkeypatch.setattr(
        lc.overhead,
        "build_binaries",
        lambda bin_dir: {n: bin_dir / n for n in ("rp-mock", "routepilot", "rp-load")},
    )
    monkeypatch.setattr(
        lc.overhead,
        "probe_clock",
        lambda binary: {"clock_step_ns": step_ns, "zero_fraction": 0.0},
    )

    def fake_cell(spec: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(spec.name)
        r = run(
            spec.path, spec.rps, spec.stream, {"direct": 37.0, "gateway": 37.5}.get(spec.path, 40.0)
        )
        if spec.path == "litellm":
            r["preflight"] = {"status": 200, "headers": {"x-litellm-version": "1.106.6"}}
        return r

    monkeypatch.setattr(lc, "measure_cell", fake_cell)
    monkeypatch.setattr(lc, "litellm_python_version", lambda binary: "Python 3.14.9")
    return calls


def test_main_writes_the_three_files(
    lc: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = fake_main_setup(lc, monkeypatch, tmp_path, 30)
    litellm_bin = tmp_path / "litellm"
    litellm_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    out = tmp_path / "out"
    code = lc.main(
        [
            "--rps",
            "100",
            "--out",
            str(out),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--litellm-bin",
            str(litellm_bin),
        ]
    )
    assert code == 0
    assert len(calls) == 6
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["settings"]["litellm_version"] == "1.106.6"
    assert results["settings"]["workers"] == 8
    assert results["clock"]["coarse"] is False
    info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert info["experiment"] == "litellm"
    assert info["measurement"]["clock_step_ns"] == 30
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# ")


def test_main_refuses_a_coarse_clock_before_measuring(
    lc: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = fake_main_setup(lc, monkeypatch, tmp_path, 503_000)
    litellm_bin = tmp_path / "litellm"
    litellm_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    out = tmp_path / "out"
    code = lc.main(
        [
            "--rps",
            "100",
            "--out",
            str(out),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--litellm-bin",
            str(litellm_bin),
        ]
    )
    assert code == 1
    assert calls == []
    assert not out.exists()
    assert "503000 ns" in capsys.readouterr().err


def test_main_fails_when_the_proxy_reports_another_version(
    lc: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_main_setup(lc, monkeypatch, tmp_path, 30)

    def other_version(spec: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        r = run(spec.path, spec.rps, spec.stream, 40.0)
        if spec.path == "litellm":
            r["preflight"] = {"status": 200, "headers": {"x-litellm-version": "1.2.3"}}
        return r

    monkeypatch.setattr(lc, "measure_cell", other_version)
    litellm_bin = tmp_path / "litellm"
    litellm_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    code = lc.main(
        [
            "--rps",
            "100",
            "--out",
            str(tmp_path / "out"),
            "--scratch-root",
            str(tmp_path / "s"),
            "--litellm-bin",
            str(litellm_bin),
        ]
    )
    assert code == 1
    assert "1.2.3" in capsys.readouterr().err


def test_main_needs_the_litellm_binary(
    lc: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = fake_main_setup(lc, monkeypatch, tmp_path, 30)
    code = lc.main(
        [
            "--rps",
            "100",
            "--out",
            str(tmp_path / "out"),
            "--scratch-root",
            str(tmp_path / "s"),
            "--litellm-bin",
            str(tmp_path / "missing"),
        ]
    )
    assert code == 2
    assert calls == []
    assert "litellm" in capsys.readouterr().err
