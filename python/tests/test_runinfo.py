import ast
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from routepilot_train import runinfo
from routepilot_train.data import selection

EXPECTED_KEYS = [
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


def test_run_json_keys(tmp_path: Path):
    started = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)
    run_dir = tmp_path / "data" / "20261027T120000Z-abc1234"
    info = runinfo.write_run_json(
        run_dir, "data", {"split_salt": "s", "numpy": 7}, ["python", "bench/x.py"], None, started
    )
    on_disk = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert on_disk == info
    assert list(on_disk) == EXPECTED_KEYS
    assert set(on_disk["machine"]) == {"cpu", "ram_gb", "os"}
    assert on_disk["run_id"] == run_dir.name
    assert on_disk["experiment"] == "data"
    assert on_disk["policy_version"] is None
    assert on_disk["dataset_sha256"] == selection.DATASET_SHA256
    assert on_disk["seeds"] == {"split_salt": "s", "numpy": 7}
    assert on_disk["command"] == ["python", "bench/x.py"]
    assert on_disk["started_at"] == "2026-10-27T12:00:00Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", on_disk["ended_at"])
    assert on_disk["ended_at"] >= on_disk["started_at"]
    assert re.fullmatch(r"[0-9a-f]{40}", on_disk["git_sha"])
    assert isinstance(on_disk["git_dirty"], bool)
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", on_disk["python_version"])
    assert on_disk["go_version"] is None or on_disk["go_version"].startswith("go")
    assert isinstance(on_disk["machine"]["cpu"], str) and on_disk["machine"]["cpu"]
    assert on_disk["machine"]["ram_gb"] is None or on_disk["machine"]["ram_gb"] > 0
    raw = (run_dir / "run.json").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw


def test_run_json_keeps_a_policy_version(tmp_path: Path):
    info = runinfo.write_run_json(
        tmp_path / "r", "e", {}, ["x"], "pol-1", datetime(2026, 1, 2, 3, 4, 5)
    )
    assert info["policy_version"] == "pol-1"
    assert info["started_at"] == "2026-01-02T03:04:05Z"  # a naive time is taken as UTC


def test_run_json_merges_extra_under_a_measurement_key(tmp_path: Path):
    extra = {"clock_step_ns": 40, "zero_fraction": 0.0}
    info = runinfo.write_run_json(
        tmp_path / "r", "e", {}, ["x"], None, datetime(2026, 1, 2, 3, 4, 5), extra=extra
    )
    on_disk = json.loads((tmp_path / "r" / "run.json").read_text(encoding="utf-8"))
    assert on_disk == info
    assert on_disk["measurement"] == extra
    assert list(on_disk) == [*EXPECTED_KEYS, "measurement"]
    assert on_disk["seeds"] == {}  # the extra fields do not leak into the other keys


def test_new_run_id_format():
    now = datetime(2026, 10, 27, 23, 59, 58, tzinfo=UTC)
    run_id = runinfo.new_run_id(now)
    assert re.fullmatch(r"20261027T235958Z-[0-9a-f]{7,}", run_id)
    assert runinfo.new_run_id(now) == run_id
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{7,}", runinfo.new_run_id())


def test_a_given_git_state_is_recorded_instead_of_the_current_one(tmp_path: Path):
    now = datetime(2026, 10, 27, 23, 59, 58, tzinfo=UTC)
    state = ("abcdef0123456789abcdef0123456789abcdef01", True)
    assert runinfo.new_run_id(now, state) == "20261027T235958Z-abcdef0"
    assert runinfo.new_run_id(now, (None, None)) == "20261027T235958Z-nogit"
    info = runinfo.write_run_json(tmp_path / "r", "e", {}, ["x"], None, now, git=state)
    assert (info["git_sha"], info["git_dirty"]) == state
    on_disk = json.loads((tmp_path / "r" / "run.json").read_text(encoding="utf-8"))
    assert (on_disk["git_sha"], on_disk["git_dirty"]) == state


RUN_WRITERS = {"write_results", "write_run_json"}


def _run_calls(path: Path) -> list[tuple[int, str, ast.Call]]:
    """Every call in ``path`` of ``new_run_id``, ``write_results`` or ``write_run_json``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in RUN_WRITERS | {"new_run_id"}:
                found.append((node.lineno, node.func.attr, node))
    return found


def test_every_run_takes_the_git_state_when_it_starts():
    """A run that calls the writers without the state taken at its start records the end state."""
    root = runinfo.repo_root()
    sources = sorted((root / "bench" / "scripts").glob("*.py"))
    sources += sorted((root / "python" / "routepilot_train").rglob("*.py"))
    missing = []
    seen = 0
    for path in sources:
        for line, name, call in _run_calls(path):
            seen += 1
            keywords = {k.arg for k in call.keywords}
            ok = "git" in keywords or (name == "new_run_id" and len(call.args) + len(keywords) >= 2)
            if not ok:
                missing.append(f"{path.relative_to(root).as_posix()}:{line} {name}")
    assert seen > 40  # the walk found the experiments, not nothing
    assert missing == []


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_git_state_reports_dirty_but_ignores_bench_results(tmp_path: Path):
    git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "commit", "-q", "-m", "init")

    sha, dirty = runinfo.git_state(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", sha) and dirty is False

    results = tmp_path / "bench" / "results" / "data" / "run"
    results.mkdir(parents=True)
    (results / "results.json").write_text("{}\n", encoding="utf-8", newline="\n")
    assert runinfo.git_state(tmp_path) == (sha, False)  # a fresh results directory is not dirt

    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8", newline="\n")
    assert runinfo.git_state(tmp_path) == (sha, True)


def test_go_version_is_none_without_go(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    assert runinfo.go_version() is None
