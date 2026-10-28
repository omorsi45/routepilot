import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "gate.py"
FIXTURE = ROOT / "testdata" / "bench" / "gate"

BASELINE = {
    "pool": "demo",
    "split": "test",
    "policy": "rules",
    "items": 100,
    "success_rate": 0.60,
    "mean_cost_picousd": 1000.0,
}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def binary() -> Path:
    """The routepilot binary built from this checkout; skips when Go cannot build it."""
    if shutil.which("go") is None:
        pytest.skip("go is not on PATH")
    try:
        return replaytool.build_go(ROOT)
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))


def fixture_args(mode: str, baseline: Path | None = None) -> list[str]:
    return [
        mode,
        "--pool",
        "smoke",
        "--policy",
        "rules",
        "--config",
        str(FIXTURE / "config.yaml"),
        "--input",
        str(FIXTURE / "items.jsonl"),
        "--baseline",
        str(baseline or FIXTURE / "baseline.json"),
    ]


def test_compare_pass():
    gate = load_script()
    assert gate.compare(dict(BASELINE), BASELINE) == []
    better = {**BASELINE, "success_rate": 0.75, "mean_cost_picousd": 400.0}
    assert gate.compare(better, BASELINE) == []  # improvements never fail
    # exactly at both limits is still a pass: a one point drop and a 5% rise
    edge = {**BASELINE, "success_rate": 0.59, "mean_cost_picousd": 1050.0}
    assert gate.compare(edge, BASELINE) == []


def test_compare_success_drop_fails():
    gate = load_script()
    current = {**BASELINE, "success_rate": 0.58}

    problems = gate.compare(current, BASELINE)

    assert len(problems) == 1
    message = problems[0]
    assert "success_rate" in message
    assert "0.6" in message and "0.58" in message  # both values
    assert "0.01" in message  # the limit
    assert gate.compare(current, BASELINE, max_success_drop=0.05) == []


def test_compare_cost_rise_fails():
    gate = load_script()
    current = {**BASELINE, "mean_cost_picousd": 1060.0}

    problems = gate.compare(current, BASELINE)

    assert len(problems) == 1
    message = problems[0]
    assert "mean_cost_picousd" in message
    assert "1000" in message and "1060" in message  # both values
    assert "5%" in message  # the limit
    assert gate.compare(current, BASELINE, max_cost_rise=0.10) == []


def test_compare_reports_every_problem_and_a_baseline_for_other_data():
    gate = load_script()
    both = {**BASELINE, "success_rate": 0.40, "mean_cost_picousd": 5000.0}
    assert len(gate.compare(both, BASELINE)) == 2

    # numbers of another pool, policy or item set are not comparable: say so, by key
    for key, other in (
        ("pool", "other"),
        ("policy", "cheapest"),
        ("items", 99),
        ("split", "calib"),
    ):
        problems = gate.compare({**BASELINE, key: other}, BASELINE)
        assert len(problems) == 1 and key in problems[0], key

    # a baseline that costs nothing allows no cost at all
    free = {**BASELINE, "mean_cost_picousd": 0.0}
    assert gate.compare(dict(free), free) == []
    assert len(gate.compare({**free, "mean_cost_picousd": 1.0}, free)) == 1


def test_smoke_fixture_passes(binary: Path, capsys: pytest.CaptureFixture[str]):
    code = load_script().main(fixture_args("--check"), binary=binary)

    out = capsys.readouterr().out
    assert code == 0, out
    assert "ok" in out and "success_rate" in out and "mean_cost_picousd" in out


def test_smoke_fixture_routes_to_more_than_one_model(binary: Path, tmp_path: Path):
    """The rules in the fixture send code, long and plain prompts to different models."""
    frame = replaytool.replay(
        binary,
        FIXTURE / "config.yaml",
        FIXTURE / "items.jsonl",
        "test",
        "rules",
        tmp_path / "rules.jsonl",
    )
    picked = dict(zip(frame["item_id"], frame["model"], strict=True))
    assert picked["gate-b/1"] == "gate-large"  # code
    assert picked["gate-b/6"] == "gate-mid"  # long
    assert picked["gate-b/3"] == "gate-small"  # neither
    assert set(picked.values()) == {"gate-small", "gate-mid", "gate-large"}


def test_check_fails_when_the_baseline_is_better(
    binary: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    baseline = json.loads((FIXTURE / "baseline.json").read_text(encoding="utf-8"))
    baseline["success_rate"] = min(1.0, baseline["success_rate"] + 0.2)
    baseline["mean_cost_picousd"] = baseline["mean_cost_picousd"] / 2
    stricter = tmp_path / "stricter.json"
    stricter.write_text(json.dumps(baseline), encoding="utf-8")

    code = load_script().main(fixture_args("--check", stricter), binary=binary)

    captured = capsys.readouterr()
    assert code == 1
    assert "success_rate" in captured.err and "mean_cost_picousd" in captured.err


def test_update_writes_the_baseline_and_check_then_passes(
    binary: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    gate = load_script()
    target = tmp_path / "baseline.json"

    assert gate.main(fixture_args("--update", target), binary=binary) == 0

    raw = target.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    written = json.loads(raw)
    assert list(written) == [
        "pool",
        "split",
        "policy",
        "items",
        "success_rate",
        "mean_cost_picousd",
        "git_sha",
        "updated_at",
    ]
    assert written["pool"] == "smoke" and written["split"] == "test"
    assert written["policy"] == "rules" and written["items"] == 10  # the test split only
    # the committed fixture baseline holds the same measured numbers
    committed = json.loads((FIXTURE / "baseline.json").read_text(encoding="utf-8"))
    for key in ("pool", "split", "policy", "items", "success_rate", "mean_cost_picousd"):
        assert written[key] == committed[key], key
    assert gate.main(fixture_args("--check", target), binary=binary) == 0
    capsys.readouterr()


def test_exactly_one_mode_is_required():
    gate = load_script()
    with pytest.raises(SystemExit) as neither:
        gate.main(["--pool", "smoke"])
    assert neither.value.code == 2
    with pytest.raises(SystemExit) as both:
        gate.main(["--check", "--update"])
    assert both.value.code == 2


def test_a_missing_baseline_is_an_input_error(
    binary: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = load_script().main(fixture_args("--check", tmp_path / "absent.json"), binary=binary)

    assert code == 2
    assert "absent.json" in capsys.readouterr().err
