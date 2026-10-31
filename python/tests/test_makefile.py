"""The Makefile targets that cannot be seen from the Go or Python tests.

``make py-setup`` has to find the interpreter of a venv it has only just created, and
``make verify`` has to run what CI runs. Both are checked here with GNU make itself.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from routepilot_train.runinfo import repo_root

MAKE = shutil.which("make")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(MAKE is None or BASH is None, reason="GNU make and bash needed")

# Stands in for ``python -m venv .venv`` on a Windows checkout: the interpreter lands in
# Scripts/, not bin/, and "installing" records what it was asked to do.
FAKE_VENV = """\
mkdir -p .venv/Scripts
printf '#!/bin/sh\\necho "$@" > pip-args.txt\\n' > .venv/Scripts/python.exe
chmod +x .venv/Scripts/python.exe
"""


def make(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(MAKE), *args], cwd=cwd, capture_output=True, text=True, timeout=120, check=False
    )


def test_py_setup_uses_the_interpreter_of_the_venv_it_just_created(tmp_path: Path) -> None:
    shutil.copy(repo_root() / "Makefile", tmp_path / "Makefile")
    (tmp_path / "make-venv.sh").write_text(FAKE_VENV, encoding="utf-8", newline="\n")

    done = make("py-setup", "PYTHON_BOOTSTRAP=bash make-venv.sh", cwd=tmp_path)

    assert done.returncode == 0, done.stderr
    installed = (tmp_path / "pip-args.txt").read_text(encoding="utf-8").strip()
    assert installed == "-m pip install -e python[dev]"


def test_verify_runs_what_ci_runs() -> None:
    done = make("-n", "verify", cwd=repo_root())

    assert done.returncode == 0, done.stderr
    plan = done.stdout
    assert "gofmt -l" in plan and "go vet ./..." in plan  # check
    assert "go test -race -count=1 ./..." in plan
    assert "go mod tidy" in plan  # tidy-check
    assert "ruff check python" in plan  # py-check
    assert "bench/scripts/gate.py --check --pool smoke" in plan  # bench-smoke
