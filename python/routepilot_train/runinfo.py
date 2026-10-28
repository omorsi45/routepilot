"""Run identity and metadata for every experiment that writes under bench/results.

``run.json`` records what ran, on what code and data, and on which machine, so a
published number can be traced back. Nothing here needs a third-party package.
"""

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routepilot_train.data import selection

# Results directories are output, not input: a new one must not make the tree look dirty.
_RESULTS_PATHSPEC = ":(exclude)bench/results"


def repo_root() -> Path:
    """The repository root, found from this file (python/routepilot_train/runinfo.py)."""
    return Path(__file__).resolve().parents[2]


def _utc(moment: datetime) -> datetime:
    """A naive time is taken to be UTC."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _iso(moment: datetime) -> str:
    return _utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_state(root: Path | None = None) -> tuple[str | None, bool | None]:
    """The full HEAD SHA and whether the tree has changes outside bench/results.

    Both are None when ``root`` is not a git checkout or git is missing.
    """
    root = root or repo_root()
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git(root, "status", "--porcelain", "--", ".", _RESULTS_PATHSPEC)
    return sha.strip(), None if status is None else bool(status.strip())


def new_run_id(now: datetime | None = None) -> str:
    """``<UTC yyyymmddTHHMMSSZ>-<git short SHA>``; the SHA part is ``nogit`` outside git."""
    moment = _utc(now or datetime.now(UTC))
    sha, _ = git_state()
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{sha[:7] if sha else 'nogit'}"


def go_version() -> str | None:
    """The Go toolchain version such as ``go1.27.2``, or None when go is not available."""
    exe = shutil.which("go")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    parts = proc.stdout.split()
    if proc.returncode != 0 or len(parts) < 3 or parts[:2] != ["go", "version"]:
        return None
    return parts[2]


def _cpu_name() -> str:
    name = ""
    try:
        system = platform.system()
        if system == "Windows":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                name = str(winreg.QueryValueEx(handle, "ProcessorNameString")[0])
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        elif system == "Darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            name = proc.stdout
    except (ImportError, OSError, subprocess.SubprocessError):
        name = ""
    name = " ".join(name.split()) or platform.processor() or platform.machine() or "unknown"
    return f"{name} ({os.cpu_count()} threads)"


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            total = status.ullTotalPhys
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return round(total / 2**30, 1) if total > 0 else None


def write_run_json(
    dir: Path,
    experiment: str,
    seeds: Mapping[str, Any],
    command: Sequence[str],
    policy_version: str | None,
    started_at: datetime,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``dir/run.json`` (creating ``dir``) and return its content.

    ``run_id`` is the directory name, which callers build with ``new_run_id``.
    ``ended_at`` is the time of this call. ``extra``, when given, is how the run was measured
    (for example the clock step of a timing experiment) and is stored under ``measurement``;
    without it the key is absent.
    """
    sha, dirty = git_state()
    info: dict[str, Any] = {
        "run_id": dir.name,
        "experiment": experiment,
        "git_sha": sha,
        "git_dirty": dirty,
        "policy_version": policy_version,
        "dataset_sha256": selection.DATASET_SHA256,
        "seeds": dict(seeds),
        "machine": {"cpu": _cpu_name(), "ram_gb": _ram_gb(), "os": platform.platform()},
        "go_version": go_version(),
        "python_version": platform.python_version(),
        "started_at": _iso(started_at),
        "ended_at": _iso(datetime.now(UTC)),
        "command": list(command),
    }
    if extra is not None:
        info["measurement"] = dict(extra)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    return info
