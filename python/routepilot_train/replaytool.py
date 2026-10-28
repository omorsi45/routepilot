"""Run the Go ``routepilot replay`` command and read its output.

The replay output holds item ids, hashes and numbers only, never prompt text.
"""

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# The columns of one replay record, in the order the command writes them.
COLUMNS = [
    "item_id",
    "dataset",
    "split",
    "prompt_sha256_16",
    "policy",
    "policy_version",
    "model",
    "success",
    "cost_picousd",
    "in_upper",
    "k",
    "p",
    "l",
    "cost_exp_picousd",
    "cost_max_picousd",
    "floor",
    "relaxed",
    "ood",
    "ood_score",
    "explored",
    "propensity",
    "ttft_ms_sim",
    "e2e_ms_sim",
    "slo_ms",
    "slo_violated",
    "infeasible",
    "error",
]
_FLOAT_COLUMNS = [
    "k",
    "p",
    "l",
    "cost_exp_picousd",
    "cost_max_picousd",
    "floor",
    "ood_score",
    "propensity",
    "ttft_ms_sim",
    "e2e_ms_sim",
    "slo_ms",
]
_BOOL_COLUMNS = ["success", "ood", "explored", "infeasible"]


class GoToolchainError(RuntimeError):
    """The installed Go toolchain cannot build this module (too old, or none installed)."""


def build_go(repo: Path) -> Path:
    """Build ``./cmd/routepilot`` into ``repo/bin`` and return the binary path.

    The build never downloads a toolchain: ``GOTOOLCHAIN`` defaults to ``local``, so a Go
    older than go.mod asks for raises GoToolchainError instead of fetching one.
    """
    binary = repo / "bin" / ("routepilot.exe" if sys.platform == "win32" else "routepilot")
    env = {**os.environ}
    env.setdefault("GOTOOLCHAIN", "local")
    try:
        proc = subprocess.run(
            ["go", "build", "-o", str(binary), "./cmd/routepilot"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError as err:
        raise GoToolchainError("the go toolchain is not on PATH") from err
    if proc.returncode != 0:
        message = f"go build failed: {proc.stderr.strip()}"
        if "requires go" in proc.stderr or "GOTOOLCHAIN" in proc.stderr:
            raise GoToolchainError(message)
        raise RuntimeError(message)
    return binary


def read_replay(path: Path) -> pd.DataFrame:
    """Read a replay JSONL file into a frame with one row per item."""
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    for column in _FLOAT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in _BOOL_COLUMNS:
        df[column] = df[column].astype(bool)
    df["cost_picousd"] = df["cost_picousd"].astype("int64")
    df["in_upper"] = df["in_upper"].astype("int64")
    df["slo_violated"] = df["slo_violated"].astype("boolean")
    df["error"] = df["error"].fillna("").astype(str)
    return df


def replay(
    binary: Path,
    config: Path,
    input: Path,
    split: str,
    policy: str,
    out: Path,
    extra: Sequence[str] = (),
) -> pd.DataFrame:
    """Run ``routepilot replay`` and return its records as a frame.

    ``extra`` holds more flags, such as ``["--seed", "3"]`` or ``["--model", "gpt-5"]``.
    A failing command raises RuntimeError with its message.
    """
    command = [
        str(binary),
        "replay",
        "--config",
        str(config),
        "--input",
        str(input),
        "--split",
        split,
        "--policy",
        policy,
        *extra,
        "--out",
        str(out),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"routepilot replay failed ({proc.returncode}): {proc.stderr.strip()}")
    return read_replay(out)
