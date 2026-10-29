"""Write one experiment's results: ``results.json``, ``report.md``, figures and ``run.json``.

Every table lands in the report as Markdown. A "Labels" line says which quantities are
simulated, fitted or assumed, so no number in the report passes as measured when it is not.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from routepilot_train import runinfo

matplotlib.use("Agg")

DEFAULT_LABELS = (
    "prices are fitted from recorded costs, not list prices",
    "latency is simulated, never measured, wherever it is shown",
    "nothing here is assumed",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _plain(value: Any) -> Any:
    """JSON-ready copy of ``value``: numpy scalars become Python numbers, NaN and inf null."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _cell(value: Any) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame) -> str:
    """``df`` as a GitHub Markdown table; floats use 6 significant digits."""
    header = "| " + " | ".join(_cell(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def write_results(
    out_dir: Path,
    experiment: str,
    results: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Figure],
    seeds: Mapping[str, Any],
    command: Sequence[str],
    policy_version: str | None,
    started_at: datetime,
    measurement: Mapping[str, Any] | None = None,
) -> Path:
    """Write the result files into ``out_dir`` (named by ``runinfo.new_run_id``) and return it.

    ``results.json`` is ``results``. ``report.md`` has the title, the run id, the
    Labels line (``results["labels"]`` when given, else ``DEFAULT_LABELS``), every table
    and every figure. Each figure is saved as ``<name>.png``. ``measurement``, when given, is
    how the run was measured (for example the protocol of a live collection) and goes under
    ``measurement`` in ``run.json``.
    """
    for name in figures:
        if not _SAFE_NAME.match(name):
            raise ValueError(f"figure name {name!r} is not a plain file stem")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _plain(results)
    _write_text(out_dir / "results.json", json.dumps(payload, indent=2) + "\n")

    labels = results.get("labels") or list(DEFAULT_LABELS)
    lines = [f"# {experiment}", "", f"Run `{out_dir.name}`.", ""]
    lines += [f"**Labels:** {'; '.join(str(label) for label in labels)}.", ""]
    for name, table in tables.items():
        lines += [f"## {name}", "", markdown_table(table), ""]
    for name, figure in figures.items():
        figure.savefig(out_dir / f"{name}.png", dpi=120)
        lines += [f"![{name}]({name}.png)", ""]
    _write_text(out_dir / "report.md", "\n".join(lines))

    runinfo.write_run_json(
        out_dir, experiment, seeds, command, policy_version, started_at, extra=measurement
    )
    return out_dir


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
