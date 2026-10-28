"""Read the gateway's decision log for off-policy evaluation.

The Go gateway appends one JSON line per routing decision, and one per piece of client
feedback, to an append-only JSONL file that it rotates by size (see ``internal/decisionlog``).
A decision line carries the candidates the policy scored, the model it chose and the
probability it had of choosing it, and what came of the request. It holds no prompt text, only
the first 16 hex digits of the prompt's SHA-256 and its size. A feedback line says whether a
decision's answer worked out.

``load_decisions`` returns the decisions, with the tenant-matched feedback joined on.
"""

import glob
import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

_STR = ["decision_id", "tenant", "policy_version", "policy_role", "prompt_sha256_16", "chosen"]
_INT = [
    "prompt_bytes",
    "status",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "cost_actual_picousd",
]
_FLOAT = ["ood_score", "propensity", "epsilon"]
_BOOL = ["ood", "explored", "usage_estimated"]
_LIST = ["required", "candidates", "relaxed", "attempts"]

# The columns of the frame ``load_decisions`` returns, in order. The first block is the decision
# record as the gateway writes it; a shadow decision has no outcome, so ``status`` to
# ``attempts`` are missing for it. The last two are the joined feedback.
COLUMNS = [
    "decision_id",
    "time",
    "tenant",
    "policy_version",
    "policy_role",
    "prompt_sha256_16",
    "prompt_bytes",
    "required",
    "ood_score",
    "ood",
    "candidates",
    "chosen",
    "propensity",
    "epsilon",
    "explored",
    "relaxed",
    "status",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "cost_actual_picousd",
    "usage_estimated",
    "attempts",
    "feedback_success",
    "feedback_time",
]


def _log_files(path_glob: str | Path) -> list[Path]:
    """The files a glob (or a directory, meaning its ``*.jsonl`` files) names, sorted by name.

    Rotated files are ``<base>.<UTC yyyymmddTHHMMSSZ>.jsonl``, which sort by time and before
    the active ``<base>.jsonl``.
    """
    path = Path(path_glob)
    if path.is_dir():
        found = list(path.glob("*.jsonl"))
    else:
        found = [Path(match) for match in glob.glob(str(path_glob))]
    files = sorted(f for f in found if f.is_file())
    if not files:
        raise FileNotFoundError(f"no decision log files match {str(path_glob)!r}")
    return files


def _read_records(files: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split the lines of ``files`` into decisions and feedback, and count the unreadable ones.

    A line that is not a JSON object is skipped: the last line of a file can be torn by a
    crash. A record of any other type is ignored, so a newer gateway can add types.
    """
    decisions: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    malformed = 0
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    malformed += 1
                elif record.get("type") == "decision":
                    decisions.append(record)
                elif record.get("type") == "feedback":
                    feedback.append(record)
    return decisions, feedback, malformed


def _decision_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per decision. Integers stay exact (nullable Int64): picodollars exceed 2**53."""
    columns: dict[str, Any] = {}
    columns["decision_id"] = pd.Series([r.get("decision_id") for r in records], dtype="str")
    columns["time"] = pd.to_datetime([r.get("time") for r in records], utc=True, format="ISO8601")
    for name in _STR[1:]:
        columns[name] = pd.Series([r.get(name) for r in records], dtype="str")
    for name in _INT:
        columns[name] = pd.array([r.get(name) for r in records], dtype="Int64")
    for name in _FLOAT:
        columns[name] = pd.array([r.get(name) for r in records], dtype="Float64").astype("float64")
    for name in _BOOL:
        columns[name] = pd.array([r.get(name) for r in records], dtype="boolean")
    for name in _LIST:
        columns[name] = pd.Series([r.get(name) for r in records], dtype="object")
    frame = pd.DataFrame(columns)
    return frame[[c for c in COLUMNS if c in frame.columns]]


def _feedback_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """The latest feedback per (decision, tenant); of two at the same time the later line wins."""
    frame = pd.DataFrame(
        {
            "_join_id": pd.Series([r.get("decision_id") for r in records], dtype="str"),
            "tenant": pd.Series([r.get("tenant") for r in records], dtype="str"),
            "feedback_success": pd.array([r.get("success") for r in records], dtype="boolean"),
            "feedback_time": pd.to_datetime(
                [r.get("time") for r in records], utc=True, format="ISO8601"
            ),
        }
    )
    frame = frame.dropna(subset=["_join_id", "tenant", "feedback_success"])  # unusable lines
    frame = frame.sort_values("feedback_time", kind="stable")
    return frame.drop_duplicates(["_join_id", "tenant"], keep="last")


def load_decisions(path_glob: str | Path, *, include_shadow: bool = False) -> pd.DataFrame:
    """Load decision records, joined with the feedback their own tenant gave.

    ``path_glob`` is a glob such as ``logs/decisions*.jsonl`` (which reads the rotated files
    and the active one), or a directory, meaning every ``*.jsonl`` in it. The rows are in time
    order. Feedback joins on the decision ID and the tenant together: a verdict from another
    tenant is ignored, and when a tenant sent several the latest counts. The joined columns are
    ``feedback_success`` (a nullable boolean, missing while nobody has answered) and
    ``feedback_time``. A feedback line for a decision that is not in the log is ignored.

    A shadow decision was never acted on, so it has no outcome and no feedback, and it shares
    its ``decision_id`` with the decision that was. It is left out unless ``include_shadow``
    is true, which keeps ``decision_id`` unique in the result.

    Raises FileNotFoundError when nothing matches. Lines that are not JSON objects are skipped
    with a warning.
    """
    files = _log_files(path_glob)
    decisions, feedback, malformed = _read_records(files)
    if malformed:
        warnings.warn(
            f"skipped {malformed} malformed line(s) in the decision log",
            UserWarning,
            stacklevel=2,
        )
    if not include_shadow:
        decisions = [r for r in decisions if r.get("policy_role") != "shadow"]

    frame = _decision_frame(decisions)
    # A shadow row gets no join key, and a missing key matches no feedback.
    frame["_join_id"] = frame["decision_id"].where(frame["policy_role"] != "shadow")
    joined = frame.merge(_feedback_frame(feedback), on=["_join_id", "tenant"], how="left")
    joined = joined.drop(columns="_join_id").sort_values("time", kind="stable")
    return joined[COLUMNS].reset_index(drop=True)
