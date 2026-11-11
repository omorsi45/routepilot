"""The repository text carries no internal planning references.

Ruling IDs, plan task numbers and design-document section numbers point at documents that are
not in the repository, so a reader meets references that resolve to nothing. This walks every
tracked text file and fails on them. Run output under ``bench/results/<experiment>/<run>/`` is
recorded script output and is not edited, so it is skipped (the generated docs scrub it).
"""

import re
import subprocess
from pathlib import Path

import pytest

from routepilot_train.runinfo import repo_root

# What each pattern means, so a failure says what to remove.
FORBIDDEN = {
    "a ruling ID": re.compile(r"\b[MV]\d+-R\d+"),
    "a ruling reference": re.compile(r"\b[Rr]ulings?\s+[MV]\d|\bThe ruling is\b"),
    "a plan task number": re.compile(r"\b[Tt]asks? \d"),
    "a design spec section": re.compile(r"\bspec \d+(\.\d+)*\b|\bdesign spec\b"),
    "a spec phrase": re.compile(
        r"\bspec (?:operating point|rule|lambda)\b|operating point \(spec\)"
    ),
    "the plan": re.compile(r"\b[Tt]he plan\b|\bplan Task\b|\bin this plan\b"),
    "process wording": re.compile(r"\bbriefed\b|\b[Tt]he brief(?:'s)?\b"),
    "a review item label": re.compile(r"\bN3\b"),
}

# The scrubber that removes these references from generated docs names them in its patterns and
# its tests, and so cannot pass its own guard.
ALLOWED_FILES = {
    "bench/scripts/render_docs.py",
    "python/tests/test_render_docs.py",
    "python/tests/test_repository_text.py",
}
RUN_OUTPUT = re.compile(r"^bench/results/[^/]+/[^/]+/")


def tracked_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as err:
        pytest.skip(f"git is not available here: {err}")
    return [name for name in proc.stdout.decode("utf-8").split("\0") if name]


def references_in(text: str) -> list[tuple[int, str, str]]:
    """Every (line number, what it is, the line) of an internal reference in ``text``."""
    found = []
    for number, line in enumerate(text.split("\n"), start=1):
        for what, pattern in FORBIDDEN.items():
            if pattern.search(line):
                found.append((number, what, line.strip()[:120]))
    return found


def test_no_tracked_text_file_carries_an_internal_reference() -> None:
    root = repo_root()
    problems = []
    checked = 0
    for name in tracked_files(root):
        if name in ALLOWED_FILES or RUN_OUTPUT.match(name):
            continue
        path = root / name
        if not path.is_file():
            continue  # staged for removal or renamed in the working tree
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue  # weights, images
        checked += 1
        problems += [f"{name}:{n}: {what}: {line}" for n, what, line in references_in(text)]
    assert checked > 300  # the walk saw the repository, not an empty list
    assert problems == [], "internal references:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    "line",
    [
        "// pins Ruling V1-R24: the bound a model is held to",
        "# Prices are an ASSUMPTION (M2-R11), not the price of anything",
        "the Task 13 pipeline enforces the split",
        "(spec 7.1)",
        "the spec operating point",
        "the plan's 0.8 relaxes nearly every decision",
        "the briefed 64",
        "N3 (a burst below the default)",
        "section 11 of the design spec",
    ],
)
def test_the_guard_catches_each_kind_of_reference(line: str) -> None:
    assert references_in(line)


@pytest.mark.parametrize(
    "line",
    [
        "feature spec v1 and the spec version in the manifest",
        "the attempt plan walks the candidates",
        "a task force of 3 tasks",
        "be brief",
        "Rules are fixed before any model is trained",
    ],
)
def test_the_guard_leaves_ordinary_wording_alone(line: str) -> None:
    assert references_in(line) == []
