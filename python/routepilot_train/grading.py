"""Grade multiple-choice answers: find the offered option letters, pull the letter a model
chose out of its free text, and compare it with the ground truth.

The rule for the chosen letter: the letter that follows the last "answer" cue (``answer``,
``Answer:``, ``the answer is``, ``final answer: (B)``, ``\\boxed{B}``), when there is one;
otherwise the last standalone capital letter among the offered letters. A standalone letter
is not part of a longer word. Lowercase letters are not read as choices.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

How = Literal["cue", "last_letter", "none"]

_EDGE_BEFORE = r"(?<![A-Za-z0-9_])"
_EDGE_AFTER = r"(?![A-Za-z0-9_])"
# Characters that may sit between an answer cue and the letter: spaces, colons, markdown
# emphasis, brackets and dashes. A word of the connective list may be followed by more of
# them. The pieces never overlap (a connective starts with a letter, a separator does not),
# so the pattern cannot backtrack badly on long input.
_SEP = r"[\s:*\-=(\[{>]*"
_BOX = r"(?:\\(?:boxed|text|textbf|mathbf)\s*\{)?"
_CONNECTIVE = r"(?:(?i:is|are|was|be|option|choice|letter|correct)\b" + _SEP + r")*"
_OFFER_LINE = re.compile(r"(?m)^[ \t]*\(?([A-J])[.):][ \t]")


@dataclass(frozen=True)
class Extraction:
    """The chosen letter (``None`` when there is none) and how it was found."""

    letter: str | None
    how: How


def _check_letters(letters: Sequence[str]) -> str:
    singles = all(isinstance(x, str) and len(x) == 1 for x in letters)
    joined = "".join(letters) if singles else ""
    if not joined or not all("A" <= c <= "Z" for c in joined) or len(set(joined)) != len(joined):
        raise ValueError(f"letters must be distinct capital letters, got {letters!r}")
    return joined


def extract(text: str, letters: Sequence[str]) -> Extraction:
    """The letter ``text`` picks among ``letters`` and whether a cue or the last letter found it.

    Raises ``ValueError`` when ``letters`` is empty, repeats a letter or holds anything but
    single capital letters.
    """
    allowed = _check_letters(letters)
    klass = "[" + "".join(re.escape(c) for c in allowed) + "]"
    cue = re.compile(
        r"(?i:answer)"
        + _SEP
        + _CONNECTIVE
        + _BOX
        + _SEP
        + _EDGE_BEFORE
        + f"({klass})"
        + _EDGE_AFTER
    )
    cues = cue.findall(text)
    if cues:
        return Extraction(cues[-1], "cue")
    standalone = re.compile(_EDGE_BEFORE + "(" + klass + ")" + _EDGE_AFTER)
    singles = standalone.findall(text)
    if singles:
        return Extraction(singles[-1], "last_letter")
    return Extraction(None, "none")


def extract_choice(text: str, letters: Sequence[str]) -> str | None:
    """The letter ``text`` picks among ``letters``, or ``None`` (see ``extract``)."""
    return extract(text, letters).letter


def offered_letters(prompt: str) -> str:
    """The option letters a multiple-choice prompt offers, such as ``ABCD``.

    Option lines start with the letter and a ``.``, ``)`` or ``:`` (optionally in
    parentheses). The offered run is the consecutive letters from the last ``A`` that starts
    an option line, so a numbered list before the options does not leak in. It is empty
    when fewer than two options are found.
    """
    starts = _OFFER_LINE.findall(prompt)
    if "A" not in starts:
        return ""
    run = starts[len(starts) - 1 - starts[::-1].index("A") :]
    out = ""
    for letter in run:
        if letter != chr(ord("A") + len(out)):
            break
        out += letter
    return out if len(out) >= 2 else ""


def is_correct(text: str, truth: str, letters: Sequence[str]) -> bool:
    """Whether the letter ``text`` picks equals ``truth`` (spaces and case in ``truth`` ignored).

    Raises ``ValueError`` when ``truth`` is not one of ``letters``.
    """
    allowed = _check_letters(letters)
    wanted = truth.strip().upper()
    if wanted not in allowed:
        raise ValueError(f"ground truth {truth!r} is not one of the letters {allowed!r}")
    return extract_choice(text, letters) == wanted
