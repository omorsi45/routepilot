"""Grading of multiple-choice answers: option letters offered, extracted and compared.

Every text here is invented. No dataset prompt or model output is used.
"""

import pytest

from routepilot_train import grading

ABCD = "ABCD"
A_TO_J = "ABCDEFGHIJ"


@pytest.mark.parametrize(
    ("text", "letters", "want"),
    [
        # an answer cue wins over any other letter in the text
        ("The answer is B.", ABCD, "B"),
        ("Answer: C", ABCD, "C"),
        ("ANSWER: D", ABCD, "D"),
        ("A is wrong because of X. B is wrong too. The answer is D.", ABCD, "D"),
        ("I think it is A. On reflection B. Answer: (C)", ABCD, "C"),
        ("**Answer: B**", ABCD, "B"),
        ("So the final answer is: **D**", ABCD, "D"),
        ("The correct answer is option C because of Y.", ABCD, "C"),
        ("Answer: \\boxed{B}", ABCD, "B"),
        ("A common mistake is Z, so the answer is C.", ABCD, "C"),
        # the last cue is the one that counts
        ("Answer: A. But on reflection the answer is C", ABCD, "C"),
        # without a cue, the last standalone option letter
        ("Option A seems plausible but B is the best.", ABCD, "B"),
        ("I choose (D)", ABCD, "D"),
        ("C", ABCD, "C"),
        # a cue with no usable letter after it falls back to the last standalone letter
        ("I choose B. Answer:", ABCD, "B"),
        ("The answer is unclear, maybe A or C.", ABCD, "C"),
        # the letters the prompt offers decide what counts
        ("The answer is J.", A_TO_J, "J"),
        ("The answer is J.", ABCD, None),
        ("The answer is E.", ABCD, None),
        # nothing usable
        ("", ABCD, None),
        ("   \n  ", ABCD, None),
        ("The answer is", ABCD, None),
        ("BAD CAB DEAD", ABCD, None),
        ("the answer is b", ABCD, None),
        ("I do not know.", ABCD, None),
    ],
)
def test_extract_choice(text: str, letters: str, want: str | None) -> None:
    assert grading.extract_choice(text, letters) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("The answer is B.", ("B", "cue")),
        ("Option A seems plausible but B is the best.", ("B", "last_letter")),
        ("I choose B. Answer:", ("B", "last_letter")),
        ("I do not know.", (None, "none")),
        ("", (None, "none")),
    ],
)
def test_extract_says_how_the_letter_was_found(text: str, want: tuple[str | None, str]) -> None:
    got = grading.extract(text, ABCD)
    assert (got.letter, got.how) == want


def test_extract_choice_takes_letters_as_any_sequence() -> None:
    assert grading.extract_choice("Answer: B", list(ABCD)) == "B"
    assert grading.extract_choice("Answer: B", tuple(ABCD)) == "B"


@pytest.mark.parametrize("letters", ["", "abcd", "AB1", "AAB", ["AB", "C"]])
def test_extract_choice_rejects_bad_letters(letters: object) -> None:
    with pytest.raises(ValueError, match="letters"):
        grading.extract_choice("Answer: A", letters)  # type: ignore[arg-type]


def test_extract_choice_is_linear_on_long_text() -> None:
    # a long run of filler must not blow up the cue pattern
    text = "answer " * 20000 + "is " * 20000 + "x"
    assert grading.extract_choice(text, ABCD) is None


@pytest.mark.parametrize(
    ("prompt", "want"),
    [
        ("Q?\nA) one\nB) two\nC) three\nD) four\nThink, then answer.", "ABCD"),
        ("Q?\nA. one\nB. two\nC. three\nD. four\nE. five\nF. six", "ABCDEF"),
        ("Q?\n(A) one\n(B) two\n(C) three", "ABC"),
        # a numbered list before the options must not leak in: the run after the last A counts
        ("Steps:\nI. first\nA. one\nB. two\nC. three\nD. four", "ABCD"),
        ("A. note\nA. one\nB. two\nC. three\nD. four", "ABCD"),
        # a gap ends the run
        ("A. one\nB. two\nD. four", "AB"),
        ("no options here", ""),
        ("A. only one option", ""),
    ],
)
def test_offered_letters(prompt: str, want: str) -> None:
    assert grading.offered_letters(prompt) == want


@pytest.mark.parametrize(
    ("truth", "text", "want"),
    [
        ("B", "The answer is B.", True),
        ("B", "The answer is C.", False),
        (" b ", "The answer is B.", True),
        ("B", "no idea", False),
        ("B", "", False),
    ],
)
def test_is_correct(truth: str, text: str, want: bool) -> None:
    assert grading.is_correct(text, truth, ABCD) is want


def test_is_correct_rejects_a_truth_outside_the_letters() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        grading.is_correct("The answer is A", "E", ABCD)
