"""Feature spec v1: unit behavior and golden parity with the Go implementation.

The Go side (internal/features) reads the same testdata/features files, so a
change that breaks parity fails in both test suites.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from routepilot_train import features

TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "features"
TOLERANCE = 1e-12
CASES = features.load_cases(TESTDATA / "cases.json")
CASES_BY_NAME = {case.name: case for case in CASES}


def _extract_dense(text: str, role: str = "user") -> np.ndarray:
    return features.extract([(role, text)])[2]


def test_fnv_known_values() -> None:
    # Published FNV-1a 64 test vectors.
    assert features.fnv1a64(b"") == 0xCBF29CE484222325
    assert features.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert features.fnv1a64(b"foobar") == 0x85944171F73967E8


def test_golden_matches() -> None:
    golden = json.loads((TESTDATA / "golden.json").read_text(encoding="utf-8"))
    assert golden["spec_version"] == features.SPEC_VERSION
    assert [g["name"] for g in golden["cases"]] == [c.name for c in CASES]
    for case, want in zip(CASES, golden["cases"], strict=True):
        assert want["dim"] == case.dim, case.name
        idx, val, dense = features.extract(case.messages, case.dim)
        assert idx.tolist() == want["idx"], case.name
        np.testing.assert_allclose(val, want["val"], rtol=0, atol=TOLERANCE, err_msg=case.name)
        np.testing.assert_allclose(dense, want["dense"], rtol=0, atol=TOLERANCE, err_msg=case.name)


def test_testdata_files_are_lf_only() -> None:
    for name in ("cases.json", "golden.json"):
        assert b"\r" not in (TESTDATA / name).read_bytes(), name


def test_cases_cover_the_required_inputs() -> None:
    assert len(CASES) >= 20
    assert any(c.dim == 16384 for c in CASES)
    assert any(len(c.messages) == 0 for c in CASES)
    assert any(len(c.messages) == 3 for c in CASES)
    assert any(role == "system" for c in CASES for role, _ in c.messages)


def test_extract_output_types_and_shapes() -> None:
    idx, val, dense = features.extract([("user", "hello world")])
    assert idx.dtype == np.int32
    assert val.dtype == np.float64
    assert dense.dtype == np.float64
    assert idx.shape == val.shape
    assert dense.shape == (features.NUM_DENSE,)


def test_idx_ascending_unique_and_val_unit_norm() -> None:
    for case in CASES:
        idx, val, _ = features.extract(case.messages, case.dim)
        assert np.all(np.diff(idx) > 0), case.name
        assert np.all(idx >= 0) and np.all(idx < case.dim), case.name
        if len(idx) > 0:
            assert math.isclose(float(np.sum(val * val)), 1.0, abs_tol=1e-12), case.name


def test_empty_inputs_have_no_hashed_features() -> None:
    for messages in ([], [("user", "")], [("user", " \n\t ")]):
        idx, val, _ = features.extract(messages)
        assert len(idx) == 0 and len(val) == 0
    assert features.extract([])[2].tolist() == [0.0] * features.NUM_DENSE
    # An empty text is still one message with zero lines.
    assert _extract_dense("").tolist() == [0.0, math.log1p(1)] + [0.0] * 10


def test_tokenize_splits_ascii_runs_and_keeps_non_ascii_runs_raw() -> None:
    assert features.tokenize("Café 42-x é, ÀB".encode()) == [
        b"caf",
        b"\xc3\xa9",
        b"42",
        b"x",
        b"\xc3\xa9",
        b"\xc3\x80",
        b"b",
    ]
    assert features.tokenize(b"") == []
    assert features.tokenize(b"  ... \n") == []


def test_repeated_token_uses_log1p_count() -> None:
    idx, val, _ = features.extract([("user", "go go go")])
    # Unigram go x3, bigram "go go" x2 (distinct buckets with overwhelming probability).
    assert len(idx) == 2
    expected = np.array([math.log1p(3), math.log1p(2)])
    expected = expected / math.sqrt(float(np.sum(expected**2)))
    assert sorted(np.abs(val).tolist()) == pytest.approx(sorted(expected.tolist()), abs=1e-15)


def test_case_is_folded_for_ascii_only() -> None:
    a = features.extract([("user", "Hello WORLD")])
    b = features.extract([("user", "hello world")])
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()
    c = features.extract([("user", "É")])
    d = features.extract([("user", "é")])
    assert c[0].tolist() != d[0].tolist()


def test_hashed_part_ignores_bytes_past_prefix() -> None:
    inside = "x " * (features.PREFIX_BYTES // 2)
    long = inside + "tail words beyond the prefix " * 50
    a_idx, a_val, a_dense = features.extract([("user", inside)])
    b_idx, b_val, b_dense = features.extract([("user", long)])
    assert a_idx.tolist() == b_idx.tolist()
    assert a_val.tolist() == b_val.tolist()
    assert b_dense[0] > a_dense[0]  # the length feature still sees the whole text


def test_long_case_cuts_mid_rune_and_ignores_tail_markers() -> None:
    case = CASES_BY_NAME["long_cut_mid_rune"]
    ((_, text),) = case.messages
    raw = text.encode("utf-8")
    assert len(raw) == 40000
    assert 0x80 <= raw[features.PREFIX_BYTES] <= 0xBF  # the cut lands inside a rune
    dense = features.extract(case.messages, case.dim)[2]
    assert dense[0] == math.log1p(40000)
    assert dense[8] == 0.0  # "$$" and "\frac" sit past the prefix
    assert dense[11] == 0.0  # so does "?"
    assert dense[2] == math.log1p(2)  # lines are counted over the full text


def test_hash_collisions_that_cancel_are_dropped() -> None:
    case = CASES_BY_NAME["small_dim_collisions"]
    ((_, text),) = case.messages
    counts = features.bucket_counts(text.encode(), case.dim)
    cancelled = [i for i, c in counts.items() if c == 0]
    assert cancelled, "the case must contain a bucket whose signed counts cancel"
    idx = features.extract(case.messages, case.dim)[0].tolist()
    assert idx == sorted(i for i, c in counts.items() if c != 0)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("A. x", 1.0),
        ("(b) x", 1.0),
        ("C: x", 1.0),
        ("  \tE) x", 1.0),
        ("(e) x", 1.0),
        ("intro\nB) yes", 1.0),
        ("F. x", 0.0),
        ("A.x", 0.0),
        ("A.\nB.", 0.0),
        ("((a) x", 0.0),
        ("xA. y", 0.0),
        ("the answer is A. maybe", 0.0),
        ("a]x", 0.0),
    ],
)
def test_choices_flag(text: str, want: float) -> None:
    assert _extract_dense(text)[9] == want


@pytest.mark.parametrize("marker", ["\\frac", "\\sum", "$$", "\\(", "\\["])
def test_math_marker_flag(marker: str) -> None:
    assert _extract_dense(f"a {marker} b")[8] == 1.0


def test_math_marker_flag_off_without_marker() -> None:
    assert _extract_dense("frac sum $ ( [ \\ ")[8] == 0.0


def test_system_flag_follows_role() -> None:
    assert features.extract([("system", "x")])[2][10] == 1.0
    assert features.extract([("user", "x"), ("system", "y")])[2][10] == 1.0
    assert features.extract([("user", "x"), ("assistant", "y")])[2][10] == 0.0


def test_dense_counts_and_ratios() -> None:
    dense = _extract_dense("a1B!é")  # 6 bytes: a 1 B ! 0xC3 0xA9
    assert dense[0] == math.log1p(6)
    assert dense[1] == math.log1p(1)
    assert dense[2] == math.log1p(1)
    assert dense[3] == 1 / 6
    assert dense[4] == 1 / 6
    assert dense[5] == 2 / 6
    assert dense[6] == 1 / 6


def test_lines_and_fences_and_questions() -> None:
    assert _extract_dense("a\nb\nc")[2] == math.log1p(3)
    assert _extract_dense("\n")[2] == math.log1p(2)
    assert features.extract([])[2][2] == 0.0
    assert _extract_dense("```a```")[7] == math.log1p(2)
    assert _extract_dense("`````")[7] == math.log1p(1)  # non-overlapping count
    assert _extract_dense("why? how??")[11] == math.log1p(3)


def test_all_32_punctuation_bytes_are_counted() -> None:
    punctuation = "".join(chr(b) for b in range(0x21, 0x7F) if not chr(b).isalnum())
    assert len(punctuation) == 32
    assert _extract_dense(punctuation)[6] == 1.0


def test_messages_join_with_newline_and_count() -> None:
    dense = features.extract([("user", "ab"), ("assistant", "cd")])[2]
    assert dense[0] == math.log1p(5)  # "ab\ncd"
    assert dense[1] == math.log1p(2)
    assert dense[2] == math.log1p(2)
    one = features.extract([("user", "ab\ncd")])
    two = features.extract([("user", "ab"), ("assistant", "cd")])
    assert one[0].tolist() == two[0].tolist()  # hashed part only sees the joined bytes


def test_extract_texts_matches_extract() -> None:
    texts = [
        case.messages[0][1]
        for case in CASES
        if len(case.messages) == 1
        and case.messages[0][0] == "user"
        and case.dim == features.DEFAULT_DIM
    ]
    assert len(texts) >= 15
    matrix, dense = features.extract_texts(texts)
    assert scipy.sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (len(texts), features.DEFAULT_DIM)
    assert dense.shape == (len(texts), features.NUM_DENSE)
    assert matrix.has_sorted_indices
    for i, text in enumerate(texts):
        idx, val, want_dense = features.extract([("user", text)])
        row = matrix.getrow(i)
        assert row.indices.tolist() == idx.tolist()
        assert row.data.tolist() == val.tolist()
        assert dense[i].tolist() == want_dense.tolist()


def test_extract_texts_empty_list_and_custom_dim() -> None:
    matrix, dense = features.extract_texts([], dim=1024)
    assert matrix.shape == (0, 1024)
    assert dense.shape == (0, features.NUM_DENSE)
    matrix, _ = features.extract_texts(["hello world", ""], dim=1024)
    assert matrix.shape == (2, 1024)
    assert matrix.getrow(1).nnz == 0
    want_idx = features.extract([("user", "hello world")], 1024)[0]
    assert matrix.getrow(0).indices.tolist() == want_idx.tolist()


def test_extract_rows_matches_extract_for_any_roles() -> None:
    rows = [
        [("user", "pick the letter\nA) one\nB) two")],
        [("system", "Reply with only the letter."), ("user", "pick the letter\nA) one\nB) two")],
        [("system", "s"), ("user", "u"), ("assistant", "a")],
        [("user", "")],
    ]
    matrix, dense = features.extract_rows(rows, dim=1024)
    assert scipy.sparse.isspmatrix_csr(matrix) and matrix.has_sorted_indices
    assert matrix.shape == (len(rows), 1024) and dense.shape == (len(rows), features.NUM_DENSE)
    for i, messages in enumerate(rows):
        idx, val, want_dense = features.extract(messages, 1024)
        assert matrix.getrow(i).indices.tolist() == idx.tolist()
        assert matrix.getrow(i).data.tolist() == val.tolist()
        assert dense[i].tolist() == want_dense.tolist()
    # the roles matter: a system message changes the dense block and the joined text
    assert dense[0].tolist() != dense[1].tolist()
    assert dense[1][10] == 1.0 and dense[0][10] == 0.0
    # extract_texts is extract_rows over one user message each
    texts_matrix, texts_dense = features.extract_texts(["x y", "z"], dim=1024)
    rows_matrix, rows_dense = features.extract_rows([[("user", "x y")], [("user", "z")]], dim=1024)
    assert (texts_matrix != rows_matrix).nnz == 0
    assert texts_dense.tolist() == rows_dense.tolist()
    empty_matrix, empty_dense = features.extract_rows([], dim=1024)
    assert empty_matrix.shape == (0, 1024) and empty_dense.shape == (0, features.NUM_DENSE)


@pytest.mark.parametrize("dim", [0, -1, 2**31])
def test_invalid_dim_is_rejected(dim: int) -> None:
    with pytest.raises(ValueError, match="dim"):
        features.extract([("user", "x")], dim)
    with pytest.raises(ValueError, match="dim"):
        features.extract_texts(["x"], dim)
    with pytest.raises(ValueError, match="dim"):
        features.extract_rows([[("user", "x")]], dim)


def test_write_golden_cli_reproduces_the_committed_file(tmp_path: Path) -> None:
    (tmp_path / "cases.json").write_bytes((TESTDATA / "cases.json").read_bytes())
    assert features.main(["--write-golden", str(tmp_path)]) == 0
    written_bytes = (tmp_path / "golden.json").read_bytes()
    assert written_bytes.endswith(b"\n")
    assert b"\r" not in written_bytes
    written = json.loads(written_bytes)
    committed = json.loads((TESTDATA / "golden.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in written["cases"]] == [c["name"] for c in committed["cases"]]
    for got, want in zip(written["cases"], committed["cases"], strict=True):
        assert got["idx"] == want["idx"], got["name"]
        np.testing.assert_allclose(got["val"], want["val"], rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(got["dense"], want["dense"], rtol=0, atol=TOLERANCE)
