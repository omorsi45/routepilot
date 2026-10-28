"""The policy artifact: a content-addressed directory the Go router loads and verifies.

Layout: ``<root>/<name>/<version>/manifest.json`` and ``weights.bin``.

``weights.bin`` is the concatenation of little-endian arrays. The manifest refers to each
as ``{"offset": bytes, "len": elements, "dtype": "f32" | "i32"}``. Every weight is ``f32``
and the OOD centroid indices are ``i32`` (Ruling V1-R11). Arrays are laid out with no gaps,
in manifest order: per model its quality weights then its length weights, then per centroid
its indices then its values.

``manifest.json`` is written with ``json.dumps(obj, sort_keys=True, indent=1,
ensure_ascii=True) + "\\n"``. ``version`` is the first 12 hex digits of SHA-256 over the
manifest bytes written with ``"version": ""`` followed by the ``weights.bin`` bytes; the
manifest is then rewritten with the version filled in and the directory is named after it.
Go recomputes the hash from the raw manifest bytes (one ``"version": "<v>"`` replaced by
``"version": ""``), so no JSON round trip can change what is hashed.

Every artifact carries golden predictions for ``GOLDEN_PROMPTS``, computed by
``predict_bundle`` on the float32-rounded weights that Go will read. Go repeats them at load
and refuses the artifact on a mismatch.

Scores accumulate in float64, the same way on both sides: the hashed terms in ascending
index order, then the 12 standardized dense terms, then the bias; weights are float32
widened to float64.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

from routepilot_train import features
from routepilot_train.train.calibrate import apply_calibrator
from routepilot_train.train.lcb import lookup_lcb
from routepilot_train.train.length import QUANTILES
from routepilot_train.train.ood import ood_score

FORMAT = "routepilot-policy-artifact"
FORMAT_VERSION = 1
FEATURE_SPEC_VERSION = features.SPEC_VERSION
VERSION_LENGTH = 12
RESIDUAL_KEYS = tuple(str(q) for q in QUANTILES)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DTYPES = {"f32": "<f4", "i32": "<i4"}

_PARAGRAPH_SENTENCES = (
    "The harbor town of Veldmoor keeps a small archive of weather diaries written by its ferry "
    "captains over many generations.",
    "Each captain noted the wind, the tide and the colour of the evening sky, and most entries "
    "were no longer than a single line.",
    "When the archivist IlsabetOrne began to copy them into ledgers, she found that the "
    "captains had quietly agreed on a shorthand for fog.",
    "A circle meant mist that lifted by noon, two circles meant a grey day, and a circle "
    "crossed by a bar meant that the lamps stayed lit until supper.",
    "The shorthand spread to the pilots, then to the net menders, and finally to the children "
    "who drew it in chalk on the quay stones.",
    "By the time the ledgers were bound, nobody in Veldmoor remembered who had started the "
    "habit, only that it made the mornings easier to read.",
    "Visitors sometimes asked whether the diaries were accurate, and Ilsabet would answer that "
    "they were kind rather than exact.",
    "A diary that records a bad morning gently is worth more to a sailor than one that records "
    "it precisely and says nothing about what to do next.",
)

# Eight invented prompts (never dataset text): arithmetic, a code fence, LaTeX, multiple
# choice, a 3 KB paragraph, a system plus user pair, non-ASCII text and a single word.
GOLDEN_PROMPTS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("user", "What is 17 times 23?"),),
    (
        (
            "user",
            "Explain what this Python function returns for n = 5:\n"
            "```python\ndef f(n):\n    return sum(i * i for i in range(n))\n```\n"
            "Is the result a prime number?",
        ),
    ),
    (("user", r"Evaluate \frac{1}{x^2 + 1} integrated from 0 to 1 and simplify the result."),),
    (
        (
            "user",
            "Which planet in the invented Orrery system has the longest year?\n"
            "A. Aldoria\nB. Brenth\nC. Corvane\nD. Dessel\nAnswer with a single letter.",
        ),
    ),
    (("user", " ".join(_PARAGRAPH_SENTENCES * 3)),),
    (
        ("system", "You are a careful assistant. Answer in one sentence."),
        ("user", "Why do ships float?"),
    ),
    (
        (
            "user",
            "Quelle est la capitale de l'\u00c9gypte ? \u041e\u0431\u044a\u044f\u0441\u043d\u0438"
            "\u0442\u0435, \u043f\u043e\u0447\u0435\u043c\u0443 \u043d\u0435\u0431\u043e "
            "\u0433\u043e\u043b\u0443\u0431\u043e\u0435. "
            "\u65e5\u672c\u8a9e\u306e\u6587\u7ae0\u3082\u542b\u307e\u308c\u3066\u3044\u307e"
            "\u3059\u3002 \U0001f642",
        ),
    ),
    (("user", "Photosynthesis"),),
)


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The canonical manifest bytes: sorted keys, indent 1, ASCII, trailing newline, LF."""
    try:
        text = json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False)
    except ValueError as err:
        raise ValueError(f"manifest holds a number that is not finite: {err}") from err
    return (text + "\n").encode("ascii")


def content_version(unversioned_manifest: bytes, weights: bytes) -> str:
    """The version hash of manifest bytes written with ``"version": ""`` plus the weights."""
    digest = hashlib.sha256(unversioned_manifest + weights).hexdigest()
    return digest[:VERSION_LENGTH]


def _plain(value: Any) -> Any:
    """A JSON-ready copy of ``value``: numpy scalars and arrays become Python values."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def assemble_bundle(
    *,
    dim: int,
    dense_mean: np.ndarray,
    dense_std: np.ndarray,
    models: Sequence[Mapping[str, Any]],
    ood: Mapping[str, Any],
    training: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Turn the outputs of the train modules into the ``(bundle, weights)`` of ``export_artifact``.

    Each entry of ``models`` is ``{"id", "quality", "calibrator", "lcb", "length"}`` holding the
    return values of ``fit_quality``, ``fit_calibrator``, ``fit_lcb`` and ``fit_length``.
    ``ood`` is the return value of ``fit_ood``. ``training`` gives ``pool``,
    ``dataset_sha256``, ``split_salt``, ``git_sha`` and optionally ``created_at`` (now,
    UTC, when absent); the splits and the per-model ``cv`` choices are filled in here.

    The bundle holds everything except the arrays; the arrays go in ``weights`` under the
    names the bundle refers to.
    """
    ids = [str(m["id"]) for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError(f"model ids must be unique, got {ids}")
    weights: dict[str, np.ndarray] = {}
    bundle_models = []
    for m in models:
        quality_name, length_name = f"{m['id']}.quality", f"{m['id']}.length"
        weights[quality_name] = np.asarray(m["quality"]["coef"])
        weights[length_name] = np.asarray(m["length"]["coef"])
        bundle_models.append(
            {
                "id": str(m["id"]),
                "quality": {"weights": quality_name, "bias": float(m["quality"]["intercept"])},
                "calibrator": _plain(m["calibrator"]),
                "lcb": _plain(m["lcb"]),
                "length": {
                    "weights": length_name,
                    "bias": float(m["length"]["intercept"]),
                    "smear": float(m["length"]["smear"]),
                    "residual_quantiles": {
                        key: float(m["length"]["residual_quantiles"][key]) for key in RESIDUAL_KEYS
                    },
                    "in_ratio": float(m["length"]["in_ratio"]),
                },
            }
        )
    centroids = []
    for j, centroid in enumerate(ood["centroids"]):
        weights[f"ood.{j}.idx"] = np.asarray(centroid["idx"])
        weights[f"ood.{j}.val"] = np.asarray(centroid["val"])
        centroids.append({"idx": f"ood.{j}.idx", "val": f"ood.{j}.val"})
    bundle = {
        "dim": int(dim),
        "dense_mean": [float(v) for v in dense_mean],
        "dense_std": [float(v) for v in dense_std],
        "models": bundle_models,
        "ood": {"k": int(ood["k"]), "threshold": float(ood["threshold"]), "centroids": centroids},
        "training": {
            "pool": str(training["pool"]),
            "dataset_sha256": str(training["dataset_sha256"]),
            "split_salt": str(training["split_salt"]),
            "git_sha": training["git_sha"],
            "created_at": training.get("created_at")
            or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trained_on": ["train"],
            "calibrated_on": ["calib"],
            "cv": {
                str(m["id"]): {"C": m["quality"]["C"], "alpha": float(m["length"]["alpha"])}
                for m in models
            },
        },
    }
    return bundle, weights


def _slots(bundle: Mapping[str, Any]) -> list[tuple[str, str, int | None]]:
    """The weight arrays the bundle refers to, in file order: (name, dtype, required length)."""
    wide = bundle["dim"] + features.NUM_DENSE
    slots: list[tuple[str, str, int | None]] = []
    for model in bundle["models"]:
        slots.append((model["quality"]["weights"], "f32", wide))
        slots.append((model["length"]["weights"], "f32", wide))
    for centroid in bundle["ood"]["centroids"]:
        slots.append((centroid["idx"], "i32", None))
        slots.append((centroid["val"], "f32", None))
    return slots


def _pack(
    bundle: Mapping[str, Any], weights: Mapping[str, np.ndarray]
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Encode the arrays into ``weights.bin`` bytes and the manifest refs by array name."""
    slots = _slots(bundle)
    names = [name for name, _, _ in slots]
    if len(set(names)) != len(names):
        raise ValueError("a weights array is referenced more than once")
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"missing weights arrays: {missing}")
    stray = sorted(set(weights) - set(names))
    if stray:
        raise ValueError(f"unreferenced weights arrays: {stray}")
    chunks, refs, offset = [], {}, 0
    for name, dtype, required in slots:
        array = np.asarray(weights[name])
        if array.ndim != 1:
            raise ValueError(f"weights {name!r} must be 1-D, got shape {array.shape}")
        if required is not None and array.shape[0] != required:
            raise ValueError(f"weights {name!r} has length {array.shape[0]}, expected {required}")
        if dtype == "f32":
            wide = array.astype(np.float64)
            with np.errstate(over="ignore"):  # an overflow is reported below, not warned about
                packed = wide.astype(_DTYPES[dtype])
            if not (np.isfinite(wide).all() and np.isfinite(packed).all()):
                raise ValueError(f"weights {name!r} must be finite float32 values")
        else:
            if array.dtype.kind not in "iu":
                raise ValueError(f"indices {name!r} must be integers, got {array.dtype}")
            if array.size and (array.min() < 0 or array.max() > np.iinfo(np.int32).max):
                raise ValueError(f"indices {name!r} do not fit int32")
            packed = array.astype(_DTYPES[dtype])
        chunk = packed.tobytes()
        chunks.append(chunk)
        refs[name] = {"offset": offset, "len": int(packed.shape[0]), "dtype": dtype}
        offset += len(chunk)
    return b"".join(chunks), refs


def _check_calibrator(model_id: str, cal: Mapping[str, Any]) -> None:
    if cal["type"] == "isotonic":
        xs, ys = cal["x"], cal["y"]
        if len(xs) == 0 or len(xs) != len(ys):
            raise ValueError(f"{model_id}: isotonic x and y need the same non-zero length")
        if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
            raise ValueError(f"{model_id}: isotonic knots must be strictly increasing")
        if any(not 0.0 <= y <= 1.0 for y in ys):
            raise ValueError(f"{model_id}: isotonic values must be in [0, 1]")
    elif cal["type"] != "platt":
        raise ValueError(f"{model_id}: unknown calibrator type {cal['type']!r}")


def _check_lcb(model_id: str, table: Mapping[str, Any]) -> None:
    upper, lower = table["upper"], table["lower"]
    if len(upper) == 0 or len(upper) != len(lower) or len(upper) != len(table["n"]):
        raise ValueError(f"{model_id}: lcb upper, lower and n need the same non-zero length")
    if any(b < a for a, b in zip(upper, upper[1:], strict=False)) or upper[-1] != 1.0:
        raise ValueError(f"{model_id}: lcb upper must be non-decreasing and end at 1.0")


def _check_bundle(bundle: Mapping[str, Any]) -> None:
    dim = bundle["dim"]
    if not 1 <= dim <= features.MAX_DIM:
        raise ValueError(f"dim must be in [1, {features.MAX_DIM}], got {dim}")
    if len(bundle["dense_mean"]) != features.NUM_DENSE:
        raise ValueError(f"dense_mean needs {features.NUM_DENSE} values")
    if len(bundle["dense_std"]) != features.NUM_DENSE or not all(
        v > 0 for v in bundle["dense_std"]
    ):
        raise ValueError(f"dense_std needs {features.NUM_DENSE} positive values")
    if not bundle["models"]:
        raise ValueError("an artifact needs at least one model")
    for model in bundle["models"]:
        _check_calibrator(model["id"], model["calibrator"])
        _check_lcb(model["id"], model["lcb"])
    ood = bundle["ood"]
    if ood["k"] != len(ood["centroids"]) or ood["k"] < 1:
        raise ValueError(f"ood k={ood['k']} does not match {len(ood['centroids'])} centroids")


def _check_centroids(bundle: Mapping[str, Any], weights: Mapping[str, np.ndarray]) -> None:
    for j, centroid in enumerate(bundle["ood"]["centroids"]):
        idx = np.asarray(weights[centroid["idx"]]).astype(np.int64)
        if idx.shape != np.asarray(weights[centroid["val"]]).shape:
            raise ValueError(f"centroid {j}: idx and val need the same length")
        if idx.size and (idx.min() < 0 or idx.max() >= bundle["dim"]):
            raise ValueError(f"centroid {j}: indices must be in [0, dim)")
        if (np.diff(idx) <= 0).any():
            raise ValueError(f"centroid {j}: indices must be strictly ascending")


def _manifest(bundle: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest without ``golden`` and with an empty ``version``."""
    models = []
    for m in bundle["models"]:
        models.append(
            {
                "id": m["id"],
                "quality": {"weights": refs[m["quality"]["weights"]], "bias": m["quality"]["bias"]},
                "calibrator": m["calibrator"],
                "lcb": m["lcb"],
                "length": {**m["length"], "weights": refs[m["length"]["weights"]]},
            }
        )
    ood = bundle["ood"]
    return _plain(
        {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "version": "",
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "dim": bundle["dim"],
            "dense_mean": bundle["dense_mean"],
            "dense_std": bundle["dense_std"],
            "models": models,
            "ood": {
                "k": ood["k"],
                "threshold": ood["threshold"],
                "centroids": [
                    {"idx": refs[c["idx"]], "val": refs[c["val"]]} for c in ood["centroids"]
                ],
            },
            "training": bundle["training"],
        }
    )


def _golden(manifest: Mapping[str, Any], blob: bytes) -> list[dict[str, Any]]:
    entries = []
    for prompt in GOLDEN_PROMPTS:
        predicted = predict_bundle(manifest, blob, prompt)
        entries.append(
            {
                "messages": [[role, text] for role, text in prompt],
                "ood_score": predicted["ood_score"],
                "models": {
                    model_id: {key: scores[key] for key in ("p", "l", "e_out", "q99")}
                    for model_id, scores in predicted["models"].items()
                },
            }
        )
    return entries


def export_artifact(
    root: Path, name: str, bundle: Mapping[str, Any], weights: Mapping[str, np.ndarray]
) -> Path:
    """Write ``<root>/<name>/<version>/{manifest.json,weights.bin}`` and return that directory.

    ``bundle`` and ``weights`` come from ``assemble_bundle``. The output depends only on the
    arguments (``training["created_at"]`` is part of the bundle), so the same inputs give the
    same version. Raises ``ValueError`` for malformed inputs before anything is written.
    """
    if not _SAFE_NAME.match(name):
        raise ValueError(f"artifact name {name!r} is not a plain directory name")
    _check_bundle(bundle)
    blob, refs = _pack(bundle, weights)
    _check_centroids(bundle, weights)
    manifest = _manifest(bundle, refs)
    manifest["golden"] = _golden(manifest, blob)
    version = content_version(manifest_bytes(manifest), blob)
    manifest["version"] = version

    out = Path(root) / name / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(blob)
    (out / "manifest.json").write_bytes(manifest_bytes(manifest))
    return out


def load_artifact(directory: Path) -> tuple[dict[str, Any], bytes]:
    """Read ``manifest.json`` and ``weights.bin`` of an exported artifact (no verification)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
    return manifest, (directory / "weights.bin").read_bytes()


def _array(blob: bytes, ref: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(blob, dtype=_DTYPES[ref["dtype"]], count=ref["len"], offset=ref["offset"])


def _linear(
    w: np.ndarray, bias: float, idx: np.ndarray, val: np.ndarray, z: np.ndarray, dim: int
) -> float:
    """The score ``w . [hashed | z] + bias``, accumulated in float64 in the order Go uses."""
    acc = 0.0
    for i, v in zip(idx.tolist(), val.tolist(), strict=True):
        acc += float(w[i]) * v
    for d in range(features.NUM_DENSE):
        acc += float(w[dim + d]) * float(z[d])
    return acc + bias


def predict_bundle(
    manifest: Mapping[str, Any], weights: bytes, messages: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """The reference prediction for ``messages``: what the Go predictor must reproduce.

    ``manifest`` is the parsed manifest and ``weights`` the ``weights.bin`` bytes. Returns
    ``{"ood_score", "ood", "models": {id: {"p", "l", "pooled_rate", "pooled_lcb", "e_out",
    "out_q", "q99", "in_ratio"}}}``; ``out_q`` holds the quantile estimates for u = 0.5, 0.9,
    0.95 and 0.99 and ``q99`` repeats the last one. Nothing is clamped.
    """
    dim = manifest["dim"]
    idx, val, dense = features.extract(list(messages), dim)
    z = (dense - np.asarray(manifest["dense_mean"])) / np.asarray(manifest["dense_std"])

    centroids = [
        {"idx": _array(weights, c["idx"]), "val": _array(weights, c["val"])}
        for c in manifest["ood"]["centroids"]
    ]
    row = scipy.sparse.csr_matrix(
        (val, idx, np.array([0, len(idx)], dtype=np.int64)), shape=(1, dim)
    )
    score = float(ood_score({"centroids": centroids}, row)[0])

    models: dict[str, dict[str, Any]] = {}
    for m in manifest["models"]:
        s_quality = _linear(
            _array(weights, m["quality"]["weights"]), m["quality"]["bias"], idx, val, z, dim
        )
        p = float(apply_calibrator(m["calibrator"], s_quality))
        length = m["length"]
        pred = _linear(_array(weights, length["weights"]), length["bias"], idx, val, z, dim)
        with np.errstate(over="ignore"):
            e_out = float(np.exp(pred) * length["smear"] - 1.0)
            out_q = [
                float(np.exp(pred + length["residual_quantiles"][key]) - 1.0)
                for key in RESIDUAL_KEYS
            ]
        models[m["id"]] = {
            "p": p,
            "l": float(lookup_lcb(m["lcb"], p)),
            "pooled_rate": m["lcb"]["pooled_rate"],
            "pooled_lcb": m["lcb"]["pooled_lcb"],
            "e_out": e_out,
            "out_q": out_q,
            "q99": out_q[3],
            "in_ratio": length["in_ratio"],
        }
    return {
        "ood_score": score,
        "ood": bool(score > manifest["ood"]["threshold"]),
        "models": models,
    }
