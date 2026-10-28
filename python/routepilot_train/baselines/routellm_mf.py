"""RouteLLM's matrix factorization router as a baseline, trained in numpy.

    python -m routepilot_train.baselines.routellm_mf --pool flagship --derived data/derived

The scorer is RouteLLM's: ``s(m, x) = w2 . (v_m * (W1 x))`` with ``x`` the design matrix row
(hashed text features plus the standardized dense features), ``W1`` of shape
``(dim_h, dim + 12)``, one vector ``v_m`` per model and ``w2`` of length ``dim_h``. Model ``a``
beats model ``b`` on an item with probability ``sigmoid(s(a, x) - s(b, x))``.

Training uses the train split only. For every train item, every ordered pair ``(a, b)`` where
``a`` succeeded and ``b`` failed is a positive example of "a beats b"; the loss is the mean
``-log sigmoid(s(a) - s(b))`` over the pairs of a minibatch of items, plus an L2 term on all
parameters, minimized with Adam. Items where every model succeeded, or every model failed, have
no pair and are skipped.

The score difference between two models is linear in ``x``:
``s(a) - s(b) = x . u`` with ``u = W1^T (w2 * (v_a - v_b))``. ``export_mf`` writes that ``u`` for
the pool's Best Single (strong) and cheapest (weak) model, which is all the Go replay policy
needs: ``score = sigmoid(u . [hashed, standardized dense])``, routed to the strong model when it
reaches a threshold. The exported file is derived data and stays out of git.
"""

import argparse
import base64
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.special import expit

from routepilot_train import features, runinfo
from routepilot_train.data.splits import require_splits
from routepilot_train.evaluate import report
from routepilot_train.train import design, pipeline

EXPERIMENT = "mf_train"
DEFAULT_DIM_H = 128
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_L2 = 1e-5
DEFAULT_BATCH_ITEMS = 256
# Initial scales: small enough that the random initial scores add little noise to the ranking.
INIT_W1 = 0.01
INIT_V = 0.1
INIT_W2 = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
LABELS = (
    "trained on the train split only: no calibration, test or held-out row was read",
    "no price and no latency is used or simulated here",
    "the exported weights are derived data and are not committed; only their SHA-256 is recorded",
)


def pair_arrays(success: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ordered pair (a, b) per item where model ``a`` succeeded and ``b`` failed.

    ``success`` is an items x models 0/1 matrix. Returns three equal-length index arrays:
    the item row, the winner column and the loser column.
    """
    success = np.asarray(success)
    wins, losses = success == 1, success == 0
    rows, winners, losers = np.nonzero(wins[:, :, None] & losses[:, None, :])
    return rows, winners, losers


def loss_and_grads(
    W1T: np.ndarray,
    V: np.ndarray,
    w2: np.ndarray,
    X: Any,
    rows: np.ndarray,
    winners: np.ndarray,
    losers: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Mean pairwise loss of a batch and its gradients, without the L2 term.

    ``W1T`` is ``W1`` transposed, shape ``(width, dim_h)``; ``X`` is the batch's design matrix
    (sparse or dense, ``(items, width)``); ``rows``, ``winners`` and ``losers`` are the pairs,
    with ``rows`` indexing ``X``. Returns ``(loss, dW1T, dV, dw2)``.

    With ``H = X W1T``, ``U = V * w2`` and ``S = H U^T`` (one score per item and model) the loss
    is the mean over pairs of ``softplus(-(S[i, a] - S[i, b]))``; the gradients follow by the
    chain rule through ``S``, ``H`` and the product ``U = V * w2``.
    """
    H = np.asarray(X @ W1T)
    U = V * w2
    S = H @ U.T
    diff = S[rows, winners] - S[rows, losers]
    n_pairs = len(rows)
    loss = float(np.logaddexp(0.0, -diff).mean())
    g = -expit(-diff) / n_pairs  # d loss / d diff, per pair
    dS = np.zeros_like(S)
    np.add.at(dS, (rows, winners), g)
    np.add.at(dS, (rows, losers), -g)
    dU = dS.T @ H
    dH = dS @ U
    dW1T = np.asarray(X.T @ dH)
    return loss, dW1T, dU * w2, (dU * V).sum(axis=0)


class _Adam:
    """Adam over a list of arrays, updated in place, with coupled L2 on every parameter."""

    def __init__(self, params: Sequence[np.ndarray], lr: float, l2: float) -> None:
        self.params = list(params)
        self.lr, self.l2 = lr, l2
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.scratch = [np.empty_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        step_size = self.lr / (1.0 - ADAM_BETA1**self.t)
        second_bias = np.sqrt(1.0 - ADAM_BETA2**self.t)
        for p, g, m, v, tmp in zip(self.params, grads, self.m, self.v, self.scratch, strict=True):
            if self.l2:
                np.multiply(p, self.l2, out=tmp)
                g += tmp
            m *= ADAM_BETA1
            np.multiply(g, 1.0 - ADAM_BETA1, out=tmp)
            m += tmp
            np.multiply(g, g, out=tmp)
            tmp *= 1.0 - ADAM_BETA2
            v *= ADAM_BETA2
            v += tmp
            np.sqrt(v, out=tmp)
            tmp /= second_bias
            tmp += ADAM_EPS
            np.divide(m, tmp, out=tmp)
            tmp *= step_size
            p -= tmp


def _success_matrix(outcomes: pd.DataFrame, models: Sequence[str], rows: int) -> np.ndarray:
    if len(outcomes) != rows:
        raise ValueError(f"X has {rows} rows but outcomes_train has {len(outcomes)}")
    for model in models:
        if model not in outcomes.columns:
            raise ValueError(f"outcomes_train has no column for model {model!r}")
    matrix = outcomes[list(models)].to_numpy()
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("outcomes_train success values must be 0 or 1")
    return matrix.astype(np.int64)


def train_mf(
    X_train: Any,
    outcomes_train: pd.DataFrame,
    models: list[str],
    dim_h: int = DEFAULT_DIM_H,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
    seed: int = 0,
    batch_items: int = DEFAULT_BATCH_ITEMS,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train RouteLLM's MF scorer on the train items.

    ``X_train`` is the train design matrix, shape ``(items, dim + 12)``, kept sparse.
    ``outcomes_train`` has one row per ``X_train`` row, a ``split`` column that must hold
    ``train`` only (checked first) and one 0/1 success column per entry of ``models``.
    Minibatches hold ``batch_items`` shuffled items with all their pairs; the shuffle, the
    initialization and so the whole run follow ``seed``. ``progress`` is called with each
    epoch's history entry.

    Returns ``{"models", "W1" (dim_h x width), "V" (models x dim_h), "w2" (dim_h),
    "dim_h", "n_items", "n_pairs", "history", "hyperparameters"}``; ``history`` has
    ``epoch``, ``loss`` (mean pairwise loss, without the L2 term) and ``seconds`` per epoch.
    """
    require_splits(outcomes_train, {"train"})
    for name, value, ok in (
        ("dim_h", dim_h, dim_h >= 1),
        ("epochs", epochs, epochs >= 1),
        ("lr", lr, lr > 0),
        ("l2", l2, l2 >= 0),
        ("batch_items", batch_items, batch_items >= 1),
    ):
        if not ok:
            raise ValueError(f"{name} is out of range: {value}")
    if not scipy.sparse.issparse(X_train):
        X_train = scipy.sparse.csr_matrix(X_train)
    X = X_train.tocsr()
    width = X.shape[1]
    if width <= features.NUM_DENSE:
        raise ValueError(
            f"X has {width} columns; it needs the hashed block and {features.NUM_DENSE} dense ones"
        )
    models = list(models)
    success = _success_matrix(outcomes_train, models, X.shape[0])
    eligible = np.flatnonzero(((success == 1).any(axis=1)) & ((success == 0).any(axis=1)))
    if len(eligible) == 0:
        raise ValueError("no pair of models with different outcomes on any item: nothing to learn")
    n_pairs = int(len(pair_arrays(success[eligible])[0]))

    rng = np.random.default_rng(seed)
    W1T = rng.normal(0.0, INIT_W1, size=(width, dim_h))
    V = rng.normal(0.0, INIT_V, size=(len(models), dim_h))
    w2 = rng.normal(0.0, INIT_W2, size=dim_h)
    adam = _Adam([W1T, V, w2], lr, l2)

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        total, count = 0.0, 0
        order = rng.permutation(eligible)
        for start in range(0, len(order), batch_items):
            batch = order[start : start + batch_items]
            rows, winners, losers = pair_arrays(success[batch])
            loss, d_w1t, d_v, d_w2 = loss_and_grads(W1T, V, w2, X[batch], rows, winners, losers)
            total += loss * len(rows)
            count += len(rows)
            adam.step([d_w1t, d_v, d_w2])
        entry = {"epoch": epoch, "loss": total / count, "seconds": time.perf_counter() - started}
        history.append(entry)
        if progress is not None:
            progress(entry)

    return {
        "models": models,
        "W1": W1T.T,
        "V": V,
        "w2": w2,
        "dim_h": dim_h,
        "n_items": int(len(eligible)),
        "n_pairs": n_pairs,
        "history": history,
        "hyperparameters": {
            "dim_h": dim_h,
            "epochs": epochs,
            "lr": lr,
            "l2": l2,
            "seed": seed,
            "batch_items": batch_items,
        },
    }


def scores(params: Mapping[str, Any], X: Any) -> np.ndarray:
    """The score ``s(m, x)`` of every model for every row of ``X``: an items x models array."""
    H = np.asarray(X @ np.ascontiguousarray(params["W1"].T))
    return H @ (params["V"] * params["w2"]).T


def _standardizer(params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(params.get("dense_mean", []), dtype=np.float64)
    std = np.asarray(params.get("dense_std", []), dtype=np.float64)
    if mean.shape != (features.NUM_DENSE,) or not np.isfinite(mean).all():
        raise ValueError(f"dense_mean must be {features.NUM_DENSE} finite values")
    if std.shape != (features.NUM_DENSE,) or not (np.isfinite(std).all() and (std > 0).all()):
        raise ValueError(f"dense_std must be {features.NUM_DENSE} finite positive values")
    return mean, std


def export_mf(params: Mapping[str, Any], strong: str, weak: str, out: Path) -> None:
    """Write the linear strong-versus-weak scorer to ``out`` as JSON.

    ``params`` is ``train_mf``'s result plus ``dense_mean`` and ``dense_std``, the standardizer
    the design matrix was built with (the Go side applies it to the dense features). The file
    is ``{"strong", "weak", "dim", "dense_mean", "dense_std", "u"}``: ``dim`` is the hashed
    dimension and ``u`` the base64 of ``dim + 12`` little-endian float32 values,
    ``W1^T (w2 * (v_strong - v_weak))``, so ``sigmoid(u . x)`` is the chance the strong model
    beats the weak one on a request with design row ``x``.
    """
    models = list(params["models"])
    for name in (strong, weak):
        if name not in models:
            raise ValueError(f"model {name!r} is not among the trained models {models}")
    if strong == weak:
        raise ValueError("strong and weak must differ")
    mean, std = _standardizer(params)
    W1, V, w2 = params["W1"], params["V"], params["w2"]
    direction = w2 * (V[models.index(strong)] - V[models.index(weak)])
    u = np.asarray(direction @ W1, dtype=np.float64).astype("<f4")
    if not np.isfinite(u).all():
        raise ValueError("the exported weights are not finite")
    document = {
        "strong": strong,
        "weak": weak,
        "dim": int(W1.shape[1]) - features.NUM_DENSE,
        "dense_mean": [float(v) for v in mean],
        "dense_std": [float(v) for v in std],
        "u": base64.b64encode(u.tobytes()).decode("ascii"),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document) + "\n", encoding="utf-8", newline="\n")


def read_mf(path: Path) -> dict[str, Any]:
    """Read an exported file: ``u`` as float32, the standardizer as float64 arrays."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    u = np.frombuffer(base64.b64decode(document["u"]), dtype="<f4")
    if u.shape != (document["dim"] + features.NUM_DENSE,):
        raise ValueError(f"u has {u.size} values, want {document['dim'] + features.NUM_DENSE}")
    return {
        "strong": document["strong"],
        "weak": document["weak"],
        "dim": int(document["dim"]),
        "dense_mean": np.asarray(document["dense_mean"], dtype=np.float64),
        "dense_std": np.asarray(document["dense_std"], dtype=np.float64),
        "u": u,
    }


def mf_score(mf: Mapping[str, Any], X: Any) -> np.ndarray:
    """``sigmoid(u . x)`` for every row of the design matrix ``X``: the strong model's score."""
    return expit(np.asarray(X @ mf["u"].astype(np.float64)).ravel())


def run(
    pool_name: str,
    derived: Path,
    out: Path | None = None,
    dim: int = features.DEFAULT_DIM,
    dim_h: int = DEFAULT_DIM_H,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
    seed: int = 0,
    results_root: Path | None = None,
    command: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train the MF scorer of ``pool_name`` on its train items and export it.

    The scorer routes between the pool's Best Single and its cheapest model. It is written to
    ``out`` (default ``<derived>/mf/<pool>.json``) and the run to
    ``<results_root>/mf_train/<run-id>/`` (default ``bench/results`` of this checkout).
    """
    started = datetime.now(UTC)
    clock = time.perf_counter()
    derived = Path(derived)
    pool = json.loads((derived / "pools" / f"{pool_name}.json").read_text(encoding="utf-8"))
    out = Path(out) if out is not None else derived / "mf" / f"{pool_name}.json"
    strong, weak = pool["best_single"], pool["cheapest"]

    train, _ = pipeline.load_fit_data(derived, pool)
    hashed, dense = features.extract_texts(train.items["prompt"].tolist(), dim)
    mean, std = design.standardizer(dense)
    X = design.stack(hashed, dense, mean, std)
    models = list(pool["models"])
    outcomes = pd.DataFrame({model: train.success[model] for model in models})
    outcomes["split"] = train.items["split"].to_numpy()

    def show(entry: dict[str, Any]) -> None:
        if progress is not None:
            progress(
                f"epoch {entry['epoch']}: loss {entry['loss']:.5f} in {entry['seconds']:.1f} s"
            )

    clock_train = time.perf_counter()
    params = train_mf(X, outcomes, models, dim_h, epochs, lr, l2, seed, progress=show)
    train_seconds = time.perf_counter() - clock_train
    export_mf({**params, "dense_mean": mean, "dense_std": std}, strong, weak, out)
    total_seconds = time.perf_counter() - clock
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    root = Path(results_root) if results_root is not None else runinfo.repo_root() / "bench/results"
    results_dir = root / EXPERIMENT / runinfo.new_run_id(started)
    results = {
        "pool": pool_name,
        "strong": strong,
        "weak": weak,
        "models": models,
        "dim": dim,
        "input_width": int(X.shape[1]),
        "hyperparameters": params["hyperparameters"],
        "n_items": params["n_items"],
        "n_pairs": params["n_pairs"],
        "epochs": params["history"],
        "train_wall_time_s": train_seconds,
        "total_wall_time_s": total_seconds,
        "mf_file": out.as_posix(),
        "mf_sha256": digest,
        "labels": list(LABELS),
    }
    setup = [
        ("pool", pool_name),
        ("strong (Best Single on train+calib)", strong),
        ("weak (cheapest on train+calib)", weak),
        ("hashed dimension", dim),
        ("train items with a pair / pairs", f"{params['n_items']} / {params['n_pairs']}"),
        ("hyperparameters", json.dumps(params["hyperparameters"])),
        ("training wall time (s)", round(train_seconds, 1)),
        ("total wall time with features (s)", round(total_seconds, 1)),
        ("weights SHA-256", digest),
    ]
    tables = {
        "Setup": pd.DataFrame(setup, columns=["key", "value"]),
        "Epochs (loss: mean pairwise loss on the train items)": pd.DataFrame(params["history"]),
    }
    shown = (
        list(command)
        if command is not None
        else [
            "python",
            "-m",
            "routepilot_train.baselines.routellm_mf",
            "--pool",
            pool_name,
            "--derived",
            derived.as_posix(),
        ]
    )
    report.write_results(
        results_dir,
        EXPERIMENT,
        results,
        tables,
        {},
        {"mf": seed},
        shown,
        digest[:12],
        started,
    )
    return {
        "mf_file": out,
        "results_dir": results_dir,
        "train_wall_time_s": train_seconds,
        "total_wall_time_s": total_seconds,
        "n_pairs": params["n_pairs"],
        "mf_sha256": digest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", required=True, help="pool name, e.g. flagship")
    parser.add_argument("--derived", type=Path, default=Path("data/derived"))
    parser.add_argument("--out", type=Path, default=None, help="default: <derived>/mf/<pool>.json")
    parser.add_argument("--dim", type=int, default=features.DEFAULT_DIM)
    parser.add_argument("--dim-h", type=int, default=DEFAULT_DIM_H)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results-root", type=Path, default=None, help="default: bench/results of this checkout"
    )
    args = parser.parse_args(argv)

    def progress(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    summary = run(
        args.pool,
        args.derived,
        args.out,
        args.dim,
        args.dim_h,
        args.epochs,
        args.lr,
        args.l2,
        args.seed,
        args.results_root,
        ["python", "-m", "routepilot_train.baselines.routellm_mf", *(argv or sys.argv[1:])],
        progress,
    )
    print(f"mf: {summary['mf_file'].as_posix()}")
    print(f"mf sha256: {summary['mf_sha256']}")
    print(f"results: {summary['results_dir'].as_posix()}")
    print(f"pairs: {summary['n_pairs']}")
    print(
        f"wall time: {summary['total_wall_time_s']:.1f} s "
        f"(training {summary['train_wall_time_s']:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
