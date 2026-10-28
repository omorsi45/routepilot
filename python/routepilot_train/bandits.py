"""Contextual bandits for routing: LinUCB and linear Thompson sampling over the model arms.

Each arm is a pool model. The context of an item is the same for every arm: the first
``SVD_COMPONENTS`` (128) components of a ``TruncatedSVD`` of the hashed text features (fitted on
the train split only, ``random_state=0``), the 12 dense features standardized with the train mean
and standard deviation, and a constant 1: 141 numbers. The reward of an arm on an item is
``success - lambda * cost_usd`` with the recorded cost of that model on that item.

``LinUCB(alpha)`` keeps a ridge regression per arm (``A = I``, ``b = 0``, inverse updated with
the Sherman-Morrison formula) and plays the arm with the largest
``theta . x + alpha * sqrt(x . A^-1 . x)``, the lower arm index winning a tie. ``LinearTS(v)``
draws, for every arm and every step, one sample from the posterior ``N(theta, v^2 A^-1)`` with a
seeded ``numpy.random.Generator`` and plays the arm whose sample scores highest. Both learn from
bandit feedback only: ``update`` receives the reward of the arm that was played.

The protocol (``select_hyperparameter``, ``run_seed``): a warm-up stream over a block of rows in
a seeded random order that is learned from and not evaluated, then an evaluation stream. The
streams take ``Block`` objects whose ``split`` column is asserted (``require_splits``), so a
tuning or warm-up step cannot read a test row and the evaluation cannot read anything else.
"""

import copy
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from routepilot_train import features
from routepilot_train.data import selection
from routepilot_train.data.splits import require_splits
from routepilot_train.train import design

SVD_COMPONENTS = 128
ALPHAS = (0.1, 0.5, 1.0, 2.0)
TS_SCALES = (0.05, 0.1, 0.5, 1.0)
SEEDS = tuple(range(10))
DOMAIN_ORDER = ("math", "code", "knowledge", "reasoning", "writing")
SPLITS = ("train", "calib", "test", "heldout")
PICO = 1e12
# Independent streams of the same seed: the warm-up order, the evaluation order, the TS draws.
WARMUP_STREAM, EVAL_STREAM, TS_STREAM = 0, 1, 2


def context_size(n_components: int = SVD_COMPONENTS) -> int:
    """Length of a context: the SVD components, the dense features and the constant 1."""
    return n_components + features.NUM_DENSE + 1


# --- learners ---------------------------------------------------------------------------------


class Learner(Protocol):
    """What ``play`` needs from a bandit."""

    def choose(self, x: np.ndarray) -> int: ...

    def update(self, arm: int, x: np.ndarray, reward: float) -> None: ...


class _Ridge:
    """Per-arm ridge regressions with ``A = I`` and ``b = 0`` at the start.

    ``a_inv[k]`` is the inverse of ``A_k``, kept up to date with the Sherman-Morrison formula
    (and re-symmetrized, so rounding cannot make it drift off symmetric); ``theta[k]`` is
    ``A_k^-1 b_k``; ``counts[k]`` is how often arm ``k`` was updated.
    """

    def __init__(self, n_arms: int, dim: int) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be at least 1, got {n_arms}")
        if dim < 1:
            raise ValueError(f"dim must be at least 1, got {dim}")
        self.n_arms, self.dim = n_arms, dim
        self.a_inv = np.tile(np.eye(dim), (n_arms, 1, 1))
        self.b = np.zeros((n_arms, dim))
        self.theta = np.zeros((n_arms, dim))
        self.counts = np.zeros(n_arms, dtype=np.int64)

    def _context(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"context has shape {x.shape}, want ({self.dim},)")
        if not np.isfinite(x).all():
            raise ValueError("context must be finite")
        return x

    def _changed(self, arm: int) -> None:
        """Hook for subclasses that cache something derived from ``a_inv[arm]``."""

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """Learn the reward of ``arm`` on context ``x``; every other arm is left as it was."""
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm {arm} is out of range [0, {self.n_arms})")
        x = self._context(x)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        inv = self.a_inv[arm]
        ax = inv @ x
        inv = inv - np.outer(ax, ax) / (1.0 + float(x @ ax))
        inv = (inv + inv.T) / 2.0
        self.a_inv[arm] = inv
        self.b[arm] += reward * x
        self.theta[arm] = inv @ self.b[arm]
        self.counts[arm] += 1
        self._changed(arm)


class LinUCB(_Ridge):
    """Play the arm with the largest ``theta . x + alpha * sqrt(x . A^-1 . x)``."""

    def __init__(self, n_arms: int, dim: int, alpha: float) -> None:
        if not (math.isfinite(alpha) and alpha >= 0.0):
            raise ValueError(f"alpha must be a finite number >= 0, got {alpha}")
        super().__init__(n_arms, dim)
        self.alpha = float(alpha)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """The upper confidence bound of every arm on context ``x``."""
        x = self._context(x)
        ax = np.einsum("kij,j->ki", self.a_inv, x)
        width = np.sqrt(np.maximum(ax @ x, 0.0))
        return self.theta @ x + self.alpha * width

    def choose(self, x: np.ndarray) -> int:
        """The arm with the largest bound; a tie goes to the lower arm index."""
        return int(np.argmax(self.scores(x)))


def _factor(matrix: np.ndarray) -> np.ndarray:
    """``F`` with ``F F^T = matrix`` (Cholesky; the clipped eigendecomposition if it fails)."""
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        return vectors * np.sqrt(np.maximum(values, 0.0))


class LinearTS(_Ridge):
    """Play the arm whose posterior draw ``theta ~ N(theta_hat, v^2 A^-1)`` scores highest.

    ``rng`` is the only source of randomness; ``choose`` takes exactly one block of
    ``n_arms * dim`` standard normals from it, so a seeded generator makes the whole
    stream reproducible. The Cholesky factor of an arm's covariance is recomputed only after
    that arm was updated.
    """

    def __init__(self, n_arms: int, dim: int, v: float, rng: np.random.Generator) -> None:
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"v must be a finite number > 0, got {v}")
        super().__init__(n_arms, dim)
        self.v = float(v)
        self.rng = rng
        self._factors = np.tile(np.eye(dim), (n_arms, 1, 1))
        self._stale = np.zeros(n_arms, dtype=bool)

    def _changed(self, arm: int) -> None:
        self._stale[arm] = True

    def draw(self) -> np.ndarray:
        """One sample of the parameters of every arm, shape ``(n_arms, dim)``."""
        z = self.rng.standard_normal((self.n_arms, self.dim))
        for arm in np.flatnonzero(self._stale):
            self._factors[arm] = _factor(self.a_inv[arm])
        self._stale[:] = False
        return self.theta + self.v * np.einsum("kij,kj->ki", self._factors, z)

    def choose(self, x: np.ndarray) -> int:
        x = self._context(x)
        return int(np.argmax(self.draw() @ x))


Factory = Callable[[float, int], Learner]


def linucb_factory(n_arms: int, dim: int) -> Factory:
    """``make(alpha, seed)`` for ``select_hyperparameter`` and ``run_seed``; the seed is unused."""

    def make(alpha: float, seed: int) -> Learner:
        return LinUCB(n_arms, dim, alpha)

    return make


def linear_ts_factory(n_arms: int, dim: int) -> Factory:
    """``make(v, seed)``: a ``LinearTS`` whose generator is the TS stream of ``seed``."""

    def make(v: float, seed: int) -> Learner:
        return LinearTS(n_arms, dim, v, np.random.default_rng([seed, TS_STREAM]))

    return make


# --- the context ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """The fitted context map: SVD of the hashed part, standardized dense part, a constant 1."""

    svd: TruncatedSVD
    dense_mean: np.ndarray
    dense_std: np.ndarray
    dim: int

    @property
    def size(self) -> int:
        return context_size(self.svd.n_components)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """One context per text (one user message each), shape ``(len(texts), size)``."""
        hashed, dense = features.extract_texts(list(texts), self.dim)
        parts = [
            self.svd.transform(hashed),
            (dense - self.dense_mean) / self.dense_std,
            np.ones((len(texts), 1)),
        ]
        return np.hstack(parts)


def fit_context(
    train_items: pd.DataFrame,
    dim: int = features.DEFAULT_DIM,
    n_components: int = SVD_COMPONENTS,
) -> Context:
    """Fit the context map on the train rows ``train_items`` (``split`` and ``prompt`` columns).

    Raises ``ValueError`` when a row is not a train row, or ``n_components`` is not in
    ``[1, min(dim, rows))``. The dense standardizer is ``design.standardizer`` on the train rows,
    the one the artifact stores.
    """
    require_splits(train_items, {"train"})
    texts = train_items["prompt"].tolist()
    limit = min(dim, len(texts))
    if not 1 <= n_components < limit:
        raise ValueError(f"n_components must be in [1, {limit}), got {n_components}")
    hashed, dense = features.extract_texts(texts, dim)
    mean, std = design.standardizer(dense)
    svd = TruncatedSVD(n_components=n_components, random_state=0).fit(hashed)
    return Context(svd=svd, dense_mean=mean, dense_std=std, dim=dim)


# --- arms, rewards and blocks -----------------------------------------------------------------


@dataclass(frozen=True)
class Arms:
    """The recorded outcome of every pool model on a set of items.

    ``items`` has ``item_id``, ``dataset``, ``split`` and ``prompt`` and is sorted by ``item_id``;
    ``success`` and ``cost_picousd`` have one column per model in ``models``.
    """

    items: pd.DataFrame
    models: tuple[str, ...]
    success: np.ndarray
    cost_picousd: np.ndarray

    @property
    def cost_usd(self) -> np.ndarray:
        return self.cost_picousd / PICO

    def of_split(self, split: str) -> "Arms":
        """The rows of one split."""
        mask = (self.items["split"] == split).to_numpy()
        return Arms(
            items=self.items.loc[mask].reset_index(drop=True),
            models=self.models,
            success=self.success[mask],
            cost_picousd=self.cost_picousd[mask],
        )


def read_arms(path: Path, splits: Collection[str], models: Sequence[str]) -> Arms:
    """The rows of ``splits`` in the replay input at ``path`` with each model's outcome.

    Only the wanted splits are kept in memory. Raises ``ValueError`` for an unknown split, no
    rows, an item without exactly one user message, or a missing outcome of a model.
    """
    wanted = set(splits)
    unknown = sorted(wanted - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split {unknown}, the splits are {', '.join(SPLITS)}")
    rows: list[tuple[str, str, str, str]] = []
    success: list[list[int]] = []
    cost: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] not in wanted:
                continue
            messages = item["messages"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError(f"item {item['item_id']}: want one user message")
            outcomes = item["outcomes"]
            for model in models:
                if model not in outcomes:
                    raise ValueError(
                        f"item {item['item_id']}: no recorded outcome for model {model}"
                    )
            rows.append((item["item_id"], item["dataset"], item["split"], messages[0]["content"]))
            success.append([int(bool(outcomes[m]["success"])) for m in models])
            cost.append([int(outcomes[m]["cost_picousd"]) for m in models])
    if not rows:
        raise ValueError(f"no items in splits {sorted(wanted)}")
    order = np.argsort(np.array([r[0] for r in rows]), kind="stable")
    frame = pd.DataFrame(rows, columns=["item_id", "dataset", "split", "prompt"])
    return Arms(
        items=frame.iloc[order].reset_index(drop=True),
        models=tuple(models),
        success=np.asarray(success, dtype=np.int64)[order],
        cost_picousd=np.asarray(cost, dtype=np.int64)[order],
    )


def reward_matrix(success: np.ndarray, cost_usd: np.ndarray, lam: float) -> np.ndarray:
    """``success - lam * cost_usd`` for every item and arm (``lam`` in USD^-1)."""
    success = np.asarray(success, dtype=np.float64)
    cost = np.asarray(cost_usd, dtype=np.float64)
    if success.shape != cost.shape:
        raise ValueError(f"success and cost need the same shape, got {success.shape}, {cost.shape}")
    if not math.isfinite(lam):
        raise ValueError(f"lambda must be finite, got {lam}")
    return success - lam * cost


@dataclass(frozen=True)
class Block:
    """Rows ready to play: contexts and the full reward table of every arm.

    ``items`` (``item_id``, ``dataset``, ``split``; no prompt text) is aligned with the rows of
    ``x`` (contexts), ``success``, ``cost_usd`` and ``reward`` (one column per arm). The learner
    only ever sees the entry of the arm it played; the full table is for evaluation.
    """

    items: pd.DataFrame
    x: np.ndarray
    success: np.ndarray
    cost_usd: np.ndarray
    reward: np.ndarray


def make_block(arms: Arms, context: Context, lam: float) -> Block:
    """Contexts and rewards of ``arms`` (the prompt text is dropped once it is featurized)."""
    cost = arms.cost_usd
    return Block(
        items=arms.items.drop(columns="prompt").reset_index(drop=True),
        x=context.transform(arms.items["prompt"].tolist()),
        success=arms.success,
        cost_usd=cost,
        reward=reward_matrix(arms.success, cost, lam),
    )


def concat_blocks(*blocks: Block) -> Block:
    """The rows of ``blocks`` one after the other."""
    return Block(
        items=pd.concat([b.items for b in blocks], ignore_index=True),
        x=np.vstack([b.x for b in blocks]),
        success=np.vstack([b.success for b in blocks]),
        cost_usd=np.vstack([b.cost_usd for b in blocks]),
        reward=np.vstack([b.reward for b in blocks]),
    )


# --- orders -----------------------------------------------------------------------------------


def _permutation(n: int, seed: int, stream: int) -> np.ndarray:
    return np.random.default_rng([seed, stream]).permutation(n)


def warmup_order(n: int, seed: int) -> np.ndarray:
    """The seeded random order of the warm-up stream over ``n`` rows."""
    return _permutation(n, seed, WARMUP_STREAM)


def random_order(n: int, seed: int) -> np.ndarray:
    """The seeded random order of an evaluation stream over ``n`` rows (no shift)."""
    return _permutation(n, seed, EVAL_STREAM)


def shift_order(items: pd.DataFrame, domains: Mapping[str, str] | None = None) -> np.ndarray:
    """Row indices ordered by domain (``DOMAIN_ORDER``) and then by ``item_id``.

    ``domains`` maps a dataset to its domain (default ``selection.DOMAINS``). Raises
    ``ValueError`` for a dataset without a domain or with one outside ``DOMAIN_ORDER``.
    """
    mapping = selection.DOMAINS if domains is None else domains
    rank = {name: i for i, name in enumerate(DOMAIN_ORDER)}
    keys: list[tuple[int, str]] = []
    for item_id, dataset in zip(items["item_id"], items["dataset"], strict=True):
        domain = mapping.get(dataset)
        if domain is None:
            raise ValueError(f"dataset {dataset!r} has no domain")
        if domain not in rank:
            raise ValueError(f"dataset {dataset!r} has domain {domain!r}, not in {DOMAIN_ORDER}")
        keys.append((rank[domain], str(item_id)))
    return np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


# --- streams ----------------------------------------------------------------------------------


def play(learner: Learner, x: np.ndarray, reward: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Run a bandit-feedback stream over the rows in ``order``; the arms played, in order.

    At each step the learner chooses an arm for the row's context and then learns the reward of
    that arm only.
    """
    arms = np.empty(len(order), dtype=np.int64)
    for step, row in enumerate(order):
        arm = int(learner.choose(x[row]))
        learner.update(arm, x[row], float(reward[row, arm]))
        arms[step] = arm
    return arms


def summarize_choices(
    reward: np.ndarray,
    success: np.ndarray,
    cost_usd: np.ndarray,
    order: np.ndarray,
    arms: np.ndarray,
) -> dict[str, Any]:
    """What a stream earned: ``arms[t]`` was played on row ``order[t]``.

    Mean reward, success rate and mean cost per item, the cumulative regret against the per-item
    best arm by reward (total and as a curve), the cumulative reward curve and the share of
    steps each arm was played. An arm of -1 stands for a routing error (no model was chosen):
    a failure at zero cost, so reward 0; it is no arm and has no share.
    """
    order, arms = np.asarray(order), np.asarray(arms)
    played = arms >= 0
    safe = np.where(played, arms, 0)
    chosen = np.where(played, reward[order, safe], 0.0)
    regret = reward.max(axis=1)[order] - chosen
    return {
        "n": int(len(order)),
        "mean_reward": float(chosen.mean()),
        "success_rate": float(np.where(played, success[order, safe], 0).mean()),
        "mean_cost_usd": float(np.where(played, cost_usd[order, safe], 0.0).mean()),
        "cumulative_regret": float(regret.sum()),
        "reward_curve": np.cumsum(chosen),
        "regret_curve": np.cumsum(regret),
        "arm_share": np.bincount(arms[played], minlength=reward.shape[1]) / len(order),
    }


def oracle_arms(reward: np.ndarray) -> np.ndarray:
    """The best arm by reward for every row (the lower index on a tie)."""
    return np.argmax(reward, axis=1)


def best_constant_arm(reward: np.ndarray) -> int:
    """The single arm with the largest mean reward over the rows (the lower index on a tie)."""
    return int(np.argmax(reward.mean(axis=0)))


def spread(values: Sequence[float]) -> dict[str, float]:
    """Mean, minimum and maximum of one number per seed."""
    if len(values) == 0:
        raise ValueError("no values to summarize")
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}


# --- the protocol -----------------------------------------------------------------------------


def select_hyperparameter(
    factory: Factory,
    values: Sequence[float],
    warm: Block,
    evaluate: Block,
    seed: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """The value of ``values`` with the best mean reward: warm up on train, evaluate on calib.

    For each value, a learner from ``factory(value, seed)`` warms up on ``warm`` (train rows, a
    seeded random order) and is then evaluated on ``evaluate`` (calibration rows, a seeded
    random order, still learning). The best mean reward wins; a tie goes to the smaller value.
    Returns the chosen value and one row per value (``value``, ``mean_reward``, ``success_rate``,
    ``mean_cost_usd``, ``cumulative_regret``). Raises ``ValueError`` when ``warm`` holds a
    non-train row or ``evaluate`` a non-calibration row.
    """
    if len(values) == 0:
        raise ValueError("need at least one value to choose from")
    require_splits(warm.items, {"train"})
    require_splits(evaluate.items, {"calib"})
    order = random_order(len(evaluate.items), seed)
    rows: list[dict[str, Any]] = []
    for value in values:
        learner = factory(value, seed)
        play(learner, warm.x, warm.reward, warmup_order(len(warm.items), seed))
        arms = play(learner, evaluate.x, evaluate.reward, order)
        s = summarize_choices(evaluate.reward, evaluate.success, evaluate.cost_usd, order, arms)
        rows.append(
            {
                "value": value,
                "mean_reward": s["mean_reward"],
                "success_rate": s["success_rate"],
                "mean_cost_usd": s["mean_cost_usd"],
                "cumulative_regret": s["cumulative_regret"],
            }
        )
    best = rows[0]
    for row in rows[1:]:
        if row["mean_reward"] > best["mean_reward"] or (
            row["mean_reward"] == best["mean_reward"] and row["value"] < best["value"]
        ):
            best = row
    return best["value"], rows


def run_seed(
    factory: Factory,
    value: float,
    warm: Block,
    evaluate: Block,
    orders: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """One seed of the test protocol: warm up once, then play each order from that state.

    The learner warms up on ``warm`` (train and calibration rows) in the seeded warm-up order,
    which is learned from and not evaluated. Each order of ``orders`` (a permutation of the
    rows of ``evaluate``, test rows only) is then played by its own copy of the warm learner,
    which keeps learning during the stream. Returns, per order name, ``summarize_choices`` plus
    the ``arms`` played.
    """
    require_splits(warm.items, {"train", "calib"})
    require_splits(evaluate.items, {"test"})
    n = len(evaluate.items)
    for name, order in orders.items():
        if not np.array_equal(np.sort(np.asarray(order)), np.arange(n)):
            raise ValueError(f"order {name!r} is not a permutation of the {n} evaluation rows")
    learner = factory(value, seed)
    play(learner, warm.x, warm.reward, warmup_order(len(warm.items), seed))
    out: dict[str, dict[str, Any]] = {}
    for name, order in orders.items():
        branch = copy.deepcopy(learner)
        arms = play(branch, evaluate.x, evaluate.reward, np.asarray(order))
        out[name] = {
            **summarize_choices(
                evaluate.reward, evaluate.success, evaluate.cost_usd, np.asarray(order), arms
            ),
            "arms": arms,
        }
    return out
