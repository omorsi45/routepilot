"""Tests for the RouteLLM matrix factorization baseline: pairs, gradients, learning, export."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from routepilot_train import features
from routepilot_train.baselines import routellm_mf
from routepilot_train.train import design

DIM = 256
MODELS = ["strong", "weak", "mid"]
FILLER = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]


def synthetic(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Texts and a 0/1 success matrix over MODELS.

    Half the texts hold ``zephyr``: strong succeeds, weak and mid fail. The others hold
    ``quartz``: weak succeeds, strong and mid fail. So strong beats weak on a zephyr text and
    loses to it on a quartz text, and mid loses to the winner everywhere.
    """
    rng = np.random.default_rng(seed)
    texts, success = [], np.zeros((n, len(MODELS)), dtype=np.int64)
    for i in range(n):
        words = [str(w) for w in rng.choice(FILLER, size=6)]
        zephyr = bool(rng.integers(0, 2))
        words[int(rng.integers(0, 6))] = "zephyr" if zephyr else "quartz"
        texts.append(" ".join(words))
        success[i, 0 if zephyr else 1] = 1
    return texts, success


def design_for(texts: list[str], mean=None, std=None):
    hashed, dense = features.extract_texts(texts, DIM)
    if mean is None:
        mean, std = design.standardizer(dense)
    return design.stack(hashed, dense, mean, std), mean, std


def frame(success: np.ndarray, split: str = "train") -> pd.DataFrame:
    out = pd.DataFrame(success, columns=MODELS)
    out["split"] = split
    return out


@pytest.fixture(scope="module")
def trained():
    texts, success = synthetic(300, seed=0)
    X, mean, std = design_for(texts)
    params = routellm_mf.train_mf(
        X, frame(success), MODELS, dim_h=8, epochs=40, lr=0.01, batch_items=64
    )
    return params, X, mean, std


def test_pair_arrays_lists_every_ordered_winner_loser_pair():
    success = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0]])
    rows, winners, losers = routellm_mf.pair_arrays(success)
    got = sorted(zip(rows.tolist(), winners.tolist(), losers.tolist(), strict=True))
    # item 0: models 0 and 2 beat model 1; item 3: model 1 beats models 0 and 2.
    assert got == [(0, 0, 1), (0, 2, 1), (3, 1, 0), (3, 1, 2)]


def test_loss_and_grads_match_finite_differences():
    rng = np.random.default_rng(3)
    n, width, dim_h, n_models = 12, 20, 4, 3
    X = scipy.sparse.random(n, width, density=0.4, format="csr", random_state=1, dtype=np.float64)
    success = rng.integers(0, 2, size=(n, n_models))
    rows, winners, losers = routellm_mf.pair_arrays(success)
    assert len(rows) > 10
    W1T = rng.normal(0, 0.5, size=(width, dim_h))
    V = rng.normal(0, 0.5, size=(n_models, dim_h))
    w2 = rng.normal(0, 0.5, size=dim_h)

    loss, grad_w1t, grad_v, grad_w2 = routellm_mf.loss_and_grads(
        W1T, V, w2, X, rows, winners, losers
    )

    base = {"W1T": W1T, "V": V, "w2": w2}

    def loss_with(name: str, array: np.ndarray) -> float:
        args = {**base, name: array}
        return routellm_mf.loss_and_grads(
            args["W1T"], args["V"], args["w2"], X, rows, winners, losers
        )[0]

    eps = 1e-6
    for name, grad in (("W1T", grad_w1t), ("V", grad_v), ("w2", grad_w2)):
        array = base[name]
        assert grad.shape == array.shape
        for flat in rng.choice(array.size, size=min(6, array.size), replace=False):
            index = np.unravel_index(flat, array.shape)
            up, down = array.copy(), array.copy()
            up[index] += eps
            down[index] -= eps
            numeric = (loss_with(name, up) - loss_with(name, down)) / (2 * eps)
            assert grad[index] == pytest.approx(numeric, rel=1e-5, abs=1e-8), (name, index)
    assert loss > 0


def test_mf_learns_pairwise_signal(trained):
    params, _, mean, std = trained
    texts, success = synthetic(400, seed=99)  # unseen texts
    X, _, _ = design_for(texts, mean, std)
    scores = routellm_mf.scores(params, X)
    assert scores.shape == (400, len(MODELS))

    rows, winners, losers = routellm_mf.pair_arrays(success)
    accuracy = float(np.mean(scores[rows, winners] > scores[rows, losers]))
    assert accuracy > 0.95

    # The strong-weak score difference separates the two kinds of text.
    diff = scores[:, 0] - scores[:, 1]
    zephyr = success[:, 0] == 1
    assert diff[zephyr].min() > diff[~zephyr].max()
    assert len(params["history"]) == 40
    assert params["history"][-1]["loss"] < 0.5 * params["history"][0]["loss"]


def test_export_u_matches_direct_difference(trained, tmp_path: Path):
    params, X, mean, std = trained
    exported = {**params, "dense_mean": mean, "dense_std": std}
    out = tmp_path / "mf" / "toy.json"

    routellm_mf.export_mf(exported, "strong", "weak", out)

    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    document = json.loads(raw)
    assert list(document) == ["strong", "weak", "dim", "dense_mean", "dense_std", "u"]
    assert document["strong"] == "strong" and document["weak"] == "weak"
    assert document["dim"] == DIM
    assert document["dense_mean"] == mean.tolist() and document["dense_std"] == std.tolist()
    u = np.frombuffer(base64.b64decode(document["u"]), dtype="<f4")
    assert u.shape == (DIM + features.NUM_DENSE,)

    direct = routellm_mf.scores(params, X)
    want = direct[:, 0] - direct[:, 1]
    np.testing.assert_allclose(X @ u.astype(np.float64), want, rtol=0, atol=1e-4)

    loaded = routellm_mf.read_mf(out)
    assert loaded["strong"] == "strong" and loaded["dim"] == DIM
    np.testing.assert_array_equal(loaded["u"], u)
    np.testing.assert_array_equal(loaded["dense_mean"], mean)
    np.testing.assert_allclose(
        routellm_mf.mf_score(loaded, X), 1 / (1 + np.exp(-(want))), rtol=0, atol=1e-4
    )


def test_training_is_deterministic_and_depends_on_the_seed():
    texts, success = synthetic(80, seed=5)
    X, _, _ = design_for(texts)

    def fit(seed: int):
        return routellm_mf.train_mf(
            X, frame(success), MODELS, dim_h=4, epochs=3, lr=0.01, seed=seed, batch_items=32
        )

    first, again, other = fit(0), fit(0), fit(1)
    for key in ("W1", "V", "w2"):
        np.testing.assert_array_equal(first[key], again[key])
    assert not np.allclose(first["V"], other["V"])
    assert [h["loss"] for h in first["history"]] == [h["loss"] for h in again["history"]]


@pytest.mark.parametrize("bad", ["calib", "test", "heldout"])
def test_train_refuses_rows_outside_the_train_split(bad: str):
    texts, success = synthetic(40, seed=1)
    X, _, _ = design_for(texts)
    outcomes = frame(success)
    outcomes.loc[7, "split"] = bad
    with pytest.raises(ValueError, match="splits not allowed"):
        routellm_mf.train_mf(X, outcomes, MODELS, dim_h=4, epochs=1)
    with pytest.raises(ValueError, match="split"):
        routellm_mf.train_mf(X, outcomes.drop(columns="split"), MODELS, dim_h=4, epochs=1)


def test_train_rejects_bad_input():
    texts, success = synthetic(40, seed=1)
    X, _, _ = design_for(texts)
    with pytest.raises(ValueError, match="rows"):
        routellm_mf.train_mf(X, frame(success[:-1]), MODELS)
    with pytest.raises(ValueError, match="ghost"):
        routellm_mf.train_mf(X, frame(success), [*MODELS, "ghost"])
    with pytest.raises(ValueError, match="0 or 1"):
        routellm_mf.train_mf(X, frame(success + 1), MODELS)
    with pytest.raises(ValueError, match="no pair"):
        routellm_mf.train_mf(X, frame(np.ones_like(success)), MODELS)
    with pytest.raises(ValueError, match="dim_h"):
        routellm_mf.train_mf(X, frame(success), MODELS, dim_h=0)
    with pytest.raises(ValueError, match="epochs"):
        routellm_mf.train_mf(X, frame(success), MODELS, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        routellm_mf.train_mf(X, frame(success), MODELS, lr=0.0)
    with pytest.raises(ValueError, match="dense"):
        routellm_mf.train_mf(X[:, : features.NUM_DENSE], frame(success), MODELS)


def test_export_rejects_unknown_models_and_a_missing_standardizer(trained, tmp_path: Path):
    params, _, mean, std = trained
    exported = {**params, "dense_mean": mean, "dense_std": std}
    out = tmp_path / "mf.json"
    with pytest.raises(ValueError, match="ghost"):
        routellm_mf.export_mf(exported, "ghost", "weak", out)
    with pytest.raises(ValueError, match="differ"):
        routellm_mf.export_mf(exported, "weak", "weak", out)
    with pytest.raises(ValueError, match="dense_mean"):
        routellm_mf.export_mf(params, "strong", "weak", out)
    bad_std = {**exported, "dense_std": np.zeros_like(std)}
    with pytest.raises(ValueError, match="dense_std"):
        routellm_mf.export_mf(bad_std, "strong", "weak", out)
    assert not out.exists()


# --- the command line ------------------------------------------------------------------------

SENTINEL = "sentinelzzword"
POOL = {
    "pool": "toy",
    "models": ["m-strong", "m-weak"],
    "best_single": "m-strong",
    "cheapest": "m-weak",
}


def write_derived(root: Path) -> Path:
    rng = np.random.default_rng(0)
    splits, prompts, outcomes = [], [], []
    for split, count in {"train": 120, "calib": 40, "test": 20, "heldout": 10}.items():
        for i in range(count):
            item_id = f"toy/{split}-{i}"
            zephyr = bool(rng.integers(0, 2))
            words = [str(w) for w in rng.choice(FILLER, size=5)] + [
                "zephyr" if zephyr else "quartz"
            ]
            if split in ("test", "heldout"):
                words.append(SENTINEL)  # a read of these rows shows up in the output
            text = " ".join(words)
            data = text.encode("utf-8")
            splits.append({"item_id": item_id, "dataset": "toy", "split": split})
            prompts.append(
                {
                    "dataset": "toy",
                    "item_id": item_id,
                    "prompt": text,
                    "prompt_sha256": hashlib.sha256(data).hexdigest(),
                    "prompt_bytes": len(data),
                }
            )
            for model, wins in (("m-strong", zephyr), ("m-weak", not zephyr)):
                outcomes.append(
                    {
                        "dataset": "toy",
                        "item_id": item_id,
                        "model": model,
                        "score": float(wins),
                        "success": wins,
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "cost_usd": 0.001,
                        "prompt_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    derived = root / "derived"
    (derived / "pools").mkdir(parents=True)
    pd.DataFrame(splits).to_parquet(derived / "splits.parquet", index=False)
    pd.DataFrame(prompts).to_parquet(derived / "prompts.parquet", index=False)
    pd.DataFrame(outcomes).to_parquet(derived / "outcomes.parquet", index=False)
    (derived / "pools" / "toy.json").write_text(json.dumps(POOL), encoding="utf-8")
    return derived


def test_main_trains_on_the_train_split_and_exports_the_pool_pair(tmp_path: Path, capsys):
    derived = write_derived(tmp_path)
    results_root = tmp_path / "results"

    code = routellm_mf.main(
        [
            "--pool",
            "toy",
            "--derived",
            str(derived),
            "--dim",
            str(DIM),
            "--dim-h",
            "4",
            "--epochs",
            "3",
            "--results-root",
            str(results_root),
        ]
    )

    assert code == 0
    out = derived / "mf" / "toy.json"
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["strong"] == "m-strong" and document["weak"] == "m-weak"
    assert document["dim"] == DIM
    assert SENTINEL not in out.read_text(encoding="utf-8")

    runs = list((results_root / "mf_train").iterdir())
    assert len(runs) == 1
    payload = json.loads((runs[0] / "results.json").read_text(encoding="utf-8"))
    assert payload["pool"] == "toy" and payload["strong"] == "m-strong"
    assert payload["weak"] == "m-weak"
    assert payload["hyperparameters"] == {
        "dim_h": 4,
        "epochs": 3,
        "lr": 0.001,
        "l2": 1e-05,
        "seed": 0,
        "batch_items": 256,
    }
    assert payload["n_items"] == 120 and payload["n_pairs"] == 120
    assert len(payload["epochs"]) == 3
    assert payload["train_wall_time_s"] > 0 and payload["total_wall_time_s"] > 0
    assert payload["mf_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    run_json = json.loads((runs[0] / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "mf_train" and run_json["seeds"] == {"mf": 0}
    report = (runs[0] / "report.md").read_text(encoding="utf-8")
    assert "no calibration, test or held-out row was read" in report
    assert SENTINEL not in report

    printed = capsys.readouterr().out
    assert "mf:" in printed and "wall time" in printed
    assert SENTINEL not in printed
