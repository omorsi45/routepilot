"""Output length model: how many completion tokens a prompt is likely to produce.

A ridge regression on ``log1p(completion_tokens)`` over the shared design matrix. The
router needs the expected length (for ``cost_exp``) and an upper quantile (for
``cost_max``). Exponentiating a log-scale mean underestimates the mean, so the expected
value uses Duan's smearing factor, and the quantiles are shifts of the log prediction by
residual quantiles. Both come from calibration-split residuals (Ruling V1-R7), so neither
is fitted in-sample. ``X_train`` and the train arrays must be train-split rows, the
``*_calib`` arrays calibration-split rows; the Task 13 pipeline enforces this with
``require_splits`` before calling ``fit_length``.
"""

import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from routepilot_train.train.quality import N_FOLDS, raw_score

ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
QUANTILES = (0.5, 0.9, 0.95, 0.99)


def predict_log(params: dict, X) -> np.ndarray:
    """The predicted ``log1p(completion_tokens)``, one value per row of ``X``."""
    return raw_score(params, X)


def expected_out(params: dict, X) -> np.ndarray:
    """Expected completion tokens: ``exp(pred) * smear - 1`` (not clamped; Go clamps)."""
    return np.exp(predict_log(params, X)) * params["smear"] - 1.0


def out_quantile(params: dict, X, u: float) -> np.ndarray:
    """The ``u`` quantile of completion tokens: ``exp(pred + r_u) - 1`` (not clamped)."""
    key = str(u)
    if key not in params["residual_quantiles"]:
        raise ValueError(f"no residual quantile fitted for {u!r}; have {list(QUANTILES)}")
    return np.exp(predict_log(params, X) + params["residual_quantiles"][key]) - 1.0


def _kept_rows(X, out, name: str) -> tuple[object, np.ndarray]:
    """Rows with a valid (non-negative) completion token count, and ``log1p`` of it."""
    out = np.asarray(out, dtype=np.float64)
    if out.ndim != 1 or X.shape[0] != out.shape[0]:
        raise ValueError(f"{name}: X has {X.shape[0]} rows but out has shape {out.shape}")
    keep = np.flatnonzero(out >= 0)
    return X[keep], np.log1p(out[keep])


def _in_ratio(prompt_tokens, in_upper, n_rows: int) -> float:
    prompt_tokens = np.asarray(prompt_tokens, dtype=np.float64)
    in_upper = np.asarray(in_upper, dtype=np.float64)
    if prompt_tokens.shape != (n_rows,) or in_upper.shape != (n_rows,):
        raise ValueError(
            "prompt_tokens_train and in_upper_model_train need one value per train row "
            f"({n_rows}), got {prompt_tokens.shape} and {in_upper.shape}"
        )
    used = prompt_tokens > 0
    if not used.any():
        raise ValueError("no train row has prompt_tokens > 0, so in_ratio is undefined")
    if (in_upper[used] <= 0).any():
        raise ValueError("in_upper_model_train must be positive where prompt_tokens > 0")
    return float(np.median(prompt_tokens[used] / in_upper[used]))


def _pick_alpha(X, target: np.ndarray) -> float:
    """The alpha in ``ALPHA_GRID`` with the lowest 5-fold mean squared error (smaller on ties)."""
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=0).split(np.arange(len(target))))
    best_alpha, best_mse = ALPHA_GRID[0], math.inf
    for alpha in ALPHA_GRID:
        errors = []
        for train, valid in folds:
            model = Ridge(alpha=alpha).fit(X[train], target[train])
            errors.append(float(np.mean((model.predict(X[valid]) - target[valid]) ** 2)))
        mse = float(np.mean(errors))
        if mse < best_mse:  # strict: the first (smallest) alpha wins a tie
            best_alpha, best_mse = alpha, mse
    return best_alpha


def fit_length(
    X_train,
    out_train: np.ndarray,
    X_calib,
    out_calib: np.ndarray,
    prompt_tokens_train: np.ndarray,
    in_upper_model_train: np.ndarray,
) -> dict:
    """Fit one model's length predictor.

    ``out_*`` are the model's recorded ``completion_tokens``; rows with a negative count
    are dropped from the fit and from the residuals. ``prompt_tokens_train`` and
    ``in_upper_model_train`` hold, per train row, the recorded prompt tokens and the
    model-level prompt bound (``prompt_bytes + 24 + prompt_overhead_tokens[model]``).

    Returns ``{"coef", "intercept", "alpha", "smear", "residual_quantiles", "in_ratio"}``:
    ridge ``alpha`` from ``ALPHA_GRID`` by 5-fold CV; ``smear = mean(exp(r))`` and the
    ``residual_quantiles`` (keys ``"0.5"``, ``"0.9"``, ``"0.95"``, ``"0.99"``) from the
    calibration residuals ``r = log1p(out_calib) - pred``; ``in_ratio`` is the median of
    ``prompt_tokens / in_upper`` over train rows with ``prompt_tokens > 0``, so the router
    estimates prompt tokens as ``ceil(in_upper * in_ratio)``.
    """
    in_ratio = _in_ratio(prompt_tokens_train, in_upper_model_train, X_train.shape[0])
    X_fit, target = _kept_rows(X_train, out_train, "train")
    X_cal, target_cal = _kept_rows(X_calib, out_calib, "calibration")
    if target.shape[0] < N_FOLDS:
        raise ValueError(f"need at least {N_FOLDS} train rows with completion_tokens >= 0")
    if target_cal.shape[0] == 0:
        raise ValueError("no calibration rows with completion_tokens >= 0")
    alpha = _pick_alpha(X_fit, target)
    model = Ridge(alpha=alpha).fit(X_fit, target)
    params = {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": float(model.intercept_),
    }
    residuals = target_cal - predict_log(params, X_cal)
    params.update(
        alpha=alpha,
        smear=float(np.mean(np.exp(residuals))),
        residual_quantiles={str(q): float(np.quantile(residuals, q)) for q in QUANTILES},
        in_ratio=in_ratio,
    )
    return params
