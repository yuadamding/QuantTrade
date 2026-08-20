"""Causal economic baselines and target-blind/null controls for alpha panels."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from rl_quant.evaluation.alpha_panel import AlphaPanelEvaluationError


def trailing_return_score(
    log_returns: np.ndarray,
    valid: np.ndarray,
    *,
    lookback_sessions: int,
    skip_recent_sessions: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a raw historical return baseline without future filling.

    Input shape is ``[date, asset, history]`` and history ends at the declared
    observation cutoff.  A score is valid only when its complete requested
    lookback is present.
    """

    if (
        not isinstance(log_returns, np.ndarray)
        or not isinstance(valid, np.ndarray)
        or log_returns.ndim != 3
        or valid.shape != log_returns.shape
        or valid.dtype != np.bool_
        or isinstance(lookback_sessions, bool)
        or not isinstance(lookback_sessions, int)
        or lookback_sessions <= 0
        or isinstance(skip_recent_sessions, bool)
        or not isinstance(skip_recent_sessions, int)
        or skip_recent_sessions < 0
        or lookback_sessions + skip_recent_sessions > log_returns.shape[-1]
    ):
        raise AlphaPanelEvaluationError("trailing return baseline inputs are malformed")
    stop = log_returns.shape[-1] - skip_recent_sessions
    start = stop - lookback_sessions
    selected_returns = log_returns[..., start:stop]
    selected_valid = valid[..., start:stop]
    score_valid = np.asarray(selected_valid.all(axis=-1), dtype=np.bool_)
    if not np.isfinite(selected_returns[selected_valid]).all():
        raise AlphaPanelEvaluationError("trailing return history is nonfinite")
    scores = np.where(
        score_valid,
        np.where(selected_valid, selected_returns, 0.0).sum(axis=-1),
        0.0,
    )
    return scores.astype(np.float64, copy=False), score_valid


def reversal_score(
    log_returns: np.ndarray,
    valid: np.ndarray,
    *,
    lookback_sessions: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, score_valid = trailing_return_score(
        log_returns,
        valid,
        lookback_sessions=lookback_sessions,
    )
    return -scores, score_valid


def fit_ridge_baseline(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    ridge_penalty: float,
) -> np.ndarray:
    """Fit one explicit linear control on caller-provided training history."""

    if (
        training_features.ndim != 2
        or training_targets.ndim != 1
        or training_features.shape[0] != training_targets.shape[0]
        or evaluation_features.ndim != 2
        or evaluation_features.shape[1] != training_features.shape[1]
        or training_features.shape[0] <= training_features.shape[1]
        or not np.isfinite(training_features).all()
        or not np.isfinite(training_targets).all()
        or not np.isfinite(evaluation_features).all()
        or not math.isfinite(ridge_penalty)
        or ridge_penalty <= 0.0
    ):
        raise AlphaPanelEvaluationError("ridge baseline inputs are malformed")
    design = np.concatenate(
        (np.ones((training_features.shape[0], 1)), training_features), axis=1
    ).astype(np.float64, copy=False)
    evaluation_design = np.concatenate(
        (np.ones((evaluation_features.shape[0], 1)), evaluation_features), axis=1
    ).astype(np.float64, copy=False)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ training_targets.astype(np.float64, copy=False),
    )
    return evaluation_design @ coefficient


def fixed_random_score(asset_ids: Sequence[str], *, seed: int) -> np.ndarray:
    """Create a target-blind score stable across processes and Python versions."""

    if (
        not asset_ids
        or len(set(asset_ids)) != len(asset_ids)
        or any(not value or value != value.strip() for value in asset_ids)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise AlphaPanelEvaluationError("fixed random-score identity is invalid")
    values: list[float] = []
    for asset_id in asset_ids:
        digest = hashlib.sha256(f"{seed}:{asset_id}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "big")
        values.append(integer / float(2**64) - 0.5)
    return np.asarray(values, dtype=np.float64)


def permute_targets_within_date(
    targets: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Destroy asset identity while preserving each date's target multiset."""

    if (
        targets.ndim != 2
        or valid.shape != targets.shape
        or valid.dtype != np.bool_
        or not np.isfinite(targets[valid]).all()
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise AlphaPanelEvaluationError("target permutation inputs are malformed")
    rng = np.random.default_rng(seed)
    result = targets.copy()
    for date in range(targets.shape[0]):
        selected = np.flatnonzero(valid[date])
        result[date, selected] = targets[date, rng.permutation(selected)]
    return result


def shift_targets_without_wrap(
    targets: np.ndarray,
    valid: np.ndarray,
    *,
    periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift targets across dates with explicit invalid edges and no circular wrap."""

    if (
        targets.ndim != 2
        or valid.shape != targets.shape
        or valid.dtype != np.bool_
        or isinstance(periods, bool)
        or not isinstance(periods, int)
        or periods == 0
        or abs(periods) >= targets.shape[0]
        or not np.isfinite(targets[valid]).all()
    ):
        raise AlphaPanelEvaluationError("target shift inputs are malformed")
    result = np.zeros_like(targets)
    result_valid = np.zeros_like(valid)
    if periods > 0:
        result[periods:] = targets[:-periods]
        result_valid[periods:] = valid[:-periods]
    else:
        width = -periods
        result[:-width] = targets[width:]
        result_valid[:-width] = valid[width:]
    return result, result_valid


__all__ = [
    "fit_ridge_baseline",
    "fixed_random_score",
    "permute_targets_within_date",
    "reversal_score",
    "shift_targets_without_wrap",
    "trailing_return_score",
]
