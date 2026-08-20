"""Paired chronological alpha metrics and non-wrapping fold bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np


class AlphaPanelEvaluationError(ValueError):
    """The predictive panel or its chronological inference is invalid."""


def _finite_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise AlphaPanelEvaluationError(f"{name} must be nonempty and finite")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def rank_information_coefficient(
    score: Sequence[float],
    target: Sequence[float],
) -> float:
    score_array = np.asarray(_finite_tuple("score", score), dtype=np.float64)
    target_array = np.asarray(_finite_tuple("target", target), dtype=np.float64)
    if score_array.shape != target_array.shape or score_array.size < 3:
        raise AlphaPanelEvaluationError("rank IC needs at least three aligned assets")
    score_rank = _average_ranks(score_array)
    target_rank = _average_ranks(target_array)
    score_rank -= score_rank.mean()
    target_rank -= target_rank.mean()
    denominator = float(np.linalg.norm(score_rank) * np.linalg.norm(target_rank))
    if denominator <= 0.0:
        raise AlphaPanelEvaluationError("rank IC is undefined for a constant cross-section")
    return float(np.dot(score_rank, target_rank) / denominator)


def tail_spread(
    score: Sequence[float],
    target: Sequence[float],
    *,
    tail_fraction: float = 0.20,
) -> float:
    score_array = np.asarray(_finite_tuple("score", score), dtype=np.float64)
    target_array = np.asarray(_finite_tuple("target", target), dtype=np.float64)
    if score_array.shape != target_array.shape or not 0.0 < tail_fraction <= 0.5:
        raise AlphaPanelEvaluationError("tail spread inputs or fraction are invalid")
    count = max(1, int(math.floor(score_array.size * tail_fraction)))
    if score_array.size < 2 * count:
        raise AlphaPanelEvaluationError("tail spread has insufficient cross-sectional support")
    order = np.argsort(score_array, kind="stable")
    return float(target_array[order[-count:]].mean() - target_array[order[:count]].mean())


@dataclass(frozen=True, slots=True)
class AlphaCrossSection:
    session_index: int
    fold_index: int
    model_score: tuple[float, ...]
    baseline_score: tuple[float, ...]
    target: tuple[float, ...]
    valid: tuple[bool, ...]

    def validate(self) -> None:
        if (
            isinstance(self.session_index, bool)
            or not isinstance(self.session_index, int)
            or self.session_index < 0
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise AlphaPanelEvaluationError("cross-section chronology is invalid")
        size = len(self.model_score)
        if (
            size < 3
            or len(self.baseline_score) != size
            or len(self.target) != size
            or len(self.valid) != size
            or any(not isinstance(value, bool) for value in self.valid)
            or sum(self.valid) < 3
        ):
            raise AlphaPanelEvaluationError("cross-section asset support is invalid")
        for name, values in (
            ("model score", self.model_score),
            ("baseline score", self.baseline_score),
            ("target", self.target),
        ):
            selected = tuple(value for value, valid in zip(values, self.valid, strict=True) if valid)
            _finite_tuple(name, selected)

    def selected(self, values: Sequence[float]) -> tuple[float, ...]:
        self.validate()
        return tuple(value for value, valid in zip(values, self.valid, strict=True) if valid)

    @property
    def model_ic(self) -> float:
        return rank_information_coefficient(
            self.selected(self.model_score), self.selected(self.target)
        )

    @property
    def baseline_ic(self) -> float:
        return rank_information_coefficient(
            self.selected(self.baseline_score), self.selected(self.target)
        )

    @property
    def model_tail_spread(self) -> float:
        return tail_spread(self.selected(self.model_score), self.selected(self.target))


@dataclass(frozen=True, slots=True)
class FoldBootstrapConfig:
    replicates: int = 2_000
    block_sessions: int = 21
    seed: int = 17
    lower_probability: float = 0.025

    def validate(self) -> None:
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates < 100
            or isinstance(self.block_sessions, bool)
            or not isinstance(self.block_sessions, int)
            or self.block_sessions <= 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not 0.0 < self.lower_probability < 0.5
        ):
            raise AlphaPanelEvaluationError("fold bootstrap configuration is invalid")


def _nonwrapping_resample(values: tuple[float, ...], block: int, rng: random.Random) -> list[float]:
    if not values:
        raise AlphaPanelEvaluationError("bootstrap fold is empty")
    width = min(block, len(values))
    starts = tuple(range(len(values) - width + 1))
    result: list[float] = []
    while len(result) < len(values):
        start = starts[rng.randrange(len(starts))]
        result.extend(values[start : start + width])
    return result[: len(values)]


def fold_cluster_block_bootstrap_lcb(
    values_by_fold: Sequence[Sequence[float]],
    config: FoldBootstrapConfig,
) -> float:
    """Resample folds as clusters and non-wrapping blocks within each fold."""

    config.validate()
    folds = tuple(_finite_tuple("bootstrap fold", values) for values in values_by_fold)
    if len(folds) < 2:
        raise AlphaPanelEvaluationError("fold bootstrap needs multiple outer folds")
    rng = random.Random(config.seed)
    draws: list[float] = []
    for _ in range(config.replicates):
        selected_folds = [folds[rng.randrange(len(folds))] for _ in folds]
        sampled: list[float] = []
        for fold in selected_folds:
            sampled.extend(_nonwrapping_resample(fold, config.block_sessions, rng))
        draws.append(math.fsum(sampled) / len(sampled))
    draws.sort()
    index = max(0, min(len(draws) - 1, int(config.lower_probability * len(draws))))
    return draws[index]


@dataclass(frozen=True, slots=True)
class AlphaPanelSummary:
    mean_model_ic: float
    mean_baseline_ic: float
    mean_model_minus_baseline_ic: float
    model_minus_baseline_ic_lcb95: float
    model_tail_spread_lcb95: float
    positive_model_ic_fold_count: int
    fold_count: int
    session_count: int

    def validate(self) -> None:
        for name in (
            "mean_model_ic",
            "mean_baseline_ic",
            "mean_model_minus_baseline_ic",
            "model_minus_baseline_ic_lcb95",
            "model_tail_spread_lcb95",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise AlphaPanelEvaluationError(f"{name} is nonfinite")
        if (
            self.fold_count < 2
            or self.session_count < self.fold_count
            or not 0 <= self.positive_model_ic_fold_count <= self.fold_count
        ):
            raise AlphaPanelEvaluationError("alpha panel summary inventory is invalid")


def evaluate_alpha_panel(
    rows: Sequence[AlphaCrossSection],
    bootstrap: FoldBootstrapConfig,
) -> AlphaPanelSummary:
    """Evaluate paired model and baseline scores on identical date/asset support."""

    if not rows:
        raise AlphaPanelEvaluationError("alpha panel is empty")
    sessions: list[int] = []
    by_fold_delta: dict[int, list[float]] = {}
    by_fold_spread: dict[int, list[float]] = {}
    by_fold_model_ic: dict[int, list[float]] = {}
    model_ic: list[float] = []
    baseline_ic: list[float] = []
    for row in rows:
        row.validate()
        sessions.append(row.session_index)
        observed_model_ic = row.model_ic
        observed_baseline_ic = row.baseline_ic
        model_ic.append(observed_model_ic)
        baseline_ic.append(observed_baseline_ic)
        by_fold_delta.setdefault(row.fold_index, []).append(
            observed_model_ic - observed_baseline_ic
        )
        by_fold_spread.setdefault(row.fold_index, []).append(row.model_tail_spread)
        by_fold_model_ic.setdefault(row.fold_index, []).append(observed_model_ic)
    if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
        raise AlphaPanelEvaluationError("alpha panel sessions must be sorted and unique")
    fold_ids = tuple(sorted(by_fold_delta))
    if fold_ids != tuple(range(len(fold_ids))):
        raise AlphaPanelEvaluationError("outer fold inventory must be contiguous")
    delta_lcb = fold_cluster_block_bootstrap_lcb(
        tuple(by_fold_delta[index] for index in fold_ids), bootstrap
    )
    # Common random draws are obtained by using the same frozen bootstrap seed.
    spread_lcb = fold_cluster_block_bootstrap_lcb(
        tuple(by_fold_spread[index] for index in fold_ids), bootstrap
    )
    result = AlphaPanelSummary(
        mean_model_ic=math.fsum(model_ic) / len(model_ic),
        mean_baseline_ic=math.fsum(baseline_ic) / len(baseline_ic),
        mean_model_minus_baseline_ic=(
            math.fsum(model - baseline for model, baseline in zip(model_ic, baseline_ic, strict=True))
            / len(model_ic)
        ),
        model_minus_baseline_ic_lcb95=delta_lcb,
        model_tail_spread_lcb95=spread_lcb,
        positive_model_ic_fold_count=sum(
            math.fsum(values) / len(values) > 0.0
            for values in by_fold_model_ic.values()
        ),
        fold_count=len(fold_ids),
        session_count=len(rows),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class AlphaDiscoveryGate:
    summary: AlphaPanelSummary
    signal_only_gross_return_lcb95: float
    maximum_year_contribution_fraction: float
    maximum_sector_contribution_fraction: float
    minimum_mean_ic: float = 0.02
    minimum_positive_folds: int = 4

    @property
    def passed(self) -> bool:
        self.validate()
        return (
            self.summary.mean_model_ic >= self.minimum_mean_ic
            and self.summary.positive_model_ic_fold_count >= self.minimum_positive_folds
            and self.summary.model_minus_baseline_ic_lcb95 > 0.0
            and self.summary.model_tail_spread_lcb95 > 0.0
            and self.signal_only_gross_return_lcb95 > 0.0
            and self.maximum_year_contribution_fraction <= 0.50
            and self.maximum_sector_contribution_fraction <= 0.35
        )

    def validate(self) -> None:
        self.summary.validate()
        for name in (
            "signal_only_gross_return_lcb95",
            "maximum_year_contribution_fraction",
            "maximum_sector_contribution_fraction",
            "minimum_mean_ic",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise AlphaPanelEvaluationError(f"{name} is nonfinite")
        if (
            not 0.0 <= self.maximum_year_contribution_fraction <= 1.0
            or not 0.0 <= self.maximum_sector_contribution_fraction <= 1.0
            or self.minimum_mean_ic != 0.02
            or self.minimum_positive_folds != 4
            or self.minimum_positive_folds > self.summary.fold_count
        ):
            raise AlphaPanelEvaluationError("alpha discovery thresholds are invalid")


__all__ = [
    "AlphaCrossSection",
    "AlphaDiscoveryGate",
    "AlphaPanelEvaluationError",
    "AlphaPanelSummary",
    "FoldBootstrapConfig",
    "evaluate_alpha_panel",
    "fold_cluster_block_bootstrap_lcb",
    "rank_information_coefficient",
    "tail_spread",
]
