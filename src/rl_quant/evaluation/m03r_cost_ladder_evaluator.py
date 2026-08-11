"""Exact 0/10/20/40-bp repricing of frozen M03R-v7 action paths."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

M03R_COST_LADDER_BASIS_POINTS = (0, 10, 20, 40)
M03R_COST_LADDER_SCHEMA = "rl-quant.top2000-dev.m03r-v7-cost-ladder-v1"


class M03RCostLadderError(RuntimeError):
    """Return or turnover arrays cannot support exact cost repricing."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finite_vector(name: str, value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise M03RCostLadderError(f"{name} must be a finite vector with at least two rows")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _annualized_mean(value: np.ndarray) -> float:
    return float(value.mean() * 252.0)


def _annualized_ir(active: np.ndarray) -> float | None:
    std = float(active.std(ddof=1))
    return None if std <= 0.0 else float(active.mean() / std * math.sqrt(252.0))


@dataclass(frozen=True, slots=True)
class M03RCostLadderInput:
    setting_index: int
    setting_id: str
    fold_index: int
    score_dates: tuple[str, ...]
    policy_net_returns_20bp: np.ndarray
    benchmark_net_returns_20bp: np.ndarray
    policy_total_one_way_turnover: np.ndarray
    benchmark_total_one_way_turnover: np.ndarray
    primary_one_way_cost_basis_points: int = 20

    def __post_init__(self) -> None:
        if (
            not 0 <= self.setting_index < 12
            or not 0 <= self.fold_index < 6
            or not isinstance(self.setting_id, str)
            or not self.setting_id
            or self.primary_one_way_cost_basis_points != 20
            or len(self.score_dates) < 2
            or tuple(sorted(self.score_dates)) != self.score_dates
            or len(set(self.score_dates)) != len(self.score_dates)
        ):
            raise M03RCostLadderError("cost-ladder identity or chronology is invalid")
        expected = (len(self.score_dates),)
        for name in (
            "policy_net_returns_20bp",
            "benchmark_net_returns_20bp",
            "policy_total_one_way_turnover",
            "benchmark_total_one_way_turnover",
        ):
            value = _finite_vector(name, getattr(self, name))
            object.__setattr__(self, name, value)
            if value.shape != expected:
                raise M03RCostLadderError(f"{name} does not align with score dates")
        if np.any(self.policy_total_one_way_turnover < 0.0) or np.any(
            self.benchmark_total_one_way_turnover < 0.0
        ):
            raise M03RCostLadderError("turnover cannot be negative")


def evaluate_m03r_cost_ladder(value: M03RCostLadderInput) -> dict[str, Any]:
    """Reprice one unchanged action path and separate policy from C1 costs."""

    primary_rate = value.primary_one_way_cost_basis_points / 10_000.0
    policy_gross = value.policy_net_returns_20bp + (
        primary_rate * value.policy_total_one_way_turnover
    )
    benchmark_gross = value.benchmark_net_returns_20bp + (
        primary_rate * value.benchmark_total_one_way_turnover
    )
    gross_active = policy_gross - benchmark_gross
    incremental_turnover = (
        value.policy_total_one_way_turnover - value.benchmark_total_one_way_turnover
    )
    rows: dict[str, dict[str, float | None]] = {}
    for basis_points in M03R_COST_LADDER_BASIS_POINTS:
        rate = basis_points / 10_000.0
        policy_net = policy_gross - rate * value.policy_total_one_way_turnover
        benchmark_net = benchmark_gross - rate * value.benchmark_total_one_way_turnover
        active = policy_net - benchmark_net
        rows[str(basis_points)] = {
            "annualized_policy_net_return": _annualized_mean(policy_net),
            "annualized_benchmark_net_return": _annualized_mean(benchmark_net),
            "annualized_net_active_return": _annualized_mean(active),
            "net_information_ratio": _annualized_ir(active),
        }
    incremental_mean = float(incremental_turnover.mean())
    gross_active_mean = float(gross_active.mean())
    break_even = (
        None
        if incremental_mean <= 0.0
        else float(gross_active_mean / incremental_mean * 10_000.0)
    )
    payload: dict[str, Any] = {
        "schema": M03R_COST_LADDER_SCHEMA,
        "setting_index": value.setting_index,
        "setting_id": value.setting_id,
        "fold_index": value.fold_index,
        "score_dates": list(value.score_dates),
        "primary_one_way_cost_basis_points": value.primary_one_way_cost_basis_points,
        "frozen_action_path_repriced": True,
        "annualized_policy_gross_return": _annualized_mean(policy_gross),
        "annualized_benchmark_gross_return": _annualized_mean(benchmark_gross),
        "annualized_gross_active_return": _annualized_mean(gross_active),
        "gross_information_ratio": _annualized_ir(gross_active),
        "annualized_policy_transaction_cost_at_20bp": _annualized_mean(
            primary_rate * value.policy_total_one_way_turnover
        ),
        "annualized_benchmark_transaction_cost_at_20bp": _annualized_mean(
            primary_rate * value.benchmark_total_one_way_turnover
        ),
        "annualized_incremental_active_transaction_cost_at_20bp": _annualized_mean(
            primary_rate * incremental_turnover
        ),
        "break_even_one_way_cost_basis_points": break_even,
        "cost_ladder": rows,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


__all__ = [
    "M03R_COST_LADDER_BASIS_POINTS",
    "M03R_COST_LADDER_SCHEMA",
    "M03RCostLadderError",
    "M03RCostLadderInput",
    "evaluate_m03r_cost_ladder",
]
