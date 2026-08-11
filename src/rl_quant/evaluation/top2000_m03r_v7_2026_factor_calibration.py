"""Pre-2026 execution-factor calibration for the 2026 TOP2000 trace.

The execution projection uses the same four numerical controls as training,
but this evaluation-specific adapter fits them once from states 0..63 and
transitions 0..62 of the bound retrospective chronology.  No 2026 state,
return, or availability observation enters the fit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import torch

from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_CONTEXT_STATES,
    TOP2000_M03R_V7_2026_START,
    Top2000M03RV72026RetrospectiveData,
)
from rl_quant.training.top2000_m03r_v7_factor_calibration import (
    TOP2000_M03R_V7_FACTOR_NAMES,
)

TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-2026-pre-score-factor-calibration-v1"
)
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-2026-pre-score-factor-rule-v1"
)
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_START = 0
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_STOP_EXCLUSIVE = 64
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_START = 0
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_STOP_EXCLUSIVE = 63


class Top2000M03RV72026FactorCalibrationError(ValueError):
    """The factor fit is not the exact bound pre-2026 prefix."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026FactorCalibrationError(
            "factor calibration receipt is not canonical-JSON safe"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026FactorCalibrationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE = {
    "schema": TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SCHEMA,
    "state_start_index": TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_START,
    "state_stop_index_exclusive": (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_STOP_EXCLUSIVE
    ),
    "transition_start_index": (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_START
    ),
    "transition_stop_index_exclusive": (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_STOP_EXCLUSIVE
    ),
    "factor_names": list(TOP2000_M03R_V7_FACTOR_NAMES),
    "numerical_formula": "top2000-m03r-v7-training-four-control-formula-v2",
    "availability_rule": "available-at-every-calibration-state-cash-excluded",
    "standardization_rule": (
        "cross-sectional-selected-mean-population-std-clamped-at-1e-6"
    ),
    "nonfinite_rule": "replace-raw-control-with-zero-before-standardization",
    "cash_rule": "cash-loading-row-exactly-zero",
    "future_observation_access": False,
}
TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256 = _canonical_sha256(
    TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE
)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026PreScoreFactorCalibration:
    """Four standardized execution controls and their complete prefix receipt."""

    loadings: torch.Tensor
    action_ids: tuple[str, ...]
    factor_names: tuple[str, ...]
    calibration_state_dates: tuple[str, ...]
    retrospective_data_receipt_sha256: str
    retrospective_cache_file_sha256: str
    action_hash: str
    action_ids_sha256: str
    calibration_dates_sha256: str
    daily_ohlcv_sha256: str
    availability_sha256: str
    asset_returns_sha256: str
    benchmark_net_returns_sha256: str
    input_array_inventory_sha256: str
    loadings_sha256: str
    rule_sha256: str = TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256
    state_start_index: int = TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_START
    state_stop_index_exclusive: int = (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_STOP_EXCLUSIVE
    )
    transition_start_index: int = (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_START
    )
    transition_stop_index_exclusive: int = (
        TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_STOP_EXCLUSIVE
    )
    fit_role: str = "pre-2026-context-prefix-execution-projection-only"
    first_2026_transition_index: int = TOP2000_M03R_V7_2026_CONTEXT_STATES - 1
    development_only: bool = True
    future_selected_universe: bool = True
    includes_2026_observation: bool = False
    policy_training_authorized: bool = False
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "retrospective_data_receipt_sha256",
            "retrospective_cache_file_sha256",
            "action_hash",
            "action_ids_sha256",
            "calibration_dates_sha256",
            "daily_ohlcv_sha256",
            "availability_sha256",
            "asset_returns_sha256",
            "benchmark_net_returns_sha256",
            "input_array_inventory_sha256",
            "loadings_sha256",
            "rule_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_SCHEMA
            or self.factor_names != TOP2000_M03R_V7_FACTOR_NAMES
            or self.rule_sha256
            != TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256
            or self.state_start_index != 0
            or self.state_stop_index_exclusive != 64
            or self.transition_start_index != 0
            or self.transition_stop_index_exclusive != 63
            or self.first_2026_transition_index != 251
            or self.fit_role
            != "pre-2026-context-prefix-execution-projection-only"
            or not self.development_only
            or not self.future_selected_universe
            or self.includes_2026_observation
            or self.policy_training_authorized
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026FactorCalibrationError(
                "factor calibration rule or nonreportable semantics drifted"
            )
        if (
            not isinstance(self.loadings, torch.Tensor)
            or self.loadings.ndim != 2
            or tuple(self.loadings.shape)
            != (len(self.action_ids), len(self.factor_names))
            or not self.loadings.is_floating_point()
            or not bool(torch.isfinite(self.loadings).all())
            or self.loadings.requires_grad
            or not self.action_ids
            or self.action_ids[0] != "CASH"
            or len(set(self.action_ids)) != len(self.action_ids)
            or bool((self.loadings[0] != 0).any())
        ):
            raise Top2000M03RV72026FactorCalibrationError(
                "factor loadings must be detached finite [action,factor] with zero CASH"
            )
        if _tensor_sha256(self.loadings) != self.loadings_sha256:
            raise Top2000M03RV72026FactorCalibrationError(
                "factor-loading content hash mismatch"
            )
        if _canonical_sha256(list(self.action_ids)) != self.action_ids_sha256:
            raise Top2000M03RV72026FactorCalibrationError(
                "factor action-axis hash mismatch"
            )
        if (
            len(self.calibration_state_dates) != 64
            or _canonical_sha256(list(self.calibration_state_dates))
            != self.calibration_dates_sha256
        ):
            raise Top2000M03RV72026FactorCalibrationError(
                "factor calibration date-axis hash or length drifted"
            )
        try:
            dates = tuple(
                dt.date.fromisoformat(value) for value in self.calibration_state_dates
            )
        except (TypeError, ValueError) as exc:
            raise Top2000M03RV72026FactorCalibrationError(
                "factor calibration dates must be ISO exchange dates"
            ) from exc
        if (
            any(left >= right for left, right in pairwise(dates))
            or dates[-1] >= TOP2000_M03R_V7_2026_START
        ):
            raise Top2000M03RV72026FactorCalibrationError(
                "factor calibration must be a strictly ordered pre-2026 prefix"
            )
        expected_inventory = _canonical_sha256(
            {
                "daily_ohlcv_sha256": self.daily_ohlcv_sha256,
                "availability_sha256": self.availability_sha256,
                "asset_returns_sha256": self.asset_returns_sha256,
                "benchmark_net_returns_sha256": self.benchmark_net_returns_sha256,
            }
        )
        if self.input_array_inventory_sha256 != expected_inventory:
            raise Top2000M03RV72026FactorCalibrationError(
                "factor input-array inventory hash drifted"
            )

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("loadings")
        return payload

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


def fit_top2000_m03r_v7_2026_pre_score_factor_calibration(
    data: Top2000M03RV72026RetrospectiveData,
) -> Top2000M03RV72026PreScoreFactorCalibration:
    """Fit the execution controls from states 0..63 and transitions 0..62."""

    if not isinstance(data, Top2000M03RV72026RetrospectiveData):
        raise Top2000M03RV72026FactorCalibrationError(
            "factor fitting requires typed 2026 retrospective data"
        )
    if data.cache_file_sha256 is None:
        raise Top2000M03RV72026FactorCalibrationError(
            "factor fitting requires an immutable retrospective cache file receipt"
        )
    _require_digest("retrospective_cache_file_sha256", data.cache_file_sha256)
    if (
        data.identity.context_state_rows != TOP2000_M03R_V7_2026_CONTEXT_STATES
        or data.identity.score_transition_start
        != TOP2000_M03R_V7_2026_CONTEXT_STATES - 1
        or len(data.exchange_dates) < 64
    ):
        raise Top2000M03RV72026FactorCalibrationError(
            "retrospective chronology cannot supply the exact pre-2026 factor prefix"
        )
    calibration_dates = data.exchange_dates[:64]
    try:
        parsed_dates = tuple(dt.date.fromisoformat(value) for value in calibration_dates)
    except ValueError as exc:
        raise Top2000M03RV72026FactorCalibrationError(
            "factor prefix contains an invalid exchange date"
        ) from exc
    if (
        len(parsed_dates) != 64
        or any(left >= right for left, right in pairwise(parsed_dates))
        or parsed_dates[-1] >= TOP2000_M03R_V7_2026_START
    ):
        raise Top2000M03RV72026FactorCalibrationError(
            "factor prefix may not contain a 2026 observation"
        )

    sequence = data.sequence
    daily_ohlcv = sequence.decision_state[:64, 0]
    availability = sequence.decision_available[:64, 0]
    asset_returns = sequence.asset_returns[:63, 0]
    benchmark_net_returns = sequence.benchmark_net_returns[:63, 0]
    if (
        daily_ohlcv.ndim != 3
        or daily_ohlcv.shape != (64, len(data.action_ids), 5)
        or availability.shape != (64, len(data.action_ids))
        or asset_returns.shape != (63, len(data.action_ids))
        or benchmark_net_returns.shape != (63,)
    ):
        raise Top2000M03RV72026FactorCalibrationError(
            "factor prefix arrays do not match the frozen state/transition axes"
        )

    # Keep the qualified training formula exactly: market beta, 63-session
    # log return, population volatility, and mean log volume, followed by
    # cross-sectional standardization over continuously available risky assets.
    with torch.no_grad():
        continuously_available = availability.all(0).clone()
        continuously_available[0] = False
        centered_market = benchmark_net_returns - benchmark_net_returns.mean()
        market_variance = centered_market.square().sum().clamp_min(1.0e-12)
        centered_assets = asset_returns - asset_returns.mean(0, keepdim=True)
        beta = (
            centered_assets * centered_market.unsqueeze(-1)
        ).sum(0) / market_variance
        momentum = torch.log1p(asset_returns.clamp_min(-0.999999)).sum(0)
        volatility = asset_returns.std(0, unbiased=False)
        volume = torch.log1p(daily_ohlcv[..., 4].clamp_min(0.0)).mean(0)
        values = torch.stack((beta, momentum, volatility, volume), dim=-1)
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        selected = values[continuously_available]
        if selected.shape[0] <= values.shape[1]:
            raise Top2000M03RV72026FactorCalibrationError(
                "factor prefix needs more continuously available risky assets than factors"
            )
        mean = selected.mean(0)
        scale = selected.std(0, unbiased=False).clamp_min(1.0e-6)
        loadings = ((values - mean) / scale).detach()
        loadings[~continuously_available] = 0.0
        loadings[0] = 0.0

    array_hashes = {
        "daily_ohlcv_sha256": _tensor_sha256(daily_ohlcv),
        "availability_sha256": _tensor_sha256(availability),
        "asset_returns_sha256": _tensor_sha256(asset_returns),
        "benchmark_net_returns_sha256": _tensor_sha256(benchmark_net_returns),
    }
    return Top2000M03RV72026PreScoreFactorCalibration(
        loadings=loadings,
        action_ids=data.action_ids,
        factor_names=TOP2000_M03R_V7_FACTOR_NAMES,
        calibration_state_dates=calibration_dates,
        retrospective_data_receipt_sha256=data.identity.receipt_sha256,
        retrospective_cache_file_sha256=data.cache_file_sha256,
        action_hash=data.identity.action_hash,
        action_ids_sha256=_canonical_sha256(list(data.action_ids)),
        calibration_dates_sha256=_canonical_sha256(list(calibration_dates)),
        daily_ohlcv_sha256=array_hashes["daily_ohlcv_sha256"],
        availability_sha256=array_hashes["availability_sha256"],
        asset_returns_sha256=array_hashes["asset_returns_sha256"],
        benchmark_net_returns_sha256=array_hashes[
            "benchmark_net_returns_sha256"
        ],
        input_array_inventory_sha256=_canonical_sha256(array_hashes),
        loadings_sha256=_tensor_sha256(loadings),
    )


__all__ = [
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_START",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_STATE_STOP_EXCLUSIVE",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_START",
    "TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_TRANSITION_STOP_EXCLUSIVE",
    "Top2000M03RV72026FactorCalibrationError",
    "Top2000M03RV72026PreScoreFactorCalibration",
    "fit_top2000_m03r_v7_2026_pre_score_factor_calibration",
]
