"""Pure numerical 2026 retrospective for the seed-17 TOP2000 M03R-v7 panel.

This module deliberately does not execute a policy or select a checkpoint.  It
consumes one already-executed, continuous 20-bp 2026 chronology for all twelve
development settings and derives a common 10/20/40-basis-point cost ladder.
The 20-bp execution decisions are never fed back through the policy when the
other two rungs are priced.

The TOP2000 universe was selected with future information.  Every receipt is
therefore development-only, nonreportable, and nonpromotable even when all
optional factor inputs are available.  Missing official factor evidence or
holding telemetry remains explicitly unavailable; this surface never invents
a proxy series or caller-authored statistic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256,
    M03R_SEED17_TOP2000_2026_YTD_CONTRASTS,
    M03R_SEED17_TOP2000_2026_YTD_COSTS_BASIS_POINTS,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE,
    M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE,
    M03R_SEED17_TOP2000_2026_YTD_PRIMARY_COST_BASIS_POINTS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_DATA_ROLE,
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SETTING_IDS,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
        Top2000M03RV72026CohortTrajectories,
    )
    from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
        Top2000M03RV72026FactorData,
    )

TOP2000_M03R_V7_2026_EVALUATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-retrospective-evaluation-v1"
)
TOP2000_M03R_V7_2026_INFERENCE_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-inference-plan-v1"
)
TOP2000_M03R_V7_2026_FACTOR_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ff5-momentum-manifest-v1"
)
TOP2000_M03R_V7_2026_COMMON_INPUT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-common-inputs-v1"
)
TOP2000_M03R_V7_2026_SETTING_INPUT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-setting-inputs-v1"
)

TOP2000_M03R_V7_2026_COST_BASIS_POINTS = (
    M03R_SEED17_TOP2000_2026_YTD_COSTS_BASIS_POINTS
)
TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS = (
    M03R_SEED17_TOP2000_2026_YTD_PRIMARY_COST_BASIS_POINTS
)
TOP2000_M03R_V7_2026_PRIMARY_BLOCK_LENGTH = 21
TOP2000_M03R_V7_2026_SENSITIVITY_BLOCK_LENGTHS = (10, 30)
TOP2000_M03R_V7_2026_AGE_BINS = 61
TOP2000_M03R_V7_2026_ACTIONS = ("HOLD", "CONTINUOUS", "EXIT")
TOP2000_M03R_V7_2026_FACTOR_NAMES = ("SMB", "HML", "RMW", "CMA", "Mom")
TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME = "Mkt-RF"
TOP2000_M03R_V7_2026_FACTOR_SOURCE = "official-Kenneth-French-Data-Library"
TOP2000_M03R_V7_2026_TURNOVER_CAUSES = tuple(
    cause.value for cause in TURNOVER_CAUSES
)
TOP2000_M03R_V7_2026_FORCED_EXIT_CAUSES = (
    TurnoverCause.MEMBERSHIP_FORCED.value,
    TurnoverCause.AVAILABILITY_FORCED.value,
    TurnoverCause.RISK_FORCED.value,
    TurnoverCause.TERMINAL.value,
)

_TOP2000_M03R_V7_2026_EXCHANGE_HOLIDAYS = frozenset(
    {
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
    }
)


def _frozen_score_date_axis() -> tuple[str, ...]:
    first = date.fromisoformat(M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE)
    last = date.fromisoformat(M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE)
    current = first
    result: list[str] = []
    while current <= last:
        if current.weekday() < 5 and current not in (
            _TOP2000_M03R_V7_2026_EXCHANGE_HOLIDAYS
        ):
            result.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(result)


TOP2000_M03R_V7_2026_SCORE_DATE_AXIS = _frozen_score_date_axis()
TOP2000_M03R_V7_2026_DECISION_COUNT = len(
    TOP2000_M03R_V7_2026_SCORE_DATE_AXIS
)

TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS = (
    tuple(
        (
            row.contrast_id,
            row.minuend_setting_id,
            row.subtrahend_setting_id,
        )
        for row in M03R_SEED17_TOP2000_2026_YTD_CONTRASTS
    )
)

_DIGEST_CHARS = frozenset("0123456789abcdef")
_BOOTSTRAP_DOMAIN = b"rl-quant.top2000.m03r-v7-2026-moving-block-v1\x00"
_COMMON_INPUT_DOMAIN = b"rl-quant.top2000.m03r-v7-2026-common-input-v1\x00"
_SETTING_INPUT_DOMAIN = b"rl-quant.top2000.m03r-v7-2026-setting-input-v1\x00"
_FACTOR_ARRAY_DOMAIN = b"rl-quant.top2000.m03r-v7-2026-factor-arrays-v1\x00"
_EVALUATION_RECEIPT_MAX_BYTES = 32 * 1024 * 1024


class Top2000M03RV72026EvaluationError(ValueError):
    """A retrospective input, frozen plan, or receipt is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026EvaluationError(
            "2026 retrospective payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_finite_number(
    name: str,
    value: object,
    *,
    nullable: bool = False,
) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Top2000M03RV72026EvaluationError(
            f"{name} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise Top2000M03RV72026EvaluationError(
            f"{name} must be a finite number"
        )
    return result


def _finite_array(
    name: str,
    value: object,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise Top2000M03RV72026EvaluationError(
            f"{name} must be finite with shape {shape}"
        )
    if nonnegative and np.any(array < 0.0):
        raise Top2000M03RV72026EvaluationError(f"{name} must be nonnegative")
    return array


def _update_array_hash(digest: Any, *, name: str, value: np.ndarray) -> None:
    if value.dtype == object:
        payload = _canonical_json(value.tolist())
        dtype = "canonical-utf8-string"
    elif value.dtype == np.bool_:
        payload = np.ascontiguousarray(value, dtype=np.uint8).tobytes(order="C")
        dtype = "uint8-boolean"
    else:
        payload = np.ascontiguousarray(value, dtype=">f8").tobytes(order="C")
        dtype = "big-endian-float64"
    metadata = _canonical_json(
        {"name": name, "shape": list(value.shape), "normalized_dtype": dtype}
    )
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026FactorManifest:
    """Exact official daily FF5+Momentum evidence used without imputation."""

    five_factor_source_file_sha256: str
    momentum_source_file_sha256: str
    source_receipt_sha256: str
    coverage_receipt_sha256: str
    exact_array_receipt_sha256: str
    manifest_sha256: str
    factor_names: tuple[str, ...] = TOP2000_M03R_V7_2026_FACTOR_NAMES
    market_factor_name: str = TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME
    risk_free_name: str = "RF"
    source: str = TOP2000_M03R_V7_2026_FACTOR_SOURCE
    frequency: str = "daily"
    return_unit: str = "decimal-simple-return"
    date_join: str = "exact-date-inner-join"
    missing_value_policy: str = "no-imputation"
    factor_set_defined_before_2026_access: bool = True
    evaluation_returns_used_to_define_factor_set: bool = False
    schema: str = TOP2000_M03R_V7_2026_FACTOR_MANIFEST_SCHEMA

    def semantics(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "five_factor_source_file_sha256": (
                self.five_factor_source_file_sha256
            ),
            "momentum_source_file_sha256": self.momentum_source_file_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "exact_array_receipt_sha256": self.exact_array_receipt_sha256,
            "factor_names": list(self.factor_names),
            "market_factor_name": self.market_factor_name,
            "risk_free_name": self.risk_free_name,
            "frequency": self.frequency,
            "return_unit": self.return_unit,
            "date_join": self.date_join,
            "missing_value_policy": self.missing_value_policy,
            "factor_set_defined_before_2026_access": (
                self.factor_set_defined_before_2026_access
            ),
            "evaluation_returns_used_to_define_factor_set": (
                self.evaluation_returns_used_to_define_factor_set
            ),
        }

    def __post_init__(self) -> None:
        for name in (
            "five_factor_source_file_sha256",
            "momentum_source_file_sha256",
            "source_receipt_sha256",
            "coverage_receipt_sha256",
            "exact_array_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_2026_FACTOR_MANIFEST_SCHEMA
            or self.source != TOP2000_M03R_V7_2026_FACTOR_SOURCE
            or self.factor_names != TOP2000_M03R_V7_2026_FACTOR_NAMES
            or self.market_factor_name != TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME
            or self.risk_free_name != "RF"
            or self.frequency != "daily"
            or self.return_unit != "decimal-simple-return"
            or self.date_join != "exact-date-inner-join"
            or self.missing_value_policy != "no-imputation"
            or not self.factor_set_defined_before_2026_access
            or self.evaluation_returns_used_to_define_factor_set
        ):
            raise Top2000M03RV72026EvaluationError(
                "factor evidence must be frozen official daily FF5+Momentum with "
                "an exact-date inner join and no imputation"
            )
        if _require_digest("manifest_sha256", self.manifest_sha256) != _sha256(
            self.semantics()
        ):
            raise Top2000M03RV72026EvaluationError("factor manifest hash mismatch")


def build_top2000_m03r_v7_2026_factor_manifest(
    *,
    five_factor_source_file_sha256: str,
    momentum_source_file_sha256: str,
    source_receipt_sha256: str,
    coverage_receipt_sha256: str,
    exact_array_receipt_sha256: str,
) -> Top2000M03RV72026FactorManifest:
    """Build the only factor manifest accepted by this retrospective."""

    fields: dict[str, Any] = {
        "five_factor_source_file_sha256": five_factor_source_file_sha256,
        "momentum_source_file_sha256": momentum_source_file_sha256,
        "source_receipt_sha256": source_receipt_sha256,
        "coverage_receipt_sha256": coverage_receipt_sha256,
        "exact_array_receipt_sha256": exact_array_receipt_sha256,
        "factor_names": TOP2000_M03R_V7_2026_FACTOR_NAMES,
        "market_factor_name": TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME,
        "risk_free_name": "RF",
        "source": TOP2000_M03R_V7_2026_FACTOR_SOURCE,
        "frequency": "daily",
        "return_unit": "decimal-simple-return",
        "date_join": "exact-date-inner-join",
        "missing_value_policy": "no-imputation",
        "factor_set_defined_before_2026_access": True,
        "evaluation_returns_used_to_define_factor_set": False,
        "schema": TOP2000_M03R_V7_2026_FACTOR_MANIFEST_SCHEMA,
    }
    unsigned = Top2000M03RV72026FactorManifest.__new__(
        Top2000M03RV72026FactorManifest
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return Top2000M03RV72026FactorManifest(
        **fields,
        manifest_sha256=_sha256(unsigned.semantics()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026InferencePlan:
    """Joint block-bootstrap and multiplicity family frozen before evaluation."""

    bootstrap_replicates: int
    bootstrap_seed_sha256: str
    plan_sha256: str
    primary_block_length_trading_sessions: int = 21
    sensitivity_block_lengths_trading_sessions: tuple[int, ...] = (10, 30)
    one_sided_alpha: float = 0.05
    quantile_method: str = "inverted_cdf"
    resampling: str = "joint-date-index-circular-moving-block"
    multiplicity_method: str = (
        "joint-max-absolute-centered-contrast-fwer-0.05"
    )
    raw_one_sided_p_value_method: str = (
        "null-centered-paired-bootstrap-upper-tail"
    )
    dispersion_standard_deviation_degrees_of_freedom: int = 1
    cohort_rmst_resampling: str = (
        "joint-complete-origin-trajectory-circular-block-by-entry-date"
    )
    cohort_rmst_block_length_origin_sessions: int = 21
    same_origin_block_draws_for_all_settings: bool = True
    date_by_age_snapshot_survival_is_descriptive_only: bool = True
    contrast_family: tuple[tuple[str, str, str], ...] = (
        TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS
    )
    schema: str = TOP2000_M03R_V7_2026_INFERENCE_PLAN_SCHEMA

    @property
    def block_lengths_trading_sessions(self) -> tuple[int, ...]:
        return (
            self.primary_block_length_trading_sessions,
            *self.sensitivity_block_lengths_trading_sessions,
        )

    def semantics(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "primary_block_length_trading_sessions": (
                self.primary_block_length_trading_sessions
            ),
            "sensitivity_block_lengths_trading_sessions": list(
                self.sensitivity_block_lengths_trading_sessions
            ),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed_sha256": self.bootstrap_seed_sha256,
            "one_sided_alpha": self.one_sided_alpha,
            "quantile_method": self.quantile_method,
            "resampling": self.resampling,
            "multiplicity_method": self.multiplicity_method,
            "raw_one_sided_p_value_method": self.raw_one_sided_p_value_method,
            "dispersion_standard_deviation_degrees_of_freedom": (
                self.dispersion_standard_deviation_degrees_of_freedom
            ),
            "cohort_rmst_resampling": self.cohort_rmst_resampling,
            "cohort_rmst_block_length_origin_sessions": (
                self.cohort_rmst_block_length_origin_sessions
            ),
            "same_origin_block_draws_for_all_settings": (
                self.same_origin_block_draws_for_all_settings
            ),
            "date_by_age_snapshot_survival_is_descriptive_only": (
                self.date_by_age_snapshot_survival_is_descriptive_only
            ),
            "contrast_family": [list(row) for row in self.contrast_family],
            "joint_draws_across_settings_metrics_and_contrasts": True,
            "same_date_indices_for_all_cost_rungs": True,
            "one_joint_family_per_checkpoint_fold": True,
            "checkpoint_fold_paths_pooled": False,
        }

    def __post_init__(self) -> None:
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates
            != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.bootstrap.replicate_count
            or self.primary_block_length_trading_sessions
            != TOP2000_M03R_V7_2026_PRIMARY_BLOCK_LENGTH
            or self.sensitivity_block_lengths_trading_sessions
            != TOP2000_M03R_V7_2026_SENSITIVITY_BLOCK_LENGTHS
            or self.bootstrap_seed_sha256
            != M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256
            or self.one_sided_alpha != 0.05
            or self.quantile_method != "inverted_cdf"
            or self.resampling != "joint-date-index-circular-moving-block"
            or self.multiplicity_method
            != "joint-max-absolute-centered-contrast-fwer-0.05"
            or self.raw_one_sided_p_value_method
            != "null-centered-paired-bootstrap-upper-tail"
            or self.dispersion_standard_deviation_degrees_of_freedom != 1
            or self.cohort_rmst_resampling
            != "joint-complete-origin-trajectory-circular-block-by-entry-date"
            or self.cohort_rmst_block_length_origin_sessions != 21
            or not self.same_origin_block_draws_for_all_settings
            or not self.date_by_age_snapshot_survival_is_descriptive_only
            or self.contrast_family != TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS
            or self.schema != TOP2000_M03R_V7_2026_INFERENCE_PLAN_SCHEMA
        ):
            raise Top2000M03RV72026EvaluationError(
                "2026 inference plan must retain the frozen 10,000-draw joint "
                "21/(10,30) moving-block and max-absolute contrast family"
            )
        _require_digest("bootstrap_seed_sha256", self.bootstrap_seed_sha256)
        if _require_digest("plan_sha256", self.plan_sha256) != _sha256(
            self.semantics()
        ):
            raise Top2000M03RV72026EvaluationError("inference plan hash mismatch")


def build_top2000_m03r_v7_2026_inference_plan(
    *,
    bootstrap_replicates: int = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.bootstrap.replicate_count
    ),
    bootstrap_seed_sha256: str = (
        M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256
    ),
) -> Top2000M03RV72026InferencePlan:
    """Build the frozen joint bootstrap/multiplicity plan."""

    fields: dict[str, Any] = {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed_sha256": bootstrap_seed_sha256,
        "primary_block_length_trading_sessions": 21,
        "sensitivity_block_lengths_trading_sessions": (10, 30),
        "one_sided_alpha": 0.05,
        "quantile_method": "inverted_cdf",
        "resampling": "joint-date-index-circular-moving-block",
        "multiplicity_method": (
            "joint-max-absolute-centered-contrast-fwer-0.05"
        ),
        "raw_one_sided_p_value_method": (
            "null-centered-paired-bootstrap-upper-tail"
        ),
        "dispersion_standard_deviation_degrees_of_freedom": 1,
        "cohort_rmst_resampling": (
            "joint-complete-origin-trajectory-circular-block-by-entry-date"
        ),
        "cohort_rmst_block_length_origin_sessions": 21,
        "same_origin_block_draws_for_all_settings": True,
        "date_by_age_snapshot_survival_is_descriptive_only": True,
        "contrast_family": TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS,
        "schema": TOP2000_M03R_V7_2026_INFERENCE_PLAN_SCHEMA,
    }
    unsigned = Top2000M03RV72026InferencePlan.__new__(
        Top2000M03RV72026InferencePlan
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return Top2000M03RV72026InferencePlan(
        **fields,
        plan_sha256=_sha256(unsigned.semantics()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026Telemetry:
    """Optional raw execution telemetry; scalar claims are not accepted."""

    requested_to_executed_projection_distance: object
    age_notional_at_risk: object
    discretionary_exit_notional_by_age: object
    forced_exit_notional_by_cause_and_age: Mapping[str, object]
    action_counts_by_type: Mapping[str, object]
    continuous_hazard: object
    continuous_hazard_observed: object


def _dates(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or array.size != TOP2000_M03R_V7_2026_DECISION_COUNT:
        raise Top2000M03RV72026EvaluationError(
            "score_dates must match the exact frozen 2026 trading-session count"
        )
    parsed: list[date] = []
    normalized = np.empty(array.shape, dtype=object)
    for index, item in enumerate(array):
        if type(item) is not str or not item or item.strip() != item:
            raise Top2000M03RV72026EvaluationError(
                "score_dates must contain canonical ISO-8601 strings"
            )
        try:
            day = date.fromisoformat(item)
        except ValueError as exc:
            raise Top2000M03RV72026EvaluationError(
                "score_dates must contain canonical ISO-8601 strings"
            ) from exc
        if day.isoformat() != item or day.year != 2026:
            raise Top2000M03RV72026EvaluationError(
                "every scored date must be a canonical 2026 date"
            )
        parsed.append(day)
        normalized[index] = item
    if any(current <= previous for previous, current in pairwise(parsed)):
        raise Top2000M03RV72026EvaluationError(
            "score_dates must be globally strictly increasing without fold resets"
        )
    if tuple(normalized.tolist()) != TOP2000_M03R_V7_2026_SCORE_DATE_AXIS:
        raise Top2000M03RV72026EvaluationError(
            "score_dates must equal the exact frozen 2026 exchange-session axis"
        )
    return normalized


def _cause_arrays(
    name: str,
    value: Mapping[str, object],
    shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(
        TOP2000_M03R_V7_2026_TURNOVER_CAUSES
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name} must contain every authoritative turnover cause exactly once"
        )
    return {
        cause: _finite_array(
            f"{name}[{cause}]", value[cause], shape, nonnegative=True
        )
        for cause in TOP2000_M03R_V7_2026_TURNOVER_CAUSES
    }


def _block_indices(
    *,
    seed: bytes,
    replicate: int,
    block_length: int,
    rows: int,
) -> np.ndarray:
    material = (
        _BOOTSTRAP_DOMAIN
        + seed
        + replicate.to_bytes(8, "big")
        + block_length.to_bytes(2, "big")
        + rows.to_bytes(4, "big")
    )
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    )
    starts = rng.integers(0, rows, size=math.ceil(rows / block_length))
    return np.concatenate(
        [
            (start + np.arange(block_length, dtype=np.int64)) % rows
            for start in starts
        ]
    )[:rows]


def _annualized_sharpe(excess: np.ndarray) -> float | None:
    if excess.size < 2:
        return None
    standard_deviation = float(np.std(excess, ddof=1))
    if standard_deviation <= 0.0:
        return None
    return float(np.mean(excess) / standard_deviation * math.sqrt(252.0))


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.exp(np.cumsum(np.log1p(returns)))
    running_maximum = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    complete_wealth = np.concatenate(([1.0], wealth))
    return float(np.min(complete_wealth / running_maximum - 1.0))


def _return_metrics(policy: np.ndarray, benchmark: np.ndarray) -> dict[str, Any]:
    policy_log = np.log1p(policy)
    benchmark_log = np.log1p(benchmark)
    active_log = policy_log - benchmark_log
    active_simple = policy - benchmark
    active_standard_deviation = float(np.std(active_simple, ddof=1))
    return {
        "portfolio_cumulative_net_return": float(np.expm1(policy_log.sum())),
        "benchmark_cumulative_net_return": float(np.expm1(benchmark_log.sum())),
        "cumulative_active_return": float(np.expm1(active_log.sum())),
        "annualized_portfolio_arithmetic_mean_return": float(
            252.0 * np.mean(policy)
        ),
        "annualized_benchmark_arithmetic_mean_return": float(
            252.0 * np.mean(benchmark)
        ),
        "annualized_active_log_return": float(252.0 * np.mean(active_log)),
        "annualized_tracking_error": float(
            math.sqrt(252.0) * active_standard_deviation
        ),
        "annualized_information_ratio": (
            None
            if active_standard_deviation <= 0.0
            else float(
                math.sqrt(252.0)
                * np.mean(active_simple)
                / active_standard_deviation
            )
        ),
        "portfolio_maximum_drawdown": _maximum_drawdown(policy),
        "active_maximum_drawdown": _maximum_drawdown(np.expm1(active_log)),
    }


def _regression(
    dependent: np.ndarray,
    regressors: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any] | None:
    design = np.column_stack((np.ones(dependent.size, dtype=np.float64), regressors))
    if dependent.ndim != 1 or regressors.ndim != 2 or regressors.shape[0] != dependent.size:
        raise Top2000M03RV72026EvaluationError("regression arrays are misaligned")
    if dependent.size <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coefficients, *_ = np.linalg.lstsq(design, dependent, rcond=None)
    return {
        "alpha_daily": float(coefficients[0]),
        "alpha_annualized_arithmetic": float(252.0 * coefficients[0]),
        "loadings": {
            name: float(coefficients[index + 1])
            for index, name in enumerate(names)
        },
        "observations": int(dependent.size),
    }


def _inverted_cdf_quantile(value: np.ndarray, probability: float) -> float:
    """Return the plan's frozen quantile without a dynamically typed method."""

    return float(np.quantile(value, probability, method="inverted_cdf"))


def _null_centered_one_sided_bootstrap_p_value(
    estimate: float,
    draws: np.ndarray,
) -> float:
    """Test a positive paired contrast against a null-centered upper tail."""

    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise Top2000M03RV72026EvaluationError(
            "paired contrast bootstrap draws must be finite and one-dimensional"
        )
    if estimate <= 0.0:
        return 1.0
    null_centered = values - float(estimate)
    return float(
        (1 + np.count_nonzero(null_centered >= estimate)) / (values.size + 1)
    )


def _telemetry_metrics(
    telemetry: Top2000M03RV72026Telemetry | None,
    *,
    settings: int,
    rows: int,
) -> tuple[list[dict[str, Any]] | None, list[tuple[str, np.ndarray]]]:
    if telemetry is None:
        return None, []
    if not isinstance(telemetry, Top2000M03RV72026Telemetry):
        raise Top2000M03RV72026EvaluationError(
            "telemetry must use the typed raw-array contract"
        )
    projection = _finite_array(
        "requested_to_executed_projection_distance",
        telemetry.requested_to_executed_projection_distance,
        (settings, rows),
        nonnegative=True,
    )
    at_risk = _finite_array(
        "age_notional_at_risk",
        telemetry.age_notional_at_risk,
        (settings, rows, TOP2000_M03R_V7_2026_AGE_BINS),
        nonnegative=True,
    )
    discretionary = _finite_array(
        "discretionary_exit_notional_by_age",
        telemetry.discretionary_exit_notional_by_age,
        (settings, rows, TOP2000_M03R_V7_2026_AGE_BINS),
        nonnegative=True,
    )
    if set(telemetry.forced_exit_notional_by_cause_and_age) != set(
        TOP2000_M03R_V7_2026_FORCED_EXIT_CAUSES
    ):
        raise Top2000M03RV72026EvaluationError(
            "forced exit telemetry must contain the four exempt exit causes"
        )
    forced = {
        cause: _finite_array(
            f"forced_exit_notional_by_cause_and_age[{cause}]",
            telemetry.forced_exit_notional_by_cause_and_age[cause],
            (settings, rows, TOP2000_M03R_V7_2026_AGE_BINS),
            nonnegative=True,
        )
        for cause in TOP2000_M03R_V7_2026_FORCED_EXIT_CAUSES
    }
    if set(telemetry.action_counts_by_type) != set(TOP2000_M03R_V7_2026_ACTIONS):
        raise Top2000M03RV72026EvaluationError(
            "action telemetry must contain HOLD, CONTINUOUS, and EXIT"
        )
    actions = {
        action: _finite_array(
            f"action_counts_by_type[{action}]",
            telemetry.action_counts_by_type[action],
            (settings, rows),
            nonnegative=True,
        )
        for action in TOP2000_M03R_V7_2026_ACTIONS
    }
    if any(np.any(value != np.floor(value)) for value in actions.values()):
        raise Top2000M03RV72026EvaluationError(
            "action count telemetry must contain integer-valued counts"
        )
    hazard = np.asarray(telemetry.continuous_hazard, dtype=np.float64)
    observed = np.asarray(telemetry.continuous_hazard_observed, dtype=np.bool_)
    if (
        hazard.ndim != 3
        or hazard.shape[:2] != (settings, rows)
        or observed.shape != hazard.shape
        or not np.isfinite(hazard).all()
        or np.any((hazard < 0.0) | (hazard > 1.0))
    ):
        raise Top2000M03RV72026EvaluationError(
            "continuous hazard/mask must be aligned finite [setting,date,asset] arrays"
        )
    if not np.array_equal(
        observed.sum(axis=-1),
        actions["CONTINUOUS"],
    ):
        raise Top2000M03RV72026EvaluationError(
            "observed hazard count must equal the CONTINUOUS action count"
        )
    total_exit_by_age = discretionary + sum(
        forced.values(), np.zeros_like(discretionary)
    )
    if np.any(total_exit_by_age > at_risk + 1.0e-10):
        raise Top2000M03RV72026EvaluationError(
            "cause-specific exits must be disjoint and cannot exceed notional at risk"
        )
    aggregate_at_risk = at_risk.sum(axis=1)
    aggregate_events = discretionary.sum(axis=1)
    if np.any(discretionary > at_risk + 1.0e-10) or np.any(
        aggregate_events > aggregate_at_risk + 1.0e-10
    ):
        raise Top2000M03RV72026EvaluationError(
            "discretionary exit notional cannot exceed notional at risk by age"
        )

    result: list[dict[str, Any]] = []
    ages = np.arange(TOP2000_M03R_V7_2026_AGE_BINS, dtype=np.float64)
    for setting in range(settings):
        risk = aggregate_at_risk[setting]
        events = aggregate_events[setting]
        age_hazard = np.divide(
            events,
            risk,
            out=np.zeros_like(events),
            where=risk > 0.0,
        ).clip(0.0, 1.0)
        survival = np.ones(TOP2000_M03R_V7_2026_AGE_BINS, dtype=np.float64)
        for age in range(1, TOP2000_M03R_V7_2026_AGE_BINS):
            survival[age] = survival[age - 1] * (1.0 - age_hazard[age - 1])
        sold_total = float(events.sum())
        sale_age_mean = (
            None if sold_total <= 0.0 else float(np.dot(events, ages) / sold_total)
        )
        sale_age_median: int | None = None
        if sold_total > 0.0:
            sale_age_median = int(
                np.searchsorted(np.cumsum(events), 0.5 * sold_total, side="left")
            )
        action_totals = {
            action: float(actions[action][setting].sum())
            for action in TOP2000_M03R_V7_2026_ACTIONS
        }
        all_actions = sum(action_totals.values())
        valid_hazards = hazard[setting][observed[setting]]
        hazard_row: dict[str, Any]
        if valid_hazards.size == 0:
            hazard_row = {
                "status": "unavailable",
                "reason": "no-observed-continuous-hazard-actions",
            }
        else:
            hazard_row = {
                "status": "available",
                "observations": int(valid_hazards.size),
                "quantiles": {
                    str(quantile): float(np.quantile(valid_hazards, quantile))
                    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99)
                },
                "near_zero_fraction": float(np.mean(valid_hazards <= 1.0e-6)),
                "near_one_fraction": float(np.mean(valid_hazards >= 1.0 - 1.0e-6)),
            }
        result.append(
            {
                "projection_distance": {
                    "mean": float(np.mean(projection[setting])),
                    "maximum": float(np.max(projection[setting])),
                    "p95": float(np.quantile(projection[setting], 0.95)),
                },
                "holding_snapshot_descriptive": {
                    "source": "date-by-age-notional-snapshot-product-limit",
                    "eligible_for_required_cohort_rmst": False,
                    "snapshot_product_limit_rmst60_sessions": float(
                        np.sum(survival[:60])
                    ),
                    "snapshot_notional_survival": {
                        "10": float(survival[10]),
                        "20": float(survival[20]),
                        "30": float(survival[30]),
                    },
                    "discretionary_exit_notional_by_age": events.tolist(),
                    "discretionary_exit_notional_total": sold_total,
                    "notional_weighted_discretionary_sale_age": sale_age_mean,
                    "median_discretionary_sale_age": sale_age_median,
                    "forced_exit_notional_by_cause_and_age": {
                        cause: forced[cause][setting].sum(axis=0).tolist()
                        for cause in TOP2000_M03R_V7_2026_FORCED_EXIT_CAUSES
                    },
                },
                "actions": {
                    "counts": action_totals,
                    "frequencies": {
                        action: (
                            None
                            if all_actions <= 0.0
                            else float(action_totals[action] / all_actions)
                        )
                        for action in TOP2000_M03R_V7_2026_ACTIONS
                    },
                },
                "continuous_hazard": hazard_row,
            }
        )
    hash_arrays: list[tuple[str, np.ndarray]] = [
        ("requested_to_executed_projection_distance", projection),
        ("age_notional_at_risk", at_risk),
        ("discretionary_exit_notional_by_age", discretionary),
        ("continuous_hazard", hazard),
        ("continuous_hazard_observed", observed),
    ]
    hash_arrays.extend(
        (f"forced_exit_notional_by_cause_and_age/{cause}", value)
        for cause, value in forced.items()
    )
    hash_arrays.extend(
        (f"action_counts_by_type/{action}", value)
        for action, value in actions.items()
    )
    return result, hash_arrays


def evaluate_top2000_m03r_v7_2026_panel(
    *,
    score_dates: object,
    setting_ids: tuple[str, ...],
    portfolio_net_returns_20bp: object,
    benchmark_net_returns_20bp: object,
    portfolio_turnover_by_cause: Mapping[str, object],
    benchmark_turnover_by_cause: Mapping[str, object],
    checkpoint_sha256_by_setting: Mapping[str, str],
    checkpoint_fold_index: int,
    checkpoint_role: str,
    training_completion_receipt_sha256: str,
    data_manifest_sha256: str,
    chronology_receipt_sha256: str,
    execution_receipt_sha256: str,
    inference_plan: Top2000M03RV72026InferencePlan,
    factor_data: Top2000M03RV72026FactorData | None = None,
    telemetry: Top2000M03RV72026Telemetry | None = None,
    cohort_trajectories: Sequence[Top2000M03RV72026CohortTrajectories]
    | None = None,
    reversal_episode_mask: object | None = None,
    reversal_episode_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate the twelve one-seed rows on one frozen 2026 chronology."""

    if setting_ids != M03R_SEED17_TOP2000_SETTING_IDS:
        raise Top2000M03RV72026EvaluationError(
            "setting_ids must be the exact ordered seed-17 twelve-setting panel"
        )
    expected_checkpoint_role = (
        "headline" if checkpoint_fold_index == 5 else "cutoff-sensitivity"
    )
    if (
        isinstance(checkpoint_fold_index, bool)
        or not isinstance(checkpoint_fold_index, int)
        or checkpoint_fold_index not in range(6)
        or checkpoint_role != expected_checkpoint_role
    ):
        raise Top2000M03RV72026EvaluationError(
            "checkpoint fold must be 0-5 with fold 5 as the sole headline"
        )
    if not isinstance(inference_plan, Top2000M03RV72026InferencePlan):
        raise Top2000M03RV72026EvaluationError(
            "typed 2026 inference plan is required"
        )
    inference_plan.__post_init__()
    dates = _dates(score_dates)
    rows = int(dates.size)
    settings = len(setting_ids)
    if any(block >= rows for block in inference_plan.block_lengths_trading_sessions):
        raise Top2000M03RV72026EvaluationError(
            "every bootstrap block must be shorter than the 2026 chronology"
        )
    portfolio_primary_net = _finite_array(
        "portfolio_net_returns_20bp",
        portfolio_net_returns_20bp,
        (settings, rows),
    )
    benchmark_primary_net = _finite_array(
        "benchmark_net_returns_20bp",
        benchmark_net_returns_20bp,
        (rows,),
    )
    portfolio_turnover = _cause_arrays(
        "portfolio_turnover_by_cause",
        portfolio_turnover_by_cause,
        (settings, rows),
    )
    benchmark_turnover = _cause_arrays(
        "benchmark_turnover_by_cause",
        benchmark_turnover_by_cause,
        (rows,),
    )
    if set(checkpoint_sha256_by_setting) != set(setting_ids):
        raise Top2000M03RV72026EvaluationError(
            "checkpoint hashes must cover every setting exactly once"
        )
    checkpoint_hashes = {
        setting_id: _require_digest(
            f"checkpoint_sha256_by_setting[{setting_id}]",
            checkpoint_sha256_by_setting[setting_id],
        )
        for setting_id in setting_ids
    }
    completion_hash = _require_digest(
        "training_completion_receipt_sha256", training_completion_receipt_sha256
    )
    data_hash = _require_digest("data_manifest_sha256", data_manifest_sha256)
    chronology_hash = _require_digest(
        "chronology_receipt_sha256", chronology_receipt_sha256
    )
    execution_hash = _require_digest(
        "execution_receipt_sha256", execution_receipt_sha256
    )
    if (
        reversal_episode_mask is not None
        or reversal_episode_receipt_sha256 is not None
    ):
        raise Top2000M03RV72026EvaluationError(
            "v1 reversal episodes require a frozen typed pre-outcome artifact; "
            "caller-authored masks and receipt hashes are prohibited"
        )

    factor_available = factor_data is not None
    factor_manifest: Top2000M03RV72026FactorManifest | None
    factor_data_receipt_sha256: str | None
    joined_factor_dates: np.ndarray | None
    risk_free: np.ndarray | None
    market_excess: np.ndarray | None
    factors: np.ndarray | None
    if factor_available:
        from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
            Top2000M03RV72026FactorData,
            Top2000M03RV72026FactorDataError,
        )

        if type(factor_data) is not Top2000M03RV72026FactorData:
            raise Top2000M03RV72026EvaluationError(
                "factor_data must use the exact typed official-factor contract"
            )
        assert factor_data is not None
        if not isinstance(
            factor_data.manifest, Top2000M03RV72026FactorManifest
        ):
            raise Top2000M03RV72026EvaluationError(
                "factor_data must contain the typed official-factor manifest"
            )
        try:
            factor_data.__post_init__()
            factor_data.manifest.__post_init__()
        except (Top2000M03RV72026FactorDataError, TypeError, ValueError) as exc:
            raise Top2000M03RV72026EvaluationError(
                "factor_data failed receipt and array validation"
            ) from exc
        factor_manifest = factor_data.manifest
        factor_data_receipt_sha256 = _require_digest(
            "factor_data.receipt_sha256", factor_data.receipt_sha256
        )
        joined_factor_dates = _dates(factor_data.score_dates)
        if not np.array_equal(joined_factor_dates, dates):
            raise Top2000M03RV72026EvaluationError(
                "official factor dates must exactly equal the scored return dates"
            )
        risk_free = _finite_array(
            "factor_data.risk_free_returns",
            factor_data.risk_free_returns,
            (rows,),
        )
        market_excess = _finite_array(
            "factor_data.market_excess_returns",
            factor_data.market_excess_returns,
            (rows,),
        )
        factors = _finite_array(
            "factor_data.factor_returns",
            factor_data.factor_returns,
            (rows, len(TOP2000_M03R_V7_2026_FACTOR_NAMES)),
        )
    else:
        factor_manifest = None
        factor_data_receipt_sha256 = None
        joined_factor_dates = None
        risk_free = None
        market_excess = None
        factors = None

    factor_arrays_sha256: str | None = None
    if factor_available:
        assert (
            factor_manifest is not None
            and joined_factor_dates is not None
            and risk_free is not None
            and market_excess is not None
            and factors is not None
        )
        factor_digest = hashlib.sha256()
        factor_digest.update(_FACTOR_ARRAY_DOMAIN)
        factor_header = _canonical_json(
            {
                "factor_data_receipt_sha256": factor_data_receipt_sha256,
                "factor_manifest_sha256": factor_manifest.manifest_sha256,
                "date_alignment": "exact-date-inner-join",
                "missing_value_policy": "no-imputation",
            }
        )
        factor_digest.update(len(factor_header).to_bytes(8, "big"))
        factor_digest.update(factor_header)
        for name, array in (
            ("factor_dates", joined_factor_dates),
            ("risk_free_returns", risk_free),
            ("market_excess_returns", market_excess),
            ("factor_returns", factors),
        ):
            _update_array_hash(factor_digest, name=name, value=array)
        factor_arrays_sha256 = factor_digest.hexdigest()

    cohort_rmst_receipt: dict[str, Any] | None = None
    cohort_rows_by_setting: dict[str, dict[str, Any]] = {}
    cohort_trajectory_hashes: dict[str, str | None] = {
        setting_id: None for setting_id in setting_ids
    }
    cohort_origin_block_schedule_sha256: str | None = None
    if cohort_trajectories is not None:
        from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
            Top2000M03RV72026CohortSurvivalError,
            evaluate_top2000_m03r_v7_2026_cohort_rmst60,
            validate_top2000_m03r_v7_2026_cohort_rmst60_receipt,
            validate_top2000_m03r_v7_2026_cohort_trajectories,
        )

        typed_cohorts = tuple(cohort_trajectories)
        try:
            for value in typed_cohorts:
                validate_top2000_m03r_v7_2026_cohort_trajectories(value)
            if (
                tuple(value.origin_dates for value in typed_cohorts)
                != (tuple(str(item) for item in dates),) * settings
                or any(
                    value.receipt.checkpoint_sha256
                    != checkpoint_hashes[value.receipt.setting_id]
                    for value in typed_cohorts
                )
                or any(
                    value.receipt.checkpoint_fold_index != checkpoint_fold_index
                    for value in typed_cohorts
                )
                or any(
                    value.receipt.chronology_receipt_sha256 != chronology_hash
                    for value in typed_cohorts
                )
            ):
                raise Top2000M03RV72026CohortSurvivalError(
                    "cohort panel does not match evaluator dates or source receipts"
                )
            cohort_rmst_receipt = (
                evaluate_top2000_m03r_v7_2026_cohort_rmst60(typed_cohorts)
            )
            validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(
                cohort_rmst_receipt
            )
        except (Top2000M03RV72026CohortSurvivalError, KeyError) as exc:
            raise Top2000M03RV72026EvaluationError(
                "cohort trajectories failed complete-origin validation"
            ) from exc
        cohort_rows_by_setting = {
            row["setting_id"]: row for row in cohort_rmst_receipt["rows"]
        }
        cohort_trajectory_hashes = {
            value.receipt.setting_id: value.receipt.receipt_sha256
            for value in typed_cohorts
        }
        cohort_origin_block_schedule_sha256 = cohort_rmst_receipt[
            "origin_block_schedule_sha256"
        ]

    telemetry_rows, telemetry_hash_arrays = _telemetry_metrics(
        telemetry,
        settings=settings,
        rows=rows,
    )
    total_portfolio_turnover = sum(
        portfolio_turnover.values(), np.zeros((settings, rows), dtype=np.float64)
    )
    total_benchmark_turnover = sum(
        benchmark_turnover.values(), np.zeros(rows, dtype=np.float64)
    )

    primary_rate = TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS * 1.0e-4
    portfolio_gross = portfolio_primary_net + primary_rate * total_portfolio_turnover
    benchmark_gross = benchmark_primary_net + primary_rate * total_benchmark_turnover

    policy_net: dict[int, np.ndarray] = {}
    benchmark_net: dict[int, np.ndarray] = {}
    for cost_bps in TOP2000_M03R_V7_2026_COST_BASIS_POINTS:
        rate = cost_bps * 1.0e-4
        if cost_bps == TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS:
            policy_net[cost_bps] = portfolio_primary_net.copy()
            benchmark_net[cost_bps] = benchmark_primary_net.copy()
        else:
            policy_net[cost_bps] = (
                portfolio_gross - rate * total_portfolio_turnover
            )
            benchmark_net[cost_bps] = (
                benchmark_gross - rate * total_benchmark_turnover
            )
        if np.any(policy_net[cost_bps] <= -1.0) or np.any(
            benchmark_net[cost_bps] <= -1.0
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{cost_bps}-bp repriced returns must exceed -1"
            )

    common_digest = hashlib.sha256()
    common_digest.update(_COMMON_INPUT_DOMAIN)
    common_header = _canonical_json(
        {
            "schema": TOP2000_M03R_V7_2026_COMMON_INPUT_SCHEMA,
            "protocol_generation": (
                M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION
            ),
            "design_id": M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID,
            "protocol_sha256": (
                M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
            ),
            "source_training_protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
            "checkpoint_fold_index": checkpoint_fold_index,
            "checkpoint_role": checkpoint_role,
            "training_completion_receipt_sha256": completion_hash,
            "data_manifest_sha256": data_hash,
            "chronology_receipt_sha256": chronology_hash,
            "execution_receipt_sha256": execution_hash,
            "inference_plan_sha256": inference_plan.plan_sha256,
            "factor_manifest_sha256": (
                None if factor_manifest is None else factor_manifest.manifest_sha256
            ),
            "factor_data_receipt_sha256": factor_data_receipt_sha256,
            "cohort_origin_block_schedule_sha256": (
                cohort_origin_block_schedule_sha256
            ),
        }
    )
    common_digest.update(len(common_header).to_bytes(8, "big"))
    common_digest.update(common_header)
    for name, array in (
        ("score_dates", dates),
        ("benchmark_net_returns_20bp", benchmark_primary_net),
        *tuple(
            (f"benchmark_turnover_by_cause/{cause}", benchmark_turnover[cause])
            for cause in TOP2000_M03R_V7_2026_TURNOVER_CAUSES
        ),
    ):
        _update_array_hash(common_digest, name=name, value=array)
    if factor_available:
        assert (
            joined_factor_dates is not None
            and risk_free is not None
            and market_excess is not None
            and factors is not None
        )
        for name, array in (
            ("factor_dates", joined_factor_dates),
            ("risk_free_returns", risk_free),
            ("market_excess_returns", market_excess),
            ("factor_returns", factors),
        ):
            _update_array_hash(common_digest, name=name, value=array)
    common_inputs_sha256 = common_digest.hexdigest()

    setting_input_hashes: dict[str, str] = {}
    for index, setting_id in enumerate(setting_ids):
        digest = hashlib.sha256()
        digest.update(_SETTING_INPUT_DOMAIN)
        header = _canonical_json(
            {
                "schema": TOP2000_M03R_V7_2026_SETTING_INPUT_SCHEMA,
                "setting_id": setting_id,
                "setting_index": index,
                "checkpoint_sha256": checkpoint_hashes[setting_id],
                "common_inputs_sha256": common_inputs_sha256,
                "cohort_trajectory_receipt_sha256": (
                    cohort_trajectory_hashes[setting_id]
                ),
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        _update_array_hash(
            digest,
            name="portfolio_net_returns_20bp",
            value=portfolio_primary_net[index],
        )
        for cause in TOP2000_M03R_V7_2026_TURNOVER_CAUSES:
            _update_array_hash(
                digest,
                name=f"portfolio_turnover_by_cause/{cause}",
                value=portfolio_turnover[cause][index],
            )
        for name, array in telemetry_hash_arrays:
            _update_array_hash(digest, name=name, value=array[index])
        setting_input_hashes[setting_id] = digest.hexdigest()

    primary_cost = TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS
    primary_policy = policy_net[primary_cost]
    primary_benchmark = benchmark_net[primary_cost]
    primary_active_log = np.log1p(primary_policy) - np.log1p(primary_benchmark)[None, :]
    point_active = 252.0 * primary_active_log.mean(axis=1)

    factor_names = (
        TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME,
        *TOP2000_M03R_V7_2026_FACTOR_NAMES,
    )
    point_portfolio_regressions: list[dict[str, Any] | None] = [None] * settings
    point_benchmark_regression: dict[str, Any] | None = None
    point_active_regressions: list[dict[str, Any] | None] = [None] * settings
    point_beta: np.ndarray | None = None
    if factor_available:
        assert risk_free is not None and market_excess is not None and factors is not None
        regressors = np.column_stack((market_excess, factors))
        point_benchmark_regression = _regression(
            primary_benchmark - risk_free,
            regressors,
            factor_names,
        )
        point_beta = np.empty(settings, dtype=np.float64)
        market_variance = float(np.sum((market_excess - market_excess.mean()) ** 2))
        if market_variance <= 0.0:
            point_beta = None
        for setting in range(settings):
            point_portfolio_regressions[setting] = _regression(
                primary_policy[setting] - risk_free,
                regressors,
                factor_names,
            )
            point_active_regressions[setting] = _regression(
                primary_policy[setting] - primary_benchmark,
                regressors,
                factor_names,
            )
            if point_beta is not None:
                active = primary_policy[setting] - primary_benchmark
                point_beta[setting] = float(
                    np.sum((active - active.mean()) * (market_excess - market_excess.mean()))
                    / market_variance
                )

    bootstrap_by_block: dict[int, dict[str, Any]] = {}
    primary_active_draws: np.ndarray | None = None
    primary_beta_draws: np.ndarray | None = None
    seed = bytes.fromhex(inference_plan.bootstrap_seed_sha256)
    confidence_quantile = inference_plan.one_sided_alpha
    for block_length in inference_plan.block_lengths_trading_sessions:
        indexes = np.stack(
            [
                _block_indices(
                    seed=seed,
                    replicate=replicate,
                    block_length=block_length,
                    rows=rows,
                )
                for replicate in range(inference_plan.bootstrap_replicates)
            ]
        )
        active_draws = np.empty(
            (settings, inference_plan.bootstrap_replicates), dtype=np.float64
        )
        sharpe_difference_draws = np.full_like(active_draws, np.nan)
        beta_draws = np.full_like(active_draws, np.nan)
        active_alpha_draws = np.full_like(active_draws, np.nan)
        for replicate, index in enumerate(indexes):
            sampled_benchmark = primary_benchmark[index]
            sampled_rf: np.ndarray | None = None
            sampled_market: np.ndarray | None = None
            if factor_available:
                assert (
                    risk_free is not None
                    and market_excess is not None
                    and factors is not None
                )
                sampled_rf = risk_free[index]
                sampled_market = market_excess[index]
                sampled_regressors = np.column_stack(
                    (sampled_market, factors[index])
                )
                sampled_design = np.column_stack(
                    (
                        np.ones(rows, dtype=np.float64),
                        sampled_regressors,
                    )
                )
                if np.linalg.matrix_rank(sampled_design) == sampled_design.shape[1]:
                    sampled_active_matrix = (
                        primary_policy[:, index].T - sampled_benchmark[:, None]
                    )
                    coefficients, *_ = np.linalg.lstsq(
                        sampled_design,
                        sampled_active_matrix,
                        rcond=None,
                    )
                    active_alpha_draws[:, replicate] = 252.0 * coefficients[0]
            for setting in range(settings):
                sampled_policy = primary_policy[setting, index]
                active_draws[setting, replicate] = float(
                    252.0
                    * np.mean(np.log1p(sampled_policy) - np.log1p(sampled_benchmark))
                )
                if factor_available:
                    assert sampled_rf is not None and sampled_market is not None
                    policy_sharpe = _annualized_sharpe(sampled_policy - sampled_rf)
                    benchmark_sharpe = _annualized_sharpe(
                        sampled_benchmark - sampled_rf
                    )
                    if policy_sharpe is not None and benchmark_sharpe is not None:
                        sharpe_difference_draws[setting, replicate] = (
                            policy_sharpe - benchmark_sharpe
                        )
                    variance = float(
                        np.sum((sampled_market - sampled_market.mean()) ** 2)
                    )
                    if variance > 0.0:
                        sampled_active = sampled_policy - sampled_benchmark
                        beta_draws[setting, replicate] = float(
                            np.sum(
                                (sampled_active - sampled_active.mean())
                                * (sampled_market - sampled_market.mean())
                            )
                            / variance
                        )
        rows_by_setting: list[dict[str, Any]] = []
        for setting in range(settings):
            row: dict[str, Any] = {
                "active_return_annualized_log_lcb": _inverted_cdf_quantile(
                    active_draws[setting],
                    confidence_quantile,
                ),
                "sharpe_difference_lcb": None,
                "active_multifactor_alpha_annualized_lcb": None,
                "valid_sharpe_replicates": int(
                    np.isfinite(sharpe_difference_draws[setting]).sum()
                ),
                "valid_active_alpha_replicates": int(
                    np.isfinite(active_alpha_draws[setting]).sum()
                ),
            }
            valid_sharpe = sharpe_difference_draws[setting][
                np.isfinite(sharpe_difference_draws[setting])
            ]
            if valid_sharpe.size == inference_plan.bootstrap_replicates:
                row["sharpe_difference_lcb"] = _inverted_cdf_quantile(
                    valid_sharpe,
                    confidence_quantile,
                )
            valid_alpha = active_alpha_draws[setting][
                np.isfinite(active_alpha_draws[setting])
            ]
            if valid_alpha.size == inference_plan.bootstrap_replicates:
                row["active_multifactor_alpha_annualized_lcb"] = (
                    _inverted_cdf_quantile(
                        valid_alpha,
                        confidence_quantile,
                    )
                )
            rows_by_setting.append(row)
        bootstrap_by_block[block_length] = {
            "block_length_trading_sessions": block_length,
            "one_sided_confidence_level": 1.0 - inference_plan.one_sided_alpha,
            "settings": rows_by_setting,
        }
        if block_length == inference_plan.primary_block_length_trading_sessions:
            primary_active_draws = active_draws
            primary_beta_draws = beta_draws
    assert primary_active_draws is not None and primary_beta_draws is not None

    setting_rows: list[dict[str, Any]] = []
    for index, setting_id in enumerate(setting_ids):
        cost_ladder: dict[str, Any] = {}
        for cost_bps in TOP2000_M03R_V7_2026_COST_BASIS_POINTS:
            metrics = _return_metrics(
                policy_net[cost_bps][index], benchmark_net[cost_bps]
            )
            if factor_available:
                assert risk_free is not None
                portfolio_sharpe = _annualized_sharpe(
                    policy_net[cost_bps][index] - risk_free
                )
                benchmark_sharpe = _annualized_sharpe(
                    benchmark_net[cost_bps] - risk_free
                )
                metrics["portfolio_sharpe"] = portfolio_sharpe
                metrics["benchmark_sharpe"] = benchmark_sharpe
                metrics["portfolio_minus_benchmark_sharpe"] = (
                    None
                    if portfolio_sharpe is None or benchmark_sharpe is None
                    else portfolio_sharpe - benchmark_sharpe
                )
            else:
                metrics["portfolio_sharpe"] = None
                metrics["benchmark_sharpe"] = None
                metrics["portfolio_minus_benchmark_sharpe"] = None
            cost_ladder[str(cost_bps)] = metrics

        turnover_by_cause: dict[str, Any] = {}
        for cause in TOP2000_M03R_V7_2026_TURNOVER_CAUSES:
            values = portfolio_turnover[cause][index]
            turnover_by_cause[cause] = {
                "total_one_way_turnover": float(values.sum()),
                "mean_daily_one_way_turnover": float(values.mean()),
                "cost_return_units": {
                    str(cost): float(values.sum() * cost * 1.0e-4)
                    for cost in TOP2000_M03R_V7_2026_COST_BASIS_POINTS
                },
            }
        if factor_available:
            portfolio_regression = point_portfolio_regressions[index]
            active_regression = point_active_regressions[index]
            factor_attribution: dict[str, Any] = {
                "status": (
                    "available"
                    if portfolio_regression is not None
                    and point_benchmark_regression is not None
                    and active_regression is not None
                    else "unavailable"
                ),
                "reason": (
                    None
                    if portfolio_regression is not None
                    and point_benchmark_regression is not None
                    and active_regression is not None
                    else "factor-regression-design-rank-deficient"
                ),
                "portfolio_multifactor_regression": portfolio_regression,
                "benchmark_multifactor_regression": point_benchmark_regression,
                "active_multifactor_regression": active_regression,
            }
        else:
            factor_attribution = {
                "status": "unavailable",
                "reason": "official-daily-ff5-momentum-evidence-not-supplied",
                "portfolio_multifactor_regression": None,
                "benchmark_multifactor_regression": None,
                "active_multifactor_regression": None,
            }
        if factor_available and point_beta is not None:
            valid_beta = primary_beta_draws[index][
                np.isfinite(primary_beta_draws[index])
            ]
            beta_standard_error = (
                None
                if valid_beta.size != inference_plan.bootstrap_replicates
                else float(np.std(valid_beta, ddof=1))
            )
            beta_equivalence = (
                None
                if beta_standard_error is None
                else abs(float(point_beta[index]))
                + NormalDist().inv_cdf(1.0 - inference_plan.one_sided_alpha)
                * beta_standard_error
            )
            beta_row: dict[str, Any] = {
                "status": "available" if beta_equivalence is not None else "unavailable",
                "reason": (
                    None
                    if beta_equivalence is not None
                    else "incomplete-valid-bootstrap-beta-replicates"
                ),
                "active_market_beta": float(point_beta[index]),
                "bootstrap_standard_error": beta_standard_error,
                "equivalence_absolute_upper_bound": beta_equivalence,
                "constraint_maximum_absolute": 0.10,
                "constraint_satisfied": (
                    None if beta_equivalence is None else beta_equivalence <= 0.10
                ),
            }
        else:
            beta_row = {
                "status": "unavailable",
                "reason": (
                    "official-daily-ff5-momentum-evidence-not-supplied"
                    if not factor_available
                    else "market-factor-variance-is-zero"
                ),
                "active_market_beta": None,
                "bootstrap_standard_error": None,
                "equivalence_absolute_upper_bound": None,
                "constraint_maximum_absolute": 0.10,
                "constraint_satisfied": None,
            }
        setting_rows.append(
            {
                "setting_index": index,
                "setting_id": setting_id,
                "checkpoint_sha256": checkpoint_hashes[setting_id],
                "setting_inputs_sha256": setting_input_hashes[setting_id],
                "cost_ladder": cost_ladder,
                "turnover_by_cause": turnover_by_cause,
                "active_beta": beta_row,
                "factor_attribution": factor_attribution,
                "bootstrap": {
                    str(block): bootstrap_by_block[block]["settings"][index]
                    for block in inference_plan.block_lengths_trading_sessions
                },
                "telemetry": (
                    {
                        "status": "unavailable",
                        "reason": "raw-cause-age-action-hazard-telemetry-not-supplied",
                    }
                    if telemetry_rows is None
                    else {"status": "available", **telemetry_rows[index]}
                ),
                "cohort_rmst60": (
                    {
                        "status": "unavailable",
                        "reason": (
                            "complete-score-origin-cohort-trajectory-panel-"
                            "not-supplied"
                        ),
                        "rmst60_sessions": None,
                        "uncertainty": {
                            "status": "unavailable",
                            "reason": (
                                "complete-score-origin-cohort-trajectory-panel-"
                                "not-supplied"
                            ),
                        },
                        "entry_units": None,
                        "trajectory_receipt_sha256": None,
                    }
                    if not cohort_rows_by_setting
                    else cohort_rows_by_setting[setting_id]
                ),
                "reversal_episode_performance": {
                    "status": "unavailable",
                    "reason": (
                        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
                        .reversal_episodes.unavailable_reason
                    ),
                },
            }
        )

    setting_index = {setting_id: index for index, setting_id in enumerate(setting_ids)}
    contrast_specs = inference_plan.contrast_family
    contrast_estimates = np.empty(len(contrast_specs), dtype=np.float64)
    contrast_draws = np.empty(
        (len(contrast_specs), inference_plan.bootstrap_replicates),
        dtype=np.float64,
    )
    for contrast_index, (_name, numerator, denominator) in enumerate(
        contrast_specs
    ):
        numerator_index = setting_index[numerator]
        denominator_index = setting_index[denominator]
        contrast_estimates[contrast_index] = (
            point_active[numerator_index] - point_active[denominator_index]
        )
        contrast_draws[contrast_index] = (
            primary_active_draws[numerator_index]
            - primary_active_draws[denominator_index]
        )
    centered_contrast_draws = contrast_draws - contrast_estimates[:, None]
    joint_max_absolute_centered = np.max(
        np.abs(centered_contrast_draws), axis=0
    )
    simultaneous_critical_value = _inverted_cdf_quantile(
        joint_max_absolute_centered,
        1.0 - inference_plan.one_sided_alpha,
    )
    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, (name, numerator, denominator) in enumerate(contrast_specs):
        estimate = float(contrast_estimates[contrast_index])
        draws = contrast_draws[contrast_index]
        raw_p = _null_centered_one_sided_bootstrap_p_value(
            estimate,
            draws,
        )
        adjusted_p = (
            1.0
            if estimate <= 0.0
            else float(
                (
                    1
                    + np.count_nonzero(
                        joint_max_absolute_centered >= estimate
                    )
                )
                / (inference_plan.bootstrap_replicates + 1)
            )
        )
        simultaneous_lcb = estimate - simultaneous_critical_value
        contrast_rows.append(
            {
                "contrast_id": name,
                "numerator_setting_id": numerator,
                "denominator_setting_id": denominator,
                "estimand": "20bp-annualized-active-log-return-difference",
                "estimate": estimate,
                "one_sided_95pct_lcb": _inverted_cdf_quantile(
                    draws,
                    inference_plan.one_sided_alpha,
                ),
                "raw_one_sided_null_centered_bootstrap_p_value": raw_p,
                "simultaneous_fwer95_lcb": simultaneous_lcb,
                "simultaneous_fwer95_ucb": estimate + simultaneous_critical_value,
                "multiplicity_adjusted_p_value": adjusted_p,
                "multiplicity_reject_at_family_alpha_0_05": bool(
                    simultaneous_lcb > 0.0
                ),
            }
        )

    benchmark_turnover_receipt: dict[str, Any] = {}
    for cause in TOP2000_M03R_V7_2026_TURNOVER_CAUSES:
        values = benchmark_turnover[cause]
        benchmark_turnover_receipt[cause] = {
            "total_one_way_turnover": float(values.sum()),
            "mean_daily_one_way_turnover": float(values.mean()),
            "cost_return_units": {
                str(cost): float(values.sum() * cost * 1.0e-4)
                for cost in TOP2000_M03R_V7_2026_COST_BASIS_POINTS
            },
        }

    unsigned: dict[str, Any] = {
        "schema": TOP2000_M03R_V7_2026_EVALUATION_SCHEMA,
        "protocol_generation": (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION
        ),
        "design_id": M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID,
        "protocol_sha256": (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
        ),
        "source_training_protocol_generation": (
            M03R_SEED17_TOP2000_PROTOCOL_GENERATION
        ),
        "source_training_design_id": M03R_SEED17_TOP2000_DESIGN_ID,
        "source_training_protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "data_role": M03R_SEED17_TOP2000_DATA_ROLE,
        "evaluation_role": "retrospective-2026-development-diagnostic",
        "development_only": True,
        "future_selected_universe": True,
        "scientific_reporting_authorized": False,
        "promotion_authorized": False,
        "promotion_eligible_settings": [],
        "checkpoint_fold_index": checkpoint_fold_index,
        "checkpoint_role": checkpoint_role,
        "evidence_limits": [
            "future-selected-TOP2000-universe",
            "incomplete-delisting-history",
            "single-seed-17-no-seed-robustness",
            "retrospective-not-untouched-lockbox",
        ],
        "chronology": {
            "single_continuous_path": True,
            "fold_resets": False,
            "start_date": str(dates[0]),
            "end_date": str(dates[-1]),
            "decision_count": rows,
        },
        "cost_ladder": {
            "basis_points": list(TOP2000_M03R_V7_2026_COST_BASIS_POINTS),
            "decision_cost_basis_points": (
                TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS
            ),
            "primary_execution_mode": (
                "authoritative-20bp-closed-loop-chronological"
            ),
            "sensitivity_execution_mode": (
                "reprice-frozen-20bp-executed-turnover"
            ),
            "policy_reexecuted_for_sensitivity_rungs": False,
        },
        "dispersion_estimator": {
            "standard_deviation": "sample-ddof-1",
            "applies_to": [
                "portfolio-sharpe",
                "benchmark-sharpe",
                "information-ratio",
                "tracking-error",
            ],
        },
        "point_in_time_evidence": {
            "universe": {
                "status": "unavailable",
                "reason": "TOP2000-membership-was-selected-with-2026-information",
            },
            "official_factors": (
                {
                    "status": "available",
                    "factor_data_receipt_sha256": factor_data_receipt_sha256,
                    "factor_arrays_sha256": factor_arrays_sha256,
                    "manifest": {
                        **factor_manifest.semantics(),
                        "manifest_sha256": factor_manifest.manifest_sha256,
                    },
                }
                if factor_manifest is not None
                else {
                    "status": "unavailable",
                    "reason": "official-daily-ff5-momentum-receipt-not-supplied",
                    "factor_data_receipt_sha256": None,
                    "factor_arrays_sha256": None,
                    "manifest": None,
                }
            ),
        },
        "reversal_episode_evidence": {
            "status": "unavailable",
            "reason": (
                M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
                .reversal_episodes.unavailable_reason
            ),
            "receipt_sha256": None,
        },
        "cohort_survival_evidence": (
            {
                "status": "unavailable",
                "reason": (
                    "complete-score-origin-cohort-trajectory-panel-not-supplied"
                ),
                "receipt": None,
            }
            if cohort_rmst_receipt is None
            else {
                "status": "available",
                "reason": None,
                "receipt": cohort_rmst_receipt,
            }
        ),
        "source_receipts": {
            "training_completion_receipt_sha256": completion_hash,
            "data_manifest_sha256": data_hash,
            "chronology_receipt_sha256": chronology_hash,
            "execution_receipt_sha256": execution_hash,
        },
        "inference_plan": {
            **inference_plan.semantics(),
            "plan_sha256": inference_plan.plan_sha256,
        },
        "common_inputs_schema": TOP2000_M03R_V7_2026_COMMON_INPUT_SCHEMA,
        "common_inputs_sha256": common_inputs_sha256,
        "setting_inputs_schema": TOP2000_M03R_V7_2026_SETTING_INPUT_SCHEMA,
        "benchmark_turnover_by_cause": benchmark_turnover_receipt,
        "settings": setting_rows,
        "paired_contrasts": {
            "multiplicity_method": (
                "joint-max-absolute-centered-contrast-fwer-0.05"
            ),
            "raw_one_sided_p_value_method": (
                "null-centered-paired-bootstrap-upper-tail"
            ),
            "family_alpha": 0.05,
            "joint_primary_block_draws": True,
            "joint_max_absolute_centered_critical_value": (
                simultaneous_critical_value
            ),
            "rows": contrast_rows,
        },
    }
    return {**unsigned, "receipt_sha256": _sha256(unsigned)}


def _numbers_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-12)


def _require_probability(name: str, value: object) -> float:
    parsed = _require_finite_number(name, value)
    assert parsed is not None
    if not 0.0 <= parsed <= 1.0:
        raise Top2000M03RV72026EvaluationError(
            f"{name} must lie in [0, 1]"
        )
    return parsed


def _validate_regression_receipt(
    name: str,
    value: object,
    *,
    decision_count: int,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "alpha_daily",
        "alpha_annualized_arithmetic",
        "loadings",
        "observations",
    }:
        raise Top2000M03RV72026EvaluationError(
            f"{name} regression fields drifted"
        )
    alpha_daily = _require_finite_number(
        f"{name}.alpha_daily", value["alpha_daily"]
    )
    alpha_annualized = _require_finite_number(
        f"{name}.alpha_annualized_arithmetic",
        value["alpha_annualized_arithmetic"],
    )
    assert alpha_daily is not None and alpha_annualized is not None
    loadings = value["loadings"]
    expected_loadings = {
        TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME,
        *TOP2000_M03R_V7_2026_FACTOR_NAMES,
    }
    if (
        not isinstance(loadings, dict)
        or set(loadings) != expected_loadings
        or isinstance(value["observations"], bool)
        or value["observations"] != decision_count
        or not _numbers_close(alpha_annualized, 252.0 * alpha_daily)
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name} regression identity drifted"
        )
    for factor_name, loading in loadings.items():
        _require_finite_number(f"{name}.loadings[{factor_name}]", loading)


def _validate_active_beta_receipt(name: str, value: object) -> None:
    expected_keys = {
        "status",
        "reason",
        "active_market_beta",
        "bootstrap_standard_error",
        "equivalence_absolute_upper_bound",
        "constraint_maximum_absolute",
        "constraint_satisfied",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value["constraint_maximum_absolute"] != 0.10
    ):
        raise Top2000M03RV72026EvaluationError(f"{name} fields drifted")
    if value["status"] == "available":
        beta = _require_finite_number(
            f"{name}.active_market_beta", value["active_market_beta"]
        )
        standard_error = _require_finite_number(
            f"{name}.bootstrap_standard_error",
            value["bootstrap_standard_error"],
        )
        upper = _require_finite_number(
            f"{name}.equivalence_absolute_upper_bound",
            value["equivalence_absolute_upper_bound"],
        )
        assert beta is not None and standard_error is not None and upper is not None
        expected_upper = abs(beta) + NormalDist().inv_cdf(0.95) * standard_error
        if (
            value["reason"] is not None
            or standard_error < 0.0
            or upper < 0.0
            or not _numbers_close(upper, expected_upper)
            or type(value["constraint_satisfied"]) is not bool
            or value["constraint_satisfied"] is not (upper <= 0.10)
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{name} available-state invariants drifted"
            )
        return
    if value["status"] != "unavailable" or value["constraint_satisfied"] is not None:
        raise Top2000M03RV72026EvaluationError(
            f"{name} status invariants drifted"
        )
    reason = value["reason"]
    expected_null: tuple[object, ...]
    if reason == "incomplete-valid-bootstrap-beta-replicates":
        _require_finite_number(
            f"{name}.active_market_beta", value["active_market_beta"]
        )
        expected_null = (
            value["bootstrap_standard_error"],
            value["equivalence_absolute_upper_bound"],
        )
    elif reason in {
        "official-daily-ff5-momentum-evidence-not-supplied",
        "market-factor-variance-is-zero",
    }:
        expected_null = (
            value["active_market_beta"],
            value["bootstrap_standard_error"],
            value["equivalence_absolute_upper_bound"],
        )
    else:
        raise Top2000M03RV72026EvaluationError(
            f"{name} unavailable reason drifted"
        )
    if any(item is not None for item in expected_null):
        raise Top2000M03RV72026EvaluationError(
            f"{name} unavailable estimates must be null"
        )


def _validate_factor_attribution_receipt(
    name: str,
    value: object,
    *,
    decision_count: int,
) -> None:
    regression_names = (
        "portfolio_multifactor_regression",
        "benchmark_multifactor_regression",
        "active_multifactor_regression",
    )
    if not isinstance(value, dict) or set(value) != {
        "status",
        "reason",
        *regression_names,
    }:
        raise Top2000M03RV72026EvaluationError(f"{name} fields drifted")
    if value["status"] == "available":
        if value["reason"] is not None:
            raise Top2000M03RV72026EvaluationError(
                f"{name} available reason must be null"
            )
        for regression_name in regression_names:
            _validate_regression_receipt(
                f"{name}.{regression_name}",
                value[regression_name],
                decision_count=decision_count,
            )
        return
    if (
        value["status"] != "unavailable"
        or value["reason"]
        not in {
            "official-daily-ff5-momentum-evidence-not-supplied",
            "factor-regression-design-rank-deficient",
        }
        or any(value[regression_name] is not None for regression_name in regression_names)
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name} unavailable-state invariants drifted"
        )


def _finite_number_list(
    name: str,
    value: object,
    *,
    length: int,
    nonnegative: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise Top2000M03RV72026EvaluationError(
            f"{name} must be a list with length {length}"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        parsed = _require_finite_number(f"{name}[{index}]", item)
        assert parsed is not None
        if nonnegative and parsed < 0.0:
            raise Top2000M03RV72026EvaluationError(
                f"{name}[{index}] must be nonnegative"
            )
        result.append(parsed)
    return result


def _validate_telemetry_receipt(name: str, value: object) -> None:
    if not isinstance(value, dict) or value.get("status") not in {
        "available",
        "unavailable",
    }:
        raise Top2000M03RV72026EvaluationError(f"{name} status drifted")
    if value["status"] == "unavailable":
        if value != {
            "status": "unavailable",
            "reason": "raw-cause-age-action-hazard-telemetry-not-supplied",
        }:
            raise Top2000M03RV72026EvaluationError(
                f"{name} unavailable payload drifted"
            )
        return
    if set(value) != {
        "status",
        "projection_distance",
        "holding_snapshot_descriptive",
        "actions",
        "continuous_hazard",
    }:
        raise Top2000M03RV72026EvaluationError(
            f"{name} available fields drifted"
        )
    projection = value["projection_distance"]
    if not isinstance(projection, dict) or set(projection) != {
        "mean",
        "maximum",
        "p95",
    }:
        raise Top2000M03RV72026EvaluationError(
            f"{name}.projection_distance fields drifted"
        )
    projection_values: dict[str, float] = {}
    for key in ("mean", "maximum", "p95"):
        parsed_projection = _require_finite_number(
            f"{name}.projection_distance.{key}", projection[key]
        )
        assert parsed_projection is not None
        projection_values[key] = parsed_projection
    if (
        any(item < 0.0 for item in projection_values.values())
        or (
            projection_values["mean"] > projection_values["maximum"]
            and not _numbers_close(
                projection_values["mean"], projection_values["maximum"]
            )
        )
        or (
            projection_values["p95"] > projection_values["maximum"]
            and not _numbers_close(
                projection_values["p95"], projection_values["maximum"]
            )
        )
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name}.projection_distance values drifted"
        )

    holding = value["holding_snapshot_descriptive"]
    expected_holding_keys = {
        "source",
        "eligible_for_required_cohort_rmst",
        "snapshot_product_limit_rmst60_sessions",
        "snapshot_notional_survival",
        "discretionary_exit_notional_by_age",
        "discretionary_exit_notional_total",
        "notional_weighted_discretionary_sale_age",
        "median_discretionary_sale_age",
        "forced_exit_notional_by_cause_and_age",
    }
    if (
        not isinstance(holding, dict)
        or set(holding) != expected_holding_keys
        or holding["source"]
        != "date-by-age-notional-snapshot-product-limit"
        or holding["eligible_for_required_cohort_rmst"] is not False
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name}.holding_snapshot_descriptive fields drifted"
        )
    rmst = _require_finite_number(
        f"{name}.snapshot_product_limit_rmst60_sessions",
        holding["snapshot_product_limit_rmst60_sessions"],
    )
    if rmst is None or not 0.0 <= rmst <= 60.0:
        raise Top2000M03RV72026EvaluationError(
            f"{name}.snapshot RMST lies outside [0, 60]"
        )
    survival = holding["snapshot_notional_survival"]
    if not isinstance(survival, dict) or set(survival) != {"10", "20", "30"}:
        raise Top2000M03RV72026EvaluationError(
            f"{name}.snapshot survival inventory drifted"
        )
    survival_values = [
        _require_probability(f"{name}.snapshot_notional_survival[{age}]", survival[age])
        for age in ("10", "20", "30")
    ]
    if not survival_values[0] >= survival_values[1] >= survival_values[2]:
        raise Top2000M03RV72026EvaluationError(
            f"{name}.snapshot survival must be nonincreasing"
        )
    discretionary = _finite_number_list(
        f"{name}.discretionary_exit_notional_by_age",
        holding["discretionary_exit_notional_by_age"],
        length=TOP2000_M03R_V7_2026_AGE_BINS,
        nonnegative=True,
    )
    discretionary_total = _require_finite_number(
        f"{name}.discretionary_exit_notional_total",
        holding["discretionary_exit_notional_total"],
    )
    assert discretionary_total is not None
    if discretionary_total < 0.0 or not _numbers_close(
        discretionary_total, sum(discretionary)
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name}.discretionary exit total drifted"
        )
    weighted_age = _require_finite_number(
        f"{name}.notional_weighted_discretionary_sale_age",
        holding["notional_weighted_discretionary_sale_age"],
        nullable=True,
    )
    median_age = holding["median_discretionary_sale_age"]
    if discretionary_total <= 0.0:
        if weighted_age is not None or median_age is not None:
            raise Top2000M03RV72026EvaluationError(
                f"{name}.empty discretionary-age statistics drifted"
            )
    else:
        expected_weighted_age = sum(
            age * amount for age, amount in enumerate(discretionary)
        ) / discretionary_total
        expected_median = int(
            np.searchsorted(
                np.cumsum(np.asarray(discretionary, dtype=np.float64)),
                0.5 * discretionary_total,
                side="left",
            )
        )
        if (
            weighted_age is None
            or not 0.0 <= weighted_age <= 60.0
            or not _numbers_close(weighted_age, expected_weighted_age)
            or isinstance(median_age, bool)
            or not isinstance(median_age, int)
            or median_age != expected_median
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{name}.discretionary-age statistics drifted"
            )
    forced = holding["forced_exit_notional_by_cause_and_age"]
    if not isinstance(forced, dict) or set(forced) != set(
        TOP2000_M03R_V7_2026_FORCED_EXIT_CAUSES
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name}.forced-exit cause inventory drifted"
        )
    for cause, values in forced.items():
        _finite_number_list(
            f"{name}.forced_exit_notional_by_cause_and_age[{cause}]",
            values,
            length=TOP2000_M03R_V7_2026_AGE_BINS,
            nonnegative=True,
        )

    actions = value["actions"]
    if not isinstance(actions, dict) or set(actions) != {"counts", "frequencies"}:
        raise Top2000M03RV72026EvaluationError(f"{name}.actions fields drifted")
    counts = actions["counts"]
    frequencies = actions["frequencies"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(TOP2000_M03R_V7_2026_ACTIONS)
        or not isinstance(frequencies, dict)
        or set(frequencies) != set(TOP2000_M03R_V7_2026_ACTIONS)
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name}.action inventory drifted"
        )
    parsed_counts: dict[str, float] = {}
    for action, count in counts.items():
        parsed = _require_finite_number(f"{name}.actions.counts[{action}]", count)
        assert parsed is not None
        if parsed < 0.0 or parsed != math.floor(parsed):
            raise Top2000M03RV72026EvaluationError(
                f"{name}.action counts must be nonnegative integers"
            )
        parsed_counts[action] = parsed
    total_actions = sum(parsed_counts.values())
    for action, frequency in frequencies.items():
        if total_actions <= 0.0:
            if frequency is not None:
                raise Top2000M03RV72026EvaluationError(
                    f"{name}.empty action frequencies must be null"
                )
        else:
            parsed = _require_probability(
                f"{name}.actions.frequencies[{action}]", frequency
            )
            if not _numbers_close(parsed, parsed_counts[action] / total_actions):
                raise Top2000M03RV72026EvaluationError(
                    f"{name}.action frequencies do not reconcile"
                )

    hazard = value["continuous_hazard"]
    if not isinstance(hazard, dict) or hazard.get("status") not in {
        "available",
        "unavailable",
    }:
        raise Top2000M03RV72026EvaluationError(
            f"{name}.continuous_hazard status drifted"
        )
    if hazard["status"] == "unavailable":
        if hazard != {
            "status": "unavailable",
            "reason": "no-observed-continuous-hazard-actions",
        } or parsed_counts["CONTINUOUS"] != 0.0:
            raise Top2000M03RV72026EvaluationError(
                f"{name}.unavailable continuous hazard drifted"
            )
    else:
        if set(hazard) != {
            "status",
            "observations",
            "quantiles",
            "near_zero_fraction",
            "near_one_fraction",
        }:
            raise Top2000M03RV72026EvaluationError(
                f"{name}.available continuous hazard fields drifted"
            )
        observations = hazard["observations"]
        quantiles = hazard["quantiles"]
        if (
            isinstance(observations, bool)
            or not isinstance(observations, int)
            or observations <= 0
            or observations != int(parsed_counts["CONTINUOUS"])
            or not isinstance(quantiles, dict)
            or set(quantiles) != {"0.01", "0.1", "0.5", "0.9", "0.99"}
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{name}.continuous hazard observations drifted"
            )
        quantile_values = [
            _require_probability(f"{name}.continuous_hazard.quantiles[{key}]", quantiles[key])
            for key in ("0.01", "0.1", "0.5", "0.9", "0.99")
        ]
        if any(
            current < previous
            for previous, current in pairwise(quantile_values)
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{name}.continuous hazard quantiles must be nondecreasing"
            )
        _require_probability(
            f"{name}.continuous_hazard.near_zero_fraction",
            hazard["near_zero_fraction"],
        )
        _require_probability(
            f"{name}.continuous_hazard.near_one_fraction",
            hazard["near_one_fraction"],
        )


def _validate_turnover_summary(
    name: str,
    value: object,
    *,
    decision_count: int,
) -> None:
    if not isinstance(value, dict) or set(value) != set(
        TOP2000_M03R_V7_2026_TURNOVER_CAUSES
    ):
        raise Top2000M03RV72026EvaluationError(
            f"{name} turnover cause inventory drifted"
        )
    expected_costs = {
        str(cost) for cost in TOP2000_M03R_V7_2026_COST_BASIS_POINTS
    }
    for cause, row in value.items():
        if not isinstance(row, dict) or set(row) != {
            "total_one_way_turnover",
            "mean_daily_one_way_turnover",
            "cost_return_units",
        }:
            raise Top2000M03RV72026EvaluationError(
                f"{name}[{cause}] fields drifted"
            )
        total = _require_finite_number(
            f"{name}[{cause}].total_one_way_turnover",
            row["total_one_way_turnover"],
        )
        mean = _require_finite_number(
            f"{name}[{cause}].mean_daily_one_way_turnover",
            row["mean_daily_one_way_turnover"],
        )
        costs = row["cost_return_units"]
        if (
            total is None
            or mean is None
            or total < 0.0
            or mean < 0.0
            or not isinstance(costs, dict)
            or set(costs) != expected_costs
        ):
            raise Top2000M03RV72026EvaluationError(
                f"{name}[{cause}] turnover values drifted"
            )
        for cost, amount in costs.items():
            parsed = _require_finite_number(
                f"{name}[{cause}].cost_return_units[{cost}]", amount
            )
            expected_cost = total * int(cost) * 1.0e-4
            if (
                parsed is None
                or parsed < 0.0
                or not _numbers_close(parsed, expected_cost)
            ):
                raise Top2000M03RV72026EvaluationError(
                    f"{name}[{cause}] cost values drifted"
                )
        if not _numbers_close(mean, total / decision_count):
            raise Top2000M03RV72026EvaluationError(
                f"{name}[{cause}] mean turnover does not reconcile"
            )


def _validate_cost_ladder(name: str, value: object) -> None:
    expected_costs = {
        str(cost) for cost in TOP2000_M03R_V7_2026_COST_BASIS_POINTS
    }
    metric_keys = {
        "portfolio_cumulative_net_return",
        "benchmark_cumulative_net_return",
        "cumulative_active_return",
        "annualized_portfolio_arithmetic_mean_return",
        "annualized_benchmark_arithmetic_mean_return",
        "annualized_active_log_return",
        "annualized_tracking_error",
        "annualized_information_ratio",
        "portfolio_maximum_drawdown",
        "active_maximum_drawdown",
        "portfolio_sharpe",
        "benchmark_sharpe",
        "portfolio_minus_benchmark_sharpe",
    }
    nullable = {
        "annualized_information_ratio",
        "portfolio_sharpe",
        "benchmark_sharpe",
        "portfolio_minus_benchmark_sharpe",
    }
    if not isinstance(value, dict) or set(value) != expected_costs:
        raise Top2000M03RV72026EvaluationError(f"{name} cost ladder drifted")
    for cost, metrics in value.items():
        if not isinstance(metrics, dict) or set(metrics) != metric_keys:
            raise Top2000M03RV72026EvaluationError(
                f"{name}[{cost}] metric inventory drifted"
            )
        for metric, amount in metrics.items():
            parsed = _require_finite_number(
                f"{name}[{cost}].{metric}",
                amount,
                nullable=metric in nullable,
            )
            if metric == "annualized_tracking_error" and (
                parsed is None or parsed < 0.0
            ):
                raise Top2000M03RV72026EvaluationError(
                    f"{name}[{cost}] tracking error drifted"
                )


def _validate_setting_receipt_row(
    row: object,
    *,
    setting_index: int,
    setting_id: str,
    decision_count: int,
    factor_available: bool,
) -> None:
    expected_keys = {
        "setting_index",
        "setting_id",
        "checkpoint_sha256",
        "setting_inputs_sha256",
        "cost_ladder",
        "turnover_by_cause",
        "active_beta",
        "factor_attribution",
        "bootstrap",
        "telemetry",
        "cohort_rmst60",
        "reversal_episode_performance",
    }
    if (
        not isinstance(row, dict)
        or set(row) != expected_keys
        or row["setting_index"] != setting_index
        or row["setting_id"] != setting_id
    ):
        raise Top2000M03RV72026EvaluationError(
            f"setting receipt row {setting_index} drifted"
        )
    _require_digest("checkpoint_sha256", row["checkpoint_sha256"])
    _require_digest("setting_inputs_sha256", row["setting_inputs_sha256"])
    _validate_cost_ladder(f"settings[{setting_index}]", row["cost_ladder"])
    _validate_turnover_summary(
        f"settings[{setting_index}]",
        row["turnover_by_cause"],
        decision_count=decision_count,
    )
    _validate_active_beta_receipt(
        f"settings[{setting_index}].active_beta", row["active_beta"]
    )
    _validate_factor_attribution_receipt(
        f"settings[{setting_index}].factor_attribution",
        row["factor_attribution"],
        decision_count=decision_count,
    )
    beta = row["active_beta"]
    attribution = row["factor_attribution"]
    assert isinstance(beta, dict) and isinstance(attribution, dict)
    unavailable_factor_reason = (
        "official-daily-ff5-momentum-evidence-not-supplied"
    )
    if not factor_available and (
        beta["reason"] != unavailable_factor_reason
        or attribution["reason"] != unavailable_factor_reason
    ):
        raise Top2000M03RV72026EvaluationError(
            f"settings[{setting_index}] missing-factor status drifted"
        )
    if factor_available and (
        beta["reason"] == unavailable_factor_reason
        or attribution["reason"] == unavailable_factor_reason
    ):
        raise Top2000M03RV72026EvaluationError(
            f"settings[{setting_index}] available-factor status drifted"
        )
    bootstrap = row["bootstrap"]
    expected_blocks = {
        str(block)
        for block in (
            TOP2000_M03R_V7_2026_PRIMARY_BLOCK_LENGTH,
            *TOP2000_M03R_V7_2026_SENSITIVITY_BLOCK_LENGTHS,
        )
    }
    bootstrap_keys = {
        "active_return_annualized_log_lcb",
        "sharpe_difference_lcb",
        "active_multifactor_alpha_annualized_lcb",
        "valid_sharpe_replicates",
        "valid_active_alpha_replicates",
    }
    if not isinstance(bootstrap, dict) or set(bootstrap) != expected_blocks:
        raise Top2000M03RV72026EvaluationError(
            f"settings[{setting_index}] bootstrap inventory drifted"
        )
    for block, metrics in bootstrap.items():
        if not isinstance(metrics, dict) or set(metrics) != bootstrap_keys:
            raise Top2000M03RV72026EvaluationError(
                f"settings[{setting_index}].bootstrap[{block}] drifted"
            )
        _require_finite_number(
            f"settings[{setting_index}].bootstrap[{block}].active_lcb",
            metrics["active_return_annualized_log_lcb"],
        )
        sharpe_lcb = _require_finite_number(
            f"settings[{setting_index}].bootstrap[{block}].sharpe_difference_lcb",
            metrics["sharpe_difference_lcb"],
            nullable=True,
        )
        alpha_lcb = _require_finite_number(
            f"settings[{setting_index}].bootstrap[{block}].active_alpha_lcb",
            metrics["active_multifactor_alpha_annualized_lcb"],
            nullable=True,
        )
        for name in (
            "valid_sharpe_replicates",
            "valid_active_alpha_replicates",
        ):
            count = metrics[name]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count not in range(10_001)
            ):
                raise Top2000M03RV72026EvaluationError(
                    f"settings[{setting_index}].bootstrap[{block}].{name} drifted"
                )
        if (
            (metrics["valid_sharpe_replicates"] == 10_000)
            is not (sharpe_lcb is not None)
            or (metrics["valid_active_alpha_replicates"] == 10_000)
            is not (alpha_lcb is not None)
            or (
                not factor_available
                and (
                    metrics["valid_active_alpha_replicates"] != 0
                    or alpha_lcb is not None
                )
            )
        ):
            raise Top2000M03RV72026EvaluationError(
                f"settings[{setting_index}].bootstrap[{block}] availability drifted"
            )
    _validate_telemetry_receipt(
        f"settings[{setting_index}].telemetry", row["telemetry"]
    )


def validate_top2000_m03r_v7_2026_receipt(
    receipt: object,
    *,
    expected_receipt_sha256: str,
) -> None:
    """Validate semantics against an externally frozen receipt identity."""

    expected_hash = _require_digest(
        "expected_receipt_sha256", expected_receipt_sha256
    )

    if not isinstance(receipt, dict):
        raise Top2000M03RV72026EvaluationError("evaluation receipt must be a dict")
    required = {
        "schema",
        "protocol_generation",
        "design_id",
        "protocol_sha256",
        "source_training_protocol_generation",
        "source_training_design_id",
        "source_training_protocol_sha256",
        "data_role",
        "evaluation_role",
        "development_only",
        "future_selected_universe",
        "scientific_reporting_authorized",
        "promotion_authorized",
        "promotion_eligible_settings",
        "checkpoint_fold_index",
        "checkpoint_role",
        "evidence_limits",
        "chronology",
        "cost_ladder",
        "dispersion_estimator",
        "point_in_time_evidence",
        "reversal_episode_evidence",
        "cohort_survival_evidence",
        "source_receipts",
        "inference_plan",
        "common_inputs_schema",
        "common_inputs_sha256",
        "setting_inputs_schema",
        "benchmark_turnover_by_cause",
        "settings",
        "paired_contrasts",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise Top2000M03RV72026EvaluationError("evaluation receipt keys drifted")
    if (
        receipt["schema"] != TOP2000_M03R_V7_2026_EVALUATION_SCHEMA
        or receipt["protocol_generation"]
        != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION
        or receipt["design_id"]
        != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID
        or receipt["protocol_sha256"]
        != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
        or receipt["source_training_protocol_generation"]
        != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
        or receipt["source_training_design_id"]
        != M03R_SEED17_TOP2000_DESIGN_ID
        or receipt["source_training_protocol_sha256"]
        != M03R_SEED17_TOP2000_PROTOCOL_SHA256
        or receipt["data_role"] != M03R_SEED17_TOP2000_DATA_ROLE
        or receipt["evaluation_role"]
        != "retrospective-2026-development-diagnostic"
        or receipt["development_only"] is not True
        or receipt["future_selected_universe"] is not True
        or receipt["scientific_reporting_authorized"] is not False
        or receipt["promotion_authorized"] is not False
        or receipt["promotion_eligible_settings"] != []
    ):
        raise Top2000M03RV72026EvaluationError(
            "development-only authorization or protocol identity drifted"
        )
    expected_checkpoint_role = (
        "headline"
        if receipt["checkpoint_fold_index"] == 5
        else "cutoff-sensitivity"
    )
    if (
        isinstance(receipt["checkpoint_fold_index"], bool)
        or not isinstance(receipt["checkpoint_fold_index"], int)
        or receipt["checkpoint_fold_index"] not in range(6)
        or receipt["checkpoint_role"] != expected_checkpoint_role
    ):
        raise Top2000M03RV72026EvaluationError("checkpoint-fold identity drifted")
    if receipt["evidence_limits"] != [
        "future-selected-TOP2000-universe",
        "incomplete-delisting-history",
        "single-seed-17-no-seed-robustness",
        "retrospective-not-untouched-lockbox",
    ]:
        raise Top2000M03RV72026EvaluationError("evidence limits drifted")
    chronology = receipt["chronology"]
    if (
        not isinstance(chronology, dict)
        or set(chronology)
        != {
            "single_continuous_path",
            "fold_resets",
            "start_date",
            "end_date",
            "decision_count",
        }
        or chronology["single_continuous_path"] is not True
        or chronology["fold_resets"] is not False
        or isinstance(chronology["decision_count"], bool)
        or not isinstance(chronology["decision_count"], int)
        or chronology["decision_count"]
        != TOP2000_M03R_V7_2026_DECISION_COUNT
        or chronology["start_date"]
        != M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE
        or chronology["end_date"]
        != M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE
    ):
        raise Top2000M03RV72026EvaluationError("chronology semantics drifted")
    for name in ("start_date", "end_date"):
        try:
            parsed = date.fromisoformat(chronology[name])
        except (TypeError, ValueError) as exc:
            raise Top2000M03RV72026EvaluationError(
                "receipt chronology dates must be canonical 2026 dates"
            ) from exc
        if parsed.year != 2026 or parsed.isoformat() != chronology[name]:
            raise Top2000M03RV72026EvaluationError(
                "receipt chronology dates must be canonical 2026 dates"
            )
    cost_ladder = receipt["cost_ladder"]
    if cost_ladder != {
        "basis_points": list(TOP2000_M03R_V7_2026_COST_BASIS_POINTS),
        "decision_cost_basis_points": (
            TOP2000_M03R_V7_2026_DECISION_COST_BASIS_POINTS
        ),
        "primary_execution_mode": "authoritative-20bp-closed-loop-chronological",
        "sensitivity_execution_mode": "reprice-frozen-20bp-executed-turnover",
        "policy_reexecuted_for_sensitivity_rungs": False,
    }:
        raise Top2000M03RV72026EvaluationError("cost-ladder semantics drifted")
    if receipt["dispersion_estimator"] != {
        "standard_deviation": "sample-ddof-1",
        "applies_to": [
            "portfolio-sharpe",
            "benchmark-sharpe",
            "information-ratio",
            "tracking-error",
        ],
    }:
        raise Top2000M03RV72026EvaluationError(
            "dispersion estimator semantics drifted"
        )
    point_in_time = receipt["point_in_time_evidence"]
    if (
        not isinstance(point_in_time, dict)
        or set(point_in_time) != {"universe", "official_factors"}
        or point_in_time["universe"]
        != {
            "status": "unavailable",
            "reason": "TOP2000-membership-was-selected-with-2026-information",
        }
        or not isinstance(point_in_time["official_factors"], dict)
        or point_in_time["official_factors"].get("status")
        not in {"available", "unavailable"}
    ):
        raise Top2000M03RV72026EvaluationError("point-in-time evidence drifted")
    factor_evidence = point_in_time["official_factors"]
    if factor_evidence["status"] == "unavailable":
        if factor_evidence != {
            "status": "unavailable",
            "reason": "official-daily-ff5-momentum-receipt-not-supplied",
            "factor_data_receipt_sha256": None,
            "factor_arrays_sha256": None,
            "manifest": None,
        }:
            raise Top2000M03RV72026EvaluationError(
                "unavailable factor evidence drifted"
            )
    else:
        if set(factor_evidence) != {
            "status",
            "factor_data_receipt_sha256",
            "factor_arrays_sha256",
            "manifest",
        }:
            raise Top2000M03RV72026EvaluationError(
                "available factor evidence keys drifted"
            )
        _require_digest(
            "factor_data_receipt_sha256",
            factor_evidence["factor_data_receipt_sha256"],
        )
        _require_digest(
            "factor_arrays_sha256",
            factor_evidence["factor_arrays_sha256"],
        )
        manifest = factor_evidence.get("manifest")
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema",
            "source",
            "five_factor_source_file_sha256",
            "momentum_source_file_sha256",
            "source_receipt_sha256",
            "coverage_receipt_sha256",
            "exact_array_receipt_sha256",
            "factor_names",
            "market_factor_name",
            "risk_free_name",
            "frequency",
            "return_unit",
            "date_join",
            "missing_value_policy",
            "factor_set_defined_before_2026_access",
            "evaluation_returns_used_to_define_factor_set",
            "manifest_sha256",
        }:
            raise Top2000M03RV72026EvaluationError("factor manifest payload drifted")
        manifest_unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        for name in (
            "five_factor_source_file_sha256",
            "momentum_source_file_sha256",
            "source_receipt_sha256",
            "coverage_receipt_sha256",
            "exact_array_receipt_sha256",
        ):
            _require_digest(name, manifest[name])
        if _require_digest(
            "factor manifest_sha256", manifest["manifest_sha256"]
        ) != _sha256(manifest_unsigned):
            raise Top2000M03RV72026EvaluationError("factor manifest hash mismatch")
        if (
            manifest["schema"] != TOP2000_M03R_V7_2026_FACTOR_MANIFEST_SCHEMA
            or manifest["source"] != TOP2000_M03R_V7_2026_FACTOR_SOURCE
            or tuple(manifest["factor_names"])
            != TOP2000_M03R_V7_2026_FACTOR_NAMES
            or manifest["market_factor_name"]
            != TOP2000_M03R_V7_2026_MARKET_FACTOR_NAME
            or manifest["risk_free_name"] != "RF"
            or manifest["frequency"] != "daily"
            or manifest["return_unit"] != "decimal-simple-return"
            or manifest["date_join"] != "exact-date-inner-join"
            or manifest["missing_value_policy"] != "no-imputation"
            or manifest["factor_set_defined_before_2026_access"] is not True
            or manifest["evaluation_returns_used_to_define_factor_set"]
            is not False
        ):
            raise Top2000M03RV72026EvaluationError("factor semantics drifted")
    reversal_evidence = receipt["reversal_episode_evidence"]
    if not isinstance(reversal_evidence, dict) or set(reversal_evidence) != {
        "status",
        "reason",
        "receipt_sha256",
    }:
        raise Top2000M03RV72026EvaluationError(
            "reversal-episode evidence drifted"
        )
    if reversal_evidence != {
        "status": "unavailable",
        "reason": (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
            .reversal_episodes.unavailable_reason
        ),
        "receipt_sha256": None,
    }:
        raise Top2000M03RV72026EvaluationError(
            "unavailable reversal-episode evidence drifted"
        )
    cohort_evidence = receipt["cohort_survival_evidence"]
    if not isinstance(cohort_evidence, dict) or set(cohort_evidence) != {
        "status",
        "reason",
        "receipt",
    }:
        raise Top2000M03RV72026EvaluationError(
            "cohort-survival evidence drifted"
        )
    if cohort_evidence["status"] == "unavailable":
        if cohort_evidence != {
            "status": "unavailable",
            "reason": (
                "complete-score-origin-cohort-trajectory-panel-not-supplied"
            ),
            "receipt": None,
        }:
            raise Top2000M03RV72026EvaluationError(
                "unavailable cohort-survival evidence drifted"
            )
    elif cohort_evidence["status"] == "available":
        if cohort_evidence["reason"] is not None:
            raise Top2000M03RV72026EvaluationError(
                "available cohort-survival evidence has a blocker"
            )
        from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
            Top2000M03RV72026CohortSurvivalError,
            validate_top2000_m03r_v7_2026_cohort_rmst60_receipt,
        )

        try:
            validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(
                cohort_evidence["receipt"]
            )
        except Top2000M03RV72026CohortSurvivalError as exc:
            raise Top2000M03RV72026EvaluationError(
                "cohort-survival receipt failed validation"
            ) from exc
    else:
        raise Top2000M03RV72026EvaluationError(
            "cohort-survival evidence status drifted"
        )
    source_receipts = receipt["source_receipts"]
    if not isinstance(source_receipts, dict) or set(source_receipts) != {
        "training_completion_receipt_sha256",
        "data_manifest_sha256",
        "chronology_receipt_sha256",
        "execution_receipt_sha256",
    }:
        raise Top2000M03RV72026EvaluationError("source receipt inventory drifted")
    if (
        receipt["common_inputs_schema"]
        != TOP2000_M03R_V7_2026_COMMON_INPUT_SCHEMA
        or receipt["setting_inputs_schema"]
        != TOP2000_M03R_V7_2026_SETTING_INPUT_SCHEMA
    ):
        raise Top2000M03RV72026EvaluationError("source-array schema drifted")
    _validate_turnover_summary(
        "benchmark_turnover_by_cause",
        receipt["benchmark_turnover_by_cause"],
        decision_count=chronology["decision_count"],
    )
    plan_payload = receipt["inference_plan"]
    if not isinstance(plan_payload, dict) or set(plan_payload) != {
        "schema",
        "primary_block_length_trading_sessions",
        "sensitivity_block_lengths_trading_sessions",
        "bootstrap_replicates",
        "bootstrap_seed_sha256",
        "one_sided_alpha",
        "quantile_method",
        "resampling",
        "multiplicity_method",
        "raw_one_sided_p_value_method",
        "dispersion_standard_deviation_degrees_of_freedom",
        "cohort_rmst_resampling",
        "cohort_rmst_block_length_origin_sessions",
        "same_origin_block_draws_for_all_settings",
        "date_by_age_snapshot_survival_is_descriptive_only",
        "contrast_family",
        "joint_draws_across_settings_metrics_and_contrasts",
        "same_date_indices_for_all_cost_rungs",
        "one_joint_family_per_checkpoint_fold",
        "checkpoint_fold_paths_pooled",
        "plan_sha256",
    }:
        raise Top2000M03RV72026EvaluationError("inference plan payload drifted")
    plan_unsigned = {
        key: value for key, value in plan_payload.items() if key != "plan_sha256"
    }
    if (
        _require_digest("plan_sha256", plan_payload["plan_sha256"])
        != _sha256(plan_unsigned)
        or plan_payload["primary_block_length_trading_sessions"] != 21
        or plan_payload["sensitivity_block_lengths_trading_sessions"] != [10, 30]
        or plan_payload["bootstrap_replicates"] != 10_000
        or plan_payload["bootstrap_seed_sha256"]
        != M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256
        or plan_payload["one_sided_alpha"] != 0.05
        or plan_payload["quantile_method"] != "inverted_cdf"
        or plan_payload["resampling"]
        != "joint-date-index-circular-moving-block"
        or plan_payload["multiplicity_method"]
        != "joint-max-absolute-centered-contrast-fwer-0.05"
        or plan_payload["raw_one_sided_p_value_method"]
        != "null-centered-paired-bootstrap-upper-tail"
        or plan_payload["dispersion_standard_deviation_degrees_of_freedom"] != 1
        or plan_payload["cohort_rmst_resampling"]
        != "joint-complete-origin-trajectory-circular-block-by-entry-date"
        or plan_payload["cohort_rmst_block_length_origin_sessions"] != 21
        or plan_payload["same_origin_block_draws_for_all_settings"] is not True
        or plan_payload["date_by_age_snapshot_survival_is_descriptive_only"]
        is not True
        or plan_payload["contrast_family"]
        != [list(row) for row in TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS]
        or plan_payload["joint_draws_across_settings_metrics_and_contrasts"]
        is not True
        or plan_payload["same_date_indices_for_all_cost_rungs"] is not True
        or plan_payload["one_joint_family_per_checkpoint_fold"] is not True
        or plan_payload["checkpoint_fold_paths_pooled"] is not False
    ):
        raise Top2000M03RV72026EvaluationError("inference plan semantics drifted")
    if (
        not isinstance(receipt["settings"], list)
        or tuple(row.get("setting_id") for row in receipt["settings"])
        != M03R_SEED17_TOP2000_SETTING_IDS
    ):
        raise Top2000M03RV72026EvaluationError("setting receipt inventory drifted")
    for name in source_receipts:
        _require_digest(name, source_receipts[name])
    _require_digest("common_inputs_sha256", receipt["common_inputs_sha256"])
    for setting_index, (setting_id, row) in enumerate(
        zip(M03R_SEED17_TOP2000_SETTING_IDS, receipt["settings"], strict=True)
    ):
        _validate_setting_receipt_row(
            row,
            setting_index=setting_index,
            setting_id=setting_id,
            decision_count=chronology["decision_count"],
            factor_available=factor_evidence["status"] == "available",
        )
        assert isinstance(row, dict)
        if row.get("reversal_episode_performance") != {
            "status": "unavailable",
            "reason": (
                M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
                .reversal_episodes.unavailable_reason
            ),
        }:
            raise Top2000M03RV72026EvaluationError(
                "setting reversal-episode metric drifted"
            )
        if cohort_evidence["status"] == "unavailable":
            expected_cohort_row: dict[str, Any] = {
                "status": "unavailable",
                "reason": (
                    "complete-score-origin-cohort-trajectory-panel-not-supplied"
                ),
                "rmst60_sessions": None,
                "uncertainty": {
                    "status": "unavailable",
                    "reason": (
                        "complete-score-origin-cohort-trajectory-panel-not-supplied"
                    ),
                },
                "entry_units": None,
                "trajectory_receipt_sha256": None,
            }
        else:
            cohort_receipt = cohort_evidence["receipt"]
            assert isinstance(cohort_receipt, dict)
            expected_cohort_row = next(
                item
                for item in cohort_receipt["rows"]
                if item["setting_id"] == row.get("setting_id")
            )
        if row.get("cohort_rmst60") != expected_cohort_row:
            raise Top2000M03RV72026EvaluationError(
                "setting cohort RMST metric drifted"
            )
    paired = receipt["paired_contrasts"]
    if (
        not isinstance(paired, dict)
        or set(paired)
        != {
            "multiplicity_method",
            "raw_one_sided_p_value_method",
            "family_alpha",
            "joint_primary_block_draws",
            "joint_max_absolute_centered_critical_value",
            "rows",
        }
        or paired.get("multiplicity_method")
        != "joint-max-absolute-centered-contrast-fwer-0.05"
        or paired.get("raw_one_sided_p_value_method")
        != "null-centered-paired-bootstrap-upper-tail"
        or paired.get("family_alpha") != 0.05
        or paired.get("joint_primary_block_draws") is not True
        or not isinstance(paired.get("rows"), list)
        or tuple(row.get("contrast_id") for row in paired["rows"])
        != tuple(row[0] for row in TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS)
    ):
        raise Top2000M03RV72026EvaluationError(
            "paired contrast or multiplicity family drifted"
        )
    expected_contrast_keys = {
        "contrast_id",
        "numerator_setting_id",
        "denominator_setting_id",
        "estimand",
        "estimate",
        "one_sided_95pct_lcb",
        "raw_one_sided_null_centered_bootstrap_p_value",
        "simultaneous_fwer95_lcb",
        "simultaneous_fwer95_ucb",
        "multiplicity_adjusted_p_value",
        "multiplicity_reject_at_family_alpha_0_05",
    }
    if any(set(row) != expected_contrast_keys for row in paired["rows"]):
        raise Top2000M03RV72026EvaluationError(
            "paired contrast row fields drifted"
        )
    critical = _require_finite_number(
        "joint_max_absolute_centered_critical_value",
        paired["joint_max_absolute_centered_critical_value"],
    )
    assert critical is not None
    if critical < 0.0:
        raise Top2000M03RV72026EvaluationError(
            "paired contrast critical value must be nonnegative"
        )
    for row, (contrast_id, numerator, denominator) in zip(
        paired["rows"],
        TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS,
        strict=True,
    ):
        if (
            row["contrast_id"] != contrast_id
            or row["numerator_setting_id"] != numerator
            or row["denominator_setting_id"] != denominator
            or row["estimand"]
            != "20bp-annualized-active-log-return-difference"
            or type(row["multiplicity_reject_at_family_alpha_0_05"])
            is not bool
        ):
            raise Top2000M03RV72026EvaluationError(
                "paired contrast identity drifted"
            )
        parsed_values: dict[str, float] = {}
        for name in (
            "estimate",
            "one_sided_95pct_lcb",
            "simultaneous_fwer95_lcb",
            "simultaneous_fwer95_ucb",
        ):
            parsed_metric = _require_finite_number(
                f"paired_contrasts[{contrast_id}].{name}", row[name]
            )
            assert parsed_metric is not None
            parsed_values[name] = parsed_metric
        for name in (
            "raw_one_sided_null_centered_bootstrap_p_value",
            "multiplicity_adjusted_p_value",
        ):
            probability = _require_finite_number(
                f"paired_contrasts[{contrast_id}].{name}", row[name]
            )
            if probability is None or not 0.0 <= probability <= 1.0:
                raise Top2000M03RV72026EvaluationError(
                    f"paired contrast probability {name} drifted"
                )
        estimate = parsed_values["estimate"]
        simultaneous_lcb = parsed_values["simultaneous_fwer95_lcb"]
        simultaneous_ucb = parsed_values["simultaneous_fwer95_ucb"]
        if (
            not _numbers_close(simultaneous_lcb, estimate - critical)
            or not _numbers_close(simultaneous_ucb, estimate + critical)
            or row["multiplicity_reject_at_family_alpha_0_05"]
            is not (simultaneous_lcb > 0.0)
        ):
            raise Top2000M03RV72026EvaluationError(
                f"paired contrast bounds or rejection {contrast_id} drifted"
            )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt_hash = _require_digest("receipt_sha256", receipt["receipt_sha256"])
    if receipt_hash != expected_hash or receipt_hash != _sha256(unsigned):
        raise Top2000M03RV72026EvaluationError("evaluation receipt hash mismatch")


def write_top2000_m03r_v7_2026_receipt(
    path: str | Path,
    receipt: object,
) -> str:
    """Publish one canonical receipt without permitting overwrite."""

    if not isinstance(receipt, dict):
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt must be a dict"
        )
    expected_receipt_sha256 = _require_digest(
        "receipt_sha256", receipt.get("receipt_sha256")
    )
    validate_top2000_m03r_v7_2026_receipt(
        receipt,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    payload = _canonical_json(receipt)
    if len(payload) > _EVALUATION_RECEIPT_MAX_BYTES:
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt exceeds the bounded artifact size"
        )
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable evaluation receipt {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable evaluation receipt {destination}"
        ) from exc
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("evaluation receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    # A failed write is intentionally preserved rather than unlinking a path
    # that another process could have replaced.  The loader rejects partial
    # bytes against the externally frozen file digest.
    return hashlib.sha256(payload).hexdigest()


def load_top2000_m03r_v7_2026_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Load a canonical receipt against external file and semantic hashes."""

    file_hash = _require_digest("expected_file_sha256", expected_file_sha256)
    receipt_hash = _require_digest(
        "expected_receipt_sha256", expected_receipt_sha256
    )
    source = Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt file identity or size drifted"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _EVALUATION_RECEIPT_MAX_BYTES
        ):
            raise Top2000M03RV72026EvaluationError(
                "evaluation receipt file identity or size drifted"
            )
        chunks: list[bytes] = []
        remaining = _EVALUATION_RECEIPT_MAX_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(payload) <= 0
        or len(payload) > _EVALUATION_RECEIPT_MAX_BYTES
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or len(payload) != after.st_size
        or hashlib.sha256(payload).hexdigest() != file_hash
    ):
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt file identity or size drifted"
        )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt is not canonical JSON"
        ) from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != payload:
        raise Top2000M03RV72026EvaluationError(
            "evaluation receipt bytes are not canonical JSON"
        )
    validate_top2000_m03r_v7_2026_receipt(
        decoded,
        expected_receipt_sha256=receipt_hash,
    )
    return decoded


__all__ = [
    "TOP2000_M03R_V7_2026_ACTIONS",
    "TOP2000_M03R_V7_2026_COST_BASIS_POINTS",
    "TOP2000_M03R_V7_2026_DECISION_COUNT",
    "TOP2000_M03R_V7_2026_EVALUATION_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_NAMES",
    "TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS",
    "TOP2000_M03R_V7_2026_SCORE_DATE_AXIS",
    "Top2000M03RV72026EvaluationError",
    "Top2000M03RV72026FactorManifest",
    "Top2000M03RV72026InferencePlan",
    "Top2000M03RV72026Telemetry",
    "build_top2000_m03r_v7_2026_factor_manifest",
    "build_top2000_m03r_v7_2026_inference_plan",
    "evaluate_top2000_m03r_v7_2026_panel",
    "load_top2000_m03r_v7_2026_receipt",
    "validate_top2000_m03r_v7_2026_receipt",
    "write_top2000_m03r_v7_2026_receipt",
]
