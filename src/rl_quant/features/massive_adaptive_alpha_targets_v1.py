"""Nonauthorizing seven-bucket economic targets for adaptive Massive alpha.

This module deliberately does not extend the frozen H63 profitability target
authority.  It consumes one explicitly receipted 127-mark economic path per
security, computes the protocol's seven contiguous and non-overlapping return
buckets, and applies one origin-time weighted factor operator on common
complete support.  Corporate actions and terminal outcomes must already be
represented in the economic values; no price-only fallback is permitted here.

The artifact is an engineering boundary.  It cannot authorize historical
training, profitability reporting, lockbox access, or reinforcement learning
until a later package-owned source reconciliation replays the 127 marks from
the frozen Massive authorities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.alpha.targets import (
    OriginExposurePanel,
    OriginResidualOperator,
    apply_origin_residual_operator,
    build_origin_residual_operator,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA = (
    "rl-quant.massive-adaptive-economic-path-v1"
)
MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SCHEMA = (
    "rl-quant.massive-adaptive-alpha-targets-v1"
)
MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "marks": 127,
        "bucket_intervals": tuple(
            (
                bucket.bucket_id,
                bucket.start_offset_sessions,
                bucket.end_offset_sessions,
            )
            for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
        ),
        "return": "non-overlapping-economic-total-simple-return",
        "support": "complete-127-mark-common-cross-sectional-support",
        "terminal": "exact-carry-or-declared-conservative-total-loss",
        "factor_operator": "origin-available-weighted-qr-common-support",
        "factor_target": "weighted-mean-fitted-component",
        "duration_prior": False,
        "downstream_authorization": False,
    }
)
_MARK_COUNT = 127
_BUCKET_COUNT = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
_ALLOWED_MARK_KINDS = frozenset(
    {"market", "terminal-disposition", "validated-fallback", "missing"}
)


class MassiveAdaptiveAlphaTargetsV1Error(ValueError):
    """An adaptive economic path or target decomposition is malformed."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveAlphaTargetsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MassiveAdaptiveAlphaTargetsV1Error(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicPathV1:
    """One post-fill economic-value path through session offset 126."""

    security_id: str
    decision_at_ms: int
    fill_at_ms: int
    economic_at_ms: tuple[int, ...]
    available_at_ms: tuple[int, ...]
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    terminal: tuple[bool, ...]
    mark_kinds: tuple[str, ...]
    mark_receipts: tuple[str, ...]
    unresolved_terminal_fallback_session_offset: int | None
    conservative_total_loss_fallback: bool
    source_economic_path_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        coordinates = (
            self.economic_at_ms,
            self.available_at_ms,
            self.values,
            self.valid,
            self.terminal,
            self.mark_kinds,
            self.mark_receipts,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA
            or not self.security_id
            or self.security_id != self.security_id.strip()
            or any(len(values) != _MARK_COUNT for values in coordinates)
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or isinstance(self.fill_at_ms, bool)
            or not isinstance(self.fill_at_ms, int)
            or not 0 <= self.decision_at_ms < self.fill_at_ms
            or self.economic_at_ms[0] != self.fill_at_ms
            or self.economic_at_ms != tuple(sorted(set(self.economic_at_ms)))
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive economic path identity or shape differs"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.economic_at_ms + self.available_at_ms
        ) or any(
            available < economic
            for economic, available in zip(
                self.economic_at_ms, self.available_at_ms, strict=True
            )
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive economic path chronology differs"
            )
        if any(not isinstance(value, bool) for value in self.valid + self.terminal):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive economic path masks differ"
            )
        if any(kind not in _ALLOWED_MARK_KINDS for kind in self.mark_kinds):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive economic mark kind differs"
            )
        terminal_seen = False
        terminal_value = 0.0
        for value, valid, terminal, kind, receipt in zip(
            self.values,
            self.valid,
            self.terminal,
            self.mark_kinds,
            self.mark_receipts,
            strict=True,
        ):
            normalized = _finite("adaptive economic value", value)
            _digest("adaptive economic mark", receipt)
            if normalized < 0.0 or (not valid and normalized != 0.0):
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "adaptive economic value or missing payload differs"
                )
            if not valid and kind != "missing":
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "an invalid adaptive mark must be explicitly missing"
                )
            if valid and kind == "missing":
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "a valid adaptive mark cannot be missing"
                )
            if terminal:
                if not valid or kind != "terminal-disposition":
                    raise MassiveAdaptiveAlphaTargetsV1Error(
                        "adaptive terminal mark differs"
                    )
                if not terminal_seen:
                    terminal_seen = True
                    terminal_value = normalized
                elif normalized != terminal_value:
                    raise MassiveAdaptiveAlphaTargetsV1Error(
                        "adaptive terminal value is not carried"
                    )
            elif terminal_seen:
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "adaptive economic path stops carrying a terminal outcome"
                )
        fallback = self.unresolved_terminal_fallback_session_offset
        if fallback is not None and (
            isinstance(fallback, bool)
            or not isinstance(fallback, int)
            or not 1 <= fallback <= 126
            or not all(self.terminal[fallback:])
            or any(
                value != self.values[fallback]
                for value in self.values[fallback:]
            )
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive conservative terminal fallback differs"
            )
        if self.conservative_total_loss_fallback != (fallback is not None):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive conservative fallback state differs"
            )
        _digest(
            "adaptive economic path source",
            self.source_economic_path_receipt_sha256,
        )
        _digest("adaptive economic path", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive economic path receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaTargetRowV1:
    security_id: str
    raw_bucket_returns: tuple[float, ...]
    factor_component_returns: tuple[float, ...]
    residual_bucket_returns: tuple[float, ...]
    economic_valid_by_bucket: tuple[bool, ...]
    training_valid_by_bucket: tuple[bool, ...]
    terminal_by_bucket: tuple[bool, ...]
    conservative_fallback_by_bucket: tuple[bool, ...]
    economic_path_receipt_sha256: str
    residual_operator_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        values = (
            self.raw_bucket_returns,
            self.factor_component_returns,
            self.residual_bucket_returns,
            self.economic_valid_by_bucket,
            self.training_valid_by_bucket,
            self.terminal_by_bucket,
            self.conservative_fallback_by_bucket,
        )
        if (
            not self.security_id
            or any(len(row) != _BUCKET_COUNT for row in values)
            or any(
                not isinstance(value, bool)
                for row in values[3:]
                for value in row
            )
            or any(
                training and not economic
                for training, economic in zip(
                    self.training_valid_by_bucket,
                    self.economic_valid_by_bucket,
                    strict=True,
                )
            )
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive target row shape or masks differ"
            )
        for index, (raw, factor, residual, training_valid) in enumerate(
            zip(
                self.raw_bucket_returns,
                self.factor_component_returns,
                self.residual_bucket_returns,
                self.training_valid_by_bucket,
                strict=True,
            )
        ):
            raw_value = _finite(f"raw bucket {index}", raw)
            factor_value = _finite(f"factor bucket {index}", factor)
            residual_value = _finite(f"residual bucket {index}", residual)
            if training_valid:
                if raw_value < -1.0 or not math.isclose(
                    raw_value,
                    factor_value + residual_value,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                ):
                    raise MassiveAdaptiveAlphaTargetsV1Error(
                        "adaptive target decomposition differs"
                    )
            elif (
                factor_value != 0.0
                or residual_value != 0.0
                or (
                    not self.economic_valid_by_bucket[index]
                    and raw_value != 0.0
                )
            ):
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "invalid adaptive target payload is nonzero"
                )
            if (
                self.conservative_fallback_by_bucket[index]
                and not self.terminal_by_bucket[index]
            ):
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "adaptive fallback bucket is not terminal"
                )
        _digest("adaptive target path", self.economic_path_receipt_sha256)
        _digest("adaptive residual operator", self.residual_operator_receipt_sha256)
        _digest("adaptive target row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive target row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaTargetsV1:
    decision_session_date: str
    decision_at_ms: int
    built_at_ms: int
    security_ids: tuple[str, ...]
    bucket_ids: tuple[str, ...]
    rows: tuple[MassiveAdaptiveAlphaTargetRowV1, ...]
    factor_return_target: tuple[float, ...]
    factor_valid: tuple[bool, ...]
    common_training_asset_mask: tuple[bool, ...]
    valid_counts_by_bucket: tuple[int, ...]
    maximum_weighted_exposure_error_by_bucket: tuple[float, ...]
    residual_operator: OriginResidualOperator
    economic_path_inventory_sha256: str
    target_row_inventory_sha256: str
    origin_receipt_sha256: str
    economic_accounting_receipt_sha256: str
    fill_source_receipt_sha256: str
    terminal_authority_receipt_sha256: str
    economic_coverage_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_paths_replayed: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        expected_buckets = tuple(
            bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SCHEMA
            or not self.decision_session_date
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or isinstance(self.built_at_ms, bool)
            or not isinstance(self.built_at_ms, int)
            or self.built_at_ms <= self.decision_at_ms
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or not self.security_ids
            or self.bucket_ids != expected_buckets
            or tuple(row.security_id for row in self.rows) != self.security_ids
            or len(self.factor_return_target) != _BUCKET_COUNT
            or len(self.factor_valid) != _BUCKET_COUNT
            or len(self.common_training_asset_mask) != len(self.security_ids)
            or len(self.valid_counts_by_bucket) != _BUCKET_COUNT
            or len(self.maximum_weighted_exposure_error_by_bucket) != _BUCKET_COUNT
            or any(not isinstance(value, bool) for value in self.factor_valid)
            or any(
                not isinstance(value, bool)
                for value in self.common_training_asset_mask
            )
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SOURCE_SHA256
            or any(
                (
                    self.source_paths_replayed,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                    self.reinforcement_learning_authorized,
                )
            )
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive target artifact identity or authorization differs"
            )
        self.residual_operator.validate()
        if (
            self.residual_operator.origin_at_ms != self.decision_at_ms
            or self.residual_operator.asset_ids != self.security_ids
            or self.residual_operator.qualified_asset_mask
            != self.common_training_asset_mask
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive residual operator support differs"
            )
        for asset_index, row in enumerate(self.rows):
            row.validate()
            if (
                row.residual_operator_receipt_sha256
                != self.residual_operator.receipt_sha256
                or row.training_valid_by_bucket
                != (self.common_training_asset_mask[asset_index],) * _BUCKET_COUNT
            ):
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "adaptive target row operator differs"
                )
        expected_counts = tuple(
            sum(row.training_valid_by_bucket[index] for row in self.rows)
            for index in range(_BUCKET_COUNT)
        )
        expected_common = (sum(self.common_training_asset_mask),) * _BUCKET_COUNT
        if (
            self.valid_counts_by_bucket != expected_counts
            or self.valid_counts_by_bucket != expected_common
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive common target support differs"
            )
        if any(
            not math.isfinite(_finite("factor return target", value))
            for value in self.factor_return_target
        ) or any(
            not math.isfinite(_finite("weighted exposure error", value))
            or value > 2.0e-10
            for value in self.maximum_weighted_exposure_error_by_bucket
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive factor target or orthogonality differs"
            )
        if self.factor_valid != (True,) * _BUCKET_COUNT:
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive common support lacks a factor target"
            )
        raw_array = np.asarray(
            [row.raw_bucket_returns for row in self.rows], dtype=np.float64
        )
        factor_array = np.asarray(
            [row.factor_component_returns for row in self.rows], dtype=np.float64
        )
        residual_array = np.asarray(
            [row.residual_bucket_returns for row in self.rows], dtype=np.float64
        )
        mask = np.asarray(self.common_training_asset_mask, dtype=np.bool_)
        weights = np.asarray(
            self.residual_operator.qualified_weights, dtype=np.float64
        )
        for bucket_index in range(_BUCKET_COUNT):
            replay = apply_origin_residual_operator(
                raw_array[:, bucket_index], self.residual_operator
            )
            expected_residual = np.asarray(replay.values, dtype=np.float64)
            expected_factor = np.zeros(len(self.rows), dtype=np.float64)
            expected_factor[mask] = (
                raw_array[mask, bucket_index] - expected_residual[mask]
            )
            expected_factor_target = float(
                np.average(expected_factor[mask], weights=weights)
            )
            if (
                not np.allclose(
                    residual_array[:, bucket_index],
                    expected_residual,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                or not np.allclose(
                    factor_array[:, bucket_index],
                    expected_factor,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                or not math.isclose(
                    self.factor_return_target[bucket_index],
                    expected_factor_target,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    self.maximum_weighted_exposure_error_by_bucket[bucket_index],
                    replay.maximum_weighted_exposure_error,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                )
            ):
                raise MassiveAdaptiveAlphaTargetsV1Error(
                    "adaptive target factor replay differs"
                )
        if self.economic_path_inventory_sha256 != semantic_sha256(
            tuple(row.economic_path_receipt_sha256 for row in self.rows)
        ) or self.target_row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive target inventory differs"
            )
        for name in (
            "economic_path_inventory_sha256",
            "target_row_inventory_sha256",
            "origin_receipt_sha256",
            "economic_accounting_receipt_sha256",
            "fill_source_receipt_sha256",
            "terminal_authority_receipt_sha256",
            "economic_coverage_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveAlphaTargetsV1Error(
                "adaptive target semantic receipt differs"
            )
        assert_no_adaptive_hold_semantics(self)


def _bucket_return(
    path: MassiveAdaptiveEconomicPathV1,
    *,
    start: int,
    end: int,
) -> tuple[float, bool, bool, bool]:
    complete = all(path.valid[start : end + 1])
    terminal = any(path.terminal[start + 1 : end + 1])
    fallback = (
        path.unresolved_terminal_fallback_session_offset is not None
        and start < path.unresolved_terminal_fallback_session_offset <= end
    )
    if not complete:
        return 0.0, False, terminal, fallback
    start_value = path.values[start]
    end_value = path.values[end]
    if start_value > 0.0:
        return end_value / start_value - 1.0, True, terminal, fallback
    if (
        start_value == 0.0
        and end_value == 0.0
        and all(path.terminal[start : end + 1])
    ):
        return 0.0, True, terminal, fallback
    return 0.0, False, terminal, fallback


def build_massive_adaptive_alpha_targets_v1(
    *,
    decision_session_date: str,
    built_at_ms: int,
    paths: Sequence[MassiveAdaptiveEconomicPathV1],
    exposure_panel: OriginExposurePanel,
    origin_receipt_sha256: str,
    economic_accounting_receipt_sha256: str,
    fill_source_receipt_sha256: str,
    terminal_authority_receipt_sha256: str,
    economic_coverage_receipt_sha256: str,
) -> MassiveAdaptiveAlphaTargetsV1:
    """Build non-overlapping raw and factor-residual target term structures."""

    exposure_panel.validate()
    ordered = tuple(paths)
    for path in ordered:
        path.validate()
    security_ids = tuple(path.security_id for path in ordered)
    if (
        not decision_session_date
        or security_ids != tuple(sorted(set(security_ids)))
        or security_ids != exposure_panel.asset_ids
        or any(path.decision_at_ms != exposure_panel.origin_at_ms for path in ordered)
        or isinstance(built_at_ms, bool)
        or not isinstance(built_at_ms, int)
        or any(built_at_ms < max(path.available_at_ms) for path in ordered)
    ):
        raise MassiveAdaptiveAlphaTargetsV1Error(
            "adaptive target source support or build chronology differs"
        )
    for name, value in (
        ("origin", origin_receipt_sha256),
        ("economic accounting", economic_accounting_receipt_sha256),
        ("fill source", fill_source_receipt_sha256),
        ("terminal authority", terminal_authority_receipt_sha256),
        ("economic coverage", economic_coverage_receipt_sha256),
    ):
        _digest(name, value)

    raw_by_asset: list[list[float]] = []
    economic_valid_by_asset: list[list[bool]] = []
    terminal_by_asset: list[list[bool]] = []
    fallback_by_asset: list[list[bool]] = []
    for path in ordered:
        raw: list[float] = []
        valid: list[bool] = []
        terminal: list[bool] = []
        fallback: list[bool] = []
        for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS:
            value, is_valid, is_terminal, is_fallback = _bucket_return(
                path,
                start=bucket.start_offset_sessions,
                end=bucket.end_offset_sessions,
            )
            raw.append(value)
            valid.append(is_valid)
            terminal.append(is_terminal)
            fallback.append(is_fallback)
        raw_by_asset.append(raw)
        economic_valid_by_asset.append(valid)
        terminal_by_asset.append(terminal)
        fallback_by_asset.append(fallback)

    common_mask = tuple(
        exposure_valid and all(economic_valid)
        for exposure_valid, economic_valid in zip(
            exposure_panel.qualified_asset_mask,
            economic_valid_by_asset,
            strict=True,
        )
    )
    common_panel = OriginExposurePanel(
        origin_at_ms=exposure_panel.origin_at_ms,
        available_at_ms=exposure_panel.available_at_ms,
        asset_ids=exposure_panel.asset_ids,
        exposure_names=exposure_panel.exposure_names,
        exposures=exposure_panel.exposures,
        regression_weights=exposure_panel.regression_weights,
        qualified_asset_mask=common_mask,
        source_receipt_sha256=semantic_sha256(
            {
                "origin_exposure_source": exposure_panel.source_receipt_sha256,
                "common_complete_path_support": common_mask,
                "economic_path_inventory": tuple(
                    path.receipt_sha256 for path in ordered
                ),
                "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
            }
        ),
    )
    try:
        operator = build_origin_residual_operator(common_panel)
    except PITAlphaDataError as error:
        raise MassiveAdaptiveAlphaTargetsV1Error(
            "adaptive common target support cannot identify the factor operator"
        ) from error

    raw_array = np.asarray(raw_by_asset, dtype=np.float64)
    factor_array = np.zeros_like(raw_array)
    residual_array = np.zeros_like(raw_array)
    factor_target: list[float] = []
    exposure_errors: list[float] = []
    mask = np.asarray(common_mask, dtype=np.bool_)
    weights = np.asarray(operator.qualified_weights, dtype=np.float64)
    for bucket_index in range(_BUCKET_COUNT):
        result = apply_origin_residual_operator(
            raw_array[:, bucket_index], operator
        )
        residual = np.asarray(result.values, dtype=np.float64)
        residual_array[:, bucket_index] = residual
        factor_array[mask, bucket_index] = (
            raw_array[mask, bucket_index] - residual[mask]
        )
        factor_target.append(
            float(
                np.average(
                    factor_array[mask, bucket_index],
                    weights=weights,
                )
            )
        )
        exposure_errors.append(result.maximum_weighted_exposure_error)

    rows: list[MassiveAdaptiveAlphaTargetRowV1] = []
    for asset_index, path in enumerate(ordered):
        training_valid = (common_mask[asset_index],) * _BUCKET_COUNT
        row_body = {
            "security_id": path.security_id,
            "raw_bucket_returns": tuple(
                float(value) for value in raw_array[asset_index]
            ),
            "factor_component_returns": tuple(
                float(value) for value in factor_array[asset_index]
            ),
            "residual_bucket_returns": tuple(
                float(value) for value in residual_array[asset_index]
            ),
            "economic_valid_by_bucket": tuple(
                economic_valid_by_asset[asset_index]
            ),
            "training_valid_by_bucket": training_valid,
            "terminal_by_bucket": tuple(terminal_by_asset[asset_index]),
            "conservative_fallback_by_bucket": tuple(
                fallback_by_asset[asset_index]
            ),
            "economic_path_receipt_sha256": path.receipt_sha256,
            "residual_operator_receipt_sha256": operator.receipt_sha256,
        }
        row = MassiveAdaptiveAlphaTargetRowV1(
            **row_body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(row_body),
        )
        row.validate()
        rows.append(row)

    path_inventory = semantic_sha256(
        tuple(path.receipt_sha256 for path in ordered)
    )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic: dict[str, object] = {
        "schema": MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "decision_at_ms": exposure_panel.origin_at_ms,
        "built_at_ms": built_at_ms,
        "security_ids": security_ids,
        "bucket_ids": tuple(
            bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
        ),
        "rows": tuple(asdict(row) for row in rows),
        "factor_return_target": tuple(factor_target),
        "factor_valid": (True,) * _BUCKET_COUNT,
        "common_training_asset_mask": common_mask,
        "valid_counts_by_bucket": (sum(common_mask),) * _BUCKET_COUNT,
        "maximum_weighted_exposure_error_by_bucket": tuple(exposure_errors),
        "residual_operator": asdict(operator),
        "economic_path_inventory_sha256": path_inventory,
        "target_row_inventory_sha256": row_inventory,
        "origin_receipt_sha256": origin_receipt_sha256,
        "economic_accounting_receipt_sha256": economic_accounting_receipt_sha256,
        "fill_source_receipt_sha256": fill_source_receipt_sha256,
        "terminal_authority_receipt_sha256": terminal_authority_receipt_sha256,
        "economic_coverage_receipt_sha256": economic_coverage_receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SOURCE_SHA256,
        "source_paths_replayed": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveAdaptiveAlphaTargetsV1(
        decision_session_date=decision_session_date,
        decision_at_ms=exposure_panel.origin_at_ms,
        built_at_ms=built_at_ms,
        security_ids=security_ids,
        bucket_ids=semantic["bucket_ids"],  # type: ignore[arg-type]
        rows=tuple(rows),
        factor_return_target=tuple(factor_target),
        factor_valid=(True,) * _BUCKET_COUNT,
        common_training_asset_mask=common_mask,
        valid_counts_by_bucket=(sum(common_mask),) * _BUCKET_COUNT,
        maximum_weighted_exposure_error_by_bucket=tuple(exposure_errors),
        residual_operator=operator,
        economic_path_inventory_sha256=path_inventory,
        target_row_inventory_sha256=row_inventory,
        origin_receipt_sha256=origin_receipt_sha256,
        economic_accounting_receipt_sha256=economic_accounting_receipt_sha256,
        fill_source_receipt_sha256=fill_source_receipt_sha256,
        terminal_authority_receipt_sha256=terminal_authority_receipt_sha256,
        economic_coverage_receipt_sha256=economic_coverage_receipt_sha256,
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        specification_sha256=MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        source_paths_replayed=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_ALPHA_TARGETS_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA",
    "MassiveAdaptiveAlphaTargetRowV1",
    "MassiveAdaptiveAlphaTargetsV1",
    "MassiveAdaptiveAlphaTargetsV1Error",
    "MassiveAdaptiveEconomicPathV1",
    "build_massive_adaptive_alpha_targets_v1",
]
