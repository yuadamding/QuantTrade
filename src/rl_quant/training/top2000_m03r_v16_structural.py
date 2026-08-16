"""Package-owned structural operators and targets for M03R-v16.

The scientific workload must not rebuild data-dependent QR operators inside
the optimizer loop.  This module materializes every scheduled origin from the
exact verified cache and risk source, seals the resulting tensors in one
no-clobber artifact, and reconstructs only validated immutable operators at
worker startup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
    m03r_v16_selection_target_weights_from_id,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
)
from rl_quant.training.top2000_m03r_v15_residual_operator import (
    M03RV15ResidualOperator,
    apply_m03r_v15_residual_operator,
    build_m03r_v15_residual_operator,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03R_V16_REQUIRED_STATE_ROWS,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)

M03R_V16_STRUCTURAL_SLAB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-package-owned-structural-slab-v2"
)
M03R_V16_STRUCTURAL_SLAB_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-package-owned-structural-slab-file-v2"
)
_MAX_SLAB_BYTES = 2 * 1024**3
_RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE = 1.0e-5


class M03RV16StructuralError(ValueError):
    """The package-owned V16 structural artifact failed or drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise M03RV16StructuralError(f"{name} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical(tuple(tensor.shape)))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def scheduled_m03r_v16_origins() -> tuple[int, ...]:
    """Return every unique training, validation, and qualification origin."""

    origins: set[int] = set()
    for geometry in render_m03r_v16_fold_geometries(M03R_V16_REQUIRED_STATE_ROWS):
        origins.update(geometry.eligible_training_origins)
        origins.update(
            range(
                geometry.inner_validation_origin_start_inclusive,
                geometry.inner_validation_origin_stop_exclusive,
            )
        )
        origins.update(
            range(
                geometry.qualification_origin_start_inclusive,
                geometry.qualification_origin_stop_exclusive,
            )
        )
    return tuple(sorted(origins))


def _returned_dtype_exposure_error(
    residual: torch.Tensor,
    operator: M03RV15ResidualOperator,
) -> float:
    selected = torch.nonzero(operator.qualified_asset_mask, as_tuple=False).flatten()
    returned = (
        residual.detach().to(device="cpu").index_select(0, selected).to(torch.float64)
    )
    return float(
        (
            operator.base.qualified_design.T
            @ (operator.base.qualified_weights * returned)
        )
        .abs()
        .max()
    )


@dataclass(frozen=True, slots=True)
class M03RV16PreparedOrigin:
    origin_state_index: int
    action_operator: M03RV15ResidualOperator
    common_target_operator: M03RV15ResidualOperator
    economic_targets: tuple[torch.Tensor, ...]
    standardized_targets: tuple[torch.Tensor, ...]
    action_returned_dtype_zero_error: float
    target_returned_dtype_exposure_errors: tuple[float, ...]

    def validate(self) -> None:
        self.action_operator.validate()
        self.common_target_operator.validate()
        assets = self.action_operator.qualified_asset_mask.numel()
        if (
            self.origin_state_index != self.action_operator.origin_state_index
            or self.origin_state_index != self.common_target_operator.origin_state_index
            or bool(
                (
                    self.common_target_operator.qualified_asset_mask
                    & ~self.action_operator.qualified_asset_mask
                ).any()
            )
            or len(self.economic_targets) != len(M03R_V16_SETTINGS)
            or len(self.standardized_targets) != len(M03R_V16_SETTINGS)
            or len(self.target_returned_dtype_exposure_errors) != len(M03R_V16_SETTINGS)
            or not math.isfinite(self.action_returned_dtype_zero_error)
            or self.action_returned_dtype_zero_error
            > _RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE
        ):
            raise M03RV16StructuralError("V16 prepared origin identity drifted")
        for setting, economic, standardized, error in zip(
            M03R_V16_SETTINGS,
            self.economic_targets,
            self.standardized_targets,
            self.target_returned_dtype_exposure_errors,
            strict=True,
        ):
            if (
                not isinstance(economic, torch.Tensor)
                or economic.device.type != "cpu"
                or economic.dtype != torch.float32
                or economic.shape != (assets,)
                or not bool(torch.isfinite(economic).all())
                or not isinstance(standardized, torch.Tensor)
                or standardized.device.type != "cpu"
                or standardized.dtype != torch.float32
                or standardized.shape != (assets,)
                or not bool(torch.isfinite(standardized).all())
                or not torch.allclose(
                    standardized,
                    economic / setting.selection_target_scale,
                    rtol=2.0e-6,
                    atol=2.0e-7,
                )
                or not math.isfinite(error)
                or error > _RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE
            ):
                raise M03RV16StructuralError("V16 prepared target drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "origin_state_index": self.origin_state_index,
                "action_operator_receipt_sha256": (self.action_operator.receipt_sha256),
                "common_target_operator_receipt_sha256": (
                    self.common_target_operator.receipt_sha256
                ),
                "economic_target_sha256": tuple(
                    _tensor_sha256(value) for value in self.economic_targets
                ),
                "standardized_target_sha256": tuple(
                    _tensor_sha256(value) for value in self.standardized_targets
                ),
                "action_returned_dtype_zero_error": (
                    self.action_returned_dtype_zero_error
                ),
                "target_returned_dtype_exposure_errors": (
                    self.target_returned_dtype_exposure_errors
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class M03RV16StructuralSlabReceipt:
    cache_sha256: str
    cache_manifest_sha256: str
    asset_axis_sha256: str
    source_manifest_sha256: str
    operator_source_sha256: str
    risk_artifact_file_sha256: str
    risk_source_manifest_file_sha256: str
    risk_source_receipt_sha256: str
    exposure_receipt_sha256: str
    projector_manifest_file_sha256: str
    projector_manifest_sha256: str
    projector_binding_sha256: str
    fold_geometry_sha256: tuple[str, ...]
    scheduled_origin_sha256: str
    scheduled_origin_count: int
    first_scheduled_origin: int
    last_scheduled_origin: int
    action_operator_root_sha256: str
    common_target_operator_root_sha256: str
    target_root_sha256: tuple[str, ...]
    minimum_action_qualified_assets: int
    minimum_target_qualified_assets: int
    minimum_action_residual_degrees_of_freedom: int
    minimum_target_residual_degrees_of_freedom: int
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    all_scheduled_origins_qualified: bool = True
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_STRUCTURAL_SLAB_SCHEMA

    def validate(self) -> None:
        origins = scheduled_m03r_v16_origins()
        geometries = render_m03r_v16_fold_geometries(M03R_V16_REQUIRED_STATE_ROWS)
        digests = (
            self.cache_sha256,
            self.cache_manifest_sha256,
            self.asset_axis_sha256,
            self.source_manifest_sha256,
            self.operator_source_sha256,
            self.risk_artifact_file_sha256,
            self.risk_source_manifest_file_sha256,
            self.risk_source_receipt_sha256,
            self.exposure_receipt_sha256,
            self.projector_manifest_file_sha256,
            self.projector_manifest_sha256,
            self.projector_binding_sha256,
            self.action_operator_root_sha256,
            self.common_target_operator_root_sha256,
            *self.target_root_sha256,
        )
        if (
            any(not _digest("receipt digest", value) for value in digests)
            or self.fold_geometry_sha256
            != tuple(value.receipt_sha256 for value in geometries)
            or self.scheduled_origin_sha256 != _sha256(origins)
            or self.scheduled_origin_count != len(origins)
            or self.first_scheduled_origin != origins[0]
            or self.last_scheduled_origin != origins[-1]
            or len(self.target_root_sha256) != len(M03R_V16_SETTINGS)
            or min(
                self.minimum_action_qualified_assets,
                self.minimum_target_qualified_assets,
                self.minimum_action_residual_degrees_of_freedom,
                self.minimum_target_residual_degrees_of_freedom,
            )
            <= 0
            or not self.all_scheduled_origins_qualified
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_STRUCTURAL_SLAB_SCHEMA
        ):
            raise M03RV16StructuralError("V16 structural slab receipt drifted")

    def validate_for_package(self, **expected: str) -> None:
        self.validate()
        allowed = {
            "cache_sha256",
            "cache_manifest_sha256",
            "asset_axis_sha256",
            "source_manifest_sha256",
            "operator_source_sha256",
            "risk_artifact_file_sha256",
            "risk_source_manifest_file_sha256",
            "risk_source_receipt_sha256",
            "exposure_receipt_sha256",
            "projector_manifest_file_sha256",
            "projector_manifest_sha256",
            "projector_binding_sha256",
        }
        if set(expected) != allowed or any(
            getattr(self, name) != value for name, value in expected.items()
        ):
            raise M03RV16StructuralError(
                "V16 structural slab does not bind the exact package"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16StructuralSlab:
    receipt: M03RV16StructuralSlabReceipt
    origins: tuple[M03RV16PreparedOrigin, ...]

    def validate(self) -> None:
        self.receipt.validate()
        expected_origins = scheduled_m03r_v16_origins()
        if (
            tuple(value.origin_state_index for value in self.origins)
            != expected_origins
        ):
            raise M03RV16StructuralError("V16 structural slab origin axis drifted")
        for value in self.origins:
            value.validate()
        action_rows = tuple(
            (value.origin_state_index, value.action_operator.receipt_sha256)
            for value in self.origins
        )
        target_rows = tuple(
            (value.origin_state_index, value.common_target_operator.receipt_sha256)
            for value in self.origins
        )
        target_roots = tuple(
            _sha256(
                tuple(
                    (
                        value.origin_state_index,
                        _tensor_sha256(value.economic_targets[index]),
                    )
                    for value in self.origins
                )
            )
            for index in range(len(M03R_V16_SETTINGS))
        )
        if (
            self.receipt.action_operator_root_sha256 != _sha256(action_rows)
            or self.receipt.common_target_operator_root_sha256 != _sha256(target_rows)
            or self.receipt.target_root_sha256 != target_roots
        ):
            raise M03RV16StructuralError("V16 structural slab content root drifted")

    def origin(self, origin_state_index: int) -> M03RV16PreparedOrigin:
        self.validate()
        first = self.receipt.first_scheduled_origin
        candidate = origin_state_index - first
        if 0 <= candidate < len(self.origins):
            value = self.origins[candidate]
            if value.origin_state_index == origin_state_index:
                return value
        # Scheduled origins can contain gaps between disjoint folds.
        for value in self.origins:
            if value.origin_state_index == origin_state_index:
                return value
        raise M03RV16StructuralError("V16 origin is absent from structural slab")


_VALIDATED_SLAB_ISSUER = object()


@dataclass(frozen=True, slots=True)
class M03RV16DeviceActionOperator:
    """One action operator materialized once on a worker device."""

    origin_state_index: int
    qualified_asset_mask: torch.Tensor
    selected_indices: torch.Tensor
    qualified_design: torch.Tensor
    qualified_weights: torch.Tensor
    coefficient_map: torch.Tensor
    operator_receipt_sha256: str

    def apply(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            value.ndim != 1
            or value.numel() != self.qualified_asset_mask.numel()
            or value.device != self.qualified_asset_mask.device
            or not value.is_floating_point()
        ):
            raise M03RV16StructuralError("V16 device action value drifted")
        target = value.index_select(0, self.selected_indices).to(torch.float64)
        selected_residual = target - self.qualified_design @ (
            self.coefficient_map @ target
        )
        residual = torch.zeros_like(value)
        residual[self.selected_indices] = selected_residual.to(value.dtype)
        returned = residual.index_select(0, self.selected_indices).to(torch.float64)
        exposure_error = (
            (self.qualified_design.T @ (self.qualified_weights * returned)).abs().max()
        )
        return residual, exposure_error


@dataclass(frozen=True, slots=True)
class M03RV16DevicePreparedOrigin:
    origin_state_index: int
    action_operator: M03RV16DeviceActionOperator
    common_target_mask: torch.Tensor
    economic_targets: tuple[torch.Tensor, ...]
    standardized_targets: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class M03RV16ValidatedStructuralSlab:
    """Startup-qualified slab authority with O(1) mutation-checked lookup."""

    slab: M03RV16StructuralSlab
    origin_to_slot: dict[int, int]
    target_tensor_versions: tuple[tuple[int, ...], ...]
    receipt_sha256: str
    device_action_operator_cache: dict[
        tuple[int, str, int | None], M03RV16DeviceActionOperator
    ]
    device_prepared_origin_cache: dict[
        tuple[int, str, int | None], M03RV16DevicePreparedOrigin
    ]
    _issuer: object

    @property
    def receipt(self) -> M03RV16StructuralSlabReceipt:
        return self.slab.receipt

    @property
    def origins(self) -> tuple[M03RV16PreparedOrigin, ...]:
        return self.slab.origins

    def require_fast_identity(self) -> None:
        if (
            self._issuer is not _VALIDATED_SLAB_ISSUER
            or len(self.origin_to_slot) != len(self.slab.origins)
            or len(self.target_tensor_versions) != len(self.slab.origins)
        ):
            raise M03RV16StructuralError("V16 validated slab authority drifted")

    def origin(self, origin_state_index: int) -> M03RV16PreparedOrigin:
        self.require_fast_identity()
        slot = self.origin_to_slot.get(origin_state_index)
        if slot is None:
            raise M03RV16StructuralError("V16 origin is absent from structural slab")
        value = self.slab.origins[slot]
        if value.origin_state_index != origin_state_index:
            raise M03RV16StructuralError("V16 structural slab lookup drifted")
        value.action_operator.require_fast_identity()
        value.common_target_operator.require_fast_identity()
        observed_versions = tuple(
            tensor._version
            for tensor in (*value.economic_targets, *value.standardized_targets)
        )
        if observed_versions != self.target_tensor_versions[slot]:
            raise M03RV16StructuralError("V16 structural target tensor mutated")
        return value

    def device_action_operator(
        self,
        origin_state_index: int,
        device: torch.device,
    ) -> M03RV16DeviceActionOperator:
        value = self.origin(origin_state_index)
        slot = self.origin_to_slot[origin_state_index]
        key = (slot, device.type, device.index)
        cached = self.device_action_operator_cache.get(key)
        if cached is not None:
            return cached
        operator = value.action_operator
        mask = operator.qualified_asset_mask.to(device=device)
        created = M03RV16DeviceActionOperator(
            origin_state_index=origin_state_index,
            qualified_asset_mask=mask,
            selected_indices=torch.nonzero(mask, as_tuple=False).flatten(),
            qualified_design=operator.base.qualified_design.to(
                device=device, dtype=torch.float64
            ),
            qualified_weights=operator.base.qualified_weights.to(
                device=device, dtype=torch.float64
            ),
            coefficient_map=operator.coefficient_map.to(
                device=device, dtype=torch.float64
            ),
            operator_receipt_sha256=operator.receipt_sha256,
        )
        self.device_action_operator_cache[key] = created
        return created

    def device_origin(
        self,
        origin_state_index: int,
        device: torch.device,
    ) -> M03RV16DevicePreparedOrigin:
        value = self.origin(origin_state_index)
        slot = self.origin_to_slot[origin_state_index]
        key = (slot, device.type, device.index)
        cached = self.device_prepared_origin_cache.get(key)
        if cached is not None:
            return cached
        created = M03RV16DevicePreparedOrigin(
            origin_state_index=origin_state_index,
            action_operator=self.device_action_operator(origin_state_index, device),
            common_target_mask=value.common_target_operator.qualified_asset_mask.to(
                device=device
            ),
            economic_targets=tuple(
                row.to(device=device) for row in value.economic_targets
            ),
            standardized_targets=tuple(
                row.to(device=device) for row in value.standardized_targets
            ),
        )
        self.device_prepared_origin_cache[key] = created
        return created


def qualify_m03r_v16_structural_slab(
    slab: M03RV16StructuralSlab,
) -> M03RV16ValidatedStructuralSlab:
    """Deep-validate once, then issue a fast runtime authority."""

    slab.validate()
    origin_to_slot = {
        value.origin_state_index: index for index, value in enumerate(slab.origins)
    }
    if len(origin_to_slot) != len(slab.origins):
        raise M03RV16StructuralError("V16 structural slab origins are not unique")
    versions = tuple(
        tuple(
            tensor._version
            for tensor in (*value.economic_targets, *value.standardized_targets)
        )
        for value in slab.origins
    )
    return M03RV16ValidatedStructuralSlab(
        slab=slab,
        origin_to_slot=origin_to_slot,
        target_tensor_versions=versions,
        receipt_sha256=slab.receipt.receipt_sha256,
        device_action_operator_cache={},
        device_prepared_origin_cache={},
        _issuer=_VALIDATED_SLAB_ISSUER,
    )


def _cache_simple_returns(cache: Top2000VerifiedDevelopmentCache) -> torch.Tensor:
    # Reproduce the training adapter's exact economic dtype before log1p.
    close = cache.daily_ohlcv[..., 3]
    pair_valid = cache.availability[:-1] & cache.availability[1:]
    previous = torch.where(pair_valid, close[:-1], torch.ones_like(close[:-1]))
    result = torch.where(
        pair_valid,
        close[1:] / previous - 1.0,
        torch.zeros_like(close[:-1]),
    )
    result[:, 0] = 0.0
    if not bool(torch.isfinite(result).all()) or bool((result <= -1.0).any()):
        raise M03RV16StructuralError("V16 cache return path is invalid")
    return result


def build_m03r_v16_structural_slab(
    cache: Top2000VerifiedDevelopmentCache,
    risk_source: M03RV9MaterializedRiskSource,
    *,
    cache_manifest_sha256: str,
    source_manifest_sha256: str,
    operator_source_sha256: str,
    risk_artifact_file_sha256: str,
    risk_source_manifest_file_sha256: str,
    projector_manifest_file_sha256: str,
    projector_manifest_sha256: str,
    projector_binding_sha256: str,
) -> M03RV16StructuralSlab:
    """Materialize every exact V16 operator and target once on CPU."""

    cache.validate_unmodified()
    risk_source.validate()
    exposures = risk_source.exposures
    if (
        cache.daily_ohlcv.shape[0] != M03R_V16_REQUIRED_STATE_ROWS
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or exposures.asset_axis_sha256 != cache.action_hash
        or exposures.state_start_index != 0
        or exposures.exposure_loadings.shape[:2] != cache.daily_ohlcv.shape[:2]
    ):
        raise M03RV16StructuralError("V16 cache, risk, or asset axis drifted")
    returns = _cache_simple_returns(cache)
    cash = exposures.cash_index
    prepared: list[M03RV16PreparedOrigin] = []
    for origin in scheduled_m03r_v16_origins():
        action_available = cache.availability[origin].clone()
        action_available[cash] = False
        future_available = cache.availability[
            origin + 1 : origin + M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS + 2
        ].all(dim=0)
        future_available[cash] = False
        action_operator = build_m03r_v15_residual_operator(
            available_mask=action_available,
            origin_state_index=origin,
            cash_index=cash,
            exposure_loadings=exposures.exposure_loadings[origin],
            regression_weights=exposures.regression_weights[origin],
            projector_exposure_names=exposures.projector_exposure_names,
            projector_exposure_families=exposures.projector_exposure_families,
            asset_axis_sha256=cache.action_hash,
            source_exposure_receipt_sha256=exposures.receipt_sha256,
        )
        target_operator = build_m03r_v15_residual_operator(
            available_mask=action_available & future_available,
            origin_state_index=origin,
            cash_index=cash,
            exposure_loadings=exposures.exposure_loadings[origin],
            regression_weights=exposures.regression_weights[origin],
            projector_exposure_names=exposures.projector_exposure_names,
            projector_exposure_families=exposures.projector_exposure_families,
            asset_axis_sha256=cache.action_hash,
            source_exposure_receipt_sha256=exposures.receipt_sha256,
        )
        economic_targets: list[torch.Tensor] = []
        standardized_targets: list[torch.Tensor] = []
        target_errors: list[float] = []
        for setting in M03R_V16_SETTINGS:
            weights = torch.tensor(
                m03r_v16_selection_target_weights_from_id(setting.selection_target),
                dtype=returns.dtype,
            )
            target_rows = returns[
                origin + 1 : origin + 1 + setting.numerical_target_support_sessions
            ]
            raw_target = (
                torch.log1p(target_rows.clamp_min(-0.999999)) * weights.unsqueeze(-1)
            ).sum(dim=0)
            target_result = apply_m03r_v15_residual_operator(
                raw_target.to(torch.float32), target_operator
            )
            economic = target_result.residual.detach().to(device="cpu").contiguous()
            economic_targets.append(economic)
            standardized_targets.append(
                (economic / setting.selection_target_scale).contiguous()
            )
            target_errors.append(
                _returned_dtype_exposure_error(economic, target_operator)
            )
        zero = torch.zeros(cache.daily_ohlcv.shape[1], dtype=torch.float32)
        action_zero = apply_m03r_v15_residual_operator(zero, action_operator).residual
        row = M03RV16PreparedOrigin(
            origin_state_index=origin,
            action_operator=action_operator,
            common_target_operator=target_operator,
            economic_targets=tuple(economic_targets),
            standardized_targets=tuple(standardized_targets),
            action_returned_dtype_zero_error=_returned_dtype_exposure_error(
                action_zero, action_operator
            ),
            target_returned_dtype_exposure_errors=tuple(target_errors),
        )
        row.validate()
        prepared.append(row)
    origins = tuple(prepared)
    geometries = render_m03r_v16_fold_geometries(M03R_V16_REQUIRED_STATE_ROWS)
    receipt = M03RV16StructuralSlabReceipt(
        cache_sha256=cache.cache_sha256,
        cache_manifest_sha256=_digest("cache_manifest_sha256", cache_manifest_sha256),
        asset_axis_sha256=cache.action_hash,
        source_manifest_sha256=_digest(
            "source_manifest_sha256", source_manifest_sha256
        ),
        operator_source_sha256=_digest(
            "operator_source_sha256", operator_source_sha256
        ),
        risk_artifact_file_sha256=_digest(
            "risk_artifact_file_sha256", risk_artifact_file_sha256
        ),
        risk_source_manifest_file_sha256=_digest(
            "risk_source_manifest_file_sha256",
            risk_source_manifest_file_sha256,
        ),
        risk_source_receipt_sha256=risk_source.receipt_sha256,
        exposure_receipt_sha256=exposures.receipt_sha256,
        projector_manifest_file_sha256=_digest(
            "projector_manifest_file_sha256", projector_manifest_file_sha256
        ),
        projector_manifest_sha256=_digest(
            "projector_manifest_sha256", projector_manifest_sha256
        ),
        projector_binding_sha256=_digest(
            "projector_binding_sha256", projector_binding_sha256
        ),
        fold_geometry_sha256=tuple(value.receipt_sha256 for value in geometries),
        scheduled_origin_sha256=_sha256(scheduled_m03r_v16_origins()),
        scheduled_origin_count=len(origins),
        first_scheduled_origin=origins[0].origin_state_index,
        last_scheduled_origin=origins[-1].origin_state_index,
        action_operator_root_sha256=_sha256(
            tuple(
                (value.origin_state_index, value.action_operator.receipt_sha256)
                for value in origins
            )
        ),
        common_target_operator_root_sha256=_sha256(
            tuple(
                (
                    value.origin_state_index,
                    value.common_target_operator.receipt_sha256,
                )
                for value in origins
            )
        ),
        target_root_sha256=tuple(
            _sha256(
                tuple(
                    (
                        value.origin_state_index,
                        _tensor_sha256(value.economic_targets[index]),
                    )
                    for value in origins
                )
            )
            for index in range(len(M03R_V16_SETTINGS))
        ),
        minimum_action_qualified_assets=min(
            value.action_operator.factor_qualified_risky_asset_count
            for value in origins
        ),
        minimum_target_qualified_assets=min(
            value.common_target_operator.factor_qualified_risky_asset_count
            for value in origins
        ),
        minimum_action_residual_degrees_of_freedom=min(
            value.action_operator.weighted_residual_degrees_of_freedom
            for value in origins
        ),
        minimum_target_residual_degrees_of_freedom=min(
            value.common_target_operator.weighted_residual_degrees_of_freedom
            for value in origins
        ),
    )
    slab = M03RV16StructuralSlab(receipt=receipt, origins=origins)
    slab.validate()
    return slab


def _pack_operator(operator: M03RV15ResidualOperator) -> dict[str, Any]:
    base = operator.base
    return {
        "base": {
            name: getattr(base, name)
            for name in base.__dataclass_fields__
            if name
            not in {"qualified_asset_mask", "qualified_design", "qualified_weights"}
        },
        "qualified_asset_mask": base.qualified_asset_mask,
        "qualified_design": base.qualified_design,
        "qualified_weights": base.qualified_weights,
        "coefficient_map": operator.coefficient_map,
        "coefficient_map_sha256": operator.coefficient_map_sha256,
        "receipt_sha256": operator.receipt_sha256,
        "solver": operator.solver,
        "operator_rule": operator.operator_rule,
        "protocol_sha256": operator.protocol_sha256,
        "schema": operator.schema,
    }


def _unpack_operator(payload: dict[str, Any]) -> M03RV15ResidualOperator:
    base_values = dict(payload["base"])
    for name in (
        "exposure_names",
        "exposure_families",
        "unsupported_sector_names",
    ):
        base_values[name] = tuple(base_values[name])
    base = M03RV11ResidualOperator(
        **base_values,
        qualified_asset_mask=payload["qualified_asset_mask"],
        qualified_design=payload["qualified_design"],
        qualified_weights=payload["qualified_weights"],
    )
    coefficient = payload["coefficient_map"]
    operator = M03RV15ResidualOperator(
        base=base,
        coefficient_map=coefficient,
        coefficient_map_sha256=payload["coefficient_map_sha256"],
        receipt_sha256=payload["receipt_sha256"],
        tensor_version_counters=(
            coefficient._version,
            base.qualified_design._version,
            base.qualified_weights._version,
            base.qualified_asset_mask._version,
        ),
        solver=payload["solver"],
        operator_rule=payload["operator_rule"],
        protocol_sha256=payload["protocol_sha256"],
        schema=payload["schema"],
    )
    operator.validate()
    return operator


def write_m03r_v16_structural_slab(
    path: str | Path,
    slab: M03RV16StructuralSlab,
) -> str:
    slab.validate()
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV16StructuralError("V16 structural slab target already exists")
    payload = {
        "schema": M03R_V16_STRUCTURAL_SLAB_FILE_SCHEMA,
        "receipt": asdict(slab.receipt),
        "receipt_sha256": slab.receipt.receipt_sha256,
        "origins": [
            {
                "origin_state_index": value.origin_state_index,
                "action_operator": _pack_operator(value.action_operator),
                "common_target_operator": _pack_operator(value.common_target_operator),
                "economic_targets": value.economic_targets,
                "standardized_targets": value.standardized_targets,
                "action_returned_dtype_zero_error": (
                    value.action_returned_dtype_zero_error
                ),
                "target_returned_dtype_exposure_errors": (
                    value.target_returned_dtype_exposure_errors
                ),
            }
            for value in slab.origins
        ],
    }
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def load_m03r_v16_structural_slab(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> M03RV16ValidatedStructuralSlab:
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    expected_receipt_sha256 = _digest(
        "expected_receipt_sha256", expected_receipt_sha256
    )
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16StructuralError("V16 structural slab is unavailable") from exc
    try:
        stream = os.fdopen(descriptor, "rb")
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_SLAB_BYTES
        ):
            raise M03RV16StructuralError("V16 structural slab size or type drifted")
        digest = hashlib.sha256()
        size = 0
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16StructuralError("V16 structural slab changed while read")
        if size != before.st_size or digest.hexdigest() != expected_file_sha256:
            raise M03RV16StructuralError("V16 structural slab file hash drifted")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        final = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise M03RV16StructuralError("V16 structural slab changed while loaded")
        receipt_values = dict(payload["receipt"])
        for name in ("fold_geometry_sha256", "target_root_sha256"):
            receipt_values[name] = tuple(receipt_values[name])
        receipt = M03RV16StructuralSlabReceipt(**receipt_values)
        origins = tuple(
            M03RV16PreparedOrigin(
                origin_state_index=int(value["origin_state_index"]),
                action_operator=_unpack_operator(value["action_operator"]),
                common_target_operator=_unpack_operator(
                    value["common_target_operator"]
                ),
                economic_targets=tuple(value["economic_targets"]),
                standardized_targets=tuple(value["standardized_targets"]),
                action_returned_dtype_zero_error=float(
                    value["action_returned_dtype_zero_error"]
                ),
                target_returned_dtype_exposure_errors=tuple(
                    float(row) for row in value["target_returned_dtype_exposure_errors"]
                ),
            )
            for value in payload["origins"]
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        raise M03RV16StructuralError("V16 structural slab is malformed") from exc
    finally:
        stream.close()
    if (
        payload.get("schema") != M03R_V16_STRUCTURAL_SLAB_FILE_SCHEMA
        or payload.get("receipt_sha256") != expected_receipt_sha256
        or receipt.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV16StructuralError("V16 structural slab receipt drifted")
    result = M03RV16StructuralSlab(receipt=receipt, origins=origins)
    return qualify_m03r_v16_structural_slab(result)


__all__ = [
    "M03R_V16_STRUCTURAL_SLAB_FILE_SCHEMA",
    "M03R_V16_STRUCTURAL_SLAB_SCHEMA",
    "M03RV16PreparedOrigin",
    "M03RV16DeviceActionOperator",
    "M03RV16DevicePreparedOrigin",
    "M03RV16StructuralError",
    "M03RV16StructuralSlab",
    "M03RV16StructuralSlabReceipt",
    "M03RV16ValidatedStructuralSlab",
    "build_m03r_v16_structural_slab",
    "load_m03r_v16_structural_slab",
    "qualify_m03r_v16_structural_slab",
    "scheduled_m03r_v16_origins",
    "write_m03r_v16_structural_slab",
]
