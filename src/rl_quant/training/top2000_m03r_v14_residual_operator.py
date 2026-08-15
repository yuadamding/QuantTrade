"""Stable reusable exposure-null operators for M03R-v14 executable scores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_EXECUTABLE_SCORE_RULE,
    M03R_V14_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
    build_m03r_v11_residual_operator,
)

M03R_V14_RESIDUAL_OPERATOR_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-qr-map-residual-operator-v1"
)
M03R_V14_RESIDUAL_RESULT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-qr-map-residual-result-v1"
)


class M03RV14ResidualOperatorError(ValueError):
    """The reusable v14 residual operator is invalid or drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(tuple(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV14ResidualOperator:
    """A v11-qualified design plus its QR-derived linear residual map."""

    base: M03RV11ResidualOperator
    coefficient_map: torch.Tensor
    coefficient_map_sha256: str
    receipt_sha256: str
    tensor_version_counters: tuple[int, int, int, int]
    solver: str = "weighted-qr-precomputed-coefficient-map"
    operator_rule: str = M03R_V14_EXECUTABLE_SCORE_RULE
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_RESIDUAL_OPERATOR_SCHEMA

    def validate(self) -> None:
        self.base.validate()
        coefficient = self.coefficient_map
        expected_shape = (
            self.base.qualified_design.shape[1],
            self.base.qualified_design.shape[0],
        )
        if (
            not isinstance(coefficient, torch.Tensor)
            or coefficient.device.type != "cpu"
            or coefficient.dtype != torch.float64
            or tuple(coefficient.shape) != expected_shape
            or not bool(torch.isfinite(coefficient).all())
            or self.coefficient_map_sha256 != _tensor_sha256(coefficient)
            or self.tensor_version_counters
            != (
                coefficient._version,
                self.base.qualified_design._version,
                self.base.qualified_weights._version,
                self.base.qualified_asset_mask._version,
            )
            or self.solver != "weighted-qr-precomputed-coefficient-map"
            or self.operator_rule != M03R_V14_EXECUTABLE_SCORE_RULE
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_RESIDUAL_OPERATOR_SCHEMA
        ):
            raise M03RV14ResidualOperatorError("v14 residual operator drifted")
        identity = coefficient @ self.base.qualified_design
        expected = torch.eye(identity.shape[0], dtype=torch.float64)
        if not torch.allclose(identity, expected, rtol=0.0, atol=2.0e-10):
            raise M03RV14ResidualOperatorError(
                "v14 QR coefficient map does not reproduce the design identity"
            )
        if self.receipt_sha256 != _operator_receipt_sha256(
            base_receipt_sha256=self.base.receipt_sha256,
            coefficient_map_sha256=self.coefficient_map_sha256,
        ):
            raise M03RV14ResidualOperatorError("v14 residual receipt drifted")

    @property
    def qualified_asset_mask(self) -> torch.Tensor:
        return self.base.qualified_asset_mask

    @property
    def origin_state_index(self) -> int:
        return self.base.origin_state_index

    @property
    def factor_qualified_risky_asset_count(self) -> int:
        return self.base.factor_qualified_risky_asset_count

    @property
    def weighted_residual_degrees_of_freedom(self) -> int:
        return self.base.weighted_residual_degrees_of_freedom

    def require_fast_identity(self) -> None:
        if (
            self.tensor_version_counters
            != (
                self.coefficient_map._version,
                self.base.qualified_design._version,
                self.base.qualified_weights._version,
                self.base.qualified_asset_mask._version,
            )
            or self.solver != "weighted-qr-precomputed-coefficient-map"
            or self.operator_rule != M03R_V14_EXECUTABLE_SCORE_RULE
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_RESIDUAL_OPERATOR_SCHEMA
        ):
            raise M03RV14ResidualOperatorError("v14 residual tensor identity changed")


def _operator_receipt_sha256(
    *,
    base_receipt_sha256: str,
    coefficient_map_sha256: str,
) -> str:
    return _sha256(
        {
            "schema": M03R_V14_RESIDUAL_OPERATOR_SCHEMA,
            "protocol_sha256": M03R_V14_PROTOCOL_SHA256,
            "operator_rule": M03R_V14_EXECUTABLE_SCORE_RULE,
            "solver": "weighted-qr-precomputed-coefficient-map",
            "base_receipt_sha256": base_receipt_sha256,
            "coefficient_map_sha256": coefficient_map_sha256,
        }
    )


def build_m03r_v14_residual_operator(
    *,
    origin_state_index: int,
    cash_index: int,
    available_mask: torch.Tensor,
    exposure_loadings: torch.Tensor,
    regression_weights: torch.Tensor,
    projector_exposure_names: tuple[str, ...],
    projector_exposure_families: tuple[str, ...],
    asset_axis_sha256: str,
    source_exposure_receipt_sha256: str,
) -> M03RV14ResidualOperator:
    """Build one stable map without forming a dense asset-by-asset projector."""

    base = build_m03r_v11_residual_operator(
        origin_state_index=origin_state_index,
        cash_index=cash_index,
        available_mask=available_mask,
        exposure_loadings=exposure_loadings,
        regression_weights=regression_weights,
        projector_exposure_names=projector_exposure_names,
        projector_exposure_families=projector_exposure_families,
        asset_axis_sha256=asset_axis_sha256,
        source_exposure_receipt_sha256=source_exposure_receipt_sha256,
    )
    design = base.qualified_design
    root_weight = base.qualified_weights.sqrt()
    weighted_design = root_weight.unsqueeze(-1) * design
    q, r = torch.linalg.qr(weighted_design, mode="reduced")
    coefficient = torch.linalg.solve(
        r,
        q.T * root_weight.unsqueeze(0),
    ).contiguous()
    coefficient_sha256 = _tensor_sha256(coefficient)
    result = M03RV14ResidualOperator(
        base=base,
        coefficient_map=coefficient,
        coefficient_map_sha256=coefficient_sha256,
        receipt_sha256=_operator_receipt_sha256(
            base_receipt_sha256=base.receipt_sha256,
            coefficient_map_sha256=coefficient_sha256,
        ),
        tensor_version_counters=(
            coefficient._version,
            base.qualified_design._version,
            base.qualified_weights._version,
            base.qualified_asset_mask._version,
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV14ResidualResult:
    residual: torch.Tensor
    qualified_asset_mask: torch.Tensor
    operator_receipt_sha256: str
    weighted_exposure_error: float
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_RESIDUAL_RESULT_SCHEMA

    def validate(self, operator: M03RV14ResidualOperator) -> None:
        operator.require_fast_identity()
        if (
            not isinstance(self.residual, torch.Tensor)
            or self.residual.ndim != 1
            or self.residual.numel() != operator.qualified_asset_mask.numel()
            or not bool(torch.isfinite(self.residual).all())
            or not torch.equal(
                self.qualified_asset_mask.to(device="cpu"),
                operator.qualified_asset_mask,
            )
            or self.operator_receipt_sha256 != operator.receipt_sha256
            or not math.isfinite(self.weighted_exposure_error)
            or self.weighted_exposure_error > operator.base.orthogonality_tolerance
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_RESIDUAL_RESULT_SCHEMA
        ):
            raise M03RV14ResidualOperatorError("v14 residual result drifted")


def apply_m03r_v14_residual_operator(
    value: torch.Tensor,
    operator: M03RV14ResidualOperator,
) -> M03RV14ResidualResult:
    """Apply the precomputed linear map while preserving score gradients."""

    operator.require_fast_identity()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() != operator.qualified_asset_mask.numel()
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV14ResidualOperatorError("v14 residual value is malformed")
    selected = torch.nonzero(
        operator.qualified_asset_mask.to(device=value.device), as_tuple=False
    ).flatten()
    design = operator.base.qualified_design.to(device=value.device, dtype=torch.float64)
    weights = operator.base.qualified_weights.to(device=value.device, dtype=torch.float64)
    coefficient = operator.coefficient_map.to(device=value.device, dtype=torch.float64)
    target = value.index_select(0, selected).to(torch.float64)
    selected_residual = target - design @ (coefficient @ target)
    exposure_error = float((design.T @ (weights * selected_residual)).abs().max())
    residual = torch.zeros_like(value)
    residual[selected] = selected_residual.to(value.dtype)
    result = M03RV14ResidualResult(
        residual=residual,
        qualified_asset_mask=operator.qualified_asset_mask.to(device=value.device),
        operator_receipt_sha256=operator.receipt_sha256,
        weighted_exposure_error=exposure_error,
    )
    result.validate(operator)
    return result


__all__ = [
    "M03R_V14_RESIDUAL_OPERATOR_SCHEMA",
    "M03R_V14_RESIDUAL_RESULT_SCHEMA",
    "M03RV14ResidualOperator",
    "M03RV14ResidualOperatorError",
    "M03RV14ResidualResult",
    "apply_m03r_v14_residual_operator",
    "build_m03r_v14_residual_operator",
]
