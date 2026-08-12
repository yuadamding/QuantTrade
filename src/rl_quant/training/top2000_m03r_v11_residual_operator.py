"""One shared weighted residual operator for M03R-v11 targets and signals."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_RESIDUAL_OPERATOR_RULE,
)

M03R_V11_RESIDUAL_OPERATOR_SCHEMA = "rl-quant.top2000-dev.m03r-v11-residual-operator-v2"
M03R_V11_RESIDUAL_RESULT_SCHEMA = "rl-quant.top2000-dev.m03r-v11-residual-result-v1"


class M03RV11ResidualOperatorError(ValueError):
    """The qualified exposure space or residual calculation drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV11ResidualOperator:
    origin_state_index: int
    cash_index: int
    available_risky_asset_count: int
    factor_qualified_risky_asset_count: int
    factor_qualified_fraction: float
    exposure_names: tuple[str, ...]
    exposure_families: tuple[str, ...]
    dropped_reference_sector: str
    unsupported_sector_names: tuple[str, ...]
    qualified_asset_mask: torch.Tensor
    qualified_design: torch.Tensor
    qualified_weights: torch.Tensor
    effective_design_rank: int
    weighted_residual_degrees_of_freedom: int
    asset_axis_sha256: str
    source_exposure_receipt_sha256: str
    qualified_asset_mask_sha256: str
    exposure_matrix_sha256: str
    weight_vector_sha256: str
    solver: str = "weighted-qr-lstsq"
    ridge_convention: str = "none-after-reference-sector-drop"
    orthogonality_tolerance: float = 1.0e-9
    operator_rule: str = M03R_V11_RESIDUAL_OPERATOR_RULE
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_RESIDUAL_OPERATOR_SCHEMA

    def validate(self) -> None:
        mask = self.qualified_asset_mask
        design = self.qualified_design
        weights = self.qualified_weights
        if (
            isinstance(self.origin_state_index, bool)
            or self.origin_state_index < 0
            or isinstance(self.cash_index, bool)
            or self.cash_index < 0
            or not isinstance(mask, torch.Tensor)
            or mask.ndim != 1
            or mask.dtype != torch.bool
            or self.cash_index >= mask.numel()
            or bool(mask[self.cash_index])
            or not isinstance(design, torch.Tensor)
            or design.ndim != 2
            or design.shape[0] != int(mask.sum())
            or design.shape[1] != len(self.exposure_names)
            or not isinstance(weights, torch.Tensor)
            or weights.ndim != 1
            or weights.shape[0] != design.shape[0]
            or design.dtype != torch.float64
            or weights.dtype != torch.float64
            or not bool(torch.isfinite(design).all())
            or not bool(torch.isfinite(weights).all())
            or bool((weights <= 0.0).any())
            or self.available_risky_asset_count
            < self.factor_qualified_risky_asset_count
            or self.factor_qualified_risky_asset_count != int(mask.sum())
            or self.factor_qualified_fraction
            != self.factor_qualified_risky_asset_count
            / self.available_risky_asset_count
            or self.effective_design_rank != design.shape[1]
            or self.weighted_residual_degrees_of_freedom
            != design.shape[0] - self.effective_design_rank
            or self.weighted_residual_degrees_of_freedom <= 0
            or self.exposure_names[0] != "intercept"
            or len(self.exposure_names) != len(self.exposure_families)
            or self.exposure_families[0] != "intercept"
            or self.dropped_reference_sector in self.exposure_names
            or any(
                name in self.exposure_names for name in self.unsupported_sector_names
            )
            or self.dropped_reference_sector in self.unsupported_sector_names
            or self.solver != "weighted-qr-lstsq"
            or self.ridge_convention != "none-after-reference-sector-drop"
            or self.orthogonality_tolerance != 1.0e-9
            or self.operator_rule != M03R_V11_RESIDUAL_OPERATOR_RULE
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_RESIDUAL_OPERATOR_SCHEMA
            or self.qualified_asset_mask_sha256 != _tensor_sha256(mask)
            or self.exposure_matrix_sha256 != _tensor_sha256(design)
            or self.weight_vector_sha256 != _tensor_sha256(weights)
        ):
            raise M03RV11ResidualOperatorError("v11 residual operator drifted")
        for value in (self.asset_axis_sha256, self.source_exposure_receipt_sha256):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise M03RV11ResidualOperatorError(
                    "v11 residual operator identity is not a SHA-256"
                )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "origin_state_index": self.origin_state_index,
                "cash_index": self.cash_index,
                "available_risky_asset_count": self.available_risky_asset_count,
                "factor_qualified_risky_asset_count": (
                    self.factor_qualified_risky_asset_count
                ),
                "factor_qualified_fraction": self.factor_qualified_fraction,
                "exposure_names": self.exposure_names,
                "exposure_families": self.exposure_families,
                "dropped_reference_sector": self.dropped_reference_sector,
                "unsupported_sector_names": self.unsupported_sector_names,
                "effective_design_rank": self.effective_design_rank,
                "weighted_residual_degrees_of_freedom": (
                    self.weighted_residual_degrees_of_freedom
                ),
                "asset_axis_sha256": self.asset_axis_sha256,
                "source_exposure_receipt_sha256": (self.source_exposure_receipt_sha256),
                "qualified_asset_mask_sha256": self.qualified_asset_mask_sha256,
                "exposure_matrix_sha256": self.exposure_matrix_sha256,
                "weight_vector_sha256": self.weight_vector_sha256,
                "solver": self.solver,
                "ridge_convention": self.ridge_convention,
                "orthogonality_tolerance": self.orthogonality_tolerance,
                "operator_rule": self.operator_rule,
            }
        )


def build_m03r_v11_residual_operator(
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
) -> M03RV11ResidualOperator:
    if (
        not isinstance(available_mask, torch.Tensor)
        or available_mask.ndim != 1
        or available_mask.dtype != torch.bool
        or not isinstance(exposure_loadings, torch.Tensor)
        or exposure_loadings.ndim != 2
        or exposure_loadings.shape[0] != available_mask.numel()
        or exposure_loadings.shape[1] != len(projector_exposure_names) + 1
        or not isinstance(regression_weights, torch.Tensor)
        or regression_weights.shape != available_mask.shape
        or len(projector_exposure_names) != len(projector_exposure_families)
        or not 0 <= cash_index < available_mask.numel()
    ):
        raise M03RV11ResidualOperatorError("v11 residual inputs are not aligned")
    available = available_mask.detach().to(device="cpu").clone()
    available[cash_index] = False
    weights_all = regression_weights.detach().to(device="cpu", dtype=torch.float64)
    loadings = exposure_loadings.detach().to(device="cpu", dtype=torch.float64)
    qualified = available & (weights_all > 0.0)
    sector_positions = tuple(
        index
        for index, family in enumerate(projector_exposure_families)
        if family == "sector"
    )
    if not sector_positions:
        raise M03RV11ResidualOperatorError(
            "v11 residual design requires a declared sector family"
        )
    selected = torch.nonzero(qualified, as_tuple=False).flatten()
    supported_sector_positions = tuple(
        index
        for index in sector_positions
        if selected.numel() > 0
        and bool(loadings.index_select(0, selected)[:, index + 1].abs().gt(0.0).any())
    )
    if not supported_sector_positions:
        raise M03RV11ResidualOperatorError(
            "v11 residual design has no supported reference sector"
        )
    unsupported_sector_positions = tuple(
        index for index in sector_positions if index not in supported_sector_positions
    )
    dropped_projector_index = supported_sector_positions[-1]
    dropped_name = projector_exposure_names[dropped_projector_index]
    keep_projector_indices = tuple(
        index
        for index in range(len(projector_exposure_names))
        if index != dropped_projector_index
        and index not in unsupported_sector_positions
    )
    keep_loading_columns = (0, *(index + 1 for index in keep_projector_indices))
    names = (
        "intercept",
        *(projector_exposure_names[index] for index in keep_projector_indices),
    )
    families = (
        "intercept",
        *(projector_exposure_families[index] for index in keep_projector_indices),
    )
    design = loadings.index_select(0, selected).index_select(
        1, torch.tensor(keep_loading_columns, dtype=torch.long)
    )
    weights = weights_all.index_select(0, selected)
    if (
        selected.numel() <= design.shape[1]
        or not bool(torch.isfinite(design).all())
        or not bool(torch.isfinite(weights).all())
    ):
        raise M03RV11ResidualOperatorError(
            "v11 residual design has insufficient qualified finite support"
        )
    weighted_design = weights.sqrt().unsqueeze(-1) * design
    rank = int(torch.linalg.matrix_rank(weighted_design))
    if rank != design.shape[1]:
        raise M03RV11ResidualOperatorError(
            "v11 residual design is rank deficient after reference-sector drop"
        )
    available_count = int(available.sum())
    qualified_count = int(qualified.sum())
    result = M03RV11ResidualOperator(
        origin_state_index=origin_state_index,
        cash_index=cash_index,
        available_risky_asset_count=available_count,
        factor_qualified_risky_asset_count=qualified_count,
        factor_qualified_fraction=qualified_count / available_count,
        exposure_names=names,
        exposure_families=families,
        dropped_reference_sector=dropped_name,
        unsupported_sector_names=tuple(
            projector_exposure_names[index] for index in unsupported_sector_positions
        ),
        qualified_asset_mask=qualified,
        qualified_design=design,
        qualified_weights=weights,
        effective_design_rank=rank,
        weighted_residual_degrees_of_freedom=qualified_count - rank,
        asset_axis_sha256=asset_axis_sha256,
        source_exposure_receipt_sha256=source_exposure_receipt_sha256,
        qualified_asset_mask_sha256=_tensor_sha256(qualified),
        exposure_matrix_sha256=_tensor_sha256(design),
        weight_vector_sha256=_tensor_sha256(weights),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV11ResidualResult:
    residual: torch.Tensor
    qualified_asset_mask: torch.Tensor
    operator_receipt_sha256: str
    weighted_exposure_error: float
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_RESIDUAL_RESULT_SCHEMA

    def validate(self, operator: M03RV11ResidualOperator) -> None:
        operator.validate()
        if (
            self.residual.ndim != 1
            or self.residual.numel() != operator.qualified_asset_mask.numel()
            or not bool(torch.isfinite(self.residual).all())
            or not torch.equal(
                self.qualified_asset_mask.to(device="cpu"),
                operator.qualified_asset_mask,
            )
            or self.operator_receipt_sha256 != operator.receipt_sha256
            or not math.isfinite(self.weighted_exposure_error)
            or self.weighted_exposure_error > operator.orthogonality_tolerance
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_RESIDUAL_RESULT_SCHEMA
        ):
            raise M03RV11ResidualOperatorError("v11 residual result drifted")


def apply_m03r_v11_residual_operator(
    value: torch.Tensor,
    operator: M03RV11ResidualOperator,
) -> M03RV11ResidualResult:
    operator.validate()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() != operator.qualified_asset_mask.numel()
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV11ResidualOperatorError("v11 residual value is malformed")
    selected = torch.nonzero(
        operator.qualified_asset_mask.to(device=value.device), as_tuple=False
    ).flatten()
    design = operator.qualified_design.to(device=value.device, dtype=torch.float64)
    weights = operator.qualified_weights.to(device=value.device, dtype=torch.float64)
    target = value.index_select(0, selected).to(torch.float64)
    root_weight = weights.sqrt()
    weighted_design = root_weight.unsqueeze(-1) * design
    weighted_target = root_weight * target
    q, r = torch.linalg.qr(weighted_design, mode="reduced")
    coefficients = torch.linalg.solve(r, q.T @ weighted_target)
    selected_residual = target - design @ coefficients
    exposure_error = float((design.T @ (weights * selected_residual)).abs().max())
    residual = torch.zeros_like(value)
    residual[selected] = selected_residual.to(value.dtype)
    result = M03RV11ResidualResult(
        residual=residual,
        qualified_asset_mask=operator.qualified_asset_mask.to(device=value.device),
        operator_receipt_sha256=operator.receipt_sha256,
        weighted_exposure_error=exposure_error,
    )
    result.validate(operator)
    return result


__all__ = [
    "M03R_V11_RESIDUAL_OPERATOR_SCHEMA",
    "M03R_V11_RESIDUAL_RESULT_SCHEMA",
    "M03RV11ResidualOperator",
    "M03RV11ResidualOperatorError",
    "M03RV11ResidualResult",
    "apply_m03r_v11_residual_operator",
    "build_m03r_v11_residual_operator",
]
