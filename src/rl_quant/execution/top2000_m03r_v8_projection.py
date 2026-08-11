"""Content-bound radial risk projection for TOP2000 M03R-v8 development.

The future-selected TOP2000 compatibility route previously projected factor
exposure to exact zero.  A 1.5x relaxation of zero is not an ablation.  V8
therefore requires explicit nonzero factor/sector slabs and uses one
benchmark-radial scale that preserves the requested cross-sectional direction
while jointly satisfying those slabs, active beta, annualized tracking error,
availability, stock caps, and gross risk.

This is a research-development projector.  A caller must separately build the
manifest from training-fold-only inputs; this module does not estimate risk or
open outcome data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from rl_quant.envs.hold30 import reconcile_cash_simplex_roundoff
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ACTIVE_POLICY,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)

M03R_V8_RISK_MANIFEST_SCHEMA = "rl-quant.top2000-dev.m03r-v8-risk-manifest-v1"
M03R_V8_PROJECTION_SCHEMA = "rl-quant.top2000-dev.m03r-v8-risk-projection-v1"
_RISK_QUALIFICATION_ISSUER = object()


class M03RV8ProjectionError(ValueError):
    """Risk inputs or the projected book violate the v8 contract."""


def _digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV8ProjectionError("identity must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _finite_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype not in {torch.float32, torch.float64}
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV8ProjectionError(
            f"{name} must be detached finite float32/float64 {shape}"
        )
    return value


@dataclass(frozen=True, slots=True)
class M03RV8QualifiedRiskManifest:
    """Immutable-content training-fold risk inputs for one asset axis."""

    asset_count: int
    cash_index: int
    exposure_names: tuple[str, ...]
    asset_axis_sha256: str
    source_receipt_sha256: str
    exposure_loadings: torch.Tensor  # [asset, exposure]
    exposure_lower_bounds: torch.Tensor  # [exposure]
    exposure_upper_bounds: torch.Tensor  # [exposure]
    active_beta_loadings: torch.Tensor  # [asset]
    daily_return_covariance: torch.Tensor  # [asset, asset]
    tensor_sha256: tuple[str, str, str, str, str]
    minimum_covariance_eigenvalue: float
    manifest_sha256: str
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_RISK_MANIFEST_SCHEMA
    _qualification_issuer: object | None = field(
        repr=False,
        compare=False,
        default=None,
    )

    def validate(self) -> None:
        if (
            isinstance(self.asset_count, bool)
            or not isinstance(self.asset_count, int)
            or self.asset_count < 3
            or isinstance(self.cash_index, bool)
            or not isinstance(self.cash_index, int)
            or not 0 <= self.cash_index < self.asset_count
            or not self.exposure_names
            or len(set(self.exposure_names)) != len(self.exposure_names)
            or any(not name for name in self.exposure_names)
            or self._qualification_issuer is not _RISK_QUALIFICATION_ISSUER
        ):
            raise M03RV8ProjectionError("risk-manifest axis identity is invalid")
        constraints = len(self.exposure_names)
        loadings = _finite_tensor(
            "exposure_loadings",
            self.exposure_loadings,
            (self.asset_count, constraints),
        )
        lower = _finite_tensor(
            "exposure_lower_bounds",
            self.exposure_lower_bounds,
            (constraints,),
        )
        upper = _finite_tensor(
            "exposure_upper_bounds",
            self.exposure_upper_bounds,
            (constraints,),
        )
        beta = _finite_tensor(
            "active_beta_loadings",
            self.active_beta_loadings,
            (self.asset_count,),
        )
        covariance = _finite_tensor(
            "daily_return_covariance",
            self.daily_return_covariance,
            (self.asset_count, self.asset_count),
        )
        if (
            bool((lower > 0.0).any())
            or bool((upper < 0.0).any())
            or bool((lower > upper).any())
            or not bool(((lower < 0.0) | (upper > 0.0)).any())
            or not torch.equal(
                loadings[self.cash_index], torch.zeros_like(loadings[self.cash_index])
            )
            or beta[self.cash_index].item() != 0.0
            or not torch.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-12)
        ):
            raise M03RV8ProjectionError(
                "risk slabs must contain zero and be nonzero; CASH and covariance drifted"
            )
        if (
            not math.isfinite(self.minimum_covariance_eigenvalue)
            or self.minimum_covariance_eigenvalue < -1.0e-10
        ):
            raise M03RV8ProjectionError(
                "daily covariance must be positive semidefinite"
            )
        observed_hashes = tuple(
            _tensor_sha256(value)
            for value in (loadings, lower, upper, beta, covariance)
        )
        if observed_hashes != self.tensor_sha256:
            raise M03RV8ProjectionError("risk-manifest tensor content changed")
        unsigned = {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "asset_count": self.asset_count,
            "cash_index": self.cash_index,
            "exposure_names": self.exposure_names,
            "asset_axis_sha256": self.asset_axis_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "tensor_sha256": self.tensor_sha256,
            "minimum_covariance_eigenvalue_float64_hex": float(
                self.minimum_covariance_eigenvalue
            ).hex(),
            "annual_tracking_error_ceiling": (
                M03R_V8_ACTIVE_POLICY.annual_tracking_error_ceiling
            ),
            "active_beta_absolute_bound": (
                M03R_V8_ACTIVE_POLICY.active_beta_equivalence_absolute_upper_bound
            ),
            "maximum_stock_weight_fraction": (
                M03R_V8_ACTIVE_POLICY.maximum_stock_weight_fraction
            ),
            "projection_mode": M03R_V8_ACTIVE_POLICY.projection_mode,
        }
        if (
            self.schema != M03R_V8_RISK_MANIFEST_SCHEMA
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or _digest(self.asset_axis_sha256) != self.asset_axis_sha256
            or _digest(self.source_receipt_sha256) != self.source_receipt_sha256
            or _payload_sha256(unsigned) != self.manifest_sha256
        ):
            raise M03RV8ProjectionError("risk-manifest receipt identity drifted")


def qualify_m03r_v8_risk_manifest(
    *,
    exposure_names: tuple[str, ...],
    asset_axis_sha256: str,
    source_receipt_sha256: str,
    exposure_loadings: torch.Tensor,
    exposure_lower_bounds: torch.Tensor,
    exposure_upper_bounds: torch.Tensor,
    active_beta_loadings: torch.Tensor,
    daily_return_covariance: torch.Tensor,
    cash_index: int,
) -> M03RV8QualifiedRiskManifest:
    """Clone, content-address, and numerically qualify one risk manifest."""

    tensors = tuple(
        value.detach().to(device="cpu", dtype=torch.float64).clone()
        for value in (
            exposure_loadings,
            exposure_lower_bounds,
            exposure_upper_bounds,
            active_beta_loadings,
            daily_return_covariance,
        )
    )
    loadings, lower, upper, beta, covariance = tensors
    asset_count = int(loadings.shape[0]) if loadings.ndim == 2 else -1
    tensor_hashes = tuple(_tensor_sha256(value) for value in tensors)
    minimum_eigenvalue = (
        float(torch.linalg.eigvalsh(covariance).min())
        if covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1]
        else float("nan")
    )
    unsigned = {
        "schema": M03R_V8_RISK_MANIFEST_SCHEMA,
        "protocol_sha256": M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
        "asset_count": asset_count,
        "cash_index": cash_index,
        "exposure_names": exposure_names,
        "asset_axis_sha256": _digest(asset_axis_sha256),
        "source_receipt_sha256": _digest(source_receipt_sha256),
        "tensor_sha256": tensor_hashes,
        "minimum_covariance_eigenvalue_float64_hex": float(minimum_eigenvalue).hex(),
        "annual_tracking_error_ceiling": (
            M03R_V8_ACTIVE_POLICY.annual_tracking_error_ceiling
        ),
        "active_beta_absolute_bound": (
            M03R_V8_ACTIVE_POLICY.active_beta_equivalence_absolute_upper_bound
        ),
        "maximum_stock_weight_fraction": (
            M03R_V8_ACTIVE_POLICY.maximum_stock_weight_fraction
        ),
        "projection_mode": M03R_V8_ACTIVE_POLICY.projection_mode,
    }
    qualified = M03RV8QualifiedRiskManifest(
        asset_count=asset_count,
        cash_index=cash_index,
        exposure_names=exposure_names,
        asset_axis_sha256=asset_axis_sha256,
        source_receipt_sha256=source_receipt_sha256,
        exposure_loadings=loadings,
        exposure_lower_bounds=lower,
        exposure_upper_bounds=upper,
        active_beta_loadings=beta,
        daily_return_covariance=covariance,
        tensor_sha256=tensor_hashes,  # type: ignore[arg-type]
        minimum_covariance_eigenvalue=minimum_eigenvalue,
        manifest_sha256=_payload_sha256(unsigned),
        _qualification_issuer=_RISK_QUALIFICATION_ISSUER,
    )
    qualified.validate()
    return qualified


@dataclass(frozen=True, slots=True)
class M03RV8ProjectionResult:
    """One radial projection and its pre/post risk telemetry."""

    projected_weights: torch.Tensor
    radial_scale: torch.Tensor
    requested_factor_exposure: torch.Tensor
    projected_factor_exposure: torch.Tensor
    requested_active_beta: torch.Tensor
    projected_active_beta: torch.Tensor
    requested_annual_tracking_error: torch.Tensor
    projected_annual_tracking_error: torch.Tensor
    risk_manifest_sha256: str
    factor_sector_bound_multiplier: float
    schema: str = M03R_V8_PROJECTION_SCHEMA


def _annual_tracking_error(
    active: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    variance = torch.einsum("ba,ac,bc->b", active, covariance, active)
    return torch.sqrt((252.0 * variance).clamp_min(0.0))


def project_m03r_v8_active_book(
    requested_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    risk_manifest: M03RV8QualifiedRiskManifest,
    *,
    factor_sector_bound_multiplier: float,
) -> M03RV8ProjectionResult:
    """Radially project one requested book around its feasible benchmark."""

    risk_manifest.validate()
    if factor_sector_bound_multiplier not in {
        1.0,
        M03R_V8_ACTIVE_POLICY.relaxed_factor_sector_bound_multiplier,
    }:
        raise M03RV8ProjectionError("factor/sector bound multiplier is not frozen")
    if (
        not isinstance(requested_weights, torch.Tensor)
        or requested_weights.ndim != 2
        or requested_weights.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(requested_weights).all())
    ):
        raise M03RV8ProjectionError(
            "requested_weights must be finite floating [batch,asset]"
        )
    batch, assets = requested_weights.shape
    expected = (batch, assets)
    if assets != risk_manifest.asset_count:
        raise M03RV8ProjectionError("risk manifest and requested asset axes differ")
    for name, value in (
        ("benchmark_weights", benchmark_weights),
        ("risk_asset_caps", risk_asset_caps),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or value.dtype != requested_weights.dtype
            or value.device != requested_weights.device
            or not bool(torch.isfinite(value).all())
        ):
            raise M03RV8ProjectionError(f"{name} must align with requested_weights")
    if (
        not isinstance(trade_mask, torch.Tensor)
        or tuple(trade_mask.shape) != expected
        or trade_mask.dtype != torch.bool
        or trade_mask.device != requested_weights.device
        or not isinstance(risk_gross_max, torch.Tensor)
        or tuple(risk_gross_max.shape) != (batch,)
        or risk_gross_max.dtype != requested_weights.dtype
        or risk_gross_max.device != requested_weights.device
        or not bool(torch.isfinite(risk_gross_max).all())
    ):
        raise M03RV8ProjectionError("projection masks or gross limits are misaligned")
    cash = risk_manifest.cash_index
    benchmark = reconcile_cash_simplex_roundoff(
        benchmark_weights,
        cash_index=cash,
        risky_gross_limit=risk_gross_max,
    )
    risky = torch.ones_like(requested_weights, dtype=torch.bool)
    risky[:, cash] = False
    available = trade_mask.clone()
    available[:, cash] = True
    caps = torch.minimum(
        risk_asset_caps.clamp_min(0.0),
        requested_weights.new_tensor(
            M03R_V8_ACTIVE_POLICY.maximum_stock_weight_fraction
        ),
    )
    caps[:, cash] = 1.0
    if (
        bool((benchmark < 0.0).any())
        or bool((benchmark - caps > 2.0e-6).any())
        or bool(
            (
                torch.where(available, torch.zeros_like(benchmark), benchmark) != 0.0
            ).any()
        )
    ):
        raise M03RV8ProjectionError("benchmark is infeasible under fill-time limits")
    requested = reconcile_cash_simplex_roundoff(
        requested_weights,
        cash_index=cash,
        risky_gross_limit=risk_gross_max,
    )
    active = requested - benchmark
    tiny = torch.finfo(requested.dtype).tiny
    scale = torch.ones(batch, device=requested.device, dtype=requested.dtype)

    positive_ratio = torch.where(
        active > 0.0,
        (caps - benchmark).clamp_min(0.0) / active.clamp_min(tiny),
        torch.full_like(active, torch.inf),
    )
    negative_ratio = torch.where(
        active < 0.0,
        benchmark.clamp_min(0.0) / (-active).clamp_min(tiny),
        torch.full_like(active, torch.inf),
    )
    scale = torch.minimum(scale, positive_ratio.amin(dim=-1))
    scale = torch.minimum(scale, negative_ratio.amin(dim=-1))
    unavailable_active = torch.where(
        available,
        torch.zeros_like(active),
        active.abs(),
    ).amax(dim=-1)
    scale = torch.where(unavailable_active > 0.0, torch.zeros_like(scale), scale)
    benchmark_gross = torch.where(risky, benchmark, torch.zeros_like(benchmark)).sum(
        dim=-1
    )
    active_gross = torch.where(risky, active, torch.zeros_like(active)).sum(dim=-1)
    gross_ratio = torch.where(
        active_gross > 0.0,
        (risk_gross_max - benchmark_gross).clamp_min(0.0)
        / active_gross.clamp_min(tiny),
        torch.full_like(active_gross, torch.inf),
    )
    scale = torch.minimum(scale, gross_ratio)

    work_dtype = torch.float64
    loadings = risk_manifest.exposure_loadings.to(
        device=requested.device,
        dtype=work_dtype,
    )
    active64 = active.to(work_dtype)
    factor = active64 @ loadings
    lower = risk_manifest.exposure_lower_bounds.to(requested.device) * float(
        factor_sector_bound_multiplier
    )
    upper = risk_manifest.exposure_upper_bounds.to(requested.device) * float(
        factor_sector_bound_multiplier
    )
    factor_ratio = torch.where(
        factor > 0.0,
        upper.unsqueeze(0) / factor.clamp_min(torch.finfo(work_dtype).tiny),
        torch.where(
            factor < 0.0,
            lower.unsqueeze(0) / factor.clamp_max(-torch.finfo(work_dtype).tiny),
            torch.full_like(factor, torch.inf),
        ),
    )
    scale = torch.minimum(scale, factor_ratio.amin(dim=-1).to(scale.dtype))

    beta_loadings = risk_manifest.active_beta_loadings.to(requested.device)
    beta = active64 @ beta_loadings
    beta_ratio = torch.where(
        beta.abs() > 0.0,
        beta.new_tensor(
            M03R_V8_ACTIVE_POLICY.active_beta_equivalence_absolute_upper_bound
        )
        / beta.abs().clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(beta, torch.inf),
    )
    scale = torch.minimum(scale, beta_ratio.to(scale.dtype))
    covariance = risk_manifest.daily_return_covariance.to(requested.device)
    tracking_error = _annual_tracking_error(active64, covariance)
    te_ratio = torch.where(
        tracking_error > 0.0,
        tracking_error.new_tensor(M03R_V8_ACTIVE_POLICY.annual_tracking_error_ceiling)
        / tracking_error.clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(tracking_error, torch.inf),
    )
    scale = torch.minimum(scale, te_ratio.to(scale.dtype)).clamp(0.0, 1.0)
    guard = 8.0 * torch.finfo(requested.dtype).eps
    scale = torch.where(scale < 1.0, scale * (1.0 - guard), scale)
    projected = benchmark + scale.unsqueeze(-1) * active
    projected = reconcile_cash_simplex_roundoff(
        projected,
        cash_index=cash,
        risky_gross_limit=risk_gross_max,
    )
    projected_active64 = (projected - benchmark).to(work_dtype)
    projected_factor = projected_active64 @ loadings
    projected_beta = projected_active64 @ beta_loadings
    projected_te = _annual_tracking_error(projected_active64, covariance)
    tolerance = 2.0e-6
    if (
        bool((projected < -tolerance).any())
        or bool((projected - caps > tolerance).any())
        or bool((projected_factor - upper.unsqueeze(0) > tolerance).any())
        or bool((lower.unsqueeze(0) - projected_factor > tolerance).any())
        or bool(
            (
                projected_beta.abs()
                > M03R_V8_ACTIVE_POLICY.active_beta_equivalence_absolute_upper_bound
                + tolerance
            ).any()
        )
        or bool(
            (
                projected_te
                > M03R_V8_ACTIVE_POLICY.annual_tracking_error_ceiling + tolerance
            ).any()
        )
    ):
        raise M03RV8ProjectionError("radial v8 risk projection failed reconciliation")
    return M03RV8ProjectionResult(
        projected_weights=projected,
        radial_scale=scale,
        requested_factor_exposure=factor,
        projected_factor_exposure=projected_factor,
        requested_active_beta=beta,
        projected_active_beta=projected_beta,
        requested_annual_tracking_error=tracking_error,
        projected_annual_tracking_error=projected_te,
        risk_manifest_sha256=risk_manifest.manifest_sha256,
        factor_sector_bound_multiplier=float(factor_sector_bound_multiplier),
    )


__all__ = [
    "M03R_V8_PROJECTION_SCHEMA",
    "M03R_V8_RISK_MANIFEST_SCHEMA",
    "M03RV8ProjectionError",
    "M03RV8ProjectionResult",
    "M03RV8QualifiedRiskManifest",
    "project_m03r_v8_active_book",
    "qualify_m03r_v8_risk_manifest",
]
