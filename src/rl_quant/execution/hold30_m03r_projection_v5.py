"""Fail-closed M03R v5 post-ensemble risk projection and execution.

This is intentionally separate from the frozen V2/V3 portfolio builders.  It
aggregates the five M03R seed intents once, constructs one age-aware requested
book, separately projects the confidence-independent learned-exit anchor and
the full replacement proposal onto the manifest-bound linear feasible set,
applies the 6% hard tracking-error ceiling to both, confidence-limits only the
replacement move between those feasible books, and finally turnover-limits the
move.
Ordinary pretrade drift is repaired first as separately accounted risk-forced
turnover, including exact age-ledger reconciliation. Convex interpolation
preserves every hard constraint.

The ensemble's ``active_risk_scale`` is a confidence-dependent budget for new
or enlarged active risk, not a compulsory tracking-error target and not a
liquidation order. Existing feasible active risk is carried unless the learned
hazard or a separately governed de-risk request releases it. The fixed 6%
limit remains an independent safety ceiling for repaired and executed books.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from rl_quant.models.daily_policy import hold30_release_hazard
from rl_quant.models.hold30_m03r_ensemble_v5 import (
    M03REnsembleIntent,
    M03REnsembleMember,
    M03RSeedCheckpointEnsembleManifest,
    aggregate_m03r_alpha_intents,
    compute_m03r_asset_order_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN,
    validate_m03r_artifact_identity,
)

M03R_RISK_MANIFEST_SCHEMA = M03R_DESIGN.ensemble_execution.risk_manifest_schema
M03R_REQUIRED_EXPOSURE_FAMILIES = frozenset(
    M03R_DESIGN.factor_sector_projection.exposure_families
)
M03R_ACTIVE_BETA_EXPOSURE_NAME = "active_market_beta"
M03R_ANNUAL_TRACKING_ERROR_CEILING = (
    M03R_DESIGN.active_risk.annual_tracking_error_ceiling
)
M03R_ACTIVE_BETA_ABSOLUTE_MAXIMUM = (
    M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
)
M03R_MAXIMUM_RISKY_ASSET_WEIGHT = M03R_DESIGN.active_risk.maximum_asset_weight_fraction
M03R_OBJECTIVE_RISK_CONTRACT_SCHEMA = "rl-quant.m03r-v5-objective-risk-contract-v1"
M03R_MAXIMUM_CONFIDENCE_INCREMENTAL_ONE_WAY_TURNOVER = (
    M03R_DESIGN.active_risk.maximum_confidence_incremental_one_way_turnover
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_FINGERPRINT_DOMAIN = b"rl-quant.m03r-v5-qualified-tensor-v1\x00"


class M03RProjectionError(ValueError):
    """An M03R projection input is unbound, infeasible, or unconverged."""


_M03R_QUALIFICATION_ISSUER = object()


def _canonical_tensor_content_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor metadata and values in a portable float64 encoding."""

    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = json.dumps(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
            "value_encoding": "little-endian-float64-c-order",
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    values = np.asarray(canonical.numpy(), dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(_TENSOR_FINGERPRINT_DOMAIN)
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _M03RQualificationCapability:
    """Process-local proof issued only after full manifest validation."""

    issuer: object
    qualified_object_id: int
    manifest_object_id: int
    tensor_object_ids: tuple[int, ...]
    manifest_sha256: str
    asset_order_sha256: str
    sorted_asset_indices: tuple[int, ...]
    tensor_versions: tuple[int, ...]
    tensor_content_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class M03RRiskManifest:
    """Point-in-time factor slabs and return covariance in one bound payload."""

    schema: str
    as_of_trading_session: str
    asset_ids: tuple[str, ...]
    exposure_names: tuple[str, ...]
    exposure_families: tuple[str, ...]
    exposure_units: tuple[str, ...]
    exposure_normalization_ids: tuple[str, ...]
    exposure_estimation_window_trading_sessions: int
    missing_value_policy: str
    covariance_estimator_id: str
    covariance_shrinkage_id: str
    covariance_return_convention: str
    stale_loading_policy: str
    infeasibility_policy: str
    exposure_loadings: torch.Tensor
    exposure_lower_bounds: torch.Tensor
    exposure_upper_bounds: torch.Tensor
    daily_return_covariance: torch.Tensor
    annual_tracking_error_ceiling: float
    maximum_risky_asset_weight: float
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class M03RQualifiedRiskManifest:
    """Once-validated immutable-by-capability risk inputs for many decisions.

    Qualification clones every tensor, performs the PSD eigendecomposition
    once, and issues a process-local capability bound to exact object identities,
    mutation counters, and canonical tensor-content fingerprints. The capability
    is intentionally non-serializable: external or restored inputs must qualify
    again. Each governed trust boundary rehashes tensor content without repeating
    covariance eigendecomposition or other numerical qualification.
    """

    manifest: M03RRiskManifest
    manifest_sha256: str
    asset_order_sha256: str
    sorted_asset_indices: tuple[int, ...]
    covariance_factor: torch.Tensor
    tensor_versions: tuple[int, ...]
    _qualification_capability: object = field(repr=False, compare=False)

    def assert_unmodified(self) -> None:
        capability = self._qualification_capability
        if (
            not isinstance(capability, _M03RQualificationCapability)
            or capability.issuer is not _M03R_QUALIFICATION_ISSUER
            or capability.qualified_object_id != id(self)
            or capability.manifest_object_id != id(self.manifest)
            or capability.manifest_sha256 != self.manifest_sha256
            or capability.asset_order_sha256 != self.asset_order_sha256
            or capability.sorted_asset_indices != self.sorted_asset_indices
            or capability.tensor_versions != self.tensor_versions
        ):
            raise M03RProjectionError(
                "risk-manifest qualification capability is absent or no longer bound"
            )
        tensors = (
            self.manifest.exposure_loadings,
            self.manifest.exposure_lower_bounds,
            self.manifest.exposure_upper_bounds,
            self.manifest.daily_return_covariance,
            self.covariance_factor,
        )
        if tuple(id(value) for value in tensors) != capability.tensor_object_ids:
            raise M03RProjectionError(
                "qualified risk-manifest tensor identities changed after qualification"
            )
        if tuple(int(value._version) for value in tensors) != self.tensor_versions:
            raise M03RProjectionError(
                "qualified risk-manifest tensors changed after qualification"
            )
        if (
            tuple(_canonical_tensor_content_sha256(value) for value in tensors)
            != capability.tensor_content_sha256s
        ):
            raise M03RProjectionError(
                "qualified risk-manifest tensor content changed after qualification"
            )


@dataclass(frozen=True, slots=True)
class M03RObjectiveRiskContract:
    """Exact ordered risk-manifest inventory shared with the training objective.

    The objective derives active exposures from bound policy/benchmark weights
    and the exact point-in-time loading matrices. This compact contract binds
    every observation to that source manifest and freezes the exposure axis,
    units, normalization, and asymmetric slabs shared with execution.
    """

    schema: str
    risk_manifest_schema: str
    ordered_risk_manifest_sha256s: tuple[str, ...]
    exposure_names: tuple[str, ...]
    exposure_families: tuple[str, ...]
    exposure_units: tuple[str, ...]
    exposure_normalization_ids: tuple[str, ...]
    exposure_lower_bounds: tuple[float, ...]
    exposure_upper_bounds: tuple[float, ...]
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class M03RAssetAlignedBook:
    """One decision book with an explicit cryptographic asset-axis binding."""

    decision_trading_session: str
    asset_ids: tuple[str, ...]
    asset_order_sha256: str
    current_weights: torch.Tensor
    benchmark_weights: torch.Tensor
    decision_available: torch.Tensor
    age_notional: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decision_trading_session, str)
            or not self.decision_trading_session.strip()
        ):
            raise M03RProjectionError("decision_trading_session is required")
        expected = compute_m03r_asset_order_sha256(self.asset_ids)
        if self.asset_order_sha256 != expected:
            raise M03RProjectionError("book asset_order_sha256 does not match asset_ids")


@dataclass(frozen=True, slots=True)
class M03RProjectionNumerics:
    """Non-economic convergence controls for deterministic Dykstra projection."""

    tolerance: float = M03R_DESIGN.ensemble_execution.projection_tolerance
    maximum_iterations: int = (
        M03R_DESIGN.ensemble_execution.projection_maximum_iterations
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance <= 1e-6:
            raise M03RProjectionError("projection tolerance must lie in (0, 1e-6]")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise M03RProjectionError("maximum_iterations must be a positive integer")


M03R_DEFAULT_PROJECTION_NUMERICS = M03RProjectionNumerics()


@dataclass(frozen=True, slots=True)
class M03RProjectionDiagnostics:
    projection_application_count: int
    hazard_anchor_solver_iterations: int
    hazard_anchor_solver_converged: bool
    hazard_anchor_maximum_linear_violation: float
    hazard_anchor_annual_tracking_error: float
    hazard_anchor_tracking_error_scale: float
    hazard_anchor_projection_l2_distance: float
    risk_forced_repair_application_count: int
    risk_forced_repair_solver_iterations: int
    risk_forced_repair_solver_converged: bool
    pre_repair_maximum_linear_violation: float
    risk_forced_repair_maximum_violation: float
    solver_iterations: int
    solver_converged: bool
    maximum_linear_violation: float
    maximum_final_violation: float
    requested_active_exposures: torch.Tensor
    pre_repair_active_exposures: torch.Tensor
    repaired_active_exposures: torch.Tensor
    linear_projected_active_exposures: torch.Tensor
    projected_active_exposures: torch.Tensor
    final_active_exposures: torch.Tensor
    current_annual_tracking_error: float
    repaired_current_annual_tracking_error: float
    requested_annual_tracking_error: float
    linear_projected_annual_tracking_error: float
    projected_annual_tracking_error: float
    final_annual_tracking_error: float
    preferred_annual_tracking_error_cap: float
    effective_annual_tracking_error_cap: float
    tracking_error_scale: float
    confidence_tracking_error_scale: float
    risk_forced_repair_tracking_error_scale: float
    risk_forced_repair_l2_distance: float
    risk_forced_repair_one_way_turnover: float
    risk_forced_repair_sell_notional: float
    risk_forced_repair_buy_notional: float
    unavailable_forced_one_way_turnover: float
    unavailable_forced_sell_notional: float
    unavailable_forced_buy_notional: float
    requested_incremental_active_risk: float
    confidence_limited_incremental_active_risk: float
    requested_incremental_one_way_turnover: float
    confidence_limited_incremental_one_way_turnover: float
    confidence_incremental_one_way_turnover_cap: float
    confidence_covariance_risk_scale: float
    confidence_one_way_turnover_scale: float
    confidence_incremental_risk_scale: float
    linear_projection_l2_distance: float
    projected_l2_distance: float
    final_l2_distance: float
    requested_one_way_turnover: float
    executed_one_way_turnover: float
    turnover_interpolation_scale: float


@dataclass(frozen=True, slots=True)
class M03RExecutionResult:
    """One M03R chronological decision after ensemble and hard risk controls."""

    ensemble: M03REnsembleIntent
    availability_repaired_weights: torch.Tensor
    repaired_current_weights: torch.Tensor
    repaired_age_notional: torch.Tensor
    raw_hazard_anchor_weights: torch.Tensor
    projected_hazard_anchor_weights: torch.Tensor
    requested_weights: torch.Tensor
    projected_weights: torch.Tensor
    executed_weights: torch.Tensor
    final_age_notional: torch.Tensor
    unavailable_delta: torch.Tensor
    risk_repair_delta: torch.Tensor
    hazard_release_delta: torch.Tensor
    benchmark_derisk_delta: torch.Tensor
    hazard_anchor_factor_projection_delta: torch.Tensor
    hazard_anchor_tracking_error_projection_delta: torch.Tensor
    entry_reallocation_delta: torch.Tensor
    factor_projection_delta: torch.Tensor
    tracking_error_projection_delta: torch.Tensor
    confidence_budget_delta: torch.Tensor
    turnover_truncation_delta: torch.Tensor
    executed_unavailable_sell_notional: torch.Tensor
    executed_unavailable_sale_age_notional: torch.Tensor
    executed_risk_repair_sell_notional: torch.Tensor
    executed_risk_repair_sale_age_notional: torch.Tensor
    executed_learned_hazard_sell_notional: torch.Tensor
    executed_learned_hazard_sale_age_notional: torch.Tensor
    executed_entry_buy_notional: torch.Tensor
    executed_benchmark_derisk_sell_notional: torch.Tensor
    executed_benchmark_derisk_sale_age_notional: torch.Tensor
    executed_projection_sell_notional: torch.Tensor
    executed_projection_sale_age_notional: torch.Tensor
    executed_projection_buy_notional: torch.Tensor
    executed_total_sale_age_notional: torch.Tensor
    diagnostics: M03RProjectionDiagnostics


def _tensor_payload(value: torch.Tensor) -> dict[str, object]:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise M03RProjectionError("risk-manifest tensors must be floating point")
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise M03RProjectionError("risk-manifest tensors must be finite")
    return {
        "shape": list(value.shape),
        "float64_hex_values": [float(item).hex() for item in flat.tolist()],
    }


def m03r_risk_manifest_payload(manifest: M03RRiskManifest) -> dict[str, object]:
    """Return the canonical semantic payload (excluding its claimed digest)."""

    return {
        "schema": manifest.schema,
        "as_of_trading_session": manifest.as_of_trading_session,
        "asset_ids": list(manifest.asset_ids),
        "exposure_names": list(manifest.exposure_names),
        "exposure_families": list(manifest.exposure_families),
        "exposure_units": list(manifest.exposure_units),
        "exposure_normalization_ids": list(manifest.exposure_normalization_ids),
        "exposure_estimation_window_trading_sessions": (
            manifest.exposure_estimation_window_trading_sessions
        ),
        "missing_value_policy": manifest.missing_value_policy,
        "covariance_estimator_id": manifest.covariance_estimator_id,
        "covariance_shrinkage_id": manifest.covariance_shrinkage_id,
        "covariance_return_convention": manifest.covariance_return_convention,
        "stale_loading_policy": manifest.stale_loading_policy,
        "infeasibility_policy": manifest.infeasibility_policy,
        "exposure_loadings": _tensor_payload(manifest.exposure_loadings),
        "exposure_lower_bounds": _tensor_payload(manifest.exposure_lower_bounds),
        "exposure_upper_bounds": _tensor_payload(manifest.exposure_upper_bounds),
        "daily_return_covariance": _tensor_payload(manifest.daily_return_covariance),
        "annual_tracking_error_ceiling": float(
            manifest.annual_tracking_error_ceiling
        ).hex(),
        "maximum_risky_asset_weight": float(manifest.maximum_risky_asset_weight).hex(),
    }


def compute_m03r_risk_manifest_sha256(manifest: M03RRiskManifest) -> str:
    """Content-address every risk-moving manifest value."""

    encoded = json.dumps(
        m03r_risk_manifest_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_m03r_risk_manifest(**values: object) -> M03RRiskManifest:
    """Construct a manifest whose digest is derived from its complete payload."""

    provisional = M03RRiskManifest(manifest_sha256="0" * 64, **values)  # type: ignore[arg-type]
    return replace(
        provisional,
        manifest_sha256=compute_m03r_risk_manifest_sha256(provisional),
    )


def _validate_risk_manifest(
    manifest: M03RRiskManifest,
    *,
    expected_manifest_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if manifest.schema != M03R_RISK_MANIFEST_SCHEMA:
        raise M03RProjectionError("unknown M03R risk-manifest schema")
    if not manifest.as_of_trading_session.strip():
        raise M03RProjectionError("risk manifest needs a point-in-time as-of session")
    if not _DIGEST.fullmatch(expected_manifest_sha256):
        raise M03RProjectionError("expected risk-manifest SHA-256 is invalid")
    if not _DIGEST.fullmatch(manifest.manifest_sha256):
        raise M03RProjectionError("claimed risk-manifest SHA-256 is invalid")
    computed = compute_m03r_risk_manifest_sha256(manifest)
    if manifest.manifest_sha256 != computed or computed != expected_manifest_sha256:
        raise M03RProjectionError("risk-manifest content hash does not match binding")
    assets = len(manifest.asset_ids)
    constraints = len(manifest.exposure_names)
    if (
        assets < 2
        or len(set(manifest.asset_ids)) != assets
        or any(not asset_id for asset_id in manifest.asset_ids)
    ):
        raise M03RProjectionError("risk manifest needs distinct nonempty asset IDs")
    if (
        constraints == 0
        or len(set(manifest.exposure_names)) != constraints
        or len(manifest.exposure_families) != constraints
        or len(manifest.exposure_units) != constraints
        or len(manifest.exposure_normalization_ids) != constraints
    ):
        raise M03RProjectionError("risk manifest exposure identities are invalid")
    if any(
        not isinstance(value, str) or not value.strip()
        for values in (
            manifest.exposure_units,
            manifest.exposure_normalization_ids,
        )
        for value in values
    ):
        raise M03RProjectionError("risk manifest exposure metadata cannot be empty")
    if (
        isinstance(manifest.exposure_estimation_window_trading_sessions, bool)
        or not isinstance(
            manifest.exposure_estimation_window_trading_sessions,
            int,
        )
        or manifest.exposure_estimation_window_trading_sessions <= 0
    ):
        raise M03RProjectionError("exposure estimation window must be positive")
    for name in (
        "missing_value_policy",
        "covariance_estimator_id",
        "covariance_shrinkage_id",
        "covariance_return_convention",
        "stale_loading_policy",
        "infeasibility_policy",
    ):
        value = getattr(manifest, name)
        if not isinstance(value, str) or not value.strip():
            raise M03RProjectionError(f"risk manifest {name} is required")
    if manifest.infeasibility_policy != (
        M03R_DESIGN.factor_sector_projection.infeasible_projection_behavior
    ):
        raise M03RProjectionError("risk manifest infeasibility policy drifted")
    if not M03R_REQUIRED_EXPOSURE_FAMILIES.issubset(set(manifest.exposure_families)):
        raise M03RProjectionError(
            "risk manifest omits a required PIT factor/sector exposure family"
        )
    if tuple(manifest.exposure_loadings.shape) != (constraints, assets):
        raise M03RProjectionError("exposure_loadings must be [constraint, asset]")
    if tuple(manifest.exposure_lower_bounds.shape) != (constraints,) or tuple(
        manifest.exposure_upper_bounds.shape
    ) != (constraints,):
        raise M03RProjectionError("exposure bounds must be [constraint]")
    if tuple(manifest.daily_return_covariance.shape) != (assets, assets):
        raise M03RProjectionError("daily_return_covariance must be [asset, asset]")
    tensors = (
        manifest.exposure_loadings,
        manifest.exposure_lower_bounds,
        manifest.exposure_upper_bounds,
        manifest.daily_return_covariance,
    )
    if any(
        not value.is_floating_point() or not bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise M03RProjectionError("risk manifest contains non-finite tensors")
    lower = manifest.exposure_lower_bounds.detach().to(torch.float64)
    upper = manifest.exposure_upper_bounds.detach().to(torch.float64)
    if (
        bool((lower > upper).any())
        or bool((lower > 0).any())
        or bool((upper < 0).any())
    ):
        raise M03RProjectionError(
            "every active-exposure slab must be ordered and contain benchmark zero"
        )
    try:
        beta_index = manifest.exposure_names.index(M03R_ACTIVE_BETA_EXPOSURE_NAME)
    except ValueError as exc:
        raise M03RProjectionError(
            "risk manifest omits active-market-beta slab"
        ) from exc
    if manifest.exposure_families[beta_index] != "market" or not (
        math.isclose(float(lower[beta_index]), -M03R_ACTIVE_BETA_ABSOLUTE_MAXIMUM)
        and math.isclose(float(upper[beta_index]), M03R_ACTIVE_BETA_ABSOLUTE_MAXIMUM)
    ):
        raise M03RProjectionError(
            "active-market-beta slab must be exactly [-0.10, 0.10]"
        )
    if not math.isclose(
        manifest.annual_tracking_error_ceiling,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
    ):
        raise M03RProjectionError("M03R annual tracking-error ceiling must be 6%")
    if not math.isclose(
        manifest.maximum_risky_asset_weight,
        M03R_MAXIMUM_RISKY_ASSET_WEIGHT,
    ):
        raise M03RProjectionError("M03R risky-asset cap must be 1%")
    covariance = manifest.daily_return_covariance.detach().to(torch.float64)
    if not bool(torch.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10)):
        raise M03RProjectionError("daily return covariance must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -1e-12 or float(eigenvalues.max()) <= 0:
        raise M03RProjectionError(
            "daily return covariance must be nonzero positive semidefinite"
        )
    return eigenvalues, eigenvectors


def qualify_m03r_risk_manifest(
    manifest: M03RRiskManifest,
    *,
    expected_manifest_sha256: str,
) -> M03RQualifiedRiskManifest:
    """Validate, clone, and pre-factor one point-in-time risk manifest."""

    eigenvalues, eigenvectors = _validate_risk_manifest(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    cloned = replace(
        manifest,
        exposure_loadings=manifest.exposure_loadings.detach().clone(),
        exposure_lower_bounds=manifest.exposure_lower_bounds.detach().clone(),
        exposure_upper_bounds=manifest.exposure_upper_bounds.detach().clone(),
        daily_return_covariance=manifest.daily_return_covariance.detach().clone(),
    )
    # Q diag(sqrt(lambda)) is a valid factor for a PSD covariance, including
    # the expected zero-variance cash coordinate.
    factor = eigenvectors @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())
    order = tuple(
        sorted(range(len(cloned.asset_ids)), key=lambda index: cloned.asset_ids[index])
    )
    tensors = (
        cloned.exposure_loadings,
        cloned.exposure_lower_bounds,
        cloned.exposure_upper_bounds,
        cloned.daily_return_covariance,
        factor,
    )
    manifest_sha256 = cloned.manifest_sha256
    asset_order_sha256 = compute_m03r_asset_order_sha256(cloned.asset_ids)
    versions = tuple(int(value._version) for value in tensors)
    content_sha256s = tuple(
        _canonical_tensor_content_sha256(value) for value in tensors
    )
    capability = _M03RQualificationCapability(
        issuer=_M03R_QUALIFICATION_ISSUER,
        qualified_object_id=0,
        manifest_object_id=id(cloned),
        tensor_object_ids=tuple(id(value) for value in tensors),
        manifest_sha256=manifest_sha256,
        asset_order_sha256=asset_order_sha256,
        sorted_asset_indices=order,
        tensor_versions=versions,
        tensor_content_sha256s=content_sha256s,
    )
    qualified = M03RQualifiedRiskManifest(
        manifest=cloned,
        manifest_sha256=manifest_sha256,
        asset_order_sha256=asset_order_sha256,
        sorted_asset_indices=order,
        covariance_factor=factor,
        tensor_versions=versions,
        _qualification_capability=capability,
    )
    object.__setattr__(capability, "qualified_object_id", id(qualified))
    validate_m03r_qualified_risk_manifest(
        qualified,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return qualified


def validate_m03r_qualified_risk_manifest(
    qualified: M03RQualifiedRiskManifest,
    *,
    expected_manifest_sha256: str,
) -> None:
    """Revalidate a qualified wrapper at every governed trust boundary.

    ``frozen=True`` is not an authorization primitive: a caller can fabricate
    a dataclass instance or deserialize stale precomputation. Qualification
    performs numerical qualification once; this boundary then requires the
    process-local capability and proves that every cached object, digest, order,
    mutation counter, and canonical tensor-content fingerprint still matches
    it. Fingerprint revalidation is content-linear; covariance eigendecomposition
    remains a one-time qualification cost.
    """

    if not isinstance(qualified, M03RQualifiedRiskManifest):
        raise M03RProjectionError("risk-manifest qualification must be typed")
    if not _DIGEST.fullmatch(expected_manifest_sha256):
        raise M03RProjectionError("expected risk-manifest SHA-256 is invalid")
    qualified.assert_unmodified()
    if qualified.manifest_sha256 != expected_manifest_sha256:
        raise M03RProjectionError("qualified risk-manifest binding mismatch")


def m03r_objective_risk_contract_payload(
    contract: M03RObjectiveRiskContract,
) -> dict[str, object]:
    """Canonical payload for the compact objective/execution risk bridge."""

    return {
        "schema": contract.schema,
        "risk_manifest_schema": contract.risk_manifest_schema,
        "ordered_risk_manifest_sha256s": list(
            contract.ordered_risk_manifest_sha256s
        ),
        "exposure_names": list(contract.exposure_names),
        "exposure_families": list(contract.exposure_families),
        "exposure_units": list(contract.exposure_units),
        "exposure_normalization_ids": list(contract.exposure_normalization_ids),
        "exposure_lower_bounds_float64_hex": [
            float(value).hex() for value in contract.exposure_lower_bounds
        ],
        "exposure_upper_bounds_float64_hex": [
            float(value).hex() for value in contract.exposure_upper_bounds
        ],
    }


def compute_m03r_objective_risk_contract_sha256(
    contract: M03RObjectiveRiskContract,
) -> str:
    encoded = json.dumps(
        m03r_objective_risk_contract_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_m03r_objective_risk_contract(
    contract: M03RObjectiveRiskContract,
) -> None:
    """Reject a risk bridge unless every identity and slab is self-consistent."""

    if not isinstance(contract, M03RObjectiveRiskContract):
        raise M03RProjectionError("objective risk contract must be typed")
    if contract.schema != M03R_OBJECTIVE_RISK_CONTRACT_SCHEMA:
        raise M03RProjectionError("unknown M03R objective risk-contract schema")
    if contract.risk_manifest_schema != M03R_RISK_MANIFEST_SCHEMA:
        raise M03RProjectionError("objective risk-manifest schema drifted")
    if not contract.ordered_risk_manifest_sha256s or any(
        not isinstance(value, str) or not _DIGEST.fullmatch(value)
        for value in contract.ordered_risk_manifest_sha256s
    ):
        raise M03RProjectionError(
            "objective risk contract needs an ordered manifest digest per observation"
        )
    width = len(contract.exposure_names)
    aligned = (
        contract.exposure_families,
        contract.exposure_units,
        contract.exposure_normalization_ids,
        contract.exposure_lower_bounds,
        contract.exposure_upper_bounds,
    )
    if (
        width == 0
        or len(set(contract.exposure_names)) != width
        or any(len(values) != width for values in aligned)
    ):
        raise M03RProjectionError(
            "objective risk exposure identities and bounds must align exactly"
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for values in (
            contract.exposure_names,
            contract.exposure_families,
            contract.exposure_units,
            contract.exposure_normalization_ids,
        )
        for value in values
    ):
        raise M03RProjectionError("objective risk exposure metadata cannot be empty")
    for lower, upper in zip(
        contract.exposure_lower_bounds,
        contract.exposure_upper_bounds,
        strict=True,
    ):
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
            or float(lower) > 0.0
            or float(upper) < 0.0
        ):
            raise M03RProjectionError(
                "objective risk slabs must be finite, ordered, and contain zero"
            )
    if (
        not isinstance(contract.contract_sha256, str)
        or not _DIGEST.fullmatch(contract.contract_sha256)
        or (
        contract.contract_sha256
        != compute_m03r_objective_risk_contract_sha256(contract)
        )
    ):
        raise M03RProjectionError("objective risk-contract content hash mismatch")


def bind_m03r_objective_risk_contract(
    qualified_manifests: Sequence[M03RQualifiedRiskManifest],
) -> M03RObjectiveRiskContract:
    """Bind an ordered objective batch to exact qualified PIT risk manifests."""

    if not qualified_manifests:
        raise M03RProjectionError(
            "objective risk contract needs at least one qualified manifest"
        )
    manifests: list[M03RRiskManifest] = []
    for qualified in qualified_manifests:
        if not isinstance(qualified, M03RQualifiedRiskManifest):
            raise M03RProjectionError(
                "objective risk contract accepts only qualified risk manifests"
            )
        validate_m03r_qualified_risk_manifest(
            qualified,
            expected_manifest_sha256=qualified.manifest_sha256,
        )
        manifest = qualified.manifest
        if (
            manifest.schema != M03R_RISK_MANIFEST_SCHEMA
            or qualified.manifest_sha256 != manifest.manifest_sha256
            or compute_m03r_risk_manifest_sha256(manifest)
            != qualified.manifest_sha256
        ):
            raise M03RProjectionError(
                "qualified objective risk-manifest content binding drifted"
            )
        manifests.append(manifest)

    reference = manifests[0]
    reference_lower = tuple(
        float(value)
        for value in reference.exposure_lower_bounds.detach()
        .to(device="cpu", dtype=torch.float64)
        .tolist()
    )
    reference_upper = tuple(
        float(value)
        for value in reference.exposure_upper_bounds.detach()
        .to(device="cpu", dtype=torch.float64)
        .tolist()
    )
    reference_identity = (
        reference.exposure_names,
        reference.exposure_families,
        reference.exposure_units,
        reference.exposure_normalization_ids,
        reference_lower,
        reference_upper,
    )
    for manifest in manifests[1:]:
        observed = (
            manifest.exposure_names,
            manifest.exposure_families,
            manifest.exposure_units,
            manifest.exposure_normalization_ids,
            tuple(
                float(value)
                for value in manifest.exposure_lower_bounds.detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            ),
            tuple(
                float(value)
                for value in manifest.exposure_upper_bounds.detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            ),
        )
        if observed != reference_identity:
            raise M03RProjectionError(
                "objective risk manifests disagree on exposure identity, units, or bounds"
            )

    provisional = M03RObjectiveRiskContract(
        schema=M03R_OBJECTIVE_RISK_CONTRACT_SCHEMA,
        risk_manifest_schema=M03R_RISK_MANIFEST_SCHEMA,
        ordered_risk_manifest_sha256s=tuple(
            manifest.manifest_sha256 for manifest in manifests
        ),
        exposure_names=reference.exposure_names,
        exposure_families=reference.exposure_families,
        exposure_units=reference.exposure_units,
        exposure_normalization_ids=reference.exposure_normalization_ids,
        exposure_lower_bounds=reference_lower,
        exposure_upper_bounds=reference_upper,
        contract_sha256="0" * 64,
    )
    bound = replace(
        provisional,
        contract_sha256=compute_m03r_objective_risk_contract_sha256(provisional),
    )
    validate_m03r_objective_risk_contract(bound)
    return bound


def _as_vector(
    name: str, value: torch.Tensor, assets: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (assets,)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RProjectionError(f"{name} must be finite floating point [{assets}]")
    if value.requires_grad:
        raise M03RProjectionError(
            f"{name} must be detached for post-ensemble execution"
        )
    return value.detach().to(device=device, dtype=torch.float64)


def _annual_tracking_error(
    delta: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    variance = torch.dot(delta, covariance @ delta).clamp_min(0.0)
    return torch.sqrt(delta.new_tensor(252.0) * variance)


def _exposures(
    weights: torch.Tensor,
    benchmark: torch.Tensor,
    loadings: torch.Tensor,
) -> torch.Tensor:
    return loadings @ (weights - benchmark)


def _linear_violation(
    weights: torch.Tensor,
    benchmark: torch.Tensor,
    upper_weights: torch.Tensor,
    loadings: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    exposure = _exposures(weights, benchmark, loadings)
    pieces = torch.stack(
        (
            (weights.sum() - 1.0).abs(),
            (-weights).clamp_min(0.0).max(),
            (weights - upper_weights).clamp_min(0.0).max(),
            (lower - exposure).clamp_min(0.0).max(),
            (exposure - upper).clamp_min(0.0).max(),
        )
    )
    return pieces.max()


def _project_linear_dykstra(
    requested: torch.Tensor,
    benchmark: torch.Tensor,
    upper_weights: torch.Tensor,
    loadings: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    numerics: M03RProjectionNumerics,
) -> tuple[torch.Tensor, int, float]:
    """Euclidean projection onto sum, box, and linear-slab intersection."""

    constraint_count = 2 + loadings.shape[0]
    corrections = [torch.zeros_like(requested) for _ in range(constraint_count)]
    current = requested.clone()
    for iteration in range(1, numerics.maximum_iterations + 1):
        prior = current.clone()
        # Affine simplex hyperplane.
        shifted = current + corrections[0]
        projected = shifted - (shifted.sum() - 1.0) / shifted.numel()
        corrections[0] = shifted - projected
        current = projected
        # Long-only and per-coordinate upper bounds.
        shifted = current + corrections[1]
        projected = torch.maximum(
            torch.zeros_like(shifted), torch.minimum(shifted, upper_weights)
        )
        corrections[1] = shifted - projected
        current = projected
        # Each two-sided exposure slab is one closed convex set.
        for row in range(loadings.shape[0]):
            correction_index = row + 2
            shifted = current + corrections[correction_index]
            loading = loadings[row]
            norm_sq = torch.dot(loading, loading)
            if float(norm_sq) <= 0:
                raise M03RProjectionError("exposure loading row cannot be all zero")
            value = torch.dot(loading, shifted - benchmark)
            bounded = value.clamp(lower[row], upper[row])
            projected = shifted + ((bounded - value) / norm_sq) * loading
            corrections[correction_index] = shifted - projected
            current = projected
        violation = _linear_violation(
            current, benchmark, upper_weights, loadings, lower, upper
        )
        movement = torch.linalg.vector_norm(current - prior)
        if (
            float(violation) <= numerics.tolerance
            and float(movement) <= numerics.tolerance
        ):
            return current, iteration, float(violation)
    failure_violation = float(
        _linear_violation(current, benchmark, upper_weights, loadings, lower, upper)
    )
    raise M03RProjectionError(
        "M03R Euclidean projection did not converge: "
        f"iterations={numerics.maximum_iterations}, violation={failure_violation:.3e}"
    )


def _verify_feasible_book(
    name: str,
    weights: torch.Tensor,
    benchmark: torch.Tensor,
    upper_weights: torch.Tensor,
    loadings: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    covariance: torch.Tensor,
    te_ceiling: float,
    tolerance: float,
) -> None:
    linear = float(
        _linear_violation(weights, benchmark, upper_weights, loadings, lower, upper)
    )
    te = float(_annual_tracking_error(weights - benchmark, covariance))
    if linear > tolerance or te > te_ceiling + tolerance:
        raise M03RProjectionError(
            f"{name} is not verified feasible: linear={linear:.3e}, annual_te={te:.6f}"
        )


def _validate_pretrade_book_sanity(
    current: torch.Tensor,
    *,
    tolerance: float,
) -> None:
    """Reject corrupted books while allowing ordinary cap/risk drift."""

    if float(current.min()) < -tolerance or float(current.max()) > 1.0 + tolerance:
        raise M03RProjectionError(
            "pretrade current book must remain a long-only unit portfolio"
        )
    if abs(float(current.sum()) - 1.0) > tolerance:
        raise M03RProjectionError(
            "pretrade current book must conserve total portfolio weight"
        )


def _reconcile_age_ledger_after_transition(
    age_notional: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    cash_index: int,
    tolerance: float,
) -> torch.Tensor:
    """Apply net sells pro rata and net buys to the age-zero cohort."""

    assets = before.numel()
    if (
        tuple(age_notional.shape) != (assets, 61)
        or not age_notional.is_floating_point()
        or age_notional.requires_grad
        or not bool(torch.isfinite(age_notional).all())
        or bool((age_notional < 0).any())
    ):
        raise M03RProjectionError(
            "age_notional must be detached, finite, nonnegative [asset, 61]"
        )
    ledger = age_notional.detach().to(device=before.device, dtype=torch.float64)
    risky = torch.ones(assets, dtype=torch.bool, device=before.device)
    risky[cash_index] = False
    if float(ledger[cash_index].abs().sum()) > tolerance or not bool(
        torch.allclose(
            ledger.sum(dim=-1)[risky],
            before[risky],
            atol=tolerance,
            rtol=tolerance,
        )
    ):
        raise M03RProjectionError(
            "age ledger does not conserve the transition's input book"
        )

    after_nonnegative = after.clamp_min(0.0)
    retained_fraction = torch.where(
        before > tolerance,
        torch.minimum(
            torch.ones_like(before),
            after_nonnegative / before.clamp_min(tolerance),
        ),
        torch.zeros_like(before),
    )
    reconciled = ledger * retained_fraction.unsqueeze(-1)
    new_buys = (after_nonnegative - before).clamp_min(0.0)
    new_buys[cash_index] = 0.0
    reconciled[:, 0] = reconciled[:, 0] + new_buys
    reconciled[cash_index] = 0.0
    if not bool(
        torch.allclose(
            reconciled.sum(dim=-1)[risky],
            after_nonnegative[risky],
            atol=max(tolerance, 1e-8),
            rtol=max(tolerance, 1e-8),
        )
    ):
        raise M03RProjectionError(
            "age-ledger transition did not conserve resulting holdings"
        )
    return reconciled


def _sale_age_notional(
    age_notional: torch.Tensor,
    before: torch.Tensor,
    sell_notional: torch.Tensor,
    *,
    cash_index: int,
    tolerance: float,
) -> torch.Tensor:
    """Attribute one cause's net sells pro rata over the available age cohorts."""

    if (
        tuple(sell_notional.shape) != tuple(before.shape)
        or not sell_notional.is_floating_point()
        or not bool(torch.isfinite(sell_notional).all())
        or bool((sell_notional < 0.0).any())
        or bool((sell_notional - before.clamp_min(0.0) > tolerance).any())
    ):
        raise M03RProjectionError(
            "cause-specific sell notional must be finite, nonnegative, and funded"
        )
    fraction = torch.where(
        before > tolerance,
        sell_notional / before.clamp_min(tolerance),
        torch.zeros_like(before),
    )
    attributed = age_notional.to(device=before.device, dtype=torch.float64) * (
        fraction.unsqueeze(-1)
    )
    attributed[cash_index] = 0.0
    risky = torch.ones(before.numel(), dtype=torch.bool, device=before.device)
    risky[cash_index] = False
    if not bool(
        torch.allclose(
            attributed.sum(dim=-1)[risky],
            sell_notional[risky],
            atol=max(tolerance, 1e-8),
            rtol=max(tolerance, 1e-8),
        )
    ):
        raise M03RProjectionError(
            "cause-specific sale-age notional does not conserve sold notional"
        )
    return attributed


def _requested_weights_from_ensemble(
    ensemble: M03REnsembleIntent,
    current: torch.Tensor,
    benchmark: torch.Tensor,
    available: torch.Tensor,
    age_notional: torch.Tensor,
    *,
    cash_index: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    intent = ensemble.intent
    assets = current.numel()
    entry_scores = intent.entry_scores
    if entry_scores is None or tuple(entry_scores.shape) != (1, assets):
        raise M03RProjectionError(
            f"ensemble entry_scores must have shape {(1, assets)}"
        )
    if entry_scores.device != current.device or not bool(
        torch.isfinite(entry_scores).all()
    ):
        raise M03RProjectionError("ensemble entry_scores is invalid")
    hazard_residual_tensor = intent.hazard_residual
    if (
        hazard_residual_tensor is None
        or tuple(hazard_residual_tensor.shape) != (1, assets)
    ):
        raise M03RProjectionError(
            f"ensemble hazard_residual must have shape {(1, assets)}"
        )
    if hazard_residual_tensor.device != current.device or not bool(
        torch.isfinite(hazard_residual_tensor).all()
    ):
        raise M03RProjectionError("ensemble hazard_residual is invalid")
    active_risk_scale = intent.active_risk_scale
    if active_risk_scale is None or tuple(active_risk_scale.shape) != (1,):
        raise M03RProjectionError("ensemble active_risk_scale must have shape (1,)")
    if active_risk_scale.device != current.device or not bool(
        torch.isfinite(active_risk_scale).all()
    ):
        raise M03RProjectionError("ensemble active_risk_scale is invalid")
    if (
        tuple(age_notional.shape) != (assets, 61)
        or not age_notional.is_floating_point()
    ):
        raise M03RProjectionError("age_notional must be floating point [asset, 61]")
    if (
        age_notional.requires_grad
        or not bool(torch.isfinite(age_notional).all())
        or bool((age_notional < 0).any())
    ):
        raise M03RProjectionError(
            "age_notional must be detached, finite, and nonnegative"
        )
    ledger = age_notional.detach().to(device=current.device, dtype=torch.float64)
    risky = available.clone()
    risky[cash_index] = False
    if float(ledger[cash_index].abs().sum()) > 1e-7 or not bool(
        torch.allclose(ledger.sum(dim=-1)[risky], current[risky], atol=1e-7, rtol=1e-7)
    ):
        raise M03RProjectionError(
            "age ledger does not conserve the verified current book"
        )

    entry = entry_scores[0].detach().to(dtype=torch.float64)
    confidence = intent.signal_confidence
    derisk = intent.benchmark_derisk_request
    if confidence is None or tuple(confidence.shape) != (1,):
        raise M03RProjectionError("ensemble signal_confidence must have shape (1,)")
    if derisk is None or tuple(derisk.shape) != (1,):
        raise M03RProjectionError(
            "ensemble benchmark_derisk_request must have shape (1,)"
        )
    # Confidence acts only after a hazard-only anchor and the full replacement
    # proposal have independently passed hard feasibility. Entry scores
    # therefore have no exact-zero branch, and learned exits are never scaled
    # by signal confidence.
    score = entry
    count = risky.sum().clamp_min(1).to(dtype=torch.float64)
    score = score - torch.where(risky, score, torch.zeros_like(score)).sum() / count
    score = score.clamp(-2.0, 2.0)
    unnormalized = torch.where(
        risky,
        benchmark.clamp_min(0.0) * torch.exp(score),
        torch.zeros_like(score),
    )
    mass = unnormalized.sum()
    if float(mass) <= 0:
        raise M03RProjectionError("benchmark-relative entry direction has zero mass")
    direction = unnormalized / mass

    ages = torch.arange(61, device=current.device, dtype=torch.float64)
    hazard_residual = hazard_residual_tensor[0].detach().to(dtype=torch.float64)
    if intent.exact_hold_probability is not None:
        raise M03RProjectionError(
            "v5 execution forbids the legacy exact_hold_probability field"
        )
    exact_hold = intent.exact_hold_decision_st
    exact = (
        None
        if exact_hold is None
        else exact_hold[0].detach().to(dtype=torch.float64).unsqueeze(-1)
    )
    hazard = hold30_release_hazard(
        ages,
        hazard_residual.unsqueeze(-1),
        exact_hold_probability=exact,
    )
    released = (ledger * hazard).sum(dim=-1)
    released = torch.where(
        risky, torch.minimum(released, current), torch.zeros_like(released)
    )
    hazard_release_delta = -released
    hazard_release_delta[cash_index] = released.sum()
    raw_hazard_anchor = current + hazard_release_delta
    derisk_fraction = float(derisk[0])
    if not 0.0 <= derisk_fraction <= 1.0:
        raise M03RProjectionError("benchmark_derisk_request must lie in [0,1]")
    benchmark_derisk_delta = derisk_fraction * (benchmark - raw_hazard_anchor)
    raw_hazard_anchor = raw_hazard_anchor + benchmark_derisk_delta

    # Replacement entry is a distinct proposal.  Explicit benchmark de-risking
    # shrinks replacement mass, while signal confidence is intentionally absent
    # here and later limits only the move from the feasible hazard anchor toward
    # the feasible replacement proposal.
    entry_reallocation_delta = (1.0 - derisk_fraction) * released.sum() * direction
    entry_reallocation_delta[cash_index] = -entry_reallocation_delta.sum()
    if (
        not bool(torch.isfinite(raw_hazard_anchor).all())
        or abs(float(raw_hazard_anchor.sum()) - 1.0) > 1e-7
        or abs(float(entry_reallocation_delta.sum())) > 1e-7
    ):
        raise M03RProjectionError(
            "age-aware ensemble request is not a finite unit book"
        )
    return (
        raw_hazard_anchor,
        hazard_release_delta,
        benchmark_derisk_delta,
        entry_reallocation_delta,
    )


def execute_m03r_post_seed_ensemble(
    members: tuple[M03REnsembleMember, ...],
    book: M03RAssetAlignedBook,
    qualified_risk_manifest: M03RQualifiedRiskManifest,
    seed_checkpoint_manifest: M03RSeedCheckpointEnsembleManifest,
    *,
    expected_risk_manifest_sha256: str,
    expected_seed_checkpoint_manifest_sha256: str,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
    cash_asset_id: str,
    maximum_one_way_turnover: float,
    numerics: M03RProjectionNumerics = M03R_DEFAULT_PROJECTION_NUMERICS,
) -> M03RExecutionResult:
    """Repair drift, project exit anchor and replacement proposal, then trade."""

    setting = validate_m03r_artifact_identity(
        protocol_generation=protocol_generation,
        design_id=design_id,
        setting_id=setting_id,
    )
    if not setting.residual_alpha_heads:
        raise M03RProjectionError("M03R constrained alpha execution needs alpha heads")
    if not setting.factor_sector_projection:
        raise M03RProjectionError(
            "this hard-projection path is forbidden for A10 no-projection identity"
        )
    if setting.annual_tracking_error_floor not in {None, 0.0}:
        raise M03RProjectionError(
            "A05 compulsory tracking-error floor requires a separately governed ablation path"
        )
    if not math.isclose(
        float(setting.annual_tracking_error_ceiling or 0.0),
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
    ):
        raise M03RProjectionError(
            "this path requires the exact M03R 6% annual tracking-error ceiling"
        )
    if setting.sharpe_mode == "separate-total-risk-overlay":
        raise M03RProjectionError(
            "A06 total-risk overlay requires its separately governed execution path"
        )
    if (
        isinstance(maximum_one_way_turnover, bool)
        or not isinstance(maximum_one_way_turnover, (int, float))
        or not math.isfinite(float(maximum_one_way_turnover))
        or not 0 <= float(maximum_one_way_turnover) <= 1
    ):
        raise M03RProjectionError("maximum_one_way_turnover must be in [0,1]")
    validate_m03r_qualified_risk_manifest(
        qualified_risk_manifest,
        expected_manifest_sha256=expected_risk_manifest_sha256,
    )
    risk_manifest = qualified_risk_manifest.manifest
    assets = len(risk_manifest.asset_ids)
    if book.asset_ids != risk_manifest.asset_ids:
        raise M03RProjectionError(
            "asset-aligned book axis does not match the risk-manifest axis"
        )
    expected_asset_order_sha256 = compute_m03r_asset_order_sha256(
        risk_manifest.asset_ids
    )
    if book.asset_order_sha256 != expected_asset_order_sha256:
        raise M03RProjectionError("asset-aligned book order hash mismatch")
    if book.decision_trading_session != risk_manifest.as_of_trading_session:
        raise M03RProjectionError(
            "book decision session and risk-manifest as-of session must match"
        )
    if cash_asset_id not in risk_manifest.asset_ids:
        raise M03RProjectionError("cash asset ID is absent from the risk manifest")
    cash_index = risk_manifest.asset_ids.index(cash_asset_id)
    if (
        tuple(book.decision_available.shape) != (assets,)
        or book.decision_available.dtype != torch.bool
    ):
        raise M03RProjectionError("decision_available must be boolean [asset]")
    if not bool(book.decision_available[cash_index]):
        raise M03RProjectionError("CASH must be decision-available")
    device = book.current_weights.device
    current = _as_vector("current_weights", book.current_weights, assets, device)
    benchmark = _as_vector("benchmark_weights", book.benchmark_weights, assets, device)
    available = book.decision_available.detach().to(device=device)

    # Lexical PIT asset order is the deterministic solver tie-break.  Returning
    # to caller order afterwards makes the whole operation permutation equivariant.
    order = torch.tensor(
        qualified_risk_manifest.sorted_asset_indices,
        device=device,
        dtype=torch.long,
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(assets, device=device)
    current_s = current[order]
    benchmark_s = benchmark[order]
    available_s = available[order]
    loadings = risk_manifest.exposure_loadings.detach().to(
        device=device, dtype=torch.float64
    )[:, order]
    lower = risk_manifest.exposure_lower_bounds.detach().to(
        device=device, dtype=torch.float64
    )
    upper = risk_manifest.exposure_upper_bounds.detach().to(
        device=device, dtype=torch.float64
    )
    covariance = risk_manifest.daily_return_covariance.detach().to(
        device=device, dtype=torch.float64
    )[order][:, order]
    cash_sorted = int(inverse[cash_index])
    upper_weights = torch.where(
        available_s,
        torch.full_like(current_s, M03R_MAXIMUM_RISKY_ASSET_WEIGHT),
        torch.zeros_like(current_s),
    )
    upper_weights[cash_sorted] = 1.0
    output_tolerance = max(numerics.tolerance * 10.0, 1e-9)
    input_tolerance = max(output_tolerance, 1e-7)
    _validate_pretrade_book_sanity(current_s, tolerance=input_tolerance)
    _verify_feasible_book(
        "benchmark",
        benchmark_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        input_tolerance,
    )

    # Availability-forced exits are separately accounted before cap/factor/TE
    # repair. Cash receives the unavailable notional and has no age ledger.
    availability_repaired = current.clone()
    unavailable_risky = ~available
    unavailable_risky[cash_index] = False
    unavailable_notional = availability_repaired[unavailable_risky].sum()
    availability_repaired[unavailable_risky] = 0.0
    availability_repaired[cash_index] += unavailable_notional
    unavailable_delta = availability_repaired - current
    unavailable_sell = (current - availability_repaired).clamp_min(0.0)
    unavailable_sell[cash_index] = 0.0
    unavailable_buy = (availability_repaired - current).clamp_min(0.0)
    unavailable_sale_age = _sale_age_notional(
        book.age_notional,
        current,
        unavailable_sell,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    availability_age = _reconcile_age_ledger_after_transition(
        book.age_notional,
        current,
        availability_repaired,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    availability_repaired_s = availability_repaired[order]

    pre_repair_violation = float(
        _linear_violation(
            availability_repaired_s,
            benchmark_s,
            upper_weights,
            loadings,
            lower,
            upper,
        )
    )
    pre_repair_te = float(
        _annual_tracking_error(availability_repaired_s - benchmark_s, covariance)
    )
    repair_required = (
        pre_repair_violation > input_tolerance
        or pre_repair_te > M03R_ANNUAL_TRACKING_ERROR_CEILING + input_tolerance
    )
    if repair_required:
        repaired_linear_s, repair_iterations, _repair_linear_violation = (
            _project_linear_dykstra(
                availability_repaired_s,
                benchmark_s,
                upper_weights,
                loadings,
                lower,
                upper,
                numerics,
            )
        )
        repaired_linear_te = float(
            _annual_tracking_error(repaired_linear_s - benchmark_s, covariance)
        )
        repair_te_scale = min(
            1.0,
            M03R_ANNUAL_TRACKING_ERROR_CEILING / max(repaired_linear_te, 1e-30),
        )
        repaired_s = benchmark_s + repair_te_scale * (repaired_linear_s - benchmark_s)
    else:
        repair_iterations = 0
        repair_te_scale = 1.0
        repaired_s = availability_repaired_s.clone()
    _verify_feasible_book(
        "risk-forced repaired current book",
        repaired_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        output_tolerance,
    )
    repair_violation = float(
        _linear_violation(
            repaired_s,
            benchmark_s,
            upper_weights,
            loadings,
            lower,
            upper,
        )
    )
    repaired = repaired_s[inverse]
    repaired_age = _reconcile_age_ledger_after_transition(
        availability_age,
        availability_repaired,
        repaired,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    risk_repair_sell = (availability_repaired - repaired).clamp_min(0.0)
    risk_repair_sell[cash_index] = 0.0
    risk_repair_sale_age = _sale_age_notional(
        availability_age,
        availability_repaired,
        risk_repair_sell,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )

    ensemble = aggregate_m03r_alpha_intents(
        members,
        available.unsqueeze(0),
        book.asset_ids,
        seed_checkpoint_manifest,
        expected_seed_checkpoint_manifest_sha256=(
            expected_seed_checkpoint_manifest_sha256
        ),
        protocol_generation=protocol_generation,
        design_id=design_id,
        setting_id=setting_id,
        cash_index=cash_index,
    )
    (
        raw_hazard_anchor,
        hazard_release_delta,
        benchmark_derisk_delta,
        entry_reallocation_delta,
    ) = _requested_weights_from_ensemble(
        ensemble,
        repaired,
        benchmark,
        available,
        repaired_age,
        cash_index=cash_index,
    )

    # Stage 1 is a separately bound hard-feasibility projection for the
    # confidence-independent learned-exit / explicit-de-risk anchor.  Without
    # this stage, zero confidence would either suppress learned exits or allow
    # them to violate factor, cap, or tracking-error limits.
    raw_hazard_anchor_s = raw_hazard_anchor[order]
    (
        hazard_anchor_linear_s,
        hazard_anchor_iterations,
        hazard_anchor_linear_violation,
    ) = _project_linear_dykstra(
        raw_hazard_anchor_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        numerics,
    )
    hazard_anchor_linear_te = float(
        _annual_tracking_error(hazard_anchor_linear_s - benchmark_s, covariance)
    )
    hazard_anchor_te_scale = min(
        1.0,
        M03R_ANNUAL_TRACKING_ERROR_CEILING
        / max(hazard_anchor_linear_te, 1e-30),
    )
    hazard_anchor_s = benchmark_s + hazard_anchor_te_scale * (
        hazard_anchor_linear_s - benchmark_s
    )
    _verify_feasible_book(
        "projected learned-hazard anchor",
        hazard_anchor_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        output_tolerance,
    )
    hazard_anchor_te = float(
        _annual_tracking_error(hazard_anchor_s - benchmark_s, covariance)
    )
    hazard_anchor = hazard_anchor_s[inverse]

    # Stage 2 projects the full replacement proposal independently.  The
    # confidence budget later interpolates between these two feasible books,
    # so it can suppress replacement entry without suppressing learned exits.
    requested = hazard_anchor + entry_reallocation_delta
    requested_s = requested[order]
    linear, iterations, linear_violation = _project_linear_dykstra(
        requested_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        numerics,
    )
    linear_te = float(_annual_tracking_error(linear - benchmark_s, covariance))
    proposal_te_scale = min(
        1.0,
        M03R_ANNUAL_TRACKING_ERROR_CEILING / max(linear_te, 1e-30),
    )
    proposal_s = benchmark_s + proposal_te_scale * (linear - benchmark_s)
    _verify_feasible_book(
        "projected replacement proposal",
        proposal_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        output_tolerance,
    )
    assert ensemble.intent.active_risk_scale is not None
    preferred_te_cap = float(ensemble.intent.active_risk_scale[0])
    # Confidence acts exactly once, on replacement/new-risk movement from the
    # already feasible hazard anchor.  Learned exits and an explicit de-risk
    # request are upstream and therefore remain executable at zero confidence.
    incremental_requested_te = float(
        _annual_tracking_error(proposal_s - hazard_anchor_s, covariance)
    )
    confidence_covariance_scale = min(
        1.0,
        preferred_te_cap / max(incremental_requested_te, 1e-30),
    )
    requested_incremental_one_way = float(
        0.5 * torch.abs(proposal_s - hazard_anchor_s).sum()
    )
    assert ensemble.intent.signal_confidence is not None
    confidence_value = float(ensemble.intent.signal_confidence[0])
    confidence_incremental_one_way_cap = (
        confidence_value * M03R_MAXIMUM_CONFIDENCE_INCREMENTAL_ONE_WAY_TURNOVER
    )
    confidence_one_way_scale = min(
        1.0,
        confidence_incremental_one_way_cap
        / max(requested_incremental_one_way, 1e-30),
    )
    # Covariance is allowed to be PSD and therefore may contain a large
    # nullspace.  The L1 cap makes confidence control nondegenerate even when a
    # full cross-sectional rotation has zero covariance norm.
    confidence_incremental_scale = min(
        confidence_covariance_scale,
        confidence_one_way_scale,
    )
    projected_s = hazard_anchor_s + confidence_incremental_scale * (
        proposal_s - hazard_anchor_s
    )
    confidence_limited_incremental_te = float(
        _annual_tracking_error(projected_s - hazard_anchor_s, covariance)
    )
    confidence_limited_incremental_one_way = float(
        0.5 * torch.abs(projected_s - hazard_anchor_s).sum()
    )
    if confidence_limited_incremental_te > preferred_te_cap + output_tolerance:
        raise M03RProjectionError(
            "confidence-dependent incremental active-risk cap was not satisfied"
        )
    if (
        confidence_limited_incremental_one_way
        > confidence_incremental_one_way_cap + output_tolerance
    ):
        raise M03RProjectionError(
            "confidence-dependent incremental one-way-turnover cap was not satisfied"
        )
    effective_te_cap = min(
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        hazard_anchor_te + preferred_te_cap,
    )
    _verify_feasible_book(
        "projected target",
        projected_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        output_tolerance,
    )
    projected_te = float(_annual_tracking_error(projected_s - benchmark_s, covariance))
    if projected_te > effective_te_cap + output_tolerance:
        raise M03RProjectionError(
            "confidence-limited projected tracking-error bound was not satisfied"
        )
    requested_turnover = 0.5 * torch.abs(projected_s - repaired_s).sum()
    turnover_scale = min(
        1.0,
        float(maximum_one_way_turnover) / max(float(requested_turnover), 1e-30),
    )
    final_s = repaired_s + turnover_scale * (projected_s - repaired_s)
    _verify_feasible_book(
        "turnover-limited execution",
        final_s,
        benchmark_s,
        upper_weights,
        loadings,
        lower,
        upper,
        covariance,
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        output_tolerance,
    )
    final_violation = float(
        _linear_violation(final_s, benchmark_s, upper_weights, loadings, lower, upper)
    )
    repair_delta = repaired_s - availability_repaired_s
    repair_turnover = 0.5 * torch.abs(repair_delta).sum()
    repair_sells = (availability_repaired_s - repaired_s).clamp_min(0.0).sum()
    repair_buys = (repaired_s - availability_repaired_s).clamp_min(0.0).sum()
    projected = projected_s[inverse]
    final = final_s[inverse]
    hazard_anchor_linear = hazard_anchor_linear_s[inverse]
    linear_caller_order = linear[inverse]
    proposal = proposal_s[inverse]
    hazard_anchor_factor_projection_delta = (
        hazard_anchor_linear - raw_hazard_anchor
    )
    hazard_anchor_tracking_error_projection_delta = (
        hazard_anchor - hazard_anchor_linear
    )
    factor_projection_delta = linear_caller_order - requested
    tracking_error_projection_delta = proposal - linear_caller_order
    confidence_budget_delta = projected - proposal
    turnover_truncation_delta = final - projected
    final_age = _reconcile_age_ledger_after_transition(
        repaired_age,
        repaired,
        final,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    # Economic trade-cause attribution uses a frozen priority convention after
    # turnover scaling: learned hazard, explicit benchmark de-risking, then
    # feasibility projection. It never treats the truncation residual as a
    # separate trade. All vectors are nonnegative notionals in caller order.
    post_repair_sell = (repaired - final).clamp_min(0.0)
    post_repair_buy = (final - repaired).clamp_min(0.0)
    hazard_sell_request = (-hazard_release_delta).clamp_min(0.0)
    hazard_sell_request[cash_index] = 0.0
    executed_hazard_sell = torch.minimum(
        post_repair_sell,
        turnover_scale * hazard_sell_request,
    )
    remaining_sell = (post_repair_sell - executed_hazard_sell).clamp_min(0.0)
    derisk_sell_request = (-benchmark_derisk_delta).clamp_min(0.0)
    derisk_sell_request[cash_index] = 0.0
    executed_derisk_sell = torch.minimum(
        remaining_sell,
        turnover_scale * derisk_sell_request,
    )
    executed_projection_sell = (
        remaining_sell - executed_derisk_sell
    ).clamp_min(0.0)
    executed_projection_sell[cash_index] = 0.0
    entry_buy_request = entry_reallocation_delta.clamp_min(0.0)
    entry_buy_request[cash_index] = 0.0
    executed_entry_buy = torch.minimum(
        post_repair_buy,
        turnover_scale * confidence_incremental_scale * entry_buy_request,
    )
    executed_projection_buy = (
        post_repair_buy - executed_entry_buy
    ).clamp_min(0.0)
    executed_hazard_sale_age = _sale_age_notional(
        repaired_age,
        repaired,
        executed_hazard_sell,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    executed_derisk_sale_age = _sale_age_notional(
        repaired_age,
        repaired,
        executed_derisk_sell,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    executed_projection_sale_age = _sale_age_notional(
        repaired_age,
        repaired,
        executed_projection_sell,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )
    executed_total_sale_age = (
        unavailable_sale_age
        + risk_repair_sale_age
        + executed_hazard_sale_age
        + executed_derisk_sale_age
        + executed_projection_sale_age
    )
    cause_sell = (
        unavailable_sell
        + risk_repair_sell
        + executed_hazard_sell
        + executed_derisk_sell
        + executed_projection_sell
    )
    if not bool(
        torch.allclose(
            executed_total_sale_age.sum(dim=-1),
            cause_sell,
            atol=max(input_tolerance, 1e-8),
            rtol=max(input_tolerance, 1e-8),
        )
    ):
        raise M03RProjectionError(
            "cause-specific sale-age tensors do not partition executed sells"
        )
    attributed = (
        unavailable_delta
        + (repaired - availability_repaired)
        + hazard_release_delta
        + benchmark_derisk_delta
        + hazard_anchor_factor_projection_delta
        + hazard_anchor_tracking_error_projection_delta
        + entry_reallocation_delta
        + factor_projection_delta
        + tracking_error_projection_delta
        + confidence_budget_delta
        + turnover_truncation_delta
    )
    if not bool(torch.allclose(current + attributed, final, atol=2e-8, rtol=2e-8)):
        raise M03RProjectionError("cause-specific execution deltas do not telescope")
    diagnostics = M03RProjectionDiagnostics(
        projection_application_count=(
            M03R_DESIGN.ensemble_execution.post_ensemble_projection_application_count
        ),
        hazard_anchor_solver_iterations=hazard_anchor_iterations,
        hazard_anchor_solver_converged=True,
        hazard_anchor_maximum_linear_violation=hazard_anchor_linear_violation,
        hazard_anchor_annual_tracking_error=hazard_anchor_te,
        hazard_anchor_tracking_error_scale=hazard_anchor_te_scale,
        hazard_anchor_projection_l2_distance=float(
            torch.linalg.vector_norm(hazard_anchor_s - raw_hazard_anchor_s)
        ),
        risk_forced_repair_application_count=int(repair_required),
        risk_forced_repair_solver_iterations=repair_iterations,
        risk_forced_repair_solver_converged=True,
        pre_repair_maximum_linear_violation=pre_repair_violation,
        risk_forced_repair_maximum_violation=repair_violation,
        solver_iterations=iterations,
        solver_converged=True,
        maximum_linear_violation=linear_violation,
        maximum_final_violation=final_violation,
        requested_active_exposures=_exposures(
            requested_s, benchmark_s, loadings
        ).detach(),
        pre_repair_active_exposures=_exposures(
            availability_repaired_s, benchmark_s, loadings
        ).detach(),
        repaired_active_exposures=_exposures(
            repaired_s, benchmark_s, loadings
        ).detach(),
        linear_projected_active_exposures=_exposures(
            linear, benchmark_s, loadings
        ).detach(),
        projected_active_exposures=_exposures(
            projected_s, benchmark_s, loadings
        ).detach(),
        final_active_exposures=_exposures(final_s, benchmark_s, loadings).detach(),
        current_annual_tracking_error=pre_repair_te,
        repaired_current_annual_tracking_error=float(
            _annual_tracking_error(repaired_s - benchmark_s, covariance)
        ),
        requested_annual_tracking_error=float(
            _annual_tracking_error(requested_s - benchmark_s, covariance)
        ),
        linear_projected_annual_tracking_error=float(linear_te),
        projected_annual_tracking_error=projected_te,
        final_annual_tracking_error=float(
            _annual_tracking_error(final_s - benchmark_s, covariance)
        ),
        preferred_annual_tracking_error_cap=preferred_te_cap,
        effective_annual_tracking_error_cap=effective_te_cap,
        tracking_error_scale=proposal_te_scale,
        confidence_tracking_error_scale=confidence_incremental_scale,
        risk_forced_repair_tracking_error_scale=repair_te_scale,
        risk_forced_repair_l2_distance=float(torch.linalg.vector_norm(repair_delta)),
        risk_forced_repair_one_way_turnover=float(repair_turnover),
        risk_forced_repair_sell_notional=float(repair_sells),
        risk_forced_repair_buy_notional=float(repair_buys),
        unavailable_forced_one_way_turnover=float(
            0.5 * torch.abs(unavailable_delta).sum()
        ),
        unavailable_forced_sell_notional=float(unavailable_sell.sum()),
        unavailable_forced_buy_notional=float(unavailable_buy.sum()),
        requested_incremental_active_risk=incremental_requested_te,
        confidence_limited_incremental_active_risk=(
            confidence_limited_incremental_te
        ),
        requested_incremental_one_way_turnover=requested_incremental_one_way,
        confidence_limited_incremental_one_way_turnover=(
            confidence_limited_incremental_one_way
        ),
        confidence_incremental_one_way_turnover_cap=(
            confidence_incremental_one_way_cap
        ),
        confidence_covariance_risk_scale=confidence_covariance_scale,
        confidence_one_way_turnover_scale=confidence_one_way_scale,
        confidence_incremental_risk_scale=confidence_incremental_scale,
        linear_projection_l2_distance=float(
            torch.linalg.vector_norm(linear - requested_s)
        ),
        projected_l2_distance=float(
            torch.linalg.vector_norm(projected_s - requested_s)
        ),
        final_l2_distance=float(torch.linalg.vector_norm(final_s - requested_s)),
        requested_one_way_turnover=float(requested_turnover),
        executed_one_way_turnover=float(0.5 * torch.abs(final_s - repaired_s).sum()),
        turnover_interpolation_scale=turnover_scale,
    )
    return M03RExecutionResult(
        ensemble=ensemble,
        availability_repaired_weights=availability_repaired,
        repaired_current_weights=repaired,
        repaired_age_notional=repaired_age,
        raw_hazard_anchor_weights=raw_hazard_anchor,
        projected_hazard_anchor_weights=hazard_anchor,
        requested_weights=requested,
        projected_weights=projected,
        executed_weights=final,
        final_age_notional=final_age,
        unavailable_delta=unavailable_delta,
        risk_repair_delta=repaired - availability_repaired,
        hazard_release_delta=hazard_release_delta,
        benchmark_derisk_delta=benchmark_derisk_delta,
        hazard_anchor_factor_projection_delta=(
            hazard_anchor_factor_projection_delta
        ),
        hazard_anchor_tracking_error_projection_delta=(
            hazard_anchor_tracking_error_projection_delta
        ),
        entry_reallocation_delta=entry_reallocation_delta,
        factor_projection_delta=factor_projection_delta,
        tracking_error_projection_delta=tracking_error_projection_delta,
        confidence_budget_delta=confidence_budget_delta,
        turnover_truncation_delta=turnover_truncation_delta,
        executed_unavailable_sell_notional=unavailable_sell,
        executed_unavailable_sale_age_notional=unavailable_sale_age,
        executed_risk_repair_sell_notional=risk_repair_sell,
        executed_risk_repair_sale_age_notional=risk_repair_sale_age,
        executed_learned_hazard_sell_notional=executed_hazard_sell,
        executed_learned_hazard_sale_age_notional=executed_hazard_sale_age,
        executed_entry_buy_notional=executed_entry_buy,
        executed_benchmark_derisk_sell_notional=executed_derisk_sell,
        executed_benchmark_derisk_sale_age_notional=executed_derisk_sale_age,
        executed_projection_sell_notional=executed_projection_sell,
        executed_projection_sale_age_notional=executed_projection_sale_age,
        executed_projection_buy_notional=executed_projection_buy,
        executed_total_sale_age_notional=executed_total_sale_age,
        diagnostics=diagnostics,
    )


__all__ = [
    "M03R_ACTIVE_BETA_ABSOLUTE_MAXIMUM",
    "M03R_ACTIVE_BETA_EXPOSURE_NAME",
    "M03R_ANNUAL_TRACKING_ERROR_CEILING",
    "M03R_DEFAULT_PROJECTION_NUMERICS",
    "M03R_MAXIMUM_CONFIDENCE_INCREMENTAL_ONE_WAY_TURNOVER",
    "M03R_MAXIMUM_RISKY_ASSET_WEIGHT",
    "M03R_OBJECTIVE_RISK_CONTRACT_SCHEMA",
    "M03R_REQUIRED_EXPOSURE_FAMILIES",
    "M03R_RISK_MANIFEST_SCHEMA",
    "M03RAssetAlignedBook",
    "M03RExecutionResult",
    "M03RObjectiveRiskContract",
    "M03RProjectionDiagnostics",
    "M03RProjectionError",
    "M03RProjectionNumerics",
    "M03RQualifiedRiskManifest",
    "M03RRiskManifest",
    "bind_m03r_objective_risk_contract",
    "bind_m03r_risk_manifest",
    "compute_m03r_objective_risk_contract_sha256",
    "compute_m03r_risk_manifest_sha256",
    "execute_m03r_post_seed_ensemble",
    "m03r_objective_risk_contract_payload",
    "m03r_risk_manifest_payload",
    "qualify_m03r_risk_manifest",
    "validate_m03r_objective_risk_contract",
    "validate_m03r_qualified_risk_manifest",
]
