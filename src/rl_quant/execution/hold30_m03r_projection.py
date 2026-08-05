"""Fail-closed M03R post-ensemble risk projection and execution.

This is intentionally separate from the frozen V2/V3 portfolio builders.  It
aggregates the five M03R seed intents once, constructs one age-aware requested
book, performs one deterministic Euclidean projection onto the manifest-bound
linear feasible set, scales the active book toward the feasible benchmark when
needed to satisfy the tracking-error ellipsoid, and finally turnover-limits the
move from a verified feasible book. Ordinary pretrade drift is repaired first
as separately accounted risk-forced turnover, including exact age-ledger
reconciliation. Convex interpolation preserves every hard constraint.

The ensemble's ``active_risk_scale`` is a confidence-dependent upper target,
not a compulsory tracking-error floor. The post-linear requested target is
scaled to at most ``min(active_risk_scale, 6%)``; zero confidence therefore
maps the projected target exactly to the benchmark. The fixed 6% limit remains
an independent safety ceiling for repaired and executed books.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace

import torch

from rl_quant.models.daily_policy import hold30_release_hazard
from rl_quant.models.hold30_m03r_ensemble import (
    M03REnsembleIntent,
    M03REnsembleMember,
    aggregate_m03r_alpha_intents,
)
from rl_quant.protocol.hold30_alpha_m03r import (
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
M03R_ACTIVE_RISK_REFERENCE = (
    M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M03RProjectionError(ValueError):
    """An M03R projection input is unbound, infeasible, or unconverged."""


@dataclass(frozen=True, slots=True)
class M03RRiskManifest:
    """Point-in-time factor slabs and return covariance in one bound payload."""

    schema: str
    as_of_trading_session: str
    asset_ids: tuple[str, ...]
    exposure_names: tuple[str, ...]
    exposure_families: tuple[str, ...]
    exposure_loadings: torch.Tensor
    exposure_lower_bounds: torch.Tensor
    exposure_upper_bounds: torch.Tensor
    daily_return_covariance: torch.Tensor
    annual_tracking_error_ceiling: float
    maximum_risky_asset_weight: float
    manifest_sha256: str


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
    repaired_current_weights: torch.Tensor
    repaired_age_notional: torch.Tensor
    requested_weights: torch.Tensor
    projected_weights: torch.Tensor
    executed_weights: torch.Tensor
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
) -> None:
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
    ):
        raise M03RProjectionError("risk manifest exposure identities are invalid")
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
    eigenvalues = torch.linalg.eigvalsh(covariance)
    if float(eigenvalues.min()) < -1e-12 or float(eigenvalues.max()) <= 0:
        raise M03RProjectionError(
            "daily return covariance must be nonzero positive semidefinite"
        )


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
    violation = float(
        _linear_violation(current, benchmark, upper_weights, loadings, lower, upper)
    )
    raise M03RProjectionError(
        "M03R Euclidean projection did not converge: "
        f"iterations={numerics.maximum_iterations}, violation={violation:.3e}"
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


def _reconcile_age_ledger_after_risk_repair(
    age_notional: torch.Tensor,
    pre_repair: torch.Tensor,
    repaired: torch.Tensor,
    *,
    cash_index: int,
    tolerance: float,
) -> torch.Tensor:
    """Apply forced sells pro rata and forced buys to the age-zero cohort."""

    assets = pre_repair.numel()
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
    ledger = age_notional.detach().to(device=pre_repair.device, dtype=torch.float64)
    risky = torch.ones(assets, dtype=torch.bool, device=pre_repair.device)
    risky[cash_index] = False
    if float(ledger[cash_index].abs().sum()) > tolerance or not bool(
        torch.allclose(
            ledger.sum(dim=-1)[risky],
            pre_repair[risky],
            atol=tolerance,
            rtol=tolerance,
        )
    ):
        raise M03RProjectionError(
            "age ledger does not conserve the pre-repair current book"
        )

    repaired_nonnegative = repaired.clamp_min(0.0)
    retained_fraction = torch.where(
        pre_repair > tolerance,
        torch.minimum(
            torch.ones_like(pre_repair),
            repaired_nonnegative / pre_repair.clamp_min(tolerance),
        ),
        torch.zeros_like(pre_repair),
    )
    reconciled = ledger * retained_fraction.unsqueeze(-1)
    forced_buys = (repaired_nonnegative - pre_repair).clamp_min(0.0)
    forced_buys[cash_index] = 0.0
    reconciled[:, 0] = reconciled[:, 0] + forced_buys
    reconciled[cash_index] = 0.0
    if not bool(
        torch.allclose(
            reconciled.sum(dim=-1)[risky],
            repaired_nonnegative[risky],
            atol=max(tolerance, 1e-8),
            rtol=max(tolerance, 1e-8),
        )
    ):
        raise M03RProjectionError(
            "risk-forced age-ledger reconciliation did not conserve repaired holdings"
        )
    return reconciled


def _requested_weights_from_ensemble(
    ensemble: M03REnsembleIntent,
    current: torch.Tensor,
    benchmark: torch.Tensor,
    available: torch.Tensor,
    age_notional: torch.Tensor,
    *,
    cash_index: int,
) -> torch.Tensor:
    intent = ensemble.intent
    assets = current.numel()
    required = {
        "entry_scores": intent.entry_scores,
        "hazard_residual": intent.hazard_residual,
        "active_risk_scale": intent.active_risk_scale,
    }
    for name, value in required.items():
        expected = (1,) if name == "active_risk_scale" else (1, assets)
        if value is None or tuple(value.shape) != expected:
            raise M03RProjectionError(f"ensemble {name} must have shape {expected}")
        if value.device != current.device or not bool(torch.isfinite(value).all()):
            raise M03RProjectionError(f"ensemble {name} is invalid")
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

    entry = intent.entry_scores[0].detach().to(dtype=torch.float64)
    scale = float(intent.active_risk_scale[0]) / M03R_ACTIVE_RISK_REFERENCE
    score = entry * scale
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
    hazard_residual = intent.hazard_residual[0].detach().to(dtype=torch.float64)
    exact_hold = intent.exact_hold_probability
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
    retained = current - released
    requested = retained + released.sum() * direction
    requested[cash_index] = current[cash_index]
    if (
        not bool(torch.isfinite(requested).all())
        or abs(float(requested.sum()) - 1.0) > 1e-7
    ):
        raise M03RProjectionError(
            "age-aware ensemble request is not a finite unit book"
        )
    return requested


def execute_m03r_post_seed_ensemble(
    members: tuple[M03REnsembleMember, ...],
    current_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    decision_available: torch.Tensor,
    age_notional: torch.Tensor,
    risk_manifest: M03RRiskManifest,
    *,
    expected_risk_manifest_sha256: str,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
    cash_asset_id: str,
    maximum_one_way_turnover: float,
    numerics: M03RProjectionNumerics = M03R_DEFAULT_PROJECTION_NUMERICS,
) -> M03RExecutionResult:
    """Repair drift, aggregate once, project requested intent once, and trade."""

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
    _validate_risk_manifest(
        risk_manifest,
        expected_manifest_sha256=expected_risk_manifest_sha256,
    )
    assets = len(risk_manifest.asset_ids)
    if cash_asset_id not in risk_manifest.asset_ids:
        raise M03RProjectionError("cash asset ID is absent from the risk manifest")
    cash_index = risk_manifest.asset_ids.index(cash_asset_id)
    if (
        tuple(decision_available.shape) != (assets,)
        or decision_available.dtype != torch.bool
    ):
        raise M03RProjectionError("decision_available must be boolean [asset]")
    if not bool(decision_available[cash_index]):
        raise M03RProjectionError("CASH must be decision-available")
    device = current_weights.device
    current = _as_vector("current_weights", current_weights, assets, device)
    benchmark = _as_vector("benchmark_weights", benchmark_weights, assets, device)
    available = decision_available.detach().to(device=device)

    # Lexical PIT asset order is the deterministic solver tie-break.  Returning
    # to caller order afterwards makes the whole operation permutation equivariant.
    order = torch.tensor(
        sorted(range(assets), key=lambda index: risk_manifest.asset_ids[index]),
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

    pre_repair_violation = float(
        _linear_violation(
            current_s,
            benchmark_s,
            upper_weights,
            loadings,
            lower,
            upper,
        )
    )
    pre_repair_te = float(_annual_tracking_error(current_s - benchmark_s, covariance))
    repair_required = (
        pre_repair_violation > input_tolerance
        or pre_repair_te > M03R_ANNUAL_TRACKING_ERROR_CEILING + input_tolerance
    )
    if repair_required:
        repaired_linear_s, repair_iterations, _repair_linear_violation = (
            _project_linear_dykstra(
                current_s,
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
        repaired_s = current_s.clone()
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
    repaired_age = _reconcile_age_ledger_after_risk_repair(
        age_notional,
        current,
        repaired,
        cash_index=cash_index,
        tolerance=input_tolerance,
    )

    ensemble = aggregate_m03r_alpha_intents(
        members,
        available.unsqueeze(0),
        protocol_generation=protocol_generation,
        design_id=design_id,
        setting_id=setting_id,
        cash_index=cash_index,
    )
    requested = _requested_weights_from_ensemble(
        ensemble,
        repaired,
        benchmark,
        available,
        repaired_age,
        cash_index=cash_index,
    )
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
    linear_te = _annual_tracking_error(linear - benchmark_s, covariance)
    assert ensemble.intent.active_risk_scale is not None
    preferred_te_cap = float(ensemble.intent.active_risk_scale[0])
    effective_te_cap = min(
        M03R_ANNUAL_TRACKING_ERROR_CEILING,
        preferred_te_cap,
    )
    te_scale = (
        0.0
        if effective_te_cap == 0.0
        else min(1.0, effective_te_cap / max(float(linear_te), 1e-30))
    )
    projected_s = benchmark_s + te_scale * (linear - benchmark_s)
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
            "confidence-dependent projected tracking-error cap was not satisfied"
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
    repair_delta = repaired_s - current_s
    repair_turnover = 0.5 * torch.abs(repair_delta).sum()
    repair_sells = (current_s - repaired_s).clamp_min(0.0).sum()
    repair_buys = (repaired_s - current_s).clamp_min(0.0).sum()
    projected = projected_s[inverse]
    final = final_s[inverse]
    diagnostics = M03RProjectionDiagnostics(
        projection_application_count=(
            M03R_DESIGN.ensemble_execution.post_ensemble_projection_application_count
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
            current_s, benchmark_s, loadings
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
        tracking_error_scale=te_scale,
        confidence_tracking_error_scale=te_scale,
        risk_forced_repair_tracking_error_scale=repair_te_scale,
        risk_forced_repair_l2_distance=float(torch.linalg.vector_norm(repair_delta)),
        risk_forced_repair_one_way_turnover=float(repair_turnover),
        risk_forced_repair_sell_notional=float(repair_sells),
        risk_forced_repair_buy_notional=float(repair_buys),
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
        repaired_current_weights=repaired,
        repaired_age_notional=repaired_age,
        requested_weights=requested,
        projected_weights=projected,
        executed_weights=final,
        diagnostics=diagnostics,
    )


__all__ = [
    "M03R_ACTIVE_BETA_ABSOLUTE_MAXIMUM",
    "M03R_ACTIVE_BETA_EXPOSURE_NAME",
    "M03R_ANNUAL_TRACKING_ERROR_CEILING",
    "M03R_DEFAULT_PROJECTION_NUMERICS",
    "M03R_MAXIMUM_RISKY_ASSET_WEIGHT",
    "M03R_REQUIRED_EXPOSURE_FAMILIES",
    "M03R_RISK_MANIFEST_SCHEMA",
    "M03RExecutionResult",
    "M03RProjectionDiagnostics",
    "M03RProjectionError",
    "M03RProjectionNumerics",
    "M03RRiskManifest",
    "bind_m03r_risk_manifest",
    "compute_m03r_risk_manifest_sha256",
    "execute_m03r_post_seed_ensemble",
    "m03r_risk_manifest_payload",
]
