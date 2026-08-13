"""Frozen projector contract and immutable risk-source binding for M03R-v9.

The point-in-time risk materialization intentionally remains immutable and
records that no projector had yet been frozen.  This module closes exactly
that boundary: it freezes the nonzero exposure slabs and the causal
factor-plus-diagonal covariance recipe, then issues a new binding receipt
without rewriting the materialized tensor artifact or its original receipt.

This module does not construct a portfolio or authorize the economic panel.
It authorizes only the three-setting predictive worker after the exact
materialized exposure names, families, asset axis, and artifact identities
match the projector contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from rl_quant.envs.hold30 import reconcile_cash_simplex_roundoff
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
    M03R_V9_PROJECTOR_EXPOSURE_FAMILIES,
    M03R_V9_PROJECTOR_EXPOSURE_NAMES,
    M03R_V9_SECTOR_EXPOSURE_NAMES,
    M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
    M03RV9MaterializedRiskSource,
    M03RV9WrittenRiskSource,
)
from rl_quant.training.top2000_m03r_v9_risk_source import (
    M03RV9RiskSourceInventory,
    M03RV9RiskSourceReadiness,
    audit_m03r_v9_risk_source,
)

M03R_V9_PROJECTOR_MANIFEST_SCHEMA = "rl-quant.top2000-dev.m03r-v9-projector-manifest-v1"
M03R_V9_PROJECTOR_BINDING_SCHEMA = "rl-quant.top2000-dev.m03r-v9-projector-binding-v1"
M03R_V9_PROJECTOR_MANIFEST_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-projector-manifest-file-v1"
)
M03R_V9_DEVICE_RISK_STATE_SCHEMA = "rl-quant.top2000-dev.m03r-v9-device-risk-state-v1"
M03R_V9_PROJECTION_RESULT_SCHEMA = "rl-quant.top2000-dev.m03r-v9-risk-projection-v1"
M03R_V9_SIGNAL_PROJECTION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-signal-null-projection-v1"
)

M03R_V9_EXPECTED_PREPROJECTOR_BLOCKERS = (
    "missing-projector-manifest",
    "target-projector-exposure-name-mismatch",
)
_DEVICE_RISK_STATE_ISSUER = object()


class M03RV9ProjectionError(ValueError):
    """The v9 projector contract or source binding drifted."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RV9ProjectionError("projector payload is not canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV9ProjectionError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_regular(path: Path, *, name: str) -> Path:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise M03RV9ProjectionError(f"{name} must be a regular non-symlink file")
    return path.resolve()


@dataclass(frozen=True, slots=True)
class M03RV9CovarianceEstimatorSpec:
    """Causal factor-plus-diagonal covariance recipe used by the sleeve."""

    lookback_sessions: int = 252
    minimum_history_sessions: int = 63
    return_definition: str = "close-to-close-log-return-v1"
    missing_return_handling: str = "pairwise-zero-after-origin-availability-mask-v1"
    cross_sectional_winsor_tail_fraction: float = 0.01
    factor_covariance_shrinkage_to_diagonal: float = 0.20
    specific_variance_shrinkage_to_cross_sectional_median: float = 0.20
    specific_variance_floor: float = 1.0e-8
    psd_eigenvalue_floor: float = 1.0e-10
    update_frequency_sessions: int = 21
    estimator: str = "weighted-ridge-factor-plus-diagonal-v1"

    def validate(self) -> None:
        if (
            self.lookback_sessions != 252
            or self.minimum_history_sessions != 63
            or self.return_definition != "close-to-close-log-return-v1"
            or self.missing_return_handling
            != "pairwise-zero-after-origin-availability-mask-v1"
            or self.cross_sectional_winsor_tail_fraction != 0.01
            or self.factor_covariance_shrinkage_to_diagonal != 0.20
            or self.specific_variance_shrinkage_to_cross_sectional_median != 0.20
            or self.specific_variance_floor != 1.0e-8
            or self.psd_eigenvalue_floor != 1.0e-10
            or self.update_frequency_sessions != 21
            or self.estimator != "weighted-ridge-factor-plus-diagonal-v1"
        ):
            raise M03RV9ProjectionError("covariance estimator contract drifted")


@dataclass(frozen=True, slots=True)
class M03RV9ProjectorManifest:
    """Static projector semantics shared by targets and simple sleeve."""

    exposure_names: tuple[str, ...]
    exposure_families: tuple[str, ...]
    exposure_lower_bounds: tuple[float, ...]
    exposure_upper_bounds: tuple[float, ...]
    active_beta_exposure_name: str
    annual_tracking_error_ceiling: float
    active_beta_absolute_bound: float
    maximum_stock_weight_fraction: float
    projection_mode: str
    covariance_estimator: M03RV9CovarianceEstimatorSpec
    manifest_sha256: str
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_PROJECTOR_MANIFEST_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "exposure_names": self.exposure_names,
            "exposure_families": self.exposure_families,
            "exposure_lower_bounds_float64_hex": tuple(
                float(value).hex() for value in self.exposure_lower_bounds
            ),
            "exposure_upper_bounds_float64_hex": tuple(
                float(value).hex() for value in self.exposure_upper_bounds
            ),
            "active_beta_exposure_name": self.active_beta_exposure_name,
            "annual_tracking_error_ceiling_float64_hex": float(
                self.annual_tracking_error_ceiling
            ).hex(),
            "active_beta_absolute_bound_float64_hex": float(
                self.active_beta_absolute_bound
            ).hex(),
            "maximum_stock_weight_fraction_float64_hex": float(
                self.maximum_stock_weight_fraction
            ).hex(),
            "projection_mode": self.projection_mode,
            "covariance_estimator": asdict(self.covariance_estimator),
            "research_only": True,
            "development_only": True,
            "economic_panel_authorized": False,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        self.covariance_estimator.validate()
        count = len(self.exposure_names)
        if (
            self.schema != M03R_V9_PROJECTOR_MANIFEST_SCHEMA
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.exposure_names != M03R_V9_PROJECTOR_EXPOSURE_NAMES
            or self.exposure_families != M03R_V9_PROJECTOR_EXPOSURE_FAMILIES
            or count == 0
            or len(set(self.exposure_names)) != count
            or len(self.exposure_families) != count
            or len(self.exposure_lower_bounds) != count
            or len(self.exposure_upper_bounds) != count
            or self.active_beta_exposure_name != M03R_V9_ACTIVE_BETA_EXPOSURE_NAME
            or self.annual_tracking_error_ceiling != 0.06
            or self.active_beta_absolute_bound != 0.10
            or self.maximum_stock_weight_fraction != 0.01
            or self.projection_mode != "benchmark-radial-factor-diagonal-beta-te-v1"
            or any(not math.isfinite(value) for value in self.exposure_lower_bounds)
            or any(not math.isfinite(value) for value in self.exposure_upper_bounds)
            or any(value >= 0.0 for value in self.exposure_lower_bounds)
            or any(value <= 0.0 for value in self.exposure_upper_bounds)
            or any(
                lower != -upper
                for lower, upper in zip(
                    self.exposure_lower_bounds,
                    self.exposure_upper_bounds,
                    strict=True,
                )
            )
            or _canonical_sha256(self.unsigned_payload()) != self.manifest_sha256
        ):
            raise M03RV9ProjectionError("projector manifest drifted")


def freeze_m03r_v9_projector_manifest() -> M03RV9ProjectorManifest:
    """Create the one frozen projector configuration for the v9 panel."""

    bounds = tuple(
        0.02 if family == "sector" else 0.10
        for family in M03R_V9_PROJECTOR_EXPOSURE_FAMILIES
    )
    covariance = M03RV9CovarianceEstimatorSpec()
    provisional = M03RV9ProjectorManifest(
        exposure_names=M03R_V9_PROJECTOR_EXPOSURE_NAMES,
        exposure_families=M03R_V9_PROJECTOR_EXPOSURE_FAMILIES,
        exposure_lower_bounds=tuple(-value for value in bounds),
        exposure_upper_bounds=bounds,
        active_beta_exposure_name=M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
        annual_tracking_error_ceiling=0.06,
        active_beta_absolute_bound=0.10,
        maximum_stock_weight_fraction=0.01,
        projection_mode="benchmark-radial-factor-diagonal-beta-te-v1",
        covariance_estimator=covariance,
        manifest_sha256="0" * 64,
    )
    result = replace(
        provisional,
        manifest_sha256=_canonical_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV9ProjectorRiskBinding:
    """Exact closure of the two pre-projector readiness blockers."""

    source_materialization_receipt_sha256: str
    source_exposure_receipt_sha256: str
    source_artifact_file_sha256: str
    source_artifact_manifest_file_sha256: str
    original_inventory_sha256: str
    original_readiness_receipt_sha256: str
    original_blocker_codes: tuple[str, ...]
    projector_manifest_sha256: str
    bound_inventory: M03RV9RiskSourceInventory
    bound_readiness: M03RV9RiskSourceReadiness
    binding_sha256: str
    schema: str = M03R_V9_PROJECTOR_BINDING_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
            "source_materialization_receipt_sha256": (
                self.source_materialization_receipt_sha256
            ),
            "source_exposure_receipt_sha256": self.source_exposure_receipt_sha256,
            "source_artifact_file_sha256": self.source_artifact_file_sha256,
            "source_artifact_manifest_file_sha256": (
                self.source_artifact_manifest_file_sha256
            ),
            "original_inventory_sha256": self.original_inventory_sha256,
            "original_readiness_receipt_sha256": (
                self.original_readiness_receipt_sha256
            ),
            "original_blocker_codes": self.original_blocker_codes,
            "projector_manifest_sha256": self.projector_manifest_sha256,
            "bound_inventory": asdict(self.bound_inventory),
            "bound_readiness": asdict(self.bound_readiness),
            "source_artifact_rewritten": False,
            "predictive_worker_authorized": True,
            "economic_panel_authorized": False,
            "research_only": True,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        self.bound_inventory.validate()
        self.bound_readiness.validate()
        for name in (
            "source_materialization_receipt_sha256",
            "source_exposure_receipt_sha256",
            "source_artifact_file_sha256",
            "source_artifact_manifest_file_sha256",
            "original_inventory_sha256",
            "original_readiness_receipt_sha256",
            "projector_manifest_sha256",
            "binding_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.schema != M03R_V9_PROJECTOR_BINDING_SCHEMA
            or self.original_blocker_codes != M03R_V9_EXPECTED_PREPROJECTOR_BLOCKERS
            or self.bound_inventory.projector_manifest_sha256
            != self.projector_manifest_sha256
            or not self.bound_inventory.target_projector_exposure_names_match
            or self.bound_readiness.blocker_codes
            or not self.bound_readiness.predictive_worker_authorized
            or self.bound_readiness.economic_panel_authorized
            or self.bound_readiness.source_inventory_sha256
            != _canonical_sha256(asdict(self.bound_inventory))
            or self.binding_sha256 != _canonical_sha256(self.unsigned_payload())
        ):
            raise M03RV9ProjectionError("projector/source binding drifted")


def bind_m03r_v9_projector_to_risk_source(
    source: M03RV9MaterializedRiskSource,
    written: M03RV9WrittenRiskSource,
    projector: M03RV9ProjectorManifest,
) -> M03RV9ProjectorRiskBinding:
    """Close only the known projector blockers on an immutable risk artifact."""

    source.validate()
    projector.validate()
    if (
        source.readiness.blocker_codes != M03R_V9_EXPECTED_PREPROJECTOR_BLOCKERS
        or source.inventory.projector_manifest_sha256 is not None
        or source.inventory.target_projector_exposure_names_match
        or source.exposures.projector_exposure_names != projector.exposure_names
        or source.exposures.projector_exposure_families != projector.exposure_families
        or source.inventory.sector_exposure_names != M03R_V9_SECTOR_EXPOSURE_NAMES
        or source.inventory.style_risk_exposure_names
        != M03R_V9_STYLE_RISK_EXPOSURE_NAMES
        or source.inventory.active_beta_exposure_name
        != projector.active_beta_exposure_name
        or source.inventory.asset_axis_sha256 != source.exposures.asset_axis_sha256
    ):
        raise M03RV9ProjectionError(
            "materialized source does not match the frozen projector boundary"
        )
    bound_inventory = replace(
        source.inventory,
        projector_manifest_sha256=projector.manifest_sha256,
        target_projector_exposure_names_match=True,
    )
    bound_readiness = audit_m03r_v9_risk_source(bound_inventory)
    bound_readiness.require_predictive_worker_authorized()
    original_inventory_sha = _canonical_sha256(asdict(source.inventory))
    provisional = M03RV9ProjectorRiskBinding(
        source_materialization_receipt_sha256=source.receipt_sha256,
        source_exposure_receipt_sha256=source.exposures.receipt_sha256,
        source_artifact_file_sha256=_digest(
            "source_artifact_file_sha256", written.artifact_file_sha256
        ),
        source_artifact_manifest_file_sha256=_digest(
            "source_artifact_manifest_file_sha256", written.manifest_file_sha256
        ),
        original_inventory_sha256=original_inventory_sha,
        original_readiness_receipt_sha256=source.readiness.receipt_sha256,
        original_blocker_codes=source.readiness.blocker_codes,
        projector_manifest_sha256=projector.manifest_sha256,
        bound_inventory=bound_inventory,
        bound_readiness=bound_readiness,
        binding_sha256="0" * 64,
    )
    result = replace(
        provisional,
        binding_sha256=_canonical_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def write_m03r_v9_projector_manifest(
    projector: M03RV9ProjectorManifest,
    binding: M03RV9ProjectorRiskBinding,
    output_path: Path,
) -> str:
    """Publish the projector and source binding once, without clobber."""

    projector.validate()
    binding.validate()
    if binding.projector_manifest_sha256 != projector.manifest_sha256:
        raise M03RV9ProjectionError("projector and source binding disagree")
    payload = {
        "schema": M03R_V9_PROJECTOR_MANIFEST_FILE_SCHEMA,
        "projector": asdict(projector),
        "binding": asdict(binding),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return _file_sha256(output_path)


def load_m03r_v9_projector_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[M03RV9ProjectorManifest, M03RV9ProjectorRiskBinding]:
    regular = _require_regular(path, name="projector manifest")
    if _file_sha256(regular) != _digest("expected_file_sha256", expected_file_sha256):
        raise M03RV9ProjectionError("projector manifest file hash mismatch")
    payload = json.loads(regular.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != M03R_V9_PROJECTOR_MANIFEST_FILE_SCHEMA
        or not isinstance(payload.get("projector"), dict)
        or not isinstance(payload.get("binding"), dict)
    ):
        raise M03RV9ProjectionError("projector manifest file schema drifted")
    projector_payload = dict(payload["projector"])
    projector_payload["exposure_names"] = tuple(projector_payload["exposure_names"])
    projector_payload["exposure_families"] = tuple(
        projector_payload["exposure_families"]
    )
    projector_payload["exposure_lower_bounds"] = tuple(
        projector_payload["exposure_lower_bounds"]
    )
    projector_payload["exposure_upper_bounds"] = tuple(
        projector_payload["exposure_upper_bounds"]
    )
    projector_payload["covariance_estimator"] = M03RV9CovarianceEstimatorSpec(
        **projector_payload["covariance_estimator"]
    )
    projector = M03RV9ProjectorManifest(**projector_payload)
    binding_payload = dict(payload["binding"])
    binding_payload["original_blocker_codes"] = tuple(
        binding_payload["original_blocker_codes"]
    )
    inventory_payload = dict(binding_payload["bound_inventory"])
    for key in (
        "source_columns",
        "sector_exposure_names",
        "style_risk_exposure_names",
    ):
        inventory_payload[key] = tuple(inventory_payload[key])
    binding_payload["bound_inventory"] = M03RV9RiskSourceInventory(**inventory_payload)
    readiness_payload = dict(binding_payload["bound_readiness"])
    readiness_payload["blocker_codes"] = tuple(readiness_payload["blocker_codes"])
    readiness_payload["required_exposure_families"] = tuple(
        readiness_payload["required_exposure_families"]
    )
    binding_payload["bound_readiness"] = M03RV9RiskSourceReadiness(**readiness_payload)
    binding = M03RV9ProjectorRiskBinding(**binding_payload)
    projector.validate()
    binding.validate()
    if binding.projector_manifest_sha256 != projector.manifest_sha256:
        raise M03RV9ProjectionError("loaded projector and source binding disagree")
    return projector, binding


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
        raise M03RV9ProjectionError(
            f"{name} must be detached finite float32/float64 {shape}"
        )
    return value


def _winsorize_available(
    values: torch.Tensor,
    available: torch.Tensor,
    tail_fraction: float,
) -> torch.Tensor:
    selected = values[available]
    if selected.numel() < 2:
        return torch.zeros_like(values)
    lower = torch.quantile(selected, tail_fraction)
    upper = torch.quantile(selected, 1.0 - tail_fraction)
    return torch.where(available, values.clamp(lower, upper), 0.0)


def _past_factor_and_residual_returns(
    source: M03RV9MaterializedRiskSource,
    daily_log_returns: torch.Tensor,
    return_available: torch.Tensor,
    stop_state_index: int,
    ridge_lambda: float,
    winsor_tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit decision-origin cross-sectional factors strictly before ``stop``."""

    exposures = source.exposures.exposure_loadings.to(dtype=torch.float64)
    weights = source.exposures.regression_weights.to(dtype=torch.float64)
    cash = source.exposures.cash_index
    factors = exposures.shape[-1]
    factor_returns = torch.zeros((stop_state_index, factors), dtype=torch.float64)
    residual_returns = torch.zeros_like(daily_log_returns[:stop_state_index])
    usable = torch.zeros_like(return_available[:stop_state_index])
    identity = torch.eye(factors, dtype=torch.float64)
    identity[0, 0] = 0.0
    for state_index in range(1, stop_state_index):
        row_available = return_available[state_index].clone() & (
            weights[state_index - 1] > 0.0
        )
        row_available[cash] = False
        if int(row_available.sum()) <= factors:
            continue
        target = _winsorize_available(
            daily_log_returns[state_index],
            row_available,
            winsor_tail_fraction,
        )
        # The return ending at ``state_index`` is explained only with the
        # loadings frozen at the preceding decision.  Endpoint metadata must
        # not leak into the covariance history.
        design = exposures[state_index - 1]
        row_weight = torch.where(row_available, weights[state_index - 1], 0.0)
        weighted_design = design * row_weight.unsqueeze(-1)
        gram = design.T @ weighted_design + ridge_lambda * identity
        rhs = weighted_design.T @ target
        coefficients = torch.linalg.solve(gram, rhs)
        fitted = design @ coefficients
        factor_returns[state_index] = coefficients
        residual_returns[state_index] = torch.where(
            row_available,
            target - fitted,
            0.0,
        )
        usable[state_index] = row_available
    return factor_returns, residual_returns, usable


def _covariance_components_at_update(
    *,
    factor_returns: torch.Tensor,
    residual_returns: torch.Tensor,
    usable: torch.Tensor,
    update_state_index: int,
    spec: M03RV9CovarianceEstimatorSpec,
    cash_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(1, update_state_index - spec.lookback_sessions)
    factor_window = factor_returns[start:update_state_index, 1:]
    if factor_window.shape[0] < spec.minimum_history_sessions:
        raise M03RV9ProjectionError(
            "risk origin lacks the frozen minimum past-only history"
        )
    centered = factor_window - factor_window.mean(dim=0, keepdim=True)
    denominator = max(1, factor_window.shape[0] - 1)
    factor_covariance = centered.T @ centered / float(denominator)
    diagonal = torch.diag(torch.diag(factor_covariance))
    factor_covariance = (
        1.0 - spec.factor_covariance_shrinkage_to_diagonal
    ) * factor_covariance + spec.factor_covariance_shrinkage_to_diagonal * diagonal
    eigenvalues, eigenvectors = torch.linalg.eigh(factor_covariance)
    eigenvalues = eigenvalues.clamp_min(spec.psd_eigenvalue_floor)
    factor_square_root = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

    residual_window = residual_returns[start:update_state_index]
    available_window = usable[start:update_state_index]
    counts = available_window.sum(dim=0)
    residual_mean = torch.where(
        counts > 0,
        residual_window.sum(dim=0) / counts.clamp_min(1),
        torch.zeros_like(counts, dtype=torch.float64),
    )
    squared = torch.where(
        available_window,
        (residual_window - residual_mean.unsqueeze(0)).square(),
        0.0,
    ).sum(dim=0)
    specific = torch.where(
        counts > 1,
        squared / (counts - 1).clamp_min(1),
        0.0,
    )
    risky = torch.ones_like(specific, dtype=torch.bool)
    risky[cash_index] = False
    observed = risky & (counts >= spec.minimum_history_sessions)
    if not bool(observed.any()):
        raise M03RV9ProjectionError("specific-risk window has no qualified assets")
    median = specific[observed].median()
    specific = torch.where(observed, specific, median)
    specific = (
        (1.0 - spec.specific_variance_shrinkage_to_cross_sectional_median) * specific
        + spec.specific_variance_shrinkage_to_cross_sectional_median * median
    ).clamp_min(spec.specific_variance_floor)
    specific[cash_index] = 0.0
    return factor_square_root, specific


@dataclass(frozen=True, slots=True)
class M03RV9DeviceRiskState:
    """Once-qualified fold-local risk tensors held on one execution device."""

    asset_count: int
    cash_index: int
    origin_state_indices: tuple[int, ...]
    manifest_sha256: str
    source_binding_sha256: str
    source_exposure_receipt_sha256: str
    daily_returns_receipt_sha256: str
    asset_axis_sha256: str
    exposure_loadings: torch.Tensor  # [origin, asset, exposure]
    exposure_lower_bounds: torch.Tensor  # [exposure]
    exposure_upper_bounds: torch.Tensor  # [exposure]
    active_beta_loadings: torch.Tensor  # [origin, asset]
    covariance_factor: torch.Tensor  # [origin, asset, factor]
    specific_variance: torch.Tensor  # [origin, asset]
    tensor_sha256: tuple[str, str, str, str, str, str]
    tensor_version_counters: tuple[int, int, int, int, int, int]
    state_sha256: str
    schema: str = M03R_V9_DEVICE_RISK_STATE_SCHEMA
    _issuer: object | None = field(repr=False, compare=False, default=None)

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
            "asset_count": self.asset_count,
            "cash_index": self.cash_index,
            "origin_state_indices": self.origin_state_indices,
            "manifest_sha256": self.manifest_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "source_exposure_receipt_sha256": self.source_exposure_receipt_sha256,
            "daily_returns_receipt_sha256": self.daily_returns_receipt_sha256,
            "asset_axis_sha256": self.asset_axis_sha256,
            "tensor_sha256": self.tensor_sha256,
            "research_only": True,
            "development_only": True,
            "economic_panel_authorized": False,
        }

    def validate(self) -> None:
        origins = len(self.origin_state_indices)
        exposures = self.exposure_loadings.shape[-1]
        factors = self.covariance_factor.shape[-1]
        tensors = (
            _finite_tensor(
                "exposure_loadings",
                self.exposure_loadings,
                (origins, self.asset_count, exposures),
            ),
            _finite_tensor(
                "exposure_lower_bounds",
                self.exposure_lower_bounds,
                (exposures,),
            ),
            _finite_tensor(
                "exposure_upper_bounds",
                self.exposure_upper_bounds,
                (exposures,),
            ),
            _finite_tensor(
                "active_beta_loadings",
                self.active_beta_loadings,
                (origins, self.asset_count),
            ),
            _finite_tensor(
                "covariance_factor",
                self.covariance_factor,
                (origins, self.asset_count, factors),
            ),
            _finite_tensor(
                "specific_variance",
                self.specific_variance,
                (origins, self.asset_count),
            ),
        )
        for name in (
            "manifest_sha256",
            "source_binding_sha256",
            "source_exposure_receipt_sha256",
            "daily_returns_receipt_sha256",
            "asset_axis_sha256",
            "state_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self._issuer is not _DEVICE_RISK_STATE_ISSUER
            or self.schema != M03R_V9_DEVICE_RISK_STATE_SCHEMA
            or self.asset_count < 3
            or not 0 <= self.cash_index < self.asset_count
            or not self.origin_state_indices
            or tuple(sorted(set(self.origin_state_indices)))
            != self.origin_state_indices
            or exposures != len(M03R_V9_PROJECTOR_EXPOSURE_NAMES)
            or factors != len(M03R_V9_PROJECTOR_EXPOSURE_NAMES)
            or bool((self.exposure_lower_bounds >= 0.0).any())
            or bool((self.exposure_upper_bounds <= 0.0).any())
            or bool((self.specific_variance < 0.0).any())
            or not torch.equal(
                self.exposure_loadings[:, self.cash_index],
                torch.zeros_like(self.exposure_loadings[:, self.cash_index]),
            )
            or not torch.equal(
                self.active_beta_loadings[:, self.cash_index],
                torch.zeros_like(self.active_beta_loadings[:, self.cash_index]),
            )
            or not torch.equal(
                self.covariance_factor[:, self.cash_index],
                torch.zeros_like(self.covariance_factor[:, self.cash_index]),
            )
            or not torch.equal(
                self.specific_variance[:, self.cash_index],
                torch.zeros_like(self.specific_variance[:, self.cash_index]),
            )
            or tuple(_tensor_sha256(value) for value in tensors) != self.tensor_sha256
            or tuple(int(value._version) for value in tensors)
            != self.tensor_version_counters
            or _canonical_sha256(self.unsigned_payload()) != self.state_sha256
        ):
            raise M03RV9ProjectionError("device risk state drifted")

    def require_fast_identity(
        self,
        *,
        sequence_asset_axis_sha256: str,
        checkpoint_asset_axis_sha256: str,
        expected_manifest_sha256: str,
    ) -> None:
        if (
            _digest("sequence_asset_axis_sha256", sequence_asset_axis_sha256)
            != self.asset_axis_sha256
            or _digest("checkpoint_asset_axis_sha256", checkpoint_asset_axis_sha256)
            != self.asset_axis_sha256
            or _digest("expected_manifest_sha256", expected_manifest_sha256)
            != self.manifest_sha256
            or tuple(
                int(value._version)
                for value in (
                    self.exposure_loadings,
                    self.exposure_lower_bounds,
                    self.exposure_upper_bounds,
                    self.active_beta_loadings,
                    self.covariance_factor,
                    self.specific_variance,
                )
            )
            != self.tensor_version_counters
        ):
            raise M03RV9ProjectionError(
                "risk state asset identity or tensor version drifted"
            )


def build_m03r_v9_device_risk_state(
    source: M03RV9MaterializedRiskSource,
    binding: M03RV9ProjectorRiskBinding,
    projector: M03RV9ProjectorManifest,
    *,
    daily_log_returns: torch.Tensor,
    return_available: torch.Tensor,
    daily_returns_receipt_sha256: str,
    sequence_asset_axis_sha256: str,
    checkpoint_asset_axis_sha256: str,
    origin_state_indices: tuple[int, ...],
    device: torch.device,
) -> M03RV9DeviceRiskState:
    """Fully validate and transfer causal factor-plus-diagonal fold risk once."""

    source.validate()
    binding.validate()
    projector.validate()
    binding.bound_readiness.require_predictive_worker_authorized()
    axis = source.exposures.asset_axis_sha256
    if (
        sequence_asset_axis_sha256 != axis
        or checkpoint_asset_axis_sha256 != axis
        or binding.source_materialization_receipt_sha256 != source.receipt_sha256
        or binding.source_exposure_receipt_sha256 != source.exposures.receipt_sha256
        or binding.projector_manifest_sha256 != projector.manifest_sha256
        or not origin_state_indices
        or tuple(sorted(set(origin_state_indices))) != origin_state_indices
    ):
        raise M03RV9ProjectionError("risk state source or asset-axis identity drifted")
    states, assets = source.exposures.exposure_loadings.shape[:2]
    if (
        not isinstance(daily_log_returns, torch.Tensor)
        or tuple(daily_log_returns.shape) != (states, assets)
        or daily_log_returns.dtype not in {torch.float32, torch.float64}
        or daily_log_returns.requires_grad
        or not bool(torch.isfinite(daily_log_returns).all())
        or not isinstance(return_available, torch.Tensor)
        or tuple(return_available.shape) != (states, assets)
        or return_available.dtype != torch.bool
        or return_available.requires_grad
        or origin_state_indices[0]
        < projector.covariance_estimator.minimum_history_sessions
        or origin_state_indices[-1] >= states
    ):
        raise M03RV9ProjectionError("past-return risk inputs are malformed")
    returns64 = daily_log_returns.detach().to(device="cpu", dtype=torch.float64)
    available = return_available.detach().to(device="cpu").clone()
    available[:, source.exposures.cash_index] = False
    factor_returns, residual_returns, usable = _past_factor_and_residual_returns(
        source,
        returns64,
        available,
        origin_state_indices[-1] + 1,
        ridge_lambda=1.0e-6,
        winsor_tail_fraction=(
            projector.covariance_estimator.cross_sectional_winsor_tail_fraction
        ),
    )
    factor_squares: dict[int, torch.Tensor] = {}
    specific_by_update: dict[int, torch.Tensor] = {}
    covariance_factors: list[torch.Tensor] = []
    specific_rows: list[torch.Tensor] = []
    first = origin_state_indices[0]
    source_loadings = source.exposures.exposure_loadings.to(dtype=torch.float64)
    for origin in origin_state_indices:
        update = (
            first
            + (
                (origin - first)
                // projector.covariance_estimator.update_frequency_sessions
            )
            * projector.covariance_estimator.update_frequency_sessions
        )
        if update not in factor_squares:
            factor_square, specific = _covariance_components_at_update(
                factor_returns=factor_returns,
                residual_returns=residual_returns,
                usable=usable,
                update_state_index=update,
                spec=projector.covariance_estimator,
                cash_index=source.exposures.cash_index,
            )
            factor_squares[update] = factor_square
            specific_by_update[update] = specific
        origin_loadings = source_loadings[origin, :, 1:]
        covariance_factors.append(origin_loadings @ factor_squares[update])
        specific_rows.append(specific_by_update[update])

    beta_index = projector.exposure_names.index(projector.active_beta_exposure_name)
    tensors_cpu = (
        source_loadings[list(origin_state_indices), :, 1:].clone(),
        torch.tensor(projector.exposure_lower_bounds, dtype=torch.float64),
        torch.tensor(projector.exposure_upper_bounds, dtype=torch.float64),
        source_loadings[list(origin_state_indices), :, 1 + beta_index].clone(),
        torch.stack(covariance_factors),
        torch.stack(specific_rows),
    )
    tensors = tuple(value.to(device=device).clone() for value in tensors_cpu)
    tensor_hashes = tuple(_tensor_sha256(value) for value in tensors)
    provisional = M03RV9DeviceRiskState(
        asset_count=assets,
        cash_index=source.exposures.cash_index,
        origin_state_indices=origin_state_indices,
        manifest_sha256=projector.manifest_sha256,
        source_binding_sha256=binding.binding_sha256,
        source_exposure_receipt_sha256=source.exposures.receipt_sha256,
        daily_returns_receipt_sha256=_digest(
            "daily_returns_receipt_sha256", daily_returns_receipt_sha256
        ),
        asset_axis_sha256=axis,
        exposure_loadings=tensors[0],
        exposure_lower_bounds=tensors[1],
        exposure_upper_bounds=tensors[2],
        active_beta_loadings=tensors[3],
        covariance_factor=tensors[4],
        specific_variance=tensors[5],
        tensor_sha256=tensor_hashes,  # type: ignore[arg-type]
        tensor_version_counters=tuple(  # type: ignore[arg-type]
            int(value._version) for value in tensors
        ),
        state_sha256="0" * 64,
        _issuer=_DEVICE_RISK_STATE_ISSUER,
    )
    result = replace(
        provisional,
        state_sha256=_canonical_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def _annual_tracking_error_factor_diagonal(
    active: torch.Tensor,
    covariance_factor: torch.Tensor,
    specific_variance: torch.Tensor,
) -> torch.Tensor:
    factor_exposure = active @ covariance_factor
    variance = factor_exposure.square().sum(-1) + (
        active.square() * specific_variance
    ).sum(-1)
    return torch.sqrt((252.0 * variance).clamp_min(0.0))


@dataclass(frozen=True, slots=True)
class M03RV9ProjectionResult:
    projected_weights: torch.Tensor
    radial_scale: torch.Tensor
    requested_factor_exposure: torch.Tensor
    projected_factor_exposure: torch.Tensor
    requested_active_beta: torch.Tensor
    projected_active_beta: torch.Tensor
    requested_annual_tracking_error: torch.Tensor
    projected_annual_tracking_error: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    risk_manifest_sha256: str
    risk_state_sha256: str
    origin_state_index: int
    schema: str = M03R_V9_PROJECTION_RESULT_SCHEMA


@dataclass(frozen=True, slots=True)
class M03RV9SignalProjection:
    projected_signal: torch.Tensor
    requested_factor_component: torch.Tensor
    projected_factor_component: torch.Tensor
    signal_retention: torch.Tensor
    risk_manifest_sha256: str
    risk_state_sha256: str
    origin_state_index: int
    schema: str = M03R_V9_SIGNAL_PROJECTION_SCHEMA


def project_m03r_v9_signal_to_exposure_null(
    signal: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_state: M03RV9DeviceRiskState,
    *,
    origin_state_index: int,
    sequence_asset_axis_sha256: str,
    checkpoint_asset_axis_sha256: str,
    expected_manifest_sha256: str,
    ridge_lambda: float = 1.0e-6,
) -> M03RV9SignalProjection:
    """Remove signal components the final projector is designed to reject."""

    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=sequence_asset_axis_sha256,
        checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if origin_state_index not in risk_state.origin_state_indices:
        raise M03RV9ProjectionError("signal risk origin is not qualified")
    if (
        not isinstance(signal, torch.Tensor)
        or signal.ndim != 2
        or signal.shape[1] != risk_state.asset_count
        or signal.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(signal).all())
        or not isinstance(trade_mask, torch.Tensor)
        or trade_mask.shape != signal.shape
        or trade_mask.dtype != torch.bool
        or trade_mask.device != signal.device
        or isinstance(ridge_lambda, bool)
        or not isinstance(ridge_lambda, (int, float))
        or not math.isfinite(float(ridge_lambda))
        or ridge_lambda <= 0.0
    ):
        raise M03RV9ProjectionError("signal-null projection inputs are malformed")
    row = risk_state.origin_state_indices.index(origin_state_index)
    loadings = risk_state.exposure_loadings[row].to(
        device=signal.device,
        dtype=torch.float64,
    )
    result_rows: list[torch.Tensor] = []
    requested_components: list[torch.Tensor] = []
    projected_components: list[torch.Tensor] = []
    retention_rows: list[torch.Tensor] = []
    for batch_index in range(signal.shape[0]):
        available = trade_mask[batch_index].clone()
        available[risk_state.cash_index] = False
        selected = torch.nonzero(available, as_tuple=False).flatten()
        if selected.numel() <= loadings.shape[1]:
            raise M03RV9ProjectionError(
                "signal-null projection has insufficient tradable assets"
            )
        design = loadings.index_select(0, selected)
        target = signal[batch_index].to(torch.float64).index_select(0, selected)
        gram = design.T @ design + float(ridge_lambda) * torch.eye(
            design.shape[1],
            device=design.device,
            dtype=design.dtype,
        )
        coefficients = torch.linalg.solve(gram, design.T @ target)
        projected = signal[batch_index].new_zeros(signal.shape[1])
        projected_selected = target - design @ coefficients
        projected[selected] = projected_selected.to(signal.dtype)
        requested_component = loadings.T @ signal[batch_index].to(torch.float64)
        projected_component = loadings.T @ projected.to(torch.float64)
        requested_norm = torch.linalg.vector_norm(target)
        projected_norm = torch.linalg.vector_norm(projected_selected)
        retention = torch.where(
            requested_norm > 0.0,
            projected_norm / requested_norm.clamp_min(torch.finfo(torch.float64).tiny),
            torch.ones_like(requested_norm),
        )
        result_rows.append(projected)
        requested_components.append(requested_component)
        projected_components.append(projected_component)
        retention_rows.append(retention)
    return M03RV9SignalProjection(
        projected_signal=torch.stack(result_rows),
        requested_factor_component=torch.stack(requested_components),
        projected_factor_component=torch.stack(projected_components),
        signal_retention=torch.stack(retention_rows),
        risk_manifest_sha256=risk_state.manifest_sha256,
        risk_state_sha256=risk_state.state_sha256,
        origin_state_index=origin_state_index,
    )


def project_m03r_v9_active_book(
    requested_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    risk_state: M03RV9DeviceRiskState,
    *,
    origin_state_index: int,
    sequence_asset_axis_sha256: str,
    checkpoint_asset_axis_sha256: str,
    expected_manifest_sha256: str,
) -> M03RV9ProjectionResult:
    """Apply the safety radial projection using only hot-path identity checks."""

    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=sequence_asset_axis_sha256,
        checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if origin_state_index not in risk_state.origin_state_indices:
        raise M03RV9ProjectionError("risk origin is not in the qualified fold state")
    risk_row = risk_state.origin_state_indices.index(origin_state_index)
    if (
        not isinstance(requested_weights, torch.Tensor)
        or requested_weights.ndim != 2
        or requested_weights.shape[1] != risk_state.asset_count
        or requested_weights.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(requested_weights).all())
    ):
        raise M03RV9ProjectionError("requested book is malformed")
    batch, assets = requested_weights.shape
    for name, value in (
        ("benchmark_weights", benchmark_weights),
        ("risk_asset_caps", risk_asset_caps),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != requested_weights.shape
            or value.dtype != requested_weights.dtype
            or value.device != requested_weights.device
            or not bool(torch.isfinite(value).all())
        ):
            raise M03RV9ProjectionError(f"{name} does not match the requested book")
    if (
        not isinstance(trade_mask, torch.Tensor)
        or trade_mask.shape != requested_weights.shape
        or trade_mask.dtype != torch.bool
        or trade_mask.device != requested_weights.device
        or not isinstance(risk_gross_max, torch.Tensor)
        or tuple(risk_gross_max.shape) != (batch,)
        or risk_gross_max.dtype != requested_weights.dtype
        or risk_gross_max.device != requested_weights.device
        or not bool(torch.isfinite(risk_gross_max).all())
    ):
        raise M03RV9ProjectionError("projection masks or gross limits are malformed")
    cash = risk_state.cash_index
    benchmark = reconcile_cash_simplex_roundoff(
        benchmark_weights,
        cash_index=cash,
        risky_gross_limit=risk_gross_max,
    )
    requested = reconcile_cash_simplex_roundoff(
        requested_weights,
        cash_index=cash,
    )
    available = trade_mask.clone()
    available[:, cash] = True
    caps = torch.minimum(risk_asset_caps.clamp_min(0.0), requested.new_tensor(0.01))
    caps[:, cash] = 1.0
    if (
        bool((benchmark < 0.0).any())
        or bool((benchmark - caps > 2.0e-6).any())
        or bool(torch.where(available, 0.0, benchmark).ne(0.0).any())
    ):
        raise M03RV9ProjectionError("benchmark is infeasible at fill time")
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
    unavailable_active = torch.where(available, 0.0, active.abs()).amax(dim=-1)
    scale = torch.where(unavailable_active > 0.0, 0.0, scale)
    risky = torch.ones(assets, device=requested.device, dtype=torch.bool)
    risky[cash] = False
    benchmark_gross = benchmark[:, risky].sum(-1)
    active_gross = active[:, risky].sum(-1)
    gross_ratio = torch.where(
        active_gross > 0.0,
        (risk_gross_max - benchmark_gross).clamp_min(0.0)
        / active_gross.clamp_min(tiny),
        torch.full_like(active_gross, torch.inf),
    )
    scale = torch.minimum(scale, gross_ratio)

    work_dtype = torch.float64
    active64 = active.to(work_dtype)
    loadings = risk_state.exposure_loadings[risk_row].to(work_dtype)
    factor = active64 @ loadings
    lower = risk_state.exposure_lower_bounds.to(work_dtype)
    upper = risk_state.exposure_upper_bounds.to(work_dtype)
    factor_ratio = torch.where(
        factor > 0.0,
        upper.unsqueeze(0) / factor.clamp_min(torch.finfo(work_dtype).tiny),
        torch.where(
            factor < 0.0,
            lower.unsqueeze(0) / factor.clamp_max(-torch.finfo(work_dtype).tiny),
            torch.full_like(factor, torch.inf),
        ),
    )
    scale = torch.minimum(scale, factor_ratio.amin(-1).to(scale.dtype))
    beta_loadings = risk_state.active_beta_loadings[risk_row].to(work_dtype)
    beta = active64 @ beta_loadings
    beta_ratio = torch.where(
        beta.abs() > 0.0,
        beta.new_tensor(0.10) / beta.abs().clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(beta, torch.inf),
    )
    scale = torch.minimum(scale, beta_ratio.to(scale.dtype))
    covariance_factor = risk_state.covariance_factor[risk_row].to(work_dtype)
    specific = risk_state.specific_variance[risk_row].to(work_dtype)
    tracking_error = _annual_tracking_error_factor_diagonal(
        active64,
        covariance_factor,
        specific,
    )
    te_ratio = torch.where(
        tracking_error > 0.0,
        tracking_error.new_tensor(0.06)
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
    projected_te = _annual_tracking_error_factor_diagonal(
        projected_active64,
        covariance_factor,
        specific,
    )
    tolerance = 2.0e-6
    # CASH is the residual simplex account, not a requested risky signal.  Its
    # value is recomputed after projection and can move by several ULPs even
    # when the risky active book is unchanged.  Including that bookkeeping
    # delta in a weak-signal norm can therefore report retention above one.
    # Measure the economically meaningful risky-active book instead, reject
    # material amplification, and canonicalize sub-tolerance roundoff to one.
    requested_risky_norm = torch.linalg.vector_norm(active64[:, risky], dim=-1)
    projected_risky_norm = torch.linalg.vector_norm(
        projected_active64[:, risky], dim=-1
    )
    raw_retention = torch.where(
        requested_risky_norm > 0.0,
        projected_risky_norm
        / requested_risky_norm.clamp_min(torch.finfo(work_dtype).tiny),
        torch.ones_like(requested_risky_norm),
    )
    if bool((raw_retention > 1.0 + tolerance).any()):
        raise M03RV9ProjectionError("v9 risk projection amplified risky active signal")
    retention = raw_retention.clamp(0.0, 1.0)
    if (
        bool((projected < -tolerance).any())
        or bool((projected - caps > tolerance).any())
        or bool((projected_factor - upper.unsqueeze(0) > tolerance).any())
        or bool((lower.unsqueeze(0) - projected_factor > tolerance).any())
        or bool((projected_beta.abs() > 0.10 + tolerance).any())
        or bool((projected_te > 0.06 + tolerance).any())
    ):
        raise M03RV9ProjectionError("v9 risk projection failed reconciliation")
    return M03RV9ProjectionResult(
        projected_weights=projected,
        radial_scale=scale,
        requested_factor_exposure=factor,
        projected_factor_exposure=projected_factor,
        requested_active_beta=beta,
        projected_active_beta=projected_beta,
        requested_annual_tracking_error=tracking_error,
        projected_annual_tracking_error=projected_te,
        requested_to_executed_retention=retention,
        risk_manifest_sha256=risk_state.manifest_sha256,
        risk_state_sha256=risk_state.state_sha256,
        origin_state_index=origin_state_index,
    )


__all__ = [
    "M03R_V9_DEVICE_RISK_STATE_SCHEMA",
    "M03R_V9_EXPECTED_PREPROJECTOR_BLOCKERS",
    "M03R_V9_PROJECTION_RESULT_SCHEMA",
    "M03R_V9_PROJECTOR_BINDING_SCHEMA",
    "M03R_V9_PROJECTOR_MANIFEST_FILE_SCHEMA",
    "M03R_V9_PROJECTOR_MANIFEST_SCHEMA",
    "M03R_V9_SIGNAL_PROJECTION_SCHEMA",
    "M03RV9CovarianceEstimatorSpec",
    "M03RV9DeviceRiskState",
    "M03RV9ProjectionError",
    "M03RV9ProjectionResult",
    "M03RV9ProjectorManifest",
    "M03RV9ProjectorRiskBinding",
    "M03RV9SignalProjection",
    "bind_m03r_v9_projector_to_risk_source",
    "build_m03r_v9_device_risk_state",
    "freeze_m03r_v9_projector_manifest",
    "load_m03r_v9_projector_manifest",
    "project_m03r_v9_active_book",
    "project_m03r_v9_signal_to_exposure_null",
    "write_m03r_v9_projector_manifest",
]
