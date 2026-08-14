"""Real-data structural qualification for the M03R-v12 predictive package."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
    M03R_V12_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    build_m03r_v11_residual_operator,
)
from rl_quant.training.top2000_m03r_v12_fold import (
    render_m03r_v12_training_shard_plan,
)
from rl_quant.training.top2000_m03r_v12_package import M03RV12PackagePlan

M03R_V12_STRUCTURAL_PREFLIGHT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-real-data-structural-preflight-v1"
)
_MAX_RECEIPT_BYTES = 1024 * 1024


class M03RV12StructuralPreflightError(ValueError):
    """The real-data origin/operator sweep is absent or drifted."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class M03RV12StructuralPreflightReceipt:
    panel_episode_schedule_sha256: str
    cache_file_sha256: str
    risk_artifact_file_sha256: str
    risk_manifest_file_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    scheduled_fold_index: int
    scheduled_origin_count: int
    minimum_origin_state_index: int
    maximum_origin_state_index: int
    maximum_target_state_index: int
    first_origin_exchange_date: str
    last_target_exchange_date: str
    horizons: tuple[int, ...]
    operator_count: int
    minimum_factor_qualified_fraction: float
    minimum_effective_design_rank: int
    maximum_effective_design_rank: int
    minimum_weighted_residual_degrees_of_freedom: int
    operator_inventory_sha256: str
    receipt_sha256: str
    real_data_replayed: bool = True
    outcome_values_read: bool = False
    qualification_tail_read: bool = False
    outer_2026_accessed: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_STRUCTURAL_PREFLIGHT_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        digests = (
            self.panel_episode_schedule_sha256,
            self.cache_file_sha256,
            self.risk_artifact_file_sha256,
            self.risk_manifest_file_sha256,
            self.asset_axis_sha256,
            self.exposure_receipt_sha256,
            self.operator_inventory_sha256,
        )
        if (
            any(not _digest(value) for value in digests)
            or self.scheduled_fold_index != 0
            or self.scheduled_origin_count <= 0
            or not 0
            <= self.minimum_origin_state_index
            <= self.maximum_origin_state_index
            or self.maximum_target_state_index
            != self.maximum_origin_state_index + max(M03R_V12_HORIZONS)
            or self.horizons != M03R_V12_HORIZONS
            or self.operator_count
            != self.scheduled_origin_count * len(M03R_V12_HORIZONS)
            or not 0.0 < self.minimum_factor_qualified_fraction <= 1.0
            or not 0
            < self.minimum_effective_design_rank
            <= self.maximum_effective_design_rank
            or self.minimum_weighted_residual_degrees_of_freedom <= 0
            or not self.real_data_replayed
            or self.outcome_values_read
            or self.qualification_tail_read
            or self.outer_2026_accessed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_STRUCTURAL_PREFLIGHT_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12StructuralPreflightError(
                "v12 real-data structural preflight drifted"
            )


def run_m03r_v12_real_data_structural_preflight(
    package: M03RV12PackagePlan,
    *,
    cache_path: str | Path,
    risk_manifest_path: str | Path,
    output_path: str | Path,
) -> M03RV12StructuralPreflightReceipt:
    """Replay every distinct scheduled fold-0 origin without reading returns."""

    package.validate()
    cache_file = Path(cache_path)
    risk_manifest = Path(risk_manifest_path)
    risk_artifact = risk_manifest.parent / "risk-exposures.pt"
    if (
        _file_sha256(cache_file) != package.artifacts.cache_artifact_sha256
        or _file_sha256(risk_manifest)
        != package.artifacts.risk_source_manifest_file_sha256
        or _file_sha256(risk_artifact) != package.artifacts.risk_artifact_sha256
    ):
        raise M03RV12StructuralPreflightError(
            "v12 preflight inputs disagree with package plan"
        )
    cache = load_verified_top2000_hold30_development_cache(
        cache_file,
        expected_cache_sha256=package.artifacts.cache_artifact_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    risk, _written = load_top2000_m03r_v9_risk_source(
        risk_manifest,
        expected_manifest_file_sha256=(
            package.artifacts.risk_source_manifest_file_sha256
        ),
    )
    if risk.action_hash != cache.action_hash:
        raise M03RV12StructuralPreflightError("v12 preflight asset axes differ")
    fold = render_top2000_m03r_v7_development_folds(len(cache.exchange_dates))[0]
    worker = package.panel.workers[0]
    origins = tuple(
        sorted(
            {
                origin
                for completed_update in range(worker.predictive_optimizer_updates)
                for origin in render_m03r_v12_training_shard_plan(
                    worker,
                    package.schedule,
                    fold,
                    completed_update=completed_update,
                ).global_origins
            }
        )
    )
    operators: list[str] = []
    fractions: list[float] = []
    ranks: list[int] = []
    degrees: list[int] = []
    exposures = risk.exposures
    for origin in origins:
        exposure_row = origin - exposures.state_start_index
        for horizon in M03R_V12_HORIZONS:
            available = cache.availability[origin + 1 : origin + horizon + 2].all(dim=0)
            operator = build_m03r_v11_residual_operator(
                origin_state_index=origin,
                cash_index=exposures.cash_index,
                available_mask=available,
                exposure_loadings=exposures.exposure_loadings[exposure_row],
                regression_weights=exposures.regression_weights[exposure_row],
                projector_exposure_names=exposures.projector_exposure_names,
                projector_exposure_families=exposures.projector_exposure_families,
                asset_axis_sha256=cache.action_hash,
                source_exposure_receipt_sha256=exposures.receipt_sha256,
            )
            operators.append(operator.receipt_sha256)
            fractions.append(operator.factor_qualified_fraction)
            ranks.append(operator.effective_design_rank)
            degrees.append(operator.weighted_residual_degrees_of_freedom)
    maximum_target = origins[-1] + max(M03R_V12_HORIZONS)
    provisional = M03RV12StructuralPreflightReceipt(
        panel_episode_schedule_sha256=package.schedule.receipt_sha256,
        cache_file_sha256=package.artifacts.cache_artifact_sha256,
        risk_artifact_file_sha256=package.artifacts.risk_artifact_sha256,
        risk_manifest_file_sha256=(package.artifacts.risk_source_manifest_file_sha256),
        asset_axis_sha256=cache.action_hash,
        exposure_receipt_sha256=exposures.receipt_sha256,
        scheduled_fold_index=0,
        scheduled_origin_count=len(origins),
        minimum_origin_state_index=origins[0],
        maximum_origin_state_index=origins[-1],
        maximum_target_state_index=maximum_target,
        first_origin_exchange_date=cache.exchange_dates[origins[0]],
        last_target_exchange_date=cache.exchange_dates[maximum_target],
        horizons=M03R_V12_HORIZONS,
        operator_count=len(operators),
        minimum_factor_qualified_fraction=min(fractions),
        minimum_effective_design_rank=min(ranks),
        maximum_effective_design_rank=max(ranks),
        minimum_weighted_residual_degrees_of_freedom=min(degrees),
        operator_inventory_sha256=_sha256(operators),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    target = Path(output_path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(asdict(result)))
        stream.flush()
        os.fsync(stream.fileno())
    return result


def load_m03r_v12_structural_preflight(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV12PackagePlan | None = None,
) -> M03RV12StructuralPreflightReceipt:
    source = Path(path)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or not 0 < status.st_size <= _MAX_RECEIPT_BYTES
        ):
            raise M03RV12StructuralPreflightError("v12 preflight file is invalid")
        content = os.read(descriptor, status.st_size + 1)
    finally:
        os.close(descriptor)
    if hashlib.sha256(content).hexdigest() != expected_file_sha256:
        raise M03RV12StructuralPreflightError("v12 preflight file hash drifted")
    row = json.loads(content)
    row["horizons"] = tuple(row["horizons"])
    result = M03RV12StructuralPreflightReceipt(**row)
    result.validate()
    if package is not None and (
        result.panel_episode_schedule_sha256 != package.schedule.receipt_sha256
        or result.cache_file_sha256 != package.artifacts.cache_artifact_sha256
        or result.risk_artifact_file_sha256 != package.artifacts.risk_artifact_sha256
        or result.risk_manifest_file_sha256
        != package.artifacts.risk_source_manifest_file_sha256
    ):
        raise M03RV12StructuralPreflightError(
            "v12 preflight is not bound to the package"
        )
    return result


__all__ = [
    "M03R_V12_STRUCTURAL_PREFLIGHT_SCHEMA",
    "M03RV12StructuralPreflightError",
    "M03RV12StructuralPreflightReceipt",
    "load_m03r_v12_structural_preflight",
    "run_m03r_v12_real_data_structural_preflight",
]
