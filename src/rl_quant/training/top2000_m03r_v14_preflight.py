"""Real-cache structural qualification for every M03R-v14 scheduled origin."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v14_residual_operator import (
    build_m03r_v14_residual_operator,
)
from rl_quant.training.top2000_m03r_v14_fold import (
    render_m03r_v14_fold_geometries,
)

M03R_V14_STRUCTURAL_PREFLIGHT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-real-cache-structural-preflight-v1"
)
M03R_V14_STRUCTURAL_PREFLIGHT_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-real-cache-structural-preflight-file-v1"
)
_MAX_PREFLIGHT_BYTES = 4 * 1024**2


class M03RV14PreflightError(ValueError):
    """The exact data-dependent v14 structural gate failed or drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV14PreflightError(f"{name} must be a lowercase SHA-256")
    return value


def _scheduled_origins() -> tuple[int, ...]:
    origins: set[int] = set()
    for geometry in render_m03r_v14_fold_geometries(1001):
        origins.update(geometry.eligible_training_origins)
        origins.update(
            range(
                geometry.qualification_origin_start_inclusive,
                geometry.qualification_origin_stop_exclusive,
            )
        )
    return tuple(sorted(origins))


@dataclass(frozen=True, slots=True)
class M03RV14StructuralPreflightReceipt:
    cache_sha256: str
    asset_axis_sha256: str
    risk_source_receipt_sha256: str
    exposure_receipt_sha256: str
    fold_geometry_sha256: tuple[str, ...]
    scheduled_origin_sha256: str
    scheduled_origin_count: int
    first_scheduled_origin: int
    last_scheduled_origin: int
    minimum_target_qualified_assets: int
    minimum_action_qualified_assets: int
    minimum_target_residual_degrees_of_freedom: int
    minimum_action_residual_degrees_of_freedom: int
    target_action_mask_difference_origin_count: int
    target_operator_root_sha256: str
    action_operator_root_sha256: str
    all_scheduled_origins_qualified: bool = True
    synthetic_fixture_only: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_STRUCTURAL_PREFLIGHT_SCHEMA

    def validate(self) -> None:
        origins = _scheduled_origins()
        geometries = render_m03r_v14_fold_geometries(1001)
        if (
            self.fold_geometry_sha256
            != tuple(row.receipt_sha256 for row in geometries)
            or self.scheduled_origin_sha256 != _sha256(origins)
            or self.scheduled_origin_count != len(origins)
            or self.first_scheduled_origin != origins[0]
            or self.last_scheduled_origin != origins[-1]
            or self.minimum_target_qualified_assets <= 0
            or self.minimum_action_qualified_assets <= 0
            or self.minimum_target_residual_degrees_of_freedom <= 0
            or self.minimum_action_residual_degrees_of_freedom <= 0
            or not 0
            <= self.target_action_mask_difference_origin_count
            <= self.scheduled_origin_count
            or not self.all_scheduled_origins_qualified
            or self.synthetic_fixture_only
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_STRUCTURAL_PREFLIGHT_SCHEMA
        ):
            raise M03RV14PreflightError("v14 structural preflight receipt drifted")
        for name, value in (
            ("cache_sha256", self.cache_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("risk_source_receipt_sha256", self.risk_source_receipt_sha256),
            ("exposure_receipt_sha256", self.exposure_receipt_sha256),
            ("target_operator_root_sha256", self.target_operator_root_sha256),
            ("action_operator_root_sha256", self.action_operator_root_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def run_m03r_v14_structural_preflight(
    cache: Top2000VerifiedDevelopmentCache,
    risk_source: M03RV9MaterializedRiskSource,
) -> M03RV14StructuralPreflightReceipt:
    """Replay target/action operators on every exact scheduled real-data origin."""

    cache.validate_unmodified()
    risk_source.validate()
    exposures = risk_source.exposures
    if (
        cache.daily_ohlcv.shape[0] != 1001
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or exposures.asset_axis_sha256 != cache.action_hash
        or exposures.state_start_index != 0
        or exposures.exposure_loadings.shape[0] != 1001
        or exposures.exposure_loadings.shape[1] != len(cache.action_ids)
    ):
        raise M03RV14PreflightError(
            "v14 real cache, risk source, or asset axis drifted"
        )
    cash = exposures.cash_index
    target_receipts: list[tuple[int, str]] = []
    action_receipts: list[tuple[int, str]] = []
    target_support: list[int] = []
    action_support: list[int] = []
    target_dof: list[int] = []
    action_dof: list[int] = []
    differing = 0
    for origin in _scheduled_origins():
        target_available = cache.availability[origin + 1 : origin + 5].all(dim=0)
        action_available = cache.availability[origin].clone()
        target_available = target_available.clone()
        target_available[cash] = False
        action_available[cash] = False
        label_available = target_available & action_available
        target = build_m03r_v14_residual_operator(
            available_mask=label_available,
            origin_state_index=origin,
            cash_index=cash,
            exposure_loadings=exposures.exposure_loadings[origin],
            regression_weights=exposures.regression_weights[origin],
            projector_exposure_names=exposures.projector_exposure_names,
            projector_exposure_families=exposures.projector_exposure_families,
            asset_axis_sha256=cache.action_hash,
            source_exposure_receipt_sha256=exposures.receipt_sha256,
        )
        action = build_m03r_v14_residual_operator(
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
        target_receipts.append((origin, target.receipt_sha256))
        action_receipts.append((origin, action.receipt_sha256))
        target_support.append(target.factor_qualified_risky_asset_count)
        action_support.append(action.factor_qualified_risky_asset_count)
        target_dof.append(target.weighted_residual_degrees_of_freedom)
        action_dof.append(action.weighted_residual_degrees_of_freedom)
        differing += int(
            not target.qualified_asset_mask.equal(action.qualified_asset_mask)
        )
    geometries = render_m03r_v14_fold_geometries(1001)
    origins = _scheduled_origins()
    result = M03RV14StructuralPreflightReceipt(
        cache_sha256=cache.cache_sha256,
        asset_axis_sha256=cache.action_hash,
        risk_source_receipt_sha256=risk_source.receipt_sha256,
        exposure_receipt_sha256=exposures.receipt_sha256,
        fold_geometry_sha256=tuple(row.receipt_sha256 for row in geometries),
        scheduled_origin_sha256=_sha256(origins),
        scheduled_origin_count=len(origins),
        first_scheduled_origin=origins[0],
        last_scheduled_origin=origins[-1],
        minimum_target_qualified_assets=min(target_support),
        minimum_action_qualified_assets=min(action_support),
        minimum_target_residual_degrees_of_freedom=min(target_dof),
        minimum_action_residual_degrees_of_freedom=min(action_dof),
        target_action_mask_difference_origin_count=differing,
        target_operator_root_sha256=_sha256(target_receipts),
        action_operator_root_sha256=_sha256(action_receipts),
    )
    result.validate()
    return result


def write_m03r_v14_structural_preflight(
    path: str | Path,
    receipt: M03RV14StructuralPreflightReceipt,
) -> str:
    receipt.validate()
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV14PreflightError("v14 preflight target already exists")
    encoded = (
        json.dumps(
            {
                "schema": M03R_V14_STRUCTURAL_PREFLIGHT_FILE_SCHEMA,
                "receipt": asdict(receipt),
                "receipt_sha256": receipt.receipt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
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
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v14_structural_preflight(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> M03RV14StructuralPreflightReceipt:
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    expected_receipt_sha256 = _digest(
        "expected_receipt_sha256", expected_receipt_sha256
    )
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV14PreflightError("v14 preflight file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PREFLIGHT_BYTES
        ):
            raise M03RV14PreflightError("v14 preflight file type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV14PreflightError("v14 preflight file changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise M03RV14PreflightError("v14 preflight file hash drifted")
    try:
        payload = json.loads(raw)
        row = dict(payload["receipt"])
        row["fold_geometry_sha256"] = tuple(row["fold_geometry_sha256"])
        receipt = M03RV14StructuralPreflightReceipt(**row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV14PreflightError("v14 preflight file is malformed") from exc
    if (
        payload.get("schema") != M03R_V14_STRUCTURAL_PREFLIGHT_FILE_SCHEMA
        or payload.get("receipt_sha256") != expected_receipt_sha256
        or receipt.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV14PreflightError("v14 preflight receipt drifted")
    return receipt


__all__ = [
    "M03R_V14_STRUCTURAL_PREFLIGHT_FILE_SCHEMA",
    "M03R_V14_STRUCTURAL_PREFLIGHT_SCHEMA",
    "M03RV14PreflightError",
    "M03RV14StructuralPreflightReceipt",
    "load_m03r_v14_structural_preflight",
    "run_m03r_v14_structural_preflight",
    "write_m03r_v14_structural_preflight",
]
