"""Content-bound package plan for the M03R-v7 TOP2000 seed-17 diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_ADMISSION_ORDER,
    M03R_SEED17_TOP2000_DATA_ROLE,
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_FOLDS,
    M03R_SEED17_TOP2000_PACKAGE_SCHEMA,
    M03R_SEED17_TOP2000_PANEL,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEEDS,
    M03R_SEED17_TOP2000_SETTING_IDS,
    runtime_setting_id,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH,
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000RuntimeProfile,
)


class M03RV7Seed17PackageError(ValueError):
    """The seed-17 package identity or inventory is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RV7Seed17PackageError(
            "seed-17 package is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7Seed17IndexPlan:
    """One setting worker owns six folds and only seed 17."""

    completion_index: int
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    admission_wave: Literal[1, 2]
    fold_indices: tuple[int, ...] = M03R_SEED17_TOP2000_FOLDS
    paired_seeds: tuple[int, ...] = M03R_SEED17_TOP2000_SEEDS
    one_member_fold_execution: bool = True
    development_only: bool = True
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.completion_index < 12:
            raise M03RV7Seed17PackageError(
                "seed-17 completion index must be in [0, 11]"
            )
        expected_index = M03R_SEED17_TOP2000_ADMISSION_ORDER[
            self.completion_index
        ]
        if (
            self.setting_index != expected_index
            or self.setting_id
            != M03R_SEED17_TOP2000_SETTING_IDS[self.setting_index]
            or self.runtime_setting_id != runtime_setting_id(self.setting_id)
            or self.admission_wave != (1 if self.completion_index < 8 else 2)
            or self.fold_indices != tuple(range(6))
            or self.paired_seeds != (17,)
            or not self.one_member_fold_execution
            or not self.development_only
            or self.promotion_eligible
        ):
            raise M03RV7Seed17PackageError(
                "seed-17 completion identity or six-cell inventory drifted"
            )

    @property
    def fold_seed_cell_count(self) -> int:
        return len(self.fold_indices) * len(self.paired_seeds)


def _build_index(completion_index: int) -> M03RV7Seed17IndexPlan:
    setting_index = M03R_SEED17_TOP2000_ADMISSION_ORDER[completion_index]
    setting_id = M03R_SEED17_TOP2000_SETTING_IDS[setting_index]
    return M03RV7Seed17IndexPlan(
        completion_index=completion_index,
        setting_index=setting_index,
        setting_id=setting_id,
        runtime_setting_id=runtime_setting_id(setting_id),
        admission_wave=1 if completion_index < 8 else 2,
    )


@dataclass(frozen=True, slots=True)
class M03RV7Seed17PackagePlan:
    """Fresh seed-17 package; never aliases the five-seed package hash."""

    artifacts: M03RV7Top2000ArtifactBindings
    indices: tuple[M03RV7Seed17IndexPlan, ...]
    runtime_profile: M03RV7Top2000RuntimeProfile
    plan_artifact_path: str
    benchmark_preflight_sha256: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH
    protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    protocol_generation: str = M03R_SEED17_TOP2000_PROTOCOL_GENERATION
    design_id: str = M03R_SEED17_TOP2000_DESIGN_ID
    data_role: Literal["development-only-nonreportable"] = (
        M03R_SEED17_TOP2000_DATA_ROLE
    )
    one_member_fold_execution: bool = True
    five_seed_ensemble_eligible: bool = False
    promotion_eligible: bool = False
    outer_evaluation_authorized: bool = False

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_SEED17_TOP2000_PACKAGE_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "protocol_sha256": self.protocol_sha256,
            "design_id": self.design_id,
            "data_role": self.data_role,
            "one_member_fold_execution": self.one_member_fold_execution,
            "five_seed_ensemble_eligible": self.five_seed_ensemble_eligible,
            "promotion_eligible": self.promotion_eligible,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "plan_artifact_path": self.plan_artifact_path,
            "benchmark_preflight_sha256": self.benchmark_preflight_sha256,
            "source_pythonpath": self.source_pythonpath,
            "artifacts": asdict(self.artifacts),
            "indices": [asdict(row) for row in self.indices],
            "runtime_profile": asdict(self.runtime_profile),
        }

    def __post_init__(self) -> None:
        expected_source = str(
            PurePosixPath(self.plan_artifact_path).parent / "source" / "src"
        )
        if (
            self.protocol_generation != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
            or self.protocol_sha256 != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or self.design_id != M03R_SEED17_TOP2000_DESIGN_ID
            or self.data_role != M03R_SEED17_TOP2000_DATA_ROLE
            or not self.plan_artifact_path.startswith("/")
            or len(self.benchmark_preflight_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.benchmark_preflight_sha256
            )
            or self.source_pythonpath != expected_source
            or self.indices != tuple(_build_index(index) for index in range(12))
            or not self.one_member_fold_execution
            or self.five_seed_ensemble_eligible
            or self.promotion_eligible
            or self.outer_evaluation_authorized
            or M03R_SEED17_TOP2000_PANEL.total_cells != 72
        ):
            raise M03RV7Seed17PackageError(
                "seed-17 package identity or development-only gate drifted"
            )
        if self.runtime_profile.expected_world_size != 2:
            raise M03RV7Seed17PackageError(
                "seed-17 package requires the same two-H100 runtime"
            )
        if self.package_plan_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17PackageError("seed-17 package hash mismatch")


def build_m03r_v7_seed17_top2000_package_plan(
    *,
    artifacts: M03RV7Top2000ArtifactBindings,
    plan_artifact_path: str,
    benchmark_preflight_sha256: str,
    runtime_profile: M03RV7Top2000RuntimeProfile | None = None,
) -> M03RV7Seed17PackagePlan:
    """Build the deterministic package plan without staging or mutation."""

    profile = runtime_profile or M03RV7Top2000RuntimeProfile()
    fields: dict[str, Any] = {
        "artifacts": artifacts,
        "indices": tuple(_build_index(index) for index in range(12)),
        "runtime_profile": profile,
        "plan_artifact_path": plan_artifact_path,
        "benchmark_preflight_sha256": benchmark_preflight_sha256,
        "source_pythonpath": str(
            PurePosixPath(plan_artifact_path).parent / "source" / "src"
        ),
    }
    unsigned = M03RV7Seed17PackagePlan.__new__(M03RV7Seed17PackagePlan)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    for name, value in {
        "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "protocol_generation": M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
        "design_id": M03R_SEED17_TOP2000_DESIGN_ID,
        "data_role": M03R_SEED17_TOP2000_DATA_ROLE,
        "one_member_fold_execution": True,
        "five_seed_ensemble_eligible": False,
        "promotion_eligible": False,
        "outer_evaluation_authorized": False,
    }.items():
        object.__setattr__(unsigned, name, value)
    digest = _sha256(unsigned.canonical_payload())
    return M03RV7Seed17PackagePlan(**fields, package_plan_sha256=digest)


__all__ = [
    "M03RV7Seed17IndexPlan",
    "M03RV7Seed17PackageError",
    "M03RV7Seed17PackagePlan",
    "build_m03r_v7_seed17_top2000_package_plan",
]
