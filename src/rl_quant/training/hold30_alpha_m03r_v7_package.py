"""Content-bound TOP2000 development package plan for M03R v7.

The module deliberately owns no cluster client and performs no file staging.
It separates source, cache, image, data, and plan identities and stays
launch-blocked until an executable worker receipt and a matching two-H100
capacity receipt are supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from rl_quant.protocol.hold30_alpha_m03r_v7_schedule import (
    M03R_V7_ADMISSION_ORDER,
    M03R_V7_H100_TOPOLOGY,
    M03R_V7_PANEL,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_DATA_ROLE,
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_TOP2000_DEV_SETTING_IDS,
    M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID,
)

M03R_TOP2000_PACKAGE_SCHEMA = "rl-quant.m03r-v7-top2000-package-v1"
M03R_TOP2000_WORKER_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-worker-qualification-v1"
)
M03R_TOP2000_CAPACITY_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-two-h100-capacity-v2"
)
M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH = "/mnt/package/source/src"
M03R_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-bounded-qualification-v1"
)
M03R_TOP2000_QUALIFICATION_EVIDENCE_SCHEMA = (
    "rl-quant.m03r-v7-top2000-verified-qualification-v1"
)
M03R_TOP2000_QUALIFICATION_STEPS = 4
M03R_TOP2000_GPU_NAME = "NVIDIA H100 80GB HBM3"
M03R_TOP2000_MIN_ALLOCATED_GIB = 60.0
M03R_TOP2000_MAX_ALLOCATED_GIB = 75.0
M03R_TOP2000_MIN_RESERVED_HEADROOM_GIB = 5.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class M03RV7Top2000PackageError(ValueError):
    """A development package identity or qualification is inconsistent."""


@dataclass(frozen=True, slots=True)
class M03RV7Top2000RuntimeProfile:
    """Content-bound optimizer and memory geometry shared by all settings."""

    optimizer_steps_per_fold_seed: int = 64
    max_origin_batch: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    token_dim: int = 512
    raw_stock_chunk: int = 512
    expected_world_size: int = 2
    activation_checkpointing: bool = True
    mixed_precision: Literal["bfloat16"] = "bfloat16"

    def __post_init__(self) -> None:
        if (
            self.optimizer_steps_per_fold_seed <= 0
            or self.max_origin_batch not in {4, 8, 16, 32}
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.grad_clip <= 0.0
            or self.token_dim not in {128, 192, 256, 320, 384, 448, 512}
            or self.raw_stock_chunk not in {128, 256, 512, 1024}
            or self.expected_world_size != 2
            or not self.activation_checkpointing
            or self.mixed_precision != "bfloat16"
        ):
            raise M03RV7Top2000PackageError(
                "TOP2000 runtime profile is outside the qualified two-H100 search surface"
            )


def _canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        value = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV7Top2000PackageError(
            "TOP2000 package payload is not canonical-JSON safe"
        ) from exc
    return value.encode("utf-8")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV7Top2000PackageError(
            f"{name} must be one lowercase hexadecimal SHA-256"
        )


def _require_absolute_container_path(name: str, value: str) -> None:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value in {"/", ""}:
        raise M03RV7Top2000PackageError(
            f"{name} must be a scoped absolute container path"
        )


def _regular_nonsymlink(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise M03RV7Top2000PackageError(f"{label} is absent: {path}") from exc
        if stat.S_ISLNK(mode):
            raise M03RV7Top2000PackageError(
                f"{label} contains a symlink component: {path}"
            )
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise M03RV7Top2000PackageError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return absolute


def _read_canonical_receipt(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    _require_sha256(f"{label}_sha256", expected_sha256)
    source = _regular_nonsymlink(path, label=label)
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise M03RV7Top2000PackageError(f"{label} SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV7Top2000PackageError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict) or raw != _canonical_json(payload) + b"\n":
        raise M03RV7Top2000PackageError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], payload), actual


def _regular_file_sha256(path: Path, *, label: str) -> str:
    source = _regular_nonsymlink(path, label=label)
    digest = hashlib.sha256()
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7Top2000ArtifactBindings:
    """Independent immutable identities; no aggregate hash substitutes for one."""

    source_archive_sha256: str
    source_manifest_sha256: str
    dependency_lock_sha256: str
    cache_artifact_sha256: str
    cache_manifest_sha256: str
    data_manifest_sha256: str
    execution_model_sha256: str
    image_reference: str
    image_digest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_archive_sha256",
            "source_manifest_sha256",
            "dependency_lock_sha256",
            "cache_artifact_sha256",
            "cache_manifest_sha256",
            "data_manifest_sha256",
            "execution_model_sha256",
            "image_digest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        match = _IMAGE_RE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise M03RV7Top2000PackageError(
                "image_reference must be digest-pinned and match image_digest_sha256"
            )


@dataclass(frozen=True, slots=True)
class M03RV7Top2000IndexPlan:
    """One Indexed-Job completion owns all thirty cells for one setting."""

    completion_index: int
    setting_index: int
    development_setting_id: str
    reviewed_v7_setting_id: str
    admission_wave: Literal[1, 2]
    fold_indices: tuple[int, ...]
    paired_seeds: tuple[int, ...]
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.completion_index < 12:
            raise M03RV7Top2000PackageError("completion index must be in [0, 11]")
        if self.setting_index != M03R_V7_ADMISSION_ORDER[self.completion_index]:
            raise M03RV7Top2000PackageError(
                "completion index must follow the frozen wave admission order"
            )
        if self.development_setting_id != M03R_TOP2000_DEV_SETTING_IDS[
            self.setting_index
        ]:
            raise M03RV7Top2000PackageError("development setting identity drifted")
        expected_reviewed = M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID[
            self.development_setting_id
        ]
        if (
            self.reviewed_v7_setting_id != expected_reviewed
            or expected_reviewed != M03R_V7_PANEL.setting_ids[self.setting_index]
        ):
            raise M03RV7Top2000PackageError(
                "TOP2000 development row no longer maps one-to-one to reviewed v7"
            )
        expected_wave = 1 if self.completion_index < 8 else 2
        if self.admission_wave != expected_wave:
            raise M03RV7Top2000PackageError("completion admission wave drifted")
        if self.fold_indices != tuple(range(M03R_V7_PANEL.fold_count)):
            raise M03RV7Top2000PackageError("each completion must own all six folds")
        if self.paired_seeds != M03R_V7_PANEL.paired_seeds:
            raise M03RV7Top2000PackageError(
                "each completion must own the same five paired seeds"
            )
        if self.promotion_eligible:
            raise M03RV7Top2000PackageError(
                "future-selected TOP2000 development rows are never promotable"
            )

    @property
    def fold_seed_cell_count(self) -> int:
        return len(self.fold_indices) * len(self.paired_seeds)


@dataclass(frozen=True, slots=True)
class M03RV7Top2000PackagePlan:
    """Plan artifact consumed by, but not generated inside, a Kubernetes Pod."""

    artifacts: M03RV7Top2000ArtifactBindings
    indices: tuple[M03RV7Top2000IndexPlan, ...]
    runtime_profile: M03RV7Top2000RuntimeProfile
    plan_artifact_path: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH
    protocol_sha256: str = M03R_TOP2000_DEV_PROTOCOL_SHA256
    protocol_generation: str = M03R_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_TOP2000_DEV_DESIGN_ID
    data_role: Literal["development-only-nonreportable"] = (
        "development-only-nonreportable"
    )
    promotion_eligible: bool = False
    outer_evaluation_authorized: bool = False

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_TOP2000_PACKAGE_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "protocol_sha256": self.protocol_sha256,
            "design_id": self.design_id,
            "data_role": self.data_role,
            "promotion_eligible": self.promotion_eligible,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "plan_artifact_path": self.plan_artifact_path,
            "source_pythonpath": self.source_pythonpath,
            "artifacts": asdict(self.artifacts),
            "indices": [asdict(value) for value in self.indices],
            "runtime_profile": asdict(self.runtime_profile),
        }

    def __post_init__(self) -> None:
        if (
            self.protocol_generation != M03R_TOP2000_DEV_PROTOCOL_GENERATION
            or self.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256
            or self.design_id != M03R_TOP2000_DEV_DESIGN_ID
        ):
            raise M03RV7Top2000PackageError("TOP2000 package protocol drifted")
        if (
            self.data_role != M03R_TOP2000_DEV_DATA_ROLE
            or self.promotion_eligible
            or self.outer_evaluation_authorized
        ):
            raise M03RV7Top2000PackageError(
                "TOP2000 compatibility work must remain development-only"
            )
        _require_absolute_container_path("plan_artifact_path", self.plan_artifact_path)
        _require_absolute_container_path("source_pythonpath", self.source_pythonpath)
        expected_source_pythonpath = str(
            PurePosixPath(self.plan_artifact_path).parent / "source" / "src"
        )
        if self.source_pythonpath != expected_source_pythonpath:
            raise M03RV7Top2000PackageError(
                "source_pythonpath must bind the staged source/src package root"
            )
        expected = tuple(_build_index_plan(i) for i in range(12))
        if self.indices != expected:
            raise M03RV7Top2000PackageError(
                "package must contain the exact twelve admission-ordered indices"
            )
        if self.package_plan_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000PackageError("package plan hash mismatch")


def _build_index_plan(completion_index: int) -> M03RV7Top2000IndexPlan:
    setting_index = M03R_V7_ADMISSION_ORDER[completion_index]
    development_setting_id = M03R_TOP2000_DEV_SETTING_IDS[setting_index]
    return M03RV7Top2000IndexPlan(
        completion_index=completion_index,
        setting_index=setting_index,
        development_setting_id=development_setting_id,
        reviewed_v7_setting_id=M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID[
            development_setting_id
        ],
        admission_wave=1 if completion_index < 8 else 2,
        fold_indices=tuple(range(M03R_V7_PANEL.fold_count)),
        paired_seeds=M03R_V7_PANEL.paired_seeds,
    )


def build_m03r_v7_top2000_package_plan(
    *,
    artifacts: M03RV7Top2000ArtifactBindings,
    plan_artifact_path: str,
    runtime_profile: M03RV7Top2000RuntimeProfile | None = None,
) -> M03RV7Top2000PackagePlan:
    """Build a deterministic, development-only plan without staging it."""

    profile = runtime_profile or M03RV7Top2000RuntimeProfile()
    fields: dict[str, Any] = {
        "artifacts": artifacts,
        "indices": tuple(_build_index_plan(index) for index in range(12)),
        "runtime_profile": profile,
        "plan_artifact_path": plan_artifact_path,
        "source_pythonpath": str(
            PurePosixPath(plan_artifact_path).parent / "source" / "src"
        ),
        "protocol_sha256": M03R_TOP2000_DEV_PROTOCOL_SHA256,
        "protocol_generation": M03R_TOP2000_DEV_PROTOCOL_GENERATION,
        "design_id": M03R_TOP2000_DEV_DESIGN_ID,
        "data_role": M03R_TOP2000_DEV_DATA_ROLE,
        "promotion_eligible": False,
        "outer_evaluation_authorized": False,
    }
    unsigned = M03RV7Top2000PackagePlan.__new__(M03RV7Top2000PackagePlan)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Top2000PackagePlan(
        **fields,
        package_plan_sha256=_sha256(unsigned.canonical_payload()),
    )


def _execution_surface_payload(
    *,
    plan: M03RV7Top2000PackagePlan,
    worker_argv_prefix: tuple[str, ...],
    worker_entrypoint_sha256: str,
    runtime_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "rl-quant.m03r-v7-top2000-execution-surface-v1",
        "protocol_sha256": plan.protocol_sha256,
        "package_plan_sha256": plan.package_plan_sha256,
        "source_archive_sha256": plan.artifacts.source_archive_sha256,
        "source_manifest_sha256": plan.artifacts.source_manifest_sha256,
        "dependency_lock_sha256": plan.artifacts.dependency_lock_sha256,
        "cache_artifact_sha256": plan.artifacts.cache_artifact_sha256,
        "cache_manifest_sha256": plan.artifacts.cache_manifest_sha256,
        "data_manifest_sha256": plan.artifacts.data_manifest_sha256,
        "execution_model_sha256": plan.artifacts.execution_model_sha256,
        "image_digest_sha256": plan.artifacts.image_digest_sha256,
        "runtime_profile": asdict(plan.runtime_profile),
        "worker_argv_prefix": list(worker_argv_prefix),
        "worker_entrypoint_sha256": worker_entrypoint_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "gpu_count_per_completion": 2,
        "torchrun_nproc_per_node": 2,
        "complete_cross_section_per_rank": True,
        "stock_axis_partitioning": False,
    }


@dataclass(frozen=True, slots=True)
class M03RV7Top2000ExecutableWorkerReceipt:
    """Qualification of the exact argv and execution surface in the image."""

    worker_argv_prefix: tuple[str, ...]
    worker_entrypoint_sha256: str
    runtime_manifest_sha256: str
    execution_surface_sha256: str
    smoke_test_receipt_sha256: str
    cuda_two_rank_parity_receipt_sha256: str
    exact_restart_receipt_sha256: str
    qualification_complete: bool
    artifact_backed_qualification: bool
    receipt_sha256: str
    protocol_sha256: str = M03R_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_TOP2000_WORKER_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "worker_argv_prefix": list(self.worker_argv_prefix),
            "worker_entrypoint_sha256": self.worker_entrypoint_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "execution_surface_sha256": self.execution_surface_sha256,
            "smoke_test_receipt_sha256": self.smoke_test_receipt_sha256,
            "cuda_two_rank_parity_receipt_sha256": (
                self.cuda_two_rank_parity_receipt_sha256
            ),
            "exact_restart_receipt_sha256": self.exact_restart_receipt_sha256,
            "qualification_complete": self.qualification_complete,
            "artifact_backed_qualification": self.artifact_backed_qualification,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_TOP2000_WORKER_RECEIPT_SCHEMA:
            raise M03RV7Top2000PackageError("worker receipt schema drifted")
        if self.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256:
            raise M03RV7Top2000PackageError("worker protocol hash drifted")
        if (
            not self.worker_argv_prefix
            or self.worker_argv_prefix[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}
            or any("\n" in value or "\x00" in value for value in self.worker_argv_prefix)
        ):
            raise M03RV7Top2000PackageError(
                "worker argv must invoke an exact executable without a shell"
            )
        for name in (
            "worker_entrypoint_sha256",
            "runtime_manifest_sha256",
            "execution_surface_sha256",
            "smoke_test_receipt_sha256",
            "cuda_two_rank_parity_receipt_sha256",
            "exact_restart_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000PackageError("worker receipt hash mismatch")

    def verify_for_plan(self, plan: M03RV7Top2000PackagePlan) -> None:
        if not self.qualification_complete or not self.artifact_backed_qualification:
            raise M03RV7Top2000PackageError(
                "worker receipt is a non-launchable caller-authored draft"
            )
        expected = _sha256(
            _execution_surface_payload(
                plan=plan,
                worker_argv_prefix=self.worker_argv_prefix,
                worker_entrypoint_sha256=self.worker_entrypoint_sha256,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
            )
        )
        if self.execution_surface_sha256 != expected:
            raise M03RV7Top2000PackageError(
                "worker receipt does not bind this source/cache/image/plan surface"
            )


def build_m03r_v7_top2000_worker_receipt(
    *,
    plan: M03RV7Top2000PackagePlan,
    worker_argv_prefix: tuple[str, ...],
    worker_entrypoint_sha256: str,
    runtime_manifest_sha256: str,
    smoke_test_receipt_sha256: str,
    cuda_two_rank_parity_receipt_sha256: str,
    exact_restart_receipt_sha256: str,
) -> M03RV7Top2000ExecutableWorkerReceipt:
    """Bind complete worker qualification to the exact package plan."""

    execution_surface_sha256 = _sha256(
        _execution_surface_payload(
            plan=plan,
            worker_argv_prefix=worker_argv_prefix,
            worker_entrypoint_sha256=worker_entrypoint_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
    )
    fields: dict[str, Any] = {
        "worker_argv_prefix": worker_argv_prefix,
        "worker_entrypoint_sha256": worker_entrypoint_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "execution_surface_sha256": execution_surface_sha256,
        "smoke_test_receipt_sha256": smoke_test_receipt_sha256,
        "cuda_two_rank_parity_receipt_sha256": cuda_two_rank_parity_receipt_sha256,
        "exact_restart_receipt_sha256": exact_restart_receipt_sha256,
        "qualification_complete": True,
        "artifact_backed_qualification": False,
        "protocol_sha256": M03R_TOP2000_DEV_PROTOCOL_SHA256,
        "schema": M03R_TOP2000_WORKER_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Top2000ExecutableWorkerReceipt.__new__(
        M03RV7Top2000ExecutableWorkerReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Top2000ExecutableWorkerReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7Top2000VerifiedQualificationArtifact:
    """Independently replayed evidence from one real four-update two-rank run."""

    completion_index: int
    setting_index: int
    setting_id: str
    qualification_receipt_sha256: str
    cell_receipt_sha256: str
    execution_plan_binding_sha256: str
    rank_model_state_sha256: tuple[str, str]
    rank_alpha_optimizer_state_sha256: tuple[str, str]
    rank_overlay_optimizer_state_sha256: tuple[str | None, str | None]
    rank_peak_allocated_bytes: tuple[int, int]
    rank_peak_reserved_bytes: tuple[int, int]
    rank_total_memory_bytes: tuple[int, int]
    rank_elapsed_seconds: tuple[float, float]
    gpu_names: tuple[str, str]
    compute_capabilities: tuple[tuple[int, int], tuple[int, int]]
    runtime_identity_sha256: str
    evidence_sha256: str
    qualification_steps: int = M03R_TOP2000_QUALIFICATION_STEPS
    schema: str = M03R_TOP2000_QUALIFICATION_EVIDENCE_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_index": self.completion_index,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "qualification_steps": self.qualification_steps,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "cell_receipt_sha256": self.cell_receipt_sha256,
            "execution_plan_binding_sha256": self.execution_plan_binding_sha256,
            "rank_model_state_sha256": list(self.rank_model_state_sha256),
            "rank_alpha_optimizer_state_sha256": list(
                self.rank_alpha_optimizer_state_sha256
            ),
            "rank_overlay_optimizer_state_sha256": list(
                self.rank_overlay_optimizer_state_sha256
            ),
            "rank_peak_allocated_bytes": list(self.rank_peak_allocated_bytes),
            "rank_peak_reserved_bytes": list(self.rank_peak_reserved_bytes),
            "rank_total_memory_bytes": list(self.rank_total_memory_bytes),
            "rank_elapsed_seconds": list(self.rank_elapsed_seconds),
            "gpu_names": list(self.gpu_names),
            "compute_capabilities": [
                list(value) for value in self.compute_capabilities
            ],
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_TOP2000_QUALIFICATION_EVIDENCE_SCHEMA:
            raise M03RV7Top2000PackageError(
                "qualification evidence schema drifted"
            )
        if not 0 <= self.completion_index < len(M03R_TOP2000_DEV_SETTING_IDS):
            raise M03RV7Top2000PackageError(
                "qualification completion index is invalid"
            )
        expected_index = M03R_V7_ADMISSION_ORDER[self.completion_index]
        if (
            self.setting_index != expected_index
            or self.setting_id != M03R_TOP2000_DEV_SETTING_IDS[expected_index]
            or self.qualification_steps != M03R_TOP2000_QUALIFICATION_STEPS
        ):
            raise M03RV7Top2000PackageError(
                "qualification setting identity or four-update geometry drifted"
            )
        for name in (
            "qualification_receipt_sha256",
            "cell_receipt_sha256",
            "execution_plan_binding_sha256",
            "runtime_identity_sha256",
            "evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for values in (
            self.rank_model_state_sha256,
            self.rank_alpha_optimizer_state_sha256,
        ):
            if len(values) != 2 or len(set(values)) != 1:
                raise M03RV7Top2000PackageError(
                    "qualification rank state or optimizer parity failed"
                )
            for value in values:
                _require_sha256("rank parity digest", value)
        if self.setting_id == "A06-sharpe-overlay-top2000-dev-v1":
            overlay_hashes = self.rank_overlay_optimizer_state_sha256
            if (
                any(value is None for value in overlay_hashes)
                or len(set(overlay_hashes)) != 1
            ):
                raise M03RV7Top2000PackageError(
                    "A06 qualification omitted overlay optimizer parity"
                )
            for overlay_value in overlay_hashes:
                if overlay_value is not None:
                    _require_sha256(
                        "A06 overlay optimizer digest", overlay_value
                    )
        elif self.rank_overlay_optimizer_state_sha256 != (None, None):
            raise M03RV7Top2000PackageError(
                "non-A06 qualification unexpectedly contains an overlay optimizer"
            )
        gib = float(1024**3)
        if not all(
            M03R_TOP2000_MIN_ALLOCATED_GIB
            <= allocated / gib
            <= M03R_TOP2000_MAX_ALLOCATED_GIB
            and allocated <= reserved <= total
            and (total - reserved) / gib
            >= M03R_TOP2000_MIN_RESERVED_HEADROOM_GIB
            for allocated, reserved, total in zip(
                self.rank_peak_allocated_bytes,
                self.rank_peak_reserved_bytes,
                self.rank_total_memory_bytes,
                strict=True,
            )
        ):
            raise M03RV7Top2000PackageError(
                "qualification must measure 60-75 GiB allocated per rank with "
                "at least 5 GiB reserved-memory headroom"
            )
        if (
            self.gpu_names != (M03R_TOP2000_GPU_NAME, M03R_TOP2000_GPU_NAME)
            or self.compute_capabilities != ((9, 0), (9, 0))
            or any(
                not 79 * 1024**3 <= value <= 81 * 1024**3
                for value in self.rank_total_memory_bytes
            )
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.rank_elapsed_seconds
            )
        ):
            raise M03RV7Top2000PackageError(
                "qualification runtime is not two actual H100 80GB ranks"
            )
        if self.evidence_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000PackageError(
                "qualification evidence hash mismatch"
            )


def _as_pair(
    value: Any,
    *,
    name: str,
    item_type: type,
) -> tuple[Any, Any]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, item_type) for item in value)
    ):
        raise M03RV7Top2000PackageError(f"{name} must contain exactly two ranks")
    return value[0], value[1]


def verify_m03r_v7_top2000_qualification_artifact(
    *,
    plan: M03RV7Top2000PackagePlan,
    completion_index: int,
    qualification_receipt_path: str | Path,
    expected_qualification_receipt_sha256: str,
) -> M03RV7Top2000VerifiedQualificationArtifact:
    """Replay one worker receipt and every referenced local artifact fail closed."""

    if not 0 <= completion_index < len(plan.indices):
        raise M03RV7Top2000PackageError("qualification completion is outside plan")
    row = plan.indices[completion_index]
    receipt_path = Path(qualification_receipt_path)
    terminal, terminal_sha = _read_canonical_receipt(
        receipt_path,
        expected_sha256=expected_qualification_receipt_sha256,
        label="qualification receipt",
    )
    required_terminal = {
        "schema": M03R_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA,
        "mode": "qualification",
        "protocol_sha256": plan.protocol_sha256,
        "cache_sha256": plan.artifacts.cache_artifact_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "world_size": 2,
        "fold_count": 1,
        "paired_seeds": [row.paired_seeds[0]],
        "completed_cells": 1,
        "optimizer_steps_per_cell": M03R_TOP2000_QUALIFICATION_STEPS,
        "intentional_restart_after_step": 1,
        "resumed_from_checkpoint": True,
        "resume_completed_steps": 1,
        "seed_validation_receipt_count": 0,
        "fold_ensemble_receipt_count": 0,
        "inference_path_count": 0,
        "output_space_ensemble_required": False,
        "development_only": True,
        "future_selected_universe": True,
        "outer_evaluation_authorized": False,
        "promotion_eligible": False,
        "complete": True,
    }
    if any(terminal.get(name) != expected for name, expected in required_terminal.items()):
        raise M03RV7Top2000PackageError(
            "qualification terminal identity or completion contract drifted"
        )

    binding_sha = terminal.get("plan_file_sha256")
    if not isinstance(binding_sha, str):
        raise M03RV7Top2000PackageError("qualification omitted execution binding")
    setting_root = receipt_path.parent.parent
    binding_path = setting_root / "execution-plan-binding.json"
    binding, actual_binding_sha = _read_canonical_receipt(
        binding_path,
        expected_sha256=binding_sha,
        label="execution plan binding",
    )
    training_plan = binding.get("training_plan")
    completion = binding.get("completion")
    if (
        binding.get("package_plan_sha256") != plan.package_plan_sha256
        or not isinstance(training_plan, dict)
        or not isinstance(completion, dict)
        or completion.get("completion_index") != completion_index
        or completion.get("setting_index") != row.setting_index
        or completion.get("development_setting_id") != row.development_setting_id
        or training_plan.get("setting_index") != row.setting_index
        or training_plan.get("setting_id") != row.development_setting_id
        or training_plan.get("cache_sha256")
        != plan.artifacts.cache_artifact_sha256
        or training_plan.get("expected_world_size") != 2
        or training_plan.get("token_dim") != plan.runtime_profile.token_dim
        or training_plan.get("max_origin_batch")
        != plan.runtime_profile.max_origin_batch
    ):
        raise M03RV7Top2000PackageError(
            "qualification execution-plan binding does not match package plan"
        )

    receipt_inventory = terminal.get("cell_receipt_sha256")
    if not isinstance(receipt_inventory, dict) or len(receipt_inventory) != 1:
        raise M03RV7Top2000PackageError(
            "qualification must bind exactly one cell receipt"
        )
    cell_name, cell_expected_sha = next(iter(receipt_inventory.items()))
    if (
        not isinstance(cell_name, str)
        or re.fullmatch(r"fold-00-seed-[0-9]+\.json", cell_name) is None
        or not isinstance(cell_expected_sha, str)
    ):
        raise M03RV7Top2000PackageError("qualification cell inventory is invalid")
    cell, cell_sha = _read_canonical_receipt(
        receipt_path.parent / "receipts" / cell_name,
        expected_sha256=cell_expected_sha,
        label="qualification cell receipt",
    )
    if (
        cell.get("mode") != "qualification"
        or cell.get("protocol_sha256") != plan.protocol_sha256
        or cell.get("plan_file_sha256") != actual_binding_sha
        or cell.get("cache_sha256") != plan.artifacts.cache_artifact_sha256
        or cell.get("setting_index") != row.setting_index
        or cell.get("setting_id") != row.development_setting_id
        or cell.get("optimizer_steps") != M03R_TOP2000_QUALIFICATION_STEPS
        or cell.get("fold_index") != 0
        or cell.get("seed") != row.paired_seeds[0]
        or cell.get("seed_validation_required") is not False
    ):
        raise M03RV7Top2000PackageError(
            "qualification cell identity or four-update evidence drifted"
        )

    model_hashes = cast(
        tuple[str, str],
        _as_pair(
            cell.get("rank_model_state_sha256"),
            name="rank model state hashes",
            item_type=str,
        ),
    )
    model_file_hashes = _as_pair(
        cell.get("rank_model_sha256"),
        name="rank model file hashes",
        item_type=str,
    )
    for rank, expected_model_sha in enumerate(model_file_hashes):
        _require_sha256("rank model file digest", expected_model_sha)
        actual_model_sha = _regular_file_sha256(
            receipt_path.parent
            / "cells"
            / f"fold-00-seed-{row.paired_seeds[0]}"
            / f"model.rank-{rank:02d}.pt",
            label="qualification rank model",
        )
        if actual_model_sha != expected_model_sha:
            raise M03RV7Top2000PackageError(
                "qualification rank model artifact SHA-256 mismatch"
            )
    last_metrics = cell.get("last_metrics")
    objective = last_metrics.get("objective") if isinstance(last_metrics, dict) else None
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
    ):
        raise M03RV7Top2000PackageError(
            "qualification cell omitted a finite measured objective"
        )
    alpha_hashes = cast(
        tuple[str, str],
        _as_pair(
            cell.get("rank_alpha_core_optimizer_state_sha256"),
            name="rank alpha optimizer hashes",
            item_type=str,
        ),
    )
    overlay_raw = cell.get("rank_overlay_optimizer_state_sha256")
    if not isinstance(overlay_raw, list) or len(overlay_raw) != 2 or any(
        value is not None and not isinstance(value, str) for value in overlay_raw
    ):
        raise M03RV7Top2000PackageError("rank overlay optimizer inventory is invalid")
    overlay_hashes = cast(tuple[str | None, str | None], tuple(overlay_raw))
    if (
        terminal.get("rank_model_state_sha256") != list(model_hashes)
        or terminal.get("rank_alpha_core_optimizer_state_sha256")
        != list(alpha_hashes)
        or terminal.get("rank_overlay_optimizer_state_sha256")
        != list(overlay_hashes)
    ):
        raise M03RV7Top2000PackageError(
            "qualification terminal does not reproduce cell rank parity"
        )

    peaks = terminal.get("rank_peak_cuda_memory")
    if not isinstance(peaks, list) or len(peaks) != 2:
        raise M03RV7Top2000PackageError(
            "qualification must contain exactly two rank memory records"
        )
    if any(not isinstance(value, dict) for value in peaks):
        raise M03RV7Top2000PackageError(
            "qualification rank memory records must be objects"
        )
    ordered: list[Mapping[str, Any]] = []
    for rank, peak in enumerate(sorted(peaks, key=lambda item: item.get("rank", -1))):
        if not isinstance(peak, dict) or peak.get("rank") != rank:
            raise M03RV7Top2000PackageError("qualification rank inventory drifted")
        if (
            peak.get("device") != f"cuda:{rank}"
            or peak.get("gpu_name") != M03R_TOP2000_GPU_NAME
            or peak.get("compute_capability") != [9, 0]
            or peak.get("allocator_oom_count") != 0
            or peak.get("allocator_retry_count") != 0
            or peak.get("torchrun_restart_count") != 1
        ):
            raise M03RV7Top2000PackageError(
                "qualification GPU identity, allocator, or launcher evidence failed"
            )
        for name in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "gpu_total_memory_bytes",
        ):
            if isinstance(peak.get(name), bool) or not isinstance(peak.get(name), int):
                raise M03RV7Top2000PackageError(
                    f"qualification {name} must be measured integer bytes"
                )
        for name in (
            "python_version",
            "torch_version",
            "torch_cuda_version",
        ):
            if not isinstance(peak.get(name), str) or not peak.get(name):
                raise M03RV7Top2000PackageError(
                    f"qualification omitted runtime {name}"
                )
        if peak.get("cudnn_version") is None or peak.get("nccl_version") is None:
            raise M03RV7Top2000PackageError(
                "qualification omitted cuDNN or NCCL runtime identity"
            )
        ordered.append(peak)
    if cell.get("rank_peak_cuda_memory") != peaks:
        raise M03RV7Top2000PackageError(
            "qualification terminal and cell memory evidence disagree"
        )
    elapsed = _as_pair(
        terminal.get("rank_elapsed_seconds"),
        name="rank elapsed seconds",
        item_type=float,
    )
    runtime_payload = {
        "schema": "rl-quant.m03r-v7-top2000-runtime-identity-v1",
        "ranks": [
            {
                name: peak[name]
                for name in (
                    "rank",
                    "device",
                    "gpu_name",
                    "gpu_total_memory_bytes",
                    "compute_capability",
                    "python_version",
                    "torch_version",
                    "torch_cuda_version",
                    "cudnn_version",
                    "nccl_version",
                )
            }
            for peak in ordered
        ],
    }
    fields: dict[str, Any] = {
        "completion_index": completion_index,
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "qualification_receipt_sha256": terminal_sha,
        "cell_receipt_sha256": cell_sha,
        "execution_plan_binding_sha256": actual_binding_sha,
        "rank_model_state_sha256": model_hashes,
        "rank_alpha_optimizer_state_sha256": alpha_hashes,
        "rank_overlay_optimizer_state_sha256": overlay_hashes,
        "rank_peak_allocated_bytes": tuple(
            cast(int, peak["peak_allocated_bytes"]) for peak in ordered
        ),
        "rank_peak_reserved_bytes": tuple(
            cast(int, peak["peak_reserved_bytes"]) for peak in ordered
        ),
        "rank_total_memory_bytes": tuple(
            cast(int, peak["gpu_total_memory_bytes"]) for peak in ordered
        ),
        "rank_elapsed_seconds": cast(tuple[float, float], elapsed),
        "gpu_names": (M03R_TOP2000_GPU_NAME, M03R_TOP2000_GPU_NAME),
        "compute_capabilities": ((9, 0), (9, 0)),
        "runtime_identity_sha256": _sha256(runtime_payload),
        "qualification_steps": M03R_TOP2000_QUALIFICATION_STEPS,
        "schema": M03R_TOP2000_QUALIFICATION_EVIDENCE_SCHEMA,
    }
    unsigned = M03RV7Top2000VerifiedQualificationArtifact.__new__(
        M03RV7Top2000VerifiedQualificationArtifact
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Top2000VerifiedQualificationArtifact(
        **fields,
        evidence_sha256=_sha256(unsigned.canonical_payload()),
    )


def _ordered_qualification_artifacts(
    values: Sequence[M03RV7Top2000VerifiedQualificationArtifact],
) -> tuple[M03RV7Top2000VerifiedQualificationArtifact, ...]:
    ordered = tuple(sorted(values, key=lambda value: value.setting_index))
    if (
        len(ordered) != len(M03R_TOP2000_DEV_SETTING_IDS)
        or tuple(value.setting_id for value in ordered)
        != M03R_TOP2000_DEV_SETTING_IDS
        or tuple(value.setting_index for value in ordered) != tuple(range(12))
    ):
        raise M03RV7Top2000PackageError(
            "qualification evidence must cover every setting exactly once"
        )
    return ordered


def _qualification_measurement_sha256(
    *,
    plan: M03RV7Top2000PackagePlan,
    qualifications: Sequence[M03RV7Top2000VerifiedQualificationArtifact],
) -> str:
    ordered = _ordered_qualification_artifacts(qualifications)
    return _sha256(
        {
            "schema": "rl-quant.m03r-v7-top2000-capacity-measurement-set-v1",
            "package_plan_sha256": plan.package_plan_sha256,
            "qualification_evidence_sha256": [
                value.evidence_sha256 for value in ordered
            ],
        }
    )


def build_m03r_v7_top2000_worker_receipt_from_qualifications(
    *,
    plan: M03RV7Top2000PackagePlan,
    qualifications: Sequence[M03RV7Top2000VerifiedQualificationArtifact],
    worker_argv_prefix: tuple[str, ...],
    worker_entrypoint_sha256: str,
    runtime_manifest_sha256: str,
) -> M03RV7Top2000ExecutableWorkerReceipt:
    """Derive runtime, parity, and intentional-restart gates from real artifacts."""

    parity_sha256 = _qualification_measurement_sha256(
        plan=plan,
        qualifications=qualifications,
    )
    execution_surface_sha256 = _sha256(
        _execution_surface_payload(
            plan=plan,
            worker_argv_prefix=worker_argv_prefix,
            worker_entrypoint_sha256=worker_entrypoint_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
    )
    fields: dict[str, Any] = {
        "worker_argv_prefix": worker_argv_prefix,
        "worker_entrypoint_sha256": worker_entrypoint_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "execution_surface_sha256": execution_surface_sha256,
        "smoke_test_receipt_sha256": parity_sha256,
        "cuda_two_rank_parity_receipt_sha256": parity_sha256,
        "exact_restart_receipt_sha256": parity_sha256,
        "qualification_complete": True,
        "artifact_backed_qualification": True,
        "protocol_sha256": M03R_TOP2000_DEV_PROTOCOL_SHA256,
        "schema": M03R_TOP2000_WORKER_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Top2000ExecutableWorkerReceipt.__new__(
        M03RV7Top2000ExecutableWorkerReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Top2000ExecutableWorkerReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7Top2000CapacityReceipt:
    """Two-H100 memory/throughput evidence for this exact execution surface."""

    execution_surface_sha256: str
    image_digest_sha256: str
    gpu_product: Literal["NVIDIA-H100-80GB-HBM3"]
    gpu_count: Literal[2]
    rank_peak_hbm_gib: tuple[float, float]
    rank_peak_reserved_hbm_gib: tuple[float, float]
    minimum_rank_reserved_headroom_gib: float
    aggregate_peak_hbm_gib: float
    qualified_setting_ids: tuple[str, ...]
    qualification_artifact_sha256s: tuple[str, ...]
    two_rank_ddp_completed: bool
    no_oom: bool
    capacity_qualified: bool
    measurement_artifact_sha256: str
    receipt_sha256: str
    schema: str = M03R_TOP2000_CAPACITY_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_surface_sha256": self.execution_surface_sha256,
            "image_digest_sha256": self.image_digest_sha256,
            "gpu_product": self.gpu_product,
            "gpu_count": self.gpu_count,
            "rank_peak_hbm_gib": list(self.rank_peak_hbm_gib),
            "rank_peak_reserved_hbm_gib": list(
                self.rank_peak_reserved_hbm_gib
            ),
            "minimum_rank_reserved_headroom_gib": (
                self.minimum_rank_reserved_headroom_gib
            ),
            "aggregate_peak_hbm_gib": self.aggregate_peak_hbm_gib,
            "qualified_setting_ids": list(self.qualified_setting_ids),
            "qualification_artifact_sha256s": list(
                self.qualification_artifact_sha256s
            ),
            "two_rank_ddp_completed": self.two_rank_ddp_completed,
            "no_oom": self.no_oom,
            "capacity_qualified": self.capacity_qualified,
            "measurement_artifact_sha256": self.measurement_artifact_sha256,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_TOP2000_CAPACITY_RECEIPT_SCHEMA:
            raise M03RV7Top2000PackageError("capacity receipt schema drifted")
        for name in (
            "execution_surface_sha256",
            "image_digest_sha256",
            "measurement_artifact_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.gpu_product != "NVIDIA-H100-80GB-HBM3" or self.gpu_count != 2:
            raise M03RV7Top2000PackageError("capacity receipt must use two H100 80GB")
        if (
            len(self.rank_peak_hbm_gib) != 2
            or len(self.rank_peak_reserved_hbm_gib) != 2
            or any(
                not M03R_TOP2000_MIN_ALLOCATED_GIB
                <= value
                <= M03R_TOP2000_MAX_ALLOCATED_GIB
                for value in self.rank_peak_hbm_gib
            )
            or any(
                reserved < allocated or reserved > 75.0
                for allocated, reserved in zip(
                    self.rank_peak_hbm_gib,
                    self.rank_peak_reserved_hbm_gib,
                    strict=True,
                )
            )
            or self.minimum_rank_reserved_headroom_gib
            < M03R_TOP2000_MIN_RESERVED_HEADROOM_GIB
            or abs(sum(self.rank_peak_hbm_gib) - self.aggregate_peak_hbm_gib) > 1e-6
            or not 120.0 <= self.aggregate_peak_hbm_gib <= 150.0
        ):
            raise M03RV7Top2000PackageError(
                "capacity qualification must demonstrate 120-150 GiB aggregate HBM"
            )
        if (
            self.qualified_setting_ids != M03R_TOP2000_DEV_SETTING_IDS
            or len(self.qualification_artifact_sha256s)
            != len(M03R_TOP2000_DEV_SETTING_IDS)
            or len(set(self.qualification_artifact_sha256s))
            != len(M03R_TOP2000_DEV_SETTING_IDS)
        ):
            raise M03RV7Top2000PackageError(
                "capacity receipt must bind actual qualification for all twelve settings"
            )
        for value in self.qualification_artifact_sha256s:
            _require_sha256("qualification artifact digest", value)
        if not (
            self.two_rank_ddp_completed and self.no_oom and self.capacity_qualified
        ):
            raise M03RV7Top2000PackageError(
                "capacity receipt must prove complete two-rank, no-OOM execution"
            )
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000PackageError("capacity receipt hash mismatch")

    def verify_for(
        self,
        *,
        plan: M03RV7Top2000PackagePlan,
        worker: M03RV7Top2000ExecutableWorkerReceipt,
    ) -> None:
        worker.verify_for_plan(plan)
        if (
            self.execution_surface_sha256 != worker.execution_surface_sha256
            or self.image_digest_sha256 != plan.artifacts.image_digest_sha256
        ):
            raise M03RV7Top2000PackageError(
                "capacity receipt belongs to a different execution surface or image"
            )


def build_m03r_v7_top2000_capacity_receipt(
    *,
    plan: M03RV7Top2000PackagePlan,
    worker: M03RV7Top2000ExecutableWorkerReceipt,
    qualifications: Sequence[M03RV7Top2000VerifiedQualificationArtifact],
) -> M03RV7Top2000CapacityReceipt:
    """Build capacity only from twelve independently replayed real artifacts."""

    worker.verify_for_plan(plan)
    ordered = _ordered_qualification_artifacts(qualifications)
    qualification_artifact_sha256s = tuple(
        value.qualification_receipt_sha256 for value in ordered
    )
    measurement_artifact_sha256 = _qualification_measurement_sha256(
        plan=plan,
        qualifications=ordered,
    )
    gib = float(1024**3)
    rank_peak_hbm_gib = tuple(
        max(value.rank_peak_allocated_bytes[rank] for value in ordered) / gib
        for rank in range(2)
    )
    rank_peak_reserved_hbm_gib = tuple(
        max(value.rank_peak_reserved_bytes[rank] for value in ordered) / gib
        for rank in range(2)
    )
    minimum_headroom = min(
        (total - reserved) / gib
        for value in ordered
        for reserved, total in zip(
            value.rank_peak_reserved_bytes,
            value.rank_total_memory_bytes,
            strict=True,
        )
    )
    fields: dict[str, Any] = {
        "execution_surface_sha256": worker.execution_surface_sha256,
        "image_digest_sha256": plan.artifacts.image_digest_sha256,
        "gpu_product": M03R_V7_H100_TOPOLOGY.gpu_product,
        "gpu_count": 2,
        "rank_peak_hbm_gib": rank_peak_hbm_gib,
        "rank_peak_reserved_hbm_gib": rank_peak_reserved_hbm_gib,
        "minimum_rank_reserved_headroom_gib": minimum_headroom,
        "aggregate_peak_hbm_gib": sum(rank_peak_hbm_gib),
        "qualified_setting_ids": M03R_TOP2000_DEV_SETTING_IDS,
        "qualification_artifact_sha256s": qualification_artifact_sha256s,
        "two_rank_ddp_completed": True,
        "no_oom": True,
        "capacity_qualified": True,
        "measurement_artifact_sha256": measurement_artifact_sha256,
        "schema": M03R_TOP2000_CAPACITY_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Top2000CapacityReceipt.__new__(
        M03RV7Top2000CapacityReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Top2000CapacityReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7Top2000QualifiedPackage:
    """Pair optional receipts with a plan; absence is an explicit blocker."""

    plan: M03RV7Top2000PackagePlan
    worker_receipt: M03RV7Top2000ExecutableWorkerReceipt | None = None
    capacity_receipt: M03RV7Top2000CapacityReceipt | None = None

    def __post_init__(self) -> None:
        if self.worker_receipt is not None:
            self.worker_receipt.verify_for_plan(self.plan)
        if self.capacity_receipt is not None:
            if self.worker_receipt is None:
                raise M03RV7Top2000PackageError(
                    "capacity evidence cannot exist without its worker receipt"
                )
            if (
                not self.worker_receipt.qualification_complete
                or not self.worker_receipt.artifact_backed_qualification
            ):
                raise M03RV7Top2000PackageError(
                    "capacity cannot pair with caller-authored worker qualification"
                )
            self.capacity_receipt.verify_for(
                plan=self.plan,
                worker=self.worker_receipt,
            )
            if (
                self.worker_receipt.cuda_two_rank_parity_receipt_sha256
                != self.capacity_receipt.measurement_artifact_sha256
            ):
                raise M03RV7Top2000PackageError(
                    "worker parity and all-setting capacity evidence disagree"
                )

    @property
    def launch_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.worker_receipt is None:
            blockers.append("executable-worker-qualification-receipt-missing")
        elif (
            not self.worker_receipt.qualification_complete
            or not self.worker_receipt.artifact_backed_qualification
        ):
            blockers.append("artifact-backed-worker-qualification-receipt-missing")
        if self.capacity_receipt is None:
            blockers.append("matching-two-h100-capacity-receipt-missing")
        return tuple(blockers)

    @property
    def launch_ready(self) -> bool:
        return not self.launch_blockers

    def require_launch_ready(self) -> None:
        if self.launch_blockers:
            raise M03RV7Top2000PackageError(
                "TOP2000 Indexed Job remains blocked: "
                + ", ".join(self.launch_blockers)
            )


__all__ = [
    "M03R_TOP2000_CAPACITY_RECEIPT_SCHEMA",
    "M03R_TOP2000_PACKAGE_SCHEMA",
    "M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH",
    "M03R_TOP2000_WORKER_RECEIPT_SCHEMA",
    "M03RV7Top2000ArtifactBindings",
    "M03RV7Top2000CapacityReceipt",
    "M03RV7Top2000ExecutableWorkerReceipt",
    "M03RV7Top2000IndexPlan",
    "M03RV7Top2000PackageError",
    "M03RV7Top2000PackagePlan",
    "M03RV7Top2000QualifiedPackage",
    "M03RV7Top2000RuntimeProfile",
    "M03RV7Top2000VerifiedQualificationArtifact",
    "build_m03r_v7_top2000_capacity_receipt",
    "build_m03r_v7_top2000_package_plan",
    "build_m03r_v7_top2000_worker_receipt_from_qualifications",
    "verify_m03r_v7_top2000_qualification_artifact",
]
