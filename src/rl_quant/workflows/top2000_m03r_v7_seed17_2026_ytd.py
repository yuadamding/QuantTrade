"""Governed retrospective 2026-YTD workflow for the completed seed-17 panel.

The workflow is deliberately split at the outcome-access boundary.  The
``freeze-plan`` command reads only immutable training/checkpoint lineage and
TOP2000 namespace metadata.  It must complete before a later command may open
2026 bars or factor-return archives.  This module never mutates the source
training run and never upgrades its future-selected, one-seed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
    Top2000M03RV72026FactorData,
    Top2000M03RV72026OfficialFactorRetrieval,
    build_top2000_m03r_v7_2026_factor_data,
    load_top2000_m03r_v7_2026_official_factor_retrieval,
    retrieve_top2000_m03r_v7_2026_official_factor_archives,
    write_top2000_m03r_v7_2026_factor_data,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveData,
    load_top2000_m03r_v7_2026_retrospective_cache,
    materialize_top2000_m03r_v7_2026_retrospective_cache,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
    M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
)
from rl_quant.training import hold30_top2000_development as training_data
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17PackagePlan,
    load_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.workflows.top2000_m03r_v7_dev import (
    CELL_RECEIPT_SCHEMA,
)

M03R_SEED17_TOP2000_2026_YTD_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-frozen-plan-v1"
)
M03R_SEED17_TOP2000_2026_YTD_CHECKPOINT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-checkpoint-binding-v1"
)
M03R_SEED17_TOP2000_2026_YTD_CACHE_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-cache-receipt-v1"
)
M03R_SEED17_TOP2000_2026_YTD_FACTOR_STAGE_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-factor-stage-receipt-v1"
)
M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID = (
    "qt-m03r-v7-t2k12-s17-20260808-a05q2"
)
M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES = (
    "src/rl_quant/protocol/hold30_alpha_m03r_v7_seed17_top2000_2026_ytd.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_retrospective_data.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_trace_telemetry.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_cohort_survival.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_checkpoint.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_execution.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_execution_view.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_factor_calibration.py",
    "src/rl_quant/evaluation/top2000_m03r_v7_2026_factor_data.py",
    "src/rl_quant/training/top2000_m03r_v7_dev.py",
    "src/rl_quant/workflows/top2000_m03r_v7_seed17_2026_execution.py",
    "src/rl_quant/workflows/top2000_m03r_v7_seed17_2026_ytd.py",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Top2000M03RV7Seed172026YTDWorkflowError(RuntimeError):
    """A frozen retrospective input, identity, or access boundary drifted."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "2026-YTD workflow payload is not canonical-JSON safe"
        ) from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _cache_axis_sha256(values: Sequence[str]) -> str:
    encoded = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} cannot be read as JSON"
        ) from exc
    if not isinstance(value, dict):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} must contain a JSON object"
        )
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} fields differ from the frozen schema"
        )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = _canonical_json_bytes(dict(payload))
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == raw:
            return hashlib.sha256(raw).hexdigest()
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"refusing to overwrite immutable receipt {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(raw)
        sink.flush()
    return hashlib.sha256(raw).hexdigest()


def _relative_regular(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} path is not a safe relative path"
        )
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} is absent or not a regular file: {path}"
        )
    return path


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDCheckpointBinding:
    """One exact final checkpoint frozen before any 2026 outcome access."""

    completion_index: int
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    training_fold_index: int
    seed: int
    writer_rank: int
    optimizer_steps: int
    checkpoint_role: str
    training_root_relative_path: str
    model_relative_path: str
    model_file_sha256: str
    model_state_sha256: str
    cell_receipt_relative_path: str
    cell_receipt_file_sha256: str
    seed_validation_receipt_relative_path: str
    seed_validation_receipt_file_sha256: str
    fold_execution_receipt_relative_path: str
    fold_execution_receipt_file_sha256: str
    completion_receipt_relative_path: str
    completion_receipt_file_sha256: str
    training_plan_file_sha256: str
    training_plan_receipt_sha256: str
    source_protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    schema: str = M03R_SEED17_TOP2000_2026_YTD_CHECKPOINT_SCHEMA
    development_only: bool = True
    future_selected_universe: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
        if not 0 <= self.setting_index < len(contract.settings):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "checkpoint setting index is outside the frozen panel"
            )
        setting = contract.settings[self.setting_index]
        expected_role = (
            "headline"
            if self.training_fold_index
            == contract.checkpoint_rule.headline_training_fold_index
            else "cutoff-sensitivity"
        )
        if (
            self.schema != M03R_SEED17_TOP2000_2026_YTD_CHECKPOINT_SCHEMA
            or not 0 <= self.completion_index < 12
            or self.setting_id != setting.seed17_setting_id
            or self.runtime_setting_id != setting.runtime_setting_id
            or self.training_fold_index not in range(6)
            or self.seed != contract.checkpoint_rule.seed
            or self.writer_rank != contract.checkpoint_rule.checkpoint_writer_rank
            or self.optimizer_steps != 64
            or self.checkpoint_role != expected_role
            or self.source_protocol_sha256 != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or not self.training_root_relative_path
            or not self.model_relative_path
            or not self.cell_receipt_relative_path
            or not self.seed_validation_receipt_relative_path
            or not self.fold_execution_receipt_relative_path
            or not self.completion_receipt_relative_path
            or not self.development_only
            or not self.future_selected_universe
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "checkpoint binding drifted from the frozen fold/seed/rank rule"
            )
        for name in (
            "model_file_sha256",
            "model_state_sha256",
            "cell_receipt_file_sha256",
            "seed_validation_receipt_file_sha256",
            "fold_execution_receipt_file_sha256",
            "completion_receipt_file_sha256",
            "training_plan_file_sha256",
            "training_plan_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    @property
    def receipt_sha256(self) -> str:
        return _content_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDSourceFile:
    relative_path: str
    file_sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.relative_path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.suffix != ".py"
            or pure.parts[:2] != ("src", "rl_quant")
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "evaluation source file path is outside the frozen inventory"
            )
        _require_sha256("evaluation source file SHA-256", self.file_sha256)


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDSourceBinding:
    source_root: str
    files: tuple[Top2000M03RV7Seed172026YTDSourceFile, ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        paths = tuple(row.relative_path for row in self.files)
        if (
            not self.source_root
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or not set(M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES).issubset(paths)
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "evaluation source inventory is incomplete or out of order"
            )
        expected = _content_sha256(
            [
                {"relative_path": row.relative_path, "file_sha256": row.file_sha256}
                for row in self.files
            ]
        )
        if self.inventory_sha256 != expected:
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "evaluation source inventory digest drifted"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDDataNamespace:
    """Metadata-only raw namespace binding made before opening 2026 bars."""

    dataset_root: str
    manifest_relative_path: str
    manifest_file_sha256: str
    universe_relative_path: str
    universe_file_sha256: str
    manifest_schema: str
    coverage: str
    last_window: str
    action_count: int
    action_hash: str
    stock_count: int
    cash_index: int
    universe_selection_date: str
    universe_selection_method: str
    membership_mode: str
    dataset_reportable: bool
    partition_contents_opened: bool = False
    future_selected_universe: bool = True
    point_in_time_membership_available: bool = False

    def __post_init__(self) -> None:
        contamination = (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.contamination
        )
        if (
            not self.dataset_root
            or self.manifest_relative_path != "manifest.json"
            or self.universe_relative_path != "universe.json"
            or self.manifest_schema != contamination.source_manifest_schema
            or self.coverage != "2022-01-03 -> 2026-06-23"
            or self.last_window != "2026-06-18_to_2026-06-24"
            or self.action_count != 1999
            or self.stock_count != 1998
            or self.cash_index != 0
            or self.universe_selection_date
            != contamination.universe_selection_date
            or self.universe_selection_method
            != contamination.universe_selection_method
            or self.membership_mode != contamination.membership_mode
            or self.dataset_reportable
            or self.partition_contents_opened
            or not self.future_selected_universe
            or self.point_in_time_membership_available
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "TOP2000 pre-access namespace binding is incomplete or overclaims PIT"
            )
        _require_sha256("manifest_file_sha256", self.manifest_file_sha256)
        _require_sha256("universe_file_sha256", self.universe_file_sha256)
        _require_sha256("action_hash", self.action_hash)


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDPre2026CacheBinding:
    """Exact completed-run cache lineage; this contains no 2026 outcomes."""

    cache_path: str
    cache_file_sha256: str
    cache_identity: str
    base_dataset_identity: str
    search_identity: str
    lockbox_partition_names_hash: str
    action_hash: str
    first_exchange_date: str
    last_exchange_date: str
    state_rows: int
    action_count: int
    contains_2026_or_later_exchange_date: bool = False
    development_only: bool = True
    bars_only: bool = True

    def __post_init__(self) -> None:
        if (
            not self.cache_path
            or self.first_exchange_date != "2022-01-03"
            or self.last_exchange_date != "2025-12-29"
            or self.state_rows != 1001
            or self.action_count != 1999
            or self.contains_2026_or_later_exchange_date
            or not self.development_only
            or not self.bars_only
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "pre-2026 cache binding drifted or contains outcome dates"
            )
        for name in (
            "cache_file_sha256",
            "cache_identity",
            "base_dataset_identity",
            "search_identity",
            "lockbox_partition_names_hash",
            "action_hash",
        ):
            _require_sha256(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed172026YTDFrozenPlan:
    """Complete pre-access plan for all twelve settings and six checkpoints."""

    source_run_id: str
    source_training_output_root: str
    source_package_plan_path: str
    source_package_plan_sha256: str
    source_package_plan_file_sha256: str
    completion_coverage_receipt_path: str
    completion_coverage_receipt_file_sha256: str
    completion_worker_runtime_proof_sha256: str
    evaluation_source: Top2000M03RV7Seed172026YTDSourceBinding
    evaluation_source_sha256: str
    data_namespace: Top2000M03RV7Seed172026YTDDataNamespace
    pre2026_cache: Top2000M03RV7Seed172026YTDPre2026CacheBinding
    checkpoints: tuple[Top2000M03RV7Seed172026YTDCheckpointBinding, ...]
    evaluation_contract_sha256: str = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
    )
    checkpoint_count: int = 72
    setting_count: int = 12
    folds_per_setting: int = 6
    outcome_partition_contents_opened_while_freezing: bool = False
    factor_archives_opened_while_freezing: bool = False
    training_artifacts_mutated: bool = False
    development_only: bool = True
    retrospective_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_SEED17_TOP2000_2026_YTD_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_2026_YTD_PLAN_SCHEMA
            or self.source_run_id != M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID
            or not self.source_training_output_root
            or not self.source_package_plan_path
            or self.evaluation_source_sha256
            != self.evaluation_source.inventory_sha256
            or self.evaluation_contract_sha256
            != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
            or self.checkpoint_count != 72
            or self.setting_count != 12
            or self.folds_per_setting != 6
            or len(self.checkpoints) != 72
            or self.outcome_partition_contents_opened_while_freezing
            or self.factor_archives_opened_while_freezing
            or self.training_artifacts_mutated
            or not self.development_only
            or not self.retrospective_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "frozen plan must remain a 12x6 pre-access retrospective plan"
            )
        for name in (
            "source_package_plan_sha256",
            "source_package_plan_file_sha256",
            "completion_coverage_receipt_file_sha256",
            "completion_worker_runtime_proof_sha256",
            "evaluation_source_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        ordered = tuple(
            (row.setting_index, row.training_fold_index) for row in self.checkpoints
        )
        if ordered != tuple(
            (setting_index, fold_index)
            for setting_index in range(12)
            for fold_index in range(6)
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "checkpoint inventory must be setting-major then fold-major"
            )
        if len({row.model_file_sha256 for row in self.checkpoints}) != 72:
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "every setting/fold checkpoint must have a distinct model artifact"
            )

    @property
    def receipt_sha256(self) -> str:
        return _content_sha256(asdict(self))


def load_top2000_m03r_v7_seed17_2026_ytd_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> Top2000M03RV7Seed172026YTDFrozenPlan:
    """Load an exact frozen plan before any outcome-bearing operation."""

    _require_sha256("expected_file_sha256", expected_file_sha256)
    _require_sha256("expected_receipt_sha256", expected_receipt_sha256)
    plan_path = Path(path)
    if _file_sha256(plan_path) != expected_file_sha256:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen plan file SHA-256 drifted"
        )
    payload = _read_json(plan_path, label="frozen 2026-YTD plan")
    _require_exact_keys(
        payload,
        {field.name for field in fields(Top2000M03RV7Seed172026YTDFrozenPlan)},
        label="frozen plan",
    )
    data_payload = payload.get("data_namespace")
    source_payload = payload.get("evaluation_source")
    cache_payload = payload.get("pre2026_cache")
    checkpoint_payloads = payload.get("checkpoints")
    if (
        not isinstance(data_payload, dict)
        or not isinstance(source_payload, dict)
        or not isinstance(cache_payload, dict)
        or not isinstance(checkpoint_payloads, list)
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen plan nested payloads are malformed"
        )
    _require_exact_keys(
        data_payload,
        {field.name for field in fields(Top2000M03RV7Seed172026YTDDataNamespace)},
        label="frozen data namespace",
    )
    _require_exact_keys(
        source_payload,
        {field.name for field in fields(Top2000M03RV7Seed172026YTDSourceBinding)},
        label="frozen evaluation source",
    )
    source_files = source_payload.get("files")
    if not isinstance(source_files, list) or any(
        not isinstance(row, dict)
        or set(row)
        != {field.name for field in fields(Top2000M03RV7Seed172026YTDSourceFile)}
        for row in source_files
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen evaluation source file inventory is malformed"
        )
    _require_exact_keys(
        cache_payload,
        {
            field.name
            for field in fields(Top2000M03RV7Seed172026YTDPre2026CacheBinding)
        },
        label="frozen pre-2026 cache",
    )
    checkpoint_keys = {
        field.name
        for field in fields(Top2000M03RV7Seed172026YTDCheckpointBinding)
    }
    if any(
        not isinstance(row, dict) or set(row) != checkpoint_keys
        for row in checkpoint_payloads
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen checkpoint fields differ from the schema"
        )
    top_level = dict(payload)
    source_top = dict(source_payload)
    source_top["files"] = tuple(
        Top2000M03RV7Seed172026YTDSourceFile(**row) for row in source_files
    )
    top_level["evaluation_source"] = Top2000M03RV7Seed172026YTDSourceBinding(
        **source_top
    )
    top_level["data_namespace"] = Top2000M03RV7Seed172026YTDDataNamespace(
        **data_payload
    )
    top_level["pre2026_cache"] = Top2000M03RV7Seed172026YTDPre2026CacheBinding(
        **cache_payload
    )
    top_level["checkpoints"] = tuple(
        Top2000M03RV7Seed172026YTDCheckpointBinding(**row)
        for row in checkpoint_payloads
    )
    try:
        plan = Top2000M03RV7Seed172026YTDFrozenPlan(**top_level)
    except TypeError as exc:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen plan cannot be reconstructed"
        ) from exc
    if plan.receipt_sha256 != expected_receipt_sha256:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "frozen plan semantic receipt SHA-256 drifted"
        )
    _validate_evaluation_source_files(plan.evaluation_source)
    return plan


def materialize_top2000_m03r_v7_seed17_2026_ytd_cache(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    expected_plan_receipt_sha256: str,
    output_cache_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, Any]:
    """Open 2026 bars once under an already-frozen plan and bind the cache."""

    plan = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_path,
        expected_file_sha256=expected_plan_file_sha256,
        expected_receipt_sha256=expected_plan_receipt_sha256,
    )
    result = materialize_top2000_m03r_v7_2026_retrospective_cache(
        plan.data_namespace.dataset_root,
        plan.pre2026_cache.cache_path,
        output_cache_path,
        expected_pre2026_cache_sha256=plan.pre2026_cache.cache_file_sha256,
        expected_base_dataset_identity=plan.pre2026_cache.base_dataset_identity,
        expected_search_identity=plan.pre2026_cache.search_identity,
        expected_lockbox_partition_names_hash=(
            plan.pre2026_cache.lockbox_partition_names_hash
        ),
        training_completion_receipt_sha256=(
            plan.completion_coverage_receipt_file_sha256
        ),
        evaluation_contract_sha256=plan.evaluation_contract_sha256,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    receipt = {
        "schema": M03R_SEED17_TOP2000_2026_YTD_CACHE_RECEIPT_SCHEMA,
        "frozen_plan_path": str(Path(plan_path)),
        "frozen_plan_file_sha256": expected_plan_file_sha256,
        "frozen_plan_receipt_sha256": expected_plan_receipt_sha256,
        "cache_path": str(Path(output_cache_path)),
        **result,
        "outcome_partition_contents_opened": True,
        "training_artifacts_mutated": False,
        "development_only": True,
        "retrospective_only": True,
        "reportable": False,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
    }
    _write_immutable_json(Path(output_receipt_path), receipt)
    return receipt


def _load_verified_2026_cache_stage(
    *,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    plan: Top2000M03RV7Seed172026YTDFrozenPlan,
) -> tuple[dict[str, Any], Top2000M03RV72026RetrospectiveData]:
    _require_sha256(
        "expected_cache_receipt_file_sha256", expected_receipt_file_sha256
    )
    path = Path(receipt_path)
    if _file_sha256(path) != expected_receipt_file_sha256:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "2026 cache-stage receipt file SHA-256 drifted"
        )
    receipt = _read_json(path, label="2026 cache-stage receipt")
    expected_keys = {
        "schema",
        "frozen_plan_path",
        "frozen_plan_file_sha256",
        "frozen_plan_receipt_sha256",
        "cache_path",
        "cache_sha256",
        "data_receipt_sha256",
        "source_evidence_sha256",
        "exchange_date_range",
        "score_return_date_range",
        "state_rows",
        "score_transition_rows",
        "action_count",
        "outcome_partition_contents_opened",
        "training_artifacts_mutated",
        "development_only",
        "retrospective_only",
        "reportable",
        "scientific_reporting_eligible",
        "promotion_eligible",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema")
        != M03R_SEED17_TOP2000_2026_YTD_CACHE_RECEIPT_SCHEMA
        or receipt.get("frozen_plan_receipt_sha256") != plan.receipt_sha256
        or receipt.get("action_count") != plan.data_namespace.action_count
        or receipt.get("outcome_partition_contents_opened") is not True
        or receipt.get("training_artifacts_mutated") is not False
        or receipt.get("development_only") is not True
        or receipt.get("retrospective_only") is not True
        or receipt.get("reportable") is not False
        or receipt.get("scientific_reporting_eligible") is not False
        or receipt.get("promotion_eligible") is not False
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "2026 cache-stage receipt semantics drifted"
        )
    cache_sha256 = _require_sha256("2026 cache SHA-256", receipt.get("cache_sha256"))
    data = load_top2000_m03r_v7_2026_retrospective_cache(
        str(receipt.get("cache_path")),
        expected_cache_sha256=cache_sha256,
        output_device="cpu",
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    if (
        data.identity.receipt_sha256 != receipt.get("data_receipt_sha256")
        or data.source_evidence.receipt_sha256
        != receipt.get("source_evidence_sha256")
        or list(receipt.get("score_return_date_range", ()))
        != [data.score_return_dates[0], data.score_return_dates[-1]]
        or data.identity.score_transition_rows != receipt.get("score_transition_rows")
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "2026 cache-stage receipt does not reproduce its arrays"
        )
    return receipt, data


def retrieve_top2000_m03r_v7_seed17_2026_ytd_factors(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    expected_plan_receipt_sha256: str,
    output_directory: str | Path,
    output_retrieval_receipt_path: str | Path,
) -> tuple[Top2000M03RV72026OfficialFactorRetrieval, str]:
    """Retrieve the two official archives only after replaying the frozen plan."""

    plan = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_path,
        expected_file_sha256=expected_plan_file_sha256,
        expected_receipt_sha256=expected_plan_receipt_sha256,
    )
    evidence, receipt_file_sha256 = (
        retrieve_top2000_m03r_v7_2026_official_factor_archives(
            output_directory=output_directory,
            output_receipt_path=output_retrieval_receipt_path,
            frozen_plan_file_sha256=expected_plan_file_sha256,
            frozen_plan_receipt_sha256=plan.receipt_sha256,
        )
    )
    if (
        evidence.frozen_plan_file_sha256 != expected_plan_file_sha256
        or evidence.frozen_plan_receipt_sha256 != plan.receipt_sha256
        or not evidence.official_source_verified
        or evidence.caller_staged_archives
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "official factor retrieval does not bind the exact frozen plan"
        )
    return evidence, receipt_file_sha256


def materialize_top2000_m03r_v7_seed17_2026_ytd_factors(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    expected_plan_receipt_sha256: str,
    cache_receipt_path: str | Path,
    expected_cache_receipt_file_sha256: str,
    retrieval_receipt_path: str | Path,
    expected_retrieval_receipt_file_sha256: str,
    output_factor_data_path: str | Path,
    output_receipt_path: str | Path,
) -> tuple[dict[str, Any], Top2000M03RV72026FactorData]:
    """Bind verified official factors after plan and chronology are frozen."""

    plan = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_path,
        expected_file_sha256=expected_plan_file_sha256,
        expected_receipt_sha256=expected_plan_receipt_sha256,
    )
    cache_receipt, data = _load_verified_2026_cache_stage(
        receipt_path=cache_receipt_path,
        expected_receipt_file_sha256=expected_cache_receipt_file_sha256,
        plan=plan,
    )
    retrieval = load_top2000_m03r_v7_2026_official_factor_retrieval(
        retrieval_receipt_path,
        expected_file_sha256=expected_retrieval_receipt_file_sha256,
    )
    if (
        retrieval.frozen_plan_file_sha256 != expected_plan_file_sha256
        or retrieval.frozen_plan_receipt_sha256 != plan.receipt_sha256
        or not retrieval.official_source_verified
        or retrieval.caller_staged_archives
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "factor retrieval evidence does not bind the current frozen plan"
        )
    factors = build_top2000_m03r_v7_2026_factor_data(
        retrieval_evidence=retrieval,
        score_dates=data.score_return_dates,
    )
    factor_file_sha256 = write_top2000_m03r_v7_2026_factor_data(
        factors,
        output_factor_data_path,
    )
    receipt = {
        "schema": M03R_SEED17_TOP2000_2026_YTD_FACTOR_STAGE_RECEIPT_SCHEMA,
        "frozen_plan_file_sha256": expected_plan_file_sha256,
        "frozen_plan_receipt_sha256": expected_plan_receipt_sha256,
        "cache_receipt_file_sha256": expected_cache_receipt_file_sha256,
        "cache_sha256": cache_receipt["cache_sha256"],
        "chronology_receipt_sha256": data.identity.receipt_sha256,
        "retrieval_receipt_path": str(Path(retrieval_receipt_path)),
        "retrieval_receipt_file_sha256": (
            expected_retrieval_receipt_file_sha256
        ),
        "retrieval_receipt_sha256": retrieval.receipt_sha256,
        "retrieval_method": retrieval.retrieval_method,
        "official_source_verified": True,
        "caller_staged_archives": False,
        "factor_data_path": str(Path(output_factor_data_path)),
        "factor_data_file_sha256": factor_file_sha256,
        "factor_data_receipt_sha256": factors.receipt_sha256,
        "factor_manifest_sha256": factors.manifest.manifest_sha256,
        "source_receipt_sha256": factors.manifest.source_receipt_sha256,
        "coverage_receipt_sha256": factors.manifest.coverage_receipt_sha256,
        "exact_array_receipt_sha256": factors.manifest.exact_array_receipt_sha256,
        "factor_archives_opened": True,
        "extraction_rule": "exact-frozen-score-dates-only",
        "source_containers_may_include_unused_post_end_rows": True,
        "post_end_source_rows_used": 0,
        "score_window_shortened": False,
        "imputed_value_count": 0,
        "development_only": True,
        "retrospective_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
    }
    _write_immutable_json(Path(output_receipt_path), receipt)
    return receipt, factors


def _require_training_receipt(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name, value in expected.items():
        if payload.get(name) != value:
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                f"{label} field {name!r} drifted"
            )


def _validated_h100_rank_proof(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} must contain exactly two H100 rank proofs"
        )
    if any(not isinstance(row, dict) for row in value):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} rank proof must be a mapping"
        )
    if any(
        isinstance(row.get("rank"), bool) or not isinstance(row.get("rank"), int)
        for row in value
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            f"{label} rank identifiers are invalid"
        )
    rows = sorted(value, key=lambda row: row["rank"])
    for rank, row in enumerate(rows):
        memory = row.get("gpu_total_memory_bytes")
        if (
            row.get("rank") != rank
            or row.get("device") != f"cuda:{rank}"
            or row.get("gpu_name") != "NVIDIA H100 80GB HBM3"
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or not 79 * 1024**3 <= memory <= 81 * 1024**3
            or row.get("compute_capability") != [9, 0]
            or row.get("allocator_oom_count") != 0
            or row.get("torchrun_restart_count") not in {0, 1}
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                f"{label} did not prove two qualified H100 ranks"
            )
    return [dict(row) for row in rows]


def _validate_completion_coverage(
    path: Path,
    *,
    package: M03RV7Seed17PackagePlan,
) -> tuple[str, dict[str, str], dict[int, dict[str, Any]]]:
    payload = _read_json(path, label="completion coverage receipt")
    if (
        payload.get("schema")
        != "rl-quant.top2000-m03r-v7-one-seed-coverage-v1"
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("source_archive_sha256")
        != package.artifacts.source_archive_sha256
        or payload.get("expected_seed") != 17
        or payload.get("expected_fold_count") != 6
        or payload.get("completion_count") != 12
        or not isinstance(payload.get("receipt_sha256"), dict)
        or len(payload["receipt_sha256"]) != 12
        or not isinstance(payload.get("worker_runtime_proof"), dict)
        or payload.get("development_only") is not True
        or payload.get("promotion_eligible") is not False
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage does not prove the exact 12x6 seed-17 panel"
        )
    coverage = payload.get("coverage_sha256")
    if not isinstance(coverage, str):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage omitted its content digest"
        )
    unsigned = dict(payload)
    unsigned.pop("coverage_sha256", None)
    if coverage != _content_sha256(unsigned):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage content digest drifted"
        )
    inventory = payload["receipt_sha256"]
    if any(
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        for relative, digest in inventory.items()
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage contains an invalid path/hash inventory"
        )
    expected_rows = {row.completion_index: row for row in package.indices}
    runtime_payload = payload["worker_runtime_proof"]
    if set(runtime_payload) != {str(value) for value in expected_rows}:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage runtime proof does not cover all twelve workers"
        )
    runtime_proof: dict[int, dict[str, Any]] = {}
    for completion_index, expected in expected_rows.items():
        row = runtime_payload[str(completion_index)]
        if (
            not isinstance(row, dict)
            or row.get("setting_index") != expected.setting_index
            or row.get("setting_id") != expected.setting_id
            or not isinstance(row.get("pod_name"), str)
            or not row["pod_name"]
            or not isinstance(row.get("pod_uid"), str)
            or not row["pod_uid"]
        ):
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                "completion coverage runtime worker identity drifted"
            )
        ranks = _validated_h100_rank_proof(
            row.get("rank_runtime"),
            label=f"completion {completion_index} runtime",
        )
        runtime_proof[completion_index] = {**row, "rank_runtime": ranks}
    return _file_sha256(path), dict(inventory), runtime_proof


def _evaluation_source_binding(
    source_root: Path,
) -> Top2000M03RV7Seed172026YTDSourceBinding:
    source_directory = source_root / "src" / "rl_quant"
    if not source_directory.is_dir() or source_directory.is_symlink():
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "evaluation source root omitted src/rl_quant"
        )
    relative_files = tuple(
        sorted(
            str(path.relative_to(source_root))
            for path in source_directory.rglob("*.py")
            if path.is_file()
        )
    )
    rows = tuple(
        Top2000M03RV7Seed172026YTDSourceFile(
            relative_path=relative,
            file_sha256=_file_sha256(
                _relative_regular(
                    source_root,
                    relative,
                    label="evaluation source file",
                )
            ),
        )
        for relative in relative_files
    )
    return Top2000M03RV7Seed172026YTDSourceBinding(
        source_root=str(source_root),
        files=rows,
        inventory_sha256=_content_sha256(
            [
                {"relative_path": row.relative_path, "file_sha256": row.file_sha256}
                for row in rows
            ]
        ),
    )


def _validate_evaluation_source_files(
    binding: Top2000M03RV7Seed172026YTDSourceBinding,
) -> None:
    root = Path(binding.source_root)
    for row in binding.files:
        path = _relative_regular(
            root,
            row.relative_path,
            label="bound evaluation source file",
        )
        if _file_sha256(path) != row.file_sha256:
            raise Top2000M03RV7Seed172026YTDWorkflowError(
                f"bound evaluation source file drifted: {row.relative_path}"
            )


def _metadata_namespace(dataset_root: Path) -> Top2000M03RV7Seed172026YTDDataNamespace:
    manifest_path = dataset_root / "manifest.json"
    universe_path = dataset_root / "universe.json"
    manifest = _read_json(manifest_path, label="TOP2000 manifest metadata")
    universe = _read_json(universe_path, label="TOP2000 universe metadata")
    manifest_universe = manifest.get("universe")
    if not isinstance(manifest_universe, dict):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "TOP2000 manifest omitted universe geometry"
        )
    actions = universe.get("actions")
    if (
        not isinstance(actions, list)
        or any(not isinstance(value, str) for value in actions)
        or len(actions) != int(universe.get("action_count", -1))
        or len(set(actions)) != len(actions)
        or not actions
        or actions[0] != "CASH"
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "TOP2000 universe must contain the complete unique CASH-first action axis"
        )
    reportability_errors = manifest.get("reportability_errors")
    if (
        not isinstance(reportability_errors, list)
        or "static future-selected universe omits point-in-time membership and delisting history"
        not in reportability_errors
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "TOP2000 manifest no longer discloses its PIT/reportability defect"
        )
    return Top2000M03RV7Seed172026YTDDataNamespace(
        dataset_root=str(dataset_root),
        manifest_relative_path="manifest.json",
        manifest_file_sha256=_file_sha256(manifest_path),
        universe_relative_path="universe.json",
        universe_file_sha256=_file_sha256(universe_path),
        manifest_schema=str(manifest.get("schema_version")),
        coverage=str(manifest.get("coverage")),
        last_window=str(manifest.get("last_window")),
        action_count=int(universe.get("action_count", -1)),
        action_hash=_cache_axis_sha256(actions),
        stock_count=int(manifest_universe.get("stocks", -1)),
        cash_index=int(universe.get("cash_index", -1)),
        universe_selection_date=str(manifest.get("universe_selection_date")),
        universe_selection_method=str(manifest.get("universe_selection_method")),
        membership_mode=str(manifest.get("membership_mode")),
        dataset_reportable=bool(manifest.get("dataset_reportable")),
    )


def _pre2026_cache_binding(
    path: Path,
    *,
    expected_cache_sha256: str,
) -> Top2000M03RV7Seed172026YTDPre2026CacheBinding:
    if not path.is_file() or path.is_symlink():
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "pre-2026 cache must be a regular non-symlink file"
        )
    try:
        payload = training_data._load_verified_daily_cache_payload(
            path,
            expected_cache_sha256=expected_cache_sha256,
        )
    except Exception as exc:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "pre-2026 cache cannot be fully verified"
        ) from exc
    dates = tuple(payload.get("exchange_dates", ()))
    actions = tuple(payload.get("actions", ()))
    if (
        not dates
        or not actions
        or any(not isinstance(value, str) for value in dates)
        or any(value >= "2026-01-01" for value in dates)
        or actions[0] != "CASH"
        or _cache_axis_sha256(actions) != payload.get("action_hash")
        or _cache_axis_sha256(dates) != payload.get("date_hash")
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "pre-2026 cache date/action/schema boundary drifted"
        )
    return Top2000M03RV7Seed172026YTDPre2026CacheBinding(
        cache_path=str(path),
        cache_file_sha256=str(payload["cache_sha256"]),
        cache_identity=_require_sha256("cache_identity", payload.get("cache_identity")),
        base_dataset_identity=_require_sha256(
            "base_dataset_identity", payload.get("base_dataset_identity")
        ),
        search_identity=_require_sha256(
            "search_identity", payload.get("search_identity")
        ),
        lockbox_partition_names_hash=_require_sha256(
            "lockbox_partition_names_hash",
            payload.get("lockbox_partition_names_hash"),
        ),
        action_hash=_require_sha256("action_hash", payload.get("action_hash")),
        first_exchange_date=dates[0],
        last_exchange_date=dates[-1],
        state_rows=len(dates),
        action_count=len(actions),
        development_only=payload.get("development_only") is True,
        bars_only=payload.get("bars_only") is True,
    )


def _checkpoint_binding(
    *,
    package: M03RV7Seed17PackagePlan,
    pre2026_cache: Top2000M03RV7Seed172026YTDPre2026CacheBinding,
    training_output_root: Path,
    completion_index: int,
    fold_index: int,
    expected_completion_receipt_sha256: str,
    expected_worker_runtime_proof: Mapping[str, Any],
) -> Top2000M03RV7Seed172026YTDCheckpointBinding:
    completion_row = package.indices[completion_index]
    setting_index = completion_row.setting_index
    setting_root_relative = (
        f"completion-{completion_index:02d}-setting-{setting_index:02d}"
    )
    setting_root = training_output_root / setting_root_relative
    run_root = setting_root / "training"
    training_root_relative = f"{setting_root_relative}/training"
    cell_relative = f"receipts/fold-{fold_index:02d}-seed-17.json"
    validation_relative = (
        f"receipts/seed-validation/fold-{fold_index:02d}-seed-17.json"
    )
    execution_relative = f"receipts/fold-execution/fold-{fold_index:02d}.json"
    model_relative_inside = (
        f"cells/fold-{fold_index:02d}-seed-17/model.rank-00.pt"
    )
    completion_relative = "completion-receipt.json"
    cell_path = _relative_regular(run_root, cell_relative, label="cell receipt")
    validation_path = _relative_regular(
        run_root, validation_relative, label="seed-validation receipt"
    )
    execution_path = _relative_regular(
        run_root, execution_relative, label="fold-execution receipt"
    )
    model_path = _relative_regular(
        run_root, model_relative_inside, label="rank-00 model"
    )
    completion_path = _relative_regular(
        run_root, completion_relative, label="completion receipt"
    )
    cell = _read_json(cell_path, label="cell receipt")
    validation = _read_json(validation_path, label="seed-validation receipt")
    execution = _read_json(execution_path, label="fold-execution receipt")
    completion = _read_json(completion_path, label="completion receipt")
    if _file_sha256(completion_path) != expected_completion_receipt_sha256:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion receipt no longer matches terminal coverage"
        )
    expected = {
        "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "setting_index": setting_index,
        "setting_id": completion_row.setting_id,
        "fold_index": fold_index,
        "seed": 17,
        "development_only": True,
        "promotion_eligible": False,
    }
    _require_training_receipt(
        cell,
        {"schema": CELL_RECEIPT_SCHEMA, "mode": "full-seed17", **expected},
        label="cell receipt",
    )
    _require_training_receipt(
        validation,
        {
            "schema": M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
            **expected,
            "outer_evaluation_authorized": False,
        },
        label="seed-validation receipt",
    )
    _require_training_receipt(
        execution,
        {
            "schema": M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
            "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
            "setting_index": setting_index,
            "setting_id": completion_row.setting_id,
            "fold_index": fold_index,
            "ordered_seeds": [17],
            "member_count": 1,
            "chronological_return_path_count": 1,
            "one_member_fold_execution": True,
            "output_space_ensemble": False,
            "five_seed_ensemble_eligible": False,
            "outer_evaluation_authorized": False,
            "development_only": True,
            "promotion_eligible": False,
        },
        label="fold-execution receipt",
    )
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("decision_count") != 63:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "seed-validation receipt does not prove its 63-decision chronology"
        )
    _validated_h100_rank_proof(
        cell.get("rank_peak_cuda_memory"),
        label=f"completion {completion_index} fold {fold_index} cell",
    )
    _require_training_receipt(
        completion,
        {
            "schema": M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
            "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
            "setting_index": setting_index,
            "setting_id": completion_row.setting_id,
            "runtime_setting_id": completion_row.runtime_setting_id,
            "world_size": 2,
            "fold_count": 6,
            "paired_seeds": [17],
            "completed_cells": 6,
            "optimizer_steps_per_cell": 64,
            "seed_validation_receipt_count": 6,
            "fold_ensemble_receipt_count": 0,
            "fold_execution_receipt_count": 6,
            "inference_path_count": 6,
            "one_member_fold_execution_required": True,
            "five_seed_ensemble_eligible": False,
            "output_space_ensemble_required": False,
            "cache_sha256": pre2026_cache.cache_file_sha256,
            "cache_identity": pre2026_cache.cache_identity,
            "search_identity": pre2026_cache.search_identity,
            "action_hash": pre2026_cache.action_hash,
            "complete": True,
            "development_only": True,
            "future_selected_universe": True,
            "promotion_eligible": False,
            "outer_evaluation_authorized": False,
        },
        label="completion receipt",
    )
    expected_cell_inventory = {
        f"fold-{index:02d}-seed-17.json" for index in range(6)
    }
    expected_validation_inventory = {
        f"receipts/seed-validation/fold-{index:02d}-seed-17.json"
        for index in range(6)
    }
    expected_execution_inventory = {
        f"receipts/fold-execution/fold-{index:02d}.json" for index in range(6)
    }
    cell_inventory = completion.get("cell_receipt_sha256")
    validation_inventory = completion.get("seed_validation_receipt_sha256")
    execution_inventory = completion.get("fold_execution_receipt_sha256")
    if (
        not isinstance(cell_inventory, dict)
        or set(cell_inventory) != expected_cell_inventory
        or not isinstance(validation_inventory, dict)
        or set(validation_inventory) != expected_validation_inventory
        or not isinstance(execution_inventory, dict)
        or set(execution_inventory) != expected_execution_inventory
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion receipt child inventory drifted"
        )
    rank_models = cell.get("rank_model_sha256")
    state_hashes = cell.get("rank_model_state_sha256")
    validation_file_sha256 = _file_sha256(validation_path)
    if (
        not isinstance(rank_models, list)
        or len(rank_models) != 2
        or not isinstance(state_hashes, list)
        or len(state_hashes) != 2
        or len(set(state_hashes)) != 1
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "cell receipt omitted its exact two-rank model inventory"
        )
    model_file_sha256 = _file_sha256(model_path)
    cell_file_sha256 = _file_sha256(cell_path)
    execution_file_sha256 = _file_sha256(execution_path)
    completion_ranks = _validated_h100_rank_proof(
        completion.get("rank_peak_cuda_memory"),
        label=f"completion {completion_index}",
    )
    if completion_ranks != expected_worker_runtime_proof.get("rank_runtime"):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion H100 proof does not match lifecycle coverage"
        )
    if (
        model_file_sha256 != rank_models[0]
        or validation.get("checkpoint_file_sha256") != model_file_sha256
        or validation.get("model_state_sha256") != state_hashes[0]
        or cell.get("seed_validation_receipt_sha256")
        != validation_file_sha256
        or execution.get("seed_validation_receipt_sha256s")
        != [validation_file_sha256]
        or execution.get("member_checkpoint_file_sha256s")
        != [model_file_sha256]
        or execution.get("member_model_state_sha256s") != [state_hashes[0]]
        or cell_inventory[f"fold-{fold_index:02d}-seed-17.json"]
        != cell_file_sha256
        or validation_inventory[validation_relative] != validation_file_sha256
        or execution_inventory[execution_relative] != execution_file_sha256
        or cell.get("fold_receipt_sha256")
        != validation.get("fold_receipt_sha256")
        or cell.get("fold_receipt_sha256")
        != execution.get("fold_receipt_sha256")
        or cell.get("optimizer_steps") != 64
        or validation.get("checkpoint_selection_rule")
        != "frozen-final-optimizer-update-no-validation-selection-v1"
        or completion.get("plan_file_sha256") != cell.get("plan_file_sha256")
        or completion.get("plan_receipt_sha256")
        != cell.get("plan_receipt_sha256")
    ):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "rank-00 checkpoint or frozen final-update rule drifted"
        )
    training_plan_file_sha256 = _require_sha256(
        "training_plan_file_sha256", cell.get("plan_file_sha256")
    )
    training_plan_receipt_sha256 = _require_sha256(
        "training_plan_receipt_sha256", cell.get("plan_receipt_sha256")
    )
    return Top2000M03RV7Seed172026YTDCheckpointBinding(
        completion_index=completion_index,
        setting_index=setting_index,
        setting_id=completion_row.setting_id,
        runtime_setting_id=completion_row.runtime_setting_id,
        training_fold_index=fold_index,
        seed=17,
        writer_rank=0,
        optimizer_steps=64,
        checkpoint_role="headline" if fold_index == 5 else "cutoff-sensitivity",
        training_root_relative_path=training_root_relative,
        model_relative_path=f"{training_root_relative}/{model_relative_inside}",
        model_file_sha256=model_file_sha256,
        model_state_sha256=str(state_hashes[0]),
        cell_receipt_relative_path=f"{training_root_relative}/{cell_relative}",
        cell_receipt_file_sha256=cell_file_sha256,
        seed_validation_receipt_relative_path=(
            f"{training_root_relative}/{validation_relative}"
        ),
        seed_validation_receipt_file_sha256=validation_file_sha256,
        fold_execution_receipt_relative_path=(
            f"{training_root_relative}/{execution_relative}"
        ),
        fold_execution_receipt_file_sha256=execution_file_sha256,
        completion_receipt_relative_path=(
            f"{training_root_relative}/{completion_relative}"
        ),
        completion_receipt_file_sha256=_file_sha256(completion_path),
        training_plan_file_sha256=training_plan_file_sha256,
        training_plan_receipt_sha256=training_plan_receipt_sha256,
    )


def freeze_top2000_m03r_v7_seed17_2026_ytd_plan(
    *,
    training_output_root: str | Path,
    package_plan_path: str | Path,
    package_plan_sha256: str,
    pre2026_cache_path: str | Path,
    completion_coverage_receipt_path: str | Path,
    dataset_root: str | Path,
    evaluation_source_root: str | Path,
) -> Top2000M03RV7Seed172026YTDFrozenPlan:
    """Validate and freeze all pre-access lineage without opening 2026 bars."""

    _require_sha256("package_plan_sha256", package_plan_sha256)
    package = load_m03r_v7_seed17_top2000_package_plan(
        package_plan_path,
        expected_package_plan_sha256=package_plan_sha256,
        require_file_location_matches_plan=False,
    )
    pre2026_cache = _pre2026_cache_binding(
        Path(pre2026_cache_path),
        expected_cache_sha256=package.artifacts.cache_artifact_sha256,
    )
    training_root = Path(training_output_root)
    if training_root.name != M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "source training root does not match the completed q2 run identity"
        )
    coverage_path = Path(completion_coverage_receipt_path)
    coverage_file_sha256, completion_inventory, runtime_proof = (
        _validate_completion_coverage(
            coverage_path,
            package=package,
        )
    )
    namespace = _metadata_namespace(Path(dataset_root))
    if namespace.action_hash != pre2026_cache.action_hash:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "pre-2026 cache and raw namespace action identities differ"
        )
    source_binding = _evaluation_source_binding(Path(evaluation_source_root))
    by_setting = {row.setting_index: row.completion_index for row in package.indices}
    if set(by_setting) != set(range(12)):
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "package completion map does not cover every setting exactly once"
        )
    expected_completion_paths = {
        "completion-"
        f"{completion_index:02d}-setting-{setting_index:02d}/"
        "training/completion-receipt.json"
        for setting_index, completion_index in by_setting.items()
    }
    if set(completion_inventory) != expected_completion_paths:
        raise Top2000M03RV7Seed172026YTDWorkflowError(
            "completion coverage path inventory drifted from the package map"
        )
    checkpoints = tuple(
        _checkpoint_binding(
            package=package,
            pre2026_cache=pre2026_cache,
            training_output_root=training_root,
            completion_index=by_setting[setting_index],
            fold_index=fold_index,
            expected_completion_receipt_sha256=completion_inventory[
                "completion-"
                f"{by_setting[setting_index]:02d}-setting-{setting_index:02d}/"
                "training/completion-receipt.json"
            ],
            expected_worker_runtime_proof=runtime_proof[by_setting[setting_index]],
        )
        for setting_index in range(12)
        for fold_index in range(6)
    )
    return Top2000M03RV7Seed172026YTDFrozenPlan(
        source_run_id=M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID,
        source_training_output_root=str(training_root),
        source_package_plan_path=str(Path(package_plan_path)),
        source_package_plan_sha256=package_plan_sha256,
        source_package_plan_file_sha256=_file_sha256(Path(package_plan_path)),
        completion_coverage_receipt_path=str(coverage_path),
        completion_coverage_receipt_file_sha256=coverage_file_sha256,
        completion_worker_runtime_proof_sha256=_content_sha256(runtime_proof),
        evaluation_source=source_binding,
        evaluation_source_sha256=source_binding.inventory_sha256,
        data_namespace=namespace,
        pre2026_cache=pre2026_cache,
        checkpoints=checkpoints,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-plan",
        help="freeze checkpoint and metadata lineage before opening 2026 outcomes",
    )
    freeze.add_argument("--training-output-root", required=True)
    freeze.add_argument("--package-plan", required=True)
    freeze.add_argument("--package-plan-sha256", required=True)
    freeze.add_argument("--pre2026-cache", required=True)
    freeze.add_argument("--completion-coverage-receipt", required=True)
    freeze.add_argument("--dataset-root", required=True)
    freeze.add_argument("--evaluation-source-root", required=True)
    freeze.add_argument("--output-plan", required=True)
    cache = commands.add_parser(
        "build-cache",
        help="materialize one immutable outcome cache under an exact frozen plan",
    )
    cache.add_argument("--plan", required=True)
    cache.add_argument("--plan-file-sha256", required=True)
    cache.add_argument("--plan-receipt-sha256", required=True)
    cache.add_argument("--output-cache", required=True)
    cache.add_argument("--output-receipt", required=True)
    retrieve_factors = commands.add_parser(
        "retrieve-factors",
        help="retrieve exact official FF5+Momentum URLs under a frozen plan",
    )
    retrieve_factors.add_argument("--plan", required=True)
    retrieve_factors.add_argument("--plan-file-sha256", required=True)
    retrieve_factors.add_argument("--plan-receipt-sha256", required=True)
    retrieve_factors.add_argument("--output-directory", required=True)
    retrieve_factors.add_argument("--output-receipt", required=True)
    factors = commands.add_parser(
        "build-factors",
        help="bind official daily FF5+Momentum after the chronology is frozen",
    )
    factors.add_argument("--plan", required=True)
    factors.add_argument("--plan-file-sha256", required=True)
    factors.add_argument("--plan-receipt-sha256", required=True)
    factors.add_argument("--cache-receipt", required=True)
    factors.add_argument("--cache-receipt-file-sha256", required=True)
    factors.add_argument("--retrieval-receipt", required=True)
    factors.add_argument("--retrieval-receipt-file-sha256", required=True)
    factors.add_argument("--output-factor-data", required=True)
    factors.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "freeze-plan":
            plan = freeze_top2000_m03r_v7_seed17_2026_ytd_plan(
                training_output_root=args.training_output_root,
                package_plan_path=args.package_plan,
                package_plan_sha256=args.package_plan_sha256,
                pre2026_cache_path=args.pre2026_cache,
                completion_coverage_receipt_path=args.completion_coverage_receipt,
                dataset_root=args.dataset_root,
                evaluation_source_root=args.evaluation_source_root,
            )
            file_sha256 = _write_immutable_json(args.output_plan, asdict(plan))
            print(
                _canonical_json_bytes(
                    {
                        "plan_path": str(Path(args.output_plan)),
                        "plan_file_sha256": file_sha256,
                        "plan_receipt_sha256": plan.receipt_sha256,
                        "checkpoint_count": len(plan.checkpoints),
                        "outcome_partition_contents_opened": False,
                        "factor_archives_opened": False,
                        "development_only": True,
                        "scientific_reporting_eligible": False,
                        "promotion_eligible": False,
                    }
                ).decode("utf-8")
            )
            return 0
        if args.command == "build-cache":
            receipt = materialize_top2000_m03r_v7_seed17_2026_ytd_cache(
                plan_path=args.plan,
                expected_plan_file_sha256=args.plan_file_sha256,
                expected_plan_receipt_sha256=args.plan_receipt_sha256,
                output_cache_path=args.output_cache,
                output_receipt_path=args.output_receipt,
            )
            print(_canonical_json_bytes(receipt).decode("utf-8"))
            return 0
        if args.command == "retrieve-factors":
            evidence, file_sha256 = (
                retrieve_top2000_m03r_v7_seed17_2026_ytd_factors(
                    plan_path=args.plan,
                    expected_plan_file_sha256=args.plan_file_sha256,
                    expected_plan_receipt_sha256=args.plan_receipt_sha256,
                    output_directory=args.output_directory,
                    output_retrieval_receipt_path=args.output_receipt,
                )
            )
            print(
                _canonical_json_bytes(
                    {
                        "retrieval_receipt_path": str(Path(args.output_receipt)),
                        "retrieval_receipt_file_sha256": file_sha256,
                        "retrieval_receipt_sha256": evidence.receipt_sha256,
                        "five_factor_archive_sha256": (
                            evidence.five_factor_archive_sha256
                        ),
                        "momentum_archive_sha256": (
                            evidence.momentum_archive_sha256
                        ),
                        "official_source_verified": True,
                        "caller_staged_archives": False,
                        "development_only": True,
                        "scientific_reporting_eligible": False,
                        "promotion_eligible": False,
                    }
                ).decode("utf-8")
            )
            return 0
        if args.command == "build-factors":
            receipt, _factors = (
                materialize_top2000_m03r_v7_seed17_2026_ytd_factors(
                    plan_path=args.plan,
                    expected_plan_file_sha256=args.plan_file_sha256,
                    expected_plan_receipt_sha256=args.plan_receipt_sha256,
                    cache_receipt_path=args.cache_receipt,
                    expected_cache_receipt_file_sha256=(
                        args.cache_receipt_file_sha256
                    ),
                    retrieval_receipt_path=args.retrieval_receipt,
                    expected_retrieval_receipt_file_sha256=(
                        args.retrieval_receipt_file_sha256
                    ),
                    output_factor_data_path=args.output_factor_data,
                    output_receipt_path=args.output_receipt,
                )
            )
            print(_canonical_json_bytes(receipt).decode("utf-8"))
            return 0
    except (OSError, ValueError, Top2000M03RV7Seed172026YTDWorkflowError) as exc:
        print(f"TOP2000 M03R-v7 2026-YTD workflow failed: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_SEED17_TOP2000_2026_YTD_CHECKPOINT_SCHEMA",
    "M03R_SEED17_TOP2000_2026_YTD_PLAN_SCHEMA",
    "M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID",
    "Top2000M03RV7Seed172026YTDCheckpointBinding",
    "Top2000M03RV7Seed172026YTDDataNamespace",
    "Top2000M03RV7Seed172026YTDFrozenPlan",
    "Top2000M03RV7Seed172026YTDPre2026CacheBinding",
    "Top2000M03RV7Seed172026YTDWorkflowError",
    "freeze_top2000_m03r_v7_seed17_2026_ytd_plan",
    "main",
    "materialize_top2000_m03r_v7_seed17_2026_ytd_factors",
    "retrieve_top2000_m03r_v7_seed17_2026_ytd_factors",
]
