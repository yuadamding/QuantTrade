"""Freeze the exact executable V5 implementation before validation inputs.

Manifest V5 owns scientific choices.  This separate create-only authority owns
the code, dependency, and numerical-runtime identity used to generate economic
outcomes.  Keeping the two registrations separate lets the scientific manifest
remain stable while the package-owned vertical implementation is completed.
The implementation and its fixed real-economic qualification suite must be
registered before any validation tape is materialized; afterward, a code
change requires a new experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import cast

import numpy as np
import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes
from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    initial_validation_inputs_authority_relative_path_v1,
    massive_adaptive_rl_forbidden_prequential_artifacts_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    validation_decision_tensor_relative_path_v1,
    validation_environment_registry_relative_path_v1,
    validation_forecast_archive_relative_path_v1,
    validation_sources_authority_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    validation_environment_registry_relative_path_v2,
    validation_sources_authority_relative_path_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    load_massive_adaptive_rl_experiment_states_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_execution_registration_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_DATASET = (
    "massive-adaptive-rl-execution-implementation-registration-v1"
)
MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA
            ),
            "encoding": "canonical-json-exact-v5-execution-implementation",
            "generic_reload": "nonauthorizing",
        }
    )
)

_IMPLEMENTATION_RELATIVE_PATHS = (
    "src/rl_quant/evaluation/massive_adaptive_profitability_env_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_cost_ladder_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_fixed_control_evaluator_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_policy_evaluator_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_prequential_validation_inputs_v1.py",
    "src/rl_quant/training/massive_adaptive_rl_fixed_control_registry_v1.py",
    "src/rl_quant/training/massive_adaptive_rl_fixed_control_selection_v1.py",
    "src/rl_quant/training/massive_adaptive_rl_policy_selection_v2.py",
    "src/rl_quant/training/massive_adaptive_rl_policy_selection_v3.py",
    "src/rl_quant/workflows/massive_adaptive_rl_execution_implementation_registration_v1.py",
    "src/rl_quant/workflows/massive_adaptive_rl_experiment_lock_v1.py",
    "src/rl_quant/workflows/massive_adaptive_rl_experiment_runner_v5.py",
    "src/rl_quant/workflows/massive_adaptive_rl_initial_validation_execution_v1.py",
    "src/rl_quant/workflows/massive_adaptive_rl_manifest_v5.py",
    "src/rl_quant/workflows/massive_adaptive_rl_manifest_v5_registration.py",
    "src/rl_quant/workflows/massive_adaptive_rl_writer_guard_v5.py",
)
_REQUIRED_V5_NATIVE_IMPLEMENTATION_RELATIVE_PATHS = (
    "src/rl_quant/evaluation/massive_adaptive_rl_validation_release_authority_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_validation_outcome_authority_v3.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_fold_validation_authority_v3.py",
    "src/rl_quant/training/massive_adaptive_rl_policy_selection_v4.py",
    "src/rl_quant/training/massive_adaptive_frozen_rl_policy_v2.py",
    "src/rl_quant/training/massive_adaptive_rl_frozen_fc06_v2.py",
    "src/rl_quant/workflows/massive_adaptive_rl_walk_forward_policy_schedule_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_outer_access_commitment_v2.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_outer_inputs_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_outer_rollout_authority_v2.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_outer_fold_seal_authority_v1.py",
    "src/rl_quant/evaluation/massive_adaptive_rl_profitability_report_authority_v2.py",
    "src/rl_quant/workflows/massive_adaptive_rl_prequential_experiment_state_v1.py",
    "src/rl_quant/workflows/massive_adaptive_rl_outer_fold_execution_v1.py",
)
_VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS = (
    "tests/test_massive_adaptive_rl_v5_vertical.py",
)
_VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS = tuple(
    f"{_VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS[0]}::{name}"
    for name in (
        "test_one_step_position_return_lag",
        "test_unchanged_position_has_zero_turnover_cost",
        "test_nonmonotone_fixed_target_cost_ladder_is_reported_as_failed_gate",
        "test_terminal_liquidation_compounding_identity",
        "test_ppo_fc06_and_benchmark_share_outer_economics",
        "test_outer_zero_seal_precedes_validation_two_release",
        "test_outer_one_seal_precedes_validation_three_release",
        "test_diagnostic_schedule_completes_outer_report",
        "test_every_stage_resumes_to_identical_receipts",
        "test_predecessor_tampering_blocks_authorization",
        "test_full_cold_replay_is_nonmaterializing",
        "test_real_v5_vertical_executes_without_economic_mocks",
    )
)
_VERTICAL_QUALIFICATION_RECEIPT_FIELD_NAMES = (
    "vertical_qualification_specification_sha256",
    "vertical_qualification_test_paths",
    "missing_vertical_qualification_test_paths",
    "vertical_qualification_test_inventory",
    "vertical_qualification_test_inventory_sha256",
    "vertical_qualification_required_node_ids",
    "vertical_qualification_command",
    "vertical_qualification_exit_code",
    "vertical_qualification_passed_node_count",
    "vertical_qualification_nonpass_outcome_labels",
    "vertical_qualification_normalized_output_sha256",
    "vertical_qualification_passed",
)
_VERTICAL_QUALIFICATION_NONPASS_PATTERNS = (
    ("failed", rb"\b[1-9][0-9]* failed\b"),
    ("error", rb"\b[1-9][0-9]* errors?\b"),
    ("skipped", rb"\b[1-9][0-9]* skipped\b"),
    ("xfailed", rb"\b[1-9][0-9]* xfailed\b"),
    ("xpassed", rb"\b[1-9][0-9]* xpassed\b"),
    ("deselected", rb"\b[1-9][0-9]* deselected\b"),
)
_VERTICAL_QUALIFICATION_NONPASS_LABELS = tuple(
    row[0] for row in _VERTICAL_QUALIFICATION_NONPASS_PATTERNS
) + ("not-run",)
_SCOPED_OUTCOME_DIRECTORY_NAMES = (
    "validation-release-v1",
    "validation-outcome-v3",
    "fold-validation-v3",
    "policy-selection-v4",
    "frozen-policy-v2",
    "frozen-fc06-v2",
    "walk-forward-policy-schedule-v1",
    "outer-access-commitment-v2",
    "outer-input-authority-v1",
    "outer-rollout-authority-v2",
    "outer-fold-seal-authority-v1",
    "profitability-report-authority-v2",
    "prequential-experiment-state-v1",
)
_LEGACY_OUTCOME_DIRECTORY_NAMES = (
    "frozen-rl-policy-v1",
    "outer-access-commitment-v1",
    "outer-evidence-v1",
    "outer-forecast-archive-v1",
    "profit-trace-v1",
    "profit-trace-v2",
    "rl-cost-ladder-authority-v1",
    "rl-fixed-control-outer-cost-ladder-v1",
    "rl-fixed-control-outer-rollout-v1",
    "rl-fixed-control-validation-authority-v1",
    "rl-fold-validation-authority-v1",
    "rl-fold-validation-authority-v2",
    "rl-fold-validation-execution-authority-v1",
    "rl-four-fold-policy-selection-authority-v1",
    "rl-outer-cost-ladder-authority-v1",
    "rl-outer-evidence-authority-v4",
    "rl-outer-evidence-v1",
    "rl-outer-evidence-v3",
    "rl-outer-forecast-archive-v1",
    "rl-outer-rollout-v1",
    "rl-policy-selection-authority-v3",
    "rl-policy-selection-v1",
    "rl-policy-selection-v2",
    "rl-policy-trace-authority-v1",
    "rl-profitability-report-authority-v1",
    "rl-validation-outcome-authority-v2",
)
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)


class MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(ValueError):
    """The V5 executable implementation cannot be frozen or replayed."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation experiment ID is not path safe"
        )
    return value


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration clock differs"
        )
    return value


def _git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("/usr/bin/git", "-C", str(repository_root), *arguments),
            check=False,
            capture_output=True,
            timeout=30,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation source identity could not be captured"
        ) from error
    if completed.returncode != 0:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation source identity command failed"
        )
    return completed.stdout.decode("utf-8").strip()


def _repository_root() -> Path:
    root = Path(
        _git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if not Path(__file__).resolve().is_relative_to(root):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation is outside its repository"
        )
    return root


def _source_inventory(
    repository_root: Path, names: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for name in names:
        unresolved = repository_root / name
        if unresolved.is_symlink():
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "execution implementation source path is a symlink"
            )
        path = unresolved.resolve(strict=True)
        if not path.is_relative_to(repository_root) or not path.is_file():
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "execution implementation source path differs"
            )
        rows.append((name, file_sha256(path)))
    return tuple(rows)


def _vertical_qualification_receipt(values: Mapping[str, object]) -> str:
    return semantic_sha256(
        {name: values[name] for name in _VERTICAL_QUALIFICATION_RECEIPT_FIELD_NAMES}
    )


def _vertical_qualification(
    *, repository_root: Path, v5_native_vertical_complete: bool
) -> dict[str, object]:
    """Run the fixed real-vertical suite for the exact checkout being frozen."""

    missing = tuple(
        name
        for name in _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        if not (repository_root / name).is_file()
        or (repository_root / name).is_symlink()
    )
    present = tuple(
        name
        for name in _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        if name not in missing
    )
    inventory = _source_inventory(repository_root, present)
    command = (
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *_VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS,
    )
    if missing or not v5_native_vertical_complete:
        normalized_output_sha256 = hashlib.sha256(
            repr(("not-run", missing, v5_native_vertical_complete)).encode("utf-8")
        ).hexdigest()
        exit_code = None
        passed_node_count = 0
        nonpass_outcome_labels: tuple[str, ...] = ("not-run",)
        passed = False
    else:
        try:
            completed = subprocess.run(
                (sys.executable, *command[1:]),
                cwd=repository_root,
                check=False,
                capture_output=True,
                timeout=1_800,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "V5 vertical qualification could not execute"
            ) from error
        normalized_output = re.sub(
            rb"\bin [0-9]+(?:\.[0-9]+)?s\b",
            b"in <duration>",
            completed.stdout + b"\0" + completed.stderr,
        )
        normalized_output_sha256 = hashlib.sha256(normalized_output).hexdigest()
        exit_code = completed.returncode
        passed_matches = re.findall(rb"\b([0-9]+) passed\b", normalized_output)
        passed_node_count = int(passed_matches[-1]) if len(passed_matches) == 1 else 0
        nonpass_outcome_labels = tuple(
            label
            for label, pattern in _VERTICAL_QUALIFICATION_NONPASS_PATTERNS
            if re.search(pattern, normalized_output)
        )
        passed = bool(
            completed.returncode == 0
            and passed_node_count == len(_VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS)
            and not nonpass_outcome_labels
        )
    body: dict[str, object] = {
        "vertical_qualification_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256
        ),
        "vertical_qualification_test_paths": (
            _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        ),
        "missing_vertical_qualification_test_paths": missing,
        "vertical_qualification_test_inventory": inventory,
        "vertical_qualification_test_inventory_sha256": semantic_sha256(inventory),
        "vertical_qualification_required_node_ids": (
            _VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
        ),
        "vertical_qualification_command": command,
        "vertical_qualification_exit_code": exit_code,
        "vertical_qualification_passed_node_count": passed_node_count,
        "vertical_qualification_nonpass_outcome_labels": nonpass_outcome_labels,
        "vertical_qualification_normalized_output_sha256": (normalized_output_sha256),
        "vertical_qualification_passed": passed,
    }
    return {
        **body,
        "vertical_qualification_receipt_sha256": _vertical_qualification_receipt(body),
    }


def _replay_registered_vertical_qualification(
    *,
    repository_root: Path,
    authority: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
) -> dict[str, object]:
    """Verify the frozen qualification identity without launching pytest again."""

    missing = tuple(
        name
        for name in _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        if not (repository_root / name).is_file()
        or (repository_root / name).is_symlink()
    )
    present = tuple(
        name
        for name in _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        if name not in missing
    )
    inventory = _source_inventory(repository_root, present)
    body = {
        name: getattr(authority, name)
        for name in _VERTICAL_QUALIFICATION_RECEIPT_FIELD_NAMES
    }
    if (
        missing != authority.missing_vertical_qualification_test_paths
        or inventory != authority.vertical_qualification_test_inventory
        or authority.vertical_qualification_test_paths
        != _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
        or authority.vertical_qualification_required_node_ids
        != _VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
        or authority.vertical_qualification_receipt_sha256
        != _vertical_qualification_receipt(body)
        or not authority.vertical_qualification_passed
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "registered V5 vertical qualification identity differs"
        )
    return {
        **body,
        "vertical_qualification_receipt_sha256": (
            authority.vertical_qualification_receipt_sha256
        ),
    }


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


def execution_implementation_registration_relative_path_v1(
    *, experiment_id: str
) -> str:
    return (
        "adaptive-rl/"
        f"{_identifier(experiment_id)}/"
        "execution-implementation-registration-v1/registration.json"
    )


def execution_implementation_registration_transaction_state_v1(
    *, root: str | Path, experiment_id: str
) -> tuple[bool, bool]:
    return _transaction_state(
        root=root,
        relative=execution_implementation_registration_relative_path_v1(
            experiment_id=experiment_id
        ),
    )


def massive_adaptive_rl_preimplementation_economic_evidence_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
) -> tuple[str, ...]:
    """Return every current, future, partial, or legacy outcome namespace."""

    manifest.validate()
    experiment_id = manifest.experiment_id
    experiment = _identifier(experiment_id)
    resolved_root = Path(root)
    candidates = tuple(
        resolved_root / "adaptive-rl" / experiment / name
        for name in _SCOPED_OUTCOME_DIRECTORY_NAMES
    ) + tuple(
        resolved_root / "massive-adaptive" / name
        for name in _LEGACY_OUTCOME_DIRECTORY_NAMES
    )
    preaccess_relatives = [
        initial_validation_inputs_authority_relative_path_v1(
            manifest=manifest.base_manifest
        )
    ]
    for fold_index in (0, 1):
        preaccess_relatives.extend(
            (
                validation_decision_tensor_relative_path_v1(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
                validation_forecast_archive_relative_path_v1(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
                validation_sources_authority_relative_path_v1(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
                validation_environment_registry_relative_path_v1(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
                validation_sources_authority_relative_path_v2(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
                validation_environment_registry_relative_path_v2(
                    manifest=manifest.base_manifest, fold_index=fold_index
                ),
            )
        )
    candidates += tuple(resolved_root / relative for relative in preaccess_relatives)
    found = {
        str(path.relative_to(resolved_root))
        for path in candidates
        if path.exists() or path.is_symlink()
    }
    found.update(
        massive_adaptive_rl_forbidden_prequential_artifacts_v1(
            root=root,
            manifest=manifest.base_manifest,
        )
    )
    return tuple(sorted(found))


def _training_lineage_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> tuple[str, str]:
    """Derive the exact completed training state and four-fold fit receipts."""

    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=root,
        experiment_id=manifest.experiment_id,
    )
    matches = tuple(
        state
        for state in states
        if state.stage
        is MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
    )
    if (
        len(matches) != 1
        or matches[0].manifest_receipt_sha256
        != manifest.base_manifest.base_manifest.semantic_receipt_sha256
        or not matches[0].source_data_qualified
        or not matches[0].stage_artifact_receipt_sha256
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration requires exact completed training"
        )
    return (
        matches[0].semantic_receipt_sha256,
        matches[0].stage_artifact_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    manifest_v5_registration_authority_receipt_sha256: str
    manifest_v5_registration_source_receipt_sha256: str
    manifest_v5_registration_commit_receipt_sha256: str
    manifest_v5_registration_committed_at_ms: int
    training_state_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    git_commit: str
    git_tree: str
    source_worktree_clean: bool
    source_worktree_status: tuple[str, ...]
    tracked_source_inventory_sha256: str
    dependency_lock_sha256: str
    implementation_inventory: tuple[tuple[str, str], ...]
    implementation_inventory_sha256: str
    required_v5_native_implementation_paths: tuple[str, ...]
    missing_v5_native_implementation_paths: tuple[str, ...]
    v5_native_vertical_complete: bool
    vertical_qualification_specification_sha256: str
    vertical_qualification_test_paths: tuple[str, ...]
    missing_vertical_qualification_test_paths: tuple[str, ...]
    vertical_qualification_test_inventory: tuple[tuple[str, str], ...]
    vertical_qualification_test_inventory_sha256: str
    vertical_qualification_required_node_ids: tuple[str, ...]
    vertical_qualification_command: tuple[str, ...]
    vertical_qualification_exit_code: int | None
    vertical_qualification_passed_node_count: int
    vertical_qualification_nonpass_outcome_labels: tuple[str, ...]
    vertical_qualification_normalized_output_sha256: str
    vertical_qualification_passed: bool
    vertical_qualification_receipt_sha256: str
    python_version: str
    python_implementation: str
    pytorch_version: str
    numpy_version: str
    execution_device_specification: str
    parameter_dtype: str
    observation_dtype: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_tf32: bool
    cudnn_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    torch_cpu_threads: int
    torch_interop_threads: int
    process_thread_environment: tuple[tuple[str, str], ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_implementation_replayed: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_manifest_registration: (
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_training_lineage: tuple[str, str] | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {"semantic_receipt_sha256", "runtime_implementation_replayed"}
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def scientific_execution_fingerprint_sha256(self) -> str:
        return semantic_sha256(
            {
                "git_commit": self.git_commit,
                "git_tree": self.git_tree,
                "tracked_source_inventory_sha256": self.tracked_source_inventory_sha256,
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "implementation_inventory_sha256": self.implementation_inventory_sha256,
                "vertical_qualification_receipt_sha256": (
                    self.vertical_qualification_receipt_sha256
                ),
                "python_version": self.python_version,
                "python_implementation": self.python_implementation,
                "pytorch_version": self.pytorch_version,
                "numpy_version": self.numpy_version,
                "execution_device_specification": self.execution_device_specification,
                "parameter_dtype": self.parameter_dtype,
                "observation_dtype": self.observation_dtype,
                "deterministic_algorithms": self.deterministic_algorithms,
                "deterministic_warn_only": self.deterministic_warn_only,
                "float32_matmul_tf32": self.float32_matmul_tf32,
                "cudnn_tf32": self.cudnn_tf32,
                "cudnn_benchmark": self.cudnn_benchmark,
                "cudnn_deterministic": self.cudnn_deterministic,
                "torch_cpu_threads": self.torch_cpu_threads,
                "torch_interop_threads": self.torch_interop_threads,
                "process_thread_environment": self.process_thread_environment,
            }
        )

    @property
    def development_execution_registered(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_implementation_replayed
            and self.source_data_qualified
        )

    def validate(self) -> None:
        runtime_roots = (
            self._runtime_manifest,
            self._runtime_manifest_registration,
            self._runtime_training_lineage,
        )
        runtime_present = all(value is not None for value in runtime_roots)
        if any(value is not None for value in runtime_roots) != runtime_present:
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "execution implementation runtime lineage is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        environment_names = tuple(row[0] for row in self.process_thread_environment)
        environment_values = dict(self.process_thread_environment)
        expected_qualified = bool(
            self.source_worktree_clean
            and not self.source_worktree_status
            and self.v5_native_vertical_complete
            and not self.missing_v5_native_implementation_paths
            and self.vertical_qualification_passed
            and not self.missing_vertical_qualification_test_paths
            and self.vertical_qualification_exit_code == 0
            and self.vertical_qualification_passed_node_count
            == len(_VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS)
            and not self.vertical_qualification_nonpass_outcome_labels
            and self.execution_device_specification == "cpu"
            and self.parameter_dtype == "torch.float32"
            and self.observation_dtype == "torch.float32"
            and self.deterministic_algorithms
            and not self.deterministic_warn_only
            and not self.float32_matmul_tf32
            and not self.cudnn_tf32
            and not self.cudnn_benchmark
            and self.cudnn_deterministic
            and self.torch_cpu_threads == 1
            and self.torch_interop_threads == 1
            and all(
                environment_values.get(name) == "1"
                for name in _THREAD_ENVIRONMENT_NAMES[:-1]
            )
            and environment_values.get("PYTHONHASHSEED", "unset").isdigit()
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA
            or _identifier(self.experiment_id) != self.experiment_id
            or len(self.git_commit) != 40
            or len(self.git_tree) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.git_commit + self.git_tree
            )
            or self.source_worktree_clean != (not self.source_worktree_status)
            or self.required_v5_native_implementation_paths
            != _REQUIRED_V5_NATIVE_IMPLEMENTATION_RELATIVE_PATHS
            or self.missing_v5_native_implementation_paths
            != tuple(
                name
                for name in self.required_v5_native_implementation_paths
                if name not in {row[0] for row in self.implementation_inventory}
            )
            or self.v5_native_vertical_complete
            != (not self.missing_v5_native_implementation_paths)
            or self.vertical_qualification_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256
            or self.vertical_qualification_test_paths
            != _VERTICAL_QUALIFICATION_TEST_RELATIVE_PATHS
            or self.missing_vertical_qualification_test_paths
            != tuple(
                name
                for name in self.vertical_qualification_test_paths
                if name
                not in {row[0] for row in self.vertical_qualification_test_inventory}
            )
            or tuple(row[0] for row in self.vertical_qualification_test_inventory)
            != tuple(
                name
                for name in self.vertical_qualification_test_paths
                if name not in self.missing_vertical_qualification_test_paths
            )
            or self.vertical_qualification_test_inventory_sha256
            != semantic_sha256(self.vertical_qualification_test_inventory)
            or self.vertical_qualification_required_node_ids
            != _VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
            or self.vertical_qualification_command
            != (
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *_VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS,
            )
            or self.vertical_qualification_passed
            != (
                self.v5_native_vertical_complete
                and not self.missing_vertical_qualification_test_paths
                and self.vertical_qualification_exit_code == 0
                and self.vertical_qualification_passed_node_count
                == len(self.vertical_qualification_required_node_ids)
                and not self.vertical_qualification_nonpass_outcome_labels
            )
            or self.vertical_qualification_receipt_sha256
            != _vertical_qualification_receipt(self.semantic_unsigned())
            or self.vertical_qualification_exit_code is not None
            and (
                isinstance(self.vertical_qualification_exit_code, bool)
                or self.vertical_qualification_exit_code < 0
            )
            or isinstance(self.vertical_qualification_passed_node_count, bool)
            or not isinstance(self.vertical_qualification_passed_node_count, int)
            or self.vertical_qualification_passed_node_count < 0
            or self.vertical_qualification_passed_node_count
            > len(self.vertical_qualification_required_node_ids)
            or len(set(self.vertical_qualification_nonpass_outcome_labels))
            != len(self.vertical_qualification_nonpass_outcome_labels)
            or any(
                label not in _VERTICAL_QUALIFICATION_NONPASS_LABELS
                for label in self.vertical_qualification_nonpass_outcome_labels
            )
            or (
                self.vertical_qualification_exit_code is None
                and (
                    self.vertical_qualification_passed_node_count != 0
                    or self.vertical_qualification_nonpass_outcome_labels
                    != ("not-run",)
                )
            )
            or (
                self.vertical_qualification_exit_code is not None
                and "not-run" in self.vertical_qualification_nonpass_outcome_labels
            )
            or tuple(row[0] for row in self.implementation_inventory)
            != _IMPLEMENTATION_RELATIVE_PATHS
            + tuple(
                name
                for name in self.required_v5_native_implementation_paths
                if name not in self.missing_v5_native_implementation_paths
            )
            or self.implementation_inventory_sha256
            != semantic_sha256(self.implementation_inventory)
            or environment_names != _THREAD_ENVIRONMENT_NAMES
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.python_version,
                    self.python_implementation,
                    self.pytorch_version,
                    self.numpy_version,
                )
            )
            or isinstance(self.manifest_v5_registration_committed_at_ms, bool)
            or self.manifest_v5_registration_committed_at_ms < 0
            or isinstance(self.torch_cpu_threads, bool)
            or self.torch_cpu_threads <= 0
            or isinstance(self.torch_interop_threads, bool)
            or self.torch_interop_threads <= 0
            or self.source_data_qualified != expected_qualified
            or self.runtime_implementation_replayed != runtime_present
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "execution implementation registration differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.manifest_v5_registration_committed_at_ms
        ):
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                "execution implementation source transaction differs"
            )
        if runtime_present:
            manifest = cast(
                MassiveAdaptiveRLExperimentManifestV5, self._runtime_manifest
            )
            registration = cast(
                MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
                self._runtime_manifest_registration,
            )
            manifest.validate()
            registration.validate()
            training_state_receipt, fit_receipt = cast(
                tuple[str, str], self._runtime_training_lineage
            )
            if (
                manifest.semantic_receipt_sha256 != self.manifest_v5_receipt_sha256
                or registration.semantic_receipt_sha256
                != self.manifest_v5_registration_authority_receipt_sha256
                or registration.source_receipt_sha256
                != self.manifest_v5_registration_source_receipt_sha256
                or registration.source_transaction_receipt_sha256
                != self.manifest_v5_registration_commit_receipt_sha256
                or registration.source_transaction_committed_at_ms
                != self.manifest_v5_registration_committed_at_ms
                or not registration.development_protocol_registered
                or training_state_receipt != self.training_state_receipt_sha256
                or fit_receipt != self.four_fold_fit_authority_receipt_sha256
            ):
                raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                    "execution implementation runtime lineage differs"
                )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for name, value in self.implementation_inventory:
            if not name:
                raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                    "execution implementation source name is absent"
                )
            _digest("execution implementation source", value)
        for name, value in self.vertical_qualification_test_inventory:
            if not name:
                raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                    "V5 vertical qualification test name is absent"
                )
            _digest("V5 vertical qualification test", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _capture_body(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    registered_authority: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1 | None
    ) = None,
) -> dict[str, object]:
    manifest.validate()
    manifest_registration.validate()
    if (
        not manifest_registration.development_protocol_registered
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration roots differ"
        )
    registration_source = manifest_registration.source_receipt_sha256
    registration_commit = manifest_registration.source_transaction_receipt_sha256
    registration_time = manifest_registration.source_transaction_committed_at_ms
    if any(
        value is None
        for value in (
            registration_source,
            registration_commit,
            registration_time,
        )
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration source lineage is absent"
        )
    training_state_receipt, four_fold_fit_receipt = _training_lineage_v1(
        root=root, manifest=manifest
    )
    repository = _repository_root()
    tracked_names = tuple(
        name
        for name in _git(repository, "ls-files", "src/rl_quant").splitlines()
        if name
    )
    tracked = _source_inventory(repository, tracked_names)
    missing_v5_native = tuple(
        name
        for name in _REQUIRED_V5_NATIVE_IMPLEMENTATION_RELATIVE_PATHS
        if not (repository / name).is_file() or (repository / name).is_symlink()
    )
    present_v5_native = tuple(
        name
        for name in _REQUIRED_V5_NATIVE_IMPLEMENTATION_RELATIVE_PATHS
        if name not in missing_v5_native
    )
    implementation = _source_inventory(
        repository,
        _IMPLEMENTATION_RELATIVE_PATHS + present_v5_native,
    )
    qualification = (
        _vertical_qualification(
            repository_root=repository,
            v5_native_vertical_complete=not missing_v5_native,
        )
        if registered_authority is None
        else _replay_registered_vertical_qualification(
            repository_root=repository,
            authority=registered_authority,
        )
    )
    status = tuple(
        row
        for row in _git(
            repository, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if row
    )
    lock_path = repository / "uv.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution dependency lock differs"
        )
    process_environment = tuple(
        (name, os.environ.get(name, "unset")) for name in _THREAD_ENVIRONMENT_NAMES
    )
    environment_values = dict(process_environment)
    source_qualified = bool(
        not status
        and not missing_v5_native
        and bool(qualification["vertical_qualification_passed"])
        and torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cudnn.benchmark
        and torch.backends.cudnn.deterministic
        and torch.get_num_threads() == 1
        and torch.get_num_interop_threads() == 1
        and all(
            environment_values.get(name) == "1"
            for name in _THREAD_ENVIRONMENT_NAMES[:-1]
        )
        and environment_values.get("PYTHONHASHSEED", "unset").isdigit()
    )
    return {
        "schema": MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "manifest_v5_registration_authority_receipt_sha256": (
            manifest_registration.semantic_receipt_sha256
        ),
        "manifest_v5_registration_source_receipt_sha256": cast(
            str, registration_source
        ),
        "manifest_v5_registration_commit_receipt_sha256": cast(
            str, registration_commit
        ),
        "manifest_v5_registration_committed_at_ms": cast(int, registration_time),
        "training_state_receipt_sha256": training_state_receipt,
        "four_fold_fit_authority_receipt_sha256": four_fold_fit_receipt,
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "source_worktree_clean": not status,
        "source_worktree_status": status,
        "tracked_source_inventory_sha256": semantic_sha256(tracked),
        "dependency_lock_sha256": file_sha256(lock_path),
        "implementation_inventory": implementation,
        "implementation_inventory_sha256": semantic_sha256(implementation),
        "required_v5_native_implementation_paths": (
            _REQUIRED_V5_NATIVE_IMPLEMENTATION_RELATIVE_PATHS
        ),
        "missing_v5_native_implementation_paths": missing_v5_native,
        "v5_native_vertical_complete": not missing_v5_native,
        **qualification,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "execution_device_specification": "cpu",
        "parameter_dtype": str(torch.float32),
        "observation_dtype": str(torch.float32),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "process_thread_environment": process_environment,
        "source_data_qualified": source_qualified,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SHA256
        ),
    }


def _authority_from_captured_body(
    *,
    body: dict[str, object],
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    training_lineage = (
        cast(str, body["training_state_receipt_sha256"]),
        cast(str, body["four_fold_fit_authority_receipt_sha256"]),
    )
    result = MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_implementation_replayed=True,
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_training_lineage=training_lineage,
    )
    result.validate()
    return result


def capture_massive_adaptive_rl_execution_implementation_registration_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    """Run qualification once while capturing a new implementation freeze."""

    return _authority_from_captured_body(
        body=_capture_body(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
        ),
        manifest=manifest,
        manifest_registration=manifest_registration,
    )


def _payload(
    authority: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
) -> dict[str, object]:
    authority.validate()
    return {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }


def _tuple_rows(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            f"execution {name} is not a JSON row inventory"
        )
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) for item in row)
        ):
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                f"execution {name} row differs"
            )
        rows.append((row[0], row[1]))
    return tuple(rows)


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    status = body.get("source_worktree_status")
    if not isinstance(status, list) or not all(isinstance(row, str) for row in status):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution tracked worktree status differs"
        )
    body["source_worktree_status"] = tuple(status)
    for name in (
        "required_v5_native_implementation_paths",
        "missing_v5_native_implementation_paths",
        "vertical_qualification_test_paths",
        "missing_vertical_qualification_test_paths",
        "vertical_qualification_required_node_ids",
        "vertical_qualification_command",
        "vertical_qualification_nonpass_outcome_labels",
    ):
        value = body.get(name)
        if not isinstance(value, list) or not all(
            isinstance(row, str) for row in value
        ):
            raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
                f"execution {name.replace('_', ' ')} differs"
            )
        body[name] = tuple(value)
    for name in (
        "implementation_inventory",
        "vertical_qualification_test_inventory",
        "process_thread_environment",
    ):
        body[name] = _tuple_rows(body.get(name), name=name.replace("_", " "))
    try:
        result = MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1(
            **body,  # type: ignore[arg-type]
            runtime_implementation_replayed=False,
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration fields differ"
        ) from error
    result.validate()
    if raw != canonical_json_file_bytes(_payload(result)):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration did not round trip"
        )
    return result


def load_massive_adaptive_rl_execution_implementation_registration_v1(
    *, root: str | Path, experiment_id: str, verified_at_ms: int
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=(
                execution_implementation_registration_relative_path_v1(
                    experiment_id=experiment_id
                )
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_execution_implementation_registration_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    authority.validate()
    if not authority.source_transaction_verified:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration is not persisted"
        )
    active = _authority_from_captured_body(
        body=_capture_body(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            registered_authority=authority,
        ),
        manifest=manifest,
        manifest_registration=manifest_registration,
    )
    if active.semantic_unsigned() != authority.semantic_unsigned():
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "active execution implementation did not replay"
        )
    result = replace(
        authority,
        runtime_implementation_replayed=True,
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_training_lineage=(
            authority.training_state_receipt_sha256,
            authority.four_fold_fit_authority_receipt_sha256,
        ),
    )
    result.validate()
    return result


def _run_or_resume_execution_implementation_registration_v1_unlocked(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    allow_materialize: bool,
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    relative = execution_implementation_registration_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration transaction is incomplete"
        )
    verified_at_ms = _wall_clock_ms()
    if complete:
        return authorize_massive_adaptive_rl_execution_implementation_registration_v1(
            root=root,
            authority=load_massive_adaptive_rl_execution_implementation_registration_v1(
                root=root,
                experiment_id=manifest.experiment_id,
                verified_at_ms=verified_at_ms,
            ),
            manifest=manifest,
            manifest_registration=manifest_registration,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration is absent"
        )
    if massive_adaptive_rl_preimplementation_economic_evidence_v1(
        root=root,
        manifest=manifest,
    ):
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation must precede every validation input"
        )
    captured = capture_massive_adaptive_rl_execution_implementation_registration_v1(
        root=root,
        manifest=manifest,
        manifest_registration=manifest_registration,
    )
    if not captured.source_data_qualified:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "active execution implementation is not scientifically qualified"
        )
    committed_at_ms = max(
        verified_at_ms,
        captured.manifest_v5_registration_committed_at_ms + 1,
    )
    capability = (
        issue_massive_adaptive_rl_manifest_v5_execution_registration_capability_v1(
            root=root,
            authority=manifest_registration,
        )
    )
    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=root,
        capability=capability,
    ):
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(_payload(captured))),
            root=root,
            relative_payload_path=relative,
            dataset_id=(
                MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_DATASET
            ),
            source_object_key=relative,
            requested_at_ms=committed_at_ms,
            downloaded_at_ms=committed_at_ms,
            schema_sha256=(
                MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SCHEMA_SHA256
            ),
            entitlement_receipt_sha256=captured.semantic_receipt_sha256,
            committed_at_ms=committed_at_ms,
            request_id=(
                f"ADAPTIVE-RL-EXECUTION-IMPLEMENTATION-{manifest.experiment_id}"
            ),
        )
    return authorize_massive_adaptive_rl_execution_implementation_registration_v1(
        root=root,
        authority=load_massive_adaptive_rl_execution_implementation_registration_v1(
            root=root,
            experiment_id=manifest.experiment_id,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        manifest_registration=manifest_registration,
    )


def run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1:
    """Freeze or replay the exact executable implementation under the global lock."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation materialization mode differs"
        )
    if not allow_materialize:
        return _run_or_resume_execution_implementation_registration_v1_unlocked(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            allow_materialize=False,
        )
    try:
        with massive_adaptive_rl_experiment_orchestration_lock_v1(
            artifact_root=root,
            experiment_id=manifest.experiment_id,
        ):
            return _run_or_resume_execution_implementation_registration_v1_unlocked(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                allow_materialize=True,
            )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLExecutionImplementationRegistrationV1Error(
            "execution implementation registration lock is invalid"
        ) from error


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SOURCE_SHA256",
    "MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1",
    "MassiveAdaptiveRLExecutionImplementationRegistrationV1Error",
    "_run_or_resume_execution_implementation_registration_v1_unlocked",
    "authorize_massive_adaptive_rl_execution_implementation_registration_v1",
    "capture_massive_adaptive_rl_execution_implementation_registration_v1",
    "execution_implementation_registration_relative_path_v1",
    "execution_implementation_registration_transaction_state_v1",
    "load_massive_adaptive_rl_execution_implementation_registration_v1",
    "massive_adaptive_rl_preimplementation_economic_evidence_v1",
    "run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1",
]
