"""Persist and replay the numerical environment used for RL validation.

The training execution environment is deliberately not reused here: validation
is a CPU-only inference/economic-replay stage with a distinct capability and
source lineage.  Generic reload proves portable integrity only.  An authority
becomes executable after the active process exactly reproduces the persisted
scientific fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field, fields, replace
from io import BytesIO, StringIO
from importlib import metadata
import json
import fcntl
import os
from pathlib import Path
import platform
import stat
import subprocess
from typing import cast, Iterator

import numpy as np
import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    fold_validation_authority_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_evidence_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256,
    fold_validation_authority_relative_path_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    validation_cost_ladder_relative_path_v1,
    validation_fixed_control_relative_path_v1,
    validation_primary_trace_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256,
    policy_selection_authority_relative_path_v3,
    policy_selection_v2_witness_relative_path_v3,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-execution-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-validation-execution-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
        ),
        "encoding": "canonical-json-validation-execution-environment",
        "generic_reload": "integrity-only",
        "runtime_authorization": "exact-active-process-recapture",
    }
)
MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "lineage": "manifest-v4-runtime-sources-v2-fit-and-validation-barrier-v2",
        "source": "clean-git-head-and-complete-tracked-runtime-source-inventory",
        "dependency_lock": "uv-lock-physical-sha256",
        "installed_distributions": "canonical-name-and-version-inventory",
        "execution_device": "cpu-only",
        "parameter_and_observation_dtype": "torch-float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "torch_threads": "intra-op-and-inter-op-equal-one",
        "process_thread_environment": "omp-mkl-openblas-numexpr-equal-one",
        "python_hash_seed": "explicit-nonnegative-integer",
        "python_torch_numpy_and_blas": "captured-exactly",
        "cpu_model_instruction_set": "captured-exactly",
        "evaluator_implementations": "captured-exactly",
        "publication": "canonical-create-only-before-any-validation-outcome",
        "verification": ("portable-integrity", "exact-computational-replay"),
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)
_EVALUATOR_IMPLEMENTATION_INVENTORY = (
    (
        "policy-trace-v1",
        MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
    ),
    (
        "cost-ladder-v1",
        MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
    ),
    (
        "fixed-control-validation-v1",
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    ),
    (
        "fold-validation-v1",
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    ),
    (
        "validation-evidence-v2",
        MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256,
    ),
    (
        "selection-computation-v2",
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256,
    ),
    (
        "selection-authority-v3",
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256,
    ),
)


class MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(ValueError):
    """The validation worker or its persisted scientific identity differs."""


class MassiveAdaptiveRLValidationActiveEnvironmentMismatch(
    MassiveAdaptiveRLValidationExecutionEnvironmentV1Error
):
    """The active process cannot replay the persisted validation environment."""


class MassiveAdaptiveRLValidationExecutionEnvironmentLeaseUnavailable(
    MassiveAdaptiveRLValidationExecutionEnvironmentV1Error
):
    """Another process owns canonical validation-environment publication."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            f"{name} must be a lowercase SHA-256 digest"
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
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation source identity could not be captured"
        ) from error
    if completed.returncode != 0:
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation source identity command failed"
        )
    return completed.stdout.decode("utf-8").strip()


def _repository_root() -> Path:
    root = Path(
        _git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if not Path(__file__).resolve().is_relative_to(root):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation implementation is outside its source repository"
        )
    return root


def _source_inventory(
    repository_root: Path,
    *,
    names: tuple[str, ...],
    inventory_name: str,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for name in names:
        unresolved = repository_root / name
        if unresolved.is_symlink():
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                f"validation {inventory_name} source path differs"
            )
        path = unresolved.resolve(strict=True)
        if not path.is_relative_to(repository_root) or not path.is_file():
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                f"validation {inventory_name} source path differs"
            )
        rows.append((name, file_sha256(path)))
    if inventory_name == "tracked" and not rows:
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation tracked source inventory is absent"
        )
    return tuple(rows)


def _tracked_source_inventory(root: Path) -> tuple[tuple[str, str], ...]:
    names = tuple(
        name for name in _git(root, "ls-files", "src/rl_quant").splitlines() if name
    )
    return _source_inventory(root, names=names, inventory_name="tracked")


def _untracked_source_inventory(root: Path) -> tuple[tuple[str, str], ...]:
    names = tuple(
        name
        for name in _git(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/rl_quant",
        ).splitlines()
        if name
    )
    return _source_inventory(root, names=names, inventory_name="untracked")


def _cpu_identity() -> tuple[str, str]:
    model = platform.processor().strip() or platform.machine().strip() or "unavailable"
    flags: tuple[str, ...] = ()
    path = Path("/proc/cpuinfo")
    if path.is_file() and not path.is_symlink():
        rows = path.read_text(encoding="utf-8", errors="strict").splitlines()
        model_rows = tuple(
            row.split(":", 1)[1].strip()
            for row in rows
            if row.lower().startswith("model name") and ":" in row
        )
        flag_rows = tuple(
            row.split(":", 1)[1].strip().split()
            for row in rows
            if row.lower().startswith(("flags", "features")) and ":" in row
        )
        if model_rows:
            model = model_rows[0]
        if flag_rows:
            flags = tuple(sorted(set(flag_rows[0])))
    return model, semantic_sha256(flags)


def _numpy_build_configuration_sha256() -> str:
    stream = StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return semantic_sha256(stream.getvalue())


def _installed_distribution_inventory() -> tuple[tuple[str, str], ...]:
    observed: dict[str, str] = {}
    for distribution in metadata.distributions():
        try:
            raw_name = distribution.metadata["Name"]
        except KeyError as error:
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation installed distribution identity differs"
            ) from error
        version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip() or not version:
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation installed distribution identity differs"
            )
        name = raw_name.strip().lower().replace("_", "-")
        previous = observed.setdefault(name, version)
        if previous != version:
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation installed distribution versions conflict"
            )
    if not observed:
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation installed distribution inventory is absent"
        )
    return tuple(sorted(observed.items()))


def _transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment transaction is incomplete"
        )
    return all(present)


def _validation_outcome_evidence_exists(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
) -> bool:
    """Detect any outcome that makes late environment attestation invalid."""

    resolved = Path(root)
    exact: list[Path] = []
    for fold_index, checkpoints in enumerate(
        four_fold_validation_inputs_v2.expected_candidate_checkpoint_authority_receipt_inventories
    ):
        exact.extend(
            (
                resolved
                / validation_fixed_control_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                resolved
                / fold_validation_authority_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                resolved
                / fold_validation_authority_relative_path_v2(
                    manifest=manifest, fold_index=fold_index
                ),
                resolved
                / policy_selection_v2_witness_relative_path_v3(
                    manifest=manifest, fold_index=fold_index
                ),
                resolved
                / policy_selection_authority_relative_path_v3(
                    manifest=manifest, fold_index=fold_index
                ),
            )
        )
        for checkpoint in checkpoints:
            exact.extend(
                (
                    resolved
                    / validation_primary_trace_relative_path_v1(
                        manifest=manifest,
                        fold_index=fold_index,
                        checkpoint_authority_receipt_sha256=checkpoint,
                    ),
                    resolved
                    / validation_cost_ladder_relative_path_v1(
                        manifest=manifest,
                        fold_index=fold_index,
                        checkpoint_authority_receipt_sha256=checkpoint,
                    ),
                )
            )
    if any(
        path.exists()
        or path.is_symlink()
        or path.with_name(path.name + ".receipt.json").exists()
        or path.with_name(path.name + ".receipt.json").is_symlink()
        or path.with_name(path.name + ".commit.json").exists()
        or path.with_name(path.name + ".commit.json").is_symlink()
        for path in exact
    ):
        return True
    patterns = (
        (
            resolved / "massive-adaptive" / "rl-policy-trace-authority-v1",
            f"v4-{manifest.semantic_receipt_sha256}-fold*-checkpoint-*-primary.json*",
        ),
        (
            resolved / "massive-adaptive" / "rl-cost-ladder-authority-v1",
            f"v4-{manifest.semantic_receipt_sha256}-fold*-checkpoint-*-cost-ladder.json*",
        ),
        (
            resolved / "massive-adaptive" / "rl-validation-outcome-authority-v2",
            f"v4-{manifest.semantic_receipt_sha256}-fold-*-*.json*",
        ),
    )
    return any(
        directory.is_dir() and next(directory.glob(pattern), None) is not None
        for directory, pattern in patterns
    )


@contextmanager
def _validation_execution_environment_lease_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> Iterator[None]:
    manifest.validate()
    directory = (
        Path(root)
        / "massive-adaptive"
        / "rl-validation-execution-environment-leases-v1"
    )
    descriptor = -1

    def close_after_setup_failure() -> None:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        if Path(root).is_symlink():
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment root is a symlink"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment lease directory is a symlink"
            )
        descriptor = os.open(
            directory / f"v4-{manifest.semantic_receipt_sha256}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment lease identity differs"
            )
    except OSError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment lease is unavailable"
        ) from error
    except MassiveAdaptiveRLValidationExecutionEnvironmentV1Error:
        close_after_setup_failure()
        raise
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLValidationExecutionEnvironmentLeaseUnavailable(
            "validation execution-environment lease is already held"
        ) from error
    except OSError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment lease is unavailable"
        ) from error
    try:
        yield
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def validation_execution_environment_relative_path_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
) -> str:
    manifest.validate()
    four_fold_validation_inputs_v2.validate()
    if (
        manifest.semantic_receipt_sha256
        != four_fold_validation_inputs_v2.manifest_v4_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment lineage differs"
        )
    return (
        "massive-adaptive/rl-validation-execution-environment-authority-v1/"
        f"v4-{manifest.semantic_receipt_sha256}-"
        f"inputs-{four_fold_validation_inputs_v2.semantic_receipt_sha256}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    four_fold_validation_inputs_v2_receipt_sha256: str
    four_fold_validation_inputs_v2_source_receipt_sha256: str
    four_fold_validation_inputs_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_committed_at_ms: int
    git_commit: str
    git_tree: str
    tracked_worktree_clean: bool
    tracked_worktree_status: tuple[str, ...]
    tracked_source_inventory: tuple[tuple[str, str], ...]
    tracked_source_inventory_sha256: str
    untracked_runtime_source_inventory: tuple[tuple[str, str], ...]
    untracked_runtime_source_count: int
    dependency_lock_sha256: str
    installed_distribution_inventory: tuple[tuple[str, str], ...]
    installed_distribution_inventory_sha256: str
    python_version: str
    python_implementation: str
    pytorch_version: str
    numpy_version: str
    torch_build_configuration_sha256: str
    numpy_build_configuration_sha256: str
    platform_machine: str
    cpu_model: str
    cpu_capability: str
    cpu_instruction_inventory_sha256: str
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
    evaluator_implementation_inventory: tuple[tuple[str, str], ...]
    evaluator_implementation_inventory_sha256: str
    executor_implementation_source_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_environment_replayed: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_four_fold_fit: MassiveAdaptiveRLFourFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2 | None
    ) = field(default=None, compare=False, repr=False)

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {"semantic_receipt_sha256", "runtime_environment_replayed"}
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
    def source_transaction_observed_published_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.payload_ctime_ns // 1_000_000
        )

    @property
    def scientific_execution_fingerprint_sha256(self) -> str:
        return semantic_sha256(
            {
                "git_commit": self.git_commit,
                "git_tree": self.git_tree,
                "tracked_source_inventory_sha256": (
                    self.tracked_source_inventory_sha256
                ),
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "installed_distribution_inventory_sha256": (
                    self.installed_distribution_inventory_sha256
                ),
                "python_version": self.python_version,
                "python_implementation": self.python_implementation,
                "pytorch_version": self.pytorch_version,
                "numpy_version": self.numpy_version,
                "torch_build_configuration_sha256": (
                    self.torch_build_configuration_sha256
                ),
                "numpy_build_configuration_sha256": (
                    self.numpy_build_configuration_sha256
                ),
                "platform_machine": self.platform_machine,
                "cpu_model": self.cpu_model,
                "cpu_capability": self.cpu_capability,
                "cpu_instruction_inventory_sha256": (
                    self.cpu_instruction_inventory_sha256
                ),
                "execution_device_specification": (self.execution_device_specification),
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
                "evaluator_implementation_inventory_sha256": (
                    self.evaluator_implementation_inventory_sha256
                ),
                "executor_implementation_source_sha256": (
                    self.executor_implementation_source_sha256
                ),
            }
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_data_qualified
            and self.source_transaction_verified
            and self.runtime_environment_replayed
        )

    def validate(self) -> None:
        runtime_roots = (
            self._runtime_manifest,
            self._runtime_sources_v2,
            self._runtime_four_fold_fit,
            self._runtime_validation_inputs_v2,
        )
        runtime_present = any(value is not None for value in runtime_roots)
        if runtime_present != all(value is not None for value in runtime_roots):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment runtime lineage is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if self._runtime_manifest is not None:
            runtime_sources_v2 = cast(
                MassiveAdaptiveRLRuntimeSourcesV2, self._runtime_sources_v2
            )
            runtime_four_fold_fit = cast(
                MassiveAdaptiveRLFourFoldFitAuthorityV1,
                self._runtime_four_fold_fit,
            )
            runtime_validation_inputs_v2 = cast(
                MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
                self._runtime_validation_inputs_v2,
            )
            self._runtime_manifest.validate()
            runtime_sources_v2.validate()
            runtime_four_fold_fit.validate()
            runtime_validation_inputs_v2.validate()
            validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
                runtime_sources_v2=runtime_sources_v2,
                four_fold_fit_authority=runtime_four_fold_fit,
            )
            if (
                self._runtime_manifest.semantic_receipt_sha256
                != self.manifest_v4_receipt_sha256
                or runtime_sources_v2.semantic_receipt_sha256
                != self.runtime_sources_v2_receipt_sha256
                or runtime_four_fold_fit.semantic_receipt_sha256
                != self.four_fold_fit_authority_receipt_sha256
                or runtime_validation_inputs_v2.semantic_receipt_sha256
                != self.four_fold_validation_inputs_v2_receipt_sha256
                or runtime_sources_v2.source_bundle_v2_receipt_sha256
                != self.source_bundle_v2_receipt_sha256
                or runtime_sources_v2.runtime_source_graph_v2_receipt_sha256
                != self.runtime_source_graph_v2_receipt_sha256
                or runtime_sources_v2.runtime_source_graph_v2_witness_receipt_sha256
                != self.runtime_source_graph_v2_witness_receipt_sha256
                or runtime_sources_v2.replay_dependency_index_v2_receipt_sha256
                != self.replay_dependency_index_v2_receipt_sha256
                or runtime_validation_inputs_v2.source_receipt_sha256
                != self.four_fold_validation_inputs_v2_source_receipt_sha256
                or runtime_validation_inputs_v2.source_transaction_receipt_sha256
                != self.four_fold_validation_inputs_v2_commit_receipt_sha256
                or runtime_validation_inputs_v2.source_transaction_committed_at_ms
                != self.four_fold_validation_inputs_v2_committed_at_ms
                or not runtime_four_fold_fit.development_stage_authorized
                or not runtime_validation_inputs_v2.development_stage_authorized
            ):
                raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                    "validation execution-environment runtime lineage differs"
                )
        names = tuple(row[0] for row in self.tracked_source_inventory)
        untracked_names = tuple(
            row[0] for row in self.untracked_runtime_source_inventory
        )
        environment_names = tuple(row[0] for row in self.process_thread_environment)
        environment_values = dict(self.process_thread_environment)
        distribution_names = tuple(
            row[0] for row in self.installed_distribution_inventory
        )
        source_rows_valid = bool(
            self.tracked_source_inventory
            and names == tuple(sorted(set(names)))
            and untracked_names == tuple(sorted(set(untracked_names)))
            and all(
                name.startswith("src/rl_quant/")
                and ".." not in Path(name).parts
                and len(receipt) == 64
                and all(character in "0123456789abcdef" for character in receipt)
                for name, receipt in (
                    *self.tracked_source_inventory,
                    *self.untracked_runtime_source_inventory,
                )
            )
        )
        boolean_fields = (
            self.tracked_worktree_clean,
            self.deterministic_algorithms,
            self.deterministic_warn_only,
            self.float32_matmul_tf32,
            self.cudnn_tf32,
            self.cudnn_benchmark,
            self.cudnn_deterministic,
            self.source_data_qualified,
            self.runtime_environment_replayed,
            self.profitability_reporting_authorized,
            self.outer_evaluation_authorized,
            self.lockbox_access_authorized,
        )
        python_hash_seed = environment_values.get("PYTHONHASHSEED", "unset")
        expected_qualified = bool(
            self.tracked_worktree_clean
            and not self.tracked_worktree_status
            and not self.untracked_runtime_source_inventory
            and self.untracked_runtime_source_count == 0
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
            and python_hash_seed.isdigit()
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or any(type(value) is not bool for value in boolean_fields)
            or len(self.git_commit) != 40
            or len(self.git_tree) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.git_commit + self.git_tree
            )
            or not source_rows_valid
            or self.tracked_worktree_clean != (not self.tracked_worktree_status)
            or self.tracked_source_inventory_sha256
            != semantic_sha256(self.tracked_source_inventory)
            or isinstance(self.untracked_runtime_source_count, bool)
            or self.untracked_runtime_source_count
            != len(self.untracked_runtime_source_inventory)
            or not self.installed_distribution_inventory
            or distribution_names != tuple(sorted(set(distribution_names)))
            or any(
                not name or not version
                for name, version in self.installed_distribution_inventory
            )
            or self.installed_distribution_inventory_sha256
            != semantic_sha256(self.installed_distribution_inventory)
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.python_version,
                    self.python_implementation,
                    self.pytorch_version,
                    self.numpy_version,
                    self.platform_machine,
                    self.cpu_model,
                    self.cpu_capability,
                )
            )
            or self.execution_device_specification != "cpu"
            or self.parameter_dtype != "torch.float32"
            or self.observation_dtype != "torch.float32"
            or isinstance(self.torch_cpu_threads, bool)
            or self.torch_cpu_threads <= 0
            or isinstance(self.torch_interop_threads, bool)
            or self.torch_interop_threads <= 0
            or environment_names != _THREAD_ENVIRONMENT_NAMES
            or self.evaluator_implementation_inventory
            != _EVALUATOR_IMPLEMENTATION_INVENTORY
            or self.evaluator_implementation_inventory_sha256
            != semantic_sha256(self.evaluator_implementation_inventory)
            or self.source_data_qualified != expected_qualified
            or self.runtime_environment_replayed != runtime_present
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment authority differs"
            )
        for name in (
            "manifest_v4_receipt_sha256",
            "training_manifest_v3_receipt_sha256",
            "runtime_sources_v2_receipt_sha256",
            "source_bundle_v2_receipt_sha256",
            "runtime_source_graph_v2_receipt_sha256",
            "runtime_source_graph_v2_witness_receipt_sha256",
            "replay_dependency_index_v2_receipt_sha256",
            "four_fold_fit_authority_receipt_sha256",
            "four_fold_validation_inputs_v2_receipt_sha256",
            "four_fold_validation_inputs_v2_source_receipt_sha256",
            "four_fold_validation_inputs_v2_commit_receipt_sha256",
            "tracked_source_inventory_sha256",
            "dependency_lock_sha256",
            "installed_distribution_inventory_sha256",
            "torch_build_configuration_sha256",
            "numpy_build_configuration_sha256",
            "cpu_instruction_inventory_sha256",
            "evaluator_implementation_inventory_sha256",
            "executor_implementation_source_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            isinstance(self.four_fold_validation_inputs_v2_committed_at_ms, bool)
            or self.four_fold_validation_inputs_v2_committed_at_ms < 0
        ):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation input-barrier timestamp differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.four_fold_validation_inputs_v2_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "validation execution-environment source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _capture_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    executor_implementation_source_sha256: str,
) -> dict[str, object]:
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    four_fold_validation_inputs_v2.validate()
    _digest(
        "validation executor implementation",
        executor_implementation_source_sha256,
    )
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    barrier_source = four_fold_validation_inputs_v2.source_receipt_sha256
    barrier_commit = four_fold_validation_inputs_v2.source_transaction_receipt_sha256
    barrier_time = four_fold_validation_inputs_v2.source_transaction_committed_at_ms
    if (
        not four_fold_fit_authority.development_stage_authorized
        or not four_fold_validation_inputs_v2.development_stage_authorized
        or barrier_source is None
        or barrier_commit is None
        or barrier_time is None
        or manifest.experiment_id != runtime_sources_v2.experiment_id
        or manifest.semantic_receipt_sha256
        != four_fold_validation_inputs_v2.manifest_v4_receipt_sha256
        or runtime_sources_v2.semantic_receipt_sha256
        != four_fold_validation_inputs_v2.runtime_sources_v2_receipt_sha256
        or four_fold_fit_authority.semantic_receipt_sha256
        != four_fold_validation_inputs_v2.four_fold_fit_authority_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment roots differ"
        )
    repository = _repository_root()
    unresolved_lock = repository / "uv.lock"
    if unresolved_lock.is_symlink():
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation dependency lock path differs"
        )
    lock = unresolved_lock.resolve(strict=True)
    if not lock.is_relative_to(repository):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation dependency lock path differs"
        )
    tracked = _tracked_source_inventory(repository)
    untracked = _untracked_source_inventory(repository)
    status = tuple(
        row
        for row in _git(
            repository, "status", "--porcelain=v1", "--untracked-files=no"
        ).splitlines()
        if row
    )
    cpu_model, cpu_flags = _cpu_identity()
    distributions = _installed_distribution_inventory()
    environment = tuple(
        (name, os.environ.get(name, "unset")) for name in _THREAD_ENVIRONMENT_NAMES
    )
    environment_values = dict(environment)
    source_qualified = bool(
        not status
        and not untracked
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
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": runtime_sources_v2.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": runtime_sources_v2.source_bundle_v2_receipt_sha256,
        "runtime_source_graph_v2_receipt_sha256": runtime_sources_v2.runtime_source_graph_v2_receipt_sha256,
        "runtime_source_graph_v2_witness_receipt_sha256": runtime_sources_v2.runtime_source_graph_v2_witness_receipt_sha256,
        "replay_dependency_index_v2_receipt_sha256": runtime_sources_v2.replay_dependency_index_v2_receipt_sha256,
        "four_fold_fit_authority_receipt_sha256": four_fold_fit_authority.semantic_receipt_sha256,
        "four_fold_validation_inputs_v2_receipt_sha256": four_fold_validation_inputs_v2.semantic_receipt_sha256,
        "four_fold_validation_inputs_v2_source_receipt_sha256": barrier_source,
        "four_fold_validation_inputs_v2_commit_receipt_sha256": barrier_commit,
        "four_fold_validation_inputs_v2_committed_at_ms": barrier_time,
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not status,
        "tracked_worktree_status": status,
        "tracked_source_inventory": tracked,
        "tracked_source_inventory_sha256": semantic_sha256(tracked),
        "untracked_runtime_source_inventory": untracked,
        "untracked_runtime_source_count": len(untracked),
        "dependency_lock_sha256": file_sha256(lock),
        "installed_distribution_inventory": distributions,
        "installed_distribution_inventory_sha256": semantic_sha256(distributions),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_build_configuration_sha256": semantic_sha256(torch.__config__.show()),
        "numpy_build_configuration_sha256": _numpy_build_configuration_sha256(),
        "platform_machine": platform.machine() or "unavailable",
        "cpu_model": cpu_model,
        "cpu_capability": str(
            getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unavailable")()
        ),
        "cpu_instruction_inventory_sha256": cpu_flags,
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
        "process_thread_environment": environment,
        "evaluator_implementation_inventory": _EVALUATOR_IMPLEMENTATION_INVENTORY,
        "evaluator_implementation_inventory_sha256": semantic_sha256(
            _EVALUATOR_IMPLEMENTATION_INVENTORY
        ),
        "executor_implementation_source_sha256": (
            executor_implementation_source_sha256
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }


def capture_massive_adaptive_rl_validation_execution_environment_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    executor_implementation_source_sha256: str,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    body = _capture_body(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        executor_implementation_source_sha256=executor_implementation_source_sha256,
    )
    result = MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_environment_replayed=True,
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    result.validate()
    return result


def _payload(
    authority: MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
) -> dict[str, object]:
    authority.validate()
    return {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }


def _tuple_rows(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            f"validation {name} is not a JSON row inventory"
        )
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) for item in row)
        ):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                f"validation {name} row differs"
            )
        rows.append((row[0], row[1]))
    return tuple(rows)


def _parse(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution environment is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    status = body.get("tracked_worktree_status")
    if not isinstance(status, list) or not all(
        isinstance(item, str) for item in status
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation tracked worktree status differs"
        )
    body["tracked_worktree_status"] = tuple(status)
    for name in (
        "tracked_source_inventory",
        "untracked_runtime_source_inventory",
        "installed_distribution_inventory",
        "process_thread_environment",
        "evaluator_implementation_inventory",
    ):
        body[name] = _tuple_rows(body.get(name), name=name.replace("_", " "))
    try:
        result = MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1(
            **body,  # type: ignore[arg-type]
            runtime_environment_replayed=False,
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment payload fields differ"
        ) from error
    result.validate()
    if raw != canonical_json_file_bytes(_payload(result)):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment payload did not round trip"
        )
    return result


def load_massive_adaptive_rl_validation_execution_environment_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=validation_execution_environment_relative_path_v1(
            manifest=manifest,
            four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        ),
        verified_at_ms=verified_at_ms,
    )
    return _parse(root=root, loaded_source=loaded)


def materialize_massive_adaptive_rl_validation_execution_environment_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    authority: MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    manifest.validate()
    four_fold_validation_inputs_v2.validate()
    authority.validate()
    barrier_time = four_fold_validation_inputs_v2.source_transaction_committed_at_ms
    if (
        not authority.runtime_environment_replayed
        or authority.source_transaction_verified
        or not authority.source_data_qualified
        or barrier_time is None
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= barrier_time
        or authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or authority.four_fold_validation_inputs_v2_receipt_sha256
        != four_fold_validation_inputs_v2.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation environment materialization is not authorized"
        )
    if _validation_outcome_evidence_exists(
        root=root,
        manifest=manifest,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    ):
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "missing validation execution environment cannot be created after outcomes"
        )
    relative = validation_execution_environment_relative_path_v1(
        manifest=manifest,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    Path(root).mkdir(parents=True, exist_ok=True)
    if Path(root).is_symlink():
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation execution-environment root is a symlink"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(authority))),
        root=root,
        relative_payload_path=relative,
        dataset_id=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET
        ),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-VALIDATION-EXECUTION-ENVIRONMENT-V1-"
            f"{manifest.semantic_receipt_sha256}"
        ),
    )
    return load_massive_adaptive_rl_validation_execution_environment_v1(
        root=root,
        manifest=manifest,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        verified_at_ms=committed_at_ms,
    )


def authorize_massive_adaptive_rl_validation_execution_environment_v1(
    *,
    authority: MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    executor_implementation_source_sha256: str,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    authority.validate()
    if not authority.source_transaction_verified:
        raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
            "validation environment replay requires persisted integrity"
        )
    try:
        active = capture_massive_adaptive_rl_validation_execution_environment_v1(
            manifest=manifest,
            runtime_sources_v2=runtime_sources_v2,
            four_fold_fit_authority=four_fold_fit_authority,
            four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
            executor_implementation_source_sha256=(
                executor_implementation_source_sha256
            ),
        )
    except MassiveAdaptiveRLValidationExecutionEnvironmentV1Error as error:
        raise MassiveAdaptiveRLValidationActiveEnvironmentMismatch(
            "active validation execution environment is unavailable"
        ) from error
    if canonical_json_file_bytes(_payload(active)) != canonical_json_file_bytes(
        _payload(authority)
    ):
        raise MassiveAdaptiveRLValidationActiveEnvironmentMismatch(
            "active validation execution environment did not replay"
        )
    result = replace(
        authority,
        runtime_environment_replayed=True,
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_validation_execution_environment_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    executor_implementation_source_sha256: str,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    relative = validation_execution_environment_relative_path_v1(
        manifest=manifest,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    if _transaction_exists(root=root, relative=relative):
        generic = load_massive_adaptive_rl_validation_execution_environment_v1(
            root=root,
            manifest=manifest,
            four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
            verified_at_ms=committed_at_ms,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "canonical validation execution environment is absent"
            )
        if _validation_outcome_evidence_exists(
            root=root,
            manifest=manifest,
            four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        ):
            raise MassiveAdaptiveRLValidationExecutionEnvironmentV1Error(
                "missing validation execution environment cannot be created after outcomes"
            )
        with _validation_execution_environment_lease_v1(
            root=root, manifest=manifest
        ):
            if _transaction_exists(root=root, relative=relative):
                generic = load_massive_adaptive_rl_validation_execution_environment_v1(
                    root=root,
                    manifest=manifest,
                    four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
                    verified_at_ms=committed_at_ms,
                )
            else:
                captured = (
                    capture_massive_adaptive_rl_validation_execution_environment_v1(
                        manifest=manifest,
                        runtime_sources_v2=runtime_sources_v2,
                        four_fold_fit_authority=four_fold_fit_authority,
                        four_fold_validation_inputs_v2=(
                            four_fold_validation_inputs_v2
                        ),
                        executor_implementation_source_sha256=(
                            executor_implementation_source_sha256
                        ),
                    )
                )
                generic = materialize_massive_adaptive_rl_validation_execution_environment_v1(
                    root=root,
                    manifest=manifest,
                    four_fold_validation_inputs_v2=(
                        four_fold_validation_inputs_v2
                    ),
                    authority=captured,
                    committed_at_ms=committed_at_ms,
                )
    return authorize_massive_adaptive_rl_validation_execution_environment_v1(
        authority=generic,
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        executor_implementation_source_sha256=executor_implementation_source_sha256,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256",
    "MassiveAdaptiveRLValidationActiveEnvironmentMismatch",
    "MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1",
    "MassiveAdaptiveRLValidationExecutionEnvironmentLeaseUnavailable",
    "MassiveAdaptiveRLValidationExecutionEnvironmentV1Error",
    "authorize_massive_adaptive_rl_validation_execution_environment_v1",
    "capture_massive_adaptive_rl_validation_execution_environment_v1",
    "load_massive_adaptive_rl_validation_execution_environment_v1",
    "materialize_massive_adaptive_rl_validation_execution_environment_v1",
    "run_or_resume_massive_adaptive_rl_validation_execution_environment_v1",
    "validation_execution_environment_relative_path_v1",
]
