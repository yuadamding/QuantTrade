"""Runtime identity for deterministic Massive adaptive-RL execution."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
import json
import os
from pathlib import Path
import platform
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
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)


MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-execution-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-execution-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-receipt-envelope",
            "integrity_verification": "portable-no-runtime-recapture",
            "computational_replay": "exact-active-environment-comparison",
        }
    )
)
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "clean-git-head-plus-exact-runtime-source-inventories",
        "tracked_worktree_clean": True,
        "untracked_runtime_source_count": 0,
        "dependencies": "uv-lock-physical-sha256",
        "initialization": "explicit-cpu-float32",
        "training_device": "manifest-v3-bound",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "cuda_cublas_workspace": ":4096:8-when-cuda",
        "thread_counts": "captured-exactly",
        "persistence": "create-only-source-transaction",
        "verification": ("portable-integrity", "exact-computational-replay"),
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLExecutionEnvironmentV1Error(ValueError):
    """The adaptive-RL execution process differs from its attestation."""


class MassiveAdaptiveRLActiveExecutionEnvironmentMismatch(
    MassiveAdaptiveRLExecutionEnvironmentV1Error
):
    """The active worker cannot exactly replay a persisted environment."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            f"{name} must be a lowercase SHA-256"
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
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL source identity could not be captured"
        ) from error
    if completed.returncode != 0:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL source identity command failed"
        )
    return completed.stdout.decode("utf-8").strip()


def _repository_root() -> Path:
    root = Path(
        _git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if not Path(__file__).resolve().is_relative_to(root):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL implementation is outside its source repository"
        )
    return root


def _tracked_source_inventory(repository_root: Path) -> tuple[tuple[str, str], ...]:
    names = tuple(
        name
        for name in _git(repository_root, "ls-files", "src/rl_quant").splitlines()
        if name
    )
    return _source_inventory(repository_root, names=names, inventory="tracked")


def _source_inventory(
    repository_root: Path,
    *,
    names: tuple[str, ...],
    inventory: str,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for name in names:
        unresolved = repository_root / name
        if unresolved.is_symlink():
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                f"adaptive RL {inventory} source path differs"
            )
        path = unresolved.resolve(strict=True)
        if not path.is_relative_to(repository_root) or not path.is_file():
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                f"adaptive RL {inventory} source path differs"
            )
        rows.append((name, file_sha256(path)))
    if inventory == "tracked" and not rows:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL tracked source inventory is absent"
        )
    return tuple(rows)


def _untracked_runtime_source_inventory(
    repository_root: Path,
) -> tuple[tuple[str, str], ...]:
    names = tuple(
        name
        for name in _git(
            repository_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/rl_quant",
        ).splitlines()
        if name
    )
    return _source_inventory(repository_root, names=names, inventory="untracked")


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution-environment artifact ID is not path safe"
        )
    return value


def _driver_version() -> str:
    path = Path("/proc/driver/nvidia/version")
    if not path.is_file() or path.is_symlink():
        return "not-applicable"
    first = path.read_text(encoding="utf-8").splitlines()
    return first[0].strip() if first else "unavailable"


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    git_commit: str
    git_tree: str
    tracked_worktree_clean: bool
    tracked_worktree_status: tuple[str, ...]
    tracked_source_inventory: tuple[tuple[str, str], ...]
    tracked_source_inventory_sha256: str
    untracked_runtime_source_inventory: tuple[tuple[str, str], ...]
    untracked_runtime_source_count: int
    dependency_lock_sha256: str
    python_version: str
    python_implementation: str
    pytorch_version: str
    numpy_version: str
    cuda_runtime_version: str
    cudnn_version: str
    nvidia_driver_version: str
    execution_device_specification: str
    execution_device_type: str
    gpu_name: str
    gpu_compute_capability: str
    parameter_dtype: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_tf32: bool
    cudnn_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cublas_workspace_config: str
    torch_cpu_threads: int
    torch_interop_threads: int
    process_thread_environment: tuple[tuple[str, str], ...]
    training_seed: int
    model_initialization_specification_sha256: str
    initial_model_state_receipt_sha256: str
    source_data_qualified: bool
    source_transaction_verified: bool
    runtime_environment_replayed: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "source_transaction_verified",
                "runtime_environment_replayed",
                "_loaded_source",
            }
        }

    @property
    def development_execution_authorized(self) -> bool:
        """Whether this persisted environment can authorize development execution."""

        return bool(
            self.source_data_qualified
            and self.source_transaction_verified
            and self.runtime_environment_replayed
        )

    @property
    def scientific_execution_fingerprint_sha256(self) -> str:
        """Hash the cross-worker scientific settings, excluding physical identity."""

        return semantic_sha256(
            {
                "git_commit": self.git_commit,
                "git_tree": self.git_tree,
                "tracked_source_inventory_sha256": (
                    self.tracked_source_inventory_sha256
                ),
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "python_version": self.python_version,
                "python_implementation": self.python_implementation,
                "pytorch_version": self.pytorch_version,
                "numpy_version": self.numpy_version,
                "cuda_runtime_version": self.cuda_runtime_version,
                "cudnn_version": self.cudnn_version,
                "execution_device_type": self.execution_device_type,
                "parameter_dtype": self.parameter_dtype,
                "deterministic_algorithms": self.deterministic_algorithms,
                "deterministic_warn_only": self.deterministic_warn_only,
                "float32_matmul_tf32": self.float32_matmul_tf32,
                "cudnn_tf32": self.cudnn_tf32,
                "cudnn_benchmark": self.cudnn_benchmark,
                "cudnn_deterministic": self.cudnn_deterministic,
                "cublas_workspace_config": self.cublas_workspace_config,
                "torch_cpu_threads": self.torch_cpu_threads,
                "torch_interop_threads": self.torch_interop_threads,
                "process_thread_environment": self.process_thread_environment,
                "training_seed": self.training_seed,
                "model_initialization_specification_sha256": (
                    self.model_initialization_specification_sha256
                ),
                "initial_model_state_receipt_sha256": (
                    self.initial_model_state_receipt_sha256
                ),
            }
        )

    @property
    def physical_worker_compatibility_sha256(self) -> str:
        """Hash the minimum physical-worker class shared by all four folds."""

        return semantic_sha256(
            {
                "execution_device_type": self.execution_device_type,
                "gpu_name": self.gpu_name,
                "gpu_compute_capability": self.gpu_compute_capability,
                "cuda_runtime_version": self.cuda_runtime_version,
                "cudnn_version": self.cudnn_version,
                "nvidia_driver_version": self.nvidia_driver_version,
                "parameter_dtype": self.parameter_dtype,
            }
        )

    def validate(self) -> None:
        expected_qualified = bool(
            self.tracked_worktree_clean
            and not self.tracked_worktree_status
            and not self.untracked_runtime_source_inventory
            and self.untracked_runtime_source_count == 0
            and self.deterministic_algorithms
            and not self.deterministic_warn_only
            and not self.float32_matmul_tf32
            and not self.cudnn_tf32
            and not self.cudnn_benchmark
            and self.cudnn_deterministic
            and self.parameter_dtype == "torch.float32"
            and (
                self.execution_device_type != "cuda"
                or self.cublas_workspace_config == ":4096:8"
            )
        )
        tracked_names = tuple(row[0] for row in self.tracked_source_inventory)
        untracked_names = tuple(
            row[0] for row in self.untracked_runtime_source_inventory
        )
        source_rows_valid = bool(
            self.tracked_source_inventory
            and tracked_names == tuple(sorted(set(tracked_names)))
            and untracked_names == tuple(sorted(set(untracked_names)))
            and all(
                name.startswith("src/rl_quant/")
                and ".." not in Path(name).parts
                and len(receipt) == 64
                and all(
                    character in "0123456789abcdef" for character in receipt
                )
                for name, receipt in (
                    *self.tracked_source_inventory,
                    *self.untracked_runtime_source_inventory,
                )
            )
        )
        loaded_present = self._loaded_source is not None
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or len(self.git_commit) != 40
            or len(self.git_tree) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.git_commit + self.git_tree
            )
            or not self.python_version
            or not self.python_implementation
            or not self.pytorch_version
            or not self.numpy_version
            or not source_rows_valid
            or self.tracked_worktree_clean != (not self.tracked_worktree_status)
            or self.tracked_source_inventory_sha256
            != semantic_sha256(self.tracked_source_inventory)
            or isinstance(self.untracked_runtime_source_count, bool)
            or self.untracked_runtime_source_count
            != len(self.untracked_runtime_source_inventory)
            or not self.execution_device_specification
            or self.execution_device_type not in {"cpu", "cuda"}
            or self.torch_cpu_threads <= 0
            or self.torch_interop_threads <= 0
            or isinstance(self.training_seed, bool)
            or self.training_seed < 0
            or self.model_initialization_specification_sha256
            != MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256
            or self.source_data_qualified != expected_qualified
            or self.source_transaction_verified != loaded_present
            or not isinstance(self.runtime_environment_replayed, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                "adaptive RL execution environment authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                "adaptive RL execution-environment source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.tracked_source_inventory_sha256,
            self.dependency_lock_sha256,
            self.model_initialization_specification_sha256,
            self.initial_model_state_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL execution environment", value)
        semantic_configuration = self.semantic_unsigned()
        for implementation_inventory in (
            "tracked_worktree_status",
            "tracked_source_inventory",
            "untracked_runtime_source_inventory",
        ):
            semantic_configuration.pop(implementation_inventory)
        assert_no_adaptive_hold_semantics(semantic_configuration)


def capture_massive_adaptive_rl_execution_environment_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    initial_model_state_receipt_sha256: str,
    device: torch.device | str,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    """Capture the active deterministic process and exact current source tree."""

    manifest.validate()
    selected_device = torch.device(device)
    if str(selected_device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution device differs from Manifest V3"
        )
    _digest("adaptive RL initial model state", initial_model_state_receipt_sha256)
    root = _repository_root()
    unresolved_lock = root / "uv.lock"
    if unresolved_lock.is_symlink():
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL dependency lock path differs"
        )
    lock = unresolved_lock.resolve(strict=True)
    if not lock.is_relative_to(root):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL dependency lock path differs"
        )
    source_rows = _tracked_source_inventory(root)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    status_rows = tuple(row for row in status.splitlines() if row)
    untracked_source_rows = _untracked_runtime_source_inventory(root)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available() or selected_device.index is None:
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                "adaptive RL CUDA device is not explicit and available"
            )
        properties = torch.cuda.get_device_properties(selected_device)
        gpu_name = properties.name
        capability = f"{properties.major}.{properties.minor}"
    else:
        gpu_name = "not-applicable"
        capability = "not-applicable"
    thread_environment = tuple(
        (name, os.environ.get(name, "unset"))
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )
    source_qualified = bool(
        not status_rows
        and not untracked_source_rows
        and torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cudnn.benchmark
        and torch.backends.cudnn.deterministic
        and (
            selected_device.type != "cuda"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not status_rows,
        "tracked_worktree_status": status_rows,
        "tracked_source_inventory": source_rows,
        "tracked_source_inventory_sha256": semantic_sha256(source_rows),
        "untracked_runtime_source_inventory": untracked_source_rows,
        "untracked_runtime_source_count": len(untracked_source_rows),
        "dependency_lock_sha256": file_sha256(lock),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": torch.version.cuda or "not-applicable",
        "cudnn_version": str(torch.backends.cudnn.version() or "not-applicable"),
        "nvidia_driver_version": _driver_version(),
        "execution_device_specification": str(selected_device),
        "execution_device_type": selected_device.type,
        "gpu_name": gpu_name,
        "gpu_compute_capability": capability,
        "parameter_dtype": str(torch.float32),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "float32_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "process_thread_environment": thread_environment,
        "training_seed": manifest.base_manifest.seeds[0],
        "model_initialization_specification_sha256": (
            MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256
        ),
        "initial_model_state_receipt_sha256": initial_model_state_receipt_sha256,
        "source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLExecutionEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        source_transaction_verified=False,
        runtime_environment_replayed=True,
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def execution_environment_relative_path_v1(*, artifact_id: str) -> str:
    return (
        "massive-adaptive/rl-execution-environment-authority-v1/"
        f"{_artifact_id(artifact_id)}.json"
    )


def _execution_environment_payload(
    authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
) -> dict[str, object]:
    authority.validate()
    return {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }


def _tuple_rows(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            f"adaptive RL {name} is not a JSON row inventory"
        )
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) for item in row)
        ):
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                f"adaptive RL {name} row differs"
            )
        rows.append((row[0], row[1]))
    return tuple(rows)


def _parse_execution_environment_payload(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution environment is not canonical JSON"
        )
    payload = dict(cast(Mapping[str, object], value))
    for name in (
        "tracked_worktree_status",
        "process_thread_environment",
    ):
        rows = payload.get(name)
        if name == "tracked_worktree_status":
            if not isinstance(rows, list) or not all(
                isinstance(item, str) for item in rows
            ):
                raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                    "adaptive RL tracked worktree status differs"
                )
            payload[name] = tuple(rows)
        else:
            payload[name] = _tuple_rows(rows, name=name)
    payload["tracked_source_inventory"] = _tuple_rows(
        payload.get("tracked_source_inventory"),
        name="tracked source inventory",
    )
    payload["untracked_runtime_source_inventory"] = _tuple_rows(
        payload.get("untracked_runtime_source_inventory"),
        name="untracked runtime source inventory",
    )
    try:
        result = MassiveAdaptiveRLExecutionEnvironmentAuthorityV1(
            **payload,  # type: ignore[arg-type]
            source_transaction_verified=True,
            runtime_environment_replayed=False,
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution-environment payload fields differ"
        ) from error
    result.validate()
    if raw != canonical_json_file_bytes(_execution_environment_payload(result)):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution-environment payload did not round trip"
        )
    return result


def materialize_massive_adaptive_rl_execution_environment_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    """Publish the complete inventory and return portable integrity evidence."""

    authority.validate()
    if (
        not authority.runtime_environment_replayed
        or authority.source_transaction_verified
    ):
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL materialization requires an active environment witness"
        )
    resolved_root = Path(root)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution-environment root is unavailable"
        ) from error
    if resolved_root.is_symlink():
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL execution-environment root is a symlink"
        )
    relative = execution_environment_relative_path_v1(artifact_id=artifact_id)
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(_execution_environment_payload(authority))
        ),
        root=resolved_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-EXECUTION-ENVIRONMENT-V1-{_artifact_id(artifact_id)}",
    )
    loaded = load_massive_adaptive_rl_execution_environment_authority_v1(
        root=resolved_root,
        artifact_id=artifact_id,
        verified_at_ms=committed_at_ms,
    )
    if loaded.semantic_receipt_sha256 != authority.semantic_receipt_sha256:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "published adaptive RL execution environment differs"
        )
    return loaded


def load_massive_adaptive_rl_execution_environment_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    verified_at_ms: int,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    """Verify persisted environment integrity without recapturing this process."""

    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=execution_environment_relative_path_v1(
            artifact_id=artifact_id
        ),
        verified_at_ms=verified_at_ms,
    )
    return _parse_execution_environment_payload(
        root=root,
        loaded_source=loaded,
    )


def verify_massive_adaptive_rl_execution_environment_integrity_v1(
    *,
    root: str | Path,
    artifact_id: str,
    verified_at_ms: int,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    """Verify the persisted authority graph without inspecting this process."""

    return load_massive_adaptive_rl_execution_environment_authority_v1(
        root=root,
        artifact_id=artifact_id,
        verified_at_ms=verified_at_ms,
    )


def verify_massive_adaptive_rl_execution_environment_replay_v1(
    *,
    authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    initial_model_state_receipt_sha256: str,
    device: torch.device | str,
) -> MassiveAdaptiveRLExecutionEnvironmentAuthorityV1:
    """Require the active process to exactly reproduce a persisted authority."""

    authority.validate()
    if not authority.source_transaction_verified:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL replay requires persisted environment integrity"
        )
    try:
        active = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial_model_state_receipt_sha256,
            device=device,
        )
    except MassiveAdaptiveRLExecutionEnvironmentV1Error as error:
        raise MassiveAdaptiveRLActiveExecutionEnvironmentMismatch(
            "adaptive RL active execution environment is unavailable"
        ) from error
    if canonical_json_file_bytes(
        _execution_environment_payload(active)
    ) != canonical_json_file_bytes(_execution_environment_payload(authority)):
        raise MassiveAdaptiveRLActiveExecutionEnvironmentMismatch(
            "adaptive RL active execution environment did not replay"
        )
    result = replace(authority, runtime_environment_replayed=True)
    result.validate()
    return result


@contextmanager
def massive_adaptive_rl_deterministic_execution_v1(
    *, device: torch.device | str
) -> Iterator[None]:
    """Temporarily install the registered deterministic Torch switches."""

    selected_device = torch.device(device)
    if (
        selected_device.type == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise MassiveAdaptiveRLActiveExecutionEnvironmentMismatch(
            "CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    prior = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])
        torch.backends.cuda.matmul.allow_tf32 = prior[2]
        torch.backends.cudnn.allow_tf32 = prior[3]
        torch.backends.cudnn.benchmark = prior[4]
        torch.backends.cudnn.deterministic = prior[5]


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MassiveAdaptiveRLActiveExecutionEnvironmentMismatch",
    "MassiveAdaptiveRLExecutionEnvironmentAuthorityV1",
    "MassiveAdaptiveRLExecutionEnvironmentV1Error",
    "capture_massive_adaptive_rl_execution_environment_v1",
    "execution_environment_relative_path_v1",
    "load_massive_adaptive_rl_execution_environment_authority_v1",
    "massive_adaptive_rl_deterministic_execution_v1",
    "materialize_massive_adaptive_rl_execution_environment_authority_v1",
    "verify_massive_adaptive_rl_execution_environment_integrity_v1",
    "verify_massive_adaptive_rl_execution_environment_replay_v1",
]
