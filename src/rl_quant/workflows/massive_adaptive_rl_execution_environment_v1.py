"""Runtime identity for deterministic Massive adaptive-RL execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import subprocess
from typing import Iterator

import numpy as np
import torch

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
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "git-head-plus-current-tracked-source-inventory",
        "dependencies": "uv-lock-physical-sha256",
        "initialization": "explicit-cpu-float32",
        "training_device": "manifest-v3-bound",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "cuda_cublas_workspace": ":4096:8-when-cuda",
        "thread_counts": "captured-exactly",
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLExecutionEnvironmentV1Error(ValueError):
    """The adaptive-RL execution process differs from its attestation."""


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
        if name.endswith(".py")
    )
    rows: list[tuple[str, str]] = []
    for name in names:
        path = (repository_root / name).resolve(strict=True)
        if not path.is_relative_to(repository_root) or path.is_symlink():
            raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
                "adaptive RL tracked source path differs"
            )
        rows.append((name, file_sha256(path)))
    if not rows:
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL tracked source inventory is absent"
        )
    return tuple(rows)


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
    tracked_source_inventory_sha256: str
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

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "runtime_environment_replayed"}
        }

    def validate(self) -> None:
        expected_qualified = bool(
            self.deterministic_algorithms
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
            or not self.execution_device_specification
            or self.execution_device_type not in {"cpu", "cuda"}
            or self.torch_cpu_threads <= 0
            or self.torch_interop_threads <= 0
            or isinstance(self.training_seed, bool)
            or self.training_seed < 0
            or self.model_initialization_specification_sha256
            != MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256
            or self.source_data_qualified != expected_qualified
            or not self.runtime_environment_replayed
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
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


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
    lock = (root / "uv.lock").resolve(strict=True)
    if not lock.is_relative_to(root) or lock.is_symlink():
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
            "adaptive RL dependency lock path differs"
        )
    source_rows = _tracked_source_inventory(root)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
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
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not status,
        "tracked_source_inventory_sha256": semantic_sha256(source_rows),
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
        "source_data_qualified": True,
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
        runtime_environment_replayed=True,
        semantic_receipt_sha256=semantic_sha256(body),
    )
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
        raise MassiveAdaptiveRLExecutionEnvironmentV1Error(
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
    "MassiveAdaptiveRLExecutionEnvironmentAuthorityV1",
    "MassiveAdaptiveRLExecutionEnvironmentV1Error",
    "capture_massive_adaptive_rl_execution_environment_v1",
    "massive_adaptive_rl_deterministic_execution_v1",
]
