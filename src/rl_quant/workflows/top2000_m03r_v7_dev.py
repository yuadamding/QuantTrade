"""Governed two-rank worker for the TOP2000 M03R-v7 development panel.

This command is deliberately nonpromotable.  It consumes the verified,
pre-2026 future-selected TOP2000 daily-OHLCV cache and runs one of the twelve
compatibility settings.  A production invocation must be launched by
``torchrun --nproc_per_node=2``; single-rank execution is accepted only by the
bounded qualification mode.

The worker owns no Kubernetes client.  It validates one immutable training
plan, loads the cache once per rank, follows a deterministic episode schedule,
and publishes rank-zero JSON receipts with atomic replacement.  Rank-local
checkpoints retain RNG state so a restarted Pod resumes the exact fold, seed,
optimizer update, and sampled chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import (  # type: ignore[attr-defined]
    PrefixStore as _TorchPrefixStore,
)
from torch.distributed import (  # type: ignore[attr-defined]
    rendezvous as _torch_distributed_rendezvous,
)

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.top2000_m03r_v7_dev import (
    Top2000M03RV7FoldEnsembleReceipt,
    Top2000M03RV7OutputSpaceEnsemblePolicy,
    Top2000M03RV7SeedValidationReceipt,
    Top2000M03RV7ValidationError,
    evaluate_top2000_m03r_v7_validation_trace,
    model_state_sha256,
    tensor_sha256,
    validate_fold_score_bounds,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_TOP2000_DEV_SETTING_IDS,
    resolve_m03r_top2000_dev_setting,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000IndexPlan,
    M03RV7Top2000PackageError,
    M03RV7Top2000PackagePlan,
    M03RV7Top2000RuntimeProfile,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    TOP2000_M03R_V7_DEV_SEEDS,
    Top2000M03RV7DevelopmentError,
    Top2000M03RV7DevelopmentFold,
    Top2000M03RV7DevelopmentPolicy,
    Top2000M03RV7DevelopmentTrainingPlan,
    bind_top2000_m03r_v7_runtime_sequence,
    build_top2000_m03r_v7_development_optimizers,
    render_top2000_m03r_v7_development_folds,
    train_top2000_m03r_v7_development_update,
)
from rl_quant.training.top2000_m03r_v7_factor_calibration import (
    TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS,
    fit_top2000_m03r_v7_warmup_factor_calibration,
)

WORKER_SCHEMA = "rl-quant.top2000-dev.m03r-v7-worker-v2"
CHECKPOINT_SCHEMA = "rl-quant.top2000-dev.m03r-v7-rank-checkpoint-v2"
CELL_MODEL_SCHEMA = "rl-quant.top2000-dev.m03r-v7-cell-model-v2"
CELL_RECEIPT_SCHEMA = "rl-quant.top2000-dev.m03r-v7-cell-receipt-v2"
PROGRESS_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-progress-manifest-v2"
)
SEED_VALIDATION_RECEIPT_DIRECTORY = "seed-validation"
FOLD_ENSEMBLE_RECEIPT_DIRECTORY = "fold-ensemble"
COMPLETION_RECEIPT_SCHEMA = "rl-quant.top2000-dev.m03r-v7-completion-receipt-v1"
QUALIFICATION_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-bounded-qualification-v1"
)
MAX_EXTENDED_QUALIFICATION_STEPS = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INTENTIONAL_RESTART_MARKER_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-intentional-restart-marker-v1"
)
_INTENTIONAL_RESTART_MARKER_ROOT = Path(
    "/tmp/rl-quant-top2000-m03r-v7-intentional-restart"
)
_INTENTIONAL_RESTART_RENDEZVOUS_TIMEOUT_SECONDS = 30.0
_INTENTIONAL_RESTART_POLL_SECONDS = 0.05
_INTENTIONAL_RESTART_SOCKET_SETTLE_SECONDS = 5.0
_PROCESS_GROUP_STORE_PREFIX_ROOT = "rl-quant.top2000-m03r-v7"


class Top2000M03RV7WorkerError(RuntimeError):
    """The worker input, distributed topology, or resume state is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optimizer_state_canonical_value(value: Any) -> Any:
    """Render optimizer state independently of device and torch serialization."""

    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "sha256": tensor_sha256(value),
        }
    if isinstance(value, Mapping):
        rows = [
            (
                f"{type(key).__name__}:{key}",
                _optimizer_state_canonical_value(item),
            )
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: row[0])
        return {"type": "mapping", "rows": rows}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "values": [_optimizer_state_canonical_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "values": [_optimizer_state_canonical_value(item) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    raise Top2000M03RV7WorkerError(
        f"optimizer state contains unsupported {type(value).__name__}"
    )


def optimizer_state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Content-bind one optimizer state without pickle/zip metadata."""

    if not isinstance(state_dict, Mapping):
        raise Top2000M03RV7WorkerError("optimizer state must be a mapping")
    return _sha256_bytes(
        _canonical_json_bytes(_optimizer_state_canonical_value(state_dict))
    )


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    return optimizer_state_dict_sha256(optimizer.state_dict())


def _requires_overlay_optimizer(
    plan: Top2000M03RV7DevelopmentTrainingPlan,
) -> bool:
    return (
        resolve_m03r_top2000_dev_setting(plan.setting_id).sharpe_mode
        == "separate-total-risk-overlay"
    )


def _require_sha256(name: str, value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise Top2000M03RV7WorkerError(
            f"{name} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
    )
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(dict(payload))
    _atomic_write_bytes(path, encoded)
    return _sha256_bytes(encoded)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(dict(payload))
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise Top2000M03RV7WorkerError(
                f"immutable receipt collision at {path}"
            )
        return _sha256_bytes(existing)
    _atomic_write_bytes(path, encoded)
    return _sha256_bytes(encoded)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def _semantic_torch_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash tensor payload content without torch archive metadata."""

    return _sha256_bytes(
        _canonical_json_bytes(_optimizer_state_canonical_value(payload))
    )


def _write_immutable_torch(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish once, reusing an existing semantically identical artifact.

    ``torch.save`` includes archive metadata derived from the destination name,
    so serializing identical tensors through two random temporary names does
    not guarantee identical file bytes.  Final cell models are bound by their
    file digest in seed-validation receipts.  Reusing a semantically identical
    artifact closes the retry window after that receipt is published but before
    the cell receipt is committed.
    """

    expected = _semantic_torch_payload_sha256(payload)
    if path.exists():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise Top2000M03RV7WorkerError(
                f"immutable torch artifact cannot be read at {path}"
            ) from exc
        if (
            not isinstance(existing, Mapping)
            or _semantic_torch_payload_sha256(existing) != expected
        ):
            raise Top2000M03RV7WorkerError(
                f"immutable torch artifact collision at {path}"
            )
        return _sha256_file(path)
    return _atomic_torch_save(path, payload)


def _read_pinned_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    _require_sha256(f"{label}_sha256", expected_sha256)
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise Top2000M03RV7WorkerError(f"cannot read {label}: {path}") from exc
    actual = _sha256_bytes(encoded)
    if actual != expected_sha256:
        raise Top2000M03RV7WorkerError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise Top2000M03RV7WorkerError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise Top2000M03RV7WorkerError(f"{label} must be a JSON object")
    return value


def load_training_plan(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_setting_index: int | None = None,
) -> tuple[Top2000M03RV7DevelopmentTrainingPlan, str]:
    """Load one exact worker plan and validate its typed scientific identity."""

    plan_path = Path(path)
    payload = _read_pinned_json(
        plan_path,
        expected_sha256,
        label="training_plan",
    )
    try:
        plan = Top2000M03RV7DevelopmentTrainingPlan(**payload)
    except (TypeError, ValueError, Top2000M03RV7DevelopmentError) as exc:
        raise Top2000M03RV7WorkerError("training plan failed typed validation") from exc
    if expected_setting_index is not None and plan.setting_index != expected_setting_index:
        raise Top2000M03RV7WorkerError(
            "worker setting index does not match its pinned training plan"
        )
    if plan.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256:
        raise Top2000M03RV7WorkerError("training plan protocol hash drifted")
    return plan, expected_sha256


def render_training_plan(
    path: str | Path,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
) -> str:
    """Atomically render a canonical per-setting plan and return its file hash."""

    return _write_immutable_json(Path(path), asdict(plan))


def load_package_plan(
    path: str | Path,
    *,
    expected_package_plan_sha256: str | None = None,
) -> M03RV7Top2000PackagePlan:
    """Reconstruct and self-validate the content-addressed Indexed-Job plan."""

    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV7WorkerError("package plan cannot be read") from exc
    if not isinstance(payload, dict):
        raise Top2000M03RV7WorkerError("package plan must be a JSON object")
    # Stagers may serialize either ``asdict(plan)`` or the canonical payload
    # plus its self hash.  The typed reconstruction below rejects every
    # scientific or admission-order drift in either representation.
    payload = dict(payload)
    payload.pop("schema", None)
    try:
        artifacts_payload = payload.pop("artifacts")
        indices_payload = payload.pop("indices")
        runtime_profile_payload = payload.pop("runtime_profile")
        if not isinstance(artifacts_payload, dict) or not isinstance(
            indices_payload, list
        ) or not isinstance(runtime_profile_payload, dict):
            raise TypeError("package children must be objects")
        artifacts = M03RV7Top2000ArtifactBindings(**artifacts_payload)
        runtime_profile = M03RV7Top2000RuntimeProfile(**runtime_profile_payload)
        indices = tuple(
            M03RV7Top2000IndexPlan(
                **{
                    **row,
                    "fold_indices": tuple(row["fold_indices"]),
                    "paired_seeds": tuple(row["paired_seeds"]),
                }
            )
            for row in indices_payload
        )
        package = M03RV7Top2000PackagePlan(
            artifacts=artifacts,
            indices=indices,
            runtime_profile=runtime_profile,
            **payload,
        )
    except (TypeError, ValueError, M03RV7Top2000PackageError) as exc:
        raise Top2000M03RV7WorkerError(
            "package plan failed typed content validation"
        ) from exc
    if Path(package.plan_artifact_path) != plan_path:
        raise Top2000M03RV7WorkerError(
            "package plan path does not match its bound container path"
        )
    if expected_package_plan_sha256 is not None:
        _require_sha256(
            "expected_package_plan_sha256", expected_package_plan_sha256
        )
        if package.package_plan_sha256 != expected_package_plan_sha256:
            raise Top2000M03RV7WorkerError(
                "package plan does not match the admitted package hash"
            )
    return package


def resolve_completion_index(explicit: int | None) -> int:
    """Resolve the Indexed-Job completion without shell interpolation."""

    environment = os.environ.get("JOB_COMPLETION_INDEX")
    from_environment: int | None = None
    if environment is not None:
        try:
            from_environment = int(environment)
        except ValueError as exc:
            raise Top2000M03RV7WorkerError(
                "JOB_COMPLETION_INDEX must be an exact integer"
            ) from exc
    if explicit is not None and from_environment is not None and explicit != from_environment:
        raise Top2000M03RV7WorkerError(
            "explicit completion index disagrees with JOB_COMPLETION_INDEX"
        )
    resolved = explicit if explicit is not None else from_environment
    if resolved is None or isinstance(resolved, bool) or not 0 <= resolved < 12:
        raise Top2000M03RV7WorkerError("completion index must lie in [0, 11]")
    return resolved


def plan_from_package_completion(
    package: M03RV7Top2000PackagePlan,
    *,
    package_plan_path: str | Path,
    completion_index: int,
    output_root: str | Path,
) -> tuple[Top2000M03RV7DevelopmentTrainingPlan, str]:
    """Derive the source-fixed worker plan for one admitted completion."""

    if not 0 <= completion_index < len(package.indices):
        raise Top2000M03RV7WorkerError("completion index is outside the package")
    row = package.indices[completion_index]
    package_path = Path(package_plan_path)
    cache_path = package_path.parent / "cache.pt"
    setting_root = (
        Path(output_root)
        / f"completion-{completion_index:02d}-setting-{row.setting_index:02d}"
    )
    plan = Top2000M03RV7DevelopmentTrainingPlan(
        setting_index=row.setting_index,
        setting_id=row.development_setting_id,
        cache_path=str(cache_path),
        cache_sha256=package.artifacts.cache_artifact_sha256,
        output_root=str(setting_root),
        total_optimizer_steps_per_fold_seed=(
            package.runtime_profile.optimizer_steps_per_fold_seed
        ),
        max_origin_batch=package.runtime_profile.max_origin_batch,
        learning_rate=package.runtime_profile.learning_rate,
        weight_decay=package.runtime_profile.weight_decay,
        grad_clip=package.runtime_profile.grad_clip,
        token_dim=package.runtime_profile.token_dim,
        raw_stock_chunk=package.runtime_profile.raw_stock_chunk,
        expected_world_size=package.runtime_profile.expected_world_size,
        activation_checkpointing=(
            package.runtime_profile.activation_checkpointing
        ),
        mixed_precision=package.runtime_profile.mixed_precision,
    )
    binding = {
        "schema": "rl-quant.top2000-dev.m03r-v7-package-worker-binding-v1",
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_path": str(package_path),
        "completion": asdict(row),
        "training_plan": asdict(plan),
        "training_plan_receipt_sha256": plan.receipt_sha256,
        "episode_schedule_sha256": plan.episode_schedule_sha256,
        "cache_path_convention": "sibling-cache.pt",
        "output_root": str(setting_root),
        "development_only": True,
        "promotion_eligible": False,
    }
    binding_path = setting_root / "execution-plan-binding.json"
    binding_sha256 = _write_immutable_json(binding_path, binding)
    return plan, binding_sha256


def deterministic_episode_start(
    *,
    episode_schedule_sha256: str,
    fold: Top2000M03RV7DevelopmentFold,
    seed: int,
    optimizer_step: int,
) -> int:
    """Map a cell/update identity to one reproducible 378-state episode."""

    _require_sha256("episode_schedule_sha256", episode_schedule_sha256)
    if seed not in TOP2000_M03R_V7_DEV_SEEDS:
        raise Top2000M03RV7WorkerError("episode seed is not in the paired inventory")
    if isinstance(optimizer_step, bool) or optimizer_step < 0:
        raise Top2000M03RV7WorkerError("optimizer_step must be nonnegative")
    minimum_start = fold.training_state_start
    maximum_start = (
        fold.training_state_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    )
    if maximum_start < minimum_start:
        raise Top2000M03RV7WorkerError(
            "fold cannot supply a complete 378-state episode"
        )
    payload = {
        "schema": "rl-quant.top2000-dev.m03r-v7-episode-draw-v2",
        "episode_schedule_sha256": episode_schedule_sha256,
        "fold_receipt_sha256": fold.receipt_sha256,
        "seed": seed,
        "optimizer_step": optimizer_step,
    }
    draw = int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(payload)).digest()[:8],
        byteorder="big",
    )
    choices = maximum_start - minimum_start + 1
    return minimum_start + draw % choices


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _rank_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device).cpu().clone()
            if device.type == "cuda"
            else None
        ),
    }


def _restore_rank_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    python_state = state.get("python")
    cpu_state = state.get("torch_cpu")
    cuda_state = state.get("torch_cuda")
    if python_state is None or not isinstance(cpu_state, torch.Tensor):
        raise Top2000M03RV7WorkerError("checkpoint RNG state is incomplete")
    random.setstate(python_state)
    torch.set_rng_state(cpu_state)
    if device.type == "cuda":
        if not isinstance(cuda_state, torch.Tensor):
            raise Top2000M03RV7WorkerError("checkpoint omitted CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state, device)
    elif cuda_state is not None:
        raise Top2000M03RV7WorkerError("CPU resume received a CUDA RNG state")


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _move_sequence_to_device(sequence: Any, device: torch.device) -> Any:
    """Move the authoritative sequence without changing any economic values."""

    ledger = CohortLedger(
        economic_value=sequence.initial_ledger.economic_value.to(device),
        retention_units=sequence.initial_ledger.retention_units.to(device),
        cash_index=sequence.initial_ledger.cash_index,
    )
    return replace(
        sequence,
        decision_state=sequence.decision_state.to(device),
        asset_returns=sequence.asset_returns.to(device),
        decision_available=sequence.decision_available.to(device),
        fill_membership=sequence.fill_membership.to(device),
        fill_availability=sequence.fill_availability.to(device),
        benchmark_weights=sequence.benchmark_weights.to(device),
        risk_asset_caps=sequence.risk_asset_caps.to(device),
        risk_gross_max=sequence.risk_gross_max.to(device),
        benchmark_net_returns=sequence.benchmark_net_returns.to(device),
        initial_ledger=ledger,
        initial_equity=(
            None
            if sequence.initial_equity is None
            else sequence.initial_equity.to(device)
        ),
        track_entry_units=(
            None
            if sequence.track_entry_units is None
            else sequence.track_entry_units.to(device)
        ),
    )


@dataclass(frozen=True, slots=True)
class _DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool


def _distributed_context(*, qualification_only: bool) -> _DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if qualification_only:
        if world_size not in {1, 2}:
            raise Top2000M03RV7WorkerError(
                "bounded qualification supports only one or two ranks"
            )
    elif world_size != 2:
        raise Top2000M03RV7WorkerError(
            "full training requires torchrun --nproc_per_node=2"
        )
    owns = False
    if world_size == 2 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise Top2000M03RV7WorkerError("two-rank execution requires CUDA")
        torch.cuda.set_device(local_rank)
        store, rendezvous_rank, rendezvous_world_size = (
            _generation_scoped_env_rendezvous(
                expected_rank=rank,
                expected_world_size=world_size,
            )
        )
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=rendezvous_rank,
            world_size=rendezvous_world_size,
        )
        owns = True
    if world_size == 2:
        if not dist.is_initialized() or dist.get_world_size() != world_size:
            raise Top2000M03RV7WorkerError("distributed process group is inconsistent")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size == 2:
        gpu_name = torch.cuda.get_device_name(device)
        properties = torch.cuda.get_device_properties(device)
        total_memory = properties.total_memory
        if (
            gpu_name != "NVIDIA H100 80GB HBM3"
            or not 79 * 1024**3 <= total_memory <= 81 * 1024**3
            or (properties.major, properties.minor) != (9, 0)
        ):
            raise Top2000M03RV7WorkerError(
                "two-rank execution requires one NVIDIA H100 80GB HBM3 per rank"
            )
    return _DistributedContext(rank, local_rank, world_size, device, owns)


def _barrier(context: _DistributedContext) -> None:
    if context.world_size == 2:
        dist.barrier()


def _torchrun_restart_count() -> int:
    raw = os.environ.get("TORCHELASTIC_RESTART_COUNT")
    try:
        value = int(raw) if raw is not None else 0
    except ValueError as exc:
        raise Top2000M03RV7WorkerError(
            "TORCHELASTIC_RESTART_COUNT must be an exact integer"
        ) from exc
    if value not in {0, 1}:
        raise Top2000M03RV7WorkerError(
            "TOP2000 M03R-v7 permits exactly one torchrun restart"
        )
    return value


def _process_group_store_prefix(restart_count: int) -> str:
    if restart_count not in {0, 1}:
        raise Top2000M03RV7WorkerError(
            "process-group store restart generation must be zero or one"
        )
    return f"{_PROCESS_GROUP_STORE_PREFIX_ROOT}/restart-generation-{restart_count}"


def _generation_scoped_env_rendezvous(
    *,
    expected_rank: int,
    expected_world_size: int,
) -> tuple[Any, int, int]:
    restart_count = _torchrun_restart_count()
    rendezvous = _torch_distributed_rendezvous("env://")
    try:
        base_store, rank, world_size = next(rendezvous)
    except StopIteration as exc:
        raise Top2000M03RV7WorkerError(
            "env:// rendezvous returned no process-group store"
        ) from exc
    if rank != expected_rank or world_size != expected_world_size:
        raise Top2000M03RV7WorkerError(
            "env:// rendezvous rank/world geometry drifted"
        )
    prefix = _process_group_store_prefix(restart_count)
    return _TorchPrefixStore(prefix, base_store), rank, world_size


def _intentional_restart_rendezvous_path(
    run_root: Path,
    *,
    restart_count: int,
) -> Path:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if not run_id:
        raise Top2000M03RV7WorkerError(
            "intentional qualification restart requires torchrun agent identity"
        )
    if restart_count not in {0, 1}:
        raise Top2000M03RV7WorkerError(
            "intentional restart rendezvous count must be zero or one"
        )
    key = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": _INTENTIONAL_RESTART_MARKER_SCHEMA,
                "torchrun_run_id": run_id,
                "torchrun_agent_pid": os.getppid(),
                "run_root": str(run_root.absolute()),
                "restart_count": restart_count,
            }
        )
    )
    return _INTENTIONAL_RESTART_MARKER_ROOT / key


def _destroy_process_group(
    context: _DistributedContext,
    *,
    force: bool = False,
) -> bool:
    """Destroy an owned group at most once; return whether this call did so."""

    if not dist.is_available() or not dist.is_initialized():
        return False
    if not force and not context.owns_process_group:
        return False
    dist.destroy_process_group()
    if dist.is_initialized():
        raise Top2000M03RV7WorkerError(
            "distributed process group remained initialized after destroy"
        )
    return True


def _publish_destroyed_process_group_marker(
    rendezvous: Path,
    *,
    context: _DistributedContext,
    rendezvous_key: str,
) -> None:
    _require_sha256("intentional restart rendezvous key", rendezvous_key)
    if rendezvous.name != rendezvous_key:
        raise Top2000M03RV7WorkerError(
            "intentional restart rendezvous path/key mismatch"
        )
    _atomic_write_json(
        rendezvous / f"destroyed.rank-{context.rank:02d}.json",
        {
            "schema": _INTENTIONAL_RESTART_MARKER_SCHEMA,
            "rendezvous_key": rendezvous_key,
            "rank": context.rank,
            "world_size": context.world_size,
            "process_group_destroyed": True,
        },
    )


def _wait_for_destroyed_process_group_markers(
    rendezvous: Path,
    *,
    rendezvous_key: str,
    world_size: int,
    timeout_seconds: float = _INTENTIONAL_RESTART_RENDEZVOUS_TIMEOUT_SECONDS,
    poll_seconds: float = _INTENTIONAL_RESTART_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if world_size != 2:
        raise Top2000M03RV7WorkerError(
            "intentional restart marker rendezvous requires exactly two ranks"
        )
    if timeout_seconds <= 0.0 or poll_seconds <= 0.0:
        raise Top2000M03RV7WorkerError(
            "intentional restart marker timing must be positive"
        )
    _require_sha256("intentional restart rendezvous key", rendezvous_key)
    deadline = monotonic() + timeout_seconds
    while True:
        complete = True
        for rank in range(world_size):
            marker_path = rendezvous / f"destroyed.rank-{rank:02d}.json"
            if not marker_path.exists():
                complete = False
                continue
            try:
                marker = json.loads(marker_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise Top2000M03RV7WorkerError(
                    "intentional restart marker cannot be read"
                ) from exc
            expected = {
                "schema": _INTENTIONAL_RESTART_MARKER_SCHEMA,
                "rendezvous_key": rendezvous_key,
                "rank": rank,
                "world_size": world_size,
                "process_group_destroyed": True,
            }
            if marker != expected:
                raise Top2000M03RV7WorkerError(
                    "intentional restart marker identity drifted"
                )
        if complete:
            return
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise Top2000M03RV7WorkerError(
                "intentional restart marker rendezvous timed out"
            )
        sleep(min(poll_seconds, remaining))


def _cleanup_intentional_restart_rendezvous(rendezvous: Path) -> None:
    """Best-effort cleanup; safe when both restarted ranks call it."""

    if rendezvous.parent != _INTENTIONAL_RESTART_MARKER_ROOT or not _SHA256_RE.fullmatch(
        rendezvous.name
    ):
        raise Top2000M03RV7WorkerError(
            "intentional restart cleanup path is outside the marker root"
        )
    shutil.rmtree(rendezvous, ignore_errors=True)


def _prepare_intentional_qualification_restart(
    context: _DistributedContext,
    run_root: Path,
    *,
    restart_count: int,
    settle_sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Quiesce both ranks before torchrun performs its one qualified restart."""

    if context.world_size != 2 or restart_count != 0:
        raise Top2000M03RV7WorkerError(
            "intentional restart handshake requires the first two-rank attempt"
        )
    rendezvous = _intentional_restart_rendezvous_path(
        run_root,
        restart_count=restart_count,
    )
    rendezvous_key = rendezvous.name
    _barrier(context)
    if not _destroy_process_group(context, force=True):
        raise Top2000M03RV7WorkerError(
            "intentional restart handshake found no initialized process group"
        )
    _publish_destroyed_process_group_marker(
        rendezvous,
        context=context,
        rendezvous_key=rendezvous_key,
    )
    _wait_for_destroyed_process_group_markers(
        rendezvous,
        rendezvous_key=rendezvous_key,
        world_size=context.world_size,
    )
    settle_sleep(_INTENTIONAL_RESTART_SOCKET_SETTLE_SECONDS)


def _gather_objects(value: Any, context: _DistributedContext) -> list[Any]:
    if context.world_size == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _checkpoint_path(root: Path, rank: int, slot: int) -> Path:
    if slot not in {0, 1}:
        raise Top2000M03RV7WorkerError("checkpoint slot must be zero or one")
    return root / "checkpoints" / f"progress.slot-{slot}.rank-{rank:02d}.pt"


def _manifest_checkpoint_path(
    run_root: Path,
    context: _DistributedContext,
) -> Path | None:
    """Resolve only the checkpoint generation committed by the manifest."""

    manifest_path = run_root / "checkpoints" / "progress-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV7WorkerError("progress manifest cannot be read") from exc
    rows = manifest.get("rank_checkpoints")
    if not isinstance(rows, list) or len(rows) != context.world_size:
        raise Top2000M03RV7WorkerError("progress manifest rank inventory is invalid")
    matched = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("rank") == context.rank
    ]
    if len(matched) != 1 or not isinstance(matched[0].get("checkpoint"), str):
        raise Top2000M03RV7WorkerError("progress manifest omitted this rank")
    name = str(matched[0]["checkpoint"])
    if Path(name).name != name or not re.fullmatch(
        rf"progress\.slot-[01]\.rank-{context.rank:02d}\.pt", name
    ):
        raise Top2000M03RV7WorkerError("progress manifest checkpoint path is unsafe")
    return run_root / "checkpoints" / name


def _recorded_rank_peak(
    context: _DistributedContext,
    *,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    live = _rank_peak(context)
    live["peak_allocated_bytes"] = max(
        int(live["peak_allocated_bytes"]), peak_allocated_bytes
    )
    live["peak_reserved_bytes"] = max(
        int(live["peak_reserved_bytes"]), peak_reserved_bytes
    )
    return live


def _publish_checkpoint_manifest(
    run_root: Path,
    checkpoint_path: Path,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    plan_file_sha256: str,
    context: _DistributedContext,
    mode: str,
    cell_index: int,
    completed_steps: int,
) -> None:
    local = {
        "rank": context.rank,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    ranks = _gather_objects(local, context)
    if context.rank == 0:
        _atomic_write_json(
            run_root / "checkpoints" / "progress-manifest.json",
            {
                "schema": PROGRESS_MANIFEST_SCHEMA,
                "mode": mode,
                "protocol_sha256": plan.protocol_sha256,
                "plan_file_sha256": plan_file_sha256,
                "plan_receipt_sha256": plan.receipt_sha256,
                "cache_sha256": plan.cache_sha256,
                "setting_index": plan.setting_index,
                "setting_id": plan.setting_id,
                "world_size": context.world_size,
                "cell_index": cell_index,
                "completed_steps": completed_steps,
                "rank_checkpoints": ranks,
                "development_only": True,
                "promotion_eligible": False,
            },
        )
    _barrier(context)


def _verify_checkpoint_manifest(
    run_root: Path,
    checkpoint_path: Path,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    plan_file_sha256: str,
    context: _DistributedContext,
    mode: str,
    checkpoint: Mapping[str, Any] | None,
) -> None:
    local = None
    if checkpoint is not None:
        local = {
            "rank": context.rank,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
    ranks = _gather_objects(local, context)
    error: str | None = None
    if context.rank == 0:
        manifest_path = run_root / "checkpoints" / "progress-manifest.json"
        if checkpoint is None:
            if manifest_path.exists() or any(value is not None for value in ranks):
                error = "partial checkpoint set exists without a complete cursor"
        elif not manifest_path.exists():
            error = "checkpoint files exist without their atomic progress manifest"
        else:
            try:
                manifest = json.loads(manifest_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                error = f"progress manifest cannot be read: {type(exc).__name__}"
            else:
                expected = {
                    "schema": PROGRESS_MANIFEST_SCHEMA,
                    "mode": mode,
                    "protocol_sha256": plan.protocol_sha256,
                    "plan_file_sha256": plan_file_sha256,
                    "plan_receipt_sha256": plan.receipt_sha256,
                    "cache_sha256": plan.cache_sha256,
                    "setting_index": plan.setting_index,
                    "setting_id": plan.setting_id,
                    "world_size": context.world_size,
                    "cell_index": checkpoint["cell_index"],
                    "completed_steps": checkpoint["completed_steps"],
                    "rank_checkpoints": ranks,
                }
                for name, expected_value in expected.items():
                    if manifest.get(name) != expected_value:
                        error = f"progress manifest {name} does not match checkpoints"
                        break
    errors = [error]
    if context.world_size == 2:
        dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise Top2000M03RV7WorkerError(str(errors[0]))


def _checkpoint_payload(
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    plan_file_sha256: str,
    context: _DistributedContext,
    mode: str,
    checkpoint_slot: int,
    checkpoint_generation: int,
    cell_index: int,
    completed_steps: int,
    policy: Top2000M03RV7DevelopmentPolicy | None,
    optimizer: torch.optim.Optimizer | None,
    overlay_optimizer: torch.optim.Optimizer | None,
    last_metrics: Mapping[str, Any] | None,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    overlay_required = _requires_overlay_optimizer(plan)
    active_cell = policy is not None
    if active_cell != (optimizer is not None):
        raise Top2000M03RV7WorkerError(
            "checkpoint policy and alpha-core optimizer must appear together"
        )
    if active_cell and overlay_required != (overlay_optimizer is not None):
        raise Top2000M03RV7WorkerError(
            "checkpoint overlay optimizer does not match the setting route"
        )
    if not active_cell and overlay_optimizer is not None:
        raise Top2000M03RV7WorkerError(
            "cursor-only checkpoint cannot retain an overlay optimizer"
        )
    alpha_state = None if optimizer is None else optimizer.state_dict()
    overlay_state = (
        None if overlay_optimizer is None else overlay_optimizer.state_dict()
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "worker_schema": WORKER_SCHEMA,
        "mode": mode,
        "checkpoint_slot": checkpoint_slot,
        "checkpoint_generation": checkpoint_generation,
        "protocol_sha256": plan.protocol_sha256,
        "plan_file_sha256": plan_file_sha256,
        "plan_receipt_sha256": plan.receipt_sha256,
        "cache_sha256": plan.cache_sha256,
        "setting_index": plan.setting_index,
        "setting_id": plan.setting_id,
        "rank": context.rank,
        "world_size": context.world_size,
        "cell_index": cell_index,
        "completed_steps": completed_steps,
        "model_state_dict": (
            None
            if policy is None
            else {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
        ),
        "overlay_optimizer_required": overlay_required,
        "alpha_core_optimizer_state_dict": alpha_state,
        "alpha_core_optimizer_state_sha256": (
            None
            if alpha_state is None
            else optimizer_state_dict_sha256(alpha_state)
        ),
        "overlay_optimizer_state_dict": overlay_state,
        "overlay_optimizer_state_sha256": (
            None
            if overlay_state is None
            else optimizer_state_dict_sha256(overlay_state)
        ),
        "rng_state": _rank_rng_state(context.device),
        "last_metrics": None if last_metrics is None else dict(last_metrics),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "development_only": True,
        "promotion_eligible": False,
    }


def _load_checkpoint(
    path: Path,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    plan_file_sha256: str,
    context: _DistributedContext,
    mode: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Top2000M03RV7WorkerError(f"cannot load checkpoint {path}") from exc
    if not isinstance(value, dict):
        raise Top2000M03RV7WorkerError("checkpoint payload must be a mapping")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "worker_schema": WORKER_SCHEMA,
        "mode": mode,
        "protocol_sha256": plan.protocol_sha256,
        "plan_file_sha256": plan_file_sha256,
        "plan_receipt_sha256": plan.receipt_sha256,
        "cache_sha256": plan.cache_sha256,
        "setting_index": plan.setting_index,
        "setting_id": plan.setting_id,
        "rank": context.rank,
        "world_size": context.world_size,
        "development_only": True,
        "promotion_eligible": False,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise Top2000M03RV7WorkerError(
                f"checkpoint {name} does not match the current worker"
            )
    cell_index = value.get("cell_index")
    completed_steps = value.get("completed_steps")
    checkpoint_slot = value.get("checkpoint_slot")
    checkpoint_generation = value.get("checkpoint_generation")
    if (
        isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or cell_index < 0
        or isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or checkpoint_slot not in {0, 1}
        or isinstance(checkpoint_generation, bool)
        or not isinstance(checkpoint_generation, int)
        or checkpoint_generation <= 0
    ):
        raise Top2000M03RV7WorkerError("checkpoint cursor is invalid")
    if path != _checkpoint_path(path.parents[1], context.rank, checkpoint_slot):
        raise Top2000M03RV7WorkerError("checkpoint slot does not match its filename")
    overlay_required = _requires_overlay_optimizer(plan)
    if value.get("overlay_optimizer_required") is not overlay_required:
        raise Top2000M03RV7WorkerError(
            "checkpoint overlay-optimizer route does not match the setting"
        )
    model_state = value.get("model_state_dict")
    alpha_state = value.get("alpha_core_optimizer_state_dict")
    alpha_hash = value.get("alpha_core_optimizer_state_sha256")
    overlay_state = value.get("overlay_optimizer_state_dict")
    overlay_hash = value.get("overlay_optimizer_state_sha256")
    active_cell = model_state is not None
    if active_cell:
        if (
            not isinstance(model_state, dict)
            or not isinstance(alpha_state, Mapping)
            or not isinstance(alpha_hash, str)
            or optimizer_state_dict_sha256(alpha_state) != alpha_hash
        ):
            raise Top2000M03RV7WorkerError(
                "checkpoint alpha-core optimizer state is absent or hash-invalid"
            )
        if overlay_required:
            if (
                not isinstance(overlay_state, Mapping)
                or not isinstance(overlay_hash, str)
                or optimizer_state_dict_sha256(overlay_state) != overlay_hash
            ):
                raise Top2000M03RV7WorkerError(
                    "checkpoint A06 overlay optimizer state is absent or hash-invalid"
                )
        elif overlay_state is not None or overlay_hash is not None:
            raise Top2000M03RV7WorkerError(
                "non-A06 checkpoint contains a forbidden overlay optimizer"
            )
    elif any(
        item is not None
        for item in (alpha_state, alpha_hash, overlay_state, overlay_hash)
    ):
        raise Top2000M03RV7WorkerError(
            "cursor-only checkpoint contains orphan optimizer state"
        )
    return value


def _cell_inventory() -> tuple[tuple[int, int], ...]:
    return tuple(
        (fold_index, seed)
        for fold_index in range(6)
        for seed in TOP2000_M03R_V7_DEV_SEEDS
    )


def _build_episode(
    cache: Top2000VerifiedDevelopmentCache,
    *,
    start: int,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=start,
        state_stop_index_exclusive=(
            start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        output_device="cpu",
    )
    calibration = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=start,
        state_stop_index_exclusive=(
            start + TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS + 1
        ),
        max_state_rows=TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS + 1,
        output_device="cpu",
    )
    fitted = fit_top2000_m03r_v7_warmup_factor_calibration(
        calibration,
        built,
    )
    return built, _move_sequence_to_device(built.sequence, device), fitted


def _build_validation_episode(
    cache: Top2000VerifiedDevelopmentCache,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    device: torch.device,
) -> tuple[Any, Any, Any, int, int]:
    """Build the sole 378-state validation chronology for one fold.

    The first 251 transitions are causal context/carry-in, the next 63 are the
    immutable scored decisions, and the final 63 are unscored auxiliary and
    holding support.  The slice includes one final state-only terminal row.
    """

    start = fold.validation_decision_start - 251
    expected_stop = fold.validation_decision_stop_exclusive + 64
    if (
        start < 0
        or expected_stop - start != TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    ):
        raise Top2000M03RV7WorkerError(
            "validation fold cannot supply the frozen 378-state chronology"
        )
    built, sequence, factor_calibration = _build_episode(
        cache,
        start=start,
        device=device,
    )
    if built.identity.state_stop_index_exclusive != expected_stop:
        raise Top2000M03RV7WorkerError(
            "validation episode does not end after its 63-transition support tail"
        )
    score_start, score_stop = validate_fold_score_bounds(
        fold,
        sequence_global_state_start=start,
        sequence_state_rows=sequence.n_positions,
    )
    return built, sequence, factor_calibration, score_start, score_stop


def _validation_trace_artifact(
    run_root: Path,
    *,
    fold_index: int,
    seed: int | None,
) -> Path:
    label = "ensemble" if seed is None else f"seed-{seed}"
    return (
        run_root
        / "validation-traces"
        / f"fold-{fold_index:02d}-{label}.pt"
    )


def _seed_validation_receipt_path(
    run_root: Path,
    *,
    fold_index: int,
    seed: int,
) -> Path:
    return (
        run_root
        / "receipts"
        / SEED_VALIDATION_RECEIPT_DIRECTORY
        / f"fold-{fold_index:02d}-seed-{seed}.json"
    )


def _fold_ensemble_receipt_path(
    run_root: Path,
    *,
    fold_index: int,
) -> Path:
    return (
        run_root
        / "receipts"
        / FOLD_ENSEMBLE_RECEIPT_DIRECTORY
        / f"fold-{fold_index:02d}.json"
    )


def _evaluate_seed_checkpoint(
    cache: Top2000VerifiedDevelopmentCache,
    fold: Top2000M03RV7DevelopmentFold,
    policy: Top2000M03RV7DevelopmentPolicy,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    run_root: Path,
    seed: int,
    checkpoint_file_sha256: str,
    device: torch.device,
) -> tuple[Path, str]:
    """Evaluate one detached final checkpoint and publish bound evidence."""

    state_sha256 = model_state_sha256(policy)
    receipt_path = _seed_validation_receipt_path(
        run_root,
        fold_index=fold.fold_index,
        seed=seed,
    )
    artifact_path = _validation_trace_artifact(
        run_root,
        fold_index=fold.fold_index,
        seed=seed,
    )
    if receipt_path.exists():
        try:
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Top2000M03RV7WorkerError(
                "prior seed validation receipt cannot be read"
            ) from exc
        expected = {
            "schema": "rl-quant.top2000-dev.m03r-v7-seed-validation-v1",
            "protocol_sha256": plan.protocol_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold.fold_index,
            "seed": seed,
            "checkpoint_selection_rule": (
                "frozen-final-optimizer-update-no-validation-selection-v1"
            ),
            "evaluation_autograd_enabled": False,
            "fold_receipt_sha256": fold.receipt_sha256,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "model_state_sha256": state_sha256,
            "development_only": True,
            "promotion_eligible": False,
        }
        artifact_hash = (
            prior.get("validation_trace_artifact_sha256")
            if isinstance(prior, dict)
            else None
        )
        if (
            not isinstance(prior, dict)
            or any(prior.get(name) != value for name, value in expected.items())
            or not isinstance(artifact_hash, str)
            or not artifact_path.is_file()
            or _sha256_file(artifact_path) != artifact_hash
        ):
            raise Top2000M03RV7WorkerError(
                "prior seed validation receipt or trace artifact drifted"
            )
        return receipt_path, _sha256_file(receipt_path)

    built, sequence, calibration, score_start, score_stop = (
        _build_validation_episode(cache, fold, device=device)
    )
    policy.bind_episode_factor_loadings(calibration.loadings)
    with torch.no_grad():
        evidence = evaluate_top2000_m03r_v7_validation_trace(
            policy,
            sequence,
            score_transition_start=score_start,
            score_transition_stop_exclusive=score_stop,
        )
    artifact_sha256 = _atomic_torch_save(
        artifact_path,
        {
            **evidence.artifact_payload(),
            "protocol_sha256": plan.protocol_sha256,
            "plan_receipt_sha256": plan.receipt_sha256,
            "cache_sha256": plan.cache_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold.fold_index,
            "fold_receipt_sha256": fold.receipt_sha256,
            "seed": seed,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "model_state_sha256": state_sha256,
            "sequence_receipt_sha256": built.identity.receipt_sha256,
            "development_only": True,
            "promotion_eligible": False,
        },
    )
    receipt = Top2000M03RV7SeedValidationReceipt(
        setting_index=plan.setting_index,
        setting_id=plan.setting_id,
        fold_index=fold.fold_index,
        seed=seed,
        fold_receipt_sha256=fold.receipt_sha256,
        sequence_receipt_sha256=built.identity.receipt_sha256,
        checkpoint_file_sha256=checkpoint_file_sha256,
        model_state_sha256=state_sha256,
        validation_trace_artifact_sha256=artifact_sha256,
        validation_trace_sha256=evidence.trace_sha256,
        array_sha256=evidence.array_sha256s(),
        metrics=evidence.metrics(),
        validation_global_decision_start=fold.validation_decision_start,
        validation_global_decision_stop_exclusive=(
            fold.validation_decision_stop_exclusive
        ),
        first_validation_date=built.exchange_dates[score_start],
        last_validation_date=built.exchange_dates[score_stop - 1],
    )
    receipt_sha256 = _write_immutable_json(receipt_path, asdict(receipt))
    return receipt_path, receipt_sha256


def _load_saved_seed_policy(
    model_path: Path,
    *,
    expected_file_sha256: str,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    fold: Top2000M03RV7DevelopmentFold,
    seed: int,
    device: torch.device,
) -> tuple[Top2000M03RV7DevelopmentPolicy, str]:
    if _sha256_file(model_path) != expected_file_sha256:
        raise Top2000M03RV7WorkerError(
            "ensemble member model no longer matches its cell receipt"
        )
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Top2000M03RV7WorkerError(
            "cannot load an ensemble member checkpoint"
        ) from exc
    expected = {
        "schema": CELL_MODEL_SCHEMA,
        "protocol_sha256": plan.protocol_sha256,
        "plan_receipt_sha256": plan.receipt_sha256,
        "cache_sha256": plan.cache_sha256,
        "setting_index": plan.setting_index,
        "setting_id": plan.setting_id,
        "fold_index": fold.fold_index,
        "fold_receipt_sha256": fold.receipt_sha256,
        "seed": seed,
        "rank": 0,
        "overlay_optimizer_required": _requires_overlay_optimizer(plan),
        "development_only": True,
        "promotion_eligible": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(name) != value for name, value in expected.items()
    ):
        raise Top2000M03RV7WorkerError(
            "ensemble member checkpoint identity drifted"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise Top2000M03RV7WorkerError(
            "ensemble member checkpoint omitted its model state"
        )
    alpha_state = payload.get("alpha_core_optimizer_state_dict")
    alpha_hash = payload.get("alpha_core_optimizer_state_sha256")
    overlay_state = payload.get("overlay_optimizer_state_dict")
    overlay_hash = payload.get("overlay_optimizer_state_sha256")
    if (
        not isinstance(alpha_state, Mapping)
        or not isinstance(alpha_hash, str)
        or optimizer_state_dict_sha256(alpha_state) != alpha_hash
    ):
        raise Top2000M03RV7WorkerError(
            "ensemble member alpha-core optimizer binding drifted"
        )
    if _requires_overlay_optimizer(plan):
        if (
            not isinstance(overlay_state, Mapping)
            or not isinstance(overlay_hash, str)
            or optimizer_state_dict_sha256(overlay_state) != overlay_hash
        ):
            raise Top2000M03RV7WorkerError(
                "A06 ensemble member overlay optimizer binding drifted"
            )
    elif overlay_state is not None or overlay_hash is not None:
        raise Top2000M03RV7WorkerError(
            "non-A06 ensemble member contains an overlay optimizer"
        )
    policy = Top2000M03RV7DevelopmentPolicy(
        plan.setting_id,
        token_dim=plan.token_dim,
        raw_stock_chunk=plan.raw_stock_chunk,
        activation_checkpointing=plan.activation_checkpointing,
    ).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, model_state_sha256(policy)


def _evaluate_fold_ensemble(
    cache: Top2000VerifiedDevelopmentCache,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    run_root: Path,
    device: torch.device,
) -> tuple[Path, str]:
    """Construct and execute one five-seed output-space fold portfolio."""

    receipt_path = _fold_ensemble_receipt_path(
        run_root,
        fold_index=fold.fold_index,
    )
    seed_receipt_paths = tuple(
        _seed_validation_receipt_path(
            run_root,
            fold_index=fold.fold_index,
            seed=seed,
        )
        for seed in TOP2000_M03R_V7_DEV_SEEDS
    )
    if not all(path.is_file() for path in seed_receipt_paths):
        raise Top2000M03RV7WorkerError(
            "fold ensemble requires all five seed validation receipts"
        )
    seed_receipt_sha256s = tuple(_sha256_file(path) for path in seed_receipt_paths)
    seed_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in seed_receipt_paths
    )
    members: list[Top2000M03RV7DevelopmentPolicy] = []
    checkpoint_hashes: list[str] = []
    model_state_hashes: list[str] = []
    for seed, seed_payload in zip(
        TOP2000_M03R_V7_DEV_SEEDS,
        seed_payloads,
        strict=True,
    ):
        if (
            not isinstance(seed_payload, dict)
            or seed_payload.get("seed") != seed
            or seed_payload.get("fold_index") != fold.fold_index
            or seed_payload.get("setting_id") != plan.setting_id
        ):
            raise Top2000M03RV7WorkerError(
                "seed validation receipt cannot enter this fold ensemble"
            )
        checkpoint_hash = str(seed_payload.get("checkpoint_file_sha256"))
        _require_sha256("checkpoint_file_sha256", checkpoint_hash)
        model_path = (
            run_root
            / "cells"
            / f"fold-{fold.fold_index:02d}-seed-{seed}"
            / "model.rank-00.pt"
        )
        member, observed_state_hash = _load_saved_seed_policy(
            model_path,
            expected_file_sha256=checkpoint_hash,
            plan=plan,
            fold=fold,
            seed=seed,
            device=device,
        )
        if observed_state_hash != seed_payload.get("model_state_sha256"):
            raise Top2000M03RV7WorkerError(
                "seed validation receipt model-state binding drifted"
            )
        members.append(member)
        checkpoint_hashes.append(checkpoint_hash)
        model_state_hashes.append(observed_state_hash)

    built, sequence, calibration, score_start, score_stop = (
        _build_validation_episode(cache, fold, device=device)
    )
    ensemble = Top2000M03RV7OutputSpaceEnsemblePolicy(members).to(device)
    ensemble.bind_episode_factor_loadings(calibration.loadings)
    with torch.no_grad():
        evidence = evaluate_top2000_m03r_v7_validation_trace(
            ensemble,
            sequence,
            score_transition_start=score_start,
            score_transition_stop_exclusive=score_stop,
        )
    artifact_path = _validation_trace_artifact(
        run_root,
        fold_index=fold.fold_index,
        seed=None,
    )
    artifact_sha256 = _atomic_torch_save(
        artifact_path,
        {
            **evidence.artifact_payload(),
            "protocol_sha256": plan.protocol_sha256,
            "plan_receipt_sha256": plan.receipt_sha256,
            "cache_sha256": plan.cache_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold.fold_index,
            "fold_receipt_sha256": fold.receipt_sha256,
            "ordered_seeds": list(TOP2000_M03R_V7_DEV_SEEDS),
            "seed_validation_receipt_sha256s": list(seed_receipt_sha256s),
            "member_checkpoint_file_sha256s": checkpoint_hashes,
            "member_model_state_sha256s": model_state_hashes,
            "output_space_ensemble": True,
            "seed_return_paths_averaged": False,
            "development_only": True,
            "promotion_eligible": False,
        },
    )
    receipt = Top2000M03RV7FoldEnsembleReceipt(
        setting_index=plan.setting_index,
        setting_id=plan.setting_id,
        fold_index=fold.fold_index,
        fold_receipt_sha256=fold.receipt_sha256,
        ordered_seeds=TOP2000_M03R_V7_DEV_SEEDS,
        seed_validation_receipt_sha256s=seed_receipt_sha256s,
        member_checkpoint_file_sha256s=tuple(checkpoint_hashes),
        member_model_state_sha256s=tuple(model_state_hashes),
        sequence_receipt_sha256=built.identity.receipt_sha256,
        validation_trace_artifact_sha256=artifact_sha256,
        validation_trace_sha256=evidence.trace_sha256,
        array_sha256=evidence.array_sha256s(),
        metrics=evidence.metrics(),
        validation_global_decision_start=fold.validation_decision_start,
        validation_global_decision_stop_exclusive=(
            fold.validation_decision_stop_exclusive
        ),
        first_validation_date=built.exchange_dates[score_start],
        last_validation_date=built.exchange_dates[score_stop - 1],
    )
    receipt_sha256 = _write_immutable_json(receipt_path, asdict(receipt))
    return receipt_path, receipt_sha256


def _collect_full_validation_evidence(
    run_root: Path,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    folds: Sequence[Top2000M03RV7DevelopmentFold],
    cells: Sequence[tuple[int, int]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Fail closed unless all 30 seed and six ensemble receipts reconcile."""

    expected_seed_paths = {
        _seed_validation_receipt_path(
            run_root,
            fold_index=fold_index,
            seed=seed,
        )
        for fold_index, seed in cells
    }
    observed_seed_paths = set(
        (
            run_root / "receipts" / SEED_VALIDATION_RECEIPT_DIRECTORY
        ).glob("fold-*-seed-*.json")
    )
    if observed_seed_paths != expected_seed_paths:
        raise Top2000M03RV7WorkerError(
            "full completion requires the exact thirty seed validation receipts"
        )
    seed_hashes: dict[tuple[int, int], str] = {}
    for fold_index, seed in cells:
        path = _seed_validation_receipt_path(
            run_root,
            fold_index=fold_index,
            seed=seed,
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Top2000M03RV7WorkerError(
                "seed validation receipt cannot be read"
            ) from exc
        expected = {
            "schema": "rl-quant.top2000-dev.m03r-v7-seed-validation-v1",
            "protocol_sha256": plan.protocol_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold_index,
            "seed": seed,
            "checkpoint_selection_rule": (
                "frozen-final-optimizer-update-no-validation-selection-v1"
            ),
            "evaluation_autograd_enabled": False,
            "development_only": True,
            "future_selected_universe": True,
            "outer_evaluation_authorized": False,
            "promotion_eligible": False,
        }
        if not isinstance(payload, dict) or any(
            payload.get(name) != value for name, value in expected.items()
        ):
            raise Top2000M03RV7WorkerError(
                "seed validation receipt identity drifted"
            )
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("decision_count") != 63:
            raise Top2000M03RV7WorkerError(
                "seed validation receipt does not contain 63 decisions"
            )
        cell_path = (
            run_root / "receipts" / f"fold-{fold_index:02d}-seed-{seed}.json"
        )
        try:
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Top2000M03RV7WorkerError(
                "cell receipt cannot bind seed validation evidence"
            ) from exc
        digest = _sha256_file(path)
        if (
            not isinstance(cell, dict)
            or cell.get("schema") != CELL_RECEIPT_SCHEMA
            or cell.get("overlay_optimizer_required")
            is not _requires_overlay_optimizer(plan)
            or cell.get("alpha_core_optimizer_steps")
            != plan.total_optimizer_steps_per_fold_seed
            or cell.get("overlay_optimizer_steps")
            != (
                plan.total_optimizer_steps_per_fold_seed
                if _requires_overlay_optimizer(plan)
                else 0
            )
            or cell.get("seed_validation_required") is not True
            or cell.get("seed_validation_receipt_sha256") != digest
        ):
            raise Top2000M03RV7WorkerError(
                "cell receipt does not bind its seed validation evidence"
            )
        seed_hashes[(fold_index, seed)] = digest

    expected_fold_paths = {
        _fold_ensemble_receipt_path(run_root, fold_index=fold.fold_index)
        for fold in folds
    }
    observed_fold_paths = set(
        (
            run_root / "receipts" / FOLD_ENSEMBLE_RECEIPT_DIRECTORY
        ).glob("fold-*.json")
    )
    if observed_fold_paths != expected_fold_paths:
        raise Top2000M03RV7WorkerError(
            "full completion requires exactly six fold ensemble receipts"
        )
    fold_hashes: dict[str, str] = {}
    for fold in folds:
        path = _fold_ensemble_receipt_path(
            run_root,
            fold_index=fold.fold_index,
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Top2000M03RV7WorkerError(
                "fold ensemble receipt cannot be read"
            ) from exc
        expected_seed_hashes = [
            seed_hashes[(fold.fold_index, seed)]
            for seed in TOP2000_M03R_V7_DEV_SEEDS
        ]
        if (
            not isinstance(payload, dict)
            or payload.get("schema")
            != "rl-quant.top2000-dev.m03r-v7-fold-ensemble-v1"
            or payload.get("protocol_sha256") != plan.protocol_sha256
            or payload.get("setting_index") != plan.setting_index
            or payload.get("setting_id") != plan.setting_id
            or payload.get("fold_index") != fold.fold_index
            or payload.get("fold_receipt_sha256") != fold.receipt_sha256
            or payload.get("ordered_seeds") != list(TOP2000_M03R_V7_DEV_SEEDS)
            or payload.get("seed_validation_receipt_sha256s")
            != expected_seed_hashes
            or payload.get("seeds_are_independent_return_paths") is not False
            or payload.get("chronological_return_path_count") != 1
            or payload.get("evaluation_autograd_enabled") is not False
            or payload.get("development_only") is not True
            or payload.get("promotion_eligible") is not False
            or not isinstance(payload.get("metrics"), dict)
            or payload["metrics"].get("decision_count") != 63
        ):
            raise Top2000M03RV7WorkerError(
                "fold ensemble receipt identity or five-seed binding drifted"
            )
        fold_hashes[str(path.relative_to(run_root))] = _sha256_file(path)
    return (
        {
            str(
                _seed_validation_receipt_path(
                    run_root,
                    fold_index=fold_index,
                    seed=seed,
                ).relative_to(run_root)
            ): digest
            for (fold_index, seed), digest in seed_hashes.items()
        },
        fold_hashes,
    )


def _new_cell(
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    *,
    seed: int,
    device: torch.device,
) -> tuple[
    Top2000M03RV7DevelopmentPolicy,
    torch.optim.Optimizer,
    torch.optim.Optimizer | None,
]:
    _seed_everything(seed)
    policy = Top2000M03RV7DevelopmentPolicy(
        plan.setting_id,
        token_dim=plan.token_dim,
        raw_stock_chunk=plan.raw_stock_chunk,
        activation_checkpointing=plan.activation_checkpointing,
    ).to(device)
    optimizer, overlay_optimizer = (
        build_top2000_m03r_v7_development_optimizers(
            policy,
            learning_rate=plan.learning_rate,
            weight_decay=plan.weight_decay,
        )
    )
    return policy, optimizer, overlay_optimizer


def _restore_cell(
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    *,
    seed: int,
    device: torch.device,
    checkpoint: Mapping[str, Any],
) -> tuple[
    Top2000M03RV7DevelopmentPolicy,
    torch.optim.Optimizer,
    torch.optim.Optimizer | None,
]:
    policy, optimizer, overlay_optimizer = _new_cell(
        plan,
        seed=seed,
        device=device,
    )
    model_state = checkpoint.get("model_state_dict")
    optimizer_state = checkpoint.get("alpha_core_optimizer_state_dict")
    optimizer_hash = checkpoint.get("alpha_core_optimizer_state_sha256")
    overlay_state = checkpoint.get("overlay_optimizer_state_dict")
    overlay_hash = checkpoint.get("overlay_optimizer_state_sha256")
    if (
        not isinstance(model_state, dict)
        or not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_hash, str)
        or optimizer_state_dict_sha256(optimizer_state) != optimizer_hash
    ):
        raise Top2000M03RV7WorkerError(
            "in-progress checkpoint omitted bound alpha-core state"
        )
    policy.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    _optimizer_to_device(optimizer, device)
    if optimizer_state_sha256(optimizer) != optimizer_hash:
        raise Top2000M03RV7WorkerError(
            "restored alpha-core optimizer state changed content"
        )
    if overlay_optimizer is None:
        if overlay_state is not None or overlay_hash is not None:
            raise Top2000M03RV7WorkerError(
                "non-A06 restore received overlay optimizer state"
            )
    else:
        if (
            not isinstance(overlay_state, dict)
            or not isinstance(overlay_hash, str)
            or optimizer_state_dict_sha256(overlay_state) != overlay_hash
        ):
            raise Top2000M03RV7WorkerError(
                "A06 checkpoint omitted bound overlay optimizer state"
            )
        overlay_optimizer.load_state_dict(overlay_state)
        _optimizer_to_device(overlay_optimizer, device)
        if optimizer_state_sha256(overlay_optimizer) != overlay_hash:
            raise Top2000M03RV7WorkerError(
                "restored A06 overlay optimizer state changed content"
            )
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise Top2000M03RV7WorkerError("checkpoint omitted rank RNG state")
    _restore_rank_rng_state(rng_state, device)
    return policy, optimizer, overlay_optimizer


def _rank_peak(context: _DistributedContext) -> dict[str, Any]:
    if context.device.type != "cuda":
        return {
            "rank": context.rank,
            "device": "cpu",
            "gpu_name": None,
            "gpu_total_memory_bytes": 0,
            "compute_capability": None,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "allocator_oom_count": 0,
            "allocator_retry_count": 0,
            "torchrun_restart_count": int(
                os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": None,
            "nccl_version": None,
        }
    properties = torch.cuda.get_device_properties(context.device)
    memory_stats = torch.cuda.memory_stats(context.device)
    nccl_version = torch.cuda.nccl.version()  # type: ignore[no-untyped-call]
    return {
        "rank": context.rank,
        "device": str(context.device),
        "gpu_name": torch.cuda.get_device_name(context.device),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(context.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(context.device),
        "allocator_oom_count": int(memory_stats.get("num_ooms", 0)),
        "allocator_retry_count": int(memory_stats.get("num_alloc_retries", 0)),
        "torchrun_restart_count": int(
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "nccl_version": (
            list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version
        ),
    }


def _existing_cell_receipt_status(
    receipt_path: Path,
    model_directory: Path,
    *,
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    plan_file_sha256: str,
    context: _DistributedContext,
    mode: str,
    cell_index: int,
    fold_index: int,
    seed: int,
    completed_steps: int,
) -> bool:
    """Return true only for a complete, hash-reconciled prior finalization."""

    status: dict[str, Any] = {"exists": False, "error": None}
    if context.rank == 0 and receipt_path.exists():
        try:
            payload = json.loads(receipt_path.read_bytes())
            expected = {
                "schema": CELL_RECEIPT_SCHEMA,
                "mode": mode,
                "protocol_sha256": plan.protocol_sha256,
                "plan_file_sha256": plan_file_sha256,
                "plan_receipt_sha256": plan.receipt_sha256,
                "cache_sha256": plan.cache_sha256,
                "setting_index": plan.setting_index,
                "setting_id": plan.setting_id,
                "cell_index": cell_index,
                "fold_index": fold_index,
                "seed": seed,
                "optimizer_steps": completed_steps,
                "overlay_optimizer_required": _requires_overlay_optimizer(plan),
                "alpha_core_optimizer_steps": completed_steps,
                "overlay_optimizer_steps": (
                    completed_steps if _requires_overlay_optimizer(plan) else 0
                ),
            }
            for name, expected_value in expected.items():
                if payload.get(name) != expected_value:
                    raise Top2000M03RV7WorkerError(
                        f"existing cell receipt {name} drifted"
                    )
            hashes = payload.get("rank_model_sha256")
            if not isinstance(hashes, list) or len(hashes) != context.world_size:
                raise Top2000M03RV7WorkerError(
                    "existing cell receipt model inventory drifted"
                )
            alpha_hashes = payload.get(
                "rank_alpha_core_optimizer_state_sha256"
            )
            overlay_hashes = payload.get(
                "rank_overlay_optimizer_state_sha256"
            )
            if (
                not isinstance(alpha_hashes, list)
                or len(alpha_hashes) != context.world_size
                or any(
                    not isinstance(value, str)
                    or _SHA256_RE.fullmatch(value) is None
                    for value in alpha_hashes
                )
                or len(set(alpha_hashes)) != 1
                or not isinstance(overlay_hashes, list)
                or len(overlay_hashes) != context.world_size
            ):
                raise Top2000M03RV7WorkerError(
                    "existing cell optimizer-state inventory drifted"
                )
            overlay_required = _requires_overlay_optimizer(plan)
            if overlay_required:
                if any(
                    not isinstance(value, str)
                    or _SHA256_RE.fullmatch(value) is None
                    for value in overlay_hashes
                ) or len(set(overlay_hashes)) != 1:
                    raise Top2000M03RV7WorkerError(
                        "A06 cell receipt omitted overlay optimizer hashes"
                    )
            elif any(value is not None for value in overlay_hashes) or len(
                set(overlay_hashes)
            ) != 1:
                raise Top2000M03RV7WorkerError(
                    "non-A06 cell receipt contains overlay optimizer hashes"
                )
            for rank, expected_hash in enumerate(hashes):
                model_path = model_directory / f"model.rank-{rank:02d}.pt"
                if (
                    not isinstance(expected_hash, str)
                    or not model_path.exists()
                    or _sha256_file(model_path) != expected_hash
                ):
                    raise Top2000M03RV7WorkerError(
                        "existing cell model no longer matches its receipt"
                    )
                try:
                    model_payload = torch.load(
                        model_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except Exception as exc:
                    raise Top2000M03RV7WorkerError(
                        "existing cell model payload cannot be read"
                    ) from exc
                alpha_state = (
                    model_payload.get("alpha_core_optimizer_state_dict")
                    if isinstance(model_payload, dict)
                    else None
                )
                overlay_state = (
                    model_payload.get("overlay_optimizer_state_dict")
                    if isinstance(model_payload, dict)
                    else None
                )
                if (
                    not isinstance(model_payload, dict)
                    or model_payload.get("schema") != CELL_MODEL_SCHEMA
                    or model_payload.get("overlay_optimizer_required")
                    is not overlay_required
                    or not isinstance(alpha_state, Mapping)
                    or optimizer_state_dict_sha256(alpha_state)
                    != alpha_hashes[rank]
                    or model_payload.get("alpha_core_optimizer_state_sha256")
                    != alpha_hashes[rank]
                ):
                    raise Top2000M03RV7WorkerError(
                        "existing cell alpha-core optimizer binding drifted"
                    )
                if overlay_required:
                    if (
                        not isinstance(overlay_state, Mapping)
                        or optimizer_state_dict_sha256(overlay_state)
                        != overlay_hashes[rank]
                        or model_payload.get("overlay_optimizer_state_sha256")
                        != overlay_hashes[rank]
                    ):
                        raise Top2000M03RV7WorkerError(
                            "existing A06 overlay optimizer binding drifted"
                        )
                elif (
                    overlay_state is not None
                    or model_payload.get("overlay_optimizer_state_sha256")
                    is not None
                ):
                    raise Top2000M03RV7WorkerError(
                        "existing non-A06 model contains an overlay optimizer"
                    )
            state_hashes = payload.get("rank_model_state_sha256")
            if (
                not isinstance(state_hashes, list)
                or len(state_hashes) != context.world_size
                or len(set(state_hashes)) != 1
                or any(
                    not isinstance(value, str)
                    or _SHA256_RE.fullmatch(value) is None
                    for value in state_hashes
                )
            ):
                raise Top2000M03RV7WorkerError(
                    "existing cell rank model-state inventory drifted"
                )
            validation_required = mode == "full"
            if payload.get("seed_validation_required") is not validation_required:
                raise Top2000M03RV7WorkerError(
                    "existing cell seed-validation requirement drifted"
                )
            if validation_required:
                validation_path = _seed_validation_receipt_path(
                    receipt_path.parents[1],
                    fold_index=fold_index,
                    seed=seed,
                )
                expected_validation_hash = payload.get(
                    "seed_validation_receipt_sha256"
                )
                if (
                    not isinstance(expected_validation_hash, str)
                    or not validation_path.is_file()
                    or _sha256_file(validation_path) != expected_validation_hash
                ):
                    raise Top2000M03RV7WorkerError(
                        "existing cell seed-validation evidence no longer reconciles"
                    )
            status["exists"] = True
        except (OSError, json.JSONDecodeError, Top2000M03RV7WorkerError) as exc:
            status["error"] = str(exc)
    values = [status]
    if context.world_size == 2:
        dist.broadcast_object_list(values, src=0)
    resolved = values[0]
    if not isinstance(resolved, dict) or resolved.get("error") is not None:
        raise Top2000M03RV7WorkerError(
            "existing cell finalization is invalid: "
            + str(resolved.get("error") if isinstance(resolved, dict) else resolved)
        )
    return bool(resolved.get("exists"))


def run_worker(
    plan: Top2000M03RV7DevelopmentTrainingPlan,
    *,
    plan_file_sha256: str,
    qualification_only: bool,
    qualification_steps: int = 1,
    qualification_restart_after_step1: bool = False,
) -> dict[str, Any] | None:
    """Run one setting, returning the rank-zero terminal receipt."""

    run_started = time.perf_counter()
    if isinstance(qualification_steps, bool) or not (
        1 <= qualification_steps <= 4
        or qualification_steps == MAX_EXTENDED_QUALIFICATION_STEPS
    ):
        raise Top2000M03RV7WorkerError(
            "qualification_steps must lie in [1, 4] or equal the approved "
            f"extended sentinel length {MAX_EXTENDED_QUALIFICATION_STEPS}"
        )
    extended_sentinel = qualification_steps == MAX_EXTENDED_QUALIFICATION_STEPS
    if extended_sentinel and (
        not qualification_only
        or qualification_restart_after_step1
        or plan.setting_index != 3
        or plan.setting_id != M03R_TOP2000_DEV_SETTING_IDS[3]
    ):
        raise Top2000M03RV7WorkerError(
            "the 20-update sentinel is restricted to qualification-only A08 "
            "without intentional restart"
        )
    if qualification_restart_after_step1 and (
        not qualification_only or qualification_steps != 4
    ):
        raise Top2000M03RV7WorkerError(
            "intentional restart qualification requires exactly four updates"
        )
    restart_count = (
        _torchrun_restart_count() if qualification_restart_after_step1 else 0
    )
    context = _distributed_context(qualification_only=qualification_only)
    mode = (
        "qualification-extended-a08-20"
        if extended_sentinel
        else "qualification"
        if qualification_only
        else "full"
    )
    try:
        if not qualification_only and context.world_size != plan.expected_world_size:
            raise Top2000M03RV7WorkerError(
                "runtime world size does not match the two-rank training plan"
            )
        output_root = Path(plan.output_root)
        run_root = output_root / (
            "qualification" if qualification_only else "training"
        )
        previous_restart_rendezvous: Path | None = None
        if qualification_restart_after_step1 and restart_count == 1:
            previous_restart_rendezvous = _intentional_restart_rendezvous_path(
                run_root,
                restart_count=0,
            )
            _wait_for_destroyed_process_group_markers(
                previous_restart_rendezvous,
                rendezvous_key=previous_restart_rendezvous.name,
                world_size=context.world_size,
            )
        if context.rank == 0:
            run_root.mkdir(parents=True, exist_ok=True)
        _barrier(context)
        if context.rank == 0 and previous_restart_rendezvous is not None:
            _cleanup_intentional_restart_rendezvous(previous_restart_rendezvous)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        cache = load_verified_top2000_hold30_development_cache(
            plan.cache_path,
            expected_cache_sha256=plan.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        if cache.daily_ohlcv.shape[0] != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
            raise Top2000M03RV7WorkerError(
                "verified cache no longer has the frozen 1001-state geometry"
            )
        folds = render_top2000_m03r_v7_development_folds(
            cache.daily_ohlcv.shape[0]
        )
        cells = _cell_inventory()
        checkpoint_path = _manifest_checkpoint_path(run_root, context)
        checkpoint = _load_checkpoint(
            _checkpoint_path(run_root, context.rank, 0)
            if checkpoint_path is None
            else checkpoint_path,
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            context=context,
            mode=mode,
        )
        resumed_from_checkpoint = checkpoint is not None
        resume_completed_steps = (
            None if checkpoint is None else int(checkpoint["completed_steps"])
        )
        _verify_checkpoint_manifest(
            run_root,
            _checkpoint_path(run_root, context.rank, 0)
            if checkpoint_path is None
            else checkpoint_path,
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            context=context,
            mode=mode,
            checkpoint=checkpoint,
        )
        cell_index = 0 if checkpoint is None else int(checkpoint["cell_index"])
        completed_steps = 0 if checkpoint is None else int(checkpoint["completed_steps"])
        checkpoint_slot = 1 if checkpoint is None else int(checkpoint["checkpoint_slot"])
        checkpoint_generation = (
            0 if checkpoint is None else int(checkpoint["checkpoint_generation"])
        )
        rank_peak_allocated = (
            0 if checkpoint is None else int(checkpoint["peak_allocated_bytes"])
        )
        rank_peak_reserved = (
            0 if checkpoint is None else int(checkpoint["peak_reserved_bytes"])
        )
        final_cell_count = 1 if qualification_only else len(cells)
        effective_steps = (
            min(plan.total_optimizer_steps_per_fold_seed, qualification_steps)
            if qualification_only
            else plan.total_optimizer_steps_per_fold_seed
        )
        if cell_index > final_cell_count or completed_steps > effective_steps:
            raise Top2000M03RV7WorkerError("checkpoint cursor exceeds this run mode")

        while cell_index < final_cell_count:
            fold_index, seed = cells[cell_index]
            fold = folds[fold_index]
            if completed_steps:
                if checkpoint is None:
                    raise Top2000M03RV7WorkerError("resume cursor lacks checkpoint")
                policy, optimizer, overlay_optimizer = _restore_cell(
                    plan,
                    seed=seed,
                    device=context.device,
                    checkpoint=checkpoint,
                )
                last_metrics = checkpoint.get("last_metrics")
            else:
                policy, optimizer, overlay_optimizer = _new_cell(
                    plan,
                    seed=seed,
                    device=context.device,
                )
                last_metrics = None

            while completed_steps < effective_steps:
                episode_start = deterministic_episode_start(
                    episode_schedule_sha256=plan.episode_schedule_sha256,
                    fold=fold,
                    seed=seed,
                    optimizer_step=completed_steps,
                )
                built, sequence, factor_calibration = _build_episode(
                    cache,
                    start=episode_start,
                    device=context.device,
                )
                bound, provider = bind_top2000_m03r_v7_runtime_sequence(
                    sequence,
                    policy,
                )
                policy.bind_episode_factor_loadings(factor_calibration.loadings)
                metrics = train_top2000_m03r_v7_development_update(
                    policy,
                    bound,
                    provider,
                    optimizer,
                    overlay_optimizer=overlay_optimizer,
                    completed_optimizer_steps=completed_steps,
                    total_optimizer_steps=plan.total_optimizer_steps_per_fold_seed,
                    max_origin_batch=plan.max_origin_batch,
                    grad_clip=plan.grad_clip,
                    distributed_rank=context.rank,
                    distributed_world_size=context.world_size,
                )
                completed_steps += 1
                last_metrics = {
                    **metrics,
                    "episode_state_start": episode_start,
                    "episode_state_stop_exclusive": (
                        episode_start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
                    ),
                    "episode_identity_sha256": built.identity.receipt_sha256,
                    "episode_schedule_sha256": plan.episode_schedule_sha256,
                    "factor_calibration_receipt_sha256": (
                        factor_calibration.receipt_sha256
                    ),
                }
                peak = _rank_peak(context)
                rank_peak_allocated = max(
                    rank_peak_allocated,
                    int(peak["peak_allocated_bytes"]),
                )
                rank_peak_reserved = max(
                    rank_peak_reserved,
                    int(peak["peak_reserved_bytes"]),
                )
                _atomic_torch_save(
                    _checkpoint_path(
                        run_root,
                        context.rank,
                        1 - checkpoint_slot,
                    ),
                    _checkpoint_payload(
                        plan=plan,
                        plan_file_sha256=plan_file_sha256,
                        context=context,
                        mode=mode,
                        checkpoint_slot=1 - checkpoint_slot,
                        checkpoint_generation=checkpoint_generation + 1,
                        cell_index=cell_index,
                        completed_steps=completed_steps,
                        policy=policy,
                        optimizer=optimizer,
                        overlay_optimizer=overlay_optimizer,
                        last_metrics=last_metrics,
                        peak_allocated_bytes=rank_peak_allocated,
                        peak_reserved_bytes=rank_peak_reserved,
                    ),
                )
                checkpoint_slot = 1 - checkpoint_slot
                checkpoint_generation += 1
                checkpoint_path = _checkpoint_path(
                    run_root, context.rank, checkpoint_slot
                )
                _publish_checkpoint_manifest(
                    run_root,
                    checkpoint_path,
                    plan=plan,
                    plan_file_sha256=plan_file_sha256,
                    context=context,
                    mode=mode,
                    cell_index=cell_index,
                    completed_steps=completed_steps,
                )
                if (
                    qualification_restart_after_step1
                    and completed_steps == 1
                    and restart_count == 0
                ):
                    _prepare_intentional_qualification_restart(
                        context,
                        run_root,
                        restart_count=restart_count,
                    )
                    raise Top2000M03RV7WorkerError(
                        "intentional qualification restart after checkpoint 1"
                    )

            model_directory = (
                run_root / "cells" / f"fold-{fold_index:02d}-seed-{seed}"
            )
            receipt_path = (
                run_root
                / "receipts"
                / f"fold-{fold_index:02d}-seed-{seed}.json"
            )
            already_finalized = _existing_cell_receipt_status(
                receipt_path,
                model_directory,
                plan=plan,
                plan_file_sha256=plan_file_sha256,
                context=context,
                mode=mode,
                cell_index=cell_index,
                fold_index=fold_index,
                seed=seed,
                completed_steps=completed_steps,
            )
            if not already_finalized:
                model_path = model_directory / f"model.rank-{context.rank:02d}.pt"
                alpha_optimizer_state = optimizer.state_dict()
                alpha_optimizer_hash = optimizer_state_dict_sha256(
                    alpha_optimizer_state
                )
                overlay_optimizer_state = (
                    None
                    if overlay_optimizer is None
                    else overlay_optimizer.state_dict()
                )
                overlay_optimizer_hash = (
                    None
                    if overlay_optimizer_state is None
                    else optimizer_state_dict_sha256(overlay_optimizer_state)
                )
                model_sha256 = _write_immutable_torch(
                    model_path,
                    {
                        "schema": CELL_MODEL_SCHEMA,
                        "protocol_sha256": plan.protocol_sha256,
                        "plan_file_sha256": plan_file_sha256,
                        "plan_receipt_sha256": plan.receipt_sha256,
                        "cache_sha256": plan.cache_sha256,
                        "setting_index": plan.setting_index,
                        "setting_id": plan.setting_id,
                        "fold_index": fold_index,
                        "fold_receipt_sha256": fold.receipt_sha256,
                        "seed": seed,
                        "rank": context.rank,
                        "world_size": context.world_size,
                        "completed_optimizer_steps": completed_steps,
                        "model_state_dict": {
                            name: value.detach().cpu().clone()
                            for name, value in policy.state_dict().items()
                        },
                        "overlay_optimizer_required": (
                            _requires_overlay_optimizer(plan)
                        ),
                        "alpha_core_optimizer_state_dict": (
                            alpha_optimizer_state
                        ),
                        "alpha_core_optimizer_state_sha256": (
                            alpha_optimizer_hash
                        ),
                        "overlay_optimizer_state_dict": (
                            overlay_optimizer_state
                        ),
                        "overlay_optimizer_state_sha256": (
                            overlay_optimizer_hash
                        ),
                        "last_metrics": last_metrics,
                        "development_only": True,
                        "promotion_eligible": False,
                    },
                )
                model_hashes = _gather_objects(model_sha256, context)
                rank_model_state_hashes = _gather_objects(
                    model_state_sha256(policy),
                    context,
                )
                rank_alpha_optimizer_hashes = _gather_objects(
                    alpha_optimizer_hash,
                    context,
                )
                rank_overlay_optimizer_hashes = _gather_objects(
                    overlay_optimizer_hash,
                    context,
                )
                if len(set(rank_model_state_hashes)) != 1:
                    raise Top2000M03RV7WorkerError(
                        "two-rank final model states diverged before validation"
                    )
                if len(set(rank_alpha_optimizer_hashes)) != 1 or len(
                    set(rank_overlay_optimizer_hashes)
                ) != 1:
                    raise Top2000M03RV7WorkerError(
                        "two-rank optimizer states diverged before finalization"
                    )
                peaks = _gather_objects(
                    _recorded_rank_peak(
                        context,
                        peak_allocated_bytes=rank_peak_allocated,
                        peak_reserved_bytes=rank_peak_reserved,
                    ),
                    context,
                )
                if context.rank == 0:
                    seed_validation_path: Path | None = None
                    seed_validation_sha256: str | None = None
                    if not qualification_only:
                        seed_validation_path, seed_validation_sha256 = (
                            _evaluate_seed_checkpoint(
                                cache,
                                fold,
                                policy,
                                plan=plan,
                                run_root=run_root,
                                seed=seed,
                                checkpoint_file_sha256=str(model_hashes[0]),
                                device=context.device,
                            )
                        )
                    _write_immutable_json(
                        receipt_path,
                        {
                            "schema": CELL_RECEIPT_SCHEMA,
                            "worker_schema": WORKER_SCHEMA,
                            "mode": mode,
                            "protocol_sha256": plan.protocol_sha256,
                            "plan_file_sha256": plan_file_sha256,
                            "plan_receipt_sha256": plan.receipt_sha256,
                            "cache_sha256": plan.cache_sha256,
                            "setting_index": plan.setting_index,
                            "setting_id": plan.setting_id,
                            "cell_index": cell_index,
                            "fold_index": fold_index,
                            "fold_receipt_sha256": fold.receipt_sha256,
                            "seed": seed,
                            "optimizer_steps": completed_steps,
                            "rank_model_sha256": model_hashes,
                            "rank_model_state_sha256": rank_model_state_hashes,
                            "overlay_optimizer_required": (
                                _requires_overlay_optimizer(plan)
                            ),
                            "rank_alpha_core_optimizer_state_sha256": (
                                rank_alpha_optimizer_hashes
                            ),
                            "rank_overlay_optimizer_state_sha256": (
                                rank_overlay_optimizer_hashes
                            ),
                            "alpha_core_optimizer_steps": completed_steps,
                            "overlay_optimizer_steps": (
                                completed_steps
                                if overlay_optimizer is not None
                                else 0
                            ),
                            "rank_peak_cuda_memory": peaks,
                            "last_metrics": last_metrics,
                            "seed_validation_required": not qualification_only,
                            "seed_validation_receipt": (
                                None
                                if seed_validation_path is None
                                else str(seed_validation_path.relative_to(run_root))
                            ),
                            "seed_validation_receipt_sha256": (
                                seed_validation_sha256
                            ),
                            "development_only": True,
                            "promotion_eligible": False,
                        },
                    )
            _barrier(context)
            if not qualification_only and seed == TOP2000_M03R_V7_DEV_SEEDS[-1]:
                fold_ensemble_status: list[dict[str, Any]] = [
                    {"receipt": None, "sha256": None, "error": None}
                ]
                if context.rank == 0:
                    try:
                        fold_receipt_path = _fold_ensemble_receipt_path(
                            run_root,
                            fold_index=fold_index,
                        )
                        if fold_receipt_path.exists():
                            payload = json.loads(
                                fold_receipt_path.read_text(encoding="utf-8")
                            )
                            if (
                                not isinstance(payload, dict)
                                or payload.get("schema")
                                != "rl-quant.top2000-dev.m03r-v7-fold-ensemble-v1"
                                or payload.get("fold_index") != fold_index
                                or payload.get("setting_id") != plan.setting_id
                                or payload.get("ordered_seeds")
                                != list(TOP2000_M03R_V7_DEV_SEEDS)
                                or payload.get("development_only") is not True
                                or payload.get("promotion_eligible") is not False
                            ):
                                raise Top2000M03RV7WorkerError(
                                    "existing fold ensemble receipt drifted"
                                )
                            artifact_hash = payload.get(
                                "validation_trace_artifact_sha256"
                            )
                            artifact_path = _validation_trace_artifact(
                                run_root,
                                fold_index=fold_index,
                                seed=None,
                            )
                            if (
                                not isinstance(artifact_hash, str)
                                or not artifact_path.is_file()
                                or _sha256_file(artifact_path) != artifact_hash
                            ):
                                raise Top2000M03RV7WorkerError(
                                    "existing fold ensemble trace artifact drifted"
                                )
                            fold_sha256 = _sha256_file(fold_receipt_path)
                        else:
                            fold_receipt_path, fold_sha256 = _evaluate_fold_ensemble(
                                cache,
                                fold,
                                plan=plan,
                                run_root=run_root,
                                device=context.device,
                            )
                        fold_ensemble_status[0] = {
                            "receipt": str(fold_receipt_path),
                            "sha256": fold_sha256,
                            "error": None,
                        }
                    except (
                        OSError,
                        ValueError,
                        json.JSONDecodeError,
                        Top2000M03RV7ValidationError,
                        Top2000M03RV7WorkerError,
                    ) as exc:
                        fold_ensemble_status[0] = {
                            "receipt": None,
                            "sha256": None,
                            "error": str(exc),
                        }
                if context.world_size == 2:
                    dist.broadcast_object_list(fold_ensemble_status, src=0)
                if fold_ensemble_status[0].get("error") is not None:
                    raise Top2000M03RV7WorkerError(
                        "fold ensemble validation failed: "
                        + str(fold_ensemble_status[0]["error"])
                    )
                _barrier(context)
            cell_index += 1
            completed_steps = 0
            checkpoint = None
            _atomic_torch_save(
                _checkpoint_path(
                    run_root,
                    context.rank,
                    1 - checkpoint_slot,
                ),
                _checkpoint_payload(
                    plan=plan,
                    plan_file_sha256=plan_file_sha256,
                    context=context,
                    mode=mode,
                    checkpoint_slot=1 - checkpoint_slot,
                    checkpoint_generation=checkpoint_generation + 1,
                    cell_index=cell_index,
                    completed_steps=0,
                    policy=None,
                    optimizer=None,
                    overlay_optimizer=None,
                    last_metrics=None,
                    peak_allocated_bytes=rank_peak_allocated,
                    peak_reserved_bytes=rank_peak_reserved,
                ),
            )
            checkpoint_slot = 1 - checkpoint_slot
            checkpoint_generation += 1
            checkpoint_path = _checkpoint_path(
                run_root, context.rank, checkpoint_slot
            )
            _publish_checkpoint_manifest(
                run_root,
                checkpoint_path,
                plan=plan,
                plan_file_sha256=plan_file_sha256,
                context=context,
                mode=mode,
                cell_index=cell_index,
                completed_steps=0,
            )

        rank_peak = _recorded_rank_peak(
            context,
            peak_allocated_bytes=rank_peak_allocated,
            peak_reserved_bytes=rank_peak_reserved,
        )
        peaks = _gather_objects(rank_peak, context)
        elapsed_seconds = _gather_objects(
            max(0.0, time.perf_counter() - run_started),
            context,
        )
        if context.rank != 0:
            return None
        receipt_files = sorted((run_root / "receipts").glob("fold-*-seed-*.json"))
        expected_receipts = final_cell_count
        if len(receipt_files) != expected_receipts:
            raise Top2000M03RV7WorkerError(
                "terminal receipt inventory is incomplete or contains stale cells"
            )
        receipt_hashes = {
            path.name: _sha256_file(path)
            for path in receipt_files
        }
        qualification_cell_evidence: dict[str, Any] | None = None
        if qualification_only:
            if len(receipt_files) != 1:
                raise Top2000M03RV7WorkerError(
                    "qualification must publish exactly one cell receipt"
                )
            try:
                qualification_cell_evidence = json.loads(
                    receipt_files[0].read_bytes()
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise Top2000M03RV7WorkerError(
                    "qualification cell receipt cannot be replayed"
                ) from exc
            if not isinstance(qualification_cell_evidence, dict):
                raise Top2000M03RV7WorkerError(
                    "qualification cell receipt must be an object"
                )
        seed_validation_hashes: dict[str, str] = {}
        fold_ensemble_hashes: dict[str, str] = {}
        if not qualification_only:
            seed_validation_hashes, fold_ensemble_hashes = (
                _collect_full_validation_evidence(
                    run_root,
                    plan=plan,
                    folds=folds,
                    cells=cells,
                )
            )
        terminal: dict[str, Any] = {
            "schema": (
                QUALIFICATION_RECEIPT_SCHEMA
                if qualification_only
                else COMPLETION_RECEIPT_SCHEMA
            ),
            "worker_schema": WORKER_SCHEMA,
            "mode": mode,
            "protocol_sha256": plan.protocol_sha256,
            "plan_file_sha256": plan_file_sha256,
            "plan_receipt_sha256": plan.receipt_sha256,
            "cache_sha256": plan.cache_sha256,
            "cache_identity": cache.cache_identity,
            "search_identity": cache.search_identity,
            "action_hash": cache.action_hash,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "world_size": context.world_size,
            "fold_count": 1 if qualification_only else len(folds),
            "paired_seeds": [cells[0][1]] if qualification_only else list(TOP2000_M03R_V7_DEV_SEEDS),
            "completed_cells": final_cell_count,
            "optimizer_steps_per_cell": effective_steps,
            "cell_receipt_sha256": receipt_hashes,
            "seed_validation_receipt_sha256": seed_validation_hashes,
            "fold_ensemble_receipt_sha256": fold_ensemble_hashes,
            "seed_validation_receipt_count": len(seed_validation_hashes),
            "fold_ensemble_receipt_count": len(fold_ensemble_hashes),
            "inference_path_count": (
                0 if qualification_only else len(fold_ensemble_hashes)
            ),
            "seeds_are_independent_return_paths": False,
            "output_space_ensemble_required": not qualification_only,
            "rank_peak_cuda_memory": peaks,
            "rank_elapsed_seconds": elapsed_seconds,
            "intentional_restart_after_step": (
                1 if qualification_restart_after_step1 else None
            ),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_completed_steps": resume_completed_steps,
            "rank_model_state_sha256": (
                None
                if qualification_cell_evidence is None
                else qualification_cell_evidence.get("rank_model_state_sha256")
            ),
            "rank_alpha_core_optimizer_state_sha256": (
                None
                if qualification_cell_evidence is None
                else qualification_cell_evidence.get(
                    "rank_alpha_core_optimizer_state_sha256"
                )
            ),
            "rank_overlay_optimizer_state_sha256": (
                None
                if qualification_cell_evidence is None
                else qualification_cell_evidence.get(
                    "rank_overlay_optimizer_state_sha256"
                )
            ),
            "development_only": True,
            "future_selected_universe": True,
            "outer_evaluation_authorized": False,
            "promotion_eligible": False,
            "complete": True,
        }
        terminal_path = run_root / (
            "qualification-receipt.json"
            if qualification_only
            else "completion-receipt.json"
        )
        if terminal_path.exists():
            try:
                existing_terminal = json.loads(terminal_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise Top2000M03RV7WorkerError(
                    "existing terminal receipt cannot be replayed"
                ) from exc
            prior_elapsed = (
                existing_terminal.get("rank_elapsed_seconds")
                if isinstance(existing_terminal, dict)
                else None
            )
            if not isinstance(prior_elapsed, list):
                raise Top2000M03RV7WorkerError(
                    "existing terminal receipt omitted rank timing"
                )
            terminal["rank_elapsed_seconds"] = prior_elapsed
            for name in (
                "intentional_restart_after_step",
                "resumed_from_checkpoint",
                "resume_completed_steps",
            ):
                terminal[name] = existing_terminal.get(name)
        terminal_sha256 = _write_immutable_json(terminal_path, terminal)
        terminal["receipt_path"] = str(terminal_path)
        terminal["receipt_sha256"] = terminal_sha256
        return terminal
    finally:
        _destroy_process_group(context)


def _setting_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("setting index must be an integer") from exc
    if not 0 <= index < len(M03R_TOP2000_DEV_SETTING_IDS):
        raise argparse.ArgumentTypeError("setting index must lie in [0, 11]")
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-plan")
    render.add_argument("--setting-index", required=True, type=_setting_index)
    render.add_argument("--cache-path", required=True)
    render.add_argument("--cache-sha256", required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--plan-path", required=True)
    render.add_argument("--optimizer-steps", type=int, default=64)
    render.add_argument("--max-origin-batch", type=int, default=16)
    render.add_argument("--learning-rate", type=float, default=1.0e-4)
    render.add_argument("--weight-decay", type=float, default=1.0e-4)
    render.add_argument("--grad-clip", type=float, default=1.0)
    render.add_argument("--token-dim", type=int, default=128)
    render.add_argument("--raw-stock-chunk", type=int, default=512)

    train = subparsers.add_parser("train")
    train.add_argument("--plan-path", required=True)
    train.add_argument("--plan-sha256", required=True)
    train.add_argument("--setting-index", required=True, type=_setting_index)
    train.add_argument("--qualification-only", action="store_true")
    train.add_argument("--qualification-steps", type=int, default=1)
    train.add_argument(
        "--qualification-restart-after-step1",
        action="store_true",
    )

    package_train = subparsers.add_parser("package-train")
    package_train.add_argument("--package-plan", required=True)
    package_train.add_argument("--package-plan-sha256", required=True)
    package_train.add_argument("--output-root", required=True)
    package_train.add_argument("--completion-index", type=_setting_index)
    package_train.add_argument("--qualification-only", action="store_true")
    package_train.add_argument("--qualification-steps", type=int, default=1)
    package_train.add_argument(
        "--qualification-restart-after-step1",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # The Kubernetes renderer invokes the exact worker prefix and appends
    # option arguments directly.  No shell or ``$(JOB_COMPLETION_INDEX)``
    # substitution is needed or accepted.
    if raw_argv and raw_argv[0] == "--package-plan":
        raw_argv.insert(0, "package-train")
    args = _parser().parse_args(raw_argv)
    try:
        if args.command == "render-plan":
            plan = Top2000M03RV7DevelopmentTrainingPlan(
                setting_index=args.setting_index,
                setting_id=M03R_TOP2000_DEV_SETTING_IDS[args.setting_index],
                cache_path=args.cache_path,
                cache_sha256=args.cache_sha256,
                output_root=args.output_root,
                total_optimizer_steps_per_fold_seed=args.optimizer_steps,
                max_origin_batch=args.max_origin_batch,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                token_dim=args.token_dim,
                raw_stock_chunk=args.raw_stock_chunk,
            )
            file_sha256 = render_training_plan(args.plan_path, plan)
            print(
                json.dumps(
                    {
                        "plan_path": args.plan_path,
                        "plan_file_sha256": file_sha256,
                        "plan_receipt_sha256": plan.receipt_sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "package-train":
            package = load_package_plan(
                args.package_plan,
                expected_package_plan_sha256=args.package_plan_sha256,
            )
            completion_index = resolve_completion_index(args.completion_index)
            plan, plan_file_sha256 = plan_from_package_completion(
                package,
                package_plan_path=args.package_plan,
                completion_index=completion_index,
                output_root=args.output_root,
            )
        else:
            plan, plan_file_sha256 = load_training_plan(
                args.plan_path,
                expected_sha256=args.plan_sha256,
                expected_setting_index=args.setting_index,
            )
        terminal = run_worker(
            plan,
            plan_file_sha256=plan_file_sha256,
            qualification_only=args.qualification_only,
            qualification_steps=args.qualification_steps,
            qualification_restart_after_step1=(
                args.qualification_restart_after_step1
            ),
        )
        if terminal is not None:
            print(json.dumps(terminal, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
        Top2000M03RV7ValidationError,
        Top2000M03RV7WorkerError,
    ) as exc:
        print(f"TOP2000 M03R-v7 worker failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_MODEL_SCHEMA",
    "CELL_RECEIPT_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_RECEIPT_SCHEMA",
    "PROGRESS_MANIFEST_SCHEMA",
    "QUALIFICATION_RECEIPT_SCHEMA",
    "Top2000M03RV7WorkerError",
    "deterministic_episode_start",
    "load_package_plan",
    "load_training_plan",
    "main",
    "optimizer_state_dict_sha256",
    "optimizer_state_sha256",
    "render_training_plan",
    "resolve_completion_index",
    "run_worker",
]
