"""Append-only training and artifact driver for Hold-30 v2 trials.

This module owns the durable boundary around
:func:`rl_quant.training.hold30.train_hold30_update`.  It intentionally does
not discover data, select folds, launch processes, or allocate GPUs.  A caller
must provide an already frozen setting/fold/seed identity and an ordered list
of content-addressed chronological sweeps.

Each sweep causes exactly one canonical pass and exactly one optimizer step.
Every completed step is published as an exclusive checkpoint plus a
self-hashed JSON receipt.  Receipts form an append-only SHA-256 chain.  Resume
is permitted only from the latest complete node; orphaned, duplicate, stale,
or mismatched artifacts fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION, resolve_hold30_setting
from rl_quant.protocol.hold30_freeze import HOLD30_FOLDS, HOLD30_SEEDS, sha256_payload
from rl_quant.training.hold30 import (
    Hold30LossContract,
    Hold30ReplayAdapter,
    Hold30ReplayGeometry,
    train_hold30_update,
)


HOLD30_DRIVER_SCHEMA_VERSION = 1
HOLD30_PRODUCTION_OPTIMIZER_UPDATES = 128
HOLD30_IDENTITY_SCHEMA = "rl-quant.hold30.trial-identity"
HOLD30_CHECKPOINT_SCHEMA = "rl-quant.hold30.training-checkpoint"
HOLD30_CHECKPOINT_RECEIPT_SCHEMA = "rl-quant.hold30.checkpoint-receipt"
HOLD30_METRICS_SCHEMA = "rl-quant.hold30.training-metrics"
HOLD30_FINAL_MODEL_SCHEMA = "rl-quant.hold30.final-model"
HOLD30_RUN_RECEIPT_SCHEMA = "rl-quant.hold30.run-receipt"
HOLD30_INITIAL_MODEL_SCHEMA = "rl-quant.hold30.initial-model"
HOLD30_INITIAL_RECEIPT_SCHEMA = "rl-quant.hold30.initial-model-receipt"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT = re.compile(r"update-([0-9]{6})\.pt\Z")
_CHECKPOINT_RECEIPT = re.compile(r"update-([0-9]{6})\.receipt\.json\Z")
_FINAL_NAMES = frozenset({"final-model.pt", "metrics.json", "run-receipt.json"})
_INITIAL_NAMES = frozenset({"initial-model.pt", "initial-model.receipt.json"})
_COHORT_MARKER = "cohort-finalization.json"


class Hold30ArtifactError(RuntimeError):
    """A trial identity, checkpoint, or receipt graph is unsafe to use."""


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    """Hash tensor trees without relying on pickle byte stability."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"N")
        elif isinstance(item, bool):
            digest.update(b"B1" if item else b"B0")
        elif isinstance(item, int):
            digest.update(b"I" + str(item).encode("ascii") + b";")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise Hold30ArtifactError("artifact trees cannot contain non-finite floats")
            digest.update(b"F" + item.hex().encode("ascii") + b";")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
        elif isinstance(item, bytes):
            digest.update(b"Y" + str(len(item)).encode("ascii") + b":" + item)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            if tensor.layout != torch.strided:
                raise Hold30ArtifactError("only strided tensors are supported in artifacts")
            digest.update(b"T")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(str(len(raw)).encode("ascii") + b":" + raw)
        elif isinstance(item, Mapping):
            digest.update(b"M" + str(len(item)).encode("ascii") + b":")
            encoded_keys: list[tuple[bytes, Any, Any]] = []
            for key, mapped in item.items():
                key_digest = hashlib.sha256()
                if isinstance(key, (str, int, bool)):
                    key_digest.update(f"{type(key).__name__}:{key}".encode("utf-8"))
                else:
                    raise Hold30ArtifactError(
                        f"unsupported artifact mapping key type {type(key).__name__}"
                    )
                encoded_keys.append((key_digest.digest(), key, mapped))
            for _, key, mapped in sorted(encoded_keys, key=lambda row: row[0]):
                update(key)
                update(mapped)
        elif isinstance(item, tuple):
            digest.update(b"U" + str(len(item)).encode("ascii") + b":")
            for child in item:
                update(child)
        elif isinstance(item, list):
            digest.update(b"L" + str(len(item)).encode("ascii") + b":")
            for child in item:
                update(child)
        else:
            raise Hold30ArtifactError(
                f"unsupported artifact value type {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _tensor_audit(value: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor-digest",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": _tree_sha256(value),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Hold30ArtifactError("metrics cannot contain non-finite floats")
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _tensor_audit(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise Hold30ArtifactError("JSON artifact mappings require string keys")
        return {key: _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise Hold30ArtifactError(f"unsupported JSON artifact value {type(value).__name__}")


def _with_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise Hold30ArtifactError("receipt payload already contains receipt_sha256")
    result = dict(payload)
    result["receipt_sha256"] = sha256_payload(result)
    return result


def _validate_self_hash(payload: Mapping[str, Any], *, path: Path) -> str:
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or _DIGEST.fullmatch(claimed) is None:
        raise Hold30ArtifactError(f"{path} lacks a valid receipt_sha256")
    unsigned = dict(payload)
    del unsigned["receipt_sha256"]
    actual = sha256_payload(unsigned)
    if actual != claimed:
        raise Hold30ArtifactError(f"{path} receipt hash mismatch")
    return claimed


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Hold30ArtifactError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Hold30ArtifactError(f"required artifact is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise Hold30ArtifactError(f"artifact must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Hold30ArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Hold30ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    """Durably publish a new file without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8"}
        with os.fdopen(descriptor, mode, **kwargs) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Hold30ArtifactError(f"refusing to overwrite existing artifact: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    def write(stream: Any) -> None:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

    _publish_new(path, write, binary=False)


def _write_new_torch(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_new(path, lambda stream: torch.save(dict(payload), stream), binary=True)


@dataclass(frozen=True, slots=True)
class Hold30TrialIdentity:
    """Exact frozen identity of one setting/fold/seed trial."""

    setting_id: str
    fold_index: int
    seed: int
    executable_manifest_sha256: str
    fold_sha256: str
    protocol_generation: str = HOLD30_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
        setting = resolve_hold30_setting(self.setting_id)
        del setting
        if self.protocol_generation != HOLD30_PROTOCOL_GENERATION:
            raise ValueError(
                f"protocol_generation must be {HOLD30_PROTOCOL_GENERATION!r}"
            )
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < HOLD30_FOLDS
        ):
            raise ValueError(f"fold_index must be in [0, {HOLD30_FOLDS - 1}]")
        if isinstance(self.seed, bool) or self.seed not in HOLD30_SEEDS:
            raise ValueError(f"seed must be one of {HOLD30_SEEDS}")
        _require_digest("executable_manifest_sha256", self.executable_manifest_sha256)
        _require_digest("fold_sha256", self.fold_sha256)

    def payload(self) -> dict[str, Any]:
        setting = resolve_hold30_setting(self.setting_id)
        return {
            "protocol_generation": self.protocol_generation,
            "setting_index": setting.setting_index,
            "setting_id": setting.setting_id,
            "mechanism": setting.mechanism,
            "fold_index": self.fold_index,
            "seed": self.seed,
            "executable_manifest_sha256": self.executable_manifest_sha256,
            "fold_sha256": self.fold_sha256,
        }


@dataclass(frozen=True, slots=True)
class Hold30StateProviderBinding:
    """Content-addressed identity of the differentiable causal state path."""

    provider_id: str
    provider_config: Mapping[str, Any]
    provider_config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(self.provider_config, Mapping):
            raise ValueError("provider_config must be a JSON-compatible mapping")
        safe_config = _json_safe(self.provider_config)
        expected = sha256_payload(safe_config)
        _require_digest("provider_config_sha256", self.provider_config_sha256)
        if self.provider_config_sha256 != expected:
            raise ValueError("provider_config_sha256 does not match provider_config")

    @classmethod
    def from_provider(
        cls,
        provider_id: str,
        provider: Any,
    ) -> "Hold30StateProviderBinding":
        config = _actual_state_provider_config(provider)
        return cls(
            provider_id=provider_id,
            provider_config=config,
            provider_config_sha256=sha256_payload(config),
        )


@dataclass(frozen=True, slots=True)
class Hold30TrainingSweep:
    """One content-addressed chronological input to one optimizer update."""

    sweep_index: int
    sweep_id: str
    sequence_sha256: str
    sequence: Any
    n_positions: int

    def __post_init__(self) -> None:
        if isinstance(self.sweep_index, bool) or not isinstance(self.sweep_index, int) or self.sweep_index < 0:
            raise ValueError("sweep_index must be a non-negative integer")
        if not isinstance(self.sweep_id, str) or not self.sweep_id:
            raise ValueError("sweep_id must be a non-empty string")
        _require_digest("sequence_sha256", self.sequence_sha256)
        if isinstance(self.n_positions, bool) or not isinstance(self.n_positions, int):
            raise ValueError("n_positions must be an integer")
        if hasattr(self.sequence, "n_positions") and int(self.sequence.n_positions) != self.n_positions:
            raise ValueError("declared n_positions does not match sequence.n_positions")

    def descriptor(self) -> dict[str, Any]:
        return {
            "sweep_index": self.sweep_index,
            "sweep_id": self.sweep_id,
            "sequence_sha256": self.sequence_sha256,
            "n_positions": self.n_positions,
        }


@dataclass(frozen=True, slots=True)
class Hold30RunProgress:
    complete: bool
    completed_sweeps: int
    total_sweeps: int
    latest_checkpoint: Path
    run_receipt: Path | None


def _validate_sweeps(
    sweeps: Sequence[Hold30TrainingSweep],
    geometry: Hold30ReplayGeometry,
    *,
    expected_updates: int,
) -> tuple[dict[str, Any], ...]:
    if not sweeps:
        raise ValueError("a Hold-30 trial requires at least one chronological sweep")
    if len(sweeps) != expected_updates:
        raise ValueError(
            f"Hold-30 requires exactly {expected_updates} optimizer-update sweeps; "
            f"got {len(sweeps)}"
        )
    descriptors = tuple(sweep.descriptor() for sweep in sweeps)
    if tuple(row["sweep_index"] for row in descriptors) != tuple(range(len(sweeps))):
        raise ValueError("sweep indexes must be contiguous, ordered, and start at zero")
    ids = [row["sweep_id"] for row in descriptors]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sweep_id values are forbidden")
    hashes = {row["sequence_sha256"] for row in descriptors}
    if len(hashes) != 1:
        raise ValueError("all optimizer updates must reuse the same chronological sequence hash")
    for row in descriptors:
        if row["n_positions"] < geometry.minimum_positions:
            raise ValueError(
                f"sweep {row['sweep_id']!r} needs at least "
                f"{geometry.minimum_positions} positions"
            )
    return descriptors


def _policy_schema(policy: torch.nn.Module) -> dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in policy.state_dict().items()
    }


def _optimizer_signature(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        values = {key: value for key, value in group.items() if key != "params"}
        values["parameter_count"] = len(group["params"])
        groups.append(_json_safe(values))
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": groups,
    }


def _validate_production_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Hold-30 production optimizer must be torch.optim.AdamW")
    for index, group in enumerate(optimizer.param_groups):
        expected = {"lr": 1e-4, "weight_decay": 1e-4, "eps": 1e-5}
        for name, value in expected.items():
            if float(group.get(name, float("nan"))) != value:
                raise ValueError(
                    f"Hold-30 AdamW group {index} requires {name}={value:g}"
                )


@dataclass(frozen=True, slots=True)
class _DistributedContext:
    world_size: int
    rank: int
    backend: str | None

    @property
    def enabled(self) -> bool:
        return self.world_size == 2


def _distributed_context(world_size: int, rank: int) -> _DistributedContext:
    if world_size not in {1, 2}:
        raise ValueError("Hold-30 supports only world_size=1 or exactly world_size=2")
    if rank not in range(world_size):
        raise ValueError("rank is outside world_size")
    if world_size == 1:
        return _DistributedContext(1, 0, None)
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != rank
    ):
        raise ValueError("world_size=2 requires a matching initialized process group")
    return _DistributedContext(2, rank, str(dist.get_backend()))


def _distributed_require_equal(
    context: _DistributedContext,
    name: str,
    value: Any,
) -> None:
    if not context.enabled:
        return
    digest = sha256_payload(_json_safe(value))
    gathered: list[str | None] = [None, None]
    dist.all_gather_object(gathered, digest)
    if gathered[0] != gathered[1]:
        raise Hold30ArtifactError(f"distributed ranks disagree on {name}")


def _rank0_call(context: _DistributedContext, callback: Callable[[], Any]) -> Any:
    if not context.enabled:
        return callback()
    packet: list[Any] = [None]
    if context.rank == 0:
        try:
            packet[0] = (True, callback())
        except Exception as exc:  # propagated so the peer never hangs at a barrier
            packet[0] = (False, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(packet, src=0)
    success, value = packet[0]
    if not success:
        raise Hold30ArtifactError(f"rank-0 artifact operation failed: {value}")
    return value


def _all_rank_values(context: _DistributedContext, value: Any) -> list[Any]:
    if not context.enabled:
        return [value]
    gathered: list[Any] = [None, None]
    dist.all_gather_object(gathered, value)
    return gathered


def _state_provider_contract(
    adapter: Hold30ReplayAdapter,
    binding: Hold30StateProviderBinding,
) -> dict[str, Any]:
    """Require and bind a genuinely differentiable upstream state provider."""

    owner = getattr(adapter, "runtime", adapter)
    provider = getattr(owner, "state_provider", None)
    if provider is None:
        raise ValueError(
            "Hold-30 production training requires an explicit trainable state_provider"
        )
    if getattr(provider, "trains_upstream_encoder", None) is not True:
        raise ValueError(
            "Hold-30 production training rejects static precomputed decision state"
        )
    if getattr(owner, "require_trainable_state_provider", None) is not True:
        raise ValueError(
            "Hold-30 runtime must set require_trainable_state_provider=True"
        )
    config = _json_safe(binding.provider_config)
    actual_config = _actual_state_provider_config(provider)
    if actual_config != config:
        raise ValueError("state-provider binding config differs from the actual provider")
    payload = {
        "provider_id": binding.provider_id,
        "provider_class": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "trains_upstream_encoder": True,
        "require_trainable_state_provider": True,
        "provider_config": config,
        "provider_config_sha256": binding.provider_config_sha256,
    }
    payload["binding_sha256"] = sha256_payload(payload)
    return payload


def _actual_state_provider_config(provider: Any) -> dict[str, Any]:
    provider_config = getattr(provider, "binding_config", None)
    if provider_config is None:
        provider_config = getattr(provider, "hold30_provider_config", None)
    if callable(provider_config):
        provider_config = provider_config()
    if not isinstance(provider_config, Mapping):
        raise ValueError(
            "trainable state_provider must expose authoritative binding_config"
        )
    actual_config = _json_safe(provider_config)
    if not isinstance(actual_config, dict):
        raise ValueError("state-provider binding_config must be a mapping")
    embedded = actual_config.get("binding_sha256")
    if embedded is not None:
        unsigned = dict(actual_config)
        del unsigned["binding_sha256"]
        if embedded != sha256_payload(unsigned):
            raise ValueError("actual state-provider binding_config digest is invalid")
    return actual_config


def _driver_contract(
    identity: Hold30TrialIdentity,
    geometry: Hold30ReplayGeometry,
    grad_clip: float,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    adapter: Hold30ReplayAdapter,
    state_provider_binding: Hold30StateProviderBinding,
    distributed_context: _DistributedContext,
    expected_updates: int,
) -> dict[str, Any]:
    _validate_production_optimizer(optimizer)
    if isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)):
        raise ValueError("grad_clip must be a finite non-negative scalar")
    if not math.isfinite(float(grad_clip)) or float(grad_clip) < 0:
        raise ValueError("grad_clip must be a finite non-negative scalar")
    contract = Hold30LossContract.for_setting(identity.setting_id)
    return {
        "loss_contract": asdict(contract),
        "replay_geometry": asdict(geometry),
        "grad_clip": float(grad_clip),
        "updates_per_sweep": 1,
        "optimizer_update_count": expected_updates,
        "production_update_contract": (
            expected_updates == HOLD30_PRODUCTION_OPTIMIZER_UPDATES
        ),
        "qualification_only": (
            expected_updates != HOLD30_PRODUCTION_OPTIMIZER_UPDATES
        ),
        "policy_schema": _policy_schema(policy),
        "optimizer": _optimizer_signature(optimizer),
        "state_provider": _state_provider_contract(adapter, state_provider_binding),
        "distributed": {
            "world_size": distributed_context.world_size,
            "backend": distributed_context.backend,
            "origin_sharding": "strided-rank-mod-world-size",
            "gradient_reduction": "SUM",
            "global_objective_denominator": True,
            "replicated_canonical_chronology": True,
        },
    }


def _identity_document(
    identity: Hold30TrialIdentity,
    descriptors: Sequence[Mapping[str, Any]],
    driver_contract: Mapping[str, Any],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    plan = list(descriptors)
    payload = {
        "schema": HOLD30_IDENTITY_SCHEMA,
        "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
        "trial": identity.payload(),
        "trial_sha256": sha256_payload(identity.payload()),
        "sweep_plan": plan,
        "sweep_plan_sha256": sha256_payload(plan),
        "driver_contract": dict(driver_contract),
        "driver_contract_sha256": sha256_payload(driver_contract),
        "initial_policy_state_sha256": _tree_sha256(policy.state_dict()),
        "initial_optimizer_state_sha256": _tree_sha256(optimizer.state_dict()),
    }
    return _with_receipt_hash(payload)


def _validate_identity_document(
    document: Mapping[str, Any],
    *,
    path: Path,
    identity: Hold30TrialIdentity,
    descriptors: Sequence[Mapping[str, Any]],
    driver_contract: Mapping[str, Any],
) -> str:
    receipt_hash = _validate_self_hash(document, path=path)
    expected_fields = {
        "schema",
        "schema_version",
        "trial",
        "trial_sha256",
        "sweep_plan",
        "sweep_plan_sha256",
        "driver_contract",
        "driver_contract_sha256",
        "initial_policy_state_sha256",
        "initial_optimizer_state_sha256",
        "receipt_sha256",
    }
    if set(document) != expected_fields:
        raise Hold30ArtifactError("identity document is partial or has unknown fields")
    if document.get("schema") != HOLD30_IDENTITY_SCHEMA or document.get("schema_version") != 1:
        raise Hold30ArtifactError("unsupported Hold-30 identity schema")
    expected_trial = identity.payload()
    if document.get("trial") != expected_trial:
        raise Hold30ArtifactError("trial setting/fold/seed identity mismatch")
    if document.get("trial_sha256") != sha256_payload(expected_trial):
        raise Hold30ArtifactError("trial identity digest mismatch")
    plan = list(descriptors)
    if document.get("sweep_plan") != plan or document.get("sweep_plan_sha256") != sha256_payload(plan):
        raise Hold30ArtifactError("chronological sweep plan mismatch")
    if document.get("driver_contract") != driver_contract:
        raise Hold30ArtifactError("driver/loss/optimizer contract mismatch")
    if document.get("driver_contract_sha256") != sha256_payload(driver_contract):
        raise Hold30ArtifactError("driver contract digest mismatch")
    for name in ("initial_policy_state_sha256", "initial_optimizer_state_sha256"):
        try:
            _require_digest(name, document[name])
        except (KeyError, ValueError) as exc:
            raise Hold30ArtifactError(f"identity document has invalid {name}") from exc
    return receipt_hash


def _seed_trial(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            [value.tolist() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else []
        ),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise Hold30ArtifactError("checkpoint RNG state is incomplete or has unknown fields")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise Hold30ArtifactError("checkpoint NumPy RNG state is malformed")
    keys = numpy_state["keys"]
    if not isinstance(keys, list) or not all(isinstance(value, int) for value in keys):
        raise Hold30ArtifactError("checkpoint NumPy RNG keys are malformed")
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(keys, dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, list) or not all(isinstance(value, int) for value in cpu_state):
        raise Hold30ArtifactError("checkpoint torch CPU RNG state is malformed")
    torch.set_rng_state(torch.tensor(cpu_state, dtype=torch.uint8))
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise Hold30ArtifactError("checkpoint CUDA RNG state must be a list")
    if cuda_states:
        if not torch.cuda.is_initialized():
            raise Hold30ArtifactError("checkpoint contains CUDA RNG state but CUDA is not initialized")
        if len(cuda_states) != torch.cuda.device_count():
            raise Hold30ArtifactError("checkpoint CUDA RNG device count mismatch")
        if not all(
            isinstance(value, list) and all(isinstance(byte, int) for byte in value)
            for value in cuda_states
        ):
            raise Hold30ArtifactError("checkpoint CUDA RNG bytes are malformed")
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8) for value in cuda_states]
        )


def _root_inventory(root: Path) -> tuple[bool, list[int]]:
    allowed = {
        "identity.json",
        "checkpoints",
        _COHORT_MARKER,
        *_INITIAL_NAMES,
        *_FINAL_NAMES,
    }
    unknown = sorted(path.name for path in root.iterdir() if path.name not in allowed)
    if unknown:
        raise Hold30ArtifactError("unknown or duplicate trial artifacts: " + ", ".join(unknown))
    finals = {name for name in _FINAL_NAMES if (root / name).exists()}
    if finals and finals != _FINAL_NAMES:
        raise Hold30ArtifactError("final artifact set is partial")
    initials = {name for name in _INITIAL_NAMES if (root / name).exists()}
    if initials and initials != _INITIAL_NAMES:
        raise Hold30ArtifactError("initial artifact set is partial")
    checkpoint_dir = root / "checkpoints"
    if not checkpoint_dir.exists():
        return bool(finals), []
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise Hold30ArtifactError("checkpoints must be a real directory")
    binary: dict[int, Path] = {}
    receipts: dict[int, Path] = {}
    for path in checkpoint_dir.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        receipt_match = _CHECKPOINT_RECEIPT.fullmatch(path.name)
        if match is not None:
            index = int(match.group(1))
            if index in binary:
                raise Hold30ArtifactError(f"duplicate checkpoint update {index}")
            binary[index] = path
        elif receipt_match is not None:
            index = int(receipt_match.group(1))
            if index in receipts:
                raise Hold30ArtifactError(f"duplicate checkpoint receipt update {index}")
            receipts[index] = path
        else:
            raise Hold30ArtifactError(f"unknown or partial checkpoint artifact: {path.name}")
    if set(binary) != set(receipts):
        raise Hold30ArtifactError("checkpoint binaries and receipts are not complete pairs")
    indices = sorted(binary)
    if indices != list(range(1, len(indices) + 1)):
        raise Hold30ArtifactError("checkpoint updates must be a contiguous one-based prefix")
    return bool(finals), indices


def _checkpoint_paths(root: Path, completed: int) -> tuple[Path, Path]:
    stem = f"update-{completed:06d}"
    return root / "checkpoints" / f"{stem}.pt", root / "checkpoints" / f"{stem}.receipt.json"


def _write_initial_artifact(
    root: Path,
    *,
    identity_receipt_sha256: str,
    sweep_plan_sha256: str,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_states: list[dict[str, Any]],
) -> None:
    model_path = root / "initial-model.pt"
    receipt_path = root / "initial-model.receipt.json"
    payload = {
        "schema": HOLD30_INITIAL_MODEL_SCHEMA,
        "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
        "identity_receipt_sha256": identity_receipt_sha256,
        "sweep_plan_sha256": sweep_plan_sha256,
        "update": 0,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": rng_states,
    }
    _write_new_torch(model_path, payload)
    receipt = _with_receipt_hash(
        {
            "schema": HOLD30_INITIAL_RECEIPT_SCHEMA,
            "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
            "identity_receipt_sha256": identity_receipt_sha256,
            "sweep_plan_sha256": sweep_plan_sha256,
            "update": 0,
            "checkpoint_id": "initial-model.pt",
            "checkpoint_sha256": _sha256_file(model_path),
            "policy_state_sha256": _tree_sha256(payload["policy_state"]),
            "optimizer_state_sha256": _tree_sha256(payload["optimizer_state"]),
            "rng_state_sha256s": [_tree_sha256(value) for value in rng_states],
        }
    )
    _write_new_json(receipt_path, receipt)


def _validate_initial_artifact(
    root: Path,
    *,
    identity_receipt_sha256: str,
    sweep_plan_sha256: str,
    expected_world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = root / "initial-model.pt"
    receipt_path = root / "initial-model.receipt.json"
    payload = _load_torch(model_path)
    expected_payload = {
        "schema",
        "schema_version",
        "identity_receipt_sha256",
        "sweep_plan_sha256",
        "update",
        "policy_state",
        "optimizer_state",
        "rng_states",
    }
    if set(payload) != expected_payload or payload.get("schema") != HOLD30_INITIAL_MODEL_SCHEMA:
        raise Hold30ArtifactError("initial model is partial or has an unknown schema")
    receipt = _read_json(receipt_path)
    _validate_self_hash(receipt, path=receipt_path)
    expected_receipt = {
        "schema",
        "schema_version",
        "identity_receipt_sha256",
        "sweep_plan_sha256",
        "update",
        "checkpoint_id",
        "checkpoint_sha256",
        "policy_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256s",
        "receipt_sha256",
    }
    if set(receipt) != expected_receipt or receipt.get("schema") != HOLD30_INITIAL_RECEIPT_SCHEMA:
        raise Hold30ArtifactError("initial receipt is partial or has an unknown schema")
    checks = {
        "identity_receipt_sha256": identity_receipt_sha256,
        "sweep_plan_sha256": sweep_plan_sha256,
        "update": 0,
    }
    for name, expected in checks.items():
        if payload.get(name) != expected or receipt.get(name) != expected:
            raise Hold30ArtifactError(f"initial artifact {name} mismatch")
    if receipt.get("checkpoint_id") != "initial-model.pt":
        raise Hold30ArtifactError("initial checkpoint ID mismatch")
    if receipt.get("checkpoint_sha256") != _sha256_file(model_path):
        raise Hold30ArtifactError("initial model file digest mismatch")
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, list) or len(rng_states) != expected_world_size:
        raise Hold30ArtifactError("initial RNG world size mismatch")
    digest_checks = {
        "policy_state_sha256": _tree_sha256(payload["policy_state"]),
        "optimizer_state_sha256": _tree_sha256(payload["optimizer_state"]),
        "rng_state_sha256s": [_tree_sha256(value) for value in rng_states],
    }
    for name, expected in digest_checks.items():
        if receipt.get(name) != expected:
            raise Hold30ArtifactError(f"initial receipt {name} mismatch")
    return payload, receipt


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Hold30ArtifactError(f"required checkpoint is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise Hold30ArtifactError(f"checkpoint must be a regular file: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Hold30ArtifactError(f"checkpoint cannot be safely loaded: {path}") from exc
    if not isinstance(value, dict):
        raise Hold30ArtifactError(f"checkpoint payload must be a mapping: {path}")
    return value


def _validate_checkpoint_chain(
    root: Path,
    indices: Sequence[int],
    *,
    identity_receipt_sha256: str,
    sweep_plan_sha256: str,
    state_provider_binding_sha256: str,
    expected_world_size: int,
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    previous_receipt: str | None = None
    terminal: dict[str, Any] | None = None
    receipt_rows: list[dict[str, Any]] = []
    expected_payload_fields = {
        "schema",
        "schema_version",
        "identity_receipt_sha256",
        "sweep_plan_sha256",
        "state_provider_binding_sha256",
        "completed_sweeps",
        "completed_sweep",
        "previous_checkpoint_receipt_sha256",
        "policy_state",
        "optimizer_state",
        "rng_states",
        "metrics",
        "metrics_sha256",
    }
    for completed in indices:
        binary_path, receipt_path = _checkpoint_paths(root, completed)
        receipt = _read_json(receipt_path)
        receipt_hash = _validate_self_hash(receipt, path=receipt_path)
        expected_receipt_fields = {
            "schema",
            "schema_version",
            "identity_receipt_sha256",
            "sweep_plan_sha256",
            "state_provider_binding_sha256",
            "completed_sweeps",
            "completed_sweep",
            "previous_checkpoint_receipt_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "policy_state_sha256",
            "optimizer_state_sha256",
            "rng_state_sha256s",
            "receipt_sha256",
        }
        if set(receipt) != expected_receipt_fields:
            raise Hold30ArtifactError("checkpoint receipt is partial or has unknown fields")
        if receipt.get("schema") != HOLD30_CHECKPOINT_RECEIPT_SCHEMA or receipt.get("schema_version") != 1:
            raise Hold30ArtifactError("unsupported checkpoint receipt schema")
        expected_relative = binary_path.relative_to(root).as_posix()
        if receipt.get("checkpoint_path") != expected_relative:
            raise Hold30ArtifactError("checkpoint receipt path mismatch")
        binary_sha = _sha256_file(binary_path)
        if receipt.get("checkpoint_sha256") != binary_sha:
            raise Hold30ArtifactError(f"checkpoint digest mismatch at update {completed}")
        expected_descriptor = dict(descriptors[completed - 1])
        checks = {
            "identity_receipt_sha256": identity_receipt_sha256,
            "sweep_plan_sha256": sweep_plan_sha256,
            "state_provider_binding_sha256": state_provider_binding_sha256,
            "completed_sweeps": completed,
            "completed_sweep": expected_descriptor,
            "previous_checkpoint_receipt_sha256": previous_receipt,
        }
        for name, expected in checks.items():
            if receipt.get(name) != expected:
                raise Hold30ArtifactError(f"checkpoint receipt {name} mismatch")
        payload = _load_torch(binary_path)
        if set(payload) != expected_payload_fields:
            raise Hold30ArtifactError("checkpoint payload is partial or has unknown fields")
        if payload.get("schema") != HOLD30_CHECKPOINT_SCHEMA or payload.get("schema_version") != 1:
            raise Hold30ArtifactError("unsupported training checkpoint schema")
        for name, expected in checks.items():
            if payload.get(name) != expected:
                raise Hold30ArtifactError(f"checkpoint payload {name} mismatch")
        metrics = payload.get("metrics")
        if not isinstance(metrics, list) or len(metrics) != completed:
            raise Hold30ArtifactError("checkpoint metric prefix is incomplete")
        if payload.get("metrics_sha256") != sha256_payload(metrics):
            raise Hold30ArtifactError("checkpoint metric prefix digest mismatch")
        policy_sha = _tree_sha256(payload.get("policy_state"))
        optimizer_sha = _tree_sha256(payload.get("optimizer_state"))
        rng_states = payload.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != expected_world_size:
            raise Hold30ArtifactError("checkpoint rank RNG state set is malformed")
        rng_shas = [_tree_sha256(value) for value in rng_states]
        for name, expected in (
            ("policy_state_sha256", policy_sha),
            ("optimizer_state_sha256", optimizer_sha),
            ("rng_state_sha256s", rng_shas),
        ):
            if receipt.get(name) != expected:
                raise Hold30ArtifactError(f"checkpoint receipt {name} mismatch")
        previous_receipt = receipt_hash
        terminal = payload
        receipt_rows.append(
            {
                "completed_sweeps": completed,
                "checkpoint_path": expected_relative,
                "checkpoint_sha256": binary_sha,
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_file_sha256": _sha256_file(receipt_path),
                "receipt_sha256": receipt_hash,
            }
        )
    return terminal, receipt_rows


def _write_checkpoint(
    root: Path,
    *,
    identity_receipt_sha256: str,
    sweep_plan_sha256: str,
    state_provider_binding_sha256: str,
    completed_sweep: Mapping[str, Any],
    previous_receipt_sha256: str | None,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
) -> tuple[Path, str]:
    completed = len(metrics)
    binary_path, receipt_path = _checkpoint_paths(root, completed)
    payload = {
        "schema": HOLD30_CHECKPOINT_SCHEMA,
        "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
        "identity_receipt_sha256": identity_receipt_sha256,
        "sweep_plan_sha256": sweep_plan_sha256,
        "state_provider_binding_sha256": state_provider_binding_sha256,
        "completed_sweeps": completed,
        "completed_sweep": dict(completed_sweep),
        "previous_checkpoint_receipt_sha256": previous_receipt_sha256,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": rng_states,
        "metrics": metrics,
        "metrics_sha256": sha256_payload(metrics),
    }
    _write_new_torch(binary_path, payload)
    receipt = _with_receipt_hash(
        {
            "schema": HOLD30_CHECKPOINT_RECEIPT_SCHEMA,
            "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
            "identity_receipt_sha256": identity_receipt_sha256,
            "sweep_plan_sha256": sweep_plan_sha256,
            "state_provider_binding_sha256": state_provider_binding_sha256,
            "completed_sweeps": completed,
            "completed_sweep": dict(completed_sweep),
            "previous_checkpoint_receipt_sha256": previous_receipt_sha256,
            "checkpoint_path": binary_path.relative_to(root).as_posix(),
            "checkpoint_sha256": _sha256_file(binary_path),
            "policy_state_sha256": _tree_sha256(payload["policy_state"]),
            "optimizer_state_sha256": _tree_sha256(payload["optimizer_state"]),
            "rng_state_sha256s": [
                _tree_sha256(value) for value in payload["rng_states"]
            ],
        }
    )
    _write_new_json(receipt_path, receipt)
    return binary_path, receipt["receipt_sha256"]


def _restore_checkpoint(
    payload: Mapping[str, Any],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    try:
        policy.load_state_dict(payload["policy_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise Hold30ArtifactError("checkpoint model/optimizer schema mismatch") from exc
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise Hold30ArtifactError("checkpoint RNG world size mismatch")
    _restore_rng(rng_states[rank])
    return list(payload["metrics"])


def _metric_row(sweep: Hold30TrainingSweep, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "sweep": sweep.descriptor(),
        "optimizer_updates": 1,
        "metrics": _json_safe(metrics),
    }
    row["metric_row_sha256"] = sha256_payload(row)
    return row


def _write_final_artifacts(
    root: Path,
    *,
    identity_document: Mapping[str, Any],
    identity_receipt_sha256: str,
    sweep_plan_sha256: str,
    state_provider: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: list[dict[str, Any]],
) -> Path:
    terminal_receipt = checkpoint_rows[-1]["receipt_sha256"]
    production_update_contract = bool(
        identity_document["driver_contract"]["production_update_contract"]
    )
    qualification_only = not production_update_contract
    final_model = {
        "schema": HOLD30_FINAL_MODEL_SCHEMA,
        "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
        "identity_receipt_sha256": identity_receipt_sha256,
        "sweep_plan_sha256": sweep_plan_sha256,
        "state_provider_binding_sha256": state_provider["binding_sha256"],
        "terminal_checkpoint_receipt_sha256": terminal_receipt,
        "completed_sweeps": len(metrics),
        "checkpoint_selection_status": "unselected-terminal-optimizer-state",
        "production_update_contract": production_update_contract,
        "qualification_only": qualification_only,
        "policy_state": policy.state_dict(),
        "optimizer_state_sha256": _tree_sha256(optimizer.state_dict()),
    }
    final_path = root / "final-model.pt"
    _write_new_torch(final_path, final_model)
    metrics_payload = _with_receipt_hash(
        {
            "schema": HOLD30_METRICS_SCHEMA,
            "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
            "identity_receipt_sha256": identity_receipt_sha256,
            "sweep_plan_sha256": sweep_plan_sha256,
            "state_provider_binding_sha256": state_provider["binding_sha256"],
            "completed_sweeps": len(metrics),
            "validation_metrics": False,
            "production_update_contract": production_update_contract,
            "qualification_only": qualification_only,
            "metrics": metrics,
            "metrics_sha256": sha256_payload(metrics),
        }
    )
    metrics_path = root / "metrics.json"
    _write_new_json(metrics_path, metrics_payload)
    graph = {
        "identity": {
            "path": "identity.json",
            "file_sha256": _sha256_file(root / "identity.json"),
            "receipt_sha256": identity_receipt_sha256,
        },
        "initial_model": {
            "path": "initial-model.pt",
            "file_sha256": _sha256_file(root / "initial-model.pt"),
            "receipt_path": "initial-model.receipt.json",
            "receipt_file_sha256": _sha256_file(root / "initial-model.receipt.json"),
        },
        "checkpoints": list(checkpoint_rows),
        "final_model": {"path": final_path.name, "file_sha256": _sha256_file(final_path)},
        "metrics": {
            "path": metrics_path.name,
            "file_sha256": _sha256_file(metrics_path),
            "receipt_sha256": metrics_payload["receipt_sha256"],
        },
    }
    receipt = _with_receipt_hash(
        {
            "schema": HOLD30_RUN_RECEIPT_SCHEMA,
            "schema_version": HOLD30_DRIVER_SCHEMA_VERSION,
            "protocol_generation": HOLD30_PROTOCOL_GENERATION,
            "trial": identity_document["trial"],
            "trial_sha256": identity_document["trial_sha256"],
            "sweep_plan_sha256": sweep_plan_sha256,
            "state_provider": dict(state_provider),
            "state_provider_binding_sha256": state_provider["binding_sha256"],
            "completed_sweeps": len(metrics),
            "optimizer_updates": len(metrics),
            "terminal_checkpoint_receipt_sha256": terminal_receipt,
            "artifact_graph": graph,
            "artifact_graph_sha256": sha256_payload(graph),
            "optimization_sweeps_complete": True,
            "end_to_end_validation_complete": False,
            "validation_checkpoint_selected": False,
            "production_update_contract": production_update_contract,
            "qualification_only": qualification_only,
            "scientific_qualification": False,
            "promotion_authorized": False,
            "gpu_launch_performed": False,
        }
    )
    receipt_path = root / "run-receipt.json"
    _write_new_json(receipt_path, receipt)
    return receipt_path


def run_hold30_trial(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    adapter: Hold30ReplayAdapter,
    identity: Hold30TrialIdentity,
    sweeps: Sequence[Hold30TrainingSweep],
    output_dir: str | Path,
    *,
    state_provider_binding: Hold30StateProviderBinding,
    geometry: Hold30ReplayGeometry | None = None,
    grad_clip: float = 0.5,
    resume: bool = False,
    max_sweeps: int | None = None,
    world_size: int = 1,
    rank: int = 0,
    qualification_update_override: int | None = None,
) -> Hold30RunProgress:
    """Run or resume one exact setting/fold/seed trial.

    ``max_sweeps`` is an interruption/testing control for the current
    invocation only.  It is deliberately absent from the scientific identity.
    A normal production invocation leaves it as ``None``.
    """

    distributed_context = _distributed_context(world_size, rank)
    if isinstance(policy, torch.nn.parallel.DistributedDataParallel):
        raise ValueError("Hold-30 performs explicit SUM reduction and rejects DDP wrappers")
    geometry = geometry or Hold30ReplayGeometry()
    if qualification_update_override is not None and (
        isinstance(qualification_update_override, bool)
        or not isinstance(qualification_update_override, int)
        or qualification_update_override <= 0
    ):
        raise ValueError("qualification_update_override must be a positive integer")
    expected_updates = (
        HOLD30_PRODUCTION_OPTIMIZER_UPDATES
        if qualification_update_override is None
        else qualification_update_override
    )
    descriptors = _validate_sweeps(
        sweeps,
        geometry,
        expected_updates=expected_updates,
    )
    driver_contract = _driver_contract(
        identity,
        geometry,
        grad_clip,
        policy,
        optimizer,
        adapter,
        state_provider_binding,
        distributed_context,
        expected_updates,
    )
    state_provider = driver_contract["state_provider"]
    state_provider_binding_sha256 = state_provider["binding_sha256"]
    root = Path(output_dir)
    if max_sweeps is not None and (
        isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or max_sweeps <= 0
    ):
        raise ValueError("max_sweeps must be a positive integer or None")
    _distributed_require_equal(
        distributed_context,
        "trial/provider/data/optimizer identity",
        {
            "trial": identity.payload(),
            "sweeps": list(descriptors),
            "driver_contract": driver_contract,
            "resume": resume,
            "max_sweeps": max_sweeps,
        },
    )
    state_hashes = _all_rank_values(
        distributed_context,
        {
            "policy": _tree_sha256(policy.state_dict()),
            "optimizer": _tree_sha256(optimizer.state_dict()),
        },
    )
    if any(value != state_hashes[0] for value in state_hashes[1:]):
        raise Hold30ArtifactError("distributed ranks began from different model/optimizer state")

    def prepare_root() -> tuple[bool, list[int]]:
        if root.exists() and (not root.is_dir() or root.is_symlink()):
            raise Hold30ArtifactError("output_dir must be a real directory")
        root.mkdir(parents=True, exist_ok=True)
        return _root_inventory(root)

    final_present, indices = _rank0_call(distributed_context, prepare_root)
    if distributed_context.enabled:
        dist.barrier()
        observed_inventory = _root_inventory(root)
        # Keep rank zero from publishing identity.json while another rank is
        # still performing this fail-closed directory scan.  Atomic writers
        # intentionally leave a visible temporary file until replacement.
        dist.barrier()
        if observed_inventory != (final_present, indices):
            raise Hold30ArtifactError("distributed rank sees a different artifact inventory")
    if (root / _COHORT_MARKER).exists():
        raise Hold30ArtifactError("a cohort-finalized trial cannot resume optimizer updates")
    identity_path = root / "identity.json"
    if resume:
        if final_present:
            raise Hold30ArtifactError("a completed trial cannot be resumed")
        if not identity_path.exists() or not indices:
            raise Hold30ArtifactError("resume requires an intact checkpoint prefix")
        identity_document = _read_json(identity_path)
        identity_receipt = _validate_identity_document(
            identity_document,
            path=identity_path,
            identity=identity,
            descriptors=descriptors,
            driver_contract=driver_contract,
        )
        _validate_initial_artifact(
            root,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            expected_world_size=world_size,
        )
        terminal, checkpoint_rows = _validate_checkpoint_chain(
            root,
            indices,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            state_provider_binding_sha256=state_provider_binding_sha256,
            expected_world_size=world_size,
            descriptors=descriptors,
        )
        assert terminal is not None
        metrics = _restore_checkpoint(
            terminal,
            policy,
            optimizer,
            rank=rank,
            world_size=world_size,
        )
        completed = len(metrics)
        previous_receipt = checkpoint_rows[-1]["receipt_sha256"]
    else:
        if any(root.iterdir()):
            raise Hold30ArtifactError("new trials require an empty output directory")
        if distributed_context.enabled:
            # Every rank must finish the empty-root assertion before rank zero
            # begins the first exclusive publication.
            dist.barrier()
        _seed_trial(identity.seed + rank)
        identity_document = _identity_document(
            identity, descriptors, driver_contract, policy, optimizer
        )
        _rank0_call(
            distributed_context,
            lambda: _write_new_json(identity_path, identity_document),
        )
        if distributed_context.enabled:
            dist.barrier()
        identity_document = _read_json(identity_path)
        identity_receipt = _validate_identity_document(
            identity_document,
            path=identity_path,
            identity=identity,
            descriptors=descriptors,
            driver_contract=driver_contract,
        )
        initial_rng_states = _all_rank_values(distributed_context, _capture_rng())
        _rank0_call(
            distributed_context,
            lambda: _write_initial_artifact(
                root,
                identity_receipt_sha256=identity_receipt,
                sweep_plan_sha256=identity_document["sweep_plan_sha256"],
                policy=policy,
                optimizer=optimizer,
                rng_states=initial_rng_states,
            ),
        )
        if distributed_context.enabled:
            dist.barrier()
        _validate_initial_artifact(
            root,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            expected_world_size=world_size,
        )
        checkpoint_rows = []
        metrics = []
        completed = 0
        previous_receipt = None

    remaining = len(sweeps) - completed
    if remaining < 0:
        raise Hold30ArtifactError("checkpoint contains more sweeps than the frozen plan")
    budget = remaining if max_sweeps is None else min(remaining, max_sweeps)
    contract = Hold30LossContract.for_setting(identity.setting_id)
    for sweep in sweeps[completed:completed + budget]:
        update_metrics = train_hold30_update(
            policy,
            sweep.sequence,
            adapter,
            optimizer,
            n_positions=sweep.n_positions,
            contract=contract,
            geometry=geometry,
            grad_clip=float(grad_clip),
            distributed_world_size=world_size,
            distributed_rank=rank,
        )
        if update_metrics.get("optimizer_steps") != 1:
            raise Hold30ArtifactError("Hold-30 core did not perform exactly one optimizer step")
        metric_row = _metric_row(sweep, update_metrics)
        metric_rows = _all_rank_values(distributed_context, metric_row)
        if any(value != metric_rows[0] for value in metric_rows[1:]):
            raise Hold30ArtifactError("distributed metric aggregation is not exact")
        metrics.append(metric_rows[0])
        synchronized_states = _all_rank_values(
            distributed_context,
            {
                "policy": _tree_sha256(policy.state_dict()),
                "optimizer": _tree_sha256(optimizer.state_dict()),
            },
        )
        if any(value != synchronized_states[0] for value in synchronized_states[1:]):
            raise Hold30ArtifactError("SUM-reduced ranks produced different optimizer states")
        rng_states = _all_rank_values(distributed_context, _capture_rng())
        _rank0_call(
            distributed_context,
            lambda: _write_checkpoint(
                root,
                identity_receipt_sha256=identity_receipt,
                sweep_plan_sha256=identity_document["sweep_plan_sha256"],
                state_provider_binding_sha256=state_provider_binding_sha256,
                completed_sweep=sweep.descriptor(),
                previous_receipt_sha256=previous_receipt,
                policy=policy,
                optimizer=optimizer,
                metrics=metrics,
                rng_states=rng_states,
            ),
        )
        if distributed_context.enabled:
            dist.barrier()
        completed += 1
        _, current_indices = _root_inventory(root)
        terminal, checkpoint_rows = _validate_checkpoint_chain(
            root,
            current_indices,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            state_provider_binding_sha256=state_provider_binding_sha256,
            expected_world_size=world_size,
            descriptors=descriptors,
        )
        if terminal is None or len(current_indices) != completed:
            raise Hold30ArtifactError("new checkpoint is not a complete append-only prefix")
        previous_receipt = checkpoint_rows[-1]["receipt_sha256"]

    if completed < len(sweeps):
        latest, _ = _checkpoint_paths(root, completed)
        return Hold30RunProgress(False, completed, len(sweeps), latest, None)

    # Re-read the complete append-only chain before binding final artifacts.
    final_present, indices = _root_inventory(root)
    if final_present:
        raise Hold30ArtifactError("final artifacts appeared concurrently")
    terminal, checkpoint_rows = _validate_checkpoint_chain(
        root,
        indices,
        identity_receipt_sha256=identity_receipt,
        sweep_plan_sha256=identity_document["sweep_plan_sha256"],
        state_provider_binding_sha256=state_provider_binding_sha256,
        expected_world_size=world_size,
        descriptors=descriptors,
    )
    if terminal is None or len(indices) != len(sweeps):
        raise Hold30ArtifactError("checkpoint graph is incomplete at finalization")
    if terminal["metrics"] != metrics:
        raise Hold30ArtifactError("in-memory metrics differ from terminal checkpoint")
    receipt_path = root / "run-receipt.json"
    _rank0_call(
        distributed_context,
        lambda: _write_final_artifacts(
            root,
            identity_document=identity_document,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            state_provider=state_provider,
            checkpoint_rows=checkpoint_rows,
            policy=policy,
            optimizer=optimizer,
            metrics=metrics,
        ),
    )
    if distributed_context.enabled:
        dist.barrier()
    verify_hold30_run(
        root,
        expected_identity=identity,
        allow_qualification_only=driver_contract["qualification_only"],
    )
    latest, _ = _checkpoint_paths(root, completed)
    return Hold30RunProgress(True, completed, len(sweeps), latest, receipt_path)


def verify_hold30_run(
    output_dir: str | Path,
    *,
    expected_identity: Hold30TrialIdentity | None = None,
    allow_qualification_only: bool = False,
) -> dict[str, Any]:
    """Verify a complete trial and every edge in its artifact graph."""

    root = Path(output_dir)
    if not root.is_dir() or root.is_symlink():
        raise Hold30ArtifactError("trial directory is absent or unsafe")
    final_present, indices = _root_inventory(root)
    if not final_present:
        return _verify_cohort_finalized_prefix(
            root,
            indices,
            expected_identity=expected_identity,
        )
    identity_path = root / "identity.json"
    identity_document = _read_json(identity_path)
    identity_receipt = _validate_self_hash(identity_document, path=identity_path)
    expected_identity_fields = {
        "schema",
        "schema_version",
        "trial",
        "trial_sha256",
        "sweep_plan",
        "sweep_plan_sha256",
        "driver_contract",
        "driver_contract_sha256",
        "initial_policy_state_sha256",
        "initial_optimizer_state_sha256",
        "receipt_sha256",
    }
    if set(identity_document) != expected_identity_fields:
        raise Hold30ArtifactError("identity document is partial or has unknown fields")
    if identity_document.get("schema") != HOLD30_IDENTITY_SCHEMA:
        raise Hold30ArtifactError("unsupported Hold-30 identity schema")
    if expected_identity is not None and identity_document.get("trial") != expected_identity.payload():
        raise Hold30ArtifactError("completed trial identity does not match expectation")
    descriptors = identity_document.get("sweep_plan")
    if not isinstance(descriptors, list) or not descriptors:
        raise Hold30ArtifactError("identity has no chronological sweep plan")
    if identity_document.get("sweep_plan_sha256") != sha256_payload(descriptors):
        raise Hold30ArtifactError("identity sweep-plan digest mismatch")
    driver_contract = identity_document.get("driver_contract")
    if not isinstance(driver_contract, dict):
        raise Hold30ArtifactError("identity has no driver contract")
    if identity_document.get("driver_contract_sha256") != sha256_payload(driver_contract):
        raise Hold30ArtifactError("identity driver-contract digest mismatch")
    state_provider = driver_contract.get("state_provider")
    if not isinstance(state_provider, dict):
        raise Hold30ArtifactError("identity has no state-provider binding")
    unsigned_provider = dict(state_provider)
    provider_binding_sha256 = unsigned_provider.pop("binding_sha256", None)
    if provider_binding_sha256 != sha256_payload(unsigned_provider):
        raise Hold30ArtifactError("state-provider binding digest mismatch")
    if (
        state_provider.get("trains_upstream_encoder") is not True
        or state_provider.get("require_trainable_state_provider") is not True
    ):
        raise Hold30ArtifactError("state-provider binding is not trainable")
    distributed_contract = driver_contract.get("distributed")
    if not isinstance(distributed_contract, dict) or distributed_contract.get("world_size") not in {1, 2}:
        raise Hold30ArtifactError("identity distributed contract is malformed")
    artifact_world_size = int(distributed_contract["world_size"])
    _validate_initial_artifact(
        root,
        identity_receipt_sha256=identity_receipt,
        sweep_plan_sha256=identity_document["sweep_plan_sha256"],
        expected_world_size=artifact_world_size,
    )
    production_update_contract = driver_contract.get("production_update_contract")
    qualification_only = driver_contract.get("qualification_only")
    if not isinstance(production_update_contract, bool) or qualification_only is not (
        not production_update_contract
    ):
        raise Hold30ArtifactError("identity update-count qualification contract is malformed")
    if qualification_only and not allow_qualification_only:
        raise Hold30ArtifactError(
            "qualification-only trial cannot pass production verification"
        )
    optimizer_update_count = driver_contract.get("optimizer_update_count")
    if (
        isinstance(optimizer_update_count, bool)
        or not isinstance(optimizer_update_count, int)
        or optimizer_update_count != len(descriptors)
        or production_update_contract
        != (optimizer_update_count == HOLD30_PRODUCTION_OPTIMIZER_UPDATES)
    ):
        raise Hold30ArtifactError("frozen optimizer-update count is inconsistent")
    if len({row.get("sequence_sha256") for row in descriptors}) != 1:
        raise Hold30ArtifactError("optimizer updates do not reuse one chronological sequence")
    if len(indices) != len(descriptors):
        raise Hold30ArtifactError("checkpoint count does not match frozen sweep plan")
    terminal, checkpoint_rows = _validate_checkpoint_chain(
        root,
        indices,
        identity_receipt_sha256=identity_receipt,
        sweep_plan_sha256=identity_document["sweep_plan_sha256"],
        state_provider_binding_sha256=provider_binding_sha256,
        expected_world_size=artifact_world_size,
        descriptors=descriptors,
    )
    assert terminal is not None
    metrics_path = root / "metrics.json"
    metrics = _read_json(metrics_path)
    metrics_receipt = _validate_self_hash(metrics, path=metrics_path)
    if metrics.get("schema") != HOLD30_METRICS_SCHEMA or metrics.get("schema_version") != 1:
        raise Hold30ArtifactError("unsupported metrics schema")
    expected_metric_fields = {
        "schema",
        "schema_version",
        "identity_receipt_sha256",
        "sweep_plan_sha256",
        "state_provider_binding_sha256",
        "completed_sweeps",
        "validation_metrics",
        "production_update_contract",
        "qualification_only",
        "metrics",
        "metrics_sha256",
        "receipt_sha256",
    }
    if set(metrics) != expected_metric_fields:
        raise Hold30ArtifactError("metrics artifact is partial or has unknown fields")
    if metrics["identity_receipt_sha256"] != identity_receipt:
        raise Hold30ArtifactError("metrics identity mismatch")
    if metrics["sweep_plan_sha256"] != identity_document["sweep_plan_sha256"]:
        raise Hold30ArtifactError("metrics sweep-plan mismatch")
    if metrics["state_provider_binding_sha256"] != provider_binding_sha256:
        raise Hold30ArtifactError("metrics state-provider mismatch")
    if metrics["completed_sweeps"] != len(descriptors):
        raise Hold30ArtifactError("metrics completed-sweep count mismatch")
    if metrics["validation_metrics"] is not False:
        raise Hold30ArtifactError("optimizer metrics cannot claim validation")
    if metrics["production_update_contract"] is not production_update_contract:
        raise Hold30ArtifactError("metrics production-update contract mismatch")
    if metrics["qualification_only"] is not qualification_only:
        raise Hold30ArtifactError("metrics qualification-only status mismatch")
    if metrics["metrics_sha256"] != sha256_payload(metrics["metrics"]):
        raise Hold30ArtifactError("metrics payload digest mismatch")
    if metrics["metrics"] != terminal["metrics"]:
        raise Hold30ArtifactError("metrics artifact differs from terminal checkpoint")
    final_path = root / "final-model.pt"
    final_model = _load_torch(final_path)
    expected_final_fields = {
        "schema",
        "schema_version",
        "identity_receipt_sha256",
        "sweep_plan_sha256",
        "state_provider_binding_sha256",
        "terminal_checkpoint_receipt_sha256",
        "completed_sweeps",
        "checkpoint_selection_status",
        "production_update_contract",
        "qualification_only",
        "policy_state",
        "optimizer_state_sha256",
    }
    if set(final_model) != expected_final_fields:
        raise Hold30ArtifactError("final model is partial or has unknown fields")
    if final_model.get("schema") != HOLD30_FINAL_MODEL_SCHEMA or final_model.get("schema_version") != 1:
        raise Hold30ArtifactError("unsupported final-model schema")
    terminal_receipt = checkpoint_rows[-1]["receipt_sha256"]
    final_checks = {
        "identity_receipt_sha256": identity_receipt,
        "sweep_plan_sha256": identity_document["sweep_plan_sha256"],
        "state_provider_binding_sha256": provider_binding_sha256,
        "terminal_checkpoint_receipt_sha256": terminal_receipt,
        "completed_sweeps": len(descriptors),
        "checkpoint_selection_status": "unselected-terminal-optimizer-state",
        "production_update_contract": production_update_contract,
        "qualification_only": qualification_only,
        "optimizer_state_sha256": _tree_sha256(terminal["optimizer_state"]),
    }
    for name, expected in final_checks.items():
        if final_model.get(name) != expected:
            raise Hold30ArtifactError(f"final-model {name} mismatch")
    if _tree_sha256(final_model["policy_state"]) != _tree_sha256(terminal["policy_state"]):
        raise Hold30ArtifactError("final policy differs from terminal checkpoint")
    receipt_path = root / "run-receipt.json"
    receipt = _read_json(receipt_path)
    _validate_self_hash(receipt, path=receipt_path)
    if receipt.get("schema") != HOLD30_RUN_RECEIPT_SCHEMA or receipt.get("schema_version") != 1:
        raise Hold30ArtifactError("unsupported run-receipt schema")
    graph = {
        "identity": {
            "path": "identity.json",
            "file_sha256": _sha256_file(identity_path),
            "receipt_sha256": identity_receipt,
        },
        "initial_model": {
            "path": "initial-model.pt",
            "file_sha256": _sha256_file(root / "initial-model.pt"),
            "receipt_path": "initial-model.receipt.json",
            "receipt_file_sha256": _sha256_file(root / "initial-model.receipt.json"),
        },
        "checkpoints": checkpoint_rows,
        "final_model": {"path": final_path.name, "file_sha256": _sha256_file(final_path)},
        "metrics": {
            "path": metrics_path.name,
            "file_sha256": _sha256_file(metrics_path),
            "receipt_sha256": metrics_receipt,
        },
    }
    receipt_checks = {
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "trial": identity_document["trial"],
        "trial_sha256": identity_document["trial_sha256"],
        "sweep_plan_sha256": identity_document["sweep_plan_sha256"],
        "state_provider": state_provider,
        "state_provider_binding_sha256": provider_binding_sha256,
        "completed_sweeps": len(descriptors),
        "optimizer_updates": len(descriptors),
        "terminal_checkpoint_receipt_sha256": terminal_receipt,
        "artifact_graph": graph,
        "artifact_graph_sha256": sha256_payload(graph),
        "optimization_sweeps_complete": True,
        "end_to_end_validation_complete": False,
        "validation_checkpoint_selected": False,
        "production_update_contract": production_update_contract,
        "qualification_only": qualification_only,
        "scientific_qualification": False,
        "promotion_authorized": False,
        "gpu_launch_performed": False,
    }
    expected_receipt_fields = {
        "schema",
        "schema_version",
        "receipt_sha256",
        *receipt_checks,
    }
    if set(receipt) != expected_receipt_fields:
        raise Hold30ArtifactError("run receipt is partial or has unknown fields")
    for name, expected in receipt_checks.items():
        if receipt.get(name) != expected:
            raise Hold30ArtifactError(f"run receipt {name} mismatch")
    return receipt


def _verify_cohort_finalized_prefix(
    root: Path,
    indices: Sequence[int],
    *,
    expected_identity: Hold30TrialIdentity | None,
) -> dict[str, Any]:
    from rl_quant.training.hold30_coordinator import (
        HOLD30_TRIAL_FINALIZATION_SCHEMA,
        verify_hold30_cohort_finalization,
    )

    marker_path = root / _COHORT_MARKER
    if not marker_path.is_file() or marker_path.is_symlink():
        raise Hold30ArtifactError(
            "trial is an interrupted prefix without a verified cohort finalization"
        )
    marker = _read_json(marker_path)
    _validate_self_hash(marker, path=marker_path)
    expected_fields = {
        "schema",
        "schema_version",
        "protocol_generation",
        "seed",
        "cohort_finalization_path",
        "cohort_finalization_file_sha256",
        "cohort_finalization_receipt_sha256",
        "selected_update",
        "stopped_update",
        "stop_reason",
        "receipt_sha256",
    }
    if set(marker) != expected_fields or marker.get("schema") != HOLD30_TRIAL_FINALIZATION_SCHEMA:
        raise Hold30ArtifactError("trial cohort-finalization marker is malformed")
    relative = marker.get("cohort_finalization_path")
    if not isinstance(relative, str) or not relative:
        raise Hold30ArtifactError("trial cohort-finalization path is absent")
    receipt_path = (root.resolve() / relative).resolve()
    if _sha256_file(receipt_path) != marker.get("cohort_finalization_file_sha256"):
        raise Hold30ArtifactError("cohort-finalization file digest mismatch")
    finalization = _read_json(receipt_path)
    try:
        outcome = verify_hold30_cohort_finalization(
            finalization,
            receipt_path=receipt_path,
        )
    except Exception as exc:
        raise Hold30ArtifactError("cohort-finalization receipt verification failed") from exc
    if finalization.get("receipt_sha256") != marker.get(
        "cohort_finalization_receipt_sha256"
    ):
        raise Hold30ArtifactError("trial marker references a different cohort receipt")
    trial_identity = Hold30TrialIdentity(
        setting_id=outcome.identity.setting_id,
        fold_index=outcome.identity.fold_index,
        seed=int(marker["seed"]),
        executable_manifest_sha256=outcome.identity.executable_manifest_sha256,
        fold_sha256=outcome.identity.fold_sha256,
    )
    if expected_identity is not None and trial_identity != expected_identity:
        raise Hold30ArtifactError("cohort-finalized trial identity does not match expectation")
    row = next(
        (
            value
            for value in finalization["trial_artifacts"]
            if value["seed"] == trial_identity.seed
            and (receipt_path.parent / value["trial_root"]).resolve() == root.resolve()
        ),
        None,
    )
    if row is None:
        raise Hold30ArtifactError("cohort receipt does not bind this trial root")
    expected_marker = {
        "selected_update": finalization["selected_update"],
        "stopped_update": finalization["stopped_update"],
        "stop_reason": finalization["stop_reason"],
    }
    if any(marker.get(name) != value for name, value in expected_marker.items()):
        raise Hold30ArtifactError("trial marker stop/selection binding mismatch")
    if len(indices) != finalization["stopped_update"]:
        raise Hold30ArtifactError("trial prefix length differs from the cohort stop update")
    return {
        "schema": "rl-quant.hold30.cohort-finalized-trial-verification",
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "trial": trial_identity.payload(),
        "optimization_sweeps_complete": False,
        "cohort_early_stop_finalized": True,
        "validation_checkpoint_selected": True,
        "selected_update": finalization["selected_update"],
        "stopped_update": finalization["stopped_update"],
        "stop_reason": finalization["stop_reason"],
        "cohort_finalization_receipt_sha256": finalization["receipt_sha256"],
        "scientific_qualification": False,
        "promotion_authorized": False,
    }


def inspect_hold30_trial_checkpoints(
    output_dir: str | Path,
    updates: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Verify one trial chain once and return multiple retained references."""

    requested = tuple(updates)
    if not requested or any(
        isinstance(update, bool) or not isinstance(update, int) or update < 0
        for update in requested
    ):
        raise ValueError("updates must be non-empty non-negative integers")
    if len(set(requested)) != len(requested):
        raise ValueError("updates must be unique")
    root = Path(output_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise Hold30ArtifactError("trial directory is absent or unsafe")
    _, indices = _root_inventory(root)
    identity_path = root / "identity.json"
    identity_document = _read_json(identity_path)
    identity_receipt = _validate_self_hash(identity_document, path=identity_path)
    driver_contract = identity_document.get("driver_contract")
    if not isinstance(driver_contract, dict):
        raise Hold30ArtifactError("trial identity has no driver contract")
    if driver_contract.get("production_update_contract") is not True:
        raise Hold30ArtifactError("cohort selection requires a production 128-update plan")
    world_size = driver_contract.get("distributed", {}).get("world_size")
    if world_size not in {1, 2}:
        raise Hold30ArtifactError("trial distributed world-size binding is invalid")
    state_provider = driver_contract.get("state_provider")
    if not isinstance(state_provider, dict) or not isinstance(
        state_provider.get("binding_sha256"), str
    ):
        raise Hold30ArtifactError("trial state-provider binding is invalid")
    _validate_initial_artifact(
        root,
        identity_receipt_sha256=identity_receipt,
        sweep_plan_sha256=identity_document["sweep_plan_sha256"],
        expected_world_size=world_size,
    )
    positive = tuple(update for update in requested if update)
    for update in positive:
        if update not in indices:
            raise Hold30ArtifactError(f"trial does not retain checkpoint update {update}")
    rows: list[dict[str, Any]] = []
    if positive:
        _, rows = _validate_checkpoint_chain(
            root,
            indices,
            identity_receipt_sha256=identity_receipt,
            sweep_plan_sha256=identity_document["sweep_plan_sha256"],
            state_provider_binding_sha256=state_provider["binding_sha256"],
            expected_world_size=world_size,
            descriptors=identity_document["sweep_plan"],
        )
    trial = identity_document.get("trial")
    if not isinstance(trial, dict):
        raise Hold30ArtifactError("trial identity payload is malformed")
    result: dict[int, dict[str, Any]] = {}
    for update in requested:
        if update == 0:
            checkpoint_path = root / "initial-model.pt"
            receipt_path = root / "initial-model.receipt.json"
        else:
            row = rows[update - 1]
            checkpoint_path = root / row["checkpoint_path"]
            receipt_path = root / row["receipt_path"]
        receipt = _read_json(receipt_path)
        _validate_self_hash(receipt, path=receipt_path)
        result[update] = {
            "trial_root": str(root),
            "trial": trial,
            "identity_receipt_sha256": identity_receipt,
            "retained_update_count": len(indices),
            "seed": trial["seed"],
            "update": update,
            "checkpoint_id": (
                f"seed-{trial['seed']}:"
                f"{checkpoint_path.relative_to(root).as_posix()}"
            ),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_receipt_sha256": receipt["receipt_sha256"],
            "checkpoint_receipt_file_sha256": _sha256_file(receipt_path),
        }
    return result


def inspect_hold30_trial_checkpoint(
    output_dir: str | Path,
    update: int,
) -> dict[str, Any]:
    """Return a verified coordinator reference for one retained checkpoint."""

    return inspect_hold30_trial_checkpoints(output_dir, (update,))[update]


__all__ = [
    "HOLD30_DRIVER_SCHEMA_VERSION",
    "HOLD30_PRODUCTION_OPTIMIZER_UPDATES",
    "Hold30ArtifactError",
    "Hold30RunProgress",
    "Hold30StateProviderBinding",
    "Hold30TrainingSweep",
    "Hold30TrialIdentity",
    "inspect_hold30_trial_checkpoint",
    "inspect_hold30_trial_checkpoints",
    "run_hold30_trial",
    "verify_hold30_run",
]
