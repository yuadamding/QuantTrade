"""Receipt-complete compact Stage-1 pretraining for Hold-30 v2.

The Hold-30 mechanism screen shares one frozen raw-OHLCV context encoder per
outer fold.  The v2 freeze defines the fold-seed derivation, 8-setting x
5-seed sharing scope, explicit normalization/update schedule artifacts, and
50-update checkpoint cadence.  A caller must supply those pre-frozen,
receipt-bound artifacts through :class:`Hold30Stage1FreezeDecision`; this
module validates the normative formulas and never substitutes defaults.

Only expanding-fold training days enter this API.  Thirty-session daily
targets are rebuilt from the supplied training close matrix, so the final 31
rows (one-day execution delay plus 30-day horizon) are censored without
borrowing validation, outer-score, or lockbox observations.  The exact date
sampling schedule is materialized once, content-addressed, and replayed by
``train_context_encoder``.  Checkpoints and receipts are append-only and form
one verifiable hash chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from rl_quant.models.context_encoder import ContextEncoder, ContextForwardHead
from rl_quant.models.daily_policy import DailyForwardHead
from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import HOLD30_FOLDS, sha256_payload
from rl_quant.training._optim import make_adamw
from rl_quant.training.context_pretrain import (
    freeze_encoder,
    ssl_targets_daily,
    train_context_encoder,
)
from rl_quant.training.designs import DESIGNS
from rl_quant.training.hold30_experiment import (
    HOLD30_BASE_DESIGN,
    build_hold30_context_config,
)

HOLD30_STAGE1_SCHEMA_VERSION = 1
HOLD30_STAGE1_STEPS = 1_000
HOLD30_STAGE1_BATCH_SIZE = 3
HOLD30_STAGE1_ACCUMULATION = 12
HOLD30_STAGE1_EFFECTIVE_BATCH = 36
HOLD30_STAGE1_LR = 2e-4
HOLD30_STAGE1_WEIGHT_DECAY = 1e-2
HOLD30_STAGE1_WARMUP_STEPS = 50
HOLD30_STAGE1_SCHEDULE = "constant"
HOLD30_STAGE1_DAILY_COEFFICIENT = 1.0
HOLD30_STAGE1_PER_STOCK_COEFFICIENT = 0.0
HOLD30_STAGE1_HORIZON = 30
HOLD30_STAGE1_EXECUTION_DELAY = 1
HOLD30_STAGE1_COVARIATE_DIM = 0
HOLD30_STAGE1_SHARING_SCOPE = "one-shared-context-per-outer-fold-8-settings-x-5-seeds"
HOLD30_STAGE1_EXECUTION_STRATEGY = (
    "two-rank-alternating-2-1-global-valid-denominator-sum-reduce-v1"
)
HOLD30_STAGE1_CHECKPOINT_EVERY = 50

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT = re.compile(r"step-([0-9]{6})\.pt\Z")
_CHECKPOINT_RECEIPT = re.compile(r"step-([0-9]{6})\.receipt\.json\Z")
_ALLOWED_DAY_FIELDS = frozenset(
    {
        "bars",
        "bar_mask",
        "cov_blocks",
        "ret",
        "ret_valid",
        "session_close_block",
    }
)
_FORBIDDEN_DAY_FIELD_TOKENS = ("outer", "validation", "test", "lockbox", "news")
_FREEZE_DECISION_FIELDS = (
    "stage1_seed",
    "sharing_scope",
    "sharing_manifest_sha256",
    "normalization_day_indices",
    "normalization_schedule_sha256",
    "optimizer_schedule_sha256",
    "checkpoint_every",
    "execution_strategy",
    "approval_receipt_sha256",
)


class Hold30Stage1Error(RuntimeError):
    """Stage-1 inputs or durable artifacts violate the frozen contract."""


class Hold30Stage1FreezeBlocker(Hold30Stage1Error):
    """A scientifically result-moving Stage-1 choice lacks explicit approval."""


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def derive_hold30_stage1_seed(fold_index: int) -> int:
    """Derive the frozen fold seed as the first SHA-256 uint64, big-endian.

    The hashed byte string is exactly ``UTF8(protocol_generation) || 0x00 ||
    fold_index:u16be || 0x00 || UTF8('stage1')``.  The first eight digest
    bytes are interpreted as one unsigned big-endian integer.
    """

    if (
        isinstance(fold_index, bool)
        or not isinstance(fold_index, int)
        or not 0 <= fold_index < HOLD30_FOLDS
    ):
        raise ValueError(f"fold_index must be in [0, {HOLD30_FOLDS - 1}]")
    digest = hashlib.sha256()
    digest.update(HOLD30_PROTOCOL_GENERATION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(fold_index.to_bytes(2, byteorder="big", signed=False))
    digest.update(b"\x00")
    digest.update(b"stage1")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    rows = [
        (key, str(value.dtype), list(value.shape), _tensor_sha256(value))
        for key, value in sorted(state.items())
    ]
    return sha256_payload(rows)


def _with_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise Hold30Stage1Error("receipt payload already contains receipt_sha256")
    result = dict(payload)
    result["receipt_sha256"] = sha256_payload(result)
    return result


def _validate_receipt(payload: Mapping[str, Any], path: Path) -> str:
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or _DIGEST.fullmatch(claimed) is None:
        raise Hold30Stage1Error(f"{path} lacks a valid receipt_sha256")
    unsigned = dict(payload)
    del unsigned["receipt_sha256"]
    if sha256_payload(unsigned) != claimed:
        raise Hold30Stage1Error(f"{path} receipt hash mismatch")
    return claimed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Hold30Stage1Error(f"required artifact is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise Hold30Stage1Error(f"artifact must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Hold30Stage1Error(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Hold30Stage1Error(f"JSON artifact must contain an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    """Durably create ``path`` without ever replacing an existing artifact."""

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
            raise Hold30Stage1Error(f"refusing to overwrite existing artifact: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    def write(stream: Any) -> None:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

    _publish_new(path, write, binary=False)


def _write_new_torch(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_new(path, lambda stream: torch.save(dict(payload), stream), binary=True)


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Hold30Stage1Error(f"required artifact is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise Hold30Stage1Error(f"artifact must be a regular file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Hold30Stage1Error(f"unreadable torch artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Hold30Stage1Error(f"torch artifact must contain a mapping: {path}")
    return payload


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu").clone()
        for key, value in module.state_dict().items()
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda", [])
    if cuda_state:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise Hold30Stage1Error("checkpoint CUDA RNG topology differs from this process")
        torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True, slots=True)
class Hold30Stage1FreezeDecision:
    """Explicit choices missing from the published scientific specification."""

    stage1_seed: int
    sharing_scope: str
    sharing_manifest_sha256: str
    normalization_day_indices: tuple[int, ...]
    normalization_schedule_sha256: str
    optimizer_schedule_sha256: str
    checkpoint_every: int
    execution_strategy: str
    approval_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.stage1_seed, bool)
            or not isinstance(self.stage1_seed, int)
            or not 0 <= self.stage1_seed < 2**64
        ):
            raise Hold30Stage1FreezeBlocker(
                "stage1_seed requires an explicit uint64 integer"
            )
        if self.sharing_scope != HOLD30_STAGE1_SHARING_SCOPE:
            raise Hold30Stage1FreezeBlocker(
                f"sharing_scope must be {HOLD30_STAGE1_SHARING_SCOPE!r}"
            )
        _require_digest("sharing_manifest_sha256", self.sharing_manifest_sha256)
        if not self.normalization_day_indices:
            raise Hold30Stage1FreezeBlocker(
                "normalization_day_indices must be explicitly frozen"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.normalization_day_indices
        ):
            raise Hold30Stage1FreezeBlocker(
                "normalization_day_indices must contain non-negative integers"
            )
        if tuple(sorted(set(self.normalization_day_indices))) != self.normalization_day_indices:
            raise Hold30Stage1FreezeBlocker(
                "normalization_day_indices must be strictly increasing and unique"
            )
        _require_digest(
            "normalization_schedule_sha256", self.normalization_schedule_sha256
        )
        _require_digest("optimizer_schedule_sha256", self.optimizer_schedule_sha256)
        if (
            isinstance(self.checkpoint_every, bool)
            or not isinstance(self.checkpoint_every, int)
            or self.checkpoint_every != HOLD30_STAGE1_CHECKPOINT_EVERY
        ):
            raise Hold30Stage1FreezeBlocker(
                "checkpoint_every must equal the frozen 50-update cadence"
            )
        if self.execution_strategy != HOLD30_STAGE1_EXECUTION_STRATEGY:
            raise Hold30Stage1FreezeBlocker(
                "Stage-1 distributed/sharding behavior is not frozen or implemented; "
                f"the receipt driver currently requires {HOLD30_STAGE1_EXECUTION_STRATEGY!r}"
            )
        _require_digest("approval_receipt_sha256", self.approval_receipt_sha256)


@dataclass(frozen=True, slots=True)
class Hold30Stage1Identity:
    """Exact source/data/fold identity of one shared fold encoder."""

    fold_index: int
    executable_manifest_sha256: str
    source_archive_sha256: str
    data_snapshot_sha256: str
    data_qualification_sha256: str
    fold_sha256: str
    training_tensor_sha256: str
    freeze: Hold30Stage1FreezeDecision
    protocol_generation: str = HOLD30_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
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
        for name in (
            "executable_manifest_sha256",
            "source_archive_sha256",
            "data_snapshot_sha256",
            "data_qualification_sha256",
            "fold_sha256",
            "training_tensor_sha256",
        ):
            _require_digest(name, getattr(self, name))
        expected_seed = derive_hold30_stage1_seed(self.fold_index)
        if self.freeze.stage1_seed != expected_seed:
            raise Hold30Stage1FreezeBlocker(
                f"fold {self.fold_index} Stage-1 seed must equal "
                f"SHA256-derived uint64 {expected_seed}"
            )

    def payload(self) -> dict[str, Any]:
        freeze = asdict(self.freeze)
        freeze["normalization_day_indices"] = list(
            self.freeze.normalization_day_indices
        )
        return {
            "protocol_generation": self.protocol_generation,
            "fold_index": self.fold_index,
            "executable_manifest_sha256": self.executable_manifest_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "data_qualification_sha256": self.data_qualification_sha256,
            "fold_sha256": self.fold_sha256,
            "training_tensor_sha256": self.training_tensor_sha256,
            "freeze_decision": freeze,
        }


@dataclass(frozen=True, slots=True)
class Hold30Stage1TrainingData:
    """Expanding-fold training inputs; outer data has no representable field."""

    fold_index: int
    day_ids: tuple[str, ...]
    train_days: Sequence[Mapping[str, Any]]
    day_close: torch.Tensor
    optimizer_day_schedule: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Hold30Stage1Progress:
    complete: bool
    completed_steps: int
    total_steps: int
    latest_checkpoint: Path | None
    run_receipt: Path | None


@dataclass(frozen=True, slots=True)
class _Stage1DistributedContext:
    world_size: int
    rank: int
    backend: str | None

    @property
    def enabled(self) -> bool:
        return self.world_size == 2


def _distributed_context(world_size: int, rank: int) -> _Stage1DistributedContext:
    if world_size not in (1, 2):
        raise Hold30Stage1Error("Stage-1 supports one-rank qualification or exact two-rank training")
    if rank not in range(world_size):
        raise Hold30Stage1Error("rank is outside world_size")
    if world_size == 1:
        return _Stage1DistributedContext(1, 0, None)
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != rank
    ):
        raise Hold30Stage1Error(
            "world_size=2 requires a matching initialized torch.distributed group"
        )
    return _Stage1DistributedContext(2, rank, str(dist.get_backend()))


def _rank0_call(context: _Stage1DistributedContext, callback: Callable[[], Any]) -> Any:
    if not context.enabled:
        return callback()
    packet: list[Any] = [None]
    if context.rank == 0:
        try:
            packet[0] = (True, callback())
        except Exception as exc:  # noqa: BLE001 - propagate rank-0 failure before a peer can hang
            packet[0] = (False, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(packet, src=0)
    success, value = packet[0]
    if not success:
        raise Hold30Stage1Error(f"rank-0 Stage-1 artifact operation failed: {value}")
    return value


def _require_rank_equal(
    context: _Stage1DistributedContext,
    name: str,
    value: Any,
) -> None:
    if not context.enabled:
        return
    gathered: list[Any] = [None, None]
    dist.all_gather_object(gathered, value)
    if gathered[0] != gathered[1]:
        raise Hold30Stage1Error(f"two Stage-1 ranks disagree on {name}")


def hold30_stage1_freeze_status(
    payload: Mapping[str, Any] | Hold30Stage1FreezeDecision | None,
) -> dict[str, Any]:
    """Report unresolved Stage-1 choices without manufacturing defaults.

    A complete decision makes the receipt driver and frozen two-rank software
    contract ready.  CPU/Gloo parity is software evidence only; this status
    never manufactures H100 capacity qualification or launch authority.
    """

    blockers: list[str] = []
    decision: Hold30Stage1FreezeDecision | None = None
    if payload is None:
        blockers.extend(f"missing:{name}" for name in _FREEZE_DECISION_FIELDS)
    elif isinstance(payload, Hold30Stage1FreezeDecision):
        decision = payload
    elif isinstance(payload, Mapping):
        missing = [name for name in _FREEZE_DECISION_FIELDS if name not in payload]
        blockers.extend(f"missing:{name}" for name in missing)
        if not missing:
            try:
                values = dict(payload)
                values["normalization_day_indices"] = tuple(
                    values["normalization_day_indices"]
                )
                decision = Hold30Stage1FreezeDecision(**values)
            except (TypeError, ValueError, Hold30Stage1FreezeBlocker) as exc:
                blockers.append(f"invalid_freeze_decision:{exc}")
    else:
        blockers.append("invalid_freeze_decision:not-a-mapping")
    return {
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "freeze_decision_complete": decision is not None,
        "receipt_driver_ready": decision is not None,
        "two_rank_software_contract_ready": decision is not None,
        "two_h100_stage1_qualified": False,
        "launch_authorized": False,
        "qualification_blockers": blockers
        + ["two-H100 Stage-1 capacity/numerical qualification has not run"],
        "decision_sha256": (
            sha256_payload(
                {
                    **asdict(decision),
                    "normalization_day_indices": list(
                        decision.normalization_day_indices
                    ),
                }
            )
            if decision is not None
            else None
        ),
    }


def hold30_stage1_contract() -> dict[str, Any]:
    """Return the package-owned, result-moving Stage-1 contract."""

    design = DESIGNS[HOLD30_BASE_DESIGN]
    expected = {
        "ssl_steps": HOLD30_STAGE1_STEPS,
        "ssl_batch_size": HOLD30_STAGE1_BATCH_SIZE,
        "ssl_accum": HOLD30_STAGE1_ACCUMULATION,
        "ssl_lr": HOLD30_STAGE1_LR,
        "ssl_weight_decay": HOLD30_STAGE1_WEIGHT_DECAY,
        "ssl_warmup_frac": 0.05,
        "schedule": HOLD30_STAGE1_SCHEDULE,
        "ssl_daily_coef": HOLD30_STAGE1_DAILY_COEFFICIENT,
        "ssl_perstock_coef": HOLD30_STAGE1_PER_STOCK_COEFFICIENT,
        "label_horizon_days": HOLD30_STAGE1_HORIZON,
    }
    drift = {
        name: (getattr(design, name), value)
        for name, value in expected.items()
        if getattr(design, name) != value
    }
    if drift:
        raise Hold30Stage1Error(f"registered Hold-30 Stage-1 contract drifted: {drift}")
    model = asdict(build_hold30_context_config())
    exact_model = {
        "bar_feature_dim": 5,
        "covariate_dim": 0,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "feedforward_dim": 256,
        "dropout": 0.0,
        "max_seconds": 390,
        "block_seconds": 5,
    }
    model_drift = {
        name: (model.get(name), value)
        for name, value in exact_model.items()
        if model.get(name) != value
    }
    if model_drift:
        raise Hold30Stage1Error(f"compact Stage-1 model drifted: {model_drift}")
    payload = {
        "model": model,
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": HOLD30_STAGE1_LR,
            "weight_decay": HOLD30_STAGE1_WEIGHT_DECAY,
            "warmup_steps": HOLD30_STAGE1_WARMUP_STEPS,
            "warmup_fraction": 0.05,
            "post_warmup_schedule": HOLD30_STAGE1_SCHEDULE,
            "optimizer_steps": HOLD30_STAGE1_STEPS,
            "micro_batch_days": HOLD30_STAGE1_BATCH_SIZE,
            "accumulation_micro_batches": HOLD30_STAGE1_ACCUMULATION,
            "effective_batch_days": HOLD30_STAGE1_EFFECTIVE_BATCH,
            "gradient_clip": design.grad_clip,
            "amp_bfloat16": design.amp,
            "distributed_strategy": HOLD30_STAGE1_EXECUTION_STRATEGY,
            "world_size": 2,
            "global_microbatch_rank_counts": {
                "even_microbatch": [2, 1],
                "odd_microbatch": [1, 2],
            },
            "dates_per_rank_per_update": 18,
            "gradient_reduction": "SUM",
            "loss_normalization": "enabled-objective sum / full-update global valid-target count",
            "partition_rng": "counter-keyed-by-fold-date-update; no stochastic masks active in v2",
        },
        "targets": {
            "market_next_block_coefficient": 1.0,
            "daily_cross_sectional_coefficient": HOLD30_STAGE1_DAILY_COEFFICIENT,
            "intraday_per_stock_coefficient": HOLD30_STAGE1_PER_STOCK_COEFFICIENT,
            "daily_horizon_sessions": HOLD30_STAGE1_HORIZON,
            "execution_delay_sessions": HOLD30_STAGE1_EXECUTION_DELAY,
            "covariate_axis_width": HOLD30_STAGE1_COVARIATE_DIM,
        },
        "input": {
            "bar_fields": ["open", "high", "low", "close", "volume"],
            "news_enabled": False,
            "fold_training_only": True,
            "daily_tail_censor_rows": 31,
        },
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def _sampling_seed(stage1_seed: int) -> int:
    return (stage1_seed * 9_973) % (2**63)


def materialize_hold30_stage1_schedule(n_days: int, stage1_seed: int) -> tuple[tuple[int, ...], ...]:
    """Create the exact 1000 x 36 training-day schedule without global RNG."""

    if isinstance(n_days, bool) or not isinstance(n_days, int) or n_days < HOLD30_STAGE1_EFFECTIVE_BATCH:
        raise Hold30Stage1Error(
            f"Stage-1 needs at least {HOLD30_STAGE1_EFFECTIVE_BATCH} fold-training days"
        )
    if (
        isinstance(stage1_seed, bool)
        or not isinstance(stage1_seed, int)
            or not 0 <= stage1_seed < 2**64
    ):
        raise Hold30Stage1FreezeBlocker(
            "stage1_seed requires an explicit uint64 integer"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sampling_seed(stage1_seed))
    return tuple(
        tuple(
            int(index)
            for index in torch.randperm(n_days, generator=generator)[
                :HOLD30_STAGE1_EFFECTIVE_BATCH
            ].tolist()
        )
        for _ in range(HOLD30_STAGE1_STEPS)
    )


def hold30_stage1_optimizer_schedule_sha256(
    schedule: Sequence[Sequence[int]],
) -> str:
    return sha256_payload([list(row) for row in schedule])


def hold30_stage1_normalization_schedule_sha256(
    day_ids: Sequence[str],
    indices: Sequence[int],
) -> str:
    if any(index < 0 or index >= len(day_ids) for index in indices):
        raise Hold30Stage1FreezeBlocker(
            "normalization schedule extends beyond the fold-training axis"
        )
    return sha256_payload(
        {
            "indices": list(indices),
            "training_day_ids": [day_ids[index] for index in indices],
        }
    )


def hold30_stage1_training_tensor_sha256(data: Hold30Stage1TrainingData) -> str:
    """Hash the exact ordered Stage-1 tensor slice using stable tensor digests."""

    if not isinstance(data, Hold30Stage1TrainingData):
        raise TypeError("data must be Hold30Stage1TrainingData")
    rows: list[dict[str, Any]] = []
    for day_index, source in enumerate(data.train_days):
        if not isinstance(source, Mapping):
            raise Hold30Stage1Error(f"training day {day_index} must be a mapping")
        missing = _ALLOWED_DAY_FIELDS - set(source)
        if missing:
            raise Hold30Stage1Error(
                f"training day {day_index} is missing Stage-1 fields {sorted(missing)}"
            )
        row: dict[str, Any] = {}
        for name in sorted(_ALLOWED_DAY_FIELDS):
            value = source[name]
            if torch.is_tensor(value):
                row[name] = _tensor_sha256(value)
            elif name == "session_close_block" and isinstance(value, int) and not isinstance(value, bool):
                row[name] = value
            else:
                raise Hold30Stage1Error(
                    f"training day {day_index} field {name!r} is not a supported tensor/scalar"
                )
        rows.append(row)
    return sha256_payload(
        {
            "fold_index": data.fold_index,
            "day_ids": list(data.day_ids),
            "day_close_sha256": _tensor_sha256(data.day_close),
            "days": rows,
        }
    )


def _validate_training_data(
    data: Hold30Stage1TrainingData,
    identity: Hold30Stage1Identity,
) -> tuple[list[dict[str, Any]], list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    if not isinstance(data, Hold30Stage1TrainingData):
        raise TypeError("data must be Hold30Stage1TrainingData")
    if data.fold_index != identity.fold_index:
        raise Hold30Stage1Error("training-data fold differs from the Stage-1 identity")
    n_days = len(data.day_ids)
    if n_days != len(data.train_days) or n_days != data.day_close.shape[0]:
        raise Hold30Stage1Error("day IDs, train days, and day_close must have equal length")
    if n_days < HOLD30_STAGE1_EFFECTIVE_BATCH:
        raise Hold30Stage1Error("too few fold-training days for one distinct effective batch")
    parsed: list[date] = []
    for value in data.day_ids:
        try:
            parsed.append(date.fromisoformat(value))
        except (TypeError, ValueError) as exc:
            raise Hold30Stage1Error(f"invalid training day ID {value!r}") from exc
    if any(left >= right for left, right in pairwise(parsed)):
        raise Hold30Stage1Error("fold-training day IDs must be strictly increasing and unique")
    if parsed[-1] >= date(2026, 1, 1):
        raise Hold30Stage1Error("Stage-1 cannot receive 2026 or later observations")
    if data.day_close.ndim != 2 or not data.day_close.dtype.is_floating_point:
        raise Hold30Stage1Error("day_close must be floating [training_day, action]")
    actual_training_sha256 = hold30_stage1_training_tensor_sha256(data)
    if actual_training_sha256 != identity.training_tensor_sha256:
        raise Hold30Stage1Error(
            "training_tensor_sha256 does not match the exact ordered Stage-1 inputs"
        )
    schedule = data.optimizer_day_schedule
    if len(schedule) != HOLD30_STAGE1_STEPS:
        raise Hold30Stage1FreezeBlocker(
            "optimizer date schedule must contain exactly 1000 update rows"
        )
    for update, row in enumerate(schedule):
        if len(row) != HOLD30_STAGE1_EFFECTIVE_BATCH:
            raise Hold30Stage1FreezeBlocker(
                f"optimizer date schedule update {update} must contain 36 dates"
            )
        if any(isinstance(index, bool) or not isinstance(index, int) for index in row):
            raise Hold30Stage1FreezeBlocker("optimizer date schedule indexes must be integers")
        if any(index < 0 or index >= n_days for index in row):
            raise Hold30Stage1FreezeBlocker(
                f"optimizer date schedule update {update} leaves the fold-training axis"
            )
        if len(set(row)) != HOLD30_STAGE1_EFFECTIVE_BATCH:
            raise Hold30Stage1FreezeBlocker(
                f"optimizer date schedule update {update} must use 36 distinct dates"
            )
    optimizer_schedule_sha256 = hold30_stage1_optimizer_schedule_sha256(schedule)
    if optimizer_schedule_sha256 != identity.freeze.optimizer_schedule_sha256:
        raise Hold30Stage1FreezeBlocker(
            "optimizer date schedule differs from its pretraining freeze digest"
        )
    normalization_schedule_sha256 = hold30_stage1_normalization_schedule_sha256(
        data.day_ids,
        identity.freeze.normalization_day_indices,
    )
    if normalization_schedule_sha256 != identity.freeze.normalization_schedule_sha256:
        raise Hold30Stage1FreezeBlocker(
            "normalization dates differ from their pretraining freeze digest"
        )

    actions = int(data.day_close.shape[1])
    projected: list[dict[str, Any]] = []
    for day_index, source in enumerate(data.train_days):
        if not isinstance(source, Mapping):
            raise Hold30Stage1Error(f"training day {day_index} must be a mapping")
        forbidden = [
            key
            for key in source
            if isinstance(key, str)
            and any(token in key.casefold() for token in _FORBIDDEN_DAY_FIELD_TOKENS)
        ]
        if forbidden:
            raise Hold30Stage1Error(
                f"training day {day_index} exposes forbidden non-training fields: {forbidden}"
            )
        missing = _ALLOWED_DAY_FIELDS - set(source)
        if missing:
            raise Hold30Stage1Error(
                f"training day {day_index} is missing Stage-1 fields {sorted(missing)}"
            )
        row = {name: source[name] for name in _ALLOWED_DAY_FIELDS}
        if any(
            not torch.is_tensor(row[name])
            for name in _ALLOWED_DAY_FIELDS - {"session_close_block"}
        ):
            raise Hold30Stage1Error(f"training day {day_index} contains a non-tensor field")
        bars = row["bars"]
        bar_mask = row["bar_mask"]
        cov = row["cov_blocks"]
        ret = row["ret"]
        ret_valid = row["ret_valid"]
        if tuple(bars.shape) != (actions, 390, 5):
            raise Hold30Stage1Error(
                f"training day {day_index} bars must be [{actions},390,5]"
            )
        if tuple(bar_mask.shape) != (actions, 390) or bar_mask.dtype != torch.bool:
            raise Hold30Stage1Error(
                f"training day {day_index} bar_mask must be bool [{actions},390]"
            )
        if tuple(cov.shape) != (78, actions, 0):
            raise Hold30Stage1Error(
                f"training day {day_index} cov_blocks must have exact zero-width shape [78,{actions},0]"
            )
        if tuple(ret.shape) != (78, actions) or tuple(ret_valid.shape) != (78, actions):
            raise Hold30Stage1Error(
                f"training day {day_index} return fields must be [78,{actions}]"
            )
        if ret_valid.dtype != torch.bool:
            raise Hold30Stage1Error(f"training day {day_index} ret_valid must be bool")
        close_block = row["session_close_block"]
        if torch.is_tensor(close_block):
            if close_block.numel() != 1:
                raise Hold30Stage1Error("session_close_block must be scalar")
            close_block = int(close_block.item())
        if isinstance(close_block, bool) or not isinstance(close_block, int) or not 0 <= close_block < 78:
            raise Hold30Stage1Error("session_close_block must lie in [0, 77]")
        row["session_close_block"] = close_block
        projected.append(row)

    targets, valid = ssl_targets_daily(
        data.day_close,
        HOLD30_STAGE1_HORIZON,
        HOLD30_STAGE1_EXECUTION_DELAY,
    )
    if bool(valid[-31:].any()):
        raise AssertionError("training-only daily targets failed the 31-row censor contract")
    if not bool(valid[:-31].any()):
        raise Hold30Stage1Error("fold-training close history has no valid 30-session target")
    daily_targets = [(targets[index], valid[index]) for index in range(n_days)]
    evidence = {
        "n_training_days": n_days,
        "n_actions": actions,
        "first_training_day": data.day_ids[0],
        "last_training_day": data.day_ids[-1],
        "training_day_axis_sha256": sha256_payload(list(data.day_ids)),
        "day_close_sha256": _tensor_sha256(data.day_close),
        "daily_target_sha256": sha256_payload(
            {
                "target": _tensor_sha256(targets),
                "valid": _tensor_sha256(valid),
            }
        ),
        "training_tensor_sha256": actual_training_sha256,
        "optimizer_schedule_sha256": optimizer_schedule_sha256,
        "normalization_schedule_sha256": normalization_schedule_sha256,
        "censored_tail_rows": 31,
        "outer_data_exposed": False,
    }
    return projected, daily_targets, evidence


def _calibrate(
    encoder: ContextEncoder,
    days: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
) -> None:
    encoder.reset_normalization()
    for offset in range(0, len(indices), HOLD30_STAGE1_BATCH_SIZE):
        rows = [days[index] for index in indices[offset:offset + HOLD30_STAGE1_BATCH_SIZE]]
        bars = torch.stack([row["bars"] for row in rows]).to(device)
        mask = torch.stack([row["bar_mask"] for row in rows]).to(device)
        cov = torch.stack([row["cov_blocks"] for row in rows]).to(device)
        encoder.calibrate_normalization(bars, mask, cov)
    if not encoder.normalization_calibrated:
        raise Hold30Stage1Error("training-only normalization did not observe all five OHLCV fields")


def _build_modules(
    seed: int,
    device: torch.device,
) -> tuple[ContextEncoder, ContextForwardHead, DailyForwardHead]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    encoder = ContextEncoder(build_hold30_context_config()).to(device)
    market_head = ContextForwardHead(128).to(device)
    daily_head = DailyForwardHead(128).to(device)
    return encoder, market_head, daily_head


def _optimizer(
    encoder: ContextEncoder,
    market_head: ContextForwardHead,
    daily_head: DailyForwardHead,
) -> torch.optim.AdamW:
    return make_adamw(
        [*encoder.parameters(), *market_head.parameters(), *daily_head.parameters()],
        lr=HOLD30_STAGE1_LR,
        weight_decay=HOLD30_STAGE1_WEIGHT_DECAY,
    )


def _identity_document(
    identity: Hold30Stage1Identity,
    contract: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
    schedule: Sequence[Sequence[int]],
    encoder: ContextEncoder,
    market_head: ContextForwardHead,
    daily_head: DailyForwardHead,
    distributed_context: _Stage1DistributedContext,
) -> dict[str, Any]:
    schedule_sha = sha256_payload([list(row) for row in schedule])
    payload = {
        "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
        "receipt_type": "hold30-stage1-identity",
        "identity": identity.payload(),
        "contract": dict(contract),
        "data": dict(data_evidence),
        "schedules": {
            "optimizer_day_schedule_sha256": schedule_sha,
            "optimizer_day_schedule_shape": [HOLD30_STAGE1_STEPS, HOLD30_STAGE1_EFFECTIVE_BATCH],
            "normalization_day_indices": list(identity.freeze.normalization_day_indices),
            "normalization_day_schedule_sha256": data_evidence[
                "normalization_schedule_sha256"
            ],
            "checkpoint_every": identity.freeze.checkpoint_every,
        },
        "initial_model": {
            "encoder_state_sha256": _state_sha256(_cpu_state(encoder)),
            "market_head_state_sha256": _state_sha256(_cpu_state(market_head)),
            "daily_head_state_sha256": _state_sha256(_cpu_state(daily_head)),
        },
        "execution": {
            "strategy": HOLD30_STAGE1_EXECUTION_STRATEGY,
            "world_size": distributed_context.world_size,
            "backend": distributed_context.backend,
            "qualification_only": distributed_context.world_size != 2,
            "h100_qualified": False,
        },
    }
    return _with_receipt_hash(payload)


def _initial_payload(
    identity_sha256: str,
    encoder: ContextEncoder,
    market_head: ContextForwardHead,
    daily_head: DailyForwardHead,
) -> dict[str, Any]:
    return {
        "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
        "artifact_type": "hold30-stage1-initial-model",
        "identity_sha256": identity_sha256,
        "encoder": _cpu_state(encoder),
        "market_head": _cpu_state(market_head),
        "daily_head": _cpu_state(daily_head),
    }


def _schedule_documents(
    identity_doc: Mapping[str, Any],
    data: Hold30Stage1TrainingData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalization_rows = [
        {"training_index": index, "day_id": data.day_ids[index]}
        for index in identity_doc["schedules"]["normalization_day_indices"]
    ]
    normalization = _with_receipt_hash(
        {
            "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
            "receipt_type": "hold30-stage1-normalization-date-schedule",
            "identity_sha256": identity_doc["receipt_sha256"],
            "previous_receipt_sha256": identity_doc["receipt_sha256"],
            "schedule_sha256": identity_doc["schedules"][
                "normalization_day_schedule_sha256"
            ],
            "rows": normalization_rows,
            "fold_training_only": True,
        }
    )
    optimizer = _with_receipt_hash(
        {
            "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
            "receipt_type": "hold30-stage1-optimizer-date-schedule",
            "identity_sha256": identity_doc["receipt_sha256"],
            "previous_receipt_sha256": normalization["receipt_sha256"],
            "schedule_sha256": identity_doc["schedules"][
                "optimizer_day_schedule_sha256"
            ],
            "shape": [HOLD30_STAGE1_STEPS, HOLD30_STAGE1_EFFECTIVE_BATCH],
            "rows": [list(row) for row in data.optimizer_day_schedule],
            "fold_training_only": True,
        }
    )
    return normalization, optimizer


def _verify_schedules(root: Path, identity_doc: Mapping[str, Any]) -> str:
    normalization_path = root / "normalization-date-schedule.json"
    optimizer_path = root / "optimizer-date-schedule.json"
    normalization = _read_json(normalization_path)
    normalization_sha = _validate_receipt(normalization, normalization_path)
    if normalization.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("normalization schedule belongs to another identity")
    if normalization.get("previous_receipt_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("normalization schedule does not follow the identity receipt")
    rows = normalization.get("rows")
    if not isinstance(rows, list):
        raise Hold30Stage1Error("normalization schedule rows are missing")
    indices = [row.get("training_index") for row in rows if isinstance(row, dict)]
    day_ids = [row.get("day_id") for row in rows if isinstance(row, dict)]
    if len(indices) != len(rows) or len(day_ids) != len(rows):
        raise Hold30Stage1Error("normalization schedule row schema is invalid")
    actual_normalization_sha = sha256_payload(
        {"indices": indices, "training_day_ids": day_ids}
    )
    if (
        normalization.get("schedule_sha256") != actual_normalization_sha
        or actual_normalization_sha
        != identity_doc["schedules"]["normalization_day_schedule_sha256"]
    ):
        raise Hold30Stage1Error("normalization schedule digest mismatch")

    optimizer = _read_json(optimizer_path)
    optimizer_sha = _validate_receipt(optimizer, optimizer_path)
    if optimizer.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("optimizer schedule belongs to another identity")
    if optimizer.get("previous_receipt_sha256") != normalization_sha:
        raise Hold30Stage1Error("optimizer schedule receipt chain is broken")
    optimizer_rows = optimizer.get("rows")
    if not isinstance(optimizer_rows, list):
        raise Hold30Stage1Error("optimizer schedule rows are missing")
    actual_optimizer_sha = hold30_stage1_optimizer_schedule_sha256(optimizer_rows)
    if (
        optimizer.get("schedule_sha256") != actual_optimizer_sha
        or actual_optimizer_sha
        != identity_doc["schedules"]["optimizer_day_schedule_sha256"]
    ):
        raise Hold30Stage1Error("optimizer schedule digest mismatch")
    return optimizer_sha


def _write_payload_with_receipt(
    artifact_path: Path,
    receipt_path: Path,
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _write_new_torch(artifact_path, payload)
    completed = _with_receipt_hash(
        {
            **receipt,
            "artifact": artifact_path.name,
            "artifact_sha256": _sha256_file(artifact_path),
        }
    )
    _write_new_json(receipt_path, completed)
    return completed


def _checkpoint_pairs(
    root: Path,
    checkpoint_every: int,
    *,
    allow_orphan_artifact: bool,
) -> list[tuple[int, Path, Path | None]]:
    directory = root / "checkpoints"
    if not directory.exists():
        return []
    if not directory.is_dir() or directory.is_symlink():
        raise Hold30Stage1Error("checkpoints must be a real directory")
    artifacts: dict[int, Path] = {}
    receipts: dict[int, Path] = {}
    for path in directory.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        receipt_match = _CHECKPOINT_RECEIPT.fullmatch(path.name)
        if match:
            artifacts[int(match.group(1))] = path
        elif receipt_match:
            receipts[int(receipt_match.group(1))] = path
        else:
            raise Hold30Stage1Error(f"unexpected checkpoint artifact: {path}")
    if set(receipts) - set(artifacts):
        raise Hold30Stage1Error("a checkpoint receipt exists without its artifact")
    orphans = sorted(set(artifacts) - set(receipts))
    if orphans and (not allow_orphan_artifact or len(orphans) != 1):
        raise Hold30Stage1Error("checkpoint and receipt sets are incomplete")
    steps = sorted(artifacts)
    expected = list(range(checkpoint_every, HOLD30_STAGE1_STEPS + 1, checkpoint_every))[: len(steps)]
    if steps != expected:
        raise Hold30Stage1Error("checkpoint chain is not a contiguous prefix")
    if orphans and orphans[0] != steps[-1]:
        raise Hold30Stage1Error("only the newest checkpoint may await receipt attachment")
    return [(step, artifacts[step], receipts.get(step)) for step in steps]


def _verify_initial(
    root: Path,
    identity_doc: Mapping[str, Any],
    *,
    attach_missing_receipt: bool,
) -> str:
    previous_receipt = _verify_schedules(root, identity_doc)
    artifact = root / "initial-model.pt"
    receipt_path = root / "initial-model.receipt.json"
    payload = _load_torch(artifact)
    if payload.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("initial model is bound to another identity")
    expected = identity_doc["initial_model"]
    for key, label in (
        ("encoder", "encoder_state_sha256"),
        ("market_head", "market_head_state_sha256"),
        ("daily_head", "daily_head_state_sha256"),
    ):
        if _state_sha256(payload[key]) != expected[label]:
            raise Hold30Stage1Error(f"initial {key} state digest mismatch")
    if not receipt_path.exists():
        if not attach_missing_receipt:
            raise Hold30Stage1Error("initial-model artifact lacks its receipt")
        receipt = _with_receipt_hash(
            {
                "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
                "receipt_type": "hold30-stage1-initial-model",
                "identity_sha256": identity_doc["receipt_sha256"],
                "previous_receipt_sha256": previous_receipt,
                "artifact": artifact.name,
                "artifact_sha256": _sha256_file(artifact),
            }
        )
        _write_new_json(receipt_path, receipt)
    else:
        receipt = _read_json(receipt_path)
    receipt_sha = _validate_receipt(receipt, receipt_path)
    if receipt.get("artifact_sha256") != _sha256_file(artifact):
        raise Hold30Stage1Error("initial-model artifact digest mismatch")
    if receipt.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("initial-model receipt is bound to another identity")
    if receipt.get("previous_receipt_sha256") != previous_receipt:
        raise Hold30Stage1Error("initial-model receipt does not follow the date schedules")
    return receipt_sha


def _verify_checkpoints(
    root: Path,
    identity_doc: Mapping[str, Any],
    *,
    attach_missing_receipts: bool = False,
) -> tuple[int, str, Path | None]:
    previous = _verify_initial(
        root,
        identity_doc,
        attach_missing_receipt=attach_missing_receipts,
    )
    latest: Path | None = None
    completed = 0
    checkpoint_every = int(identity_doc["schedules"]["checkpoint_every"])
    for step, artifact, receipt_path in _checkpoint_pairs(
        root,
        checkpoint_every,
        allow_orphan_artifact=attach_missing_receipts,
    ):
        payload = _load_torch(artifact)
        if payload.get("step") != step or payload.get("identity_sha256") != identity_doc["receipt_sha256"]:
            raise Hold30Stage1Error("checkpoint payload identity/step mismatch")
        if receipt_path is None:
            receipt_path = artifact.with_name(f"step-{step:06d}.receipt.json")
            receipt = _with_receipt_hash(
                {
                    "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
                    "receipt_type": "hold30-stage1-checkpoint",
                    "identity_sha256": identity_doc["receipt_sha256"],
                    "step": step,
                    "previous_receipt_sha256": previous,
                    "artifact": artifact.name,
                    "artifact_sha256": _sha256_file(artifact),
                }
            )
            _write_new_json(receipt_path, receipt)
        else:
            receipt = _read_json(receipt_path)
        receipt_sha = _validate_receipt(receipt, receipt_path)
        if receipt.get("step") != step:
            raise Hold30Stage1Error("checkpoint receipt step mismatch")
        if receipt.get("identity_sha256") != identity_doc["receipt_sha256"]:
            raise Hold30Stage1Error("checkpoint belongs to another identity")
        if receipt.get("previous_receipt_sha256") != previous:
            raise Hold30Stage1Error("checkpoint receipt chain is broken")
        if receipt.get("artifact_sha256") != _sha256_file(artifact):
            raise Hold30Stage1Error("checkpoint artifact digest mismatch")
        previous, completed, latest = receipt_sha, step, artifact
    return completed, previous, latest


def _publish_or_validate_final(
    root: Path,
    identity_doc: Mapping[str, Any],
    identity: Hold30Stage1Identity,
    encoder: ContextEncoder,
    *,
    previous_receipt: str,
    execution_world_size: int,
) -> None:
    freeze_encoder(encoder)
    encoder_path = root / "frozen-encoder.pt"
    encoder_receipt_path = root / "frozen-encoder.receipt.json"
    run_receipt_path = root / "run-receipt.json"
    encoder_payload = {
        "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
        "artifact_type": "hold30-stage1-frozen-encoder",
        "identity_sha256": identity_doc["receipt_sha256"],
        "completed_steps": HOLD30_STAGE1_STEPS,
        "normalization_calibrated": encoder.normalization_calibrated,
        "encoder": _cpu_state(encoder),
    }
    if encoder_path.exists():
        existing_encoder = _load_torch(encoder_path)
        if (
            existing_encoder.get("identity_sha256")
            != identity_doc["receipt_sha256"]
            or existing_encoder.get("completed_steps") != HOLD30_STAGE1_STEPS
            or _state_sha256(existing_encoder.get("encoder", {}))
            != _state_sha256(encoder_payload["encoder"])
        ):
            raise Hold30Stage1Error("existing frozen encoder differs from the final checkpoint")
    else:
        _write_new_torch(encoder_path, encoder_payload)
    expected_encoder_receipt = _with_receipt_hash(
        {
            "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
            "receipt_type": "hold30-stage1-frozen-encoder",
            "identity_sha256": identity_doc["receipt_sha256"],
            "completed_steps": HOLD30_STAGE1_STEPS,
            "previous_receipt_sha256": previous_receipt,
            "encoder_state_sha256": _state_sha256(encoder_payload["encoder"]),
            "artifact": encoder_path.name,
            "artifact_sha256": _sha256_file(encoder_path),
        }
    )
    if encoder_receipt_path.exists():
        if _read_json(encoder_receipt_path) != expected_encoder_receipt:
            raise Hold30Stage1Error("existing frozen-encoder receipt differs from the final checkpoint")
    else:
        _write_new_json(encoder_receipt_path, expected_encoder_receipt)

    expected_run_receipt = _with_receipt_hash(
        {
            "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
            "receipt_type": "hold30-stage1-run",
            "identity_sha256": identity_doc["receipt_sha256"],
            "complete": True,
            "completed_steps": HOLD30_STAGE1_STEPS,
            "previous_receipt_sha256": expected_encoder_receipt["receipt_sha256"],
            "frozen_encoder_receipt_sha256": expected_encoder_receipt[
                "receipt_sha256"
            ],
            "scientific_qualification": False,
            "two_rank_software_contract": execution_world_size == 2,
            "h100_qualified": False,
            "outer_data_exposed": False,
        }
    )
    if run_receipt_path.exists():
        if _read_json(run_receipt_path) != expected_run_receipt:
            raise Hold30Stage1Error("existing run receipt differs from exact completion")
    else:
        _write_new_json(run_receipt_path, expected_run_receipt)
    verify_hold30_stage1_run(root, expected_identity=identity)


def verify_hold30_stage1_run(
    root: str | Path,
    *,
    expected_identity: Hold30Stage1Identity | None = None,
) -> dict[str, Any]:
    """Verify the complete append-only Stage-1 receipt graph."""

    run_root = Path(root)
    identity_path = run_root / "identity.json"
    identity_doc = _read_json(identity_path)
    _validate_receipt(identity_doc, identity_path)
    if expected_identity is not None and identity_doc.get("identity") != expected_identity.payload():
        raise Hold30Stage1Error("Stage-1 identity differs from the expected identity")
    completed, previous, latest = _verify_checkpoints(run_root, identity_doc)
    if completed != HOLD30_STAGE1_STEPS or latest is None:
        raise Hold30Stage1Error("Stage-1 run is incomplete")
    encoder_path = run_root / "frozen-encoder.pt"
    encoder_receipt_path = run_root / "frozen-encoder.receipt.json"
    encoder_receipt = _read_json(encoder_receipt_path)
    encoder_receipt_sha = _validate_receipt(encoder_receipt, encoder_receipt_path)
    if encoder_receipt.get("previous_receipt_sha256") != previous:
        raise Hold30Stage1Error("frozen-encoder receipt does not close the checkpoint chain")
    if encoder_receipt.get("artifact_sha256") != _sha256_file(encoder_path):
        raise Hold30Stage1Error("frozen encoder digest mismatch")
    encoder_payload = _load_torch(encoder_path)
    if encoder_payload.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("frozen encoder belongs to another identity")
    checkpoint_payload = _load_torch(latest)
    final_state_sha256 = _state_sha256(encoder_payload.get("encoder", {}))
    if final_state_sha256 != _state_sha256(checkpoint_payload.get("encoder", {})):
        raise Hold30Stage1Error("frozen encoder differs from the final checkpoint")
    if encoder_receipt.get("encoder_state_sha256") != final_state_sha256:
        raise Hold30Stage1Error("frozen-encoder state receipt digest mismatch")
    run_receipt_path = run_root / "run-receipt.json"
    receipt = _read_json(run_receipt_path)
    _validate_receipt(receipt, run_receipt_path)
    if receipt.get("previous_receipt_sha256") != encoder_receipt_sha:
        raise Hold30Stage1Error("run receipt does not close the encoder receipt")
    if receipt.get("identity_sha256") != identity_doc["receipt_sha256"]:
        raise Hold30Stage1Error("run receipt belongs to another identity")
    if receipt.get("completed_steps") != HOLD30_STAGE1_STEPS or not receipt.get("complete"):
        raise Hold30Stage1Error("run receipt does not declare exact completion")
    return receipt


def run_hold30_stage1(
    root: str | Path,
    identity: Hold30Stage1Identity,
    data: Hold30Stage1TrainingData,
    *,
    device: torch.device,
    world_size: int = 1,
    rank: int = 0,
    max_new_steps: int | None = None,
    _train_fn: Callable[..., torch.optim.Optimizer] = train_context_encoder,
) -> Hold30Stage1Progress:
    """Train or exactly resume one shared fold encoder.

    ``max_new_steps`` is an operational yield boundary, not a changed training
    budget.  It must land on the approved checkpoint cadence.  The immutable
    plan always contains 1,000 optimizer steps.
    """

    if not isinstance(identity, Hold30Stage1Identity):
        raise Hold30Stage1FreezeBlocker(
            "an explicit Hold30Stage1Identity and freeze decision are required"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Hold30Stage1Error("CUDA device requested but CUDA is unavailable")
    distributed_context = _distributed_context(world_size, rank)
    days, daily_targets, data_evidence = _validate_training_data(data, identity)
    if any(index >= len(days) for index in identity.freeze.normalization_day_indices):
        raise Hold30Stage1FreezeBlocker(
            "normalization_day_indices extend beyond this fold's training axis"
        )
    schedule = data.optimizer_day_schedule
    contract = hold30_stage1_contract()
    run_root = Path(root)
    identity_path = run_root / "identity.json"

    encoder, market_head, daily_head = _build_modules(identity.freeze.stage1_seed, device)
    initial_path = run_root / "initial-model.pt"
    initial_receipt_path = run_root / "initial-model.receipt.json"
    if initial_path.exists():
        initial = _load_torch(initial_path)
        encoder.load_state_dict(initial["encoder"])
        market_head.load_state_dict(initial["market_head"])
        daily_head.load_state_dict(initial["daily_head"])
    else:
        _calibrate(
            encoder,
            days,
            identity.freeze.normalization_day_indices,
            device,
        )
    expected_doc = _identity_document(
        identity,
        contract,
        data_evidence,
        schedule,
        encoder,
        market_head,
        daily_head,
        distributed_context,
    )
    _require_rank_equal(
        distributed_context,
        "identity document",
        expected_doc["receipt_sha256"],
    )

    def publish_or_validate_initial() -> None:
        if identity_path.exists():
            identity_doc = _read_json(identity_path)
            _validate_receipt(identity_doc, identity_path)
            if identity_doc != expected_doc:
                raise Hold30Stage1Error(
                    "existing Stage-1 identity differs from requested run"
                )
        else:
            if run_root.exists() and any(run_root.iterdir()):
                raise Hold30Stage1Error("non-empty Stage-1 root lacks identity.json")
            _write_new_json(identity_path, expected_doc)
        normalization_doc, optimizer_doc = _schedule_documents(expected_doc, data)
        for schedule_path, schedule_doc in (
            (run_root / "normalization-date-schedule.json", normalization_doc),
            (run_root / "optimizer-date-schedule.json", optimizer_doc),
        ):
            if schedule_path.exists():
                if _read_json(schedule_path) != schedule_doc:
                    raise Hold30Stage1Error(
                        f"existing date-schedule artifact differs: {schedule_path}"
                    )
            else:
                _write_new_json(schedule_path, schedule_doc)
        optimizer_schedule_receipt = _verify_schedules(run_root, expected_doc)
        if not initial_path.exists():
            if initial_receipt_path.exists():
                raise Hold30Stage1Error(
                    "initial-model receipt exists without its artifact"
                )
            initial = _initial_payload(
                expected_doc["receipt_sha256"], encoder, market_head, daily_head
            )
            _write_payload_with_receipt(
                initial_path,
                initial_receipt_path,
                initial,
                {
                    "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
                    "receipt_type": "hold30-stage1-initial-model",
                    "identity_sha256": expected_doc["receipt_sha256"],
                    "previous_receipt_sha256": optimizer_schedule_receipt,
                },
            )
        _verify_initial(
            run_root,
            expected_doc,
            attach_missing_receipt=True,
        )

    _rank0_call(distributed_context, publish_or_validate_initial)
    identity_doc = _read_json(identity_path)
    _validate_receipt(identity_doc, identity_path)
    initial = _load_torch(initial_path)
    encoder.load_state_dict(initial["encoder"])
    market_head.load_state_dict(initial["market_head"])
    daily_head.load_state_dict(initial["daily_head"])
    _require_rank_equal(
        distributed_context,
        "initial encoder state",
        _state_sha256(_cpu_state(encoder)),
    )

    def inspect_progress() -> dict[str, Any]:
        completed, previous, latest = _verify_checkpoints(
            run_root,
            identity_doc,
            attach_missing_receipts=True,
        )
        return {
            "completed": completed,
            "previous": previous,
            "latest": str(latest) if latest is not None else None,
        }

    progress = _rank0_call(distributed_context, inspect_progress)
    completed = int(progress["completed"])
    previous_receipt = str(progress["previous"])
    latest = Path(progress["latest"]) if progress["latest"] is not None else None
    run_receipt_path = run_root / "run-receipt.json"
    if completed == HOLD30_STAGE1_STEPS:
        if latest is None:
            raise Hold30Stage1Error("complete Stage-1 progress lacks its final checkpoint")
        checkpoint = _load_torch(latest)
        encoder.load_state_dict(checkpoint["encoder"])
        market_head.load_state_dict(checkpoint["market_head"])
        daily_head.load_state_dict(checkpoint["daily_head"])
        _require_rank_equal(
            distributed_context,
            "completed encoder state",
            _state_sha256(_cpu_state(encoder)),
        )
        _rank0_call(
            distributed_context,
            lambda: _publish_or_validate_final(
                run_root,
                identity_doc,
                identity,
                encoder,
                previous_receipt=previous_receipt,
                execution_world_size=world_size,
            ),
        )
        return Hold30Stage1Progress(True, completed, HOLD30_STAGE1_STEPS, latest, run_receipt_path)
    def reject_premature_final() -> None:
        if run_receipt_path.exists() or (run_root / "frozen-encoder.pt").exists():
            raise Hold30Stage1Error(
                "incomplete checkpoint chain has premature final artifacts"
            )

    _rank0_call(distributed_context, reject_premature_final)

    initial = _load_torch(run_root / "initial-model.pt")
    encoder.load_state_dict(initial["encoder"])
    market_head.load_state_dict(initial["market_head"])
    daily_head.load_state_dict(initial["daily_head"])
    optimizer = _optimizer(encoder, market_head, daily_head)
    if latest is not None:
        checkpoint = _load_torch(latest)
        encoder.load_state_dict(checkpoint["encoder"])
        market_head.load_state_dict(checkpoint["market_head"])
        daily_head.load_state_dict(checkpoint["daily_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng_by_rank = checkpoint.get("rng_by_rank")
        if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
            raise Hold30Stage1Error(
                "checkpoint RNG topology differs from the requested Stage-1 world"
            )
        _restore_rng_state(rng_by_rank[rank])
    else:
        training_seed = _sampling_seed(identity.freeze.stage1_seed)
        random.seed(training_seed)
        np.random.seed(training_seed % (2**32))
        torch.manual_seed(training_seed)

    cadence = identity.freeze.checkpoint_every
    remaining = HOLD30_STAGE1_STEPS - completed
    if max_new_steps is None:
        target = HOLD30_STAGE1_STEPS
    else:
        if isinstance(max_new_steps, bool) or not isinstance(max_new_steps, int) or max_new_steps < 0:
            raise ValueError("max_new_steps must be a non-negative integer or None")
        allowed = min(max_new_steps, remaining)
        target = completed + allowed
        if target != HOLD30_STAGE1_STEPS and target % cadence:
            raise ValueError("max_new_steps must stop on the approved checkpoint cadence")
    if target == completed:
        return Hold30Stage1Progress(False, completed, HOLD30_STAGE1_STEPS, latest, None)

    chain = {"previous": previous_receipt}
    checkpoint_snapshot: dict[str, Any] = {"by_rank": None}

    def sum_reduce(parameters: list[torch.nn.Parameter]) -> None:
        if not distributed_context.enabled:
            return
        for parameter in parameters:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def prepare_checkpoint() -> None:
        local = {
            "rank": rank,
            "rng": _cpu_tree(_rng_state()),
            "encoder_state_sha256": _state_sha256(_cpu_state(encoder)),
            "market_head_state_sha256": _state_sha256(_cpu_state(market_head)),
            "daily_head_state_sha256": _state_sha256(_cpu_state(daily_head)),
        }
        if distributed_context.enabled:
            gathered: list[Any] = [None, None]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        state_rows = [
            (
                row["encoder_state_sha256"],
                row["market_head_state_sha256"],
                row["daily_head_state_sha256"],
            )
            for row in gathered
        ]
        if len(set(state_rows)) != 1:
            raise Hold30Stage1Error(
                "two Stage-1 ranks diverged before checkpoint publication"
            )
        checkpoint_snapshot["by_rank"] = gathered

    def on_checkpoint(step: int, current_optimizer: torch.optim.Optimizer) -> None:
        if checkpoint_snapshot["by_rank"] is None:
            prepare_checkpoint()
        snapshot = checkpoint_snapshot["by_rank"]
        checkpoint_snapshot["by_rank"] = None
        if distributed_context.rank != 0:
            return
        artifact = run_root / "checkpoints" / f"step-{step:06d}.pt"
        receipt_path = run_root / "checkpoints" / f"step-{step:06d}.receipt.json"
        payload = {
            "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
            "artifact_type": "hold30-stage1-checkpoint",
            "identity_sha256": identity_doc["receipt_sha256"],
            "step": step,
            "encoder": _cpu_state(encoder),
            "market_head": _cpu_state(market_head),
            "daily_head": _cpu_state(daily_head),
            "optimizer": _cpu_tree(current_optimizer.state_dict()),
            "rng_by_rank": [
                row["rng"] for row in snapshot
            ],
        }
        receipt = _write_payload_with_receipt(
            artifact,
            receipt_path,
            payload,
            {
                "schema_version": HOLD30_STAGE1_SCHEMA_VERSION,
                "receipt_type": "hold30-stage1-checkpoint",
                "identity_sha256": identity_doc["receipt_sha256"],
                "step": step,
                "previous_receipt_sha256": chain["previous"],
            },
        )
        chain["previous"] = receipt["receipt_sha256"]

    design = DESIGNS[HOLD30_BASE_DESIGN]
    _train_fn(
        encoder,
        market_head,
        days,
        device=device,
        perstock_head=None,
        perstock_coef=HOLD30_STAGE1_PER_STOCK_COEFFICIENT,
        daily_head=daily_head,
        daily_targets=daily_targets,
        daily_coef=HOLD30_STAGE1_DAILY_COEFFICIENT,
        steps=target,
        lr=HOLD30_STAGE1_LR,
        weight_decay=HOLD30_STAGE1_WEIGHT_DECAY,
        batch_size=HOLD30_STAGE1_BATCH_SIZE,
        accum_steps=HOLD30_STAGE1_ACCUMULATION,
        warmup_steps=HOLD30_STAGE1_WARMUP_STEPS,
        schedule=HOLD30_STAGE1_SCHEDULE,
        grad_clip=design.grad_clip,
        amp=design.amp and device.type == "cuda",
        start_step=completed,
        optimizer=optimizer,
        checkpoint_every=cadence,
        on_checkpoint=on_checkpoint,
        grad_reduce=sum_reduce if distributed_context.enabled else None,
        grad_reduce_mode="sum" if distributed_context.enabled else None,
        prepare_checkpoint=prepare_checkpoint,
        sync_after_checkpoint=(dist.barrier if distributed_context.enabled else None),
        effective_index_schedule=schedule[:target],
        distributed_rank=rank,
        distributed_world_size=world_size,
        global_valid_normalization=True,
    )
    progress = _rank0_call(distributed_context, inspect_progress)
    completed = int(progress["completed"])
    previous_receipt = str(progress["previous"])
    latest = Path(progress["latest"]) if progress["latest"] is not None else None
    if completed != target:
        raise Hold30Stage1Error("trainer returned without publishing the expected checkpoint")
    if completed != HOLD30_STAGE1_STEPS:
        return Hold30Stage1Progress(False, completed, HOLD30_STAGE1_STEPS, latest, None)

    _require_rank_equal(
        distributed_context,
        "completed encoder state",
        _state_sha256(_cpu_state(encoder)),
    )
    _rank0_call(
        distributed_context,
        lambda: _publish_or_validate_final(
            run_root,
            identity_doc,
            identity,
            encoder,
            previous_receipt=previous_receipt,
            execution_world_size=world_size,
        ),
    )
    return Hold30Stage1Progress(True, completed, HOLD30_STAGE1_STEPS, latest, run_receipt_path)


__all__ = [
    "HOLD30_STAGE1_ACCUMULATION",
    "HOLD30_STAGE1_BATCH_SIZE",
    "HOLD30_STAGE1_CHECKPOINT_EVERY",
    "HOLD30_STAGE1_DAILY_COEFFICIENT",
    "HOLD30_STAGE1_EFFECTIVE_BATCH",
    "HOLD30_STAGE1_EXECUTION_STRATEGY",
    "HOLD30_STAGE1_HORIZON",
    "HOLD30_STAGE1_LR",
    "HOLD30_STAGE1_PER_STOCK_COEFFICIENT",
    "HOLD30_STAGE1_SCHEDULE",
    "HOLD30_STAGE1_SHARING_SCOPE",
    "HOLD30_STAGE1_STEPS",
    "HOLD30_STAGE1_WARMUP_STEPS",
    "HOLD30_STAGE1_WEIGHT_DECAY",
    "Hold30Stage1Error",
    "Hold30Stage1FreezeBlocker",
    "Hold30Stage1FreezeDecision",
    "Hold30Stage1Identity",
    "Hold30Stage1Progress",
    "Hold30Stage1TrainingData",
    "derive_hold30_stage1_seed",
    "hold30_stage1_contract",
    "hold30_stage1_freeze_status",
    "hold30_stage1_normalization_schedule_sha256",
    "hold30_stage1_optimizer_schedule_sha256",
    "hold30_stage1_training_tensor_sha256",
    "materialize_hold30_stage1_schedule",
    "run_hold30_stage1",
    "verify_hold30_stage1_run",
]
