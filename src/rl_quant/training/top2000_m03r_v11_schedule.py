"""Setting-neutral paired episode schedule for M03R-v11."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_EPISODE_SCHEDULE_RULE,
    M03R_V11_PREDICTIVE_SPEC,
    M03R_V11_PROTOCOL_SHA256,
)

M03R_V11_PANEL_SCHEDULE_SCHEMA = "rl-quant.top2000-dev.m03r-v11-panel-schedule-v1"
M03R_V11_PAIRED_INPUT_SCHEMA = "rl-quant.top2000-dev.m03r-v11-paired-input-v1"


class M03RV11ScheduleError(ValueError):
    """The v11 paired episode or rank-shard schedule drifted."""


class M03RV11TrainingShardLike(Protocol):
    @property
    def panel_episode_schedule_sha256(self) -> str: ...

    @property
    def fold_index(self) -> int: ...

    @property
    def completed_update(self) -> int: ...

    @property
    def episode_start(self) -> int: ...

    @property
    def global_origins(self) -> tuple[int, ...]: ...

    @property
    def rank_origins(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11ScheduleError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV11PanelEpisodeSchedule:
    protocol_common_data_sha256: str
    cache_sha256: str
    fold_geometry_sha256: tuple[str, ...]
    seed: int = 17
    optimizer_updates: int = 64
    schedule_rule: str = M03R_V11_EPISODE_SCHEDULE_RULE
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_PANEL_SCHEDULE_SCHEMA

    def validate(self) -> None:
        if (
            len(self.fold_geometry_sha256)
            != M03R_V11_PREDICTIVE_SPEC.chronological_fold_count
            or len(set(self.fold_geometry_sha256)) != len(self.fold_geometry_sha256)
            or self.seed != M03R_V11_PREDICTIVE_SPEC.seed
            or self.optimizer_updates != M03R_V11_PREDICTIVE_SPEC.optimizer_updates
            or self.schedule_rule != M03R_V11_EPISODE_SCHEDULE_RULE
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_PANEL_SCHEDULE_SCHEMA
        ):
            raise M03RV11ScheduleError("v11 panel episode schedule drifted")
        for name, value in (
            ("protocol_common_data_sha256", self.protocol_common_data_sha256),
            ("cache_sha256", self.cache_sha256),
            *(("fold_geometry_sha256", value) for value in self.fold_geometry_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def deterministic_m03r_v11_episode_start(
    schedule: M03RV11PanelEpisodeSchedule,
    *,
    fold_index: int,
    completed_updates: int,
    admissible_start_count: int,
) -> int:
    """Choose an episode without consulting setting or worker identity."""

    schedule.validate()
    if (
        isinstance(fold_index, bool)
        or not isinstance(fold_index, int)
        or fold_index not in range(len(schedule.fold_geometry_sha256))
        or isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates not in range(schedule.optimizer_updates)
        or isinstance(admissible_start_count, bool)
        or not isinstance(admissible_start_count, int)
        or admissible_start_count <= 0
    ):
        raise M03RV11ScheduleError("v11 episode cursor or geometry is invalid")
    digest = hashlib.sha256(
        (
            f"{schedule.receipt_sha256}:"
            f"{schedule.fold_geometry_sha256[fold_index]}:"
            f"{fold_index}:{completed_updates}:{schedule.schedule_rule}"
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % admissible_start_count


def m03r_v11_complementary_rank_shards(
    global_origins: tuple[int, ...],
    *,
    world_size: int = 2,
) -> tuple[tuple[int, ...], ...]:
    if (
        not global_origins
        or any(isinstance(value, bool) or value < 0 for value in global_origins)
        or any(later <= earlier for earlier, later in pairwise(global_origins))
        or world_size != M03R_V11_PREDICTIVE_SPEC.expected_world_size
    ):
        raise M03RV11ScheduleError("v11 global origins or world size drifted")
    usable = global_origins[: len(global_origins) - len(global_origins) % world_size]
    if len(usable) < world_size:
        raise M03RV11ScheduleError("v11 paired origin set is too small")
    shards = tuple(usable[rank::world_size] for rank in range(world_size))
    if (
        any(not shard for shard in shards)
        or set(shards[0]).intersection(shards[1])
        or tuple(sorted((*shards[0], *shards[1]))) != usable
    ):
        raise M03RV11ScheduleError("v11 rank shards are not complementary")
    return shards


@dataclass(frozen=True, slots=True)
class M03RV11PairedInputReceipt:
    schedule_sha256: str
    fold_index: int
    completed_update: int
    episode_start: int
    global_origins: tuple[int, ...]
    rank_origin_sha256: tuple[str, str]
    input_tensor_sha256: tuple[str, ...]
    source_array_sha256: str
    asset_axis_sha256: str
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_PAIRED_INPUT_SCHEMA

    def validate(self) -> None:
        shards = m03r_v11_complementary_rank_shards(self.global_origins)
        expected_rank_hashes = tuple(_sha256(shard) for shard in shards)
        if (
            self.fold_index not in range(6)
            or self.completed_update not in range(64)
            or self.episode_start < 0
            or not self.input_tensor_sha256
            or self.rank_origin_sha256 != expected_rank_hashes
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_PAIRED_INPUT_SCHEMA
        ):
            raise M03RV11ScheduleError("v11 paired input receipt drifted")
        for name, value in (
            ("schedule_sha256", self.schedule_sha256),
            ("source_array_sha256", self.source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            *(("input_tensor_sha256", value) for value in self.input_tensor_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise M03RV11ScheduleError(
            "PyTorch is required for v11 tensor schedule operations"
        ) from exc
    return torch


def _tensor_sha256(value: torch.Tensor) -> str:
    torch = _torch_module()
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_m03r_v11_paired_input_receipt(
    training_shard: M03RV11TrainingShardLike,
    input_tensors: tuple[torch.Tensor, ...],
    *,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> M03RV11PairedInputReceipt:
    """Bind the setting-neutral full-episode tensors before rank sharding."""

    torch = _torch_module()

    global_origins = training_shard.global_origins
    rank_origins = training_shard.rank_origins
    if (
        not input_tensors
        or any(not isinstance(value, torch.Tensor) for value in input_tensors)
        or any(value.numel() == 0 for value in input_tensors)
        or rank_origins != m03r_v11_complementary_rank_shards(global_origins)
    ):
        raise M03RV11ScheduleError("v11 paired input tensor inventory drifted")
    result = M03RV11PairedInputReceipt(
        schedule_sha256=str(training_shard.panel_episode_schedule_sha256),
        fold_index=training_shard.fold_index,
        completed_update=training_shard.completed_update,
        episode_start=training_shard.episode_start,
        global_origins=global_origins,
        rank_origin_sha256=(
            _sha256(rank_origins[0]),
            _sha256(rank_origins[1]),
        ),
        input_tensor_sha256=tuple(_tensor_sha256(value) for value in input_tensors),
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V11_PAIRED_INPUT_SCHEMA",
    "M03R_V11_PANEL_SCHEDULE_SCHEMA",
    "M03RV11PairedInputReceipt",
    "M03RV11PanelEpisodeSchedule",
    "M03RV11ScheduleError",
    "M03RV11TrainingShardLike",
    "build_m03r_v11_paired_input_receipt",
    "deterministic_m03r_v11_episode_start",
    "m03r_v11_complementary_rank_shards",
]
