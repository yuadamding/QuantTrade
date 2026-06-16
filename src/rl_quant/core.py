from __future__ import annotations

from contextlib import nullcontext
import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class DQNLearningConfig:
    num_envs: int
    episode_length: int
    replay_capacity: int
    batch_size: int
    train_steps: int
    warmup_steps: int
    gamma: float
    learning_rate: float
    weight_decay: float
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    eval_interval: int
    grad_clip: float
    use_amp: bool = False
    amp_dtype: str = "fp16"  # AMP autocast precision when use_amp: "fp16" (default) or "bf16".


class TemporalQNetwork(nn.Module):
    """Small shared Q-network for windowed market states plus a previous action."""

    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        action_count: int,
        previous_action_count: int,
        hidden_size: int = 128,
        previous_action_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if action_count <= 1:
            raise ValueError("action_count must be greater than one")
        if previous_action_count <= 0:
            raise ValueError("previous_action_count must be positive")

        self.lookback = lookback
        self.action_count = action_count
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.previous_action_embedding = nn.Embedding(
            previous_action_count,
            previous_action_embedding_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(64 * lookback + previous_action_embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, state_windows: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        x = self.temporal(state_windows.transpose(1, 2)).reshape(state_windows.shape[0], -1)
        previous_action_features = self.previous_action_embedding(previous_actions.long())
        return self.head(torch.cat([x, previous_action_features], dim=1))


def _validate_replay_batch(
    storage: dict[str, torch.Tensor], transition: dict[str, torch.Tensor]
) -> int:
    """Validate declared replay fields share one leading batch dim and match the stored trailing
    shape, and RETURN that canonical batch size. The returned count is derived only from declared
    fields, so an extra transition key (e.g. legs/resets) appearing first cannot define the write
    size. Extra keys are otherwise ignored, matching add()'s field-driven write."""
    counts: set[int] = set()
    for name, target in storage.items():
        value = transition[name]
        if value.ndim == 0:
            raise ValueError(f"replay field {name!r} needs a leading batch dimension")
        if tuple(value.shape[1:]) != tuple(target.shape[1:]):
            raise ValueError(
                f"replay field {name!r} trailing shape {tuple(value.shape[1:])} != "
                f"expected {tuple(target.shape[1:])}"
            )
        counts.add(int(value.shape[0]))
    if len(counts) > 1:
        raise ValueError(f"mismatched replay batch sizes across fields: {sorted(counts)}")
    return next(iter(counts)) if counts else 0


class TensorReplayBuffer:
    """Circular replay buffer for tensor transitions with fixed field names."""

    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device,
        fields: dict[str, torch.dtype],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.storage = {
            name: torch.zeros(capacity, dtype=dtype, device=device)
            for name, dtype in fields.items()
        }
        self.size = 0
        self.cursor = 0

    def add(self, **transition: torch.Tensor) -> None:
        missing = set(self.storage) - set(transition)
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        # Canonical batch size comes from declared fields only -- never from an arbitrary (possibly
        # extra) first transition value, which could have a different leading dim.
        count = _validate_replay_batch(self.storage, transition)
        if count == 0:
            return

        if count >= self.capacity:
            # A single add() larger than capacity keeps only the most recent `capacity` rows.
            # Callers adding `num_envs` transitions per step should size capacity >> num_envs so
            # a step's batch is never silently dropped.
            for name in self.storage:
                transition[name] = transition[name][-self.capacity :]
            count = self.capacity

        first = min(count, self.capacity - self.cursor)
        second = count - first
        for name, target in self.storage.items():
            values = transition[name].to(device=self.device, dtype=target.dtype)
            target[self.cursor : self.cursor + first] = values[:first]
            if second:
                target[:second] = values[first:]

        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_ids = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {name: values[batch_ids] for name, values in self.storage.items()}


class TensorDictReplayBuffer:
    """Circular replay buffer for tensor transitions with scalar or shaped fields."""

    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device,
        fields: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.storage = {
            name: torch.zeros((capacity, *shape), dtype=dtype, device=device)
            for name, (shape, dtype) in fields.items()
        }
        self.size = 0
        self.cursor = 0

    def add(self, **transition: torch.Tensor) -> None:
        missing = set(self.storage) - set(transition)
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        # Canonical batch size comes from declared fields only -- never from an arbitrary (possibly
        # extra) first transition value, which could have a different leading dim.
        count = _validate_replay_batch(self.storage, transition)
        if count == 0:
            return
        if count >= self.capacity:
            for name in self.storage:
                transition[name] = transition[name][-self.capacity :]
            count = self.capacity
        first = min(count, self.capacity - self.cursor)
        second = count - first
        for name, target in self.storage.items():
            values = transition[name].to(device=self.device, dtype=target.dtype)
            target[self.cursor : self.cursor + first] = values[:first]
            if second:
                target[:second] = values[first:]
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_ids = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {name: values[batch_ids] for name, values in self.storage.items()}


def epsilon_by_step(*, step: int, train_steps: int, start: float, end: float) -> float:
    if train_steps <= 0:
        return end
    fraction_left = max(0.0, 1.0 - step / train_steps)
    return end + (start - end) * fraction_left


def dqn_td_target(
    rewards: torch.Tensor,
    gamma: float,
    terminated: torch.Tensor,
    next_q: torch.Tensor,
) -> torch.Tensor:
    """Double-DQN Bellman target that bootstraps through episode-length TRUNCATIONS.

    Only ``terminated`` (a true terminal with no valid next row) zeros the bootstrap; a mere
    rollout-length truncation keeps bootstrapping because its next row is a real continuation.
    Passing the episode-end reset mask here instead would wrongly treat every truncation as
    terminal and bias values toward short horizons.

    Uses ``torch.where`` rather than ``(1 - terminated) * next_q`` so a terminal row's ``next_q`` of
    NaN/Inf (e.g. from an empty next action mask or a clamped out-of-data dummy state) cannot
    propagate into the target -- the unselected branch never contaminates terminal positions.
    ``next_q`` is detached (a TD target must not backpropagate) and the target is computed in
    float32 (AMP-safe: with large reward_scale, fp16 target resolution is comparable to per-step
    rewards). Requires rewards/terminated/next_q to share one shape to avoid silent broadcasting
    (e.g. (B, 1) vs (B,) collapsing to (B, B))."""
    if rewards.shape != next_q.shape or rewards.shape != terminated.shape:
        raise ValueError(
            "dqn_td_target expects rewards, terminated, and next_q to share one shape; got "
            f"rewards={tuple(rewards.shape)}, terminated={tuple(terminated.shape)}, "
            f"next_q={tuple(next_q.shape)}."
        )
    gamma_f = float(gamma)
    # The range check also rejects NaN/inf (any comparison with NaN is False; inf fails the upper bound).
    if not 0.0 <= gamma_f <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1]; got {gamma!r}.")
    rewards_f = rewards.float()
    bootstrap = torch.where(terminated.bool(), torch.zeros_like(rewards_f), next_q.detach().float())
    return rewards_f + gamma_f * bootstrap


def annualized_sharpe(values: list[float], periods_per_year: float = 252.0) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    sigma = math.sqrt(variance)
    return None if sigma <= 0 else avg / sigma * math.sqrt(periods_per_year)


def fractional_max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def absolute_max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def resolve_torch_device(preference: str = "auto") -> torch.device:
    normalized = preference.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{preference} was requested, but CUDA is unavailable.")
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {preference}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is outside available range 0..{torch.cuda.device_count() - 1}."
            )
        return torch.device(normalized)
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def configure_torch_runtime(device: torch.device) -> None:
    use_cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = use_cuda
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = use_cuda
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = use_cuda
    torch.set_float32_matmul_precision("high")


def cuda_amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")


_AMP_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_amp_dtype(name: str) -> torch.dtype:
    """Map an AMP precision name to a torch dtype (whitespace/case-insensitive). fp32 is not an AMP
    dtype (disable AMP instead)."""
    normalized = str(name).strip().lower()
    try:
        return _AMP_DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported amp dtype {name!r}; choose fp16 or bf16.") from exc


def autocast_context(device: torch.device, requested: bool, amp_dtype: str = "fp16"):
    # Resolve eagerly so a typo'd amp_dtype is rejected even when AMP is disabled (e.g. a CPU dry run),
    # rather than silently ignored until a CUDA run.
    dtype = resolve_amp_dtype(amp_dtype)
    if not cuda_amp_enabled(device, requested):
        return nullcontext()
    # bf16 has a wider exponent range than fp16 and is preferred on Ampere/Hopper-class GPUs; fail
    # clearly rather than silently if the device cannot do bf16.
    if dtype is torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 AMP requested, but this device does not support bf16.")
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, requested: bool, amp_dtype: str = "fp16") -> torch.amp.GradScaler:
    if requested and device.type != "cuda":
        warnings.warn(
            f"AMP/mixed precision was requested but the device is {device.type!r}, not CUDA; "
            "AMP is disabled and this run uses fp32. Pass --device cuda for mixed precision.",
            stacklevel=2,
        )
    # Loss scaling is an fp16-only concern; bf16 has fp32-like range and needs no GradScaler, so the
    # scaler is enabled only for cuda + requested + fp16. Resolve eagerly so an invalid amp_dtype is
    # rejected even when AMP is disabled (rather than silently passing on CPU).
    dtype = resolve_amp_dtype(amp_dtype)
    enabled = cuda_amp_enabled(device, requested) and dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def cuda_memory_report(device: torch.device, *, round_digits: int | None = None) -> dict[str, float]:
    """Point-in-time CUDA memory snapshot (allocated/reserved/peak/free/total GiB).

    A true accounting of what training actually occupies, unlike the VRAM-ballast reservation
    (which *increases* usage toward a target). Returns zeros for non-CUDA devices so callers can
    log/guard unconditionally. Values are RAW by default (use them for threshold guards); pass
    ``round_digits`` only for human-readable logs/manifests."""
    if device.type != "cuda":
        report = {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}
    else:
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        report = {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            "free_gb": free / 1024**3,
            "total_gb": total / 1024**3,
        }
    if round_digits is not None:
        return {key: round(value, round_digits) for key, value in report.items()}
    return report


def require_min_free_vram(device: torch.device, min_free_gb: float, *, phase: str = "training") -> None:
    """Fail fast when free CUDA memory is below ``min_free_gb`` GiB (no-op off CUDA or when <= 0).

    Uses raw (unrounded) free memory for the comparison. A preflight guard against OOM, distinct
    from the VRAM-ballast reservation, which intentionally consumes memory."""
    if device.type != "cuda" or min_free_gb <= 0:
        return
    free_gb = cuda_memory_report(device)["free_gb"]
    if free_gb < min_free_gb:
        raise SystemExit(
            f"insufficient free CUDA memory before {phase}: {free_gb:.2f} GiB free < "
            f"--min-free-vram-gb {min_free_gb:.2f} GiB. Free memory or lower the requirement."
        )


class CudaVramReservation:
    def __init__(self, *, target_gb: float | None, safety_gb: float) -> None:
        self.target_gb = target_gb
        self.safety_gb = safety_gb
        self.chunks: list[torch.Tensor] = []
        self.report: dict[str, float | int | str] = {}

    def maybe_reserve(self, device: torch.device) -> None:
        if self.target_gb is None or device.type != "cuda" or self.chunks:
            return
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        target = min(int(self.target_gb * 1024**3), total - int(self.safety_gb * 1024**3))
        bytes_to_reserve = max(target - used, 0)
        max_chunk = 1_024**3
        remaining = bytes_to_reserve
        while remaining > 0:
            chunk_bytes = min(remaining, max_chunk)
            chunk = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
            chunk.zero_()
            self.chunks.append(chunk)
            remaining -= chunk_bytes
        torch.cuda.synchronize(device)
        free_after, total_after = torch.cuda.mem_get_info(device)
        self.report = {
            "target_gb": float(self.target_gb),
            "safety_gb": float(self.safety_gb),
            "reserved_ballast_gb": round(bytes_to_reserve / 1024**3, 4),
            "device_used_after_reserve_gb": round((total_after - free_after) / 1024**3, 4),
            "device_total_gb": round(total_after / 1024**3, 4),
            "chunks": len(self.chunks),
        }


def torch_runtime_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / 1024**3, 3),
                "cuda_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
    return summary
