from __future__ import annotations

from contextlib import nullcontext
import math
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
        first_value = next(iter(transition.values()))
        count = int(first_value.shape[0])
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
        first_value = next(iter(transition.values()))
        count = int(first_value.shape[0])
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


def autocast_context(device: torch.device, requested: bool):
    enabled = cuda_amp_enabled(device, requested)
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(device: torch.device, requested: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=cuda_amp_enabled(device, requested))


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
