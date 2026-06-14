from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from os import PathLike
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.core import (
    DQNLearningConfig,
    TensorReplayBuffer,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    epsilon_by_step,
    fractional_max_drawdown,
    make_grad_scaler,
)


@dataclass
class HourlyDataSplit:
    name: str
    timestamps: list[str]
    feature_names: list[str]
    action_names: list[str]
    features: torch.Tensor
    action_returns: torch.Tensor
    session_dates: list[str] | None
    valid_start_indices: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    lookback: int
    periods_per_year: float = 252.0 * 6.5
    bar_interval: str = "1h"

    def to(self, device: torch.device | str) -> "HourlyDataSplit":
        return replace(
            self,
            features=self.features.to(device),
            action_returns=self.action_returns.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
        )

    def state_windows(self, indices: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.lookback, device=indices.device, dtype=torch.long)
        window_indices = indices.unsqueeze(1) - (self.lookback - 1) + offsets.unsqueeze(0)
        return self.features[window_indices]


def _load_payload(path: str | bytes | PathLike[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"timestamps", "feature_names", "action_names", "features", "action_returns"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Transformer dataset is missing required keys: {sorted(missing)}")
    return payload


def _build_split(
    *,
    name: str,
    payload: dict[str, Any],
    lookback: int,
    start_ts: str | None = None,
    end_ts: str | None = None,
    reward_start_ts: str | None = None,
    reward_after_ts: str | None = None,
    reward_end_ts: str | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> HourlyDataSplit:
    all_timestamps = list(payload["timestamps"])
    all_features = payload["features"].float()
    all_returns = payload["action_returns"].float()
    all_session_dates = payload.get("session_dates")
    selected = [
        i
        for i, ts in enumerate(all_timestamps)
        if (start_ts is None or ts >= start_ts) and (end_ts is None or ts <= end_ts)
    ]
    if len(selected) < lookback + 2:
        raise ValueError(f"Need at least lookback + 2 rows for split {name!r}, got {len(selected)}.")

    timestamps = [all_timestamps[i] for i in selected]
    session_dates = [all_session_dates[i] for i in selected] if all_session_dates is not None else None
    raw_features = all_features[selected]
    action_returns = all_returns[selected]
    if feature_mean is None:
        feature_mean = raw_features.mean(dim=0)
    if feature_std is None:
        feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    features = ((raw_features - feature_mean) / feature_std).clamp_(-8.0, 8.0)
    valid: list[int] = []
    require_same_session = bool(payload.get("require_same_session_lookback", False))
    for index in range(lookback - 1, len(timestamps) - 1):
        reward_ts = timestamps[index]
        if reward_after_ts is not None and reward_ts <= reward_after_ts:
            continue
        if reward_start_ts is not None and reward_ts < reward_start_ts:
            continue
        if reward_end_ts is not None and reward_ts > reward_end_ts:
            continue
        if require_same_session and session_dates is not None:
            window_dates = session_dates[index - lookback + 1 : index + 1]
            if any(date != window_dates[-1] for date in window_dates):
                continue
        valid.append(index)
    if not valid:
        raise ValueError(f"No valid reward indices remain for split {name!r}.")

    return HourlyDataSplit(
        name=name,
        timestamps=timestamps,
        feature_names=list(payload["feature_names"]),
        action_names=list(payload["action_names"]),
        features=features,
        action_returns=action_returns,
        session_dates=session_dates,
        valid_start_indices=torch.tensor(valid, dtype=torch.long),
        feature_mean=feature_mean,
        feature_std=feature_std,
        lookback=lookback,
        periods_per_year=float(payload.get("periods_per_year", 252.0 * 6.5)),
        bar_interval=str(payload.get("bar_interval", "1h")),
    )


def build_hourly_splits(
    *,
    dataset_path,
    lookback: int,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> tuple[HourlyDataSplit, HourlyDataSplit, HourlyDataSplit]:
    payload = _load_payload(dataset_path)
    train = _build_split(
        name="train",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=train_end,
        reward_end_ts=train_end,
    )
    val = _build_split(
        name="val",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=val_end,
        reward_after_ts=train_end,
        reward_end_ts=val_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    test = _build_split(
        name="test",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=test_end,
        reward_start_ts=test_start,
        reward_end_ts=test_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    return train, val, test


class CausalTransformerQNetwork(nn.Module):
    """Causal Q-network over bar-based market context windows."""

    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        action_count: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.lookback = int(lookback)
        self.action_count = int(action_count)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(lookback, d_model))
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.previous_action_proj = nn.Linear(action_embedding_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, action_count),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), torch.finfo(torch.float32).min, device=device),
            diagonal=1,
        )

    def forward(self, state_windows: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        length = state_windows.shape[1]
        if length > self.lookback:
            raise ValueError(f"Window length {length} exceeds configured lookback {self.lookback}.")
        x = self.input_proj(state_windows)
        x = x + self.position_embedding[-length:][None, :, :]
        action_context = self.previous_action_proj(self.previous_action_embedding(previous_actions.long()))
        x = x + action_context[:, None, :]
        x = self.encoder(x, mask=self._causal_mask(length, x.device))
        return self.head(self.out_norm(x[:, -1, :]))


@dataclass
class HourlyEnvConfig:
    lookback: int
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    switch_cost_bps: float = 1.0
    initial_action: int = 0


@dataclass
class HourlyTransformerTrainingConfig:
    env: HourlyEnvConfig
    learning: DQNLearningConfig
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.05
    action_embedding_dim: int = 32
    target_vram_gb: float | None = None
    vram_safety_gb: float = 0.12


class VectorizedHourlyAllocationEnv:
    def __init__(self, data: HourlyDataSplit, config: HourlyEnvConfig, device: torch.device) -> None:
        if config.lookback != data.lookback:
            raise ValueError("HourlyEnvConfig.lookback must match the loaded split.")
        if not (0 <= config.initial_action < len(data.action_names)):
            raise ValueError("initial_action is outside the action space.")
        self.data = data if data.features.device == device else data.to(device)
        self.config = config
        self.device = device
        self.start_indices = self._build_start_index_pool()
        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.previous_actions = torch.full(
            (config.num_envs,),
            int(config.initial_action),
            dtype=torch.long,
            device=device,
        )
        self.steps = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        starts = self.data.valid_start_indices
        starts = starts[starts + 1 < self.data.action_returns.shape[0]]
        if starts.numel() == 0:
            raise ValueError("No valid bar start indices remain.")
        return starts.to(self.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.config.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count == 0:
            return
        random_ids = torch.randint(0, self.start_indices.shape[0], (count,), device=self.device)
        self.indices[mask] = self.start_indices[random_ids]
        self.previous_actions[mask] = int(self.config.initial_action)
        self.steps[mask] = 0

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data.state_windows(self.indices), self.previous_actions

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        raw_returns = self.data.action_returns[current_indices, actions]
        switch_penalty = (
            (actions != previous_actions).float()
            * float(self.config.switch_cost_bps)
            * float(self.config.reward_scale)
            / 10_000.0
        )
        rewards = raw_returns * float(self.config.reward_scale) - switch_penalty
        next_indices = current_indices + 1

        self.indices = next_indices
        self.previous_actions = actions.long()
        self.steps = self.steps + 1
        dones = (next_indices + 1 >= self.data.action_returns.shape[0]) | (
            self.steps >= int(self.config.episode_length)
        )
        return {
            "indices": current_indices,
            "previous_actions": previous_actions,
            "actions": actions,
            "rewards": rewards,
            "next_indices": next_indices,
            "next_previous_actions": self.previous_actions,
            "dones": dones.float(),
        }


@dataclass
class HourlyEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    total_switches: int
    max_drawdown: float
    hourly_sharpe: float | None
    rollout_records: list[dict[str, float | str | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "total_switches": self.total_switches,
            "max_drawdown": self.max_drawdown,
            "hourly_sharpe": self.hourly_sharpe,
            "annualized_sharpe": self.hourly_sharpe,
            "rollout_records": self.rollout_records,
        }


@torch.no_grad()
def evaluate_hourly_policy(
    data: HourlyDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    switch_cost_bps: float = 1.0,
    capture_rollout: bool = False,
) -> HourlyEvaluationResult:
    data = data if data.features.device == device else data.to(device)
    model.eval()
    previous_action = int(initial_action)
    equity = 1.0
    equity_curve = [equity]
    bar_returns: list[float] = []
    total_reward_bps = 0.0
    switches = 0
    records: list[dict[str, float | str | int]] = []
    start = int(data.valid_start_indices[0].item())
    end = data.action_returns.shape[0] - 1
    for index in range(start, end):
        index_tensor = torch.tensor([index], dtype=torch.long, device=device)
        previous_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        q_values = model(data.state_windows(index_tensor), previous_tensor)
        action = int(torch.argmax(q_values, dim=1)[0].item())
        gross_return = float(data.action_returns[index, action].item())
        switch_cost = float(switch_cost_bps) / 10_000.0 if action != previous_action else 0.0
        net_return = gross_return - switch_cost
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        bar_returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        if action != previous_action:
            switches += 1
        if capture_rollout:
            records.append(
                {
                    "timestamp": data.timestamps[index],
                    "action": action,
                    "asset": data.action_names[action],
                    "previous_action": previous_action,
                    "bar_interval": data.bar_interval,
                    "bar_return": round(net_return, 8),
                    "hourly_return": round(net_return, 8),
                    "equity": round(equity, 8),
                }
            )
        previous_action = action
    return HourlyEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        total_switches=switches,
        max_drawdown=fractional_max_drawdown(equity_curve),
        hourly_sharpe=annualized_sharpe(bar_returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
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


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_hourly_transformer_dqn(
    train_data: HourlyDataSplit,
    val_data: HourlyDataSplit,
    *,
    device: torch.device,
    config: HourlyTransformerTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    train_data = train_data if train_data.features.device == device else train_data.to(device)
    val_data = val_data if val_data.features.device == device else val_data.to(device)
    action_count = len(train_data.action_names)
    if action_count != len(val_data.action_names):
        raise ValueError("Train and validation action spaces differ.")

    q_network = CausalTransformerQNetwork(
        feature_dim=train_data.features.shape[1],
        lookback=config.env.lookback,
        action_count=action_count,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        action_embedding_dim=config.action_embedding_dim,
    ).to(device)
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp)
    replay = TensorReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields={
            "indices": torch.long,
            "previous_actions": torch.long,
            "actions": torch.long,
            "rewards": torch.float32,
            "next_indices": torch.long,
            "next_previous_actions": torch.long,
            "dones": torch.float32,
        },
    )
    env = VectorizedHourlyAllocationEnv(train_data, config.env, device)
    reservation = CudaVramReservation(
        target_gb=config.target_vram_gb,
        safety_gb=config.vram_safety_gb,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_return = -float("inf")
    best_val_switches = 10**12
    best_state = _state_dict_to_cpu(q_network)
    loss_trace: list[float] = []
    reward_trace: list[float] = []
    eval_trace: list[dict[str, float | int | None | str]] = []

    for step in range(1, config.learning.train_steps + 1):
        states, previous_actions = env.observe()
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp):
                q_values = q_network(states, previous_actions)
            greedy_actions = torch.argmax(q_values, dim=1)
            random_actions = torch.randint(0, action_count, greedy_actions.shape, device=device)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        transition = env.step(actions)
        replay.add(**transition)
        reward_trace.append(float(transition["rewards"].mean().item()))
        env.reset(transition["dones"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            current_states = train_data.state_windows(batch["indices"])
            next_states = train_data.state_windows(batch["next_indices"])
            with autocast_context(device, config.learning.use_amp):
                chosen_q = q_network(current_states, batch["previous_actions"]).gather(
                    1,
                    batch["actions"].unsqueeze(1),
                ).squeeze(1)
                with torch.no_grad():
                    next_actions = torch.argmax(
                        q_network(next_states, batch["next_previous_actions"]),
                        dim=1,
                    )
                    next_q = target_network(next_states, batch["next_previous_actions"]).gather(
                        1,
                        next_actions.unsqueeze(1),
                    ).squeeze(1)
                    target_q = batch["rewards"] + config.learning.gamma * (1.0 - batch["dones"]) * next_q
                loss = F.smooth_l1_loss(chosen_q, target_q)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.learning.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.learning.grad_clip)
                optimizer.step()
            loss_trace.append(float(loss.item()))
            reservation.maybe_reserve(device)

        if step % config.learning.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if step % config.learning.eval_interval == 0 or step == config.learning.train_steps:
            val_result = evaluate_hourly_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                switch_cost_bps=config.env.switch_cost_bps,
            )
            avg_loss = sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1)
            avg_reward = sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1)
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_switches": val_result.total_switches,
                    "val_sharpe": val_result.hourly_sharpe,
                    "average_loss": avg_loss,
                    "average_train_reward": avg_reward,
                }
            )
            if (
                val_result.total_return > best_val_return
                or (
                    abs(val_result.total_return - best_val_return) <= 1e-12
                    and val_result.total_switches < best_val_switches
                )
            ):
                best_val_return = val_result.total_return
                best_val_switches = val_result.total_switches
                best_state = _state_dict_to_cpu(q_network)

    q_network.load_state_dict(best_state)
    artifacts: dict[str, object] = {
        "best_val_return": best_val_return,
        "best_val_switches": best_val_switches,
        "amp_enabled": scaler.is_enabled(),
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        artifacts.update(
            {
                "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
                "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 4),
                "cuda_device_used_end_gb": round((total - free) / 1024**3, 4),
                "cuda_device_free_end_gb": round(free / 1024**3, 4),
            }
        )
    return q_network, artifacts


def action_index(action_names: list[str], action_name: str) -> int:
    try:
        return action_names.index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action_name!r}") from exc
