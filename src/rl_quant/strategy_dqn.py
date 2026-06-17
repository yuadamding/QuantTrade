from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.models.strategy import StrategyQNetwork
from rl_quant.core import (
    DQNLearningConfig,
    TensorReplayBuffer,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    dqn_td_target,
    epsilon_by_step,
    fractional_max_drawdown,
    make_grad_scaler,
    safe_next_row_indices,
)
from rl_quant.strategy_data import StrategyDataSplit
from rl_quant.strategy_data import assert_matching_strategy_schema


@dataclass
class StrategyEnvConfig:
    lookback: int
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    switch_cost_bps: float = 0.0
    initial_action: int = 0


@dataclass
class StrategyTrainingConfig:
    env: StrategyEnvConfig
    learning: DQNLearningConfig
    hidden_size: int = 128
    action_embedding_dim: int = 16


class VectorizedStrategyAllocationEnv:
    def __init__(self, data: StrategyDataSplit, config: StrategyEnvConfig, device: torch.device) -> None:
        if config.lookback != data.lookback:
            raise ValueError("StrategyEnvConfig.lookback must match the loaded StrategyDataSplit.")
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
            raise ValueError("No valid start indices remain for the strategy allocation environment.")
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
        next_indices = current_indices + 1

        raw_returns = self.data.action_returns[next_indices, actions]
        switch_penalty = (
            (actions != previous_actions).float()
            * float(self.config.switch_cost_bps)
            * float(self.config.reward_scale)
            / 10_000.0
        )
        rewards = raw_returns * float(self.config.reward_scale) - switch_penalty

        self.indices = next_indices
        self.previous_actions = actions.long()
        self.steps = self.steps + 1
        in_bounds = next_indices + 1 < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        # Separate the genuine terminal (no more valid data) from the artificial episode-length
        # truncation. The TD target bootstraps THROUGH truncations (the strategy curve continues),
        # zeroing the bootstrap only at true terminals; `resets` drives the env episode reset.
        terminal = ~next_valid
        truncated = self.steps >= int(self.config.episode_length)
        resets = terminal | truncated

        return {
            "indices": current_indices,
            "previous_actions": previous_actions,
            "actions": actions,
            "rewards": rewards,
            "next_indices": next_indices,
            "next_previous_actions": self.previous_actions,
            "terminated": terminal,
            "resets": resets,
        }


@dataclass
class StrategyEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    total_switches: int
    max_drawdown: float
    daily_sharpe: float | None
    rollout_records: list[dict[str, float | str | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "total_switches": self.total_switches,
            "max_drawdown": self.max_drawdown,
            "daily_sharpe": self.daily_sharpe,
            "rollout_records": self.rollout_records,
        }


@torch.no_grad()
def evaluate_strategy_policy(
    data: StrategyDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    switch_cost_bps: float = 0.0,
    capture_rollout: bool = False,
) -> StrategyEvaluationResult:
    if not (0 <= initial_action < len(data.action_names)):
        raise ValueError("initial_action is outside the action space.")

    data = data if data.features.device == device else data.to(device)
    model.eval()
    previous_action = int(initial_action)
    equity = 1.0
    equity_curve = [equity]
    daily_returns: list[float] = []
    total_reward_bps = 0.0
    switches = 0
    rollout_records: list[dict[str, float | str | int]] = []

    previous_index: int | None = None
    for index in data.valid_start_indices.detach().cpu().tolist():
        segment_reset = previous_index is None or index != previous_index + 1
        if segment_reset:
            previous_action = int(initial_action)
        index_tensor = torch.tensor([index], dtype=torch.long, device=device)
        previous_action_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        q_values = model(data.state_windows(index_tensor), previous_action_tensor)
        action = int(torch.argmax(q_values, dim=1)[0].item())
        realized_return = float(data.action_returns[index + 1, action].item())
        switch_cost = float(switch_cost_bps) / 10_000.0 if action != previous_action else 0.0
        net_return = realized_return - switch_cost
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        daily_returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        if action != previous_action:
            switches += 1

        if capture_rollout:
            rollout_records.append(
                {
                    "date": data.dates[index + 1],
                    "action": action,
                    "strategy": data.action_names[action],
                    "previous_action": previous_action,
                    "segment_reset": int(segment_reset),
                    "daily_return": round(net_return, 8),
                    "equity": round(equity, 8),
                }
            )
        previous_action = action
        previous_index = index

    return StrategyEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        total_switches=switches,
        max_drawdown=fractional_max_drawdown(equity_curve),
        daily_sharpe=annualized_sharpe(daily_returns),
        rollout_records=rollout_records,
    )


def train_strategy_dqn_agent(
    train_data: StrategyDataSplit,
    val_data: StrategyDataSplit,
    *,
    device: torch.device,
    config: StrategyTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    train_data = train_data if train_data.features.device == device else train_data.to(device)
    val_data = val_data if val_data.features.device == device else val_data.to(device)
    assert_matching_strategy_schema(train_data, val_data)

    action_count = len(train_data.action_names)
    if train_data.lookback != config.env.lookback:
        raise ValueError("Training config lookback must match the training split.")
    q_network = StrategyQNetwork(
        feature_dim=train_data.features.shape[1],
        lookback=config.env.lookback,
        action_count=action_count,
        hidden_size=config.hidden_size,
        action_embedding_dim=config.action_embedding_dim,
    ).to(device)
    target_network = deepcopy(q_network).to(device)
    target_network.eval()

    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp, config.learning.amp_dtype)
    amp_enabled = scaler.is_enabled()
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
            "terminated": torch.float32,
        },
    )
    env = VectorizedStrategyAllocationEnv(train_data, config.env, device)

    best_val_return = -float("inf")
    best_val_switches = 10**12
    best_state = deepcopy(q_network.state_dict())
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
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q_values = q_network(states, previous_actions)
            greedy_actions = torch.argmax(q_values, dim=1)
            random_actions = torch.randint(0, action_count, greedy_actions.shape, device=device)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        transition = env.step(actions)
        replay.add(**transition)
        reward_trace.append(float(transition["rewards"].mean().item()))
        # Reset on terminal OR truncation; the TD target bootstraps on `terminated` (true terminal only).
        env.reset(transition["resets"])

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            # Clamp next_indices for the window lookup: a true terminal can store an out-of-data next
            # row (its bootstrap is zeroed via `terminated`); no-op for in-range non-terminal rows.
            n_rows = int(train_data.features.shape[0])
            # min_index = lookback-1 so a clamped terminal dummy never builds a tail-wrapped window;
            # valid_index_mask rejects any in-range-but-invalid non-terminal next row defensively.
            safe_next_indices = safe_next_row_indices(
                batch["next_indices"],
                batch["terminated"],
                min_index=train_data.lookback - 1,
                max_index=n_rows - 1,
                valid_index_mask=train_data.valid_index_mask,
            )
            current_states = train_data.state_windows(batch["indices"])
            next_states = train_data.state_windows(safe_next_indices)

            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
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
                    # Shared truncation-aware fp32 target (core.dqn_td_target): bootstrap through
                    # truncations, zero only on true terminals, NaN-safe for terminal next_q.
                    target_q = dqn_td_target(batch["rewards"], config.learning.gamma, batch["terminated"], next_q)
                loss = F.smooth_l1_loss(chosen_q.float(), target_q)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
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

        if step % config.learning.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if step % config.learning.eval_interval == 0 or step == config.learning.train_steps:
            val_result = evaluate_strategy_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                switch_cost_bps=config.env.switch_cost_bps,
                capture_rollout=False,
            )
            avg_loss = sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1)
            avg_reward = sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1)
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_switches": val_result.total_switches,
                    "val_sharpe": val_result.daily_sharpe,
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
                best_state = deepcopy(q_network.state_dict())

    q_network.load_state_dict(best_state)
    return q_network, {
        "best_val_return": best_val_return,
        "best_val_switches": best_val_switches,
        "amp_enabled": amp_enabled,
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "eval_trace": eval_trace,
    }
