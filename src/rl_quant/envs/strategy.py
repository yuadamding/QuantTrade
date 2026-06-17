"""Envs layer: the vectorized strategy-allocation environment (state / transition / reward authority).

Extracted verbatim from rl_quant.strategy_dqn in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Per the architecture rule, only the env/execution layer changes portfolio state and computes reward. Re-exported via rl_quant.training.strategy and the rl_quant.strategy_dqn shim; behaviour is byte-identical."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.strategy_data import StrategyDataSplit


@dataclass
class StrategyEnvConfig:
    lookback: int
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    switch_cost_bps: float = 0.0
    initial_action: int = 0


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
