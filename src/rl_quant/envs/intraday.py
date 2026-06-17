"""Envs layer: the vectorized intraday signed-position environment (state / transition / reward authority).

Extracted verbatim from rl_quant.intraday_dqn in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Only the env/execution layer changes portfolio state and computes reward. Re-exported via rl_quant.training.intraday and the rl_quant.intraday_dqn shim; byte-identical."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rl_quant.execution import (
    fill_indices as compute_fill_indices,
    require_nonnegative_int,
    require_positive_int,
    transition_pnl,
)
from rl_quant.intraday_data import MarketDataSplit

if TYPE_CHECKING:  # avoid a runtime envs<->training import cycle; used only in the config annotation
    from rl_quant.training.intraday import TrainingConfig

ACTION_TO_POSITION = torch.tensor([-1, 0, 1], dtype=torch.long)


class VectorizedMarketEnv:
    def __init__(self, data: MarketDataSplit, config: TrainingConfig, device: torch.device) -> None:
        self.data = data
        self.config = config
        self.device = device
        self.action_to_position = ACTION_TO_POSITION.to(device)
        self.trade_scale = float(config.trade_lot_size * 100)
        self.extra_cost = float(config.extra_cost_per_share)
        self.commission = float(config.commission_per_share)
        # Validate integer-like instead of int()-truncating: a fractional step_horizon/latency_steps (e.g.
        # from a non-CLI caller) must fail closed, not silently truncate the holding/latency horizon.
        self.step_horizon = require_positive_int("step_horizon", config.step_horizon)
        self.latency_steps = require_nonnegative_int("latency_steps", config.latency_steps)
        self.start_indices = self._build_start_index_pool()

        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.positions = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.steps = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        candidates = self.data.valid_start_indices
        day_end_for_candidate = self.data.day_ends[self.data.day_ids[candidates]]
        keep_mask = candidates + self.step_horizon < day_end_for_candidate
        starts = candidates[keep_mask]
        if starts.numel() == 0:
            raise ValueError("No valid start indices remain after applying the step horizon.")
        return starts

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.config.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count == 0:
            return
        random_ids = torch.randint(0, self.start_indices.shape[0], (count,), device=self.device)
        self.indices[mask] = self.start_indices[random_ids]
        self.positions[mask] = 0
        self.steps[mask] = 0

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data.state_windows(self.indices), self.positions

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        action_positions = self.action_to_position[actions]
        current_indices = self.indices.clone()
        current_positions = self.positions.clone()
        next_indices = current_indices + self.step_horizon
        fill_indices = compute_fill_indices(
            current_indices,
            step_horizon=self.step_horizon,
            latency_steps=self.latency_steps,
        )

        mid_now = self.data.close_mid[current_indices]
        mid_fill = self.data.close_mid[fill_indices]
        mid_next = self.data.close_mid[next_indices]
        half_spread_fill = self.data.half_spread[fill_indices]
        half_spread_next = self.data.half_spread[next_indices]

        current_day = self.data.day_ids[current_indices]
        next_day = self.data.day_ids[next_indices]
        day_ends = self.data.day_ends[current_day]
        # `terminal` = a genuine end of the tradable MDP (day/segment boundary): the position
        # must be liquidated and the bootstrap must NOT continue. `truncated` = the artificial
        # episode-length cutoff: the position economically persists, so we bootstrap THROUGH it
        # (via the real next state stored in this transition) and charge no liquidation cost.
        # See Pardo et al., "Time Limits in Reinforcement Learning".
        terminal = (next_day != current_day) | (next_indices + self.step_horizon >= day_ends)
        truncated = self.steps + 1 >= self.config.episode_length
        resets = terminal | truncated
        # Shared old/new-position latency P&L + turnover/terminal-liquidation cost (see execution.py).
        reward = transition_pnl(
            current_positions,
            action_positions,
            mid_now,
            mid_fill,
            mid_next,
            half_spread_fill,
            half_spread_next,
            terminal,
            trade_scale=self.trade_scale,
            commission_per_share=self.commission,
            extra_cost_per_share=self.extra_cost,
        )

        self.indices = next_indices
        self.positions = action_positions
        self.steps = self.steps + 1

        return {
            "indices": current_indices,
            "positions": current_positions,
            "actions": actions,
            "rewards": reward,
            "next_indices": next_indices,
            "next_positions": action_positions,
            # `terminated` carries the BOOTSTRAP terminal (true terminal only) so the TD target
            # bootstraps through episode truncations; `resets` drives env episode resets.
            "terminated": terminal,
            "resets": resets,
        }
