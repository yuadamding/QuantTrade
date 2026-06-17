"""Envs layer: the hourly allocation environment -- state/transition/reward authority (extracted from rl_quant.hourly_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from rl_quant.action_risk import (
    ExposureConstraintConfig,
    action_is_inverse_tensor,
    action_is_leveraged_tensor,
    action_leverage_tensor,
    action_weight_tensor,
    apply_exposure_masks,
    build_action_metadata,
    group_ids_for_actions,
    make_exposure_features,
    trade_notional,
)
from rl_quant.trading_constraints import (
    TradingConstraintConfig,
    build_action_mask,
    make_constraint_features,
    trade_legs,
)
from rl_quant.datasets.hourly import HourlyDataSplit


@dataclass
class HourlyEnvConfig:
    lookback: int
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    switch_cost_bps: float = 1.0
    initial_action: int = 0
    constraints: TradingConstraintConfig = field(default_factory=TradingConstraintConfig)
    exposure_constraints: ExposureConstraintConfig = field(default_factory=ExposureConstraintConfig)

    def __post_init__(self) -> None:
        if self.switch_cost_bps != 1.0 and self.constraints.one_way_cost_bps == 1.0:
            self.constraints = replace(self.constraints, one_way_cost_bps=self.switch_cost_bps)


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
        self.bars_held = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.cooldown_remaining = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_episode = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.order_legs_today = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.order_legs_episode = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.action_meta = build_action_metadata(self.data.action_names)
        self.action_weights = action_weight_tensor(
            self.action_meta,
            device=device,
            max_effective_leverage=config.exposure_constraints.max_effective_leverage,
        )
        self.action_leverage = action_leverage_tensor(self.action_meta, device=device)
        self.action_is_leveraged = action_is_leveraged_tensor(self.action_meta, device=device)
        self.action_is_inverse = action_is_inverse_tensor(self.action_meta, device=device)
        self.action_group_ids, self.action_groups = group_ids_for_actions(self.action_meta, device=device)
        self.steps_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.leveraged_bars_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.consecutive_leveraged_bars = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.group_counts_today = torch.zeros(
            (config.num_envs, len(self.action_groups)),
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
        self.bars_held[mask] = int(self.config.constraints.min_hold_bars)
        self.cooldown_remaining[mask] = 0
        self.switches_today[mask] = 0
        self.switches_episode[mask] = 0
        self.order_legs_today[mask] = 0.0
        self.order_legs_episode[mask] = 0.0
        self.steps_today[mask] = 0
        self.leveraged_bars_today[mask] = 0
        self.consecutive_leveraged_bars[mask] = 0
        self.group_counts_today[mask] = 0
        self.steps[mask] = 0

    def _date_labels(self, indices: torch.Tensor) -> list[str]:
        if self.data.session_dates is not None:
            source = self.data.session_dates
        else:
            source = [timestamp[:10] for timestamp in self.data.timestamps]
        return [source[int(index.item())] for index in indices.detach().cpu()]

    def action_mask(self) -> torch.Tensor:
        constraint_mask = build_action_mask(
            current_action=self.previous_actions,
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            max_switches_per_day=self.config.constraints.max_switches_per_day,
            min_hold_bars=self.config.constraints.min_hold_bars,
            action_count=len(self.data.action_names),
            switches_episode=self.switches_episode,
            max_switches_per_episode=self.config.constraints.max_switches_per_episode,
            order_legs_today=self.order_legs_today,
            max_order_legs_per_day=self.config.constraints.max_order_legs_per_day,
            order_legs_episode=self.order_legs_episode,
            max_order_legs_per_episode=self.config.constraints.max_order_legs_per_episode,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = self.data.valid_actions(self.indices)
        availability_mask[:, int(self.config.constraints.cash_index)] = True
        mask = constraint_mask & availability_mask
        mask = apply_exposure_masks(
            mask,
            current_action=self.previous_actions,
            action_leverage=self.action_leverage,
            action_weights=self.action_weights,
            action_is_leveraged=self.action_is_leveraged,
            action_is_inverse=self.action_is_inverse,
            action_group_ids=self.action_group_ids,
            group_counts_today=self.group_counts_today,
            steps_today=self.steps_today,
            leveraged_bars_today=self.leveraged_bars_today,
            consecutive_leveraged_bars=self.consecutive_leveraged_bars,
            constraints=self.config.exposure_constraints,
            cash_index=self.config.constraints.cash_index,
        )
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, int(self.config.constraints.cash_index)] = True
        return mask

    def constraint_features(self) -> torch.Tensor:
        base = make_constraint_features(
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            switches_episode=self.switches_episode,
            constraints=self.config.constraints,
            episode_length=self.config.episode_length,
            order_legs_today=self.order_legs_today,
            order_legs_episode=self.order_legs_episode,
        )
        exposure = make_exposure_features(
            current_action=self.previous_actions,
            action_leverage=self.action_leverage,
            action_weights=self.action_weights,
            action_is_leveraged=self.action_is_leveraged,
            action_group_ids=self.action_group_ids,
            group_counts_today=self.group_counts_today,
            steps_today=self.steps_today,
            leveraged_bars_today=self.leveraged_bars_today,
            consecutive_leveraged_bars=self.consecutive_leveraged_bars,
            constraints=self.config.exposure_constraints,
            episode_length=self.config.episode_length,
        )
        return torch.cat([base, exposure], dim=1)

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data.state_windows(self.indices), self.previous_actions, self.constraint_features(), self.action_mask()

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        constraint_features = self.constraint_features()
        action_mask = self.action_mask()
        actions = actions.long()
        selected_valid = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        fallback_actions = torch.argmax(action_mask.long(), dim=1)
        actions = torch.where(selected_valid, actions, fallback_actions)
        raw_returns = self.data.action_returns[current_indices, actions]
        position_weights = self.action_weights[actions]
        gross_returns = position_weights * raw_returns
        legs = trade_legs(
            previous_actions,
            actions,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        traded_notional = trade_notional(
            previous_actions,
            actions,
            self.action_weights,
            cash_index=self.config.constraints.cash_index,
        )
        is_switch = actions != previous_actions
        per_notional_cost_bps = float(self.config.constraints.one_way_cost_bps)
        per_notional_cost_bps += is_switch.float() * float(self.config.constraints.extra_switch_penalty_bps)
        cost_bps = traded_notional * per_notional_cost_bps
        net_returns = gross_returns - cost_bps / 10_000.0
        rewards = net_returns * float(self.config.reward_scale)
        next_indices = current_indices + 1

        self.indices = next_indices
        self.previous_actions = actions.long()
        self.bars_held = torch.where(is_switch, torch.ones_like(self.bars_held), self.bars_held + 1)
        self.cooldown_remaining = torch.where(
            is_switch,
            torch.full_like(self.cooldown_remaining, int(self.config.constraints.cooldown_bars)),
            torch.clamp_min(self.cooldown_remaining - 1, 0),
        )
        self.switches_today = self.switches_today + is_switch.long()
        self.switches_episode = self.switches_episode + is_switch.long()
        self.order_legs_today = self.order_legs_today + legs
        self.order_legs_episode = self.order_legs_episode + legs
        selected_groups = self.action_group_ids[actions]
        self.group_counts_today.scatter_add_(
            1,
            selected_groups.unsqueeze(1),
            torch.ones((actions.shape[0], 1), dtype=torch.long, device=self.device),
        )
        selected_leveraged = self.action_is_leveraged[actions]
        self.steps_today = self.steps_today + 1
        self.leveraged_bars_today = self.leveraged_bars_today + selected_leveraged.long()
        self.consecutive_leveraged_bars = torch.where(
            selected_leveraged,
            self.consecutive_leveraged_bars + 1,
            torch.zeros_like(self.consecutive_leveraged_bars),
        )
        self.steps = self.steps + 1
        in_bounds = next_indices + 1 < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        # Separate a TRUE terminal (no valid next row) from an episode-length TRUNCATION (a rollout
        # boundary whose next row is a real continuation). `resets` ends the episode (terminal OR
        # truncation) and drives env reset; only `terminated` may zero the TD bootstrap (see
        # core.dqn_td_target). Naming matches strategy_dqn/intraday_dqn: terminated vs resets.
        terminated = ~next_valid
        truncated = self.steps >= int(self.config.episode_length)
        resets = terminated | truncated
        if bool(in_bounds.any().item()):
            old_dates = self._date_labels(current_indices[in_bounds])
            new_dates = self._date_labels(next_indices[in_bounds])
            reset_today = torch.tensor(
                [old != new for old, new in zip(old_dates, new_dates)],
                dtype=torch.bool,
                device=self.device,
            )
            valid_positions = torch.where(in_bounds)[0]
            reset_positions = valid_positions[reset_today]
            self.switches_today[reset_positions] = 0
            self.order_legs_today[reset_positions] = 0.0
            self.steps_today[reset_positions] = 0
            self.leveraged_bars_today[reset_positions] = 0
            self.consecutive_leveraged_bars[reset_positions] = 0
            self.group_counts_today[reset_positions] = 0
        next_action_mask = self.action_mask()
        return {
            "indices": current_indices,
            "previous_actions": previous_actions,
            "constraint_features": constraint_features,
            "action_mask": action_mask,
            "actions": actions,
            "rewards": rewards,
            "next_indices": next_indices,
            "next_previous_actions": self.previous_actions,
            "next_constraint_features": self.constraint_features(),
            "next_action_mask": next_action_mask,
            "resets": resets.float(),
            "terminated": terminated.float(),
            "legs": legs,
            "raw_action_returns": raw_returns,
            "position_weights": position_weights,
            "gross_returns": gross_returns,
            "traded_notional": traded_notional,
        }
