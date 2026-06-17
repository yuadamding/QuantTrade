"""Envs layer: the hour-allocation environment over sub-hour context -- state/transition/reward authority (extracted from rl_quant.minute_to_hour_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rl_quant.trading_constraints import (
    TradingConstraintConfig,
    build_action_mask,
    build_dynamic_transition_features,
    make_constraint_features,
    trade_legs,
)

from rl_quant.datasets.hour_from_subhour import (
    HourFromMinuteDataSplit,
    default_minute_to_hour_constraints,
)


@dataclass
class MinuteToHourEnvConfig:
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    initial_action: int = 0
    cash_idle_penalty_bps: float = 0.0
    constraints: TradingConstraintConfig = field(default_factory=default_minute_to_hour_constraints)


class VectorizedMinuteToHourEnv:
    def __init__(self, data: HourFromMinuteDataSplit, config: MinuteToHourEnvConfig, device: torch.device) -> None:
        if not (0 <= config.initial_action < len(data.action_names)):
            raise ValueError("initial_action is outside the action space.")
        self.data = data if data.minute_features.device == device else data.to(device)
        self.config = config
        self.device = device
        self.start_indices = self._build_start_index_pool()
        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.previous_actions = torch.full((config.num_envs,), int(config.initial_action), dtype=torch.long, device=device)
        self.bars_held = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.cooldown_remaining = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_episode = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.order_legs_today = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.order_legs_episode = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.steps = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        # PR-D / D0 dynamic position bookkeeping: maintained as internal env state but NOT yet consumed by
        # the reward, the model forward, the replay buffer, or the step() dict -- so training is byte-identical
        # (see pr_d_dynamic_state_design.md). This is a RETURN-based env (no prices/target weights), so we track
        # the entry row, the compounded return since entry, and the max adverse/favorable excursion since entry.
        self.entry_index = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.unrealized_pnl = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.mae = torch.zeros(config.num_envs, dtype=torch.float32, device=device)  # max adverse excursion (<= 0)
        self.mfe = torch.zeros(config.num_envs, dtype=torch.float32, device=device)  # max favorable excursion (>= 0)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        starts = self.data.valid_start_indices
        starts = starts[starts + 1 < self.data.action_returns.shape[0]]
        if starts.numel() == 0:
            raise ValueError("No valid minute-to-hour start indices remain.")
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
        self.steps[mask] = 0
        # D0 dynamic bookkeeping resets with the episode (entry starts at the freshly-drawn start row).
        self.entry_index[mask] = self.indices[mask]
        self.unrealized_pnl[mask] = 0.0
        self.mae[mask] = 0.0
        self.mfe[mask] = 0.0

    def constraint_features(self) -> torch.Tensor:
        return make_constraint_features(
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            switches_episode=self.switches_episode,
            constraints=self.config.constraints,
            episode_length=self.config.episode_length,
            order_legs_today=self.order_legs_today,
            order_legs_episode=self.order_legs_episode,
        )

    def action_mask(self, row_indices: torch.Tensor | None = None) -> torch.Tensor:
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
        if row_indices is None:
            row_indices = self.indices
        if row_indices.shape != self.previous_actions.shape:
            raise ValueError(
                "row_indices must have the same shape as the vectorized environment state; "
                f"got {tuple(row_indices.shape)} and {tuple(self.previous_actions.shape)}."
            )
        row_count = int(self.data.action_returns.shape[0])
        in_bounds = (row_indices >= 0) & (row_indices < row_count)
        safe_indices = row_indices.clamp(0, max(row_count - 1, 0))
        availability_mask = self.data.valid_actions(safe_indices)
        if bool((~in_bounds).any().item()):
            availability_mask[~in_bounds] = False
        availability_mask[:, int(self.config.constraints.cash_index)] = True
        mask = constraint_mask & availability_mask
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, int(self.config.constraints.cash_index)] = True
        return mask

    def observe(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        minute, mask, hour = self.data.state(self.indices)
        return (
            minute,
            mask,
            hour,
            self.data.action_feature_state(self.indices),
            self.previous_actions,
            self.constraint_features(),
            self.action_mask(),
        )

    def dynamic_state(self) -> torch.Tensor:
        """Per-env [B, DYNAMIC_TRANSITION_FEATURE_DIM] dynamic position-state features (PR-D) of the position
        held entering the current decision. Fed to the Q-network only when use_dynamic_transition_features."""
        return build_dynamic_transition_features(
            unrealized_pnl=self.unrealized_pnl, mae=self.mae, mfe=self.mfe
        )

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        # PR-D: snapshot the dynamic state of the position held ENTERING this decision (before the update
        # below). Always computed and returned (the replay add() filters unknown keys, so it is harmless when
        # use_dynamic_transition_features is off); consumed only when the flag is on.
        position_dynamic = self.dynamic_state()
        constraint_features = self.constraint_features()
        action_mask = self.action_mask()
        actions = actions.long()
        selected_valid = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        fallback_actions = torch.argmax(action_mask.long(), dim=1)
        actions = torch.where(selected_valid, actions, fallback_actions)
        label_mask = self.data.label_valid_actions(current_indices)
        selected_label_valid = label_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        cash_actions = torch.full_like(actions, int(self.config.constraints.cash_index))
        actions = torch.where(selected_label_valid, actions, cash_actions)
        raw_returns = self.data.action_returns[current_indices, actions]
        legs = trade_legs(
            previous_actions,
            actions,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        is_switch = actions != previous_actions
        cost_bps = legs * float(self.config.constraints.one_way_cost_bps)
        cost_bps = cost_bps + is_switch.float() * float(self.config.constraints.extra_switch_penalty_bps)
        cash_idle_penalty_bps = (
            (actions == int(self.config.constraints.cash_index)).float() * float(self.config.cash_idle_penalty_bps)
        )
        rewards = raw_returns * float(self.config.reward_scale) - (
            cost_bps + cash_idle_penalty_bps
        ) * float(self.config.reward_scale) / 10_000.0

        next_indices = current_indices + 1
        self.indices = next_indices
        self.previous_actions = actions
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
        self.steps = self.steps + 1
        # D0 dynamic bookkeeping (computed from existing state only; not fed to reward/model/replay): on a HOLD
        # compound this step's return into the position held since entry and extend MAE/MFE; on a SWITCH the new
        # position starts fresh this step (entry row = the current decision row). Held across a day boundary is
        # still held, so unlike the daily switch/leg counters below, this state is NOT reset on a new day.
        held = ~is_switch
        cum = torch.where(held, (1.0 + self.unrealized_pnl) * (1.0 + raw_returns) - 1.0, raw_returns)
        zeros = torch.zeros_like(cum)
        self.entry_index = torch.where(is_switch, current_indices, self.entry_index)
        self.mae = torch.where(held, torch.minimum(self.mae, cum), torch.minimum(zeros, cum))
        self.mfe = torch.where(held, torch.maximum(self.mfe, cum), torch.maximum(zeros, cum))
        self.unrealized_pnl = cum
        next_position_dynamic = self.dynamic_state()  # PR-D: post-action dynamic state (enters the next bar)

        in_bounds = next_indices < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        # Distinguish a TRUE terminal (no valid next row -> nothing to bootstrap from) from a mere
        # episode-length TRUNCATION (a rollout boundary whose next row is a real continuation). DQN
        # must bootstrap through truncations; only `terminated` may zero the TD bootstrap. `resets`
        # ends the episode (terminal OR truncation) and drives env reset (matches strategy/intraday).
        terminated = ~next_valid
        truncated = self.steps >= int(self.config.episode_length)
        resets = terminated | truncated
        if bool(in_bounds.any().item()):
            old_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in current_indices[in_bounds].detach().cpu()]
            new_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in next_indices[in_bounds].detach().cpu()]
            reset_today = torch.tensor([old != new for old, new in zip(old_dates, new_dates)], dtype=torch.bool, device=self.device)
            valid_positions = torch.where(in_bounds)[0]
            self.switches_today[valid_positions[reset_today]] = 0
            self.order_legs_today[valid_positions[reset_today]] = 0.0

        next_constraint_features = self.constraint_features()
        next_action_mask = self.action_mask(next_indices)
        return {
            "indices": current_indices,
            "previous_actions": previous_actions,
            "constraint_features": constraint_features,
            "action_mask": action_mask,
            "actions": actions,
            "rewards": rewards,
            "next_indices": next_indices,
            "next_previous_actions": self.previous_actions,
            "next_constraint_features": next_constraint_features,
            "next_action_mask": next_action_mask,
            "resets": resets.float(),
            "terminated": terminated.float(),
            "legs": legs,
            "position_dynamic": position_dynamic,
            "next_position_dynamic": next_position_dynamic,
        }
