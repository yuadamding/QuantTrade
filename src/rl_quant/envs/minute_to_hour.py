"""Envs layer: the hour-allocation environment over sub-hour context -- state/transition/reward authority (extracted from rl_quant.minute_to_hour_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rl_quant.trading_constraints import (
    TradingConstraintConfig,
    advance_position_excursion,
    build_action_mask,
    build_dynamic_transition_features,
    make_constraint_features,
    trade_legs,
)

from rl_quant.datasets.hour_from_subhour import (
    HourFromMinuteDataSplit,
    default_minute_to_hour_constraints,
)
from rl_quant.execution import WeightExecutionCostConfig, weight_transition_cost_bps
from rl_quant.features.action_risk import action_weight_tensor, build_action_metadata


@dataclass
class MinuteToHourEnvConfig:
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    initial_action: int = 0
    cash_idle_penalty_bps: float = 0.0
    # PR-3 shadow mode (default off, byte-identical): when on, the env ALSO computes the leg-engine weight-bps
    # execution reward/cost per transition and logs it + deltas in the step dict; training still uses the
    # legacy `rewards`. A cost-model A/B (turnover-weighted vs leg-count), NOT real-executable (no NBBO).
    execution_env_reward_shadow: bool = False
    constraints: TradingConstraintConfig = field(default_factory=default_minute_to_hour_constraints)


def transition_trade_cost_bps(
    previous_actions: torch.Tensor,
    actions: torch.Tensor,
    *,
    constraints: TradingConstraintConfig,
    cash_idle_penalty_bps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The minute->hour transition cost, SHARED by the env's reward and the evaluation rollout so the two can
    never drift (the architecture rule: only env/execution defines reward). Returns
    ``(legs, trade_cost_bps, cash_idle_bps)`` where ``trade_cost_bps = legs*one_way_cost_bps +
    switch*extra_switch_penalty_bps`` and ``cash_idle_bps`` is charged when the executed action is cash. The
    net return is ``raw_return - (trade_cost_bps + cash_idle_bps)/1e4`` and the env reward is
    ``reward_scale * net_return``. Tensor-shaped, so it serves the vectorized env and a 1-element eval step."""
    legs = trade_legs(
        previous_actions,
        actions,
        cash_index=constraints.cash_index,
        count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
    )
    is_switch = (actions != previous_actions).float()
    trade_cost_bps = legs * float(constraints.one_way_cost_bps) + is_switch * float(constraints.extra_switch_penalty_bps)
    cash_idle_bps = (actions == int(constraints.cash_index)).float() * float(cash_idle_penalty_bps)
    return legs, trade_cost_bps, cash_idle_bps


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
        # PR-3 shadow mode: per-action STATIC weights from action metadata (cash zeroed in step), so the prior
        # held weight is determined by previous_action -- no separate shadow-holdings state needed. The shadow
        # fee rate is the env's one_way_cost_bps, so the A/B isolates the costing METHOD, not the rate.
        self.execution_env_reward_shadow = bool(config.execution_env_reward_shadow)
        if self.execution_env_reward_shadow:
            self._shadow_action_weights = action_weight_tensor(
                build_action_metadata(list(self.data.action_names)), device=device
            )
            self._shadow_weight_cost = WeightExecutionCostConfig(fee_bps=float(config.constraints.one_way_cost_bps))
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
        is_switch = actions != previous_actions
        legs, cost_bps, cash_idle_penalty_bps = transition_trade_cost_bps(
            previous_actions,
            actions,
            constraints=self.config.constraints,
            cash_idle_penalty_bps=self.config.cash_idle_penalty_bps,
        )
        rewards = raw_returns * float(self.config.reward_scale) - (
            cost_bps + cash_idle_penalty_bps
        ) * float(self.config.reward_scale) / 10_000.0
        # PR-3 shadow (computed from the SAME state, only logged -- `rewards` above, used for training, is
        # untouched, so this is byte-identical to shadow-off). Weight-bps cost of the transition's two legs
        # (sell prior weight + buy new weight; cash = no exposure; only a real switch trades), carrying the same
        # cash-idle term as the legacy reward so reward_delta isolates the trade-cost-MODEL change.
        if self.execution_env_reward_shadow:
            cash_idx = int(self.config.constraints.cash_index)
            w_prev = self._shadow_action_weights[previous_actions]
            w_next = self._shadow_action_weights[actions]
            zeros = torch.zeros_like(w_prev)
            sell_weight = torch.where(is_switch & (previous_actions != cash_idx), w_prev, zeros)
            buy_weight = torch.where(is_switch & (actions != cash_idx), w_next, zeros)
            execution_cost_bps_shadow = weight_transition_cost_bps(
                sell_weight, buy_weight, weight_cost=self._shadow_weight_cost
            )
            execution_env_reward_shadow = raw_returns * float(self.config.reward_scale) - (
                execution_cost_bps_shadow + cash_idle_penalty_bps
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
        self.entry_index = torch.where(is_switch, current_indices, self.entry_index)
        self.unrealized_pnl, self.mae, self.mfe = advance_position_excursion(
            self.unrealized_pnl, self.mae, self.mfe, raw_returns, held=held
        )
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
        out: dict[str, torch.Tensor] = {
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
        # PR-3 shadow side-channel: replay stores only its declared fields, so these extra keys never reach
        # training -- they are for logging / the shadow A/B only. cost_delta vs the legacy TRADE cost (cost_bps).
        if self.execution_env_reward_shadow:
            out["execution_env_reward_shadow"] = execution_env_reward_shadow
            out["execution_cost_bps_shadow"] = execution_cost_bps_shadow
            out["reward_delta_shadow"] = execution_env_reward_shadow - rewards
            out["cost_delta_shadow"] = execution_cost_bps_shadow - cost_bps
        return out
