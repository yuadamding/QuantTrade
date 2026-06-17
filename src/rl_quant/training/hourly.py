"""Training layer: hourly DQN training loop + evaluation + configs (extracted from rl_quant.hourly_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.models.hourly import CausalTransformerQNetwork
from rl_quant.action_risk import (
    RISK_AWARE_POLICY_MODEL_VERSION,
    ActionMeta,
    ExposureConstraintConfig,
    action_is_inverse_tensor,
    action_is_leveraged_tensor,
    action_leverage_tensor,
    action_metadata_to_dicts,
    action_weight_tensor,
    apply_exposure_masks,
    build_action_metadata,
    group_ids_for_actions,
    make_exposure_features,
    stable_action_metadata_hash,
    stable_action_risk_config_hash,
    trade_notional,
)
from rl_quant.core import (
    CudaVramReservation,
    DQNLearningConfig,
    TensorDictReplayBuffer,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    dqn_td_target,
    epsilon_by_step,
    fractional_max_drawdown,
    make_grad_scaler,
    safe_next_row_indices,
)
from rl_quant.trading_constraints import (
    TradingConstraintConfig,
    apply_notional_aware_hysteresis,
    build_action_mask,
    make_constraint_features,
    sample_valid_actions,
    trade_legs,
)
from rl_quant.datasets.hourly import (
    HOURLY_CONSTRAINT_FEATURE_DIM,
    HOURLY_CONSTRAINT_FEATURE_NAMES,
    HourlyDataSplit,
    assert_matching_hourly_schema,
)
from rl_quant.envs.hourly import HourlyEnvConfig, VectorizedHourlyAllocationEnv


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


@dataclass
class HourlyEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    total_switches: int
    market_order_legs: float
    total_traded_notional: float
    max_drawdown: float
    hourly_sharpe: float | None
    rollout_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "total_switches": self.total_switches,
            "market_order_legs": self.market_order_legs,
            "total_traded_notional": self.total_traded_notional,
            "max_drawdown": self.max_drawdown,
            "hourly_sharpe": self.hourly_sharpe,
            "annualized_sharpe": self.hourly_sharpe,
            "rollout_records": self.rollout_records,
        }


def _constraint_features_for_model(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    expected_dim = getattr(model, "constraint_feature_dim", features.shape[1])
    if expected_dim == features.shape[1]:
        return features
    if expected_dim < features.shape[1]:
        return features[:, :expected_dim]
    padding = torch.zeros(
        (features.shape[0], int(expected_dim) - features.shape[1]),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, padding], dim=1)


@torch.no_grad()
def evaluate_hourly_policy(
    data: HourlyDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    switch_cost_bps: float = 1.0,
    constraints: TradingConstraintConfig | None = None,
    exposure_constraints: ExposureConstraintConfig | None = None,
    action_meta: list[ActionMeta] | None = None,
    episode_length: int | None = None,
    capture_rollout: bool = False,
) -> HourlyEvaluationResult:
    constraints = constraints or TradingConstraintConfig(one_way_cost_bps=switch_cost_bps)
    exposure_constraints = exposure_constraints or ExposureConstraintConfig()
    data = data if data.features.device == device else data.to(device)
    action_meta = action_meta or build_action_metadata(data.action_names)
    action_weights = action_weight_tensor(
        action_meta,
        device=device,
        max_effective_leverage=exposure_constraints.max_effective_leverage,
    )
    action_leverage = action_leverage_tensor(action_meta, device=device)
    action_is_leveraged = action_is_leveraged_tensor(action_meta, device=device)
    action_is_inverse = action_is_inverse_tensor(action_meta, device=device)
    action_group_ids, action_groups = group_ids_for_actions(action_meta, device=device)
    group_counts_today = torch.zeros((1, len(action_groups)), dtype=torch.long, device=device)
    model.eval()
    previous_action = int(initial_action)
    bars_held = int(constraints.min_hold_bars)
    cooldown_remaining = 0
    switches_today = 0
    switches_episode = 0
    order_legs_today = 0.0
    order_legs_episode = 0.0
    steps_today = 0
    leveraged_bars_today = 0
    consecutive_leveraged_bars = 0
    equity = 1.0
    equity_curve = [equity]
    bar_returns: list[float] = []
    total_reward_bps = 0.0
    switches = 0
    order_legs = 0.0
    traded_notional_total = 0.0
    records: list[dict[str, Any]] = []
    previous_index: int | None = None
    previous_date: str | None = None
    episode_steps = 0
    for index in data.valid_start_indices.detach().cpu().tolist():
        current_date = data.session_dates[index] if data.session_dates is not None else data.timestamps[index][:10]
        segment_reset = previous_index is None or index != previous_index + 1
        if segment_reset:
            previous_action = int(initial_action)
            bars_held = int(constraints.min_hold_bars)
            cooldown_remaining = 0
            switches_today = 0
            switches_episode = 0
            order_legs_today = 0.0
            order_legs_episode = 0.0
            steps_today = 0
            leveraged_bars_today = 0
            consecutive_leveraged_bars = 0
            group_counts_today.zero_()
            episode_steps = 0
        elif previous_date is not None and current_date != previous_date:
            switches_today = 0
            order_legs_today = 0.0
            steps_today = 0
            leveraged_bars_today = 0
            consecutive_leveraged_bars = 0
            group_counts_today.zero_()
        if episode_length is not None and episode_steps >= int(episode_length):
            switches_episode = 0
            order_legs_episode = 0.0
            episode_steps = 0
        index_tensor = torch.tensor([index], dtype=torch.long, device=device)
        previous_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        base_constraint_features = make_constraint_features(
            bars_held=torch.tensor([bars_held], dtype=torch.long, device=device),
            cooldown_remaining=torch.tensor([cooldown_remaining], dtype=torch.long, device=device),
            switches_today=torch.tensor([switches_today], dtype=torch.long, device=device),
            switches_episode=torch.tensor([switches_episode], dtype=torch.long, device=device),
            constraints=constraints,
            episode_length=int(episode_length or max(int(data.valid_start_indices.numel()), 1)),
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
        )
        exposure_features = make_exposure_features(
            current_action=previous_tensor,
            action_leverage=action_leverage,
            action_weights=action_weights,
            action_is_leveraged=action_is_leveraged,
            action_group_ids=action_group_ids,
            group_counts_today=group_counts_today,
            steps_today=torch.tensor([steps_today], dtype=torch.long, device=device),
            leveraged_bars_today=torch.tensor([leveraged_bars_today], dtype=torch.long, device=device),
            consecutive_leveraged_bars=torch.tensor([consecutive_leveraged_bars], dtype=torch.long, device=device),
            constraints=exposure_constraints,
            episode_length=int(episode_length or max(int(data.valid_start_indices.numel()), 1)),
        )
        constraint_features = torch.cat([base_constraint_features, exposure_features], dim=1)
        constraint_mask = build_action_mask(
            current_action=previous_tensor,
            bars_held=torch.tensor([bars_held], dtype=torch.long, device=device),
            cooldown_remaining=torch.tensor([cooldown_remaining], dtype=torch.long, device=device),
            switches_today=torch.tensor([switches_today], dtype=torch.long, device=device),
            max_switches_per_day=constraints.max_switches_per_day,
            min_hold_bars=constraints.min_hold_bars,
            action_count=len(data.action_names),
            switches_episode=torch.tensor([switches_episode], dtype=torch.long, device=device),
            max_switches_per_episode=constraints.max_switches_per_episode,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            max_order_legs_per_day=constraints.max_order_legs_per_day,
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
            max_order_legs_per_episode=constraints.max_order_legs_per_episode,
            cash_index=constraints.cash_index,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = data.valid_actions(index_tensor)
        availability_mask[:, int(constraints.cash_index)] = True
        pre_exposure_mask = constraint_mask & availability_mask
        action_mask = apply_exposure_masks(
            pre_exposure_mask,
            current_action=previous_tensor,
            action_leverage=action_leverage,
            action_weights=action_weights,
            action_is_leveraged=action_is_leveraged,
            action_is_inverse=action_is_inverse,
            action_group_ids=action_group_ids,
            group_counts_today=group_counts_today,
            steps_today=torch.tensor([steps_today], dtype=torch.long, device=device),
            leveraged_bars_today=torch.tensor([leveraged_bars_today], dtype=torch.long, device=device),
            consecutive_leveraged_bars=torch.tensor([consecutive_leveraged_bars], dtype=torch.long, device=device),
            constraints=exposure_constraints,
            cash_index=constraints.cash_index,
        )
        if not bool(action_mask.any().item()):
            action_mask[:, int(constraints.cash_index)] = True
        model_constraint_features = _constraint_features_for_model(model, constraint_features)
        q_values = model(data.state_windows(index_tensor), previous_tensor, model_constraint_features)
        action = int(
            apply_notional_aware_hysteresis(
                q_values,
                previous_tensor,
                action_mask,
                action_weights=action_weights,
                one_way_cost_bps=constraints.one_way_cost_bps,
                extra_switch_penalty_bps=constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=constraints.q_switch_margin_bps,
                cash_index=constraints.cash_index,
            )[0].item()
        )
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        position_weight = float(action_weights[action].item())
        effective_leverage = float((action_weights[action] * action_leverage[action]).item())
        legs = float(
            trade_legs(
                previous_tensor,
                action_tensor,
                cash_index=constraints.cash_index,
                count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            )[0].item()
        )
        traded_notional = float(
            trade_notional(
                previous_tensor,
                action_tensor,
                action_weights,
                cash_index=constraints.cash_index,
            )[0].item()
        )
        raw_action_return = float(data.action_returns[index, action].item())
        gross_return = position_weight * raw_action_return
        is_switch = action != previous_action
        per_notional_cost_bps = float(constraints.one_way_cost_bps)
        per_notional_cost_bps += float(is_switch) * float(constraints.extra_switch_penalty_bps)
        cost_bps = traded_notional * per_notional_cost_bps
        net_return = gross_return - cost_bps / 10_000.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        bar_returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        if is_switch:
            switches += 1
        order_legs += legs
        traded_notional_total += traded_notional
        if capture_rollout:
            q_row = q_values[0].detach().cpu()
            final_mask = action_mask[0].detach().cpu()
            constraint_row = constraint_mask[0].detach().cpu()
            availability_row = availability_mask[0].detach().cpu()
            pre_exposure_row = pre_exposure_mask[0].detach().cpu()
            candidate_actions = torch.arange(len(data.action_names), dtype=torch.long, device=device)
            previous_candidates = torch.full_like(candidate_actions, previous_action)
            candidate_notional = trade_notional(
                previous_candidates,
                candidate_actions,
                action_weights,
                cash_index=constraints.cash_index,
            )
            candidate_is_switch = candidate_actions.ne(previous_action)
            candidate_cost_bps = candidate_notional * (
                float(constraints.one_way_cost_bps)
                + candidate_is_switch.float() * float(constraints.extra_switch_penalty_bps)
            )
            candidate_costs = candidate_cost_bps.detach().cpu()
            q_values_by_action: dict[str, float] = {}
            candidates: dict[str, dict[str, Any]] = {}
            mask_reasons: dict[str, str] = {}
            for action_id, action_name in enumerate(data.action_names):
                valid_candidate = bool(final_mask[action_id].item())
                if valid_candidate:
                    reason = None
                elif not bool(availability_row[action_id].item()):
                    reason = "not_tradable_at_timestamp"
                elif not bool(constraint_row[action_id].item()):
                    reason = "trading_constraint"
                elif bool(pre_exposure_row[action_id].item()):
                    reason = "exposure_constraint"
                else:
                    reason = "masked"
                q_value = round(float(q_row[action_id].item()), 8)
                q_values_by_action[action_name] = q_value
                candidates[action_name] = {
                    "valid": valid_candidate,
                    "q_value": q_value,
                    "expected_cost_bps": round(float(candidate_costs[action_id].item()), 8),
                    "risk_bucket": action_meta[action_id].asset_class,
                    "reason": reason,
                }
                if reason is not None:
                    mask_reasons[action_name] = reason
            records.append(
                {
                    "decision_id": f"{data.name}:{data.timestamps[index]}",
                    "timestamp": data.timestamps[index],
                    "decision_ts": data.timestamps[index],
                    "next_timestamp": data.next_timestamps[index],
                    "model_id": f"{data.bar_interval}_causal_transformer_v{RISK_AWARE_POLICY_MODEL_VERSION}",
                    "action": action,
                    "asset": data.action_names[action],
                    "selected_action": data.action_names[action],
                    "previous_action": previous_action,
                    "previous_asset": data.action_names[previous_action],
                    "segment_reset": int(segment_reset),
                    "market_order_legs": legs,
                    "traded_notional": round(traded_notional, 8),
                    "position_weight": round(position_weight, 8),
                    "effective_leverage": round(effective_leverage, 8),
                    "raw_action_return": round(raw_action_return, 8),
                    "gross_return": round(gross_return, 8),
                    "cost_bps": round(cost_bps, 8),
                    "bar_interval": data.bar_interval,
                    "bar_return": round(net_return, 8),
                    "net_return": round(net_return, 8),
                    "hourly_return": round(net_return, 8),
                    "equity": round(equity, 8),
                    "action_mask_reasons": mask_reasons,
                    "q_values": q_values_by_action,
                    "risk_checks": {
                        "has_valid_action": bool(final_mask.any().item()),
                        "selected_action_valid": bool(final_mask[action].item()),
                        "selected_action_available": bool(availability_row[action].item()),
                    },
                    "expected_cost_bps": round(cost_bps, 8),
                    "data_quality_score": 1.0,
                    "readiness_score": 1.0 if bool(final_mask[action].item()) else 0.0,
                    "readiness_config_hash": stable_action_risk_config_hash(exposure_constraints),
                    "candidates": candidates,
                }
            )
        if is_switch:
            bars_held = 1
            cooldown_remaining = int(constraints.cooldown_bars)
            switches_today += 1
            switches_episode += 1
        else:
            bars_held += 1
            cooldown_remaining = max(cooldown_remaining - 1, 0)
        order_legs_today += legs
        order_legs_episode += legs
        selected_group = int(action_group_ids[action].item())
        group_counts_today[0, selected_group] += 1
        selected_leveraged = bool(action_is_leveraged[action].item())
        steps_today += 1
        leveraged_bars_today += int(selected_leveraged)
        consecutive_leveraged_bars = consecutive_leveraged_bars + 1 if selected_leveraged else 0
        previous_action = action
        previous_index = index
        previous_date = current_date
        episode_steps += 1
    return HourlyEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        total_switches=switches,
        market_order_legs=order_legs,
        total_traded_notional=traded_notional_total,
        max_drawdown=fractional_max_drawdown(equity_curve),
        hourly_sharpe=annualized_sharpe(bar_returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
    )


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
    assert_matching_hourly_schema(train_data, val_data)
    action_count = len(train_data.action_names)
    action_meta = build_action_metadata(train_data.action_names)
    action_weights = action_weight_tensor(
        action_meta,
        device=device,
        max_effective_leverage=config.env.exposure_constraints.max_effective_leverage,
    )

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
        constraint_feature_dim=HOURLY_CONSTRAINT_FEATURE_DIM,
    ).to(device)
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp, config.learning.amp_dtype)
    replay = TensorDictReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields={
            "indices": ((), torch.long),
            "previous_actions": ((), torch.long),
            "constraint_features": ((HOURLY_CONSTRAINT_FEATURE_DIM,), torch.float32),
            "action_mask": ((action_count,), torch.bool),
            "actions": ((), torch.long),
            "rewards": ((), torch.float32),
            "next_indices": ((), torch.long),
            "next_previous_actions": ((), torch.long),
            "next_constraint_features": ((HOURLY_CONSTRAINT_FEATURE_DIM,), torch.float32),
            "next_action_mask": ((action_count,), torch.bool),
            "terminated": ((), torch.float32),
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
    valid_action_count_trace: list[float] = []
    eval_trace: list[dict[str, float | int | None | str]] = []

    for step in range(1, config.learning.train_steps + 1):
        states, previous_actions, constraint_features, action_mask = env.observe()
        valid_action_count_trace.append(float(action_mask.sum(dim=1).float().mean().item()))
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q_values = q_network(states, previous_actions, constraint_features)
            greedy_actions = apply_notional_aware_hysteresis(
                q_values,
                previous_actions,
                action_mask,
                action_weights=action_weights,
                one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                cash_index=config.env.constraints.cash_index,
                reward_scale=config.env.reward_scale,
            )
            random_actions = sample_valid_actions(action_mask)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        transition = env.step(actions)
        replay.add(**transition)
        reward_trace.append(float(transition["rewards"].mean().item()))
        env.reset(transition["resets"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            # Clamp next_indices for the state lookup: a true terminal can store an out-of-data next
            # row (its bootstrap is zeroed below); a no-op for non-terminal in-range transitions.
            n_rows = int(train_data.action_returns.shape[0])
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
                chosen_q = q_network(
                    current_states,
                    batch["previous_actions"],
                    batch["constraint_features"],
                ).gather(
                    1,
                    batch["actions"].unsqueeze(1),
                ).squeeze(1)
                with torch.no_grad():
                    next_online = q_network(
                        next_states,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    )
                    next_actions = apply_notional_aware_hysteresis(
                        next_online,
                        batch["next_previous_actions"],
                        batch["next_action_mask"],
                        action_weights=action_weights,
                        one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                        extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                        q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                        cash_index=config.env.constraints.cash_index,
                        reward_scale=config.env.reward_scale,
                    )
                    next_q = target_network(
                        next_states,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    ).gather(
                        1,
                        next_actions.unsqueeze(1),
                    ).squeeze(1)
                    # Bootstrap through episode-length truncations; zero only on true terminals.
                    # Shared float32 target with the minute-to-hour trainer (AMP-safe). Using the
                    # reset mask here would wrongly treat truncation as terminal.
                    target_q = dqn_td_target(batch["rewards"], config.learning.gamma, batch["terminated"], next_q)
                loss = F.smooth_l1_loss(chosen_q.float(), target_q)

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
                constraints=config.env.constraints,
                exposure_constraints=config.env.exposure_constraints,
                action_meta=action_meta,
                episode_length=config.env.episode_length,
            )
            # evaluate_hourly_policy() puts the shared q_network in eval() mode; restore train()
            # so the rest of training keeps dropout active (otherwise dropout is silently disabled
            # from the first eval onward).
            q_network.train()
            avg_loss = sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1)
            avg_reward = sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1)
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_switches": val_result.total_switches,
                    "val_order_legs": val_result.market_order_legs,
                    "val_sharpe": val_result.hourly_sharpe,
                    "average_loss": avg_loss,
                    "average_train_reward": avg_reward,
                    "average_valid_action_count": sum(valid_action_count_trace[-200:])
                    / max(len(valid_action_count_trace[-200:]), 1),
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
        "valid_action_count_trace": valid_action_count_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
        "model_version": RISK_AWARE_POLICY_MODEL_VERSION,
        "uses_constraint_features": True,
        "constraint_feature_names": HOURLY_CONSTRAINT_FEATURE_NAMES,
        "action_metadata": action_metadata_to_dicts(action_meta),
        "action_metadata_hash": stable_action_metadata_hash(action_meta),
        "exposure_constraints": asdict(config.env.exposure_constraints),
        "action_risk_config_hash": stable_action_risk_config_hash(config.env.exposure_constraints),
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
