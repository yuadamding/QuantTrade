from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.models.intraday import ConvQNetwork
from rl_quant.execution import fill_index as compute_fill_index
from rl_quant.execution import fill_indices as compute_fill_indices
from rl_quant.execution import (
    require_nonnegative_int,
    require_positive_int,
    transition_pnl,
)
from rl_quant.intraday_data import MarketDataSplit
from rl_quant.core import (
    TensorReplayBuffer,
    absolute_max_drawdown,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    dqn_td_target,
    epsilon_by_step,
    make_grad_scaler,
    safe_next_row_indices,
)

ACTION_TO_POSITION = torch.tensor([-1, 0, 1], dtype=torch.long)


@dataclass
class TrainingConfig:
    lookback: int
    step_horizon: int
    latency_steps: int
    action_threshold: float
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
    trade_lot_size: int
    commission_per_share: float
    extra_cost_per_share: float
    grad_clip: float
    pretrain_epochs: int
    pretrain_batch_size: int
    use_amp: bool = False
    amp_dtype: str = "fp16"  # AMP autocast precision when use_amp: "fp16" (default) or "bf16".


def initialize_flat_policy(model: ConvQNetwork) -> None:
    with torch.no_grad():
        final_linear = model.network.head[-1]
        if isinstance(final_linear, nn.Linear):
            final_linear.weight.zero_()
            final_linear.bias.copy_(torch.tensor([0.0, 1.0, 0.0], device=final_linear.bias.device))


def _apply_action_threshold(q_values: torch.Tensor, current_positions: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0:
        return torch.argmax(q_values, dim=1)

    greedy_actions = torch.argmax(q_values, dim=1)
    greedy_values = q_values.gather(1, greedy_actions.unsqueeze(1)).squeeze(1)
    current_actions = (current_positions.long() + 1).clamp(0, q_values.shape[1] - 1)
    current_values = q_values.gather(1, current_actions.unsqueeze(1)).squeeze(1)
    should_switch = greedy_values > current_values + threshold
    return torch.where(should_switch, greedy_actions, current_actions)


class ReplayBuffer(TensorReplayBuffer):
    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            capacity=capacity,
            device=device,
            fields={
                "indices": torch.long,
                "positions": torch.long,
                "actions": torch.long,
                "rewards": torch.float32,
                "next_indices": torch.long,
                "next_positions": torch.long,
                "terminated": torch.float32,
            },
        )


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


@dataclass
class EvaluationResult:
    split_name: str
    total_pnl: float
    total_trades: int
    total_turnover: float
    win_rate: float
    average_step_reward: float
    max_drawdown: float
    daily_sharpe: float | None
    daily_summaries: list[dict[str, float | str | int]]
    rollout_records: list[dict[str, float | str | int]]
    action_threshold: float

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "total_turnover": self.total_turnover,
            "win_rate": self.win_rate,
            "average_step_reward": self.average_step_reward,
            "max_drawdown": self.max_drawdown,
            "daily_sharpe": self.daily_sharpe,
            "daily_summaries": self.daily_summaries,
            "action_threshold": self.action_threshold,
        }


def _max_drawdown(equity_curve: list[float]) -> float:
    return absolute_max_drawdown(equity_curve)


def _build_pretraining_targets(
    data: MarketDataSplit,
    *,
    device: torch.device,
    step_horizon: int,
    latency_steps: int,
    trade_lot_size: int,
    commission_per_share: float,
    extra_cost_per_share: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_indices = data.valid_start_indices
    day_end_for_candidate = data.day_ends[data.day_ids[candidate_indices]]
    valid_mask = candidate_indices + step_horizon < day_end_for_candidate
    indices = candidate_indices[valid_mask]
    next_indices = indices + step_horizon
    fill_indices = compute_fill_indices(indices, step_horizon=step_horizon, latency_steps=latency_steps)

    mid_now = data.close_mid[indices].view(-1, 1, 1)
    mid_fill = data.close_mid[fill_indices].view(-1, 1, 1)
    mid_next = data.close_mid[next_indices].view(-1, 1, 1)
    half_spread_fill = data.half_spread[fill_indices].view(-1, 1, 1)
    half_spread_next = data.half_spread[next_indices].view(-1, 1, 1)
    terminal_mask = (next_indices + step_horizon >= day_end_for_candidate[valid_mask]).view(-1, 1, 1)
    trade_scale = float(trade_lot_size * 100)

    action_positions = ACTION_TO_POSITION.to(device=device, dtype=torch.float32).view(1, 1, 3)
    current_positions = ACTION_TO_POSITION.to(device=device, dtype=torch.float32).view(1, 3, 1)
    # Same shared transition reward as the env step, broadcast over the [current, candidate] grid.
    target_q = transition_pnl(
        current_positions,
        action_positions,
        mid_now,
        mid_fill,
        mid_next,
        half_spread_fill,
        half_spread_next,
        terminal_mask,
        trade_scale=trade_scale,
        commission_per_share=commission_per_share,
        extra_cost_per_share=extra_cost_per_share,
    )

    expanded_indices = indices.unsqueeze(1).expand(-1, 3).reshape(-1)
    expanded_positions = ACTION_TO_POSITION.to(device=device).view(1, 3).expand(indices.shape[0], -1).reshape(-1)
    return expanded_indices, expanded_positions, target_q.reshape(-1, 3)


def _pretrain_q_network(
    q_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: MarketDataSplit,
    val_data: MarketDataSplit,
    *,
    device: torch.device,
    config: TrainingConfig,
    best_val_pnl: float,
    best_val_trades: int,
    best_state: dict[str, torch.Tensor],
    eval_trace: list[dict[str, float | int | None | str]],
    scaler: torch.amp.GradScaler,
) -> tuple[float, int, dict[str, torch.Tensor]]:
    if config.pretrain_epochs <= 0:
        return best_val_pnl, best_val_trades, best_state

    indices, positions, targets = _build_pretraining_targets(
        train_data,
        device=device,
        step_horizon=config.step_horizon,
        latency_steps=config.latency_steps,
        trade_lot_size=config.trade_lot_size,
        commission_per_share=config.commission_per_share,
        extra_cost_per_share=config.extra_cost_per_share,
    )
    sample_count = int(indices.shape[0])
    amp_enabled = scaler.is_enabled()
    for epoch in range(1, config.pretrain_epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        epoch_losses: list[float] = []
        q_network.train()

        for start in range(0, sample_count, config.pretrain_batch_size):
            batch_ids = permutation[start : start + config.pretrain_batch_size]
            batch_indices = indices[batch_ids]
            batch_positions = positions[batch_ids]
            batch_targets = targets[batch_ids]

            with autocast_context(device, config.use_amp, config.amp_dtype):
                predicted_q = q_network(train_data.state_windows(batch_indices), batch_positions)
                loss = F.smooth_l1_loss(predicted_q, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
                optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_result = evaluate_policy(
            val_data,
            q_network,
            device=device,
            step_horizon=config.step_horizon,
            latency_steps=config.latency_steps,
            trade_lot_size=config.trade_lot_size,
            commission_per_share=config.commission_per_share,
            extra_cost_per_share=config.extra_cost_per_share,
            capture_rollout=False,
            action_threshold=config.action_threshold,
        )
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        eval_trace.append(
            {
                "phase": "pretrain",
                "step": epoch,
                "epsilon": None,
                "val_pnl": val_result.total_pnl,
                "val_trades": val_result.total_trades,
                "average_loss": avg_loss,
                "average_train_reward": None,
            }
        )
        print(
            f"pretrain_epoch={epoch} "
            f"val_pnl={val_result.total_pnl:.2f} "
            f"val_trades={val_result.total_trades} "
            f"avg_loss={avg_loss:.6f}"
        )
        if (
            val_result.total_pnl > best_val_pnl
            or (
                abs(val_result.total_pnl - best_val_pnl) <= 1e-6
                and val_result.total_trades > best_val_trades
            )
        ):
            best_val_pnl = val_result.total_pnl
            best_val_trades = val_result.total_trades
            best_state = deepcopy(q_network.state_dict())

    return best_val_pnl, best_val_trades, best_state


def select_action_threshold(
    data: MarketDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    step_horizon: int,
    latency_steps: int,
    trade_lot_size: int,
    commission_per_share: float,
    extra_cost_per_share: float,
    candidate_thresholds: list[float],
) -> tuple[float, list[dict[str, float | int | None]]]:
    best_threshold = candidate_thresholds[0]
    best_pnl = -float("inf")
    best_trades = -1
    results: list[dict[str, float | int | None]] = []

    for threshold in candidate_thresholds:
        result = evaluate_policy(
            data,
            model,
            device=device,
            step_horizon=step_horizon,
            latency_steps=latency_steps,
            trade_lot_size=trade_lot_size,
            commission_per_share=commission_per_share,
            extra_cost_per_share=extra_cost_per_share,
            capture_rollout=False,
            action_threshold=threshold,
        )
        results.append(
            {
                "threshold": threshold,
                "val_pnl": result.total_pnl,
                "val_trades": result.total_trades,
                "val_drawdown": result.max_drawdown,
            }
        )
        if result.total_pnl > best_pnl or (
            abs(result.total_pnl - best_pnl) <= 1e-6 and result.total_trades > best_trades
        ):
            best_threshold = threshold
            best_pnl = result.total_pnl
            best_trades = result.total_trades

    return best_threshold, results


@torch.no_grad()
def evaluate_policy(
    data: MarketDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    step_horizon: int,
    latency_steps: int,
    trade_lot_size: int,
    commission_per_share: float,
    extra_cost_per_share: float,
    capture_rollout: bool,
    action_threshold: float = 0.0,
) -> EvaluationResult:
    action_to_position = ACTION_TO_POSITION.to(device)
    trade_scale = float(trade_lot_size * 100)
    model.eval()

    cumulative_pnl = 0.0
    total_trades = 0
    total_turnover = 0.0
    step_rewards: list[float] = []
    daily_summaries: list[dict[str, float | str | int]] = []
    rollout_records: list[dict[str, float | str | int]] = []
    equity_curve: list[float] = []

    for day_id, date in enumerate(data.dates):
        day_start = int(data.day_starts[day_id].item())
        day_end = int(data.day_ends[day_id].item())
        start_index = day_start + data.lookback - 1
        if start_index >= day_end - step_horizon:
            continue

        position = torch.zeros(1, dtype=torch.long, device=device)
        day_pnl = 0.0
        day_turnover = 0.0
        day_trades = 0
        day_wins = 0
        day_steps = 0

        for index in range(start_index, day_end - step_horizon, step_horizon):
            index_tensor = torch.tensor([index], dtype=torch.long, device=device)
            state = data.state_windows(index_tensor)
            q_values = model(state, position)
            action = int(_apply_action_threshold(q_values, position, action_threshold)[0].item())
            new_position = int(action_to_position[action].item())
            old_position = int(position.item())

            mid_now = float(data.close_mid[index].item())
            next_index = index + step_horizon
            fill_index = compute_fill_index(index, step_horizon=step_horizon, latency_steps=latency_steps)
            mid_fill = float(data.close_mid[fill_index].item())
            mid_next = float(data.close_mid[next_index].item())
            half_spread_fill = float(data.half_spread[fill_index].item())
            half_spread_next = float(data.half_spread[next_index].item())
            turnover_units = abs(new_position - old_position)
            next_is_terminal = next_index + step_horizon >= day_end
            # Same shared transition reward as the env step / pretraining targets (see execution.py).
            reward = transition_pnl(
                old_position,
                new_position,
                mid_now,
                mid_fill,
                mid_next,
                half_spread_fill,
                half_spread_next,
                next_is_terminal,
                trade_scale=trade_scale,
                commission_per_share=commission_per_share,
                extra_cost_per_share=extra_cost_per_share,
            )

            day_steps += 1
            day_pnl += reward
            cumulative_pnl += reward
            day_turnover += turnover_units
            total_turnover += turnover_units
            step_rewards.append(reward)
            equity_curve.append(cumulative_pnl)
            if turnover_units > 0:
                day_trades += 1
                total_trades += 1
            if reward > 0.0:
                day_wins += 1

            if capture_rollout:
                rollout_records.append(
                    {
                        "date": date,
                        "time": data.times[index],
                        "time_fill": data.times[fill_index],
                        "time_next": data.times[next_index],
                        "position_before": old_position,
                        "action": action,
                        "position_after": new_position,
                        "latency_steps": latency_steps,
                        "reward_dollars": round(reward, 4),
                        "mid_now": round(mid_now, 6),
                        "mid_fill": round(mid_fill, 6),
                        "mid_next": round(mid_next, 6),
                        "equity_dollars": round(cumulative_pnl, 4),
                    }
                )

            position.fill_(new_position)

        daily_summaries.append(
            {
                "date": date,
                "pnl_dollars": round(day_pnl, 4),
                "trades": day_trades,
                "turnover_units": day_turnover,
                "steps": day_steps,
                "win_rate": (day_wins / max(day_steps, 1)),
            }
        )

    daily_pnls = [float(item["pnl_dollars"]) for item in daily_summaries]
    daily_sharpe = annualized_sharpe(daily_pnls)

    winning_steps = sum(1 for reward in step_rewards if reward > 0.0)
    return EvaluationResult(
        split_name=data.name,
        total_pnl=float(cumulative_pnl),
        total_trades=total_trades,
        total_turnover=float(total_turnover),
        win_rate=(winning_steps / max(len(step_rewards), 1)),
        average_step_reward=(sum(step_rewards) / max(len(step_rewards), 1)),
        max_drawdown=_max_drawdown(equity_curve),
        daily_sharpe=daily_sharpe,
        daily_summaries=daily_summaries,
        rollout_records=rollout_records,
        action_threshold=action_threshold,
    )


def train_dqn_agent(
    train_data: MarketDataSplit,
    val_data: MarketDataSplit,
    *,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    if config.lookback != train_data.lookback:
        raise ValueError("TrainingConfig.lookback must match the training data lookback.")

    q_network = ConvQNetwork(train_data.features.shape[1], config.lookback).to(device)
    initialize_flat_policy(q_network)
    target_network = deepcopy(q_network).to(device)
    target_network.eval()

    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = make_grad_scaler(device, config.use_amp, config.amp_dtype)
    amp_enabled = scaler.is_enabled()
    replay = ReplayBuffer(config.replay_capacity, device)
    env = VectorizedMarketEnv(train_data, config, device)

    eval_trace: list[dict[str, float | int | None | str]] = []
    # Start at -inf so the best learned policy is always captured, even when every learned
    # policy underperforms the flat-CASH baseline (0.0 PnL); a 0.0 floor would silently return
    # the untrained CASH init and report it as "the trained agent". The CASH baseline is tracked
    # separately below so a net-negative-but-best run is not mistaken for beating cash.
    cash_baseline_val_pnl = 0.0
    best_val_pnl = float("-inf")
    best_val_trades = 0
    best_state = deepcopy(q_network.state_dict())

    best_val_pnl, best_val_trades, best_state = _pretrain_q_network(
        q_network,
        optimizer,
        train_data,
        val_data,
        device=device,
        config=config,
        best_val_pnl=best_val_pnl,
        best_val_trades=best_val_trades,
        best_state=best_state,
        eval_trace=eval_trace,
        scaler=scaler,
    )
    q_network.load_state_dict(best_state)
    target_network.load_state_dict(best_state)

    loss_trace: list[float] = []
    train_reward_trace: list[float] = []

    for step in range(1, config.train_steps + 1):
        states, positions = env.observe()
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.train_steps,
            start=config.epsilon_start,
            end=config.epsilon_end,
        )

        with torch.no_grad():
            with autocast_context(device, config.use_amp, config.amp_dtype):
                q_values = q_network(states, positions)
            greedy_actions = _apply_action_threshold(q_values, positions, config.action_threshold)
            random_actions = torch.randint(0, 3, greedy_actions.shape, device=device)
            explore_mask = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore_mask, random_actions, greedy_actions)

        transition = env.step(actions)
        replay.add(**transition)
        train_reward_trace.append(float(transition["rewards"].mean().item()))
        # Reset on terminal OR truncation; the TD target bootstraps on `terminated` (true terminal only).
        env.reset(transition["resets"])

        if replay.size >= max(config.warmup_steps, config.batch_size):
            batch = replay.sample(config.batch_size)
            # Clamp next_indices for the window lookup: a true terminal can store an out-of-data next
            # row (its bootstrap is zeroed via `terminated`); no-op for in-range non-terminal rows.
            n_rows = int(train_data.features.shape[0])
            # min_index = lookback-1 so a clamped terminal dummy never builds a tail-wrapped window;
            # valid_index_mask (the full per-day valid range) rejects any in-range-but-invalid
            # non-terminal next row. Every non-terminal next is provably mask-True (it lies in
            # [day_start+lookback, day_end-2]), so this rejects nothing legitimate and only surfaces bugs.
            safe_next_indices = safe_next_row_indices(
                batch["next_indices"],
                batch["terminated"],
                min_index=train_data.lookback - 1,
                max_index=n_rows - 1,
                valid_index_mask=train_data.valid_index_mask,
            )
            current_states = train_data.state_windows(batch["indices"])
            next_states = train_data.state_windows(safe_next_indices)

            with autocast_context(device, config.use_amp, config.amp_dtype):
                current_q = q_network(current_states, batch["positions"])
                chosen_q = current_q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double-DQN action selection via the ONLINE network, but using the SAME
                    # threshold hysteresis that the behavior and evaluation policies apply, so the
                    # bootstrap estimates the value of the policy that is actually executed/scored
                    # (not a plain-greedy policy that is never run).
                    next_online = q_network(next_states, batch["next_positions"])
                    next_actions = _apply_action_threshold(
                        next_online, batch["next_positions"], config.action_threshold
                    )
                    next_q = target_network(next_states, batch["next_positions"]).gather(
                        1,
                        next_actions.unsqueeze(1),
                    ).squeeze(1)
                    # Shared truncation-aware fp32 target (core.dqn_td_target): bootstrap through
                    # truncations, zero only on true terminals, NaN-safe for terminal next_q.
                    target_q = dqn_td_target(batch["rewards"], config.gamma, batch["terminated"], next_q)
                loss = F.smooth_l1_loss(chosen_q.float(), target_q)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
                optimizer.step()
            loss_trace.append(float(loss.item()))

        if step % config.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if step % config.eval_interval == 0 or step == config.train_steps:
            val_result = evaluate_policy(
                val_data,
                q_network,
                device=device,
                step_horizon=config.step_horizon,
                latency_steps=config.latency_steps,
                trade_lot_size=config.trade_lot_size,
                commission_per_share=config.commission_per_share,
                extra_cost_per_share=config.extra_cost_per_share,
                capture_rollout=False,
                action_threshold=config.action_threshold,
            )
            avg_loss = sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1)
            avg_reward = sum(train_reward_trace[-200:]) / max(len(train_reward_trace[-200:]), 1)
            eval_trace.append(
                {
                    "phase": "rl",
                    "step": step,
                    "epsilon": epsilon,
                    "val_pnl": val_result.total_pnl,
                    "val_trades": val_result.total_trades,
                    "average_loss": avg_loss,
                    "average_train_reward": avg_reward,
                }
            )
            print(
                f"step={step} "
                f"epsilon={epsilon:.4f} "
                f"val_pnl={val_result.total_pnl:.2f} "
                f"val_trades={val_result.total_trades} "
                f"avg_loss={avg_loss:.6f}"
            )
            if (
                val_result.total_pnl > best_val_pnl
                or (
                    abs(val_result.total_pnl - best_val_pnl) <= 1e-6
                    and val_result.total_trades > best_val_trades
                )
            ):
                best_val_pnl = val_result.total_pnl
                best_val_trades = val_result.total_trades
                best_state = deepcopy(q_network.state_dict())

    q_network.load_state_dict(best_state)
    return q_network, {
        "best_val_pnl": best_val_pnl,
        "cash_baseline_val_pnl": cash_baseline_val_pnl,
        "beats_cash_baseline_val": bool(best_val_pnl > cash_baseline_val_pnl),
        "amp_enabled": amp_enabled,
        "loss_trace": loss_trace,
        "train_reward_trace": train_reward_trace,
        "eval_trace": eval_trace,
    }
