"""Training layer: minute->hour DQN training loop + evaluation + checkpointing + recency weighting (extracted from rl_quant.minute_to_hour_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

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
    CONSTRAINED_POLICY_MODEL_VERSION,
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
    POSITION_AWARE_POLICY_MODEL_VERSION,
    TRANSITION_FEATURE_DIM,
    TRANSITION_FEATURE_NAMES,
    TRANSITION_FEATURE_SCHEMA_VERSION,
    TradingConstraintConfig,
    DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION,
    DYNAMIC_TRANSITION_FEATURE_DIM,
    DYNAMIC_TRANSITION_FEATURE_NAMES,
    DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION,
    advance_position_excursion,
    apply_leg_aware_hysteresis,
    build_action_mask,
    build_dynamic_transition_features,
    build_transition_feature_table,
    make_constraint_features,
    sample_valid_actions,
)

from rl_quant.models.minute_to_hour import (  # re-export: model moved to the models layer
    DEFAULT_MAX_SUBHOUR_TOKENS,
    MinuteToHourCausalTransformerQNetwork,
)
from rl_quant.datasets.hour_from_subhour import (
    HourFromMinuteDataSplit,
    _timestamp_to_epoch_ms,
    assert_matching_hour_from_minute_schema,
    default_minute_to_hour_constraints,
    minute_to_hour_missing_label_report,
)
from rl_quant.envs.minute_to_hour import (
    MinuteToHourEnvConfig,
    VectorizedMinuteToHourEnv,
    transition_trade_cost_bps,
)


@dataclass
class RecencyWeightConfig:
    """Recency-focus weighting of TRAINING transitions. ``mode='none'`` -> uniform (default).

    With ``mode='exponential'`` a training row with decision timestamp ``t`` gets weight
        ``min_weight + (1 - min_weight) * exp(-ln2 * age_days / half_life_days)``
    where ``age_days`` is measured relative to the VALIDATION start (never the test block), so
    older training rows are down-weighted toward the recent pre-validation regime but never fully
    ignored (weight stays >= ``min_weight``). The test split is never passed to the trainer, so
    recency weighting is structurally incapable of touching it.
    """

    mode: str = "none"
    half_life_days: float = 120.0
    min_weight: float = 0.05


@dataclass
class MinuteToHourTrainingConfig:
    env: MinuteToHourEnvConfig
    learning: DQNLearningConfig
    d_model: int = 256
    n_heads: int = 8
    minute_layers: int = 2
    hour_layers: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.05
    action_embedding_dim: int = 32
    target_vram_gb: float | None = None
    vram_safety_gb: float = 0.12
    warm_start_model: str | bytes | PathLike[str] | None = None
    resume_training_state: str | bytes | PathLike[str] | None = None
    checkpoint_training_state: str | bytes | PathLike[str] | None = None
    checkpoint_every_steps: int = 0
    max_subhour_tokens: int | None = DEFAULT_MAX_SUBHOUR_TOKENS
    recency: RecencyWeightConfig = field(default_factory=RecencyWeightConfig)
    # Opt-in position-aware transition features (default off -> no new model params, existing
    # checkpoints load unchanged). When True, the Q-network scores each candidate with the cost/risk of
    # moving from the held position to it (see build_transition_feature_table).
    use_transition_features: bool = False
    # Opt-in PR-D dynamic position-state features (default off -> byte-identical, existing checkpoints load
    # strict). When True, the Q-network also scores each candidate with the HELD position's realized-P&L
    # excursion (unrealized_pnl / MAE / MFE / drawdown / runup), threaded from the env through replay. This
    # MOVES training numbers when on, so it ships behind this flag and flips to default only after a
    # latest-period A/B (no default flip here).
    use_dynamic_transition_features: bool = False


@dataclass
class MinuteToHourEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    allocation_switches: int
    market_order_legs: float
    max_drawdown: float
    annualized_sharpe: float | None
    rollout_records: list[dict[str, float | str | int]]
    evaluation_reportable: bool = True
    reportability_errors: list[str] = field(default_factory=list)
    selectable_missing_label_count: int = 0
    rows_with_any_selectable_missing_label: int = 0
    requested_action_missing_label_count: int = 0
    executed_action_missing_label_count: int = 0
    policy_unscorable_rows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "after_cost_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "allocation_switches": self.allocation_switches,
            "market_order_legs": self.market_order_legs,
            "max_drawdown": self.max_drawdown,
            "annualized_sharpe": self.annualized_sharpe,
            "rollout_records": self.rollout_records,
            "evaluation_reportable": self.evaluation_reportable,
            "reportability_errors": self.reportability_errors,
            "selectable_missing_label_count": self.selectable_missing_label_count,
            "rows_with_any_selectable_missing_label": self.rows_with_any_selectable_missing_label,
            "requested_action_missing_label_count": self.requested_action_missing_label_count,
            "executed_action_missing_label_count": self.executed_action_missing_label_count,
            "policy_unscorable_rows": self.policy_unscorable_rows,
        }


@torch.no_grad()
def evaluate_minute_to_hour_policy(
    data: HourFromMinuteDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    constraints: TradingConstraintConfig | None = None,
    episode_length: int | None = None,
    reward_scale: float = 10_000.0,
    cash_idle_penalty_bps: float = 0.0,
    capture_rollout: bool = False,
) -> MinuteToHourEvaluationResult:
    constraints = constraints or default_minute_to_hour_constraints()
    constraint_episode_length = int(episode_length or max(int(data.valid_start_indices.numel()), 1))
    data = data if data.minute_features.device == device else data.to(device)
    model.eval()
    # PR-D: evaluate a dynamic-aware model with its dynamic features. The held-position excursion is tracked
    # continuously and reset only on a data-segment break (a walk-forward backtest has no artificial episode
    # truncations), built each step via the SAME recurrence the env uses (advance_position_excursion) so the
    # eval dynamic features cannot drift from training's. dyn_dim == 0 => the model is non-dynamic and the
    # forward guard requires dynamic_state to stay None (unchanged default behaviour).
    dyn_dim = int(getattr(model, "dynamic_feature_dim", 0))
    unrealized_pnl = 0.0
    position_mae = 0.0
    position_mfe = 0.0
    previous_action = int(initial_action)
    bars_held = int(constraints.min_hold_bars)
    cooldown_remaining = 0
    switches_today = 0
    switches_episode = 0
    order_legs_today = 0.0
    order_legs_episode = 0.0
    previous_index: int | None = None
    previous_date: str | None = None
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    total_reward_bps = 0.0
    allocation_switches = 0
    order_legs = 0.0
    records: list[dict[str, float | str | int]] = []
    evaluated_rows: list[int] = []
    requested_actions: list[int] = []
    executed_actions: list[int] = []
    episode_steps = 0
    for index in data.valid_start_indices.detach().cpu().tolist():
        current_date = data.decision_timestamps[index][:10]
        segment_reset = previous_index is None or index != previous_index + 1
        if segment_reset:
            previous_action = int(initial_action)
            bars_held = int(constraints.min_hold_bars)
            cooldown_remaining = 0
            switches_today = 0
            switches_episode = 0
            order_legs_today = 0.0
            order_legs_episode = 0.0
            episode_steps = 0
            unrealized_pnl = 0.0
            position_mae = 0.0
            position_mfe = 0.0
        elif previous_date is not None and current_date != previous_date:
            switches_today = 0
            order_legs_today = 0.0
        if episode_steps >= constraint_episode_length:
            switches_episode = 0
            order_legs_episode = 0.0
            episode_steps = 0
        minute, mask, hour = data.state(torch.tensor([index], dtype=torch.long, device=device))
        action_features = data.action_feature_state(torch.tensor([index], dtype=torch.long, device=device))
        prev_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        bars_tensor = torch.tensor([bars_held], dtype=torch.long, device=device)
        cooldown_tensor = torch.tensor([cooldown_remaining], dtype=torch.long, device=device)
        switches_today_tensor = torch.tensor([switches_today], dtype=torch.long, device=device)
        switches_episode_tensor = torch.tensor([switches_episode], dtype=torch.long, device=device)
        constraints_tensor = make_constraint_features(
            bars_held=bars_tensor,
            cooldown_remaining=cooldown_tensor,
            switches_today=switches_today_tensor,
            switches_episode=switches_episode_tensor,
            constraints=constraints,
            episode_length=constraint_episode_length,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
        )
        action_mask = build_action_mask(
            current_action=prev_tensor,
            bars_held=bars_tensor,
            cooldown_remaining=cooldown_tensor,
            switches_today=switches_today_tensor,
            max_switches_per_day=constraints.max_switches_per_day,
            min_hold_bars=constraints.min_hold_bars,
            action_count=len(data.action_names),
            switches_episode=switches_episode_tensor,
            max_switches_per_episode=constraints.max_switches_per_episode,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            max_order_legs_per_day=constraints.max_order_legs_per_day,
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
            max_order_legs_per_episode=constraints.max_order_legs_per_episode,
            cash_index=constraints.cash_index,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = data.valid_actions(torch.tensor([index], dtype=torch.long, device=device))
        availability_mask[:, int(constraints.cash_index)] = True
        action_mask = action_mask & availability_mask
        if not bool(action_mask.any().item()):
            action_mask[:, int(constraints.cash_index)] = True
        # Pass action_features / dynamic_state only when present so a minimal forward(5-arg) policy (e.g. a
        # test mock or a non-dynamic model) is still called exactly as before -- unchanged default behaviour.
        forward_kwargs: dict[str, torch.Tensor] = {}
        if action_features is not None:
            forward_kwargs["action_features"] = action_features
        if dyn_dim > 0:
            forward_kwargs["dynamic_state"] = build_dynamic_transition_features(
                unrealized_pnl=torch.tensor([unrealized_pnl], device=device),
                mae=torch.tensor([position_mae], device=device),
                mfe=torch.tensor([position_mfe], device=device),
            )
        q_values = model(minute, mask, hour, prev_tensor, constraints_tensor, **forward_kwargs)
        action = int(
            apply_leg_aware_hysteresis(
                q_values,
                prev_tensor,
                action_mask,
                one_way_cost_bps=constraints.one_way_cost_bps,
                extra_switch_penalty_bps=constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=constraints.q_switch_margin_bps,
                cash_index=constraints.cash_index,
                reward_scale=reward_scale,
                count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            )[0].item()
        )
        requested_action = action
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        label_mask = data.label_valid_actions(torch.tensor([index], dtype=torch.long, device=device))
        requested_label_missing = (
            action != int(constraints.cash_index)
            and (not bool(label_mask[0, action].item()) or not bool(torch.isfinite(data.action_returns[index, action]).item()))
        )
        if not bool(label_mask[0, action].item()) or not bool(torch.isfinite(data.action_returns[index, action]).item()):
            action = int(constraints.cash_index)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        evaluated_rows.append(int(index))
        requested_actions.append(int(requested_action))
        executed_actions.append(int(action))
        # Shared with the env (transition_trade_cost_bps) so the eval ledger cannot drift from the training
        # reward, and it now applies the env's cash_idle_penalty_bps (omitted before -> a latent drift for
        # nonzero-penalty runs). For the default penalty (0) this is byte-identical to the prior inline cost.
        cost = transition_trade_cost_bps(
            prev_tensor, action_tensor, constraints=constraints, cash_idle_penalty_bps=cash_idle_penalty_bps
        )
        legs = float(cost.legs[0].item())
        is_switch = action != previous_action
        cost_bps = float(cost.trade_cost_bps[0].item())  # leg cost + switch penalty (the legacy combined cost)
        gross_return = float(data.action_returns[index, action].item())
        net_return = gross_return - (cost_bps + float(cost.cash_idle_bps[0].item())) / 10_000.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        allocation_switches += int(is_switch)
        order_legs += legs
        if capture_rollout:
            records.append(
                {
                    "decision_timestamp": data.decision_timestamps[index],
                    "next_timestamp": data.next_timestamps[index],
                    "action": action,
                    "requested_action": requested_action,
                    "executed_action": action,
                    "asset": data.action_names[action],
                    "requested_asset": data.action_names[requested_action],
                    "executed_asset": data.action_names[action],
                    "previous_action": previous_action,
                    "segment_reset": int(segment_reset),
                    "fallback_due_to_missing_label": int(requested_label_missing),
                    "market_order_legs": legs,
                    "net_return": round(net_return, 8),
                    "equity": round(equity, 8),
                }
            )
        if dyn_dim > 0:
            # Advance the held-position excursion by this step's GROSS return (raw_returns in the env), via
            # the shared recurrence; reset-on-switch is encoded by held = not is_switch.
            next_upnl, next_mae, next_mfe = advance_position_excursion(
                torch.tensor([unrealized_pnl], device=device),
                torch.tensor([position_mae], device=device),
                torch.tensor([position_mfe], device=device),
                torch.tensor([gross_return], device=device),
                held=torch.tensor([not is_switch], device=device),
            )
            unrealized_pnl = float(next_upnl.item())
            position_mae = float(next_mae.item())
            position_mfe = float(next_mfe.item())
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
        previous_action = action
        previous_index = index
        previous_date = current_date
        episode_steps += 1

    report = minute_to_hour_missing_label_report(
        data,
        row_indices=evaluated_rows,
        requested_actions=requested_actions,
        executed_actions=executed_actions,
        cash_index=int(constraints.cash_index),
    )
    return MinuteToHourEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        allocation_switches=allocation_switches,
        market_order_legs=order_legs,
        max_drawdown=fractional_max_drawdown(equity_curve),
        annualized_sharpe=annualized_sharpe(returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
        evaluation_reportable=bool(report["evaluation_reportable"]),
        reportability_errors=list(report["reportability_errors"]),
        selectable_missing_label_count=int(report["selectable_missing_label_count"]),
        rows_with_any_selectable_missing_label=int(report["rows_with_any_selectable_missing_label"]),
        requested_action_missing_label_count=int(report["requested_action_missing_label_count"] or 0),
        executed_action_missing_label_count=int(report["executed_action_missing_label_count"] or 0),
        policy_unscorable_rows=int(report["policy_unscorable_rows"] or 0),
    )


class _ConstantActionModel(nn.Module):
    """A deterministic baseline 'policy': emits a Q-vector that, after the eval's cost-aware hysteresis,
    enters and holds a single target action. Used only by evaluate_minute_to_hour_baselines -- the Q value
    is large so it dominates the switch-cost margin (the target is entered when valid and then held)."""

    def __init__(self, action_count: int, target_action: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.target_action = int(target_action)

    def forward(self, minute, mask, hour, previous_actions, constraint_features, action_features=None, dynamic_state=None):
        del minute, mask, hour, constraint_features, action_features, dynamic_state
        q = torch.zeros(previous_actions.shape[0], self.action_count, device=previous_actions.device)
        q[:, self.target_action] = 1.0e6
        return q


def evaluate_minute_to_hour_baselines(
    data: HourFromMinuteDataSplit,
    *,
    device: torch.device,
    constraints: TradingConstraintConfig | None = None,
    episode_length: int | None = None,
    reward_scale: float = 10_000.0,
    cash_idle_penalty_bps: float = 0.0,
    include_buy_and_hold: bool = True,
) -> dict[str, MinuteToHourEvaluationResult]:
    """Deterministic reference policies run through the SAME eval path as a trained model (identical cost /
    constraint / reportability / drawdown / Sharpe accounting), so a policy -- or a PR-D flag-on-vs-off A/B
    -- can be benchmarked against cash and buy-and-hold. A model that does not beat these under cost should
    not be promoted (per the review). NOTE: the action space is single-slot/discrete, so an equal-weight
    baseline is not expressible here; the references are always-cash and per-action buy-and-hold. This is an
    EVALUATION-ONLY helper: it changes no training/reward path."""
    constraints = constraints or default_minute_to_hour_constraints()
    cash_index = int(constraints.cash_index)
    action_count = len(data.action_names)
    common = dict(
        device=device, constraints=constraints, episode_length=episode_length,
        reward_scale=reward_scale, cash_idle_penalty_bps=cash_idle_penalty_bps, initial_action=cash_index,
    )
    results: dict[str, MinuteToHourEvaluationResult] = {
        "always_cash": evaluate_minute_to_hour_policy(data, _ConstantActionModel(action_count, cash_index), **common),
    }
    if include_buy_and_hold:
        for action in range(action_count):
            if action == cash_index:
                continue
            results[f"buy_and_hold:{data.action_names[action]}"] = evaluate_minute_to_hour_policy(
                data, _ConstantActionModel(action_count, action), **common
            )
    return results


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _tensor_dict_to_cpu(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return move(optimizer.state_dict())


def _replay_state_to_cpu(replay: TensorDictReplayBuffer) -> dict[str, object]:
    return {
        "capacity": int(replay.capacity),
        "size": int(replay.size),
        "cursor": int(replay.cursor),
        "storage": _tensor_dict_to_cpu(replay.storage),
    }


def _load_replay_state(replay: TensorDictReplayBuffer, state: dict[str, object], device: torch.device) -> None:
    if int(state.get("capacity", -1)) != int(replay.capacity):
        raise ValueError("Resume checkpoint replay capacity does not match the current training config.")
    storage = state.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("Resume checkpoint is missing replay storage.")
    if set(storage) != set(replay.storage):
        raise ValueError("Resume checkpoint replay fields do not match the current training config.")
    for key, target in replay.storage.items():
        value = storage[key]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Resume checkpoint replay field {key!r} has an incompatible shape.")
        target.copy_(value.to(device=device, dtype=target.dtype))
    replay.size = int(state.get("size", 0))
    replay.cursor = int(state.get("cursor", 0))


_LEGACY_ENV_STATE_KEYS = (
    "indices", "previous_actions", "bars_held", "cooldown_remaining",
    "switches_today", "switches_episode", "order_legs_today", "order_legs_episode", "steps",
)
# PR-D dynamic position-state bookkeeping. The env tracks these every step regardless of the flag, but they
# are only CONSUMED (fed to the model / replay) when use_dynamic_transition_features is on -- so they are
# required in a resume checkpoint only for a dynamic run. We always save them (cheap, correct); we require
# them on load only when the dynamic flag is on.
_DYNAMIC_ENV_STATE_KEYS = ("entry_index", "unrealized_pnl", "mae", "mfe")


def _env_state_to_cpu(env: VectorizedMinuteToHourEnv) -> dict[str, torch.Tensor]:
    return {
        key: getattr(env, key).detach().cpu().clone()
        for key in (*_LEGACY_ENV_STATE_KEYS, *_DYNAMIC_ENV_STATE_KEYS)
    }


def _load_env_state(
    env: VectorizedMinuteToHourEnv,
    state: dict[str, torch.Tensor],
    device: torch.device,
    *,
    require_dynamic: bool = False,
) -> None:
    keys = list(_LEGACY_ENV_STATE_KEYS)
    if require_dynamic:
        missing = [k for k in _DYNAMIC_ENV_STATE_KEYS if k not in state]
        if missing:
            raise ValueError(
                "Resume checkpoint is missing dynamic env state required by "
                f"use_dynamic_transition_features=True: {missing}. Resuming a dynamic run without restored "
                "position state would corrupt the in-flight episodes' dynamic features (entry/MAE/MFE/"
                "unrealized P&L) while replay still holds dynamic-aware samples -- re-train or migrate the "
                "checkpoint instead of silently resetting them."
            )
        keys += list(_DYNAMIC_ENV_STATE_KEYS)
    else:
        # Restore dynamic bookkeeping if a checkpoint carries it (harmless when the flag is off -- the fields
        # are tracked but unconsumed), while staying tolerant of older checkpoints that predate the fields.
        keys += [k for k in _DYNAMIC_ENV_STATE_KEYS if k in state]
    for key in keys:
        value = state.get(key)
        target = getattr(env, key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Resume checkpoint env field {key!r} has an incompatible shape.")
        target.copy_(value.to(device=device, dtype=target.dtype))


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object], device: torch.device) -> None:
    torch_rng_state = state.get("torch_rng_state")
    if torch.is_tensor(torch_rng_state):
        torch.set_rng_state(torch_rng_state.cpu())
    cuda_state = state.get("cuda_rng_state_all")
    if device.type == "cuda" and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all([item.cpu() if torch.is_tensor(item) else item for item in cuda_state])


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _save_minute_to_hour_training_state(
    path: Path,
    *,
    step: int,
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: TensorDictReplayBuffer,
    env: VectorizedMinuteToHourEnv,
    best_val_return: float,
    best_val_legs: float,
    best_state: dict[str, torch.Tensor],
    loss_trace: list[float],
    reward_trace: list[float],
    valid_action_count_trace: list[float],
    eval_trace: list[dict[str, float | int | None | str]],
    device: torch.device,
) -> None:
    _atomic_torch_save(
        {
            "checkpoint_kind": "minute_to_hour_dqn_training_state",
            "checkpoint_version": 1,
            "step": int(step),
            "q_network_state_dict": _state_dict_to_cpu(q_network),
            "target_network_state_dict": _state_dict_to_cpu(target_network),
            "optimizer_state_dict": _optimizer_state_to_cpu(optimizer),
            "scaler_state_dict": scaler.state_dict(),
            "replay": _replay_state_to_cpu(replay),
            "env": _env_state_to_cpu(env),
            "best_val_return": float(best_val_return),
            "best_val_legs": float(best_val_legs),
            "best_state": _tensor_dict_to_cpu(best_state),
            "loss_trace": list(loss_trace),
            "train_reward_trace": list(reward_trace),
            "valid_action_count_trace": list(valid_action_count_trace),
            "eval_trace": list(eval_trace),
            "rng_state": _capture_rng_state(device),
        },
        path,
    )


def _assert_checkpoint_schema(
    checkpoint: dict[str, Any],
    *,
    minute_feature_names: list[str],
    hour_feature_names: list[str],
    action_names: list[str],
    action_feature_names: list[str],
    transition_feature_dim: int = 0,
    dynamic_feature_dim: int = 0,
) -> None:
    expected = {
        "minute_feature_names": minute_feature_names,
        "hour_feature_names": hour_feature_names,
        "action_names": action_names,
    }
    for key, expected_values in expected.items():
        actual = checkpoint.get(key)
        if actual is None:
            raise ValueError(f"Warm-start checkpoint is missing {key}; refusing unverified fine-tune.")
        if list(actual) != list(expected_values):
            raise ValueError(f"Warm-start checkpoint {key} does not match the current dataset schema.")

    checkpoint_action_features = checkpoint.get("action_feature_names", [])
    if action_feature_names or checkpoint_action_features:
        if list(checkpoint_action_features) != list(action_feature_names):
            raise ValueError("Warm-start checkpoint action_feature_names does not match the current dataset schema.")

    constraint_names = checkpoint.get("constraint_feature_names")
    if constraint_names is None:
        raise ValueError("Warm-start checkpoint is missing constraint_feature_names; refusing unverified fine-tune.")
    if list(constraint_names) != list(CONSTRAINT_FEATURE_NAMES):
        raise ValueError("Warm-start checkpoint constraint feature schema does not match current code.")

    # Transition (position-aware) schema must match the model being warm-started into, in BOTH
    # directions: a v3 transition checkpoint cannot load a v2 (transition-off) model and vice versa,
    # and a schema-version/name drift is rejected with a clear message rather than a cryptic strict-load
    # state_dict error.
    expected_transition = list(TRANSITION_FEATURE_NAMES) if transition_feature_dim > 0 else []
    checkpoint_transition = list(checkpoint.get("transition_feature_names", []))
    if checkpoint_transition != expected_transition:
        raise ValueError(
            "Warm-start checkpoint transition feature schema does not match the current model "
            f"(use_transition_features mismatch or schema drift): checkpoint={checkpoint_transition}, "
            f"expected={expected_transition}."
        )

    # Same bidirectional guard for the PR-D dynamic position-state schema: a dynamic-aware checkpoint
    # (wider input) cannot warm-start a non-dynamic model and vice versa.
    expected_dynamic = list(DYNAMIC_TRANSITION_FEATURE_NAMES) if dynamic_feature_dim > 0 else []
    checkpoint_dynamic = list(checkpoint.get("dynamic_transition_feature_names", []))
    if checkpoint_dynamic != expected_dynamic:
        raise ValueError(
            "Warm-start checkpoint dynamic transition feature schema does not match the current model "
            f"(use_dynamic_transition_features mismatch or schema drift): checkpoint={checkpoint_dynamic}, "
            f"expected={expected_dynamic}."
        )


def load_minute_to_hour_warm_start(
    model: nn.Module,
    *,
    checkpoint_path: str | bytes | PathLike[str],
    train_data: HourFromMinuteDataSplit,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Warm-start checkpoint must be a saved minute-to-hour model artifact with model_state_dict.")
    _assert_checkpoint_schema(
        checkpoint,
        minute_feature_names=train_data.minute_feature_names,
        hour_feature_names=train_data.hour_feature_names,
        action_names=train_data.action_names,
        action_feature_names=train_data.action_feature_names,
        transition_feature_dim=int(getattr(model, "transition_feature_dim", 0)),
        dynamic_feature_dim=int(getattr(model, "dynamic_feature_dim", 0)),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Warm-start checkpoint architecture does not match current model hyperparameters.") from exc
    return {
        "loaded": True,
        "path": str(checkpoint_path),
        "model_version": checkpoint.get("model_version"),
        "uses_constraint_features": checkpoint.get("uses_constraint_features"),
        "uses_transition_features": checkpoint.get("uses_transition_features"),
    }


def compute_recency_weights(
    decision_timestamps: list[str],
    validation_start_ms: int,
    *,
    mode: str,
    half_life_days: float,
    min_weight: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-row recency weights for training transitions (see :class:`RecencyWeightConfig`).

    Ages are clamped at 0 and measured against ``validation_start_ms`` (the earliest validation
    decision), so a training row is weighted only by how far it sits BEFORE validation -- the test
    block is never referenced. Returns a float32 tensor of shape ``(len(decision_timestamps),)``.
    ``mode='none'`` returns all ones (so a weighted mean is identical to an unweighted mean).
    """
    count = len(decision_timestamps)
    if mode == "none":
        return torch.ones(count, dtype=torch.float32, device=device)
    if mode != "exponential":
        raise ValueError(f"Unsupported recency weighting mode: {mode!r}")
    if half_life_days <= 0.0:
        raise ValueError("recency half_life_days must be positive.")
    if not 0.0 < min_weight <= 1.0:
        # A zero floor lets a batch of only-old rows collapse to ~0 weight (unstable loss scale after
        # the clamp_min denominator), and contradicts "older regimes are never fully ignored".
        raise ValueError("recency min_weight must be in (0, 1].")
    day_ms = 86_400_000.0
    decay = math.log(2.0) / float(half_life_days)
    weights = torch.empty(count, dtype=torch.float32, device=device)
    for index, timestamp in enumerate(decision_timestamps):
        age_days = max(0.0, (validation_start_ms - _timestamp_to_epoch_ms(timestamp)) / day_ms)
        weights[index] = min_weight + (1.0 - min_weight) * math.exp(-decay * age_days)
    return weights


def train_minute_to_hour_dqn(
    train_data: HourFromMinuteDataSplit,
    val_data: HourFromMinuteDataSplit,
    *,
    device: torch.device,
    config: MinuteToHourTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    train_data = train_data if train_data.minute_features.device == device else train_data.to(device)
    val_data = val_data if val_data.minute_features.device == device else val_data.to(device)
    assert_matching_hour_from_minute_schema(train_data, val_data)
    # Recency weighting is anchored to the earliest VALIDATION decision; the test split is never
    # passed to this function, so older training rows can be down-weighted without any risk of
    # touching the held-out test block. mode='none' yields uniform weights (no behavior change).
    validation_start_ms = (
        min(_timestamp_to_epoch_ms(ts) for ts in val_data.decision_timestamps)
        if val_data.decision_timestamps
        else None
    )
    recency_mode = config.recency.mode if validation_start_ms is not None else "none"
    if recency_mode != "none" and validation_start_ms is not None and train_data.decision_timestamps:
        # Recency weighting is precisely where the train/validation boundary should be re-asserted:
        # a training row at/after validation start would silently get weight 1.0 (age clamped to 0)
        # and mask an upstream split bug. Fail loudly instead.
        train_max_ms = max(_timestamp_to_epoch_ms(ts) for ts in train_data.decision_timestamps)
        if train_max_ms >= validation_start_ms:
            raise ValueError(
                "train split overlaps validation start; refusing recency-weighted training "
                f"(train_max_ms={train_max_ms} >= validation_start_ms={validation_start_ms})."
            )
    train_recency_weights = compute_recency_weights(
        train_data.decision_timestamps,
        validation_start_ms or 0,
        mode=recency_mode,
        half_life_days=config.recency.half_life_days,
        min_weight=config.recency.min_weight,
        device=device,
    )
    recency_active = recency_mode != "none"
    action_count = len(train_data.action_names)
    transition_feature_dim = 0
    transition_table = None
    if config.use_transition_features:
        from rl_quant.action_risk import action_leverage_tensor, build_action_metadata, group_ids_for_actions

        action_meta = build_action_metadata(train_data.action_names)
        action_group_ids, _ = group_ids_for_actions(action_meta, device=device)
        cons = config.env.constraints
        # Use the env's cash_index / leg convention so the table's legs/cost columns match realized cost.
        transition_table = build_transition_feature_table(
            action_count=action_count,
            cash_index=int(cons.cash_index),
            one_way_cost_bps=cons.one_way_cost_bps,
            extra_switch_penalty_bps=cons.extra_switch_penalty_bps,
            count_etf_to_etf_as_two_legs=cons.count_etf_to_etf_as_two_legs,
            action_leverage=action_leverage_tensor(action_meta, device=device),
            action_group_ids=action_group_ids,
            device=device,
        )
        transition_feature_dim = TRANSITION_FEATURE_DIM
    dynamic_feature_dim = DYNAMIC_TRANSITION_FEATURE_DIM if config.use_dynamic_transition_features else 0
    q_network = MinuteToHourCausalTransformerQNetwork(
        minute_feature_dim=train_data.minute_features.shape[-1],
        hour_feature_dim=train_data.hour_features.shape[-1],
        action_count=action_count,
        hours_lookback=train_data.hours_lookback,
        minutes_per_hour=train_data.minutes_per_hour,
        d_model=config.d_model,
        n_heads=config.n_heads,
        minute_layers=config.minute_layers,
        hour_layers=config.hour_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        action_embedding_dim=config.action_embedding_dim,
        max_subhour_tokens=config.max_subhour_tokens,
        action_feature_dim=0 if train_data.action_features is None else int(train_data.action_features.shape[-1]),
        transition_feature_dim=transition_feature_dim,
        transition_table=transition_table,
        dynamic_feature_dim=dynamic_feature_dim,
    ).to(device)
    warm_start_info: dict[str, object] | None = None
    if config.warm_start_model is not None:
        warm_start_info = load_minute_to_hour_warm_start(
            q_network,
            checkpoint_path=config.warm_start_model,
            train_data=train_data,
        )
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp, config.learning.amp_dtype)
    replay_fields = {
        "indices": ((), torch.long),
        "previous_actions": ((), torch.long),
        "constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
        "action_mask": ((action_count,), torch.bool),
        "actions": ((), torch.long),
        "rewards": ((), torch.float32),
        "next_indices": ((), torch.long),
        "next_previous_actions": ((), torch.long),
        "next_constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
        "next_action_mask": ((action_count,), torch.bool),
        "terminated": ((), torch.float32),
    }
    if config.use_dynamic_transition_features:
        # Only declared when the flag is on -> storage has exactly the 11 legacy keys otherwise, so the
        # buffer (and a flag-off resume) is byte-identical. The step dict always carries these keys; the
        # add() call below filters to declared fields, so they are silently dropped when the flag is off.
        replay_fields["position_dynamic"] = ((DYNAMIC_TRANSITION_FEATURE_DIM,), torch.float32)
        replay_fields["next_position_dynamic"] = ((DYNAMIC_TRANSITION_FEATURE_DIM,), torch.float32)
    replay = TensorDictReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields=replay_fields,
    )
    env = VectorizedMinuteToHourEnv(train_data, config.env, device)
    reservation = CudaVramReservation(target_gb=config.target_vram_gb, safety_gb=config.vram_safety_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_return = -float("inf")
    best_val_legs = float("inf")
    best_state = _state_dict_to_cpu(q_network)
    loss_trace: list[float] = []
    reward_trace: list[float] = []
    valid_action_count_trace: list[float] = []
    eval_trace: list[dict[str, float | int | None | str]] = []
    resume_info: dict[str, object] = {"loaded": False}
    start_step = 1
    resume_path = Path(config.resume_training_state) if config.resume_training_state is not None else None
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_kind") != "minute_to_hour_dqn_training_state":
            raise ValueError("Resume checkpoint is not a minute-to-hour DQN training state.")
        q_network.load_state_dict(checkpoint["q_network_state_dict"], strict=True)
        target_network.load_state_dict(checkpoint["target_network_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        replay_state = checkpoint.get("replay")
        env_state = checkpoint.get("env")
        if not isinstance(replay_state, dict) or not isinstance(env_state, dict):
            raise ValueError("Resume checkpoint is missing replay or environment state.")
        _load_replay_state(replay, replay_state, device)
        _load_env_state(env, env_state, device, require_dynamic=config.use_dynamic_transition_features)
        best_val_return = float(checkpoint.get("best_val_return", best_val_return))
        best_val_legs = float(checkpoint.get("best_val_legs", best_val_legs))
        raw_best_state = checkpoint.get("best_state")
        if not isinstance(raw_best_state, dict):
            raise ValueError("Resume checkpoint is missing best_state.")
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in raw_best_state.items()
            if torch.is_tensor(value)
        }
        loss_trace = [float(item) for item in checkpoint.get("loss_trace", [])]
        reward_trace = [float(item) for item in checkpoint.get("train_reward_trace", [])]
        valid_action_count_trace = [float(item) for item in checkpoint.get("valid_action_count_trace", [])]
        eval_trace = list(checkpoint.get("eval_trace", []))
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state, device)
        resumed_step = int(checkpoint.get("step", 0))
        start_step = min(resumed_step + 1, config.learning.train_steps + 1)
        resume_info = {
            "loaded": True,
            "path": str(resume_path),
            "resumed_from_step": resumed_step,
            "start_step": start_step,
        }

    checkpoint_path = Path(config.checkpoint_training_state) if config.checkpoint_training_state is not None else None
    checkpoint_every_steps = max(0, int(config.checkpoint_every_steps))
    shadow_reward_deltas: list[float] = []  # PR-3: per-step mean execution-shadow deltas (stays empty if flag off)
    shadow_cost_deltas: list[float] = []
    for step in range(start_step, config.learning.train_steps + 1):
        minute, mask, hour, action_features, previous_actions, constraint_features, action_mask = env.observe()
        valid_action_count_trace.append(float(action_mask.sum(dim=1).float().mean().item()))
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q_values = q_network(
                    minute,
                    mask,
                    hour,
                    previous_actions,
                    constraint_features,
                    action_features=action_features,
                    dynamic_state=env.dynamic_state() if config.use_dynamic_transition_features else None,
                )
            greedy_actions = apply_leg_aware_hysteresis(
                q_values,
                previous_actions,
                action_mask,
                one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                cash_index=config.env.constraints.cash_index,
                reward_scale=config.env.reward_scale,
                count_etf_to_etf_as_two_legs=config.env.constraints.count_etf_to_etf_as_two_legs,
            )
            random_actions = sample_valid_actions(action_mask)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)
        transition = env.step(actions)
        replay.add(**{key: value for key, value in transition.items() if key in replay.storage})
        reward_trace.append(float(transition["rewards"].mean().item()))
        if "reward_delta_shadow" in transition:  # PR-3 shadow side-channel (present only when the flag is on)
            shadow_reward_deltas.append(float(transition["reward_delta_shadow"].mean().item()))
            shadow_cost_deltas.append(float(transition["cost_delta_shadow"].mean().item()))
        env.reset(transition["resets"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            # Clamp next_indices for the state lookup: a TRUE terminal transition can store an
            # out-of-data next row, whose bootstrapped value is zeroed below via `terminated` anyway.
            # For non-terminal transitions next_indices is always in range, so this is a no-op there.
            n_rows = int(train_data.action_returns.shape[0])
            # min_index = 0: state() is plain row indexing (each row carries its own self-contained
            # window), so there is no rolling-window floor to respect and no tail-wrap to avoid.
            # valid_index_mask is the same tensor the env uses to DEFINE terminated (terminated =
            # ~valid_index_mask[next]), so every non-terminal next is mask-True by construction --
            # passing it rejects nothing legitimate and turns a mask/terminated mismatch into a loud error.
            safe_next_indices = safe_next_row_indices(
                batch["next_indices"],
                batch["terminated"],
                min_index=0,
                max_index=n_rows - 1,
                valid_index_mask=train_data.valid_index_mask,
            )
            current_minute, current_mask, current_hour = train_data.state(batch["indices"])
            next_minute, next_mask, next_hour = train_data.state(safe_next_indices)
            current_action_features = train_data.action_feature_state(batch["indices"])
            next_action_features = train_data.action_feature_state(safe_next_indices)
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q = q_network(
                    current_minute,
                    current_mask,
                    current_hour,
                    batch["previous_actions"],
                    batch["constraint_features"],
                    action_features=current_action_features,
                    dynamic_state=batch.get("position_dynamic"),
                )
                chosen_q = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online = q_network(
                        next_minute,
                        next_mask,
                        next_hour,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                        action_features=next_action_features,
                        dynamic_state=batch.get("next_position_dynamic"),
                    )
                    next_actions = apply_leg_aware_hysteresis(
                        next_online,
                        batch["next_previous_actions"],
                        batch["next_action_mask"],
                        one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                        extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                        q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                        cash_index=config.env.constraints.cash_index,
                        reward_scale=config.env.reward_scale,
                        count_etf_to_etf_as_two_legs=config.env.constraints.count_etf_to_etf_as_two_legs,
                    )
                    next_target = target_network(
                        next_minute,
                        next_mask,
                        next_hour,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                        action_features=next_action_features,
                        dynamic_state=batch.get("next_position_dynamic"),
                    )
                    next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    # fp32 TD target/loss under AMP: with reward_scale=10_000 the bootstrapped
                    # targets reach magnitudes where fp16 precision is comparable to per-step
                    # rewards, so compute the target and smooth_l1 loss in float32.
                    # Bootstrap through episode-length TRUNCATIONS; zero the bootstrap only on TRUE
                    # terminals. Shared with the hourly trainer via core.dqn_td_target.
                    target_q = dqn_td_target(batch["rewards"], config.learning.gamma, batch["terminated"], next_q)
                if recency_active:
                    # Recency-weighted smooth_l1: per-sample loss scaled by each transition's source
                    # training row weight (looked up via the replay-stored decision-row `indices`).
                    per_sample_loss = F.smooth_l1_loss(chosen_q.float(), target_q, reduction="none")
                    sample_weights = train_recency_weights[batch["indices"]]
                    loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
                else:
                    # Default (uniform) path: identical fused mean reduction as before recency support,
                    # so disabling recency is a byte-for-byte no-op on the training objective.
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
            val_result = evaluate_minute_to_hour_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                constraints=config.env.constraints,
                episode_length=config.env.episode_length,
                reward_scale=config.env.reward_scale,
                cash_idle_penalty_bps=config.env.cash_idle_penalty_bps,
            )
            # Restore train() mode: the evaluator puts the shared q_network in eval(), which would
            # otherwise leave dropout disabled for all subsequent gradient steps.
            q_network.train()
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_order_legs": val_result.market_order_legs,
                    "val_sharpe": val_result.annualized_sharpe,
                    "average_loss": sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1),
                    "average_train_reward": sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1),
                    "average_valid_action_count": sum(valid_action_count_trace[-200:])
                    / max(len(valid_action_count_trace[-200:]), 1),
                }
            )
            if val_result.total_return > best_val_return or (
                abs(val_result.total_return - best_val_return) <= 1e-12
                and val_result.market_order_legs < best_val_legs
            ):
                best_val_return = val_result.total_return
                best_val_legs = val_result.market_order_legs
                best_state = _state_dict_to_cpu(q_network)
        if checkpoint_path is not None and checkpoint_every_steps > 0 and (
            step % checkpoint_every_steps == 0 or step == config.learning.train_steps
        ):
            _save_minute_to_hour_training_state(
                checkpoint_path,
                step=step,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                scaler=scaler,
                replay=replay,
                env=env,
                best_val_return=best_val_return,
                best_val_legs=best_val_legs,
                best_state=best_state,
                loss_trace=loss_trace,
                reward_trace=reward_trace,
                valid_action_count_trace=valid_action_count_trace,
                eval_trace=eval_trace,
                device=device,
            )

    q_network.load_state_dict(best_state)
    recency_policy: dict[str, object] = {
        "mode": recency_mode,
        "half_life_days": config.recency.half_life_days,
        "min_weight": config.recency.min_weight,
        "validation_start_ms": validation_start_ms,
        # The trainer only ever receives train_data + val_data; the test split is never visible here.
        "test_used_for_recency_selection": False,
    }
    if train_recency_weights.numel() > 0:
        recency_policy["weight_min"] = float(train_recency_weights.min().item())
        recency_policy["weight_max"] = float(train_recency_weights.max().item())
        recency_policy["weight_mean"] = float(train_recency_weights.mean().item())
    artifacts: dict[str, object] = {
        "best_val_return": best_val_return,
        "best_val_order_legs": best_val_legs,
        "recency_policy": recency_policy,
        "amp_enabled": scaler.is_enabled(),
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "valid_action_count_trace": valid_action_count_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
        "cash_idle_penalty_bps": float(config.env.cash_idle_penalty_bps),
        # PR-3: shadow execution-reward diagnostics (None when the flag is off). Label-changing only -- the
        # trained model + every other metric are byte-identical to a shadow-off run. Honesty labels: this is a
        # STATIC single-slot weight-bps COST-MODEL shadow, NOT real-executable (no NBBO / quote-side fills /
        # latency P&L on this dataset).
        "execution_env_reward_shadow": bool(config.env.execution_env_reward_shadow),
        "execution_shadow_cost_model": (
            "static_single_slot_weight_bps" if config.env.execution_env_reward_shadow else None
        ),
        "execution_shadow_real_executable": False,
        # Schema version (the cost_delta basis CHANGED in v2: it is now vs the legacy LEG cost, switch penalty
        # held constant) + the auditable weight-source assumption (see docs/execution_wiring_design.md §3).
        "execution_shadow_schema_version": 2 if config.env.execution_env_reward_shadow else None,
        "execution_shadow_cost_delta_basis": (
            "execution_cost_bps_shadow - legacy_leg_cost_bps (switch_penalty + cash_idle held constant)"
            if config.env.execution_env_reward_shadow else None
        ),
        "execution_shadow_weight_source": (
            "action_metadata.max_weight" if config.env.execution_env_reward_shadow else None
        ),
        # reward delta in REWARD units (scale-dependent) AND scale-normalised bps (comparable across runs).
        "execution_shadow_reward_delta_mean": (
            sum(shadow_reward_deltas) / len(shadow_reward_deltas) if shadow_reward_deltas else None
        ),
        "execution_shadow_reward_delta_bps_mean": (
            (sum(shadow_reward_deltas) / len(shadow_reward_deltas)) / float(config.env.reward_scale) * 10_000.0
            if shadow_reward_deltas else None
        ),
        "execution_shadow_cost_delta_bps_mean": (
            sum(shadow_cost_deltas) / len(shadow_cost_deltas) if shadow_cost_deltas else None
        ),
        "model_version": (
            DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION
            if config.use_dynamic_transition_features
            else POSITION_AWARE_POLICY_MODEL_VERSION
            if config.use_transition_features
            else CONSTRAINED_POLICY_MODEL_VERSION
        ),
        "uses_constraint_features": True,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "uses_transition_features": bool(config.use_transition_features),
        "transition_feature_names": list(TRANSITION_FEATURE_NAMES) if config.use_transition_features else [],
        "transition_feature_dim": TRANSITION_FEATURE_DIM if config.use_transition_features else 0,
        "transition_feature_schema_version": TRANSITION_FEATURE_SCHEMA_VERSION if config.use_transition_features else 0,
        "uses_dynamic_transition_features": bool(config.use_dynamic_transition_features),
        "dynamic_transition_feature_names": (
            list(DYNAMIC_TRANSITION_FEATURE_NAMES) if config.use_dynamic_transition_features else []
        ),
        "dynamic_transition_feature_dim": (
            DYNAMIC_TRANSITION_FEATURE_DIM if config.use_dynamic_transition_features else 0
        ),
        "dynamic_transition_feature_schema_version": (
            DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION if config.use_dynamic_transition_features else 0
        ),
        "warm_start": warm_start_info or {"loaded": False},
        "resume": resume_info,
        "last_completed_step": int(config.learning.train_steps if start_step <= config.learning.train_steps else start_step - 1),
        "checkpoint_training_state": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_every_steps": checkpoint_every_steps,
        "source_bar_interval": train_data.source_bar_interval,
        "context_bars_per_hour": train_data.effective_context_bars_per_hour,
        "max_subhour_tokens": config.max_subhour_tokens,
        "split_policy": train_data.split_policy,
        "action_feature_names": train_data.action_feature_names,
        "action_feature_dim": 0 if train_data.action_features is None else int(train_data.action_features.shape[-1]),
        "action_feature_groups": train_data.action_feature_groups,
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
